"""End-to-end latency and resource-overhead evaluation for segmentation models.

This module loads a trained PyTorch segmentation checkpoint (a full ``nn.Module``
serialized by the training scripts) and benchmarks steady-state inference on the
BDD100k test split. Three aligned per-image latency distributions are reported:

* inference latency: model forward pass only;
* computational overhead: image decoding, preprocessing, DataLoader collation,
  and host-to-device transfer;
* end-to-end latency: computational overhead plus inference latency.

The latency DataLoader runs synchronously with ``num_workers=0``. This is
intentional: worker prefetching would overlap data preparation with model
execution and make per-image preprocessing overhead impossible to attribute
reliably. CUDA synchronisation is used where needed so GPU work is completed
inside the corresponding timing interval.

In addition to latency percentiles, the benchmark records resource-use indicators
that are useful when discussing deployment overhead:

* peak process CPU utilisation during a measured end-to-end sample;
* peak process RAM (resident-set size / RSS);
* peak CPU temperature when the operating system exposes a CPU sensor;
* peak PyTorch-managed GPU memory (allocated and reserved) on CUDA;
* peak NVIDIA GPU temperature and total device VRAM usage when NVML is available.

The first warm-up iterations are excluded from both latency and resource summaries
so that one-off CUDA/cuDNN/allocator initialisation costs do not dominate the
steady-state results. Temperature support is platform dependent. NVIDIA
temperature/NVML metrics require the optional ``nvidia-ml-py`` package
(``import pynvml``). CPU/RAM metrics require ``psutil``; if an optional sensor is
unavailable, the benchmark still runs and reports that metric as unavailable.
"""

import argparse
import glob
import os
import threading
import time
from typing import Dict, List, Optional, Tuple, cast

import albumentations as A
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Float, jaxtyped
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import psutil
except ImportError:  # Optional: latency and CUDA-memory metrics can still run.
    psutil = None  # type: ignore[assignment]

try:
    import pynvml
except ImportError:  # Optional: only needed for NVIDIA temperature/NVML metrics.
    pynvml = None  # type: ignore[assignment]


# Shape aliases documented alongside each pipeline stage. ``jaxtyped`` +
# ``beartype`` enforce them at runtime, turning shape mismatches into clear
# ``TypeCheckError`` exceptions rather than confusing downstream broadcasts.
ImageTensor = Float[torch.Tensor, "3 h w"]  # single image, channel-first layout
BatchImage = Float[torch.Tensor, "b 3 h w"]  # collated batch of images
Logits = Float[torch.Tensor, "b c h w"]  # model output, channel count is model-defined


class Configuration:
    """Constants controlling end-to-end latency evaluation.

    The benchmark uses these values to select the execution device, reconstruct
    the deterministic test split, size model inputs, discard warm-up samples, and
    configure low-overhead resource monitoring.

    Steps
    -----
    1. Select CUDA when available, otherwise fall back to CPU.
    2. Force a synchronous single-image DataLoader for attributable overhead
       timing.
    3. Define the fixed image size, warm-up count, and resource sampling cadence.
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # End-to-end timing requires data preparation to execute inside the measured
    # ``next(data_iterator)`` call. Multiple workers would prefetch samples and
    # hide part of that work behind the previous model invocation.
    NUM_WORKERS = 0
    SEED = 768

    IMAGE_HEIGHT = 360
    IMAGE_WIDTH = 640

    # Latency is measured per image, so inference is run with batch size 1.
    LATENCY_BATCH_SIZE = 1
    # Number of leading samples discarded before recording timings. These absorb
    # the one-off costs of device warm-up so the reported distribution reflects
    # steady-state inference only.
    WARMUP_ITERATIONS = 20

    # Hardware sensors are relatively slow to query, so they are sampled in a
    # low-overhead background thread during the measured part of the benchmark.
    RESOURCE_SAMPLE_INTERVAL_SECONDS = 0.25
    BYTES_PER_MIB = 1024.0 ** 2


class Path:
    """Filesystem directory locations for BDD100k images and masks."""

    BASE = "./data/bdd100k"

    SEGMENTATION_MASK_LABEL_FOLDER = BASE + "/segmentation_maps/color_labels"
    SEGMENTATION_MASK_TRAIN_PATH = SEGMENTATION_MASK_LABEL_FOLDER + "/train"

    IMAGE_FOLDER = BASE + "/images_10k"
    IMAGE_TRAIN_PATH = IMAGE_FOLDER + "/train"


class BDDInferenceDataset(Dataset[ImageTensor]):
    """BDD100k image-only dataset used for deployment-style latency timing.

    Ground-truth masks are deliberately excluded from this dataset because mask
    decoding and one-hot encoding are evaluation-only costs that would not occur
    when a deployed model receives an image for inference. Keeping only the image
    path makes the computational-overhead timer represent the work required to
    prepare a real model input.

    Steps
    -----
    1. Read the image path selected by the reconstructed BDD100k test split.
    2. Decode the image as RGB and normalise pixel values to ``[0, 1]`` floats.
    3. Apply the inference transform pipeline and return a channel-first tensor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an ``image_paths`` column.
    transform : A.Compose | None, optional
        Albumentations transform pipeline applied to each image. Defaults to
        ``None``.

    Raises
    ------
    IndexError
        If an index outside the dataset path list is requested.
    """

    def __init__(self, df: pd.DataFrame, transform: A.Compose | None = None) -> None:
        super().__init__()

        # Store only image paths so deployment latency does not accidentally
        # include label/mask work that is absent from real inference requests.
        self.image_paths: List[str] = df["image_paths"].to_list()
        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of inference images available for timing."""
        return len(self.image_paths)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> ImageTensor:
        """Decode and preprocess one image into a model-ready tensor.

        Steps
        -----
        1. Decode the selected JPEG and force a three-channel RGB layout.
        2. Convert the image to a normalised ``float32`` NumPy array.
        3. Apply the configured inference transforms, or ``ToTensorV2`` when no
           transform was supplied.

        Parameters
        ----------
        index : int
            Zero-based position of the image to load.

        Returns
        -------
        ImageTensor
            Preprocessed image tensor with shape ``(3, H, W)``.

        Raises
        ------
        IndexError
            If ``index`` is outside the range of the dataset path list.
        """
        image_path = self.image_paths[index]

        # RGB conversion makes input channel count deterministic even when a
        # source image happens to carry grayscale or alpha-channel metadata.
        image_pil = Image.open(image_path).convert("RGB")
        image = np.array(image_pil).astype(np.float32) / 255.0

        if self.transform:
            transformed = self.transform(image=image)
        else:
            transformed = ToTensorV2()(image=image)

        return cast(ImageTensor, transformed["image"])


def find_image_path_from_mask(complete_mask_path: str, base_image_path: str) -> str:
    """Derive the corresponding JPEG image path from a mask PNG path.

    Steps
    -----
    1. Extract the image ID by stripping the mask-specific suffix.
    2. Construct and return the full path in the target image folder.

    Parameters
    ----------
    complete_mask_path : str
        Full filesystem path to a segmentation mask PNG.
    base_image_path : str
        Directory containing the corresponding JPEG images.

    Returns
    -------
    str
        Full filesystem path to the matching image file.
    """
    file_path_split = complete_mask_path.split("/")
    mask_file_name = file_path_split[-1].split("_")[0]

    return base_image_path + "/" + mask_file_name + ".jpg"


def find_train_image_path_from_mask(complete_mask_path: str) -> str:
    """Locate the training image corresponding to a training mask.

    Steps
    -----
    1. Delegate to :func:`find_image_path_from_mask` using
       ``Path.IMAGE_TRAIN_PATH``.

    Parameters
    ----------
    complete_mask_path : str
        Full path to a training mask PNG.

    Returns
    -------
    str
        Full path to the corresponding training image JPEG.
    """
    return find_image_path_from_mask(complete_mask_path, Path.IMAGE_TRAIN_PATH)


def find_mask_path_from_image(complete_image_path: str, base_mask_path: str) -> str:
    """Derive the corresponding PNG mask path from a JPEG image path.

    Steps
    -----
    1. Extract the sample stem by stripping the ``.jpg`` extension.
    2. Append the ``_train_color.png`` suffix used by BDD100k masks.

    Parameters
    ----------
    complete_image_path : str
        Full filesystem path to an image JPEG.
    base_mask_path : str
        Directory containing the corresponding mask PNGs.

    Returns
    -------
    str
        Full filesystem path to the matching mask file.
    """
    file_path_split = complete_image_path.split("/")
    mask_file_name = file_path_split[-1].split(".")[0]

    return base_mask_path + "/" + mask_file_name + "_train_color.png"


def find_train_mask_path_from_image(complete_image_path: str) -> str:
    """Locate the training mask corresponding to a training image.

    Steps
    -----
    1. Delegate to :func:`find_mask_path_from_image` using
       ``Path.SEGMENTATION_MASK_TRAIN_PATH``.

    Parameters
    ----------
    complete_image_path : str
        Full path to a training image JPEG.

    Returns
    -------
    str
        Full path to the corresponding training mask PNG.
    """
    return find_mask_path_from_image(complete_image_path, Path.SEGMENTATION_MASK_TRAIN_PATH)


def load_test_split() -> pd.DataFrame:
    """Scan disk and reconstruct the BDD100k test split.

    This duplicates the exact partitioning logic from the training scripts so
    the latency is measured on the identical test split that was held out
    during training. The official BDD100k ``val`` split is intentionally not
    loaded here because it is irrelevant to latency measurement.

    Steps
    -----
    1. Collect training masks and drop any that do not match the expected
       ``1280x720`` resolution.
    2. Map them to corresponding training images and drop any missing or
       malformed pairs.
    3. Partition the resulting pool 80/20 into train/test using the fixed
       ``Configuration.SEED`` and return the test partition.

    Returns
    -------
    pd.DataFrame
        DataFrame holding the ``image_paths`` and ``mask_paths`` columns of the
        test split.
    """
    train_mask_paths = glob.glob(f"{Path.SEGMENTATION_MASK_TRAIN_PATH}/*.png")
    problematic_masks: List[str] = []
    for complete_mask_path in train_mask_paths:
        with Image.open(complete_mask_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_masks.append(complete_mask_path)
                train_mask_paths.remove(complete_mask_path)
    print(f"Problematic masks: {problematic_masks}")

    train_image_paths = list(map(find_train_image_path_from_mask, train_mask_paths))
    problematic_images: List[str] = []
    for complete_image_path in train_image_paths:
        with Image.open(complete_image_path) as img:
            width, height = img.size
            if width != 1280 or height != 720:
                problematic_images.append(complete_image_path)
                train_image_paths.remove(complete_image_path)
                train_mask_paths.remove(find_train_mask_path_from_image(complete_image_path))
    print(f"Problematic images: {problematic_images}")

    train_test_df = pd.DataFrame({
        "image_paths": train_image_paths,
        "mask_paths": train_mask_paths,
    })
    _, test_df = train_test_split(
        train_test_df, test_size=0.2, random_state=Configuration.SEED
    )

    return test_df


@jaxtyped(typechecker=beartype)
def forward(model: nn.Module, x: BatchImage) -> Logits:
    """Run a single forward pass and assert the input/output shapes.

    Centralising the forward pass here lets ``jaxtyping`` verify that the input
    batch is always ``(B, 3, H, W)`` and that the model produces
    ``(B, C, H, W)`` logits, which is where shape confusion most often arises.

    Parameters
    ----------
    model : nn.Module
        The segmentation model to run.
    x : BatchImage
        Input batch of images with shape ``(B, 3, H, W)``.

    Returns
    -------
    Logits
        Raw model logits with shape ``(B, C, H, W)``.
    """
    return cast(Logits, model(x))


def load_trained_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a trained segmentation model checkpoint from disk.

    Steps
    -----
    1. Verify the checkpoint file exists.
    2. Load it with ``torch.load`` using ``weights_only=False`` to unpack the
       serialized ``nn.Module`` object produced by the training scripts.
    3. Transfer it to the target ``device`` and switch into ``eval()`` mode.

    Parameters
    ----------
    checkpoint_path : str
        Filesystem path to the model ``.pt`` file.
    device : torch.device
        Device (CUDA or CPU) on which to place the model.

    Returns
    -------
    nn.Module
        Loaded PyTorch segmentation model ready for evaluation.

    Raises
    ------
    FileNotFoundError
        If ``checkpoint_path`` does not exist.
    TypeError
        If the loaded checkpoint object is not an ``nn.Module``.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # weights_only=False is required because the training scripts save entire
    # nn.Module objects rather than raw state dicts.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, nn.Module):
        raise TypeError(
            f"Checkpoint at '{checkpoint_path}' loaded an object of type {type(checkpoint)}, "
            "but an nn.Module instance was expected."
        )

    model = checkpoint.to(device)
    model.eval()

    return model


def _read_max_cpu_temperature_c() -> Optional[float]:
    """Return the hottest CPU-related sensor currently exposed by psutil.

    CPU temperature availability is operating-system and hardware dependent. On
    many Linux systems ``psutil.sensors_temperatures`` exposes ``coretemp``,
    ``k10temp``, ``zenpower`` or similarly named sensors. Windows/macOS commonly
    expose no value through psutil, in which case ``None`` is returned.
    """
    if psutil is None or not hasattr(psutil, "sensors_temperatures"):
        return None

    try:
        temperature_groups = psutil.sensors_temperatures(fahrenheit=False)
    except (AttributeError, NotImplementedError, OSError):
        return None

    if not temperature_groups:
        return None

    preferred_group_tokens = (
        "coretemp",
        "k10temp",
        "zenpower",
        "cpu_thermal",
        "x86_pkg_temp",
    )
    preferred_label_tokens = ("cpu", "package", "core", "tdie", "tctl")

    candidates: List[float] = []
    for group_name, entries in temperature_groups.items():
        group_lower = group_name.lower()
        group_is_cpu = any(token in group_lower for token in preferred_group_tokens)

        for entry in entries:
            current = getattr(entry, "current", None)
            if current is None:
                continue

            label = str(getattr(entry, "label", "") or "").lower()
            label_is_cpu = any(token in label for token in preferred_label_tokens)
            if group_is_cpu or label_is_cpu:
                candidates.append(float(current))

    return max(candidates) if candidates else None


class ResourceMonitor:
    """Low-overhead background sampler for RAM and temperature measurements.

    The monitor deliberately does *not* use sampled CPU percentage as the primary
    CPU metric, because an end-to-end sample may be much shorter than a sensor
    sampling interval. Peak CPU utilisation is instead derived in the benchmark
    from process CPU-time deltas measured around every individual end-to-end
    sample.

    Steps
    -----
    1. Capture a baseline resource snapshot before measured samples begin.
    2. Sample process RAM, system RAM, and available NVIDIA sensors in a daemon
       thread at the configured interval.
    3. Stop the thread after measurement and expose peak/baseline values through
       :meth:`summary`.

    Parameters
    ----------
    device : torch.device
        Device used by the benchmark; CUDA enables optional NVML sampling.
    sample_interval_seconds : float
        Requested background sensor-sampling interval in seconds. Values below
        50 ms are clamped to reduce measurement perturbation.
    """

    def __init__(self, device: torch.device, sample_interval_seconds: float) -> None:
        self.device = device
        self.sample_interval_seconds = max(float(sample_interval_seconds), 0.05)
        self.process = psutil.Process(os.getpid()) if psutil is not None else None

        self.baseline_process_rss_bytes: Optional[int] = None
        self.peak_process_rss_bytes: Optional[int] = None
        self.peak_system_ram_percent: Optional[float] = None
        self.max_gpu_temperature_c: Optional[float] = None
        self.peak_nvml_device_vram_used_bytes: Optional[int] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_handle = None
        self._nvml_initialized = False

        self._initialise_nvml()

    def _initialise_nvml(self) -> None:
        """Initialise NVML for NVIDIA temperature/device-memory sampling."""
        if self.device.type != "cuda" or pynvml is None:
            return

        try:
            pynvml.nvmlInit()
            device_index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml_initialized = True
        except Exception:
            # NVML may be unavailable inside containers, WSL, unsupported driver
            # versions, or systems without permission to query the GPU.
            self._nvml_handle = None
            self._nvml_initialized = False
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    @staticmethod
    def _update_max(current: Optional[float], value: Optional[float]) -> Optional[float]:
        if value is None:
            return current
        return value if current is None else max(current, value)

    def sample_once(self) -> None:
        """Capture one snapshot of RAM and available thermal/NVML sensors."""
        if self.process is not None:
            try:
                rss_bytes = int(self.process.memory_info().rss)
                if self.baseline_process_rss_bytes is None:
                    self.baseline_process_rss_bytes = rss_bytes
                self.peak_process_rss_bytes = (
                    rss_bytes
                    if self.peak_process_rss_bytes is None
                    else max(self.peak_process_rss_bytes, rss_bytes)
                )
            except (psutil.Error, OSError):
                pass

            try:
                system_ram_percent = float(psutil.virtual_memory().percent)
                self.peak_system_ram_percent = self._update_max(
                    self.peak_system_ram_percent, system_ram_percent
                )
            except (AttributeError, OSError):
                pass

        if self._nvml_initialized and self._nvml_handle is not None:
            try:
                gpu_temp = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
                self.max_gpu_temperature_c = self._update_max(
                    self.max_gpu_temperature_c, gpu_temp
                )
            except Exception:
                pass

            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                used_bytes = int(memory_info.used)
                self.peak_nvml_device_vram_used_bytes = (
                    used_bytes
                    if self.peak_nvml_device_vram_used_bytes is None
                    else max(self.peak_nvml_device_vram_used_bytes, used_bytes)
                )
            except Exception:
                pass

    def _sampling_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self.sample_once()

    def start(self) -> None:
        self.sample_once()
        self._thread = threading.Thread(
            target=self._sampling_loop,
            name="resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 4.0))
        self.sample_once()

        if self._nvml_initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def summary(self) -> Dict[str, Optional[float]]:
        mib = Configuration.BYTES_PER_MIB
        baseline_ram_mb = (
            None
            if self.baseline_process_rss_bytes is None
            else self.baseline_process_rss_bytes / mib
        )
        peak_ram_mb = (
            None
            if self.peak_process_rss_bytes is None
            else self.peak_process_rss_bytes / mib
        )
        incremental_ram_mb = (
            None
            if baseline_ram_mb is None or peak_ram_mb is None
            else max(0.0, peak_ram_mb - baseline_ram_mb)
        )
        peak_nvml_vram_mb = (
            None
            if self.peak_nvml_device_vram_used_bytes is None
            else self.peak_nvml_device_vram_used_bytes / mib
        )

        return {
            "baseline_process_ram_mb": baseline_ram_mb,
            "peak_process_ram_mb": peak_ram_mb,
            "incremental_process_ram_mb": incremental_ram_mb,
            "peak_system_ram_percent": self.peak_system_ram_percent,
            "max_gpu_temperature_c": self.max_gpu_temperature_c,
            # NVML device memory is *whole-device* usage and can include other
            # processes. PyTorch-managed values below are better for model-level
            # attribution when CUDA is used.
            "peak_nvml_device_vram_used_mb": peak_nvml_vram_mb,
        }


def _process_cpu_seconds(process: object) -> Optional[float]:
    """Return cumulative user+system CPU seconds for the benchmark process."""
    if psutil is None or process is None:
        return None

    try:
        cpu_times = process.cpu_times()
        return float(cpu_times.user + cpu_times.system)
    except (psutil.Error, OSError):
        return None


def measure_latency_ms(
    model: nn.Module,
    dataloader: DataLoader[ImageTensor],
    device: torch.device,
    warmup_iterations: int = Configuration.WARMUP_ITERATIONS,
    resource_sample_interval_seconds: float = Configuration.RESOURCE_SAMPLE_INTERVAL_SECONDS,
) -> Tuple[Dict[str, List[float]], Dict[str, Optional[float]]]:
    """Measure inference, computational-overhead, and end-to-end latency.

    Computational overhead is the synchronous work required to turn the next
    dataset sample into a model-ready device tensor: image decoding,
    preprocessing, DataLoader collation, and host-to-device transfer. Inference
    latency measures only the model forward pass. End-to-end latency is computed
    per sample as ``computational_overhead + inference`` so the reported values
    are exactly additive.

    A synchronous DataLoader is required because worker prefetching would overlap
    data preparation with earlier inference and therefore under-report the true
    per-sample computational overhead. The resource summary covers the measured
    steady-state portion after warm-up. CPU utilisation is the maximum
    per-end-to-end-sample average derived from process CPU-time deltas; it may
    exceed 100% when multiple CPU cores are used. RAM is process RSS. CUDA memory
    uses PyTorch's exact peak allocator counters. CPU/GPU temperatures are
    best-effort hardware-sensor measurements.

    Steps
    -----
    1. Warm up the complete input-preparation and forward-pass path without
       recording latency or resource peaks.
    2. For each remaining sample, time synchronous data preparation/device
       transfer and the model forward pass separately.
    3. Sum the two per-sample timings into end-to-end latency and collect resource
       peaks over the same measured phase.

    Parameters
    ----------
    model : nn.Module
        Segmentation model to benchmark in evaluation mode.
    dataloader : DataLoader[ImageTensor]
        Synchronous image-only DataLoader with ``num_workers=0``.
    device : torch.device
        CPU or CUDA device on which inference executes.
    warmup_iterations : int, optional
        Number of leading complete samples to exclude from measurements.
    resource_sample_interval_seconds : float, optional
        Background interval for RAM and temperature sensor sampling.

    Returns
    -------
    Tuple[Dict[str, List[float]], Dict[str, Optional[float]]]
        A mapping containing aligned per-image ``inference``, ``overhead``, and
        ``end_to_end`` latency lists in milliseconds, plus a resource-usage
        summary.

    Raises
    ------
    ValueError
        If warm-up iterations are negative, the DataLoader uses background
        workers, too few samples exist for warm-up, or no samples remain for the
        measured phase.
    """
    model.eval()
    use_cuda = device.type == "cuda"
    inference_latencies_ms: List[float] = []
    overhead_latencies_ms: List[float] = []
    end_to_end_latencies_ms: List[float] = []

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative.")
    if dataloader.num_workers != 0:
        raise ValueError(
            "End-to-end overhead timing requires dataloader.num_workers == 0 "
            "so decoding and preprocessing are not hidden by worker prefetching."
        )

    data_iterator = iter(dataloader)

    # Warm up the same end-to-end path that will be measured. This prevents
    # one-time decoder, transform, allocator, CUDA, and cuDNN setup from
    # dominating the steady-state distributions.
    with torch.inference_mode():
        for _warmup_index in range(warmup_iterations):
            try:
                X = next(data_iterator)
            except StopIteration as exc:
                raise ValueError(
                    "The dataloader yielded too few samples to complete the "
                    "requested warm-up iterations."
                ) from exc

            X = X.to(device)
            _ = forward(model, X)
            if use_cuda:
                torch.cuda.synchronize(device)

        # Discard references to the final warm-up tensors while preserving the
        # CUDA caching allocator's warmed state.
        if warmup_iterations > 0:
            del X, _

        monitor = ResourceMonitor(device, resource_sample_interval_seconds)
        process = monitor.process
        cpu_count = max(os.cpu_count() or 1, 1)
        peak_process_cpu_percent: Optional[float] = None

        baseline_vram_allocated_bytes: Optional[int] = None
        baseline_vram_reserved_bytes: Optional[int] = None

        if use_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline_vram_allocated_bytes = int(torch.cuda.memory_allocated(device))
            baseline_vram_reserved_bytes = int(torch.cuda.memory_reserved(device))

        monitor.start()
        try:
            while True:
                # CPU deltas span the complete measured sample so the utilisation
                # metric corresponds to the same work as end-to-end latency.
                cpu_before = _process_cpu_seconds(process)

                # Start before requesting the next item. With num_workers=0 this
                # synchronously captures JPEG decoding, NumPy conversion,
                # Albumentations preprocessing, and DataLoader collation.
                overhead_start = time.perf_counter()
                try:
                    X = next(data_iterator)
                except StopIteration:
                    break

                # Device transfer is part of computational overhead rather than
                # model inference. Synchronisation makes CUDA transfer completion
                # fall inside this timing interval.
                X = X.to(device)
                if use_cuda:
                    torch.cuda.synchronize(device)
                overhead_end = time.perf_counter()

                # The forward-pass interval begins only after the model-ready
                # tensor is resident on the target device.
                inference_start = time.perf_counter()
                _ = forward(model, X)
                if use_cuda:
                    torch.cuda.synchronize(device)
                inference_end = time.perf_counter()
                cpu_after = _process_cpu_seconds(process)

                overhead_seconds = overhead_end - overhead_start
                inference_seconds = inference_end - inference_start
                end_to_end_seconds = overhead_seconds + inference_seconds

                overhead_latencies_ms.append(overhead_seconds * 1000.0)
                inference_latencies_ms.append(inference_seconds * 1000.0)
                end_to_end_latencies_ms.append(end_to_end_seconds * 1000.0)

                if (
                    cpu_before is not None
                    and cpu_after is not None
                    and end_to_end_seconds > 0.0
                ):
                    # 100% means one logical CPU core was fully occupied during
                    # this end-to-end sample. Multi-threaded preprocessing or CPU
                    # inference can exceed 100%, matching psutil's convention.
                    process_cpu_percent = max(
                        0.0,
                        (cpu_after - cpu_before) / end_to_end_seconds * 100.0,
                    )
                    peak_process_cpu_percent = (
                        process_cpu_percent
                        if peak_process_cpu_percent is None
                        else max(peak_process_cpu_percent, process_cpu_percent)
                    )
        finally:
            monitor.stop()

    if not end_to_end_latencies_ms:
        raise ValueError(
            "The dataloader yielded no measured samples after the warm-up phase."
        )

    resource_stats = monitor.summary()
    resource_stats["peak_process_cpu_percent"] = peak_process_cpu_percent
    resource_stats["peak_cpu_total_capacity_percent"] = (
        None
        if peak_process_cpu_percent is None
        else peak_process_cpu_percent / cpu_count
    )

    if use_cuda:
        mib = Configuration.BYTES_PER_MIB
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))

        resource_stats.update({
            "baseline_torch_vram_allocated_mb": (
                None
                if baseline_vram_allocated_bytes is None
                else baseline_vram_allocated_bytes / mib
            ),
            "peak_torch_vram_allocated_mb": peak_allocated_bytes / mib,
            "incremental_torch_vram_allocated_mb": (
                None
                if baseline_vram_allocated_bytes is None
                else max(0, peak_allocated_bytes - baseline_vram_allocated_bytes) / mib
            ),
            "baseline_torch_vram_reserved_mb": (
                None
                if baseline_vram_reserved_bytes is None
                else baseline_vram_reserved_bytes / mib
            ),
            "peak_torch_vram_reserved_mb": peak_reserved_bytes / mib,
            "incremental_torch_vram_reserved_mb": (
                None
                if baseline_vram_reserved_bytes is None
                else max(0, peak_reserved_bytes - baseline_vram_reserved_bytes) / mib
            ),
        })
    else:
        resource_stats.update({
            "baseline_torch_vram_allocated_mb": None,
            "peak_torch_vram_allocated_mb": None,
            "incremental_torch_vram_allocated_mb": None,
            "baseline_torch_vram_reserved_mb": None,
            "peak_torch_vram_reserved_mb": None,
            "incremental_torch_vram_reserved_mb": None,
        })

    # Keep the three lists aligned by sample index. ``end_to_end`` is constructed
    # from the other two lists instead of timing a third overlapping interval, so
    # it is mathematically equal to overhead + inference for every sample.
    latency_measurements = {
        "inference": inference_latencies_ms,
        "overhead": overhead_latencies_ms,
        "end_to_end": end_to_end_latencies_ms,
    }

    return latency_measurements, resource_stats


def compute_latency_statistics(latencies_ms: List[float]) -> Dict[str, float]:
    """Summarise a latency distribution using percentiles and the mean.

    Steps
    -----
    1. Convert the latency list to a NumPy array.
    2. Compute the requested percentiles (25, 50, 75, 90, 95, 99) and the mean.

    Parameters
    ----------
    latencies_ms : List[float]
        Per-image latency values in milliseconds for any one measured component.

    Returns
    -------
    Dict[str, float]
        Mapping of metric name to value with keys ``p25``, ``p50``, ``average``,
        ``p75``, ``p90``, ``p95`` and ``p99``, all in milliseconds.
    """
    latencies = np.array(latencies_ms, dtype=np.float64)

    return {
        "p25": float(np.percentile(latencies, 25)),
        "p50": float(np.percentile(latencies, 50)),
        "average": float(np.mean(latencies)),
        "p75": float(np.percentile(latencies, 75)),
        "p90": float(np.percentile(latencies, 90)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }


def print_latency_statistics(title: str, stats: Dict[str, float]) -> None:
    """Print one latency distribution using the benchmark's standard layout.

    Steps
    -----
    1. Print the supplied section title.
    2. Emit the percentile and mean values in a consistent order so inference,
       overhead, and end-to-end sections are directly comparable.

    Parameters
    ----------
    title : str
        Human-readable heading describing the latency component.
    stats : Dict[str, float]
        Statistics returned by :func:`compute_latency_statistics`.

    Returns
    -------
    None
        This function writes formatted values to standard output.
    """
    print(title)
    print(f"  P25      : {stats['p25']:.3f}")
    print(f"  P50      : {stats['p50']:.3f}  (median)")
    print(f"  Average  : {stats['average']:.3f}")
    print(f"  P75      : {stats['p75']:.3f}")
    print(f"  P90      : {stats['p90']:.3f}")
    print(f"  P95      : {stats['p95']:.3f}")
    print(f"  P99      : {stats['p99']:.3f}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for latency and resource evaluation.

    Steps
    -----
    1. Require the trained model checkpoint path.
    2. Accept an optional resource-monitor sampling interval in milliseconds.

    Returns
    -------
    argparse.Namespace
        Parsed checkpoint and resource-sampling arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Measure inference, computational-overhead, and end-to-end latency "
            "plus CPU/RAM/VRAM/temperature usage on the BDD100k test split."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the PyTorch checkpoint file (.pt) to evaluate.",
    )
    parser.add_argument(
        "--resource-sample-interval-ms",
        type=float,
        default=Configuration.RESOURCE_SAMPLE_INTERVAL_SECONDS * 1000.0,
        help=(
            "Background RAM/temperature/NVML sampling interval in milliseconds. "
            "Defaults to 250 ms. Lower values may perturb latency more."
        ),
    )
    return parser.parse_args()


def _format_optional(value: Optional[float], decimals: int = 2) -> str:
    """Format an optional numeric resource metric for console output."""
    return "N/A" if value is None else f"{value:.{decimals}f}"


def main() -> None:
    """Run the complete end-to-end latency and resource evaluation pipeline.

    Steps
    -----
    1. Parse options and load the trained segmentation checkpoint.
    2. Reconstruct the BDD100k test split and build an image-only synchronous
       DataLoader so preprocessing overhead is measurable per image.
    3. Measure inference, computational overhead, and their additive end-to-end
       latency while collecting resource peaks.
    4. Print latency distributions and resource-use summaries.

    Returns
    -------
    None
        Results are written to standard output.

    Raises
    ------
    ValueError
        If the requested resource sampling interval is not positive or latency
        measurement constraints are violated.
    """
    args = parse_arguments()

    if args.resource_sample_interval_ms <= 0:
        raise ValueError("--resource-sample-interval-ms must be greater than zero.")

    print("\n" + "=" * 76)
    print("END-TO-END LATENCY + RESOURCE OVERHEAD EVALUATION PIPELINE")
    print("=" * 76)

    print(f"Loading checkpoint from: {args.checkpoint}")
    model = load_trained_model(args.checkpoint, Configuration.DEVICE)
    print(f"Target execution device: {Configuration.DEVICE}")

    if Configuration.DEVICE.type == "cuda":
        device_index = (
            Configuration.DEVICE.index
            if Configuration.DEVICE.index is not None
            else torch.cuda.current_device()
        )
        print(f"CUDA device: {torch.cuda.get_device_name(device_index)}")

    if psutil is None:
        print(
            "WARNING: psutil is not installed; CPU utilisation, RAM, and CPU "
            "temperature will be reported as N/A. Install with: pip install psutil"
        )
    if Configuration.DEVICE.type == "cuda" and pynvml is None:
        print(
            "NOTE: pynvml is not installed; NVIDIA GPU temperature and NVML "
            "whole-device VRAM will be N/A. PyTorch VRAM peaks are still measured. "
            "Install with: pip install nvidia-ml-py"
        )

    print("Loading BDD100k test split...")
    test_df = load_test_split()
    print(f"Loaded {len(test_df)} test samples for evaluation.")

    inference_transforms = A.Compose([
        A.Resize(height=Configuration.IMAGE_HEIGHT, width=Configuration.IMAGE_WIDTH),
        ToTensorV2(),
    ])
    # Use the image-only dataset because ground-truth mask preparation belongs to
    # offline evaluation, not the runtime inference path being benchmarked.
    test_ds = BDDInferenceDataset(test_df, transform=inference_transforms)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=Configuration.LATENCY_BATCH_SIZE,
        shuffle=False,
        num_workers=Configuration.NUM_WORKERS,
    )

    latency_measurements, resources = measure_latency_ms(
        model,
        test_loader,
        Configuration.DEVICE,
        resource_sample_interval_seconds=args.resource_sample_interval_ms / 1000.0,
    )
    inference_stats = compute_latency_statistics(latency_measurements["inference"])
    overhead_stats = compute_latency_statistics(latency_measurements["overhead"])
    end_to_end_stats = compute_latency_statistics(latency_measurements["end_to_end"])

    print(
        f"\nMeasured {len(latency_measurements['end_to_end'])} samples (batch size "
        f"{Configuration.LATENCY_BATCH_SIZE}, after "
        f"{Configuration.WARMUP_ITERATIONS} warm-up iterations)."
    )
    print("-" * 76)
    print_latency_statistics(
        "Inference latency summary (forward pass only, ms):",
        inference_stats,
    )
    print("-" * 76)
    print_latency_statistics(
        "Computational overhead summary "
        "(decode + preprocess + collate + host-to-device, ms):",
        overhead_stats,
    )
    print("-" * 76)
    print_latency_statistics(
        "End-to-end latency summary (overhead + inference, ms):",
        end_to_end_stats,
    )

    print("-" * 76)
    print("Resource-overhead summary (steady-state measured phase):")
    print(
        "  Peak process CPU      : "
        f"{_format_optional(resources['peak_process_cpu_percent'])}% "
        "(100% = one logical CPU core)"
    )
    print(
        "  Peak CPU total capacity: "
        f"{_format_optional(resources['peak_cpu_total_capacity_percent'])}%"
    )
    print(
        "  Peak process RAM      : "
        f"{_format_optional(resources['peak_process_ram_mb'])} MiB"
    )
    print(
        "  Incremental RAM peak  : "
        f"{_format_optional(resources['incremental_process_ram_mb'])} MiB "
        "(above measured-phase baseline)"
    )
    print(
        "  Peak system RAM usage : "
        f"{_format_optional(resources['peak_system_ram_percent'])}%"
    )

    if Configuration.DEVICE.type == "cuda":
        print(
            "  Peak PyTorch VRAM alloc: "
            f"{_format_optional(resources['peak_torch_vram_allocated_mb'])} MiB"
        )
        print(
            "  Peak PyTorch VRAM reserv: "
            f"{_format_optional(resources['peak_torch_vram_reserved_mb'])} MiB"
        )
        print(
            "  Incremental VRAM alloc : "
            f"{_format_optional(resources['incremental_torch_vram_allocated_mb'])} MiB "
            "(above model/baseline allocation)"
        )
        print(
            "  Maximum GPU temp       : "
            f"{_format_optional(resources['max_gpu_temperature_c'])} °C"
        )
        print(
            "  Peak device VRAM (NVML): "
            f"{_format_optional(resources['peak_nvml_device_vram_used_mb'])} MiB "
            "(whole GPU; may include other processes)"
        )

    print("-" * 76)
    print(
        "Notes: CPU/GPU temperatures are best-effort sensor readings and may be "
        "N/A on unsupported systems. PyTorch VRAM is model-process attributable; "
        "NVML device VRAM is whole-device usage."
    )
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
