"""Latency evaluation script for trained semantic segmentation models.

This module loads a trained PyTorch segmentation checkpoint (a full ``nn.Module``
serialized by the training scripts) and measures the *inference* latency of the
model on the BDD100k test split. Latency is defined here as the total wall-clock
time between submitting a single input image to the model and receiving the
output logits, so it is measured per-image (batch size 1) rather than as batch
throughput.

The measured latencies are summarised on the console using common percentiles
(P25, P50/median, P75, P90, P95, P99) plus the mean, all reported in
milliseconds. A small warm-up phase is discarded so that one-off costs (CUDA
context initialisation, cuDNN algorithm autotuning, memory allocator warm-up)
do not distort the distribution.
"""

import argparse
import glob
import os
import time
from typing import Dict, List, Tuple, cast

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


# Shape aliases documented alongside each pipeline stage. ``jaxtyped`` +
# ``beartype`` enforce them at runtime, turning shape mismatches into clear
# ``TypeCheckError`` exceptions rather than confusing downstream broadcasts.
ImageTensor = Float[torch.Tensor, "3 h w"]  # single image, channel-first layout
MaskTensor = Float[torch.Tensor, "c h w"]  # one-hot mask, c == NUM_CLASSES
BatchImage = Float[torch.Tensor, "b 3 h w"]  # collated batch of images
Logits = Float[torch.Tensor, "b c h w"]  # model output, c == NUM_CLASSES


# BDD100k color-label palette. The row index is the class id (0-19), matching
# ``Configuration.NUM_CLASSES``. It is required only to turn the RGB color-label
# masks stored on disk into a single class id per pixel.
CLASS_COLORS = np.array([
    [128,  64, 128],   # 0  - Road
    [244,  35, 232],   # 1  - Sidewalk
    [ 70,  70,  70],   # 2  - Building
    [102, 102, 156],   # 3  - Wall
    [190, 153, 153],   # 4  - Fence
    [153, 153, 153],   # 5  - Pole
    [250, 170,  30],   # 6  - Traffic Light
    [220, 220,   0],   # 7  - Traffic Sign
    [107, 142,  35],   # 8  - Vegetation
    [152, 251, 152],   # 9  - Terrain
    [ 70, 130, 180],   # 10 - Sky
    [220,  20,  60],   # 11 - Person
    [255,   0,   0],   # 12 - Rider
    [  0,   0, 142],   # 13 - Car
    [  0,   0,  70],   # 14 - Truck
    [  0,  60, 100],   # 15 - Bus
    [  0,  80, 100],   # 16 - Train
    [  0,   0, 230],   # 17 - Motorcycle
    [119,  11,  32],   # 18 - Bicycle
    [  0,   0,   0],   # 19 - Unknown
], dtype=np.uint8)


class Configuration:
    """Latency evaluation constants and dataset image dimension specifications."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 20
    NUM_WORKERS = 2
    SEED = 768

    IMAGE_HEIGHT = 360
    IMAGE_WIDTH = 640
    CHANNELS = 3

    # Latency is measured per image, so inference is run with batch size 1.
    LATENCY_BATCH_SIZE = 1
    # Number of leading samples discarded before recording timings. These absorb
    # the one-off costs of device warm-up so the reported distribution reflects
    # steady-state inference only.
    WARMUP_ITERATIONS = 20


class Path:
    """Filesystem directory locations for BDD100k images and masks."""

    BASE = "./data/bdd100k"

    SEGMENTATION_MASK_LABEL_FOLDER = BASE + "/segmentation_maps/color_labels"
    SEGMENTATION_MASK_TRAIN_PATH = SEGMENTATION_MASK_LABEL_FOLDER + "/train"

    IMAGE_FOLDER = BASE + "/images_10k"
    IMAGE_TRAIN_PATH = IMAGE_FOLDER + "/train"


def color_label_to_class_index(label: np.ndarray) -> np.ndarray:
    """Map an RGB color-label image to a per-pixel class-index map.

    BDD100k stores segmentation masks as RGB PNGs whose colours are exactly the
    entries of ``CLASS_COLORS``. The dataset loader collapses this RGB
    representation into a single class id per pixel because the one-hot encoder
    and model expect integer class ids rather than RGB triples.

    Steps
    -----
    1. Initialise the output with the id of the last palette entry so unknown
       colours degrade to ``Unknown`` instead of producing an invalid index.
    2. For each palette colour, boolean-mask the pixels whose RGB values match
       it exactly and assign the corresponding class id.

    Parameters
    ----------
    label : np.ndarray
        RGB color-label array of shape ``(H, W, 3)`` with integer values.

    Returns
    -------
    np.ndarray
        Class-index array of shape ``(H, W)`` and dtype ``uint8``.
    """
    class_ids = np.full(label.shape[:2], CLASS_COLORS.shape[0] - 1, dtype=np.uint8)

    for class_id, (red, green, blue) in enumerate(CLASS_COLORS):
        match = (
            (label[..., 0] == red)
            & (label[..., 1] == green)
            & (label[..., 2] == blue)
        )
        class_ids[match] = class_id

    return class_ids


class BDDSegmentationDataset(Dataset[Tuple[ImageTensor, MaskTensor]]):
    """BDD100k semantic segmentation dataset loader.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``image_paths`` and ``mask_paths`` columns.
    transform : A.Compose | None, optional
        Albumentations transform pipeline applied to image and mask. Defaults
        to ``None``.
    """

    def __init__(self, df: pd.DataFrame, transform: A.Compose | None = None) -> None:
        super().__init__()

        self.image_paths: List[str] = df["image_paths"].to_list()
        self.mask_paths: List[str] = df["mask_paths"].to_list()
        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)

    @jaxtyped(typechecker=beartype)
    def __getitem__(self, index: int) -> Tuple[ImageTensor, MaskTensor]:
        """Return the transformed ``(image, mask)`` pair at position ``index``.

        ``ToTensorV2`` converts the image from ``(H, W, 3)`` to ``(3, H, W)``
        while leaving the 2-D class-index mask as ``(H, W)``. The mask is then
        one-hot encoded to ``(NUM_CLASSES, H, W)`` so the returned pair mirrors
        exactly what the model received during training and performance
        evaluation.

        Parameters
        ----------
        index : int
            Zero-based position of the sample to load.

        Returns
        -------
        Tuple[ImageTensor, MaskTensor]
            The ``(image, mask)`` pair where ``image`` has shape ``(3, H, W)``
            and ``mask`` is a one-hot tensor of shape ``(NUM_CLASSES, H, W)``.

        Raises
        ------
        IndexError
            If ``index`` is out of the range of the dataset path lists.
        """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Force RGB so that RGBA color-label PNGs with an alpha channel are
        # reduced to three channels before being processed.
        image_pil = Image.open(image_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("RGB")

        image = np.array(image_pil).astype(np.float32) / 255.0
        class_mask = color_label_to_class_index(np.array(mask_pil))

        if self.transform:
            transformed = self.transform(image=image, mask=class_mask)
        else:
            transformed = ToTensorV2()(image=image, mask=class_mask)

        # One-hot encode the (H, W) class ids into (NUM_CLASSES, H, W) floats
        # so the mask channel axis lines up with the model logits.
        mask_one_hot = torch.nn.functional.one_hot(
            transformed["mask"].to(torch.int64), Configuration.NUM_CLASSES
        ).permute(2, 0, 1).float()

        return transformed["image"], mask_one_hot


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
    ``(B, NUM_CLASSES, H, W)`` logits, which is where shape confusion most often
    arises.

    Parameters
    ----------
    model : nn.Module
        The segmentation model to run.
    x : BatchImage
        Input batch of images with shape ``(B, 3, H, W)``.

    Returns
    -------
    Logits
        Raw model logits with shape ``(B, NUM_CLASSES, H, W)``.
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


def measure_inference_latency_ms(
    model: nn.Module,
    dataloader: DataLoader[Tuple[ImageTensor, MaskTensor]],
    device: torch.device,
    warmup_iterations: int = Configuration.WARMUP_ITERATIONS,
) -> List[float]:
    """Measure per-image inference latency over a dataloader.

    Each image is passed through the model one at a time (batch size 1) and the
    wall-clock time of the forward pass is recorded after the device has been
    synchronised. On CUDA, synchronising before and after the forward pass
    guarantees the kernel queue is flushed so the measured interval includes the
    full GPU execution time rather than just the host-side launch cost.

    The first ``warmup_iterations`` samples are executed but not recorded so
    that device warm-up (CUDA context creation, cuDNN algorithm selection,
    memory allocator growth) does not inflate the reported distribution.

    Steps
    -----
    1. Switch the model into inference mode.
    2. For each batch, move the image to ``device``.
    3. Synchronise the device, clock a forward pass, and synchronise again.
    4. Skip the warm-up samples, then record the elapsed time in milliseconds.

    Parameters
    ----------
    model : nn.Module
        Segmentation model being benchmarked.
    dataloader : DataLoader[Tuple[ImageTensor, MaskTensor]]
        DataLoader yielding ``(image, mask)`` batches of size 1. The mask is
        ignored because latency does not depend on it.
    device : torch.device
        Device the inference runs on.
    warmup_iterations : int, optional
        Number of leading samples discarded before recording. Defaults to
        ``Configuration.WARMUP_ITERATIONS``.

    Returns
    -------
    List[float]
        Per-image inference latency in milliseconds, excluding warm-up samples.

    Raises
    ------
    ValueError
        If the dataloader yields fewer samples than ``warmup_iterations``, so no
        latency would be recorded.
    """
    model.eval()

    latencies_ms: List[float] = []
    use_cuda = device.type == "cuda"

    # inference_mode disables autograd and version counters, which both reduces
    # overhead and matches the runtime execution profile of deployment.
    with torch.inference_mode():
        for batch_index, (X, _) in enumerate(dataloader):
            X = X.to(device)

            # Synchronise so the start/end clocks bracket the actual GPU work.
            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = forward(model, X)
            if use_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()

            if batch_index >= warmup_iterations:
                latencies_ms.append((end - start) * 1000.0)

    if not latencies_ms:
        raise ValueError(
            "The dataloader yielded too few samples to record any latency "
            "after the warm-up phase."
        )

    return latencies_ms


def compute_latency_statistics(latencies_ms: List[float]) -> Dict[str, float]:
    """Summarise a latency distribution using percentiles and the mean.

    Steps
    -----
    1. Convert the latency list to a NumPy array.
    2. Compute the requested percentiles (25, 50, 75, 90, 95, 99) and the mean.

    Parameters
    ----------
    latencies_ms : List[float]
        Per-image inference latencies in milliseconds.

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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the latency evaluation script.

    Steps
    -----
    1. Configure an ArgumentParser with an optional checkpoint path argument.
    2. Parse and return the argument namespace.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Measure inference latency percentiles of a trained model on the BDD100k test split."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the PyTorch checkpoint file (.pt) to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint and report inference latency percentiles on the test split.

    Steps
    -----
    1. Parse the checkpoint argument.
    2. Load the trained model onto the target device.
    3. Load the BDD100k test partition with the training-compatible transforms.
    4. Build a batch-size-1 DataLoader.
    5. Measure per-image latency and log the percentile summary in milliseconds.
    """
    args = parse_arguments()

    print("\n" + "=" * 68)
    print("LATENCY EVALUATION PIPELINE")
    print("=" * 68)

    # Load the checkpoint with the whole nn.Module serialized by the trainers.
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = load_trained_model(args.checkpoint, Configuration.DEVICE)
    print(f"Target execution device: {Configuration.DEVICE}")

    # Recreate the same train/test partitioning used during training so the test
    # split benchmarked here is identical to the one held out at training time.
    print("Loading BDD100k test split...")
    test_df = load_test_split()
    print(f"Loaded {len(test_df)} test samples for evaluation.")

    # Use the same inference transforms as performance evaluation so the model
    # receives images at the resolution it was trained and evaluated on.
    inference_transforms = A.Compose([
        A.Resize(height=Configuration.IMAGE_HEIGHT, width=Configuration.IMAGE_WIDTH),
        ToTensorV2(),
    ])
    test_ds = BDDSegmentationDataset(test_df, transform=inference_transforms)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=Configuration.LATENCY_BATCH_SIZE,
        shuffle=False,  # deterministic ordering for reproducible measurement
        num_workers=Configuration.NUM_WORKERS,
    )

    latencies_ms = measure_inference_latency_ms(
        model, test_loader, Configuration.DEVICE
    )
    stats = compute_latency_statistics(latencies_ms)

    print(f"\nMeasured {len(latencies_ms)} samples (batch size "
          f"{Configuration.LATENCY_BATCH_SIZE}, after {Configuration.WARMUP_ITERATIONS} warm-up iterations).")
    print("-" * 68)
    print("Inference latency summary (ms):")
    print(f"  P25      : {stats['p25']:.3f}")
    print(f"  P50      : {stats['p50']:.3f}  (median)")
    print(f"  Average  : {stats['average']:.3f}")
    print(f"  P75      : {stats['p75']:.3f}")
    print(f"  P90      : {stats['p90']:.3f}")
    print(f"  P95      : {stats['p95']:.3f}")
    print(f"  P99      : {stats['p99']:.3f}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
