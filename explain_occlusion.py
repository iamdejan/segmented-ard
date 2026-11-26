"""Evaluate Grad-CAM++ road-segmentation faithfulness with input occlusion.

Commands: pixi run explain occlusion -- --checkpoint NAME=PATH [--checkpoint ...]

Sliding attribution uses captum.attr.Occlusion. Independent nonoverlapping tile
ablations and cumulative deletion use batched PyTorch inference. The primary
score is the global spatial mean of the target-class *raw logit*, matching
explain_gradcam.MeanClassSegmentationTarget exactly.

Deletion requires float32 .npy CAMs; colorized heatmap PNGs are not usable as
quantitative evidence. Robust inputs MUST be the exact saved PGD tensors from
the Grad-CAM run, not attacks regenerated in a separate execution.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union, cast

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, Float, Int64, Num, UInt8, jaxtyped
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr
from torch import Tensor, nn

# Set the backend before pyplot is imported; batch experiments run headlessly.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Shared names encode expected tensor dimensions and are checked at runtime by
# jaxtyping + beartype on numerical boundaries. Batches can grow for occlusion.
BatchImage = Float[Tensor, "batch 3 height width"]
SingleImage = Float[Tensor, "1 3 height width"]
SegmentationLogits = Float[Tensor, "batch classes height width"]
ClassMap = Int64[Tensor, "height width"]
BatchPredictions = Int64[Tensor, "batch height width"]
BatchScores = Float[Tensor, "batch"]
NumpyHeatmap = Float[np.ndarray, "height width"]
NumpyClassMap = Int64[np.ndarray, "height width"]
NumpyRGB = Float[np.ndarray, "height width 3"]
NumpyPalette = UInt8[np.ndarray, "classes 3"]
NumpyBoolMap = Bool[np.ndarray, "height width"]
# Define one-dimensional shapes outside the TypedDict: Ruff otherwise interprets
# "batch" in field annotations as an unresolved forward-referenced Python name.
# These aliases preserve the same jaxtyping dtype and shape constraints.
NumpyBatchScores = Float[np.ndarray, "batch"]
NumpyBatchPredictions = Int64[np.ndarray, "batch height width"]
NumericArray = Num[np.ndarray, "*dimensions"]


class InferenceMetrics(TypedDict):
    """Batched model evidence and predictions for one fixed segmentation target.

    Steps
    -----
    1. Preserve signed scalar outcomes for each perturbed input in a batch.
    2. Keep the corresponding dense predictions for independent IoU analysis.

    Attributes
    ----------
    global_logit : Float[np.ndarray, "batch"]
        Global mean target-class raw logits, one per image.
    gt_logit : Float[np.ndarray, "batch"]
        Target-class logits averaged over the fixed ground-truth region.
    road_iou : Float[np.ndarray, "batch"]
        Target-class segmentation IoU with ignored pixels excluded.
    prediction : Int64[np.ndarray, "batch height width"]
        Dense class predictions, retained for diagnostic figures.
    """

    global_logit: NumpyBatchScores
    gt_logit: NumpyBatchScores
    road_iou: NumpyBatchScores
    prediction: NumpyBatchPredictions


IGNORE_ID = 19
# These three metrics are always evaluated on the same reference image and
# fixed GT mask; keep their keys literal to help static TypedDict checking.
MetricName = Literal["global_logit", "gt_logit", "road_iou"]
METRIC_NAMES: Tuple[MetricName, ...] = ("global_logit", "gt_logit", "road_iou")
# White is the default perturbation selected on the CLI; blurred, constant-color,
# and noise baselines remain available for subsequent sensitivity comparisons.
BaselineMode = Literal["blur", "gray", "image_mean", "white", "bright", "noise"]
ConcreteInputMode = Literal["clean", "robust"]
ModelSpec = Tuple[str, str]
CamKey = Tuple[str, str, int, str, str]
CamIndex = Dict[CamKey, Dict[str, Any]]


class OcclusionConfiguration:
    """Hold the fixed experiment settings outside the user-facing CLI.

    Every checkpoint uses identical geometry, budgets, and random controls so
    the requested replacement baseline is the only configurable intervention.

    Steps
    -----
    1. Define stable project/output locations and semantic target defaults.
    2. Fix sliding, deletion, and random-control settings for all checkpoints.

    Attributes
    ----------
    PROJECT_ROOT : str
        Root directory used to resolve selected cases and checkpoint paths.
    OUTPUT_DIR : str
        Root directory for all occlusion outputs.
    TARGET_CLASS : str
        Semantic class to evaluate, normally road.
    PATCH_SIZE, STRIDE, TILE_SIZE : Tuple[int, int]
        Spatial dimensions for sliding windows and nonoverlapping deletion tiles.
    BLUR_KERNEL : int
        Odd Gaussian kernel width used by the blur replacement baseline.
    BLUR_SIGMA : float
        Standard deviation of the Gaussian blur replacement baseline.
    PERTURBATIONS_PER_EVAL : int
        Maximum number of perturbed images in each inference microbatch.
    MAX_CASES : Optional[int]
        Optional cap on expanded selected cases; None processes every case.
    BUDGETS : Tuple[float, ...]
        Fractions of image area removed during cumulative deletion.
    RANDOM_REPEATS : int
        Independent repetitions for random deletion policies.
    INTERIOR_THRESHOLD : float
        Minimum road coverage for road-interior tile classification.
    SLIDING_GROUND_TRUTH_TARGET : bool
        Whether to also generate the auxiliary fixed-GT-logit sensitivity map.
    SEED : int
        Shared seed for reproducible noise and random deletion controls.
    """

    PROJECT_ROOT: str = "."
    OUTPUT_DIR: str = "./model/xai_occlusion"
    TARGET_CLASS: str = "road"
    PATCH_SIZE: Tuple[int, int] = (32, 32)
    STRIDE: Tuple[int, int] = (32, 32)
    BLUR_KERNEL: int = 31
    BLUR_SIGMA: float = 8.0
    PERTURBATIONS_PER_EVAL: int = 4
    MAX_CASES: Optional[int] = None
    TILE_SIZE: Tuple[int, int] = (32, 32)
    BUDGETS: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)
    RANDOM_REPEATS: int = 10
    INTERIOR_THRESHOLD: float = 0.8
    SLIDING_GROUND_TRUTH_TARGET: bool = False
    SEED: int = 1234


def resolve_path(path: Union[str, Path], root: Path) -> Path:
    """Resolve an input or artifact path relative to the repository root.

    Steps
    -----
    1. Expand the user-home shortcut in the supplied path.
    2. Anchor relative paths at the project root and resolve the resulting path.

    Parameters
    ----------
    path : Union[str, Path]
        Filesystem path, absolute or relative to the project root.
    root : Path
        Resolved project root used to interpret relative paths.

    Returns
    -------
    Path
        Resolved absolute filesystem path.
    """
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


@jaxtyped(typechecker=beartype)
def sha256_array(data: NumericArray) -> str:
    """Compute a reproducible hash of the raw bytes of a NumPy array.

    Steps
    -----
    1. Make the array C-contiguous to remove dependence on view strides.
    2. Hash its bytes for manifest and artifact provenance.

    Parameters
    ----------
    data : NumericArray
        Numeric NumPy array whose contiguous bytes are hashed.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of the array contents.
    """
    contiguous = np.ascontiguousarray(data)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


@jaxtyped(typechecker=beartype)
def tensor_hash(image: SingleImage) -> str:
    """Fingerprint a tensor to verify that the exact explained input is reused.

    Steps
    -----
    1. Detach and move the image to CPU without changing its values.
    2. Hash its contiguous NumPy representation.

    Parameters
    ----------
    image : SingleImage
        Reference NCHW RGB image tensor, normally a single example.

    Returns
    -------
    str
        SHA-256 digest of the image tensor bytes.
    """
    return sha256_array(image.detach().cpu().contiguous().numpy())


@jaxtyped(typechecker=beartype)
def global_mean_logits(logits: SegmentationLogits, class_id: int) -> BatchScores:
    """Calculate the exact globally averaged raw-logit target used by Grad-CAM++.

    Steps
    -----
    1. Verify that the model output is four-dimensional and the class exists.
    2. Average the selected class channel over every output pixel for each batch item.

    Parameters
    ----------
    logits : SegmentationLogits
        Raw segmentation logits in NCHW layout.
    class_id : int
        Zero-based target class channel, usually road (0).

    Returns
    -------
    BatchScores
        One unmasked target-class mean logit per input image.

    Raises
    ------
    ValueError
        If logits are not NCHW or the requested class is unavailable.
    """
    # Logits, not probabilities, must be reduced: the original Grad-CAM++
    # explanation targets this unmasked global response across every pixel.
    if logits.ndim != 4 or not 0 <= class_id < logits.shape[1]:
        raise ValueError(f"Invalid logits/target-class shape: {tuple(logits.shape)}, {class_id}")
    return logits[:, class_id].mean(dim=(1, 2))


@jaxtyped(typechecker=beartype)
def gt_region_logits(logits: SegmentationLogits, class_id: int, class_map: ClassMap) -> BatchScores:
    """Score a class only at its fixed ground-truth pixels for secondary diagnosis.

    Steps
    -----
    1. Select valid target-class pixels from the unchanged ground-truth map.
    2. Average each image’s target-channel logits over those same pixels.

    Parameters
    ----------
    logits : SegmentationLogits
        Raw segmentation logits in NCHW layout.
    class_id : int
        Zero-based target class channel, usually road (0).
    class_map : ClassMap
        Fixed two-dimensional ground-truth class ID map.

    Returns
    -------
    BatchScores
        Per-image target-class mean logits within the fixed GT region.

    Raises
    ------
    ValueError
        If there are no GT target pixels or logits and mask shapes differ.
    """
    # Fix the road support from GT before perturbation. Reusing predicted masks
    # here would change the scalar being evaluated after each intervention.
    mask = (class_map == class_id) & (class_map != IGNORE_ID)
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("There are no ground-truth target pixels for this case.")
    if tuple(mask.shape) != tuple(logits.shape[-2:]):
        raise ValueError("Ground-truth and logits spatial dimensions do not match.")
    return (logits[:, class_id] * mask[None]).sum(dim=(1, 2)) / count


@jaxtyped(typechecker=beartype)
def target_iou(predictions: Union[ClassMap, BatchPredictions], class_map: ClassMap, class_id: int) -> BatchScores:
    """Compute per-image target-class IoU while excluding ignored GT pixels.

    Steps
    -----
    1. Promote a single prediction map to a batch when necessary.
    2. Count valid intersection and union pixels; use NaN for an empty union.

    Parameters
    ----------
    predictions : Union[ClassMap, BatchPredictions]
        Two- or three-dimensional argmax semantic predictions.
    class_map : ClassMap
        Fixed two-dimensional ground-truth class ID map.
    class_id : int
        Zero-based target class channel, usually road (0).

    Returns
    -------
    BatchScores
        One target-class IoU per prediction; undefined unions give NaN.
    """
    if predictions.ndim == 2:
        predictions = predictions.unsqueeze(0)
    valid = class_map != IGNORE_ID
    truth = class_map == class_id
    positive = predictions == class_id
    tp = (positive & truth & valid).sum(dim=(-2, -1)).float()
    union = ((positive | truth) & valid).sum(dim=(-2, -1))
    return torch.where(union > 0, tp / union.clamp(min=1), torch.nan)


@jaxtyped(typechecker=beartype)
@torch.no_grad()
def infer_metrics(model: nn.Module, images: BatchImage, class_map: ClassMap, class_id: int) -> InferenceMetrics:
    """Evaluate logits and segmentation quality for a batch of perturbations.

    Steps
    -----
    1. Run the frozen segmentation model without gradient tracking.
    2. Compute global logits, GT-region logits, IoU, and class predictions on CPU.

    Parameters
    ----------
    model : nn.Module
        Loaded evaluation-mode segmentation network.
    images : BatchImage
        NCHW batch of RGB images or independently occluded variants.
    class_map : ClassMap
        Fixed two-dimensional ground-truth class ID map.
    class_id : int
        Zero-based target class channel, usually road (0).

    Returns
    -------
    InferenceMetrics
        Named arrays of global score, GT score, IoU, and dense predictions.

    Raises
    ------
    TypeError
        If the model does not return a tensor.
    ValueError
        If output resolution differs from the ground-truth mask.
    """
    logits = model(images)
    if not isinstance(logits, Tensor):
        raise TypeError("Model must return an NCHW segmentation-logit Tensor.")
    if tuple(logits.shape[-2:]) != tuple(class_map.shape):
        raise ValueError("Model logits and ground-truth mask must have matching dimensions.")
    pred = logits.argmax(dim=1)
    return {
        "global_logit": global_mean_logits(logits, class_id).cpu().numpy(),
        "gt_logit": gt_region_logits(logits, class_id, class_map).cpu().numpy(),
        "road_iou": target_iou(pred, class_map, class_id).cpu().numpy(),
        "prediction": pred.cpu().numpy(),
    }


@jaxtyped(typechecker=beartype)
def make_score_forward(model: nn.Module, class_id: int, class_map: Optional[ClassMap] = None) -> Callable[[BatchImage], BatchScores]:
    """Build a batch-safe scalar forward function for Captum.

    Steps
    -----
    1. Retain the model, requested class, and optional fixed ground-truth mask.
    2. Return a callable that reduces each batch item to its target score without gradients.

    Parameters
    ----------
    model : nn.Module
        Loaded evaluation-mode segmentation network.
    class_id : int
        Zero-based target class channel, usually road (0).
    class_map : Optional[ClassMap]
        Fixed two-dimensional ground-truth class ID map.

    Returns
    -------
    Callable[[BatchImage], BatchScores]
        Callable taking RGB image batches and returning one target score per image.
    """
    @jaxtyped(typechecker=beartype)
    def forward(images: BatchImage) -> BatchScores:
        """Compute the selected scalar response for every Captum perturbation.

        Steps
        -----
        1. Run the model on the incoming batch without gradient tracking.
        2. Reduce logits globally or over the fixed GT region, according to the closure.

        Parameters
        ----------
        images : BatchImage
            NCHW batch of RGB images or independently occluded variants.

        Returns
        -------
        BatchScores
            One scalar score per perturbed batch element.
        """
        with torch.no_grad():
            logits = model(images)
            if class_map is None:
                return global_mean_logits(logits, class_id)
            return gt_region_logits(logits, class_id, class_map)
    return forward


@jaxtyped(typechecker=beartype)
def gaussian_baseline(image: SingleImage, kernel_size: int, sigma: float) -> SingleImage:
    """Blur an RGB input to construct a structure-preserving replacement image.

    Steps
    -----
    1. Validate the Gaussian kernel geometry and standard deviation.
    2. Construct a normalized separable Gaussian filter and apply it channel-wise.

    Parameters
    ----------
    image : SingleImage
        Reference NCHW RGB image tensor, normally a single example.
    kernel_size : int
        Odd width/height of the Gaussian blur kernel.
    sigma : float
        Positive Gaussian standard deviation in image pixels.

    Returns
    -------
    SingleImage
        Blurred RGB image with the original shape, dtype, and device.

    Raises
    ------
    ValueError
        If the kernel size is not odd and at least three, or sigma is nonpositive.
    """
    if kernel_size < 3 or kernel_size % 2 != 1 or sigma <= 0:
        raise ValueError("Gaussian kernel must be odd and >= 3; sigma must be positive.")
    coords = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)
    groups = image.shape[1]
    weights = kernel[None, None].expand(groups, 1, -1, -1)
    pad = kernel_size // 2
    padding_mode = "reflect" if min(image.shape[-2:]) > pad else "replicate"
    padded = F.pad(image, (pad, pad, pad, pad), mode=padding_mode)
    return F.conv2d(padded, weights, groups=groups)


@jaxtyped(typechecker=beartype)
def image_mean_baseline(image: SingleImage) -> SingleImage:
    """Fill an input with its own per-channel mean color.

    This is a controlled constant-color alternative to neutral gray. It adapts to
    the brightness and color balance of each scene, but still removes all local
    spatial evidence inside an occluded tile. The baseline is useful for
    sensitivity analysis, while Gaussian blur remains the recommended default.

    Steps
    -----
    1. Average each RGB channel over the full spatial extent of the reference image.
    2. Expand those three means back to the original image shape without changing
       dtype or device.

    Parameters
    ----------
    image : SingleImage
        Reference NCHW RGB image tensor with one batch item.

    Returns
    -------
    SingleImage
        Spatially constant RGB tensor whose channel values match the reference
        image's global channel means.
    """
    channel_means = image.mean(dim=(-2, -1), keepdim=True)
    return channel_means.expand_as(image).clone()


def derive_noise_seed(base_seed: int, image_index: int) -> int:
    """Derive a stable per-image random seed for the colored-noise baseline.

    Using one global noise realization for every source image would repeatedly
    place identical random structures at the same coordinates. A per-image seed
    avoids that unwanted spatial coupling while preserving exact reproducibility
    for repeated runs and paired checkpoint comparisons.

    Steps
    -----
    1. Combine the user-provided base seed with the stable dataset image index.
    2. Hash the pair so nearby image indices do not produce trivially related seeds.
    3. Restrict the result to PyTorch's nonnegative 63-bit seed range.

    Parameters
    ----------
    base_seed : int
        Global reproducibility seed supplied through ``--seed``.
    image_index : int
        Stable dataset/test-split image index used to identify the source image.

    Returns
    -------
    int
        Deterministic nonnegative seed used only for the colored-noise baseline.
    """
    payload = f"{base_seed}:{image_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


@jaxtyped(typechecker=beartype)
def uniform_noise_baseline(image: SingleImage, seed: int) -> SingleImage:
    """Generate a reproducible uniform colored-noise replacement image.

    Each RGB value is sampled independently from ``Uniform(0, 1)``. Independent
    channel sampling is important: using one random value shared across RGB would
    create grayscale noise rather than the requested colored noise. The complete
    noise tensor is generated once and then reused for every tile and deletion
    budget in the case so the occlusion baseline itself does not change between
    interventions.

    Steps
    -----
    1. Create a private CPU random generator and initialize it with ``seed``.
    2. Sample one independent value in ``[0, 1)`` for every RGB pixel/channel.
    3. Move the noise tensor to the reference image's device and dtype.

    Parameters
    ----------
    image : SingleImage
        Reference NCHW RGB tensor whose shape, dtype, and device are preserved.
    seed : int
        Deterministic seed controlling this case's colored-noise realization.

    Returns
    -------
    SingleImage
        RGB tensor with independent uniformly distributed values in ``[0, 1)``.
    """
    # Generate on CPU with a private RNG so creating the baseline does not consume
    # CUDA/global random state that is also used by other experiment components.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.rand(
        tuple(image.shape),
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    )
    return cast(
        SingleImage,
        noise.to(device=image.device, dtype=image.dtype),
    )


@jaxtyped(typechecker=beartype)
def replacement_baseline(
    image: SingleImage,
    mode: BaselineMode,
    blur_kernel: int,
    blur_sigma: float,
    noise_seed: int = 0,
) -> SingleImage:
    """Select the replacement image used by every occlusion policy.

    White is the default, high-contrast occluder for the road class. Gaussian blur
    preserves coarse scene structure, while gray, bright, noise, and image_mean
    provide alternative controls for baseline-sensitivity experiments.

    Steps
    -----
    1. Validate the NCHW RGB input and its expected ``[0, 1]`` intensity domain.
    2. Return a blurred, constant-color, colored-noise, or image-adaptive replacement.

    Parameters
    ----------
    image : SingleImage
        Reference NCHW RGB image tensor, normally a single example.
    mode : BaselineMode
        Replacement strategy: ``blur``, ``gray``, ``white``, ``bright``, ``noise``, or ``image_mean``.
    blur_kernel : int
        Odd Gaussian kernel size used only for ``blur`` mode.
    blur_sigma : float
        Positive Gaussian standard deviation used only for ``blur`` mode.
    noise_seed : int, optional
        Random seed used only for ``noise`` mode. Defaults to ``0``.

    Returns
    -------
    SingleImage
        RGB replacement tensor with the same shape, dtype, and device as ``image``.

    Raises
    ------
    ValueError
        If the input is malformed, outside the expected value range, or the
        baseline mode is unknown.
    """
    if not torch.isfinite(image).all() or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected a finite NCHW RGB tensor.")
    if float(image.min()) < -1e-5 or float(image.max()) > 1 + 1e-5:
        raise ValueError("Expected [0, 1] image tensors; check model preprocessing.")

    # Blurring is an alternate control that suppresses local discriminative
    # structure without replacing it with an out-of-distribution solid color.
    if mode == "blur":
        return gaussian_baseline(image, blur_kernel, blur_sigma)
    if mode == "gray":
        return torch.full_like(image, 0.5)
    if mode == "white":
        # A saturated bright patch strongly removes road-like appearance and is a
        # useful ablation when the user wants a deliberately conspicuous occluder.
        return torch.ones_like(image)
    if mode == "bright":
        # A slightly off-white patch preserves the "bright occluder" intention while
        # avoiding exact saturation at every pixel.
        return torch.full_like(image, 0.9)
    if mode == "noise":
        # Colored noise intentionally destroys local texture and color continuity.
        # One full-image realization is reused across all interventions in this case.
        return uniform_noise_baseline(image, noise_seed)
    if mode == "image_mean":
        return image_mean_baseline(image)
    raise ValueError(f"Unknown baseline: {mode}")


@jaxtyped(typechecker=beartype)
def sliding_occlusion(
    model: nn.Module, image: SingleImage, class_id: int, class_map: ClassMap,
    patch: Tuple[int, int], stride: Tuple[int, int], baseline: SingleImage,
    batch_size: int, gt_target: bool = False,
) -> NumpyHeatmap:
    """Compute Captum sliding-window sensitivity for a selected class.

    Steps
    -----
    1. Validate full RGB window coverage and patch/stride dimensions.
    2. Call Captum with a batched scalar scoring function and a fixed replacement tensor.
    3. Check channel agreement and export the signed spatial attribution map.

    Parameters
    ----------
    model : nn.Module
        Loaded evaluation-mode segmentation network.
    image : SingleImage
        Reference NCHW RGB image tensor, normally a single example.
    class_id : int
        Zero-based target class channel, usually road (0).
    class_map : ClassMap
        Fixed two-dimensional ground-truth class ID map.
    patch : Tuple[int, int]
        Spatial height and width of each Captum occlusion window.
    stride : Tuple[int, int]
        Spatial height and width of sliding-window displacement.
    baseline : SingleImage
        RGB replacement tensor aligned with the reference input.
    batch_size : int
        Number of perturbed images processed in one inference batch.
    gt_target : bool
        If True, score fixed GT-class pixels rather than all pixels.

    Returns
    -------
    NumpyHeatmap
        Signed occlusion sensitivity averaged across joint RGB-channel effects.

    Raises
    ------
    ImportError
        If Captum is unavailable in the environment.
    ValueError
        If a patch is too large or a stride would leave coverage gaps.
    RuntimeError
        If Captum returns incompatible, nonfinite, or nonmatching channel attributions.
    """
    try:
        from captum.attr import Occlusion
    except ImportError as exc:
        raise ImportError("Install Captum (pixi install) to use sliding occlusion.") from exc
    if any(s > p for s, p in zip(stride, patch, strict=True)):
        raise ValueError("Sliding stride cannot exceed patch size (would leave uncovered gaps).")
    if any(p > dimension for p, dimension in zip(patch, image.shape[-2:], strict=True)):
        raise ValueError("Sliding patch dimensions cannot exceed input height/width.")
    forward = make_score_forward(model, class_id, class_map if gt_target else None)
    attribution = Occlusion(forward).attribute(
        image,
        sliding_window_shapes=(3, *patch),
        strides=(3, *stride),
        baselines=baseline,
        perturbations_per_eval=batch_size,
        show_progress=False,
    )
    if tuple(attribution.shape) != tuple(image.shape):
        raise RuntimeError("Unexpected Captum Occlusion result shape.")
    # Every occlusion jointly replaces all RGB channels. Their resulting
    # attribution entries should be identical (within floating point rounding).
    channels = attribution[0].detach()
    if not torch.allclose(channels, channels[0:1].expand_as(channels), atol=1e-4, rtol=1e-4):
        raise RuntimeError("Expected all-channel occlusion effects to agree.")
    spatial = channels.mean(dim=0).cpu().numpy().astype(np.float32)
    if not np.isfinite(spatial).all():
        raise RuntimeError("Occlusion returned nonfinite attributions.")
    return spatial


@dataclass(frozen=True)
class Tile:
    """Describe one fixed evaluation tile and its semantic overlap metadata.

    The image is partitioned before occlusion; these immutable descriptors keep
    CAM ranking, perturbation geometry, and area accounting in sync.

    Steps
    -----
    1. Store each tile's pixel bounds and average CAM evidence.
    2. Retain road-occupancy, boundary, and vertical-band metadata for controls.

    Attributes
    ----------
    index : int
        Zero-based index in deterministic raster-scan order.
    y0, y1, x0, x1 : int
        Half-open spatial bounds, including partial edge tiles.
    cam_mean : float
        Mean normalized Grad-CAM++ value within the tile.
    road_fraction : float
        Fraction of tile pixels with road ground-truth labels.
    boundary_fraction : float
        Fraction adjacent to a valid road/non-road transition.
    vertical_band : int
        Coarse perspective band used in spatially matched random controls.
    region : str
        Semantic region category: interior, boundary, or non-road context.
    """

    index: int
    y0: int
    y1: int
    x0: int
    x1: int
    cam_mean: float
    road_fraction: float
    boundary_fraction: float
    vertical_band: int
    region: str

    @property
    def area(self) -> int:
        """Report the exact pixel area of a possibly partial edge tile.

        Steps
        -----
        1. Subtract each tile’s lower bound from its upper bound.
        2. Multiply height and width, avoiding assumptions about full-sized edge tiles.

        Returns
        -------
        int
            Number of spatial pixels covered by this tile.
        """
        return (self.y1 - self.y0) * (self.x1 - self.x0)


@jaxtyped(typechecker=beartype)
def road_boundary_mask(class_map: ClassMap, class_id: int) -> NumpyBoolMap:
    """Locate valid target-versus-nontarget ground-truth transitions.

    Steps
    -----
    1. Compute differences between horizontal and vertical adjacent road indicators.
    2. Mark both sides of each transition, excluding ignored-label neighbors.

    Parameters
    ----------
    class_map : ClassMap
        Fixed two-dimensional ground-truth class ID map.
    class_id : int
        Zero-based target class channel, usually road (0).

    Returns
    -------
    NumpyBoolMap
        Boolean spatial mask identifying pixels adjacent to road boundaries.
    """
    gt = class_map.detach().cpu().numpy()
    road = gt == class_id
    valid = gt != IGNORE_ID
    boundary = np.zeros_like(road)
    horiz = (road[:, :-1] != road[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    vert = (road[:-1, :] != road[1:, :]) & valid[:-1, :] & valid[1:, :]
    boundary[:, :-1] |= horiz
    boundary[:, 1:] |= horiz
    boundary[:-1] |= vert
    boundary[1:] |= vert
    return boundary


@jaxtyped(typechecker=beartype)
def build_tiles(
    cam_map: NumpyHeatmap, class_map: ClassMap, class_id: int,
    tile_size: Tuple[int, int], interior_threshold: float,
) -> List[Tile]:
    """Partition CAM and GT maps into nonoverlapping evaluation tiles.

    Steps
    -----
    1. Check CAM/ground-truth alignment and derive the semantic boundary mask.
    2. Build full and partial edge tiles, measuring CAM importance and road overlap.
    3. Classify each tile by semantic region and coarse vertical position.

    Parameters
    ----------
    cam_map : NumpyHeatmap
        Finite numerical Grad-CAM++ attribution at input resolution.
    class_map : ClassMap
        Fixed two-dimensional ground-truth class ID map.
    class_id : int
        Zero-based target class channel, usually road (0).
    tile_size : Tuple[int, int]
        Height and width of a nonoverlapping deletion tile.
    interior_threshold : float
        Minimum road-pixel fraction for a road-interior tile.

    Returns
    -------
    List[Tile]
        Tile metadata in a deterministic row-major order.

    Raises
    ------
    ValueError
        If CAM and GT spatial dimensions do not match.
    """
    h, w = cam_map.shape
    if tuple(class_map.shape) != (h, w):
        raise ValueError("Numerical CAM and ground-truth mask have different sizes.")
    gt = class_map.cpu().numpy()
    boundary = road_boundary_mask(class_map, class_id)
    tiles: List[Tile] = []
    # Tiles must span non-road context as well as road pixels: segmentation
    # networks can rely on vehicles, curb edges, and scene layout to infer road.
    for y in range(0, h, tile_size[0]):
        for x in range(0, w, tile_size[1]):
            y1, x1 = min(y + tile_size[0], h), min(x + tile_size[1], w)
            crop = gt[y:y1, x:x1]
            boundary_fraction = float(boundary[y:y1, x:x1].mean())
            road_fraction = float((crop == class_id).mean())
            if boundary_fraction > 0:
                region = "road_boundary"
            elif road_fraction >= interior_threshold:
                region = "road_interior"
            else:
                region = "non_road_context"
            tiles.append(Tile(
                len(tiles), y, y1, x, x1,
                float(cam_map[y:y1, x:x1].mean()), road_fraction,
                boundary_fraction, min(3, int(4 * ((y + y1) / 2) / h)), region,
            ))
    return tiles


@jaxtyped(typechecker=beartype)
def apply_tiles(image: SingleImage, baseline: SingleImage, tiles: Sequence[Tile], chosen: Sequence[int]) -> SingleImage:
    """Replace precisely the selected tile regions without modifying the original image.

    Steps
    -----
    1. Clone the reference tensor to keep the base image unchanged.
    2. Replace every selected rectangle across all RGB channels with baseline values.

    Parameters
    ----------
    image : SingleImage
        Reference NCHW RGB image tensor, normally a single example.
    baseline : SingleImage
        RGB replacement tensor aligned with the reference input.
    tiles : Sequence[Tile]
        Deterministic tile metadata spanning the input image.
    chosen : Sequence[int]
        Indices of tiles whose RGB pixels will be replaced.

    Returns
    -------
    SingleImage
        New tensor with only the selected regions replaced.
    """
    # Every policy starts from the identical reference rather than stacking
    # interventions from a previous, potentially different policy.
    out = image.clone()
    for i in chosen:
        tile = tiles[int(i)]
        out[..., tile.y0:tile.y1, tile.x0:tile.x1] = baseline[..., tile.y0:tile.y1, tile.x0:tile.x1]
    return out


@jaxtyped(typechecker=beartype)
def tile_effects(
    model: nn.Module, image: SingleImage, baseline: SingleImage, class_map: ClassMap,
    class_id: int, tiles: Sequence[Tile], batch_size: int,
    original: InferenceMetrics,
) -> pd.DataFrame:
    """Measure independent, signed output changes from removing each tile.

    Steps
    -----
    1. Form microbatches in which exactly one tile is replaced per image.
    2. Run batched inference and subtract each perturbed metric from its original value.
    3. Return one row per tile, including metadata and signed target/IoU changes.

    Parameters
    ----------
    model : nn.Module
        Loaded evaluation-mode segmentation network.
    image : SingleImage
        Reference NCHW RGB image tensor, normally a single example.
    baseline : SingleImage
        RGB replacement tensor aligned with the reference input.
    class_map : ClassMap
        Fixed two-dimensional ground-truth class ID map.
    class_id : int
        Zero-based target class channel, usually road (0).
    tiles : Sequence[Tile]
        Deterministic tile metadata spanning the input image.
    batch_size : int
        Number of perturbed images processed in one inference batch.
    original : InferenceMetrics
        Metrics computed on the unchanged reference image.

    Returns
    -------
    pd.DataFrame
        Table containing tile metadata and independent logit/IoU deltas.
    """
    rows: List[Dict[str, Any]] = []
    # Independently removing one tile at a time makes the signed differences
    # directly comparable with a CAM ranking (unlike overlapping CAM windows).
    for start in range(0, len(tiles), batch_size):
        group = tiles[start:start + batch_size]
        perturbed = torch.cat([apply_tiles(image, baseline, tiles, [t.index]) for t in group])
        metrics = infer_metrics(model, perturbed, class_map, class_id)
        for j, tile in enumerate(group):
            row = asdict(tile)
            row["area"] = tile.area
            for metric in METRIC_NAMES:
                row[f"{metric}_occluded"] = float(metrics[metric][j])
                row[f"delta_{metric}"] = float(original[metric][0] - metrics[metric][j])
            rows.append(row)
        # A completed tile microbatch must not retain GPU inputs while the next
        # one is allocated, especially on an 8 GB accelerator.
        del perturbed, metrics
    return pd.DataFrame(rows)


def deletion_order(tiles: Sequence[Tile], policy: str, seed: int = 0) -> List[int]:
    """Generate deterministic Grad-CAM++, random, or road-matched tile rankings.

    Steps
    -----
    1. Sort tile indices by CAM for top-first or bottom-first policies.
    2. Randomize uniformly for the plain random control.
    3. For the spatially matched control, greedily match CAM-ranked road coverage and vertical band without using tile effects.

    Parameters
    ----------
    tiles : Sequence[Tile]
        Deterministic tile metadata spanning the input image.
    policy : str
        Deletion ranking policy (top, bottom, random, or matched_random).
    seed : int
        Random seed controlling repeatable ordering or bootstrap sampling.

    Returns
    -------
    List[int]
        Permutation of tile indices describing the deletion order.

    Raises
    ------
    ValueError
        If the requested ranking policy is unsupported.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(tiles))
    if policy == "top":
        return sorted(indices.tolist(), key=lambda i: (-tiles[i].cam_mean, i))
    if policy == "bottom":
        return sorted(indices.tolist(), key=lambda i: (tiles[i].cam_mean, i))
    if policy == "random":
        return rng.permutation(indices).tolist()
    if policy == "matched_random":
        top = deletion_order(tiles, "top")
        available = set(range(len(tiles)))
        chosen: List[int] = []
        # Greedy matching by road fraction and vertical position; random selection
        # among the closest available matches. Never use the measured tile effects.
        for _pos, reference_idx in enumerate(top):
            reference = tiles[reference_idx]
            scores = []
            for candidate_idx in sorted(available):
                candidate = tiles[candidate_idx]
                cost = (abs(candidate.road_fraction - reference.road_fraction)
                        + 0.25 * abs(candidate.vertical_band - reference.vertical_band)
                        + 0.05 * abs(candidate.area - reference.area) / max(1, reference.area))
                # Avoid selecting the same top tile when a comparably matched
                # alternative exists. This is a soft preference, not a constraint.
                if candidate_idx == reference_idx:
                    cost += 0.1
                scores.append((cost, candidate_idx))
            scores.sort()
            best_cost = scores[0][0]
            close = [i for cost, i in scores if cost <= best_cost + 0.10 + 1e-12]
            choice = int(rng.choice(close))
            chosen.append(choice)
            available.remove(choice)
        return chosen
    raise ValueError(f"Unsupported deletion policy: {policy}")


def selected_at_budget(tiles: Sequence[Tile], ranking: Sequence[int], fraction: float, total_area: int) -> List[int]:
    """Choose whole tiles closest to a requested cumulative area budget.

    Steps
    -----
    1. Walk a precomputed ranking while accumulating the actual pixel areas.
    2. Keep the closer of the final two complete-tile allocations.

    Parameters
    ----------
    tiles : Sequence[Tile]
        Deterministic tile metadata spanning the input image.
    ranking : Sequence[int]
        Precomputed permutation of tile indices for deletion.
    fraction : float
        Requested fraction of total image area to delete.
    total_area : int
        Total number of input spatial pixels.

    Returns
    -------
    List[int]
        Indices selected for approximately the requested area fraction.
    """
    if fraction <= 0:
        return []
    goal = fraction * total_area
    cumulative = 0
    for count, tile_index in enumerate(ranking, start=1):
        next_area = cumulative + tiles[tile_index].area
        if next_area >= goal:
            if count > 1 and abs(cumulative - goal) < abs(next_area - goal):
                return list(ranking[:count - 1])
            return list(ranking[:count])
        cumulative = next_area
    return list(ranking)


@jaxtyped(typechecker=beartype)
def ordered_deletion(
    model: nn.Module, image: SingleImage, baseline: SingleImage, class_map: ClassMap, class_id: int,
    tiles: Sequence[Tile], order: Sequence[int], policy: str, repeat: int, budgets: Sequence[float],
    original: InferenceMetrics, batch_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Tensor]]:
    """Evaluate cumulative area-controlled deletions for one tile ranking.

    Steps
    -----
    1. Compute tile selections for every budget without materializing GPU images.
    2. Generate only one microbatch of variants at a time for inference.
    3. Move top-first previews to CPU and record the area-controlled metrics.

    Parameters
    ----------
    model : nn.Module
        Loaded evaluation-mode segmentation network.
    image : SingleImage
        Reference NCHW RGB image tensor, normally a single example.
    baseline : SingleImage
        RGB replacement tensor aligned with the reference input.
    class_map : ClassMap
        Fixed two-dimensional ground-truth class ID map.
    class_id : int
        Zero-based target class channel, usually road (0).
    tiles : Sequence[Tile]
        Deterministic tile metadata spanning the input image.
    order : Sequence[int]
        Ranking of tiles under the current deletion policy.
    policy : str
        Deletion ranking policy (top, bottom, random, or matched_random).
    repeat : int
        Index of this random-control replication.
    budgets : Sequence[float]
        Ascending cumulative area fractions to evaluate.
    original : InferenceMetrics
        Metrics computed on the unchanged reference image.
    batch_size : int
        Number of perturbed images processed in one inference batch.

    Returns
    -------
    Tuple[List[Dict[str, Any]], Dict[str, Tensor]]
        Deletion-record dictionaries and optional top-first preview tensors.
    """
    h, w = image.shape[-2:]
    gt_road = class_map == class_id
    boundary = torch.from_numpy(road_boundary_mask(class_map, class_id)).to(image.device)
    rows: List[Dict[str, Any]] = []
    preview: Dict[str, Tensor] = {}
    selections: List[List[int]] = []
    # Keep only tile indices for all budgets. Holding every perturbed full-size
    # image on CUDA before inference unnecessarily increases peak VRAM usage.
    for budget in budgets:
        selections.append(selected_at_budget(tiles, order, budget, h * w))
    for start in range(0, len(selections), batch_size):
        # Generate variants from the same original reference for comparability,
        # but only allocate up to batch_size images on the GPU at once.
        group_images = torch.cat([
            apply_tiles(image, baseline, tiles, chosen)
            for chosen in selections[start:start + batch_size]
        ])
        metrics = infer_metrics(model, group_images, class_map, class_id)
        for offset in range(len(group_images)):
            j = start + offset
            chosen = selections[j]
            region_mask = torch.zeros((h, w), dtype=torch.bool, device=image.device)
            for tile_idx in chosen:
                tile = tiles[tile_idx]
                region_mask[tile.y0:tile.y1, tile.x0:tile.x1] = True
            area = int(region_mask.sum().item())
            row: Dict[str, Any] = {
                "policy": policy, "repeat": repeat,
                "budget": float(budgets[j]), "actual_fraction": area / (h * w),
                "occluded_pixels": area, "tiles_removed": len(chosen),
                "gt_road_coverage": float((region_mask & gt_road).sum().item() / max(1, gt_road.sum().item())),
                "boundary_coverage": float((region_mask & boundary).sum().item() / max(1, boundary.sum().item())),
                "chosen_tile_indices": json.dumps(chosen),
            }
            for metric in METRIC_NAMES:
                row[metric] = float(metrics[metric][offset])
                row[f"delta_{metric}"] = float(original[metric][0] - metrics[metric][offset])
            rows.append(row)
            if policy == "top" and (j == len(budgets) - 1 or math.isclose(budgets[j], 0.2)):
                # Diagnostics are saved later, so hold preview tensors on CPU
                # rather than keeping large CUDA allocations alive for each case.
                preview[f"image_{j}"] = group_images[offset:offset + 1].detach().cpu().clone()
                preview[f"prediction_{j}"] = torch.from_numpy(metrics["prediction"][offset].copy())
                preview[f"indices_{j}"] = torch.tensor(chosen, dtype=torch.int64)
        del group_images, metrics
    return rows, preview


def auc_curves(frame: pd.DataFrame) -> Dict[str, float]:
    """Summarize deletion trajectories and improvements over random baselines.

    Steps
    -----
    1. Average repeated deletion runs at each nominal budget.
    2. Integrate each policy’s score curves and extract selected-budget deltas.
    3. Calculate paired top-first advantages over random and road-matched controls.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame of tile effects, deletion records, or case selections.

    Returns
    -------
    Dict[str, float]
        Mapping from summary metric names to scalar values.
    """
    summaries: Dict[str, float] = {}
    for policy in sorted(frame["policy"].unique()):
        subset = frame[frame["policy"] == policy]
        for metric in METRIC_NAMES:
            per_budget = subset.groupby("budget")[metric].mean().sort_index()
            summaries[f"auc_{metric}_{policy}"] = float(np.trapezoid(per_budget.to_numpy(), per_budget.index.to_numpy()))
        for budget in (0.1, 0.2):
            selected = subset[np.isclose(subset["budget"], budget)]
            if not selected.empty:
                for metric in ("global_logit", "road_iou"):
                    summaries[f"delta_{metric}_{policy}_{int(100 * budget)}pct"] = float(selected[f"delta_{metric}"].mean())
    for policy in ("random", "matched_random"):
        for budget in (0.1, 0.2):
            top = frame[(frame.policy == "top") & np.isclose(frame.budget, budget)]
            control = frame[(frame.policy == policy) & np.isclose(frame.budget, budget)]
            if not top.empty and not control.empty:
                for metric in ("global_logit", "road_iou"):
                    summaries[f"advantage_top_over_{policy}_{metric}_{int(budget*100)}pct"] = (
                        float(top[f"delta_{metric}"].mean()) - float(control[f"delta_{metric}"].mean())
                    )
    return summaries


@jaxtyped(typechecker=beartype)
def save_heatmap(image: SingleImage, sensitivity: NumpyHeatmap, path: Path, title: str) -> None:
    """Save a signed occlusion map and an aligned input overlay.

    Steps
    -----
    1. Choose a zero-centered diverging color scale to preserve negative evidence.
    2. Render standalone sensitivity and image-overlay panels with a shared legend.

    Parameters
    ----------
    image : SingleImage
        Reference NCHW RGB image tensor, normally a single example.
    sensitivity : NumpyHeatmap
        Signed spatial occlusion attribution map.
    path : Path
        Filesystem path, absolute or relative to the project root.
    title : str
        Human-readable title for the figure.

    Returns
    -------
    None
        Nothing; writes a diagnostic PNG to the specified path.
    """
    v = max(float(np.max(np.abs(sensitivity))), 1e-10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rgb = image[0].permute(1, 2, 0).detach().cpu().numpy()
    norm = TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)
    axes[0].imshow(sensitivity, cmap="coolwarm", norm=norm)
    axes[0].set_title("Signed occlusion effect")
    axes[1].imshow(rgb)
    mappable = axes[1].imshow(sensitivity, cmap="coolwarm", norm=norm, alpha=0.48)
    axes[1].set_title("Input + sensitivity")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(mappable, ax=axes, fraction=0.024, pad=0.04, label="Change in target score")
    fig.suptitle(title)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_curves(frame: pd.DataFrame, directory: Path) -> None:
    """Plot progressive-deletion trajectories separately for each evaluation metric.

    Steps
    -----
    1. Aggregate repeated trials by deletion budget and policy.
    2. Draw mean curves and optional repeat variability bands.
    3. Write separate raw-logit, GT-logit, and IoU figures.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame of tile effects, deletion records, or case selections.
    directory : Path
        Directory for figures and artifacts of the current case.

    Returns
    -------
    None
        Nothing; writes deletion-curve PNGs into the output directory.
    """
    names = {"global_logit": "Global target-class logit", "gt_logit": "GT target-region logit", "road_iou": "Target-class IoU"}
    for metric, title in names.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for policy in ("top", "bottom", "random", "matched_random"):
            subset = frame[frame.policy == policy]
            if subset.empty:
                continue
            agg = subset.groupby("budget")[metric].agg(["mean", "std"]).sort_index()
            x = agg.index.to_numpy(dtype=float)
            y = agg["mean"].to_numpy(dtype=float)
            ax.plot(100 * x, y, marker="o", label=policy)
            if subset.repeat.nunique() > 1:
                std = agg["std"].fillna(0).to_numpy(dtype=float)
                ax.fill_between(100 * x, y - std, y + std, alpha=0.16)
        ax.set(xlabel="Nominal area occluded (%)", ylabel=title, title=f"Deletion: {title}")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"deletion_{metric}.png", dpi=160)
        plt.close(fig)


@jaxtyped(typechecker=beartype)
def colourise(class_map: NumpyClassMap, palette: NumpyPalette) -> NumpyRGB:
    """Convert a dense semantic class map into an RGB diagnostic image.

    Steps
    -----
    1. Replace invalid class IDs with the last palette entry.
    2. Map class IDs to their BDD100k colors and normalize to [0, 1].

    Parameters
    ----------
    class_map : NumpyClassMap
        Fixed two-dimensional ground-truth class ID map.
    palette : NumpyPalette
        RGB class-color lookup table from dataset configuration.

    Returns
    -------
    NumpyRGB
        RGB floating-point image suitable for plotting.
    """
    indices = np.where((class_map >= 0) & (class_map < len(palette)), class_map, len(palette) - 1)
    return np.asarray(palette[indices], dtype=np.float32) / 255


@jaxtyped(typechecker=beartype)
def save_diagnostic(
    directory: Path, image: SingleImage, gt: ClassMap, original_pred: NumpyClassMap,
    cam: Optional[NumpyHeatmap], sensitivity: Optional[NumpyHeatmap],
    preview: Dict[str, Tensor], tiles: Sequence[Tile], palette: NumpyPalette,
    baseline_mode: BaselineMode,
) -> None:
    """Write the eight-panel reference, attribution, and deletion comparison.

    Steps
    -----
    1. Colorize the GT and original predictions alongside the input image.
    2. Display CAM and signed occlusion evidence if available.
    3. Highlight deleted top-ranked tiles and show the corresponding changed prediction.

    Parameters
    ----------
    directory : Path
        Directory for figures and artifacts of the current case.
    image : SingleImage
        Reference NCHW RGB image tensor, normally a single example.
    gt : ClassMap
        Fixed ground-truth semantic class IDs at evaluation resolution.
    original_pred : NumpyClassMap
        Dense prediction on the unmodified reference input.
    cam : Optional[NumpyHeatmap]
        Numerical Grad-CAM++ heatmap, or None when not requested.
    sensitivity : Optional[NumpyHeatmap]
        Signed spatial occlusion attribution map.
    preview : Dict[str, Tensor]
        Saved top-first deleted images and their predicted class maps.
    tiles : Sequence[Tile]
        Deterministic tile metadata spanning the input image.
    palette : NumpyPalette
        RGB class-color lookup table from dataset configuration.
    baseline_mode : BaselineMode
        Replacement strategy used to create the displayed occluded input.

    Returns
    -------
    None
        Nothing; saves diagnostic.png in the case directory.
    """
    rgb = image[0].detach().cpu().permute(1, 2, 0).numpy()
    panels: List[Tuple[str, np.ndarray]] = [
        ("Reference input", rgb),
        ("Ground truth", colourise(gt.cpu().numpy(), palette)),
        ("Original prediction", colourise(original_pred, palette)),
    ]
    if cam is not None:
        panels.append(("Grad-CAM++", plt.get_cmap("jet")(cam)[..., :3]))
    else:
        panels.append(("Grad-CAM++ (unavailable)", np.zeros_like(rgb)))
    if sensitivity is not None:
        v = max(float(np.max(np.abs(sensitivity))), 1e-9)
        panels.append(("Signed sensitivity", plt.get_cmap("coolwarm")((sensitivity + v) / (2 * v))[..., :3]))
    else:
        panels.append(("Sensitivity (not requested)", np.zeros_like(rgb)))
    preview_indices = [int(k.split("_")[1]) for k in preview if k.startswith("image_")]
    if preview_indices:
        j = max(preview_indices)
        visual = rgb.copy()
        for tile_idx in preview[f"indices_{j}"].tolist():
            t = tiles[tile_idx]
            visual[t.y0:t.y1, t.x0:t.x1] = 0.65 * visual[t.y0:t.y1, t.x0:t.x1] + 0.35 * np.array([1, 0.2, 0.1])
        panels.append(("Top-ranked deletion tiles", visual))
        panels.append((
            f"Occluded input ({baseline_mode})",
            preview[f"image_{j}"][0].cpu().permute(1, 2, 0).numpy(),
        ))
        panels.append(("Occluded prediction", colourise(preview[f"prediction_{j}"].numpy(), palette)))
    else:
        panels.extend([(label, np.zeros_like(rgb)) for label in ("Selected tiles", "Occluded input", "Occluded prediction")])
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    for axis, (title, array) in zip(axs.flat, panels, strict=True):
        axis.imshow(np.clip(array, 0, 1))
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(directory / "diagnostic.png", dpi=160)
    plt.close(fig)


def load_cam_index(manifest_path: Path, root: Path) -> Dict[Tuple[str, str, int, str, str], Dict[str, Any]]:
    """Index Grad-CAM++ records by model, input mode, class, and source paths.

    Steps
    -----
    1. Read the existing Grad-CAM++ manifest and validate its required columns.
    2. Resolve image/mask paths and deduplicate repeated selection provenance.
    3. Reject duplicate records whose saved input hashes disagree.

    Parameters
    ----------
    manifest_path : Path
        Path to the Grad-CAM++ CSV manifest.
    root : Path
        Resolved project root used to interpret relative paths.

    Returns
    -------
    Dict[Tuple[str, str, int, str, str], Dict[str, Any]]
        Lookup from a canonical case key to its CAM manifest record.

    Raises
    ------
    FileNotFoundError
        If the manifest is absent.
    ValueError
        If required columns are missing or duplicate inputs conflict.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Grad-CAM++ manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    required = {"model_name", "input_mode", "target_class_id", "image_path", "mask_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Grad-CAM++ manifest missing: {sorted(missing)}")
    index: Dict[Tuple[str, str, int, str, str], Dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = (str(row.model_name), str(row.input_mode), int(row.target_class_id),
               str(resolve_path(str(row.image_path), root)), str(resolve_path(str(row.mask_path), root)))
        record = row.to_dict()
        if key in index:
            previous = index[key]
            _, mode_hash_field = manifest_input_fields(str(row.input_mode))
            current_hash = record.get(mode_hash_field)
            previous_hash = previous.get(mode_hash_field)
            # Backward compatibility: older manifests stored only one exact-input
            # hash column, already scoped by the row's concrete input mode.
            if not isinstance(current_hash, str) or not current_hash.strip():
                current_hash = record.get("input_sha256")
            if not isinstance(previous_hash, str) or not previous_hash.strip():
                previous_hash = previous.get("input_sha256")
            if (
                isinstance(current_hash, str)
                and isinstance(previous_hash, str)
                and current_hash.strip()
                and previous_hash.strip()
                and current_hash != previous_hash
            ):
                raise ValueError(f"Inconsistent Grad-CAM input hashes for {key}.")
        else:
            index[key] = record
    return index


@jaxtyped(typechecker=beartype)
def load_cam_data(record: Dict[str, Any], root: Path, shape: Tuple[int, int]) -> NumpyHeatmap:
    """Read the precise numerical CAM rather than a color-rendered PNG.

    Steps
    -----
    1. Find the .npy file recorded in the CAM manifest.
    2. Validate the expected resolution, finite values, and normalized range.

    Parameters
    ----------
    record : Dict[str, Any]
        One optional or required CAM manifest record with saved paths.
    root : Path
        Resolved project root used to interpret relative paths.
    shape : Tuple[int, int]
        Required (height, width) of the numerical CAM.

    Returns
    -------
    NumpyHeatmap
        Validated float32 Grad-CAM++ map at input resolution.

    Raises
    ------
    ValueError
        If the CAM path is missing or the array has invalid shape/values.
    OSError
        If the underlying .npy file cannot be read.
    """
    value = record.get("heatmap_npy_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Grad-CAM manifest lacks heatmap_npy_path. Regenerate CAMs with updated explain_gradcam.py.")
    path = resolve_path(value, root)
    cam = np.load(path, allow_pickle=False)
    if cam.shape != shape or not np.isfinite(cam).all():
        raise ValueError(f"Invalid CAM shape or values at {path}: {cam.shape} vs {shape}")
    if cam.min() < -1e-5 or cam.max() > 1 + 1e-5:
        raise ValueError(f"Expected normalized [0,1] Grad-CAM++ attribution: {path}")
    return cam.astype(np.float32, copy=False)


def manifest_input_fields(mode: str) -> Tuple[str, str]:
    """Return the manifest columns for one concrete saved input variant.

    Steps
    -----
    1. Validate that the requested mode is either ``clean`` or ``robust``.
    2. Map that mode to its explicit path and SHA-256 manifest columns.

    Parameters
    ----------
    mode : str
        Concrete input variant to recover.

    Returns
    -------
    Tuple[str, str]
        Path-column name followed by SHA-256-column name.

    Raises
    ------
    ValueError
        If ``mode`` is not ``clean`` or ``robust``.
    """
    if mode == "clean":
        return "clean_input_tensor_path", "clean_input_sha256"
    if mode == "robust":
        return "robust_input_tensor_path", "robust_input_sha256"
    raise ValueError("mode must be either 'clean' or 'robust'.")


@jaxtyped(typechecker=beartype)
def load_reference_input(
    record: Optional[Dict[str, Any]], clean: SingleImage, mode: str, root: Path,
) -> SingleImage:
    """Recover the exact saved tensor that Grad-CAM++ originally explained.

    Robust inputs are never regenerated in this program. The function prefers
    the explicit mode-specific artifact columns written by the updated
    ``explain_gradcam.py`` and only falls back to the legacy ``input_tensor_path``
    when that record itself is for the requested mode.

    Steps
    -----
    1. Resolve the explicit clean/robust artifact path for the requested mode.
    2. Fall back to the legacy exact-input field only for a matching manifest row.
    3. Validate shape and SHA-256 before moving the tensor to the model device.
    4. For clean mode, additionally require bitwise equality with current preprocessing.
    5. Reject robust mode if no persisted PGD artifact is available.

    Parameters
    ----------
    record : Optional[Dict[str, Any]]
        CAM manifest record corresponding to the same model, source image, and mode.
    clean : SingleImage
        Preprocessed unperturbed image for the same dataset sample.
    mode : str
        Concrete input variant, either ``clean`` or ``robust``.
    root : Path
        Resolved project root used to interpret relative manifest paths.

    Returns
    -------
    SingleImage
        Exact clean or PGD-perturbed reference tensor on the current model device.

    Raises
    ------
    ValueError
        If the manifest mode is inconsistent, a robust artifact is unavailable,
        or saved shape/hash/preprocessing checks fail.
    OSError
        If a referenced saved NPY artifact cannot be read.
    """
    path_field, hash_field = manifest_input_fields(mode)

    if record is None:
        if mode == "robust":
            raise ValueError(
                "Robust occlusion requires the exact PGD tensor saved by "
                "Grad-CAM++; rerun explain gradcam with --input-mode robust."
            )
        return clean

    record_mode = str(record.get("input_mode", "")).strip()
    if record_mode and record_mode != mode:
        raise ValueError(
            f"Grad-CAM manifest input_mode={record_mode!r} does not match "
            f"requested occlusion mode={mode!r}."
        )

    # New manifests record both variant-specific columns. Legacy manifests are
    # still accepted when their row's concrete input_mode matches this request.
    filename = record.get(path_field)
    expected_hash = record.get(hash_field)
    if not isinstance(filename, str) or not filename.strip():
        filename = record.get("input_tensor_path")
        expected_hash = record.get("input_sha256")

    if not isinstance(filename, str) or not filename.strip():
        if mode == "robust":
            raise ValueError(
                "Robust Grad-CAM manifest has no persisted PGD input tensor. "
                "Regenerate Grad-CAM++ with the updated explain_gradcam.py."
            )
        return clean

    artifact_path = resolve_path(filename, root)
    array = np.load(artifact_path, allow_pickle=False)
    if tuple(array.shape) != tuple(clean.shape):
        raise ValueError(
            f"Saved {mode} Grad-CAM input shape {array.shape} differs from "
            f"current sample shape {tuple(clean.shape)}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Saved {mode} Grad-CAM input contains nonfinite values: {artifact_path}")

    result = torch.from_numpy(np.array(array, copy=True)).to(
        device=clean.device,
        dtype=clean.dtype,
    )
    if (
        isinstance(expected_hash, str)
        and expected_hash.strip()
        and tensor_hash(result) != expected_hash
    ):
        raise ValueError(
            f"Saved {mode} Grad-CAM input SHA-256 does not match the manifest."
        )

    # Clean mode has an independently reconstructible reference, so require exact
    # equality. Robust mode cannot be reconstructed without rerunning PGD and must
    # therefore trust only the persisted, hash-verified artifact above.
    if mode == "clean" and not torch.equal(result, clean):
        raise ValueError(
            "Saved Grad-CAM clean input differs from current evaluation preprocessing."
        )
    return cast(SingleImage, result)


def save_csv_upsert(path: Path, rows: List[Dict[str, Any]], keys: Sequence[str]) -> None:
    """Maintain a deduplicated CSV manifest across repeated runs.

    Steps
    -----
    1. Load existing records when an output manifest already exists.
    2. Append new rows and keep the newest record for each compound key.

    Parameters
    ----------
    path : Path
        Filesystem path, absolute or relative to the project root.
    rows : List[Dict[str, Any]]
        New output records to add to the CSV manifest.
    keys : Sequence[str]
        Column names that uniquely identify one manifest record.

    Returns
    -------
    None
        Nothing; saves the updated CSV manifest in place.
    """
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if path.is_file():
        previous = pd.read_csv(path)
        incoming = pd.concat([previous, incoming], ignore_index=True)
    incoming.drop_duplicates(subset=list(keys), keep="last").to_csv(path, index=False)


@jaxtyped(typechecker=beartype)
def compute_summary(
    case: Dict[str, Any], original: InferenceMetrics, patch_frame: Optional[pd.DataFrame],
    deletion_frame: Optional[pd.DataFrame], tiles: Sequence[Tile],
) -> Dict[str, Any]:
    """Summarize one source image and checkpoint without duplicating case selections.

    Steps
    -----
    1. Store original response and IoU values for the unique analyzed input.
    2. Relate CAM ranks to independent tile effects using Spearman correlation.
    3. Attach region distribution and progressive-deletion metrics when available.

    Parameters
    ----------
    case : Dict[str, Any]
        Provenance metadata describing the current unique case.
    original : InferenceMetrics
        Metrics computed on the unchanged reference image.
    patch_frame : Optional[pd.DataFrame]
        Independent per-tile ablation results, if evaluated.
    deletion_frame : Optional[pd.DataFrame]
        Cumulative deletion results, if evaluated.
    tiles : Sequence[Tile]
        Deterministic tile metadata spanning the input image.

    Returns
    -------
    Dict[str, Any]
        One flat summary record containing provenance and available faithfulness metrics.
    """
    result = dict(case)
    result.update({f"original_{metric}": float(original[metric][0]) for metric in METRIC_NAMES})
    if patch_frame is not None and len(patch_frame) > 1:
        statistic = spearmanr(patch_frame["cam_mean"], patch_frame["delta_global_logit"])
        result["tile_spearman"] = float(statistic.statistic)
        result["tile_spearman_pvalue"] = float(statistic.pvalue)
        top_tiles = sorted(tiles, key=lambda tile: -tile.cam_mean)[:max(1, math.ceil(0.1 * len(tiles)))]
        for category in ("road_interior", "road_boundary", "non_road_context"):
            result[f"top10_tiles_{category}_fraction"] = sum(t.region == category for t in top_tiles) / len(top_tiles)
    if deletion_frame is not None:
        result.update(auc_curves(deletion_frame))
    return result


def write_model_comparisons(output: Path, summaries: List[Dict[str, Any]], seed: int) -> None:
    """Aggregate unique-image results and paired model contrasts.

    Steps
    -----
    1. Deduplicate category-selected rows by source image, model, mode, and baseline.
    2. Compute per-model estimates and image-level bootstrap intervals where sample size permits.
    3. Pair only identical source images across models, recording model-specific PGD input caveats.

    Parameters
    ----------
    output : Path
        Root directory receiving the aggregate comparison CSV files.
    summaries : List[Dict[str, Any]]
        Per-image faithfulness summary records across checkpoints.
    seed : int
        Random seed controlling repeatable ordering or bootstrap sampling.

    Returns
    -------
    None
        Nothing; writes aggregate and paired-comparison CSV files.
    """
    if not summaries:
        return
    frame = pd.DataFrame(summaries).drop_duplicates(
        subset=["model_name", "input_mode", "baseline", "unique_image_id"],
        keep="last",
    )
    comparisons = [
        "original_road_iou", "tile_spearman",
        "advantage_top_over_random_global_logit_10pct",
        "advantage_top_over_matched_random_global_logit_10pct",
        "advantage_top_over_random_road_iou_20pct",
    ]
    comparisons = [column for column in comparisons if column in frame.columns]
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    # Bootstrap distinct source images, not selection-category duplicates or
    # repeated random rankings, which are not independent experimental units.
    for (model, mode, baseline), group in frame.groupby(["model_name", "input_mode", "baseline"]):
        for metric in comparisons:
            data = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            row: Dict[str, Any] = {
                "model_name": model, "input_mode": mode, "baseline": baseline,
                "metric": metric, "n_images": len(data),
                "mean": float(np.mean(data)) if len(data) else float("nan"),
                "median": float(np.median(data)) if len(data) else float("nan"),
                "bootstrap_ci_low": float("nan"), "bootstrap_ci_high": float("nan"),
            }
            if len(data) >= 5:
                draws = rng.choice(data, size=(2000, len(data)), replace=True).mean(axis=1)
                row["bootstrap_ci_low"], row["bootstrap_ci_high"] = (
                    float(v) for v in np.quantile(draws, (0.025, 0.975))
                )
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / "model_comparison.csv", index=False)

    # Same source image and concrete input mode are paired. On 'robust' mode,
    # note that each checkpoint usually has its own PGD-generated input tensor.
    contrasts: List[Dict[str, Any]] = []
    for (mode, baseline), group in frame.groupby(["input_mode", "baseline"]):
        models = sorted(group.model_name.unique())
        for pos, left in enumerate(models):
            left_rows = group[group.model_name == left].set_index("unique_image_id")
            for right in models[pos + 1:]:
                right_rows = group[group.model_name == right].set_index("unique_image_id")
                common = sorted(set(left_rows.index) & set(right_rows.index))
                for image_id in common:
                    for metric in comparisons:
                        a = left_rows.loc[image_id, metric]
                        b = right_rows.loc[image_id, metric]
                        if pd.isna(a) or pd.isna(b):
                            continue
                        contrasts.append({
                            "input_mode": mode, "baseline": baseline,
                            "model_left": left, "model_right": right,
                            "unique_image_id": image_id, "metric": metric,
                            "left_minus_right": float(a) - float(b),
                            "shared_adversarial_input": bool(mode == "clean"),
                        })
    if contrasts:
        pd.DataFrame(contrasts).to_csv(output / "paired_model_comparisons.csv", index=False)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the five options that vary between occlusion runs.

    Steps
    -----
    1. Accept selected cases, Grad-CAM manifest, input mode, and baseline.
    2. Accept one or more named checkpoints; fixed settings live in
       :class:`OcclusionConfiguration`.
    3. Parse the supplied tokens or the current process arguments.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Optional explicit CLI tokens; defaults to process arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line namespace.

    Raises
    ------
    SystemExit
        If argparse reports invalid user arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="NAME=PATH (repeatable), matching selected_cases.csv.",
    )
    parser.add_argument("--selected-cases", default="./model/xai_selection/selected_cases.csv")
    parser.add_argument("--gradcam-manifest", default="./model/xai_gradcam/gradcam_manifest.csv")
    parser.add_argument("--input-mode", choices=("clean", "robust", "all"), default="clean")
    parser.add_argument(
        "--baseline",
        choices=("blur", "gray", "white", "bright", "noise", "image_mean"),
        default="white",
        help=(
            "Replacement for both sliding and deletion experiments. White is the "
            "default high-contrast occluder for the road class; blur, gray, "
            "bright, image_mean, and deterministic noise are available as controls."
        ),
    )
    return parser.parse_args(argv)


def validate_configuration(config: OcclusionConfiguration) -> None:
    """Check fixed experiment settings before loading any checkpoints.

    Steps
    -----
    1. Check positive batching and repetition counts and valid deletion budgets.
    2. Check tile/stride coverage and Gaussian replacement parameters.

    Parameters
    ----------
    config : OcclusionConfiguration
        Fixed experiment configuration shared by every checkpoint.

    Returns
    -------
    None
        Nothing; raises when an option is invalid.

    Raises
    ------
    ValueError
        If any fixed parameter is outside its supported range.
    """
    if OcclusionConfiguration.PERTURBATIONS_PER_EVAL < 1 or OcclusionConfiguration.RANDOM_REPEATS < 1:
        raise ValueError("Batch size and random repeats must be positive.")
    if not 0 < OcclusionConfiguration.INTERIOR_THRESHOLD <= 1:
        raise ValueError("interior_threshold must lie in (0,1].")
    if any(value <= 0 for pair in (OcclusionConfiguration.PATCH_SIZE, OcclusionConfiguration.STRIDE, OcclusionConfiguration.TILE_SIZE) for value in pair):
        raise ValueError("Patch, stride, and tile dimensions must be positive.")
    if any(x > y for x, y in zip(OcclusionConfiguration.STRIDE, OcclusionConfiguration.PATCH_SIZE, strict=True)):
        raise ValueError("Sliding stride cannot exceed patch size.")
    if OcclusionConfiguration.BLUR_KERNEL < 3 or OcclusionConfiguration.BLUR_KERNEL % 2 == 0 or OcclusionConfiguration.BLUR_SIGMA <= 0:
        raise ValueError("Gaussian blur requires an odd kernel >= 3 and sigma > 0.")
    if not OcclusionConfiguration.BUDGETS or OcclusionConfiguration.BUDGETS[0] != 0 or any(
        not 0 <= budget <= 1 for budget in OcclusionConfiguration.BUDGETS
    ) or tuple(sorted(set(OcclusionConfiguration.BUDGETS))) != OcclusionConfiguration.BUDGETS:
        raise ValueError("Deletion budgets must be sorted, unique, and start at zero.")


def validate_cam_reference_score(
    cam_record: Optional[Dict[str, Any]],
    original: InferenceMetrics,
    mode: ConcreteInputMode,
) -> None:
    """Verify that occlusion starts from the same scalar response as Grad-CAM++.

    The saved input tensor hash proves input identity, while this score check also
    catches changed checkpoint weights or materially different inference settings.
    Records without a reference score are accepted for backward compatibility.

    Steps
    -----
    1. Read the optional Grad-CAM++ target score from the manifest record.
    2. Compare it with the freshly evaluated global target logit.
    3. Raise with a provenance-focused message when the values disagree.

    Parameters
    ----------
    cam_record : Optional[Dict[str, Any]]
        Matching Grad-CAM++ manifest record; None skips legacy score verification.
    original : InferenceMetrics
        Metrics computed on the unmodified reference tensor.
    mode : ConcreteInputMode
        Concrete input variant being evaluated: ``clean`` or ``robust``.

    Returns
    -------
    None
        Nothing; validation succeeds silently.

    Raises
    ------
    ValueError
        If a finite saved Grad-CAM++ score differs from the freshly evaluated score
        beyond the configured numerical tolerance.
    """
    if cam_record is None:
        return

    expected = cam_record.get("target_score")
    if expected is None or pd.isna(expected):
        return

    actual = float(original["global_logit"][0])
    expected_float = float(expected)
    if not np.isclose(actual, expected_float, atol=1e-4, rtol=1e-4):
        raise ValueError(
            "Grad-CAM reference score mismatch after loading the exact "
            f"saved {mode} input: expected {expected_float:.10f}, got {actual:.10f}. "
            "Regenerate Grad-CAM++ with the updated saved-input artifacts and verify "
            "that the same checkpoint is used."
        )


@jaxtyped(typechecker=beartype)
def prepare_case_directory(
    output_dir: Path,
    baseline_mode: BaselineMode,
    model_name: str,
    mode: ConcreteInputMode,
    image_index: int,
    image_path: Path,
    mask_path: Path,
    image: SingleImage,
    cam: NumpyHeatmap,
) -> Tuple[Path, str]:
    """Create the case directory using the requested baseline as its final key.

    Fixed geometry and budget settings are documented in configuration.json;
    baseline is the only per-run setting that changes the directory name.

    Steps
    -----
    1. Hash the source image/mask identity to avoid filename collisions.
    2. Build the model/mode/image/baseline directory hierarchy.
    3. Save the reproducible fixed settings and reference hashes as JSON.

    Parameters
    ----------
    output_dir : Path
        Resolved root directory for occlusion artifacts.
    baseline_mode : BaselineMode
        Selected replacement baseline and final directory component.
    model_name : str
        Human-readable checkpoint identifier.
    mode : ConcreteInputMode
        Concrete input variant being evaluated.
    image_index : int
        Stable dataset/test-split image index.
    image_path : Path
        Resolved source RGB image path.
    mask_path : Path
        Resolved source segmentation-mask path.
    image : SingleImage
        Exact clean or saved PGD tensor used as the occlusion reference.
    cam : NumpyHeatmap
        Numerical Grad-CAM++ map used for deletion.

    Returns
    -------
    Tuple[Path, str]
        Case output directory and source identity digest.
    """
    from explain_gradcam import sanitise_filename

    source_id = hashlib.sha256(f"{image_path}|{mask_path}".encode()).hexdigest()[:12]
    parameters: Dict[str, Any] = {
        "experiment": "sliding+deletion",
        "patch_size": OcclusionConfiguration.PATCH_SIZE,
        "stride": OcclusionConfiguration.STRIDE,
        "tile_size": OcclusionConfiguration.TILE_SIZE,
        "baseline": baseline_mode,
        "blur_kernel": OcclusionConfiguration.BLUR_KERNEL,
        "blur_sigma": OcclusionConfiguration.BLUR_SIGMA,
        "budgets": OcclusionConfiguration.BUDGETS,
        "random_repeats": OcclusionConfiguration.RANDOM_REPEATS,
        "seed": OcclusionConfiguration.SEED,
        "noise_seed": (
            derive_noise_seed(OcclusionConfiguration.SEED, image_index)
            if baseline_mode == "noise"
            else None
        ),
        "interior_threshold": OcclusionConfiguration.INTERIOR_THRESHOLD,
        "sliding_gt_target": OcclusionConfiguration.SLIDING_GROUND_TRUTH_TARGET,
        "input_hash": tensor_hash(image),
        "cam_hash": sha256_array(cam),
    }
    directory = (
        output_dir
        / sanitise_filename(model_name)
        / mode
        / f"img{image_index:05d}_{source_id}"
        / baseline_mode
    )
    directory.mkdir(parents=True, exist_ok=True)

    # Persisting the configuration beside the outputs makes every figure and CSV
    # auditable without needing to reconstruct CLI arguments from shell history.
    with open(directory / "configuration.json", "w", encoding="utf-8") as file:
        json.dump(parameters, file, indent=2)

    return directory, source_id


@jaxtyped(typechecker=beartype)
def save_reference_artifacts(
    directory: Path,
    image: SingleImage,
    original: InferenceMetrics,
) -> None:
    """Persist the unmodified reference tensor and prediction for one case.

    These artifacts establish the baseline from which all perturbation deltas are
    measured and make later debugging possible without rerunning model inference.

    Steps
    -----
    1. Save the exact NCHW float32 input and dense original prediction as NumPy arrays.
    2. Save a human-readable RGB PNG of the same reference input.

    Parameters
    ----------
    directory : Path
        Output directory for the current model/input/image/configuration case.
    image : SingleImage
        Exact reference tensor used for all perturbation experiments.
    original : InferenceMetrics
        Metrics and prediction evaluated on the unmodified reference tensor.

    Returns
    -------
    None
        Nothing; writes NumPy and PNG artifacts to ``directory``.
    """
    input_rgb = image.detach().cpu().numpy().astype(np.float32)
    np.save(directory / "reference_input.npy", input_rgb)
    np.save(directory / "original_prediction.npy", original["prediction"][0])
    plt.imsave(directory / "original_input.png", input_rgb[0].transpose(1, 2, 0))


@jaxtyped(typechecker=beartype)
def run_sliding_artifacts(
    config: OcclusionConfiguration,
    model: nn.Module,
    image: SingleImage,
    baseline: SingleImage,
    gt: ClassMap,
    target_id: int,
    directory: Path,
) -> NumpyHeatmap:
    """Run and persist Captum sliding-window attribution for every case.

    Sliding attribution is independent from CAM-ranked deletion. It measures the
    signed change in the same scalar target when windows are replaced throughout
    the full image, providing an additional perturbation-based explanation.

    Steps
    -----
    1. Compute and save the global-logit signed occlusion map.
    2. Optionally compute the fixed-GT-road-logit diagnostic map.

    Parameters
    ----------
    config : OcclusionConfiguration
        Fixed sliding window and inference microbatch settings.
    model : nn.Module
        Loaded evaluation-mode segmentation checkpoint.
    image : SingleImage
        Exact reference input tensor.
    baseline : SingleImage
        Replacement tensor used inside each occlusion window.
    gt : ClassMap
        Fixed ground-truth semantic class map.
    target_id : int
        Semantic class ID whose evidence is evaluated.
    directory : Path
        Case output directory receiving arrays and figures.

    Returns
    -------
    NumpyHeatmap
        Global signed sensitivity map for the current input.

    Raises
    ------
    ImportError
        If Captum is unavailable.
    ValueError
        If patch/stride geometry is invalid for the current image.
    RuntimeError
        If Captum returns malformed or nonfinite attribution values.
    """
    sensitivity = sliding_occlusion(
        model,
        image,
        target_id,
        gt,
        OcclusionConfiguration.PATCH_SIZE,
        OcclusionConfiguration.STRIDE,
        baseline,
        OcclusionConfiguration.PERTURBATIONS_PER_EVAL,
    )
    np.save(directory / "occlusion_global_signed.npy", sensitivity)
    save_heatmap(
        image,
        sensitivity,
        directory / "occlusion_global.png",
        "Full-image road-logit sensitivity",
    )

    if OcclusionConfiguration.SLIDING_GROUND_TRUTH_TARGET:
        secondary = sliding_occlusion(
            model,
            image,
            target_id,
            gt,
            OcclusionConfiguration.PATCH_SIZE,
            OcclusionConfiguration.STRIDE,
            baseline,
            OcclusionConfiguration.PERTURBATIONS_PER_EVAL,
            gt_target=True,
        )
        np.save(directory / "occlusion_gt_signed.npy", secondary)
        save_heatmap(
            image,
            secondary,
            directory / "occlusion_gt.png",
            "GT-road-logit sensitivity",
        )

    return sensitivity


@jaxtyped(typechecker=beartype)
def run_deletion_artifacts(
    config: OcclusionConfiguration,
    model: nn.Module,
    image: SingleImage,
    baseline: SingleImage,
    gt: ClassMap,
    target_id: int,
    cam: NumpyHeatmap,
    original: InferenceMetrics,
    directory: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tensor], List[Tile]]:
    """Run CAM-ranked tile ablation and progressive deletion experiments.

    Deletion is the direct Grad-CAM++ faithfulness test: top-ranked regions are
    destructively replaced and compared with low-ranked and random controls under
    the same area budgets. A numerical CAM is required for every input.

    Steps
    -----
    1. Build the deterministic tile grid and measure each tile independently.
    2. Evaluate top, bottom, random, and road/location-matched deletion orders.
    3. Save raw tables, deletion curves, and the strongest top-first preview artifacts.

    Parameters
    ----------
    config : OcclusionConfiguration
        Fixed deletion, tile, and random-control settings.
    model : nn.Module
        Loaded evaluation-mode segmentation checkpoint.
    image : SingleImage
        Exact clean or robust reference tensor.
    baseline : SingleImage
        Replacement tensor used for destructive interventions.
    gt : ClassMap
        Fixed ground-truth semantic class map.
    target_id : int
        Semantic target class ID, normally road.
    cam : NumpyHeatmap
        Numerical Grad-CAM++ attribution map used for tile ranking.
    original : InferenceMetrics
        Baseline metrics evaluated before any tile replacement.
    directory : Path
        Case output directory receiving deletion artifacts.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tensor], List[Tile]]
        Independent tile-effect table, progressive deletion table, diagnostic preview
        tensors, and the tile grid.
    """
    # The same tile geometry is used for all policies so only ranking—not deleted
    # area or patch shape—changes between Grad-CAM++ and control interventions.
    tiles = build_tiles(cam, gt, target_id, OcclusionConfiguration.TILE_SIZE, OcclusionConfiguration.INTERIOR_THRESHOLD)
    patch_frame = tile_effects(
        model,
        image,
        baseline,
        gt,
        target_id,
        tiles,
        OcclusionConfiguration.PERTURBATIONS_PER_EVAL,
        original,
    )
    patch_frame.to_csv(directory / "tile_effects.csv", index=False)

    deletion_rows: List[Dict[str, Any]] = []
    previews: Dict[str, Tensor] = {}
    for policy in ("top", "bottom", "random", "matched_random"):
        repeats = OcclusionConfiguration.RANDOM_REPEATS if policy in ("random", "matched_random") else 1
        for repeat in range(repeats):
            # Use separated deterministic seeds so random controls are reproducible
            # without accidentally sharing the same permutation across repetitions.
            order = deletion_order(tiles, policy, seed=OcclusionConfiguration.SEED + repeat * 7919)
            rows, preview = ordered_deletion(
                model,
                image,
                baseline,
                gt,
                target_id,
                tiles,
                order,
                policy,
                repeat,
                OcclusionConfiguration.BUDGETS,
                original,
                OcclusionConfiguration.PERTURBATIONS_PER_EVAL,
            )
            deletion_rows.extend(rows)
            if policy == "top":
                previews = preview

    deletion_frame = pd.DataFrame(deletion_rows)
    deletion_frame.to_csv(directory / "deletion.csv", index=False)
    save_curves(deletion_frame, directory)

    if previews:
        preview_indices = [
            int(key.split("_")[1]) for key in previews if key.startswith("image_")
        ]
        preview_index = max(preview_indices)
        preview_image = previews[f"image_{preview_index}"][0].cpu().permute(1, 2, 0).numpy()
        plt.imsave(directory / "occluded_top.png", preview_image)
        np.save(
            directory / "occluded_top_prediction.npy",
            previews[f"prediction_{preview_index}"].numpy(),
        )

    return patch_frame, deletion_frame, previews, tiles


def build_case_record(
    config: OcclusionConfiguration,
    target_id: int,
    baseline_mode: BaselineMode,
    model_name: str,
    checkpoint_path: Path,
    image_index: int,
    image_path: Path,
    mask_path: Path,
    mode: ConcreteInputMode,
    image: SingleImage,
    group: pd.DataFrame,
    cam_record: Optional[Dict[str, Any]],
    directory: Path,
    source_id: str,
) -> Dict[str, Any]:
    """Build provenance metadata used by per-case and global summary manifests.

    Selection rows can repeat the same source image under multiple categories. The
    record preserves those categories as provenance while identifying the actual
    experiment by model, input mode, source image, and baseline name.

    Steps
    -----
    1. Resolve the target class name and optional CAM path.
    2. Collapse repeated selection categories/source rows into semicolon lists.
    3. Record checkpoint, input hash, output directory, and baseline identifier.

    Parameters
    ----------
    config : OcclusionConfiguration
        Shared reproducibility settings, including the fixed random seed.
    target_id : int
        Semantic class ID whose logits are evaluated.
    baseline_mode : BaselineMode
        Selected replacement strategy used for both experiments.
    model_name : str
        Human-readable checkpoint identifier.
    checkpoint_path : Path
        Resolved checkpoint file used for inference.
    image_index : int
        Stable test-split index of the source image.
    image_path : Path
        Resolved RGB source image path.
    mask_path : Path
        Resolved segmentation-mask path.
    mode : ConcreteInputMode
        Concrete input variant being evaluated.
    image : SingleImage
        Exact reference tensor used by the experiment.
    group : pd.DataFrame
        All selection rows referring to this source image for the current model.
    cam_record : Optional[Dict[str, Any]]
        Matching Grad-CAM++ manifest row, if one is required/available.
    directory : Path
        Case output directory.
    source_id : str
        Stable digest identifying the source image/mask pair.

    Returns
    -------
    Dict[str, Any]
        Provenance record consumed by :func:`compute_summary`.
    """
    from evaluate_performance import CLASS_NAMES

    cam_path = ""
    if cam_record is not None:
        cam_path = str(cam_record.get("heatmap_npy_path", ""))

    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "image_index": image_index,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "input_mode": mode,
        "target_class_id": target_id,
        "target_class_name": CLASS_NAMES[target_id],
        "baseline": baseline_mode,
        "noise_seed": (
            derive_noise_seed(OcclusionConfiguration.SEED, image_index)
            if baseline_mode == "noise"
            else None
        ),
        "experiment": "sliding+deletion",
        "experiment_id": baseline_mode,
        "input_sha256": tensor_hash(image),
        "cam_path": cam_path,
        "selection_categories": ";".join(dict.fromkeys(group.category.astype(str))),
        "selection_rows": ";".join(str(int(value)) for value in group.source_row_index),
        "output_dir": str(directory),
        "unique_image_id": source_id,
    }


def persist_case_summary(
    output_dir: Path,
    case_record: Dict[str, Any],
    original: InferenceMetrics,
    patch_frame: pd.DataFrame,
    deletion_frame: pd.DataFrame,
    tiles: Sequence[Tile],
    directory: Path,
) -> Dict[str, Any]:
    """Compute and persist one case summary plus the deduplicated global manifest.

    Steps
    -----
    1. Derive faithfulness statistics from original, tile, and deletion measurements.
    2. Save both CSV and strict JSON representations inside the case directory.
    3. Upsert the summary into the run-level occlusion manifest.

    Parameters
    ----------
    output_dir : Path
        Global output directory containing the run-level manifest.
    case_record : Dict[str, Any]
        Provenance metadata for the current model/input/image/configuration case.
    original : InferenceMetrics
        Baseline metrics evaluated on the unmodified reference input.
    patch_frame : pd.DataFrame
        Independent per-tile effects for the numerical CAM ranking.
    deletion_frame : pd.DataFrame
        Progressive deletion results for the ranked and control policies.
    tiles : Sequence[Tile]
        Tile geometry used for computing spatial aggregate statistics.
    directory : Path
        Case output directory receiving local summary files.

    Returns
    -------
    Dict[str, Any]
        Fully populated case summary used for cross-model aggregation.
    """
    summary = compute_summary(
        case_record,
        original,
        patch_frame,
        deletion_frame,
        tiles,
    )
    pd.DataFrame([summary]).to_csv(directory / "summary.csv", index=False)

    # JSON does not permit NaN/Inf. Converting only nonfinite float values to null
    # preserves standards-compliant machine-readable summaries without changing CSV.
    serializable = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in summary.items()
    }
    with open(directory / "summary.json", "w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, allow_nan=False)

    save_csv_upsert(
        output_dir / "occlusion_manifest.csv",
        [summary],
        (
            "model_name",
            "image_path",
            "mask_path",
            "input_mode",
            "target_class_id",
            "experiment_id",
        ),
    )
    return summary


@jaxtyped(typechecker=beartype)
def explain_cases_for_model(
    model: nn.Module,
    model_name: str,
    checkpoint_path: Path,
    model_cases: pd.DataFrame,
    project_root: Path,
    output_dir: Path,
    target_id: int,
    input_modes: Tuple[ConcreteInputMode, ...],
    baseline_mode: BaselineMode,
    cam_index: CamIndex,
    palette: NumpyPalette,
    config: OcclusionConfiguration,
) -> List[Dict[str, Any]]:
    """Generate sliding occlusion and CAM-ranked deletion for one checkpoint.

    The checkpoint is loaded by main. Grouping repeated selection rows
    avoids recomputing the same image while retaining category provenance.

    Steps
    -----
    1. Load each distinct image/mask pair once and skip missing target regions.
    2. Recover the exact Grad-CAM input and numerical CAM for each input mode.
    3. Run sliding attribution and all deletion policies with one baseline.
    4. Save diagnostics and summaries, releasing per-case tensors afterward.

    Parameters
    ----------
    model : nn.Module
        Current evaluation-mode segmentation checkpoint.
    model_name : str
        Checkpoint identifier used in selected cases and Grad-CAM records.
    checkpoint_path : Path
        Resolved checkpoint path recorded in output metadata.
    model_cases : pd.DataFrame
        Selected-case rows belonging to the current model.
    project_root : Path
        Repository root for resolving image/mask and Grad-CAM artifact paths.
    output_dir : Path
        Root directory where occlusion results are saved.
    target_id : int
        Semantic class whose evidence and IoU are evaluated.
    input_modes : Tuple[ConcreteInputMode, ...]
        Concrete input variants to process for each selected image.
    baseline_mode : BaselineMode
        Replacement baseline selected on the command line.
    cam_index : CamIndex
        Manifest lookup for matching numerical CAMs and exact saved inputs.
    palette : NumpyPalette
        Semantic class colors used in diagnostic figures.
    config : OcclusionConfiguration
        Fixed geometry, batching, deletion, and random-control settings.

    Returns
    -------
    List[Dict[str, Any]]
        Per-image/per-input summaries containing only CPU scalar metadata.

    Raises
    ------
    ValueError
        If an expected CAM record is missing or saved reference artifacts differ.
    FileNotFoundError
        If a source image, mask, or saved CAM artifact cannot be found.
    ImportError
        If the Captum dependency required for sliding attribution is unavailable.
    """
    from evaluate_performance import Configuration
    from explain_gradcam import load_single_sample

    summaries: List[Dict[str, Any]] = []
    grouped = model_cases.groupby(
        ["image_index", "image_path", "mask_path"],
        sort=False,
        dropna=False,
    )
    for (index_value, image_value, mask_value), group in grouped:
        image_index = int(float(str(index_value)))
        image_path = resolve_path(str(image_value), project_root)
        mask_path = resolve_path(str(mask_value), project_root)
        clean, gt = load_single_sample(image_path, mask_path, Configuration.DEVICE)

        try:
            # The target-region diagnostic is undefined if the selected class is
            # absent. This also avoids doing costly attribution for that image.
            if not bool((gt == target_id).any()):
                print(f"Skipping {image_path.name}: no target-class GT pixels.")
                continue

            for mode in input_modes:
                key: CamKey = (
                    model_name, mode, target_id, str(image_path), str(mask_path)
                )
                cam_record = cam_index.get(key)
                if cam_record is None:
                    raise ValueError(f"Missing Grad-CAM++ record for {key}.")

                # Use the exact clean/PGD tensor explained by Grad-CAM++; a newly
                # generated robust input would invalidate the comparison.
                image = load_reference_input(cam_record, clean, mode, project_root)
                if tuple(image.shape[-2:]) != tuple(gt.shape):
                    raise ValueError("Input and GT spatial dimensions differ.")
                cam = load_cam_data(cam_record, project_root, tuple(gt.shape))
                baseline = replacement_baseline(
                    image, baseline_mode, OcclusionConfiguration.BLUR_KERNEL, OcclusionConfiguration.BLUR_SIGMA,
                    noise_seed=derive_noise_seed(OcclusionConfiguration.SEED, image_index),
                )
                original = infer_metrics(model, image, gt, target_id)
                validate_cam_reference_score(cam_record, original, mode)

                directory, source_id = prepare_case_directory(
                    config, output_dir, baseline_mode, model_name, mode, image_index,
                    image_path, mask_path, image, cam,
                )
                save_reference_artifacts(directory, image, original)

                # Every case runs BOTH experiments with the same original input
                # and replacement tensor, preserving a fair faithfulness test.
                sensitivity = run_sliding_artifacts(
                    config, model, image, baseline, gt, target_id, directory
                )
                patch_frame, deletion_frame, previews, tiles = run_deletion_artifacts(
                    config, model, image, baseline, gt, target_id, cam,
                    original, directory,
                )
                save_diagnostic(
                    directory, image, gt, original["prediction"][0],
                    cam, sensitivity, previews, tiles, palette, baseline_mode,
                )
                case_record = build_case_record(
                    config, target_id, baseline_mode, model_name,
                    checkpoint_path, image_index, image_path, mask_path, mode,
                    image, group, cam_record, directory, source_id,
                )
                summary = persist_case_summary(
                    output_dir, case_record, original,
                    patch_frame, deletion_frame, tiles, directory,
                )
                summaries.append(summary)
                print(f"  {mode}: {image_path.name} -> {directory}", flush=True)

                # Explicitly discard per-mode arrays, tables, and images. Only
                # small summary dictionaries survive into the next checkpoint.
                del (
                    image, cam, baseline, original, sensitivity, patch_frame,
                    deletion_frame, previews, tiles, case_record, summary,
                )
        finally:
            # A clean reference can remain on CUDA across both input modes, but
            # should never stay alive when advancing to the next selected image.
            del clean, gt

    return summaries


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run both occlusion experiments across the requested named checkpoints.

    Match the Grad-CAM entry point: resolve selections and artifacts once, then
    load, explain, persist, and release checkpoints directly in main.

    Steps
    -----
    1. Parse the five supported CLI options and validate fixed configuration.
    2. Expand selected cases, validate checkpoints, and index saved Grad-CAMs.
    3. For each checkpoint, run both experiments on selected input modes.
    4. Garbage-collect CPU objects and clear unused CUDA cache after each model.
    5. Write paired model-comparison tables and report the finished case count.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Explicit CLI tokens for testing, or None to parse process arguments.

    Returns
    -------
    None
        Saves per-case occlusion artifacts and aggregate manifests.

    Raises
    ------
    FileNotFoundError
        If a required selected-case file, manifest, or checkpoint is missing.
    ValueError
        If checkpoint names, target class, or saved CAM artifacts are inconsistent.
    ImportError
        If Captum is not installed in the execution environment.
    """
    from evaluate_performance import CLASS_COLORS, CLASS_NAMES, Configuration, load_trained_model
    from explain_gradcam import (
        expand_and_filter_cases,
        parse_checkpoint_spec,
        resolve_target_class,
    )

    args = parse_arguments(argv)
    config = OcclusionConfiguration()
    validate_configuration(config)
    torch.manual_seed(OcclusionConfiguration.SEED)
    np.random.seed(OcclusionConfiguration.SEED)

    project_root = Path(OcclusionConfiguration.PROJECT_ROOT).expanduser().resolve()
    selected_cases_path = resolve_path(args.selected_cases, project_root)
    if not selected_cases_path.is_file():
        raise FileNotFoundError(
            f"Selected-case manifest does not exist: {selected_cases_path}"
        )

    # Validate all names and checkpoint paths before loading any model so a
    # missing later checkpoint cannot waste hours of earlier attribution work.
    model_specs: Tuple[ModelSpec, ...] = tuple(
        parse_checkpoint_spec(specification) for specification in args.checkpoint
    )
    model_names = [name for name, _ in model_specs]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Every --checkpoint must use a unique model name.")
    checkpoint_map: Dict[str, Path] = {}
    for model_name, checkpoint_name in model_specs:
        checkpoint_path = resolve_path(checkpoint_name, project_root)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint for '{model_name}' does not exist: {checkpoint_path}"
            )
        checkpoint_map[model_name] = checkpoint_path

    target_id = resolve_target_class(OcclusionConfiguration.TARGET_CLASS)
    if target_id == IGNORE_ID:
        raise ValueError("The ignored class cannot be the explanation target.")
    if args.input_mode == "all":
        input_modes: Tuple[ConcreteInputMode, ...] = ("clean", "robust")
    else:
        input_modes = (cast(ConcreteInputMode, args.input_mode),)
    baseline_mode = cast(BaselineMode, args.baseline)

    selected = pd.read_csv(selected_cases_path)
    cases = expand_and_filter_cases(selected, model_names, max_cases=OcclusionConfiguration.MAX_CASES)
    output_dir = resolve_path(OcclusionConfiguration.OUTPUT_DIR, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_path = output_dir / "expanded_cases.csv"
    cases.to_csv(expanded_path, index=False)
    if cases.empty:
        raise ValueError("No selected cases match the supplied checkpoints.")

    # Deletion always runs alongside sliding occlusion, so every processed
    # case requires a numerical CAM from the prior Grad-CAM++ execution.
    gradcam_path = resolve_path(args.gradcam_manifest, project_root)
    cam_index = load_cam_index(gradcam_path, project_root)
    palette = cast(NumpyPalette, np.asarray(CLASS_COLORS, dtype=np.uint8))

    print("\n" + "=" * 72)
    print("OCCLUSION SEMANTIC-SEGMENTATION FAITHFULNESS")
    print("=" * 72)
    print(f"Execution device : {Configuration.DEVICE}")
    print(f"Selected cases   : {selected_cases_path}")
    print(f"Cases to explain : {len(cases)}")
    print(f"Expanded cases   : {expanded_path}")
    print(f"Grad-CAM manifest: {gradcam_path}")
    print(f"Target class     : {CLASS_NAMES[target_id]} ({target_id})")
    print(f"Input mode       : {args.input_mode}")
    print(f"Baseline         : {baseline_mode}")
    print("Experiments      : sliding + deletion")
    print(f"Output directory : {output_dir}")

    all_summaries: List[Dict[str, Any]] = []
    for model_name, checkpoint_path in checkpoint_map.items():
        model_cases = cases[cases["model_name"] == model_name].copy()
        if model_cases.empty:
            continue

        print(
            f"\nExplaining {len(model_cases)} selected row(s) for "
            f"'{model_name}' from {checkpoint_path}...", flush=True,
        )
        model = load_trained_model(str(checkpoint_path), Configuration.DEVICE)
        try:
            model.eval()
            model_summaries = explain_cases_for_model(
                model, model_name, checkpoint_path, model_cases,
                project_root, output_dir, target_id, input_modes,
                baseline_mode, cam_index, palette, config,
            )
            all_summaries.extend(model_summaries)
            del model_summaries
        finally:
            # Clear live model references first, then collect unreachable CPU
            # objects, THEN release the CUDA allocator's unused cached blocks.
            # empty_cache() alone cannot free tensors that are still referenced.
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model_cases

    if all_summaries:
        write_model_comparisons(output_dir, all_summaries, OcclusionConfiguration.SEED)
    print(
        "\nOcclusion complete: "
        f"{len(all_summaries)} unique model/input/image cases."
    )


if __name__ == "__main__":
    main()
