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
DEFAULT_BUDGETS = (0.0, 0.05, 0.10, 0.20, 0.30)
# These three metrics are always evaluated on the same reference image and
# fixed GT mask; keep their keys literal to help static TypedDict checking.
MetricName = Literal["global_logit", "gt_logit", "road_iou"]
METRIC_NAMES: Tuple[MetricName, ...] = ("global_logit", "gt_logit", "road_iou")
# Blurred replacement is the recommended primary perturbation because it removes
# local high-frequency evidence without introducing an arbitrary solid color.
BaselineMode = Literal["blur", "gray", "image_mean", "white", "bright"]
ConcreteInputMode = Literal["clean", "robust"]
ModelSpec = Tuple[str, str]
CamKey = Tuple[str, str, int, str, str]
CamIndex = Dict[CamKey, Dict[str, Any]]


@dataclass(frozen=True)
class PipelineContext:
    """Store immutable paths and selections shared by the whole pipeline.

    The context separates one-time setup from per-model and per-image processing.
    Keeping these values together makes the orchestration helpers explicit about
    which state is global to the run and avoids repeatedly resolving manifests.

    Steps
    -----
    1. Retain resolved repository/output paths and validated checkpoint specs.
    2. Keep expanded selected cases, input modes, target class, CAM index, and palette.

    Attributes
    ----------
    root : Path
        Resolved repository root used for project-relative input paths.
    output : Path
        Root directory receiving occlusion artifacts and aggregate manifests.
    model_specs : Tuple[ModelSpec, ...]
        Validated ``(model_name, checkpoint_path)`` pairs from the CLI.
    target_id : int
        Semantic class ID whose Grad-CAM++ faithfulness is evaluated.
    input_modes : Tuple[ConcreteInputMode, ...]
        Concrete input variants to process: clean, robust, or both.
    cases : pd.DataFrame
        Expanded selected-case rows after checkpoint filtering and optional limiting.
    cam_index : CamIndex
        Lookup of Grad-CAM++ manifest records keyed by model, mode, class, and paths.
    palette : NumpyPalette
        Dataset RGB class palette used only for diagnostic visualisation.
    """

    root: Path
    output: Path
    model_specs: Tuple[ModelSpec, ...]
    target_id: int
    input_modes: Tuple[ConcreteInputMode, ...]
    cases: pd.DataFrame
    cam_index: CamIndex
    palette: NumpyPalette


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


@jaxtyped(typechecker=beartype)
def replacement_baseline(
    image: SingleImage,
    mode: BaselineMode,
    blur_kernel: int,
    blur_sigma: float,
) -> SingleImage:
    """Select the replacement image used by every occlusion policy.

    A Gaussian-blurred reference is preferred over a fixed solid color because it
    suppresses local texture and object detail while preserving coarse illumination
    and scene color. This reduces the risk that the perturbation itself becomes an
    out-of-distribution cue. ``gray``, ``white``, ``bright``, and ``image_mean``
    remain available for baseline-sensitivity experiments.

    Steps
    -----
    1. Validate the NCHW RGB input and its expected ``[0, 1]`` intensity domain.
    2. Return a blurred, fixed-gray, white, bright, or image-adaptive mean-color replacement.

    Parameters
    ----------
    image : SingleImage
        Reference NCHW RGB image tensor, normally a single example.
    mode : BaselineMode
        Replacement strategy: ``blur`` (recommended), ``gray``, ``white``, ``bright``, or ``image_mean``.
    blur_kernel : int
        Odd Gaussian kernel size used only for ``blur`` mode.
    blur_sigma : float
        Positive Gaussian standard deviation used only for ``blur`` mode.

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

    # Blurring is the primary baseline: it removes local discriminative structure
    # without painting the road an arbitrary color that the model may interpret as
    # a new semantic object or, conversely, as another road-like surface.
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
    1. Generate each budget variant from the unchanged reference image.
    2. Infer global logits, GT logits, IoU, and predictions in batches.
    3. Record actual area, GT-road/boundary coverage, and selected high-CAM previews.

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
    batches: List[Tensor] = []
    selections: List[List[int]] = []
    # All budgets are constructed from the original image, ensuring that the
    # only changing variable is the union of selected tile regions.
    for budget in budgets:
        chosen = selected_at_budget(tiles, order, budget, h * w)
        selections.append(chosen)
        batches.append(apply_tiles(image, baseline, tiles, chosen))
    for start in range(0, len(batches), batch_size):
        group_images = torch.cat(batches[start:start + batch_size])
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
                preview[f"image_{j}"] = batches[j].clone()
                preview[f"prediction_{j}"] = torch.from_numpy(metrics["prediction"][offset].copy())
                preview[f"indices_{j}"] = torch.tensor(chosen, dtype=torch.int64)
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

    Robust inputs are never regenerated in this pipeline. The function prefers
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


def parse_pair(raw: str) -> Tuple[int, int]:
    """Parse square or HxW CLI patch and tile geometry.

    Steps
    -----
    1. Accept one positive integer as a square dimension.
    2. Otherwise parse two positive integer dimensions separated by x.

    Parameters
    ----------
    raw : str
        Unparsed command-line value.

    Returns
    -------
    Tuple[int, int]
        Positive (height, width) tuple.

    Raises
    ------
    argparse.ArgumentTypeError
        If the geometry is malformed or contains nonpositive dimensions.
    """
    parts = raw.lower().split("x")
    if len(parts) == 1:
        parts *= 2
    if len(parts) != 2 or any(not p.isdecimal() or int(p) <= 0 for p in parts):
        raise argparse.ArgumentTypeError("Expected a positive integer or HxW, e.g. 32 or 16x32.")
    return int(parts[0]), int(parts[1])


def parse_budgets(raw: str) -> Tuple[float, ...]:
    """Parse and normalize the cumulative deletion budget schedule.

    Steps
    -----
    1. Convert comma-separated fractions to floats and deduplicate them.
    2. Sort budgets and require zero plus values within [0, 1].

    Parameters
    ----------
    raw : str
        Unparsed command-line value.

    Returns
    -------
    Tuple[float, ...]
        Sorted tuple of distinct cumulative area fractions.

    Raises
    ------
    argparse.ArgumentTypeError
        If fractions are invalid, outside [0, 1], or omit zero.
    """
    try:
        budgets = tuple(sorted(set(float(v) for v in raw.split(","))))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--budgets must be comma-separated fractions.") from exc
    if not budgets or budgets[0] < 0 or budgets[-1] > 1 or 0.0 not in budgets:
        raise argparse.ArgumentTypeError("Budgets must include 0 and lie in [0,1].")
    return budgets


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Configure checkpoint, attribution, perturbation, and reporting options.

    Steps
    -----
    1. Register the manifest, checkpoint, input-mode, and output arguments.
    2. Register baseline geometry, budget, batching, and reproducibility controls.
    3. Parse the supplied command-line tokens or process arguments.

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
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH (repeatable), matching selected_cases.csv.")
    parser.add_argument("--selected-cases", default="./model/xai_selection/selected_cases.csv")
    parser.add_argument("--gradcam-manifest", default="./model/xai_gradcam/gradcam_manifest.csv")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="./model/xai_occlusion")
    parser.add_argument("--target-class", default="road")
    parser.add_argument("--input-mode", choices=("clean", "robust", "all"), default="clean")
    parser.add_argument("--experiment", choices=("sliding", "deletion", "all"), default="all")
    parser.add_argument("--patch-size", type=parse_pair, default=(32, 32))
    parser.add_argument("--stride", type=parse_pair, default=(32, 32))
    parser.add_argument(
        "--baseline",
        choices=("blur", "gray", "white", "bright", "image_mean"),
        default="white",
        help=(
            "Occlusion replacement. 'white' is recommended because road, the default target class, has dark grey or grey color."
            "By using 'white' color to occlude, the change in the segmentation will be more visible,"
            " and this proves the Grad-CAM explanation;"
            "'bright' is stronger high-intensity occluders, while 'gray' and "
            "'image_mean' remain useful sensitivity baselines."
            "'blur' helps to blur the occlusion area."
        ),
    )
    parser.add_argument("--blur-kernel", type=int, default=31)
    parser.add_argument("--blur-sigma", type=float, default=8.0)
    parser.add_argument("--perturbations-per-eval", type=int, default=4)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--tile-size", type=parse_pair, default=(32, 32))
    parser.add_argument("--budgets", type=parse_budgets, default=DEFAULT_BUDGETS)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--interior-threshold", type=float, default=0.8)
    parser.add_argument("--sliding-gt-target", action="store_true", help="Also compute a GT-target-logit Captum map.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject inconsistent runtime options before expensive model inference.

    Steps
    -----
    1. Check positive batch/repetition sizes and valid road-interior fractions.
    2. Ensure strides have full coverage and Gaussian parameters are usable.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.

    Returns
    -------
    None
        Nothing; raises when an option is invalid.

    Raises
    ------
    ValueError
        If numerical CLI options are outside their supported ranges.
    """
    if args.perturbations_per_eval < 1 or args.random_repeats < 1:
        raise ValueError("Batch size and random repeats must be positive.")
    if not 0 < args.interior_threshold <= 1:
        raise ValueError("--interior-threshold must lie in (0,1].")
    if any(x > y for x, y in zip(args.stride, args.patch_size, strict=True)):
        raise ValueError("Sliding stride cannot exceed patch size.")
    if args.baseline == "blur" and (
        args.blur_kernel < 3 or args.blur_kernel % 2 == 0 or args.blur_sigma <= 0
    ):
        raise ValueError("Gaussian blur requires an odd kernel >= 3 and sigma > 0.")


@dataclass
class CaseExperimentArtifacts:
    """Collect optional outputs produced for one model/input/image experiment.

    The pipeline can run sliding attribution, deletion, or both. A dedicated
    container keeps those conditional outputs together so orchestration code does
    not need several loosely related local variables.

    Steps
    -----
    1. Store the optional sliding sensitivity map.
    2. Store deletion tables, preview tensors, and tile metadata when CAM deletion runs.

    Attributes
    ----------
    sensitivity : Optional[NumpyHeatmap]
        Signed Captum occlusion map for the global target logit, when requested.
    patch_frame : Optional[pd.DataFrame]
        Independent one-tile ablation effects, when a numerical CAM is available.
    deletion_frame : Optional[pd.DataFrame]
        Progressive deletion measurements for all requested ranking policies.
    previews : Dict[str, Tensor]
        Top-first deleted images, predictions, and tile indices used in diagnostics.
    tiles : List[Tile]
        Deterministic nonoverlapping tile grid used for CAM ranking and deletion.
    """

    sensitivity: Optional[NumpyHeatmap]
    patch_frame: Optional[pd.DataFrame]
    deletion_frame: Optional[pd.DataFrame]
    previews: Dict[str, Tensor]
    tiles: List[Tile]


def prepare_pipeline_context(args: argparse.Namespace) -> PipelineContext:
    """Resolve one-time pipeline configuration and load selection metadata.

    This helper performs all inexpensive setup before any checkpoint is loaded.
    Centralising setup keeps :func:`run_pipeline` focused on orchestration and
    ensures every downstream helper receives resolved, validated paths.

    Steps
    -----
    1. Resolve project/output paths, checkpoint specifications, and target class.
    2. Expand selected cases for the requested checkpoints and concrete input modes.
    3. Load the Grad-CAM++ manifest only when the requested experiment requires it.
    4. Package shared immutable state into :class:`PipelineContext`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.

    Returns
    -------
    PipelineContext
        Shared resolved state used by per-model and per-case processing helpers.

    Raises
    ------
    ValueError
        If checkpoint names are duplicated, the ignored class is targeted, or no
        selected cases match the supplied checkpoints.
    FileNotFoundError
        If a required Grad-CAM++ manifest is missing.
    """
    # Delay project-specific imports until execution so numerical helpers remain
    # independently importable in synthetic tests that do not have the dataset.
    from explain_gradcam import (
        expand_and_filter_cases,
        parse_checkpoint_spec,
        resolve_target_class,
    )
    from evaluate_performance import CLASS_COLORS

    root = Path(args.project_root).expanduser().resolve()
    output = resolve_path(args.output_dir, root)
    output.mkdir(parents=True, exist_ok=True)

    # A model name is part of the CAM lookup key and output path. Duplicate names
    # would therefore make provenance ambiguous even if checkpoint paths differ.
    model_specs = tuple(parse_checkpoint_spec(value) for value in args.checkpoint)
    model_names = [name for name, _ in model_specs]
    if len(set(model_names)) != len(model_names):
        raise ValueError("Duplicate model names in --checkpoint.")

    target_id = resolve_target_class(args.target_class)
    if target_id == IGNORE_ID:
        raise ValueError("The ignored class cannot be the explanation target.")

    if args.input_mode == "all":
        input_modes: Tuple[ConcreteInputMode, ...] = ("clean", "robust")
    else:
        input_modes = (cast(ConcreteInputMode, args.input_mode),)

    selected_path = resolve_path(args.selected_cases, root)
    selected = pd.read_csv(selected_path)
    cases = expand_and_filter_cases(selected, model_names, max_cases=args.max_cases)
    if cases.empty:
        raise ValueError("No selected cases match the supplied checkpoints.")
    cases.to_csv(output / "expanded_cases.csv", index=False)

    # Robust attribution must reuse the exact PGD tensor saved by Grad-CAM++, and
    # deletion always needs a numerical CAM. Clean sliding-only runs can proceed
    # without the manifest, which keeps that mode independently usable.
    cam_path = resolve_path(args.gradcam_manifest, root)
    manifest_required = args.experiment != "sliding" or "robust" in input_modes
    cam_index = load_cam_index(cam_path, root) if manifest_required or cam_path.exists() else {}
    palette = cast(NumpyPalette, np.asarray(CLASS_COLORS, dtype=np.uint8))

    return PipelineContext(
        root=root,
        output=output,
        model_specs=model_specs,
        target_id=target_id,
        input_modes=input_modes,
        cases=cases,
        cam_index=cam_index,
        palette=palette,
    )


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
        Matching Grad-CAM++ manifest record, or ``None`` for clean sliding-only runs.
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
            "Regenerate Grad-CAM++ with the updated artifact pipeline and verify "
            "that the same checkpoint is used."
        )


@jaxtyped(typechecker=beartype)
def prepare_case_directory(
    args: argparse.Namespace,
    context: PipelineContext,
    model_name: str,
    mode: ConcreteInputMode,
    image_index: int,
    image_path: Path,
    mask_path: Path,
    image: SingleImage,
    cam: Optional[NumpyHeatmap],
) -> Tuple[Path, str, str]:
    """Create a deterministic output directory for one experiment configuration.

    The experiment identifier hashes all settings that change the perturbation or
    reference tensors. This prevents reruns with different baselines, budgets, or
    CAM/input artifacts from silently overwriting one another.

    Steps
    -----
    1. Hash the source image/mask identity and all experiment-defining parameters.
    2. Build the model/mode/image/config directory hierarchy.
    3. Persist the exact configuration JSON used by the experiment.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.
    context : PipelineContext
        Shared resolved pipeline state.
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
    cam : Optional[NumpyHeatmap]
        Numerical Grad-CAM++ map used for deletion, if requested.

    Returns
    -------
    Tuple[Path, str, str]
        Case output directory, source identity digest, and experiment-config digest.
    """
    from explain_gradcam import sanitise_filename

    source_id = hashlib.sha256(f"{image_path}|{mask_path}".encode()).hexdigest()[:12]
    parameters: Dict[str, Any] = {
        "experiment": args.experiment,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "tile_size": args.tile_size,
        "baseline": args.baseline,
        "blur_kernel": args.blur_kernel,
        "blur_sigma": args.blur_sigma,
        "budgets": args.budgets,
        "random_repeats": args.random_repeats,
        "seed": args.seed,
        "interior_threshold": args.interior_threshold,
        "sliding_gt_target": args.sliding_gt_target,
        "input_hash": tensor_hash(image),
    }
    if cam is not None:
        parameters["cam_hash"] = sha256_array(cam)

    config_id = hashlib.sha256(
        json.dumps(parameters, sort_keys=True).encode()
    ).hexdigest()[:12]
    directory = (
        context.output
        / sanitise_filename(model_name)
        / mode
        / f"img{image_index:05d}_{source_id}"
        / config_id
    )
    directory.mkdir(parents=True, exist_ok=True)

    # Persisting the configuration beside the outputs makes every figure and CSV
    # auditable without needing to reconstruct CLI arguments from shell history.
    with open(directory / "configuration.json", "w", encoding="utf-8") as file:
        json.dump(parameters, file, indent=2)

    return directory, source_id, config_id


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
    args: argparse.Namespace,
    model: nn.Module,
    image: SingleImage,
    baseline: SingleImage,
    gt: ClassMap,
    target_id: int,
    directory: Path,
) -> Optional[NumpyHeatmap]:
    """Run and persist requested Captum sliding-window attribution maps.

    Sliding attribution is independent from CAM-ranked deletion. It measures the
    signed change in the same scalar target when windows are replaced throughout
    the full image, providing an additional perturbation-based explanation.

    Steps
    -----
    1. Skip the computation when the CLI requests deletion only.
    2. Compute and save the global-logit signed occlusion map.
    3. Optionally compute the secondary fixed-GT-road-logit map.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.
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
    Optional[NumpyHeatmap]
        Global signed sensitivity map, or ``None`` when sliding attribution is disabled.

    Raises
    ------
    ImportError
        If Captum is unavailable while sliding attribution is requested.
    ValueError
        If patch/stride geometry is invalid for the current image.
    RuntimeError
        If Captum returns malformed or nonfinite attribution values.
    """
    if args.experiment not in ("all", "sliding"):
        return None

    sensitivity = sliding_occlusion(
        model,
        image,
        target_id,
        gt,
        args.patch_size,
        args.stride,
        baseline,
        args.perturbations_per_eval,
    )
    np.save(directory / "occlusion_global_signed.npy", sensitivity)
    save_heatmap(
        image,
        sensitivity,
        directory / "occlusion_global.png",
        "Full-image road-logit sensitivity",
    )

    if args.sliding_gt_target:
        secondary = sliding_occlusion(
            model,
            image,
            target_id,
            gt,
            args.patch_size,
            args.stride,
            baseline,
            args.perturbations_per_eval,
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
    args: argparse.Namespace,
    model: nn.Module,
    image: SingleImage,
    baseline: SingleImage,
    gt: ClassMap,
    target_id: int,
    cam: Optional[NumpyHeatmap],
    original: InferenceMetrics,
    directory: Path,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Tensor], List[Tile]]:
    """Run CAM-ranked tile ablation and progressive deletion experiments.

    Deletion is the direct Grad-CAM++ faithfulness test: top-ranked regions are
    destructively replaced and compared with low-ranked and random controls under
    the same area budgets. A missing CAM means the caller requested sliding only.

    Steps
    -----
    1. Build the deterministic tile grid and measure each tile independently.
    2. Evaluate top, bottom, random, and road/location-matched deletion orders.
    3. Save raw tables, deletion curves, and the strongest top-first preview artifacts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.
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
    cam : Optional[NumpyHeatmap]
        Numerical Grad-CAM++ attribution map; ``None`` disables deletion.
    original : InferenceMetrics
        Baseline metrics evaluated before any tile replacement.
    directory : Path
        Case output directory receiving deletion artifacts.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Tensor], List[Tile]]
        Independent tile-effect table, progressive deletion table, diagnostic preview
        tensors, and the tile grid. Empty/``None`` outputs are returned when no CAM exists.
    """
    if cam is None:
        return None, None, {}, []

    # The same tile geometry is used for all policies so only ranking—not deleted
    # area or patch shape—changes between Grad-CAM++ and control interventions.
    tiles = build_tiles(cam, gt, target_id, args.tile_size, args.interior_threshold)
    patch_frame = tile_effects(
        model,
        image,
        baseline,
        gt,
        target_id,
        tiles,
        args.perturbations_per_eval,
        original,
    )
    patch_frame.to_csv(directory / "tile_effects.csv", index=False)

    deletion_rows: List[Dict[str, Any]] = []
    previews: Dict[str, Tensor] = {}
    for policy in ("top", "bottom", "random", "matched_random"):
        repeats = args.random_repeats if policy in ("random", "matched_random") else 1
        for repeat in range(repeats):
            # Use separated deterministic seeds so random controls are reproducible
            # without accidentally sharing the same permutation across repetitions.
            order = deletion_order(tiles, policy, seed=args.seed + repeat * 7919)
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
                args.budgets,
                original,
                args.perturbations_per_eval,
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


@jaxtyped(typechecker=beartype)
def execute_case_experiments(
    args: argparse.Namespace,
    model: nn.Module,
    image: SingleImage,
    baseline: SingleImage,
    gt: ClassMap,
    target_id: int,
    cam: Optional[NumpyHeatmap],
    original: InferenceMetrics,
    directory: Path,
) -> CaseExperimentArtifacts:
    """Execute all attribution/deletion components requested for one reference input.

    This helper is intentionally thin: it composes the independent sliding and
    deletion helpers and returns their outputs in one typed container for later
    diagnostics and summary generation.

    Steps
    -----
    1. Run optional full-image sliding occlusion.
    2. Run optional CAM-ranked deletion and collect its tables/previews.
    3. Return every conditional artifact in a single container.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.
    model : nn.Module
        Loaded evaluation-mode segmentation checkpoint.
    image : SingleImage
        Exact reference tensor used by all interventions.
    baseline : SingleImage
        Occlusion replacement tensor aligned with ``image``.
    gt : ClassMap
        Fixed ground-truth semantic class map.
    target_id : int
        Semantic target class ID.
    cam : Optional[NumpyHeatmap]
        Numerical Grad-CAM++ map for ranked deletion, when required.
    original : InferenceMetrics
        Metrics on the unmodified reference input.
    directory : Path
        Case output directory.

    Returns
    -------
    CaseExperimentArtifacts
        Sliding/deletion outputs required for diagnostics and summary statistics.
    """
    sensitivity = run_sliding_artifacts(
        args, model, image, baseline, gt, target_id, directory
    )
    patch_frame, deletion_frame, previews, tiles = run_deletion_artifacts(
        args,
        model,
        image,
        baseline,
        gt,
        target_id,
        cam,
        original,
        directory,
    )
    return CaseExperimentArtifacts(
        sensitivity=sensitivity,
        patch_frame=patch_frame,
        deletion_frame=deletion_frame,
        previews=previews,
        tiles=tiles,
    )


def build_case_record(
    args: argparse.Namespace,
    context: PipelineContext,
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
    config_id: str,
) -> Dict[str, Any]:
    """Build provenance metadata used by per-case and global summary manifests.

    Selection rows can repeat the same source image under multiple categories. The
    record preserves those categories as provenance while identifying the actual
    experiment by model, input mode, source image, and configuration digest.

    Steps
    -----
    1. Resolve the target class name and optional CAM path.
    2. Collapse repeated selection categories/source rows into semicolon lists.
    3. Record checkpoint, input hash, output directory, and experiment identifiers.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.
    context : PipelineContext
        Shared pipeline state containing the target ID and resolved output context.
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
    config_id : str
        Digest identifying the experiment configuration and reference artifacts.

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
        "target_class_id": context.target_id,
        "target_class_name": CLASS_NAMES[context.target_id],
        "baseline": args.baseline,
        "experiment": args.experiment,
        "experiment_id": config_id,
        "input_sha256": tensor_hash(image),
        "cam_path": cam_path,
        "selection_categories": ";".join(dict.fromkeys(group.category.astype(str))),
        "selection_rows": ";".join(str(int(value)) for value in group.source_row_index),
        "output_dir": str(directory),
        "unique_image_id": source_id,
    }


def persist_case_summary(
    context: PipelineContext,
    case_record: Dict[str, Any],
    original: InferenceMetrics,
    artifacts: CaseExperimentArtifacts,
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
    context : PipelineContext
        Shared pipeline state containing the global output directory.
    case_record : Dict[str, Any]
        Provenance metadata for the current model/input/image/configuration case.
    original : InferenceMetrics
        Baseline metrics evaluated on the unmodified reference input.
    artifacts : CaseExperimentArtifacts
        Optional sliding/deletion outputs produced for this case.
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
        artifacts.patch_frame,
        artifacts.deletion_frame,
        artifacts.tiles,
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
        context.output / "occlusion_manifest.csv",
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
def process_input_mode(
    args: argparse.Namespace,
    context: PipelineContext,
    model: nn.Module,
    model_name: str,
    checkpoint_path: Path,
    image_index: int,
    image_path: Path,
    mask_path: Path,
    clean: SingleImage,
    gt: ClassMap,
    group: pd.DataFrame,
    mode: ConcreteInputMode,
) -> Dict[str, Any]:
    """Process one clean or robust reference input from loading through persistence.

    This is the smallest orchestration unit that represents one scientific case:
    one checkpoint, one source image, one concrete input mode, and one target class.
    Keeping this workflow separate makes each validation and artifact transition easy
    to follow without nesting it inside model/image loops.

    Steps
    -----
    1. Resolve the matching CAM record and exact saved reference tensor.
    2. Validate geometry/reference score and construct the replacement baseline.
    3. Create a deterministic output directory and save unmodified artifacts.
    4. Execute sliding/deletion experiments, diagnostics, and summary persistence.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.
    context : PipelineContext
        Shared pipeline state and manifest indexes.
    model : nn.Module
        Loaded evaluation-mode segmentation checkpoint.
    model_name : str
        Human-readable checkpoint identifier.
    checkpoint_path : Path
        Resolved checkpoint file path.
    image_index : int
        Stable test-split source image index.
    image_path : Path
        Resolved RGB source image path.
    mask_path : Path
        Resolved source segmentation-mask path.
    clean : SingleImage
        Freshly preprocessed clean dataset tensor.
    gt : ClassMap
        Fixed ground-truth semantic class map.
    group : pd.DataFrame
        Selection rows referring to this source image for the current model.
    mode : ConcreteInputMode
        Concrete input variant to evaluate.

    Returns
    -------
    Dict[str, Any]
        Completed summary record for this unique scientific case.

    Raises
    ------
    ValueError
        If a required CAM record is absent, saved input/GT geometry disagrees, or
        the saved Grad-CAM++ reference score is inconsistent.
    OSError
        If a referenced NumPy CAM/input artifact cannot be read.
    ImportError
        If Captum is requested but unavailable.
    """
    key: CamKey = (
        model_name,
        mode,
        context.target_id,
        str(image_path),
        str(mask_path),
    )
    cam_record = context.cam_index.get(key)
    if args.experiment != "sliding" and cam_record is None:
        raise ValueError(f"Missing Grad-CAM++ record for {key}.")

    # Robust mode must reuse the exact PGD tensor from Grad-CAM++; regenerating an
    # attack here would test a different input and invalidate the faithfulness link.
    image = load_reference_input(cam_record, clean, mode, context.root)
    if tuple(image.shape[-2:]) != tuple(gt.shape):
        raise ValueError("Input and GT spatial dimensions differ.")

    cam = None
    if args.experiment != "sliding":
        if cam_record is None:
            raise ValueError(f"Missing Grad-CAM++ record for {key}.")
        cam = load_cam_data(cam_record, context.root, tuple(gt.shape))

    baseline_mode = cast(BaselineMode, args.baseline)
    baseline = replacement_baseline(
        image, baseline_mode, args.blur_kernel, args.blur_sigma
    )
    original = infer_metrics(model, image, gt, context.target_id)
    validate_cam_reference_score(cam_record, original, mode)

    directory, source_id, config_id = prepare_case_directory(
        args,
        context,
        model_name,
        mode,
        image_index,
        image_path,
        mask_path,
        image,
        cam,
    )
    save_reference_artifacts(directory, image, original)

    artifacts = execute_case_experiments(
        args,
        model,
        image,
        baseline,
        gt,
        context.target_id,
        cam,
        original,
        directory,
    )
    save_diagnostic(
        directory,
        image,
        gt,
        original["prediction"][0],
        cam,
        artifacts.sensitivity,
        artifacts.previews,
        artifacts.tiles,
        context.palette,
        baseline_mode,
    )

    case_record = build_case_record(
        args,
        context,
        model_name,
        checkpoint_path,
        image_index,
        image_path,
        mask_path,
        mode,
        image,
        group,
        cam_record,
        directory,
        source_id,
        config_id,
    )
    summary = persist_case_summary(
        context, case_record, original, artifacts, directory
    )
    print(f"  {mode}: {image_path.name} -> {directory}", flush=True)
    return summary


def process_image_group(
    args: argparse.Namespace,
    context: PipelineContext,
    model: nn.Module,
    model_name: str,
    checkpoint_path: Path,
    image_index_value: object,
    image_path_value: object,
    mask_path_value: object,
    group: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Process every requested input mode for one unique source image/mask pair.

    Steps
    -----
    1. Resolve the selected source paths and load the clean image/ground truth once.
    2. Skip cases without any target-class ground-truth pixels.
    3. Reuse the loaded clean sample while processing each concrete input mode.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.
    context : PipelineContext
        Shared pipeline state including requested input modes and target class.
    model : nn.Module
        Loaded evaluation-mode segmentation checkpoint.
    model_name : str
        Human-readable checkpoint identifier.
    checkpoint_path : Path
        Resolved checkpoint path used by ``model``.
    image_index_value : object
        Pandas group-key value containing the stable image index.
    image_path_value : object
        Pandas group-key value containing the source RGB path.
    mask_path_value : object
        Pandas group-key value containing the source mask path.
    group : pd.DataFrame
        All selection rows for this model and source image/mask pair.

    Returns
    -------
    List[Dict[str, Any]]
        One summary record per processed concrete input mode.

    Raises
    ------
    FileNotFoundError
        If the selected source image or mask cannot be loaded.
    ValueError
        If input/CAM/reference artifacts are inconsistent.
    """
    from explain_gradcam import load_single_sample
    from evaluate_performance import Configuration

    image_index = int(float(str(image_index_value)))
    image_path = resolve_path(str(image_path_value), context.root)
    mask_path = resolve_path(str(mask_path_value), context.root)
    clean, gt = load_single_sample(image_path, mask_path, Configuration.DEVICE)

    # A GT-region diagnostic is undefined when the target class is absent. Skipping
    # here prevents later metric failures and mirrors the original selection intent.
    if not bool((gt == context.target_id).any()):
        print(f"Skipping {image_path_value}: no GT target pixels.")
        return []

    summaries: List[Dict[str, Any]] = []
    for mode in context.input_modes:
        summaries.append(
            process_input_mode(
                args,
                context,
                model,
                model_name,
                checkpoint_path,
                image_index,
                image_path,
                mask_path,
                clean,
                gt,
                group,
                mode,
            )
        )
    return summaries


def process_model(
    args: argparse.Namespace,
    context: PipelineContext,
    model_name: str,
    checkpoint_name: str,
) -> List[Dict[str, Any]]:
    """Load one checkpoint once and process its unique selected source images.

    Steps
    -----
    1. Filter selected rows for the model and validate/load its checkpoint.
    2. Group repeated selection categories by source image/mask pair.
    3. Delegate each group to :func:`process_image_group` and release GPU memory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.
    context : PipelineContext
        Shared pipeline paths, selections, target class, and CAM index.
    model_name : str
        Human-readable checkpoint identifier supplied on the CLI.
    checkpoint_name : str
        Checkpoint path as supplied by ``parse_checkpoint_spec``.

    Returns
    -------
    List[Dict[str, Any]]
        Summary records for all processed source images and input modes of this model.

    Raises
    ------
    FileNotFoundError
        If the checkpoint path does not exist.
    ValueError
        If any selected case has inconsistent reference/CAM artifacts.
    """
    from evaluate_performance import Configuration, load_trained_model

    frame = context.cases[context.cases.model_name == model_name]
    if frame.empty:
        return []

    checkpoint_path = resolve_path(checkpoint_name, context.root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    print(f"\nModel: {model_name} ({len(frame)} selected rows)", flush=True)
    model = load_trained_model(str(checkpoint_path), Configuration.DEVICE)
    model.eval()
    summaries: List[Dict[str, Any]] = []

    try:
        # Deduplicate repeated selection categories before expensive inference. The
        # category provenance is preserved in ``group`` and written into each summary.
        grouped = frame.groupby(
            ["image_index", "image_path", "mask_path"],
            sort=False,
            dropna=False,
        )
        for (image_index, image_path, mask_path), group in grouped:
            summaries.extend(
                process_image_group(
                    args,
                    context,
                    model,
                    model_name,
                    checkpoint_path,
                    image_index,
                    image_path,
                    mask_path,
                    group,
                )
            )
    finally:
        # Explicitly release the model between checkpoints because robust deletion
        # can create many perturbation batches and otherwise leave cached GPU memory.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summaries


def run_pipeline(args: argparse.Namespace) -> None:
    """Coordinate occlusion faithfulness analysis across all requested checkpoints.

    The detailed work is intentionally delegated to smaller helpers. This function
    now describes the top-level program flow only: prepare shared state, process each
    checkpoint, aggregate summaries, and report completion.

    Steps
    -----
    1. Build one shared :class:`PipelineContext` from CLI arguments and manifests.
    2. Process each checkpoint independently while accumulating per-case summaries.
    3. Write paired model-comparison outputs when at least one case completed.
    4. Print the final number of unique model/input/image cases.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and validated command-line namespace.

    Returns
    -------
    None
        Nothing; delegates artifact creation to the processing helpers.

    Raises
    ------
    ValueError
        If selections, CAM/reference artifacts, or experiment geometry are inconsistent.
    FileNotFoundError
        If a checkpoint or required manifest/artifact cannot be found.
    ImportError
        If Captum is required for sliding attribution but is unavailable.
    """
    context = prepare_pipeline_context(args)
    all_summaries: List[Dict[str, Any]] = []

    # Each model is loaded exactly once. Per-image and per-mode helpers handle the
    # nested scientific cases without exposing that implementation detail here.
    for model_name, checkpoint_name in context.model_specs:
        all_summaries.extend(
            process_model(args, context, model_name, checkpoint_name)
        )

    if all_summaries:
        write_model_comparisons(context.output, all_summaries, args.seed)

    print(
        "\nOcclusion pipeline complete: "
        f"{len(all_summaries)} unique model/input/image cases."
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Validate arguments and seed all random controls before executing the pipeline.

    Steps
    -----
    1. Parse and validate the command line.
    2. Seed PyTorch and NumPy random streams for reproducible analysis.
    3. Dispatch to the occlusion pipeline.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Optional explicit CLI tokens; defaults to process arguments.

    Returns
    -------
    None
        Nothing; performs the requested command-line workflow.
    """
    args = parse_arguments(argv)
    validate_args(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_pipeline(args)


if __name__ == "__main__":
    main()
