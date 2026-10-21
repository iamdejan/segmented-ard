"""Generate global class-logit Grad-CAM++ explanations for selected cases.

The program consumes the ``selected_cases.csv`` manifest written by
``explain_choose.py`` and explains the mean logit of one semantic class over
*all* output pixels, including pixels not labeled as that class. Inputs may be
clean, PGD-perturbed using ``evaluate_robustness.py``, or both. The explained
class defaults to road and can be changed with ``--target-class``. Ground-truth
masks remain necessary for diagnostic figures and PGD, but never restrict the
Grad-CAM++ score. Unusable Grad-CAM++ maps raise errors without a fallback.
"""

import argparse
import hashlib
import re
from pathlib import Path as FilePath
from typing import Callable, Dict, List, Optional, Sequence, Tuple, cast

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Float, Int64, jaxtyped
from pytorch_grad_cam import GradCAMPlusPlus
from torch import Tensor, nn

from evaluate_performance import (
    BDDSegmentationDataset,
    CLASS_COLORS,
    CLASS_NAMES,
    Configuration,
    load_trained_model,
)
from evaluate_robustness import pgd_attack


BatchImage = Float[Tensor, "1 3 h w"]
BatchMask = Float[Tensor, "1 c h w"]
Logits = Float[Tensor, "1 c h w"]
SingleLogits = Float[Tensor, "c h w"]
Scalar = Float[Tensor, ""]
ClassMap = Int64[Tensor, "h w"]
NumpyImage = Float[np.ndarray, "h w 3"]
NumpyCamBatch = Float[np.ndarray, "b h w"]
NumpyHeatmap = Float[np.ndarray, "h w"]
NumpyClassMap = Int64[np.ndarray, "h w"]


def parse_checkpoint_spec(specification: str) -> Tuple[str, str]:
    """Parse a checkpoint CLI value into a model name and filesystem path.

    Steps
    -----
    1. Split ``NAME=PATH`` at the first equals sign when a name is supplied.
    2. Derive the model name from the file stem for a plain checkpoint path.
    3. Reject empty names or paths before checkpoint loading begins.

    Parameters
    ----------
    specification : str
        Checkpoint specification in ``PATH`` or ``NAME=PATH`` form.

    Returns
    -------
    Tuple[str, str]
        Human-readable model name and checkpoint path.

    Raises
    ------
    ValueError
        If the model name or checkpoint path is empty.
    """
    if "=" in specification:
        model_name, checkpoint_path = specification.split("=", maxsplit=1)
    else:
        checkpoint_path = specification
        model_name = FilePath(checkpoint_path).stem

    model_name = model_name.strip()
    checkpoint_path = checkpoint_path.strip()
    if not model_name or not checkpoint_path:
        raise ValueError(
            "Checkpoint specifications must be PATH or NAME=PATH with non-empty values."
        )
    return model_name, checkpoint_path


def resolve_project_path(path_value: str, project_root: FilePath) -> FilePath:
    """Resolve a manifest path relative to the project root when necessary.

    ``explain_choose.py`` stores the original project-relative image and mask
    paths. Resolving them at explanation time allows this script to be launched
    from another directory without silently reading a different dataset.

    Steps
    -----
    1. Expand the user home directory in the supplied path.
    2. Return absolute paths unchanged.
    3. Prefix relative paths with ``project_root``.

    Parameters
    ----------
    path_value : str
        Absolute or project-relative filesystem path.
    project_root : FilePath
        Repository root used to resolve relative paths.

    Returns
    -------
    FilePath
        Resolved filesystem path.
    """
    candidate = FilePath(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def sanitise_filename(value: str) -> str:
    """Convert arbitrary case metadata into a filesystem-safe name component.

    Steps
    -----
    1. Replace non-alphanumeric separators with underscores.
    2. Collapse repeated underscores for readable output filenames.
    3. Use ``case`` when the result would otherwise be empty.

    Parameters
    ----------
    value : str
        Text value to convert into a filename component.

    Returns
    -------
    str
        Sanitised filename component.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "case"


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying network when a checkpoint is wrapped in DataParallel.

    Steps
    -----
    1. Detect ``nn.DataParallel`` wrappers saved by multi-GPU training.
    2. Return ``model.module`` for wrapped networks or the original model otherwise.

    Parameters
    ----------
    model : nn.Module
        Loaded segmentation checkpoint.

    Returns
    -------
    nn.Module
        Unwrapped model used for layer discovery.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


def resolve_module_path(model: nn.Module, module_path: str) -> nn.Module:
    """Resolve a dotted module path such as ``decoder.blocks.4.conv2.0``.

    Numeric path components index ``Sequential``/``ModuleList`` containers while
    textual components access named attributes. Supporting both forms makes the
    explicit target-layer option work with segmentation-models-pytorch U-Nets.

    Steps
    -----
    1. Remove an optional leading ``module.`` from DataParallel-style paths.
    2. Walk each dotted component through attribute lookup or integer indexing.
    3. Verify that the resolved object is an ``nn.Module``.

    Parameters
    ----------
    model : nn.Module
        Unwrapped segmentation model.
    module_path : str
        Dotted path identifying the requested target layer.

    Returns
    -------
    nn.Module
        Resolved model module.

    Raises
    ------
    ValueError
        If the module path is empty.
    AttributeError
        If a named path component does not exist.
    IndexError
        If a numeric container index is out of range.
    TypeError
        If a numeric component targets a non-indexable module or the final
        resolved object is not an ``nn.Module``.
    """
    stripped = module_path.strip()
    if stripped.startswith("module."):
        stripped = stripped[len("module.") :]
    if not stripped:
        raise ValueError("--target-layer must not be empty.")

    current: object = model
    for component in stripped.split("."):
        if component.isdigit():
            if not isinstance(current, (nn.Sequential, nn.ModuleList)):
                raise TypeError(
                    f"Cannot index component '{component}' inside "
                    f"{type(current).__name__}."
                )
            current = current[int(component)]
        else:
            if not hasattr(current, component):
                raise AttributeError(
                    f"Module path '{module_path}' has no component '{component}'."
                )
            current = getattr(current, component)

    if not isinstance(current, nn.Module):
        raise TypeError(f"Resolved target '{module_path}' is not an nn.Module.")
    return current


def find_last_convolution(module: nn.Module, prefix: str) -> Tuple[str, nn.Conv2d]:
    """Find the last convolution in one model subtree.

    A late convolution is preferred because Grad-CAM needs spatial activations
    that are close enough to the segmentation output to encode class-specific
    evidence. For U-Net this naturally selects a late decoder convolution when
    ``prefix`` points to the decoder.

    Steps
    -----
    1. Traverse all named descendants of the requested subtree.
    2. Retain every ``nn.Conv2d`` candidate in traversal order.
    3. Return the final candidate and its fully qualified name.

    Parameters
    ----------
    module : nn.Module
        Model subtree searched for convolutional layers.
    prefix : str
        Human-readable prefix prepended to the returned module name.

    Returns
    -------
    Tuple[str, nn.Conv2d]
        Fully qualified layer name and final convolution module.

    Raises
    ------
    ValueError
        If no convolutional layer exists in the requested subtree.
    """
    candidates: List[Tuple[str, nn.Conv2d]] = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            qualified_name = prefix if not name else f"{prefix}.{name}"
            candidates.append((qualified_name, child))

    if not candidates:
        raise ValueError(f"No nn.Conv2d layer was found under '{prefix}'.")
    return candidates[-1]


def choose_target_layer(
    model: nn.Module,
    layer_scope: str,
    target_layer_path: Optional[str] = None,
) -> Tuple[str, nn.Module]:
    """Choose the feature-producing module used to build Grad-CAM activations.

    Decoder features are the default because semantic segmentation requires
    spatially detailed explanations. For segmentation-models-pytorch U-Nets the
    decoder module itself is targeted so its post-activation output is captured,
    avoiding hooks immediately before in-place ReLU operations. Encoder mode
    remains available for higher-level, lower-resolution evidence, while ``auto``
    prefers the decoder and falls back to an encoder/model convolution.

    Steps
    -----
    1. Resolve an explicit dotted target-layer path when provided.
    2. For decoder scope, use the decoder module output directly when available.
    3. Otherwise search encoder or full-model subtrees for their last convolution.

    Parameters
    ----------
    model : nn.Module
        Loaded segmentation model, possibly wrapped by DataParallel.
    layer_scope : str
        One of ``decoder``, ``encoder``, or ``auto``.
    target_layer_path : Optional[str], optional
        Explicit dotted module path. Defaults to ``None``.

    Returns
    -------
    Tuple[str, nn.Module]
        Layer name and module used for explanation generation.

    Raises
    ------
    ValueError
        If ``layer_scope`` is invalid or no suitable feature module can be found.
    AttributeError
        If an explicit target path cannot be resolved.
    TypeError
        If an explicit target path does not resolve to a module.
    """
    raw_model = unwrap_model(model)
    if target_layer_path is not None:
        return target_layer_path, resolve_module_path(raw_model, target_layer_path)

    if layer_scope not in {"decoder", "encoder", "auto"}:
        raise ValueError("layer_scope must be one of: decoder, encoder, auto.")

    scopes = [layer_scope] if layer_scope != "auto" else ["decoder", "encoder", "model"]
    errors: List[str] = []
    for scope in scopes:
        if scope == "decoder":
            if not hasattr(raw_model, "decoder"):
                errors.append("model has no 'decoder' subtree")
                continue
            decoder = raw_model.decoder
            if not isinstance(decoder, nn.Module):
                errors.append("'decoder' is not an nn.Module")
                continue
            return "decoder", decoder

        if scope == "model":
            subtree = raw_model
            prefix = "model"
        else:
            if not hasattr(raw_model, scope):
                errors.append(f"model has no '{scope}' subtree")
                continue
            subtree = getattr(raw_model, scope)
            if not isinstance(subtree, nn.Module):
                errors.append(f"'{scope}' is not an nn.Module")
                continue
            prefix = scope

        try:
            return find_last_convolution(subtree, prefix)
        except ValueError as error:
            errors.append(str(error))

    joined = "; ".join(errors)
    raise ValueError(f"Unable to find a Grad-CAM feature layer: {joined}")


class MeanClassSegmentationTarget:
    """Reduce segmentation logits to a global mean for one semantic class.

    ``pytorch-grad-cam`` expects a callable returning a scalar for each image.
    This target treats every output pixel equally, regardless of ground truth,
    predictions, or case-selection category.

    Steps
    -----
    1. Validate and store the requested semantic class id.
    2. Select that class channel from the model's per-image logits.
    3. Average the channel across its entire spatial extent for Grad-CAM++.

    Parameters
    ----------
    target_class : int
        Zero-based model output channel to explain, typically road (0).

    Raises
    ------
    ValueError
        If ``target_class`` is negative.
    """

    def __init__(self, target_class: int) -> None:
        """Store the requested global segmentation class target.

        Steps
        -----
        1. Reject negative class ids before model inference begins.
        2. Retain the class id for evaluation of every input variant.

        Parameters
        ----------
        target_class : int
            Zero-based semantic class channel whose spatial logits are averaged.

        Returns
        -------
        None
            Initialises the callable target for Grad-CAM++.

        Raises
        ------
        ValueError
            If ``target_class`` is negative.
        """
        if target_class < 0:
            raise ValueError("Grad-CAM++ target class must be non-negative.")

        # The target is independent of the ground-truth map, allowing the same
        # score definition for clean and PGD-robust variants of an image.
        self.target_class = target_class

    @jaxtyped(typechecker=beartype)
    def __call__(self, model_output: SingleLogits) -> Scalar:
        """Calculate the mean target-class logit over all output pixels.

        Steps
        -----
        1. Check that the selected class channel exists in the model output.
        2. Average its raw logits across height and width without masking.

        Parameters
        ----------
        model_output : SingleLogits
            One image's unnormalised segmentation logits, shaped ``(C, H, W)``.

        Returns
        -------
        Scalar
            Differentiable scalar mean of the selected class's ``H * W`` logits.

        Raises
        ------
        ValueError
            If the requested class id is outside the model's output channels.
        """
        if self.target_class >= model_output.shape[0]:
            raise ValueError(
                f"Target class {self.target_class} is outside logits range "
                f"[0, {model_output.shape[0] - 1}]."
            )

        # An unmasked spatial mean explains the class response everywhere in
        # the image, including locations the model assigns to other classes.
        score = model_output[self.target_class].mean()
        return cast(Scalar, score)


@jaxtyped(typechecker=beartype)
def run_cam_algorithm(
    cam: Callable[..., NumpyCamBatch],
    image: BatchImage,
    target: MeanClassSegmentationTarget,
) -> NumpyHeatmap:
    """Run one ``pytorch-grad-cam`` algorithm and validate its output shape.

    The library owns activation hooks, gradient collection, CAM weighting,
    resizing to input resolution, and normalisation. This helper only converts
    the single-image result into the repository's typed NumPy representation.

    Steps
    -----
    1. Invoke the configured library CAM with the custom segmentation target.
    2. Select the sole batch item and require a two-dimensional heatmap.
    3. Return float32 data without copying when the library already provides it.

    Parameters
    ----------
    cam : Callable[..., NumpyCamBatch]
        Configured ``pytorch-grad-cam`` algorithm bound to the model/layer.
    image : BatchImage
        Input image tensor with shape ``(1, 3, H, W)``.
    target : MeanClassSegmentationTarget
        Callable whose scalar score is the mean class logit over all spatial pixels.

    Returns
    -------
    NumpyHeatmap
        Library-normalised heatmap of shape ``(H, W)``.

    Raises
    ------
    RuntimeError
        If the library returns an unexpected batch or spatial shape.
    """
    # The library performs its own differentiable forward/backward pass because
    # its hooks must observe the exact activations used to build the CAM.
    grayscale_cams = cam(input_tensor=image, targets=[target])
    if grayscale_cams.ndim != 3 or grayscale_cams.shape[0] != 1:
        raise RuntimeError(
            "pytorch-grad-cam returned an unexpected CAM batch shape: "
            f"{grayscale_cams.shape}."
        )

    heatmap = np.asarray(grayscale_cams[0], dtype=np.float32)
    if heatmap.ndim != 2:
        raise RuntimeError(
            "pytorch-grad-cam returned an unexpected CAM spatial shape: "
            f"{heatmap.shape}."
        )
    return cast(NumpyHeatmap, heatmap)


@jaxtyped(typechecker=beartype)
def cam_is_usable(heatmap: NumpyHeatmap, epsilon: float) -> bool:
    """Return whether a normalised library CAM is finite and non-degenerate.

    This check prevents invalid or effectively empty Grad-CAM++ maps from being
    saved as successful explanations; the caller raises if it fails.

    Steps
    -----
    1. Reject CAMs containing NaN or infinite values.
    2. Require at least one value greater than the degeneracy threshold.

    Parameters
    ----------
    heatmap : NumpyHeatmap
        Normalised CAM of shape ``(H, W)``.
    epsilon : float
        Positive threshold used to identify effectively empty heatmaps.

    Returns
    -------
    bool
        ``True`` when the CAM is finite and contains meaningful positive evidence.

    Raises
    ------
    ValueError
        If ``epsilon`` is not positive.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than 0.")
    return bool(np.isfinite(heatmap).all() and float(heatmap.max()) > epsilon)


@jaxtyped(typechecker=beartype)
def generate_cam(
    model: nn.Module,
    target_layer: nn.Module,
    image: BatchImage,
    target: MeanClassSegmentationTarget,
    epsilon: float,
) -> NumpyHeatmap:
    """Generate a valid Grad-CAM++ heatmap using ``pytorch-grad-cam``.

    The external library computes all activation and gradient weights. A failed
    computation or unusable normalised CAM is an error; no other method is run.

    Steps
    -----
    1. Validate the degeneracy threshold.
    2. Run Grad-CAM++ on the selected layer and global mean class-logit target.
    3. Reject non-finite, zero, or effectively empty CAMs instead of falling back.

    Parameters
    ----------
    model : nn.Module
        Loaded semantic-segmentation network to explain.
    target_layer : nn.Module
        Feature-producing module observed by the Grad-CAM++ hooks.
    image : BatchImage
        Model input with shape ``(1, 3, H, W)``.
    target : MeanClassSegmentationTarget
        Callable averaging the selected class logits over every output pixel.
    epsilon : float
        Positive threshold below which the normalised CAM is degenerate.

    Returns
    -------
    NumpyHeatmap
        Finite, non-degenerate Grad-CAM++ heatmap of shape ``(H, W)``.

    Raises
    ------
    ValueError
        If ``epsilon`` is not positive.
    RuntimeError
        If Grad-CAM++ returns an unusable heatmap or an invalid CAM shape.

    Notes
    -----
    Other library errors propagate; no Grad-CAM fallback is attempted.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than 0.")

    # Restrict generation to Grad-CAM++ so invalid maps are surfaced, not
    # quietly replaced by another explanation algorithm.
    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        heatmap = run_cam_algorithm(cam=cam, image=image, target=target)
    if not cam_is_usable(heatmap, epsilon):
        raise RuntimeError(
            "Grad-CAM++ produced a non-finite or degenerate heatmap "
            f"(no finite value greater than {epsilon})."
        )
    return heatmap


@jaxtyped(typechecker=beartype)
def tensor_image_to_numpy(image: BatchImage) -> NumpyImage:
    """Convert one model input batch into an HWC float image for plotting.

    Steps
    -----
    1. Remove the singleton batch dimension.
    2. Move the tensor to CPU and transpose CHW to HWC.
    3. Clamp values into the valid display range ``[0, 1]``.

    Parameters
    ----------
    image : BatchImage
        Input tensor with shape ``(1, 3, H, W)``.

    Returns
    -------
    NumpyImage
        RGB image of shape ``(H, W, 3)`` with float32 values.
    """
    array = image[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    return cast(NumpyImage, np.clip(array, 0.0, 1.0))


@jaxtyped(typechecker=beartype)
def tensor_class_map_to_numpy(class_map: ClassMap) -> NumpyClassMap:
    """Convert a torch class-index map to a NumPy int64 array.

    Steps
    -----
    1. Detach the map, move it to CPU, and preserve integer class ids.

    Parameters
    ----------
    class_map : ClassMap
        Class-index tensor of shape ``(H, W)``.

    Returns
    -------
    NumpyClassMap
        CPU NumPy class map with shape ``(H, W)``.
    """
    array = class_map.detach().cpu().numpy().astype(np.int64, copy=False)
    return cast(NumpyClassMap, array)


@jaxtyped(typechecker=beartype)
def colourise_class_map(class_map: NumpyClassMap) -> NumpyImage:
    """Convert class ids to the repository's BDD100k RGB palette.

    Steps
    -----
    1. Clip invalid ids to the final ``Unknown`` palette entry.
    2. Index the shared ``CLASS_COLORS`` palette.
    3. Scale RGB values from ``0..255`` to display-ready ``0..1`` floats.

    Parameters
    ----------
    class_map : NumpyClassMap
        Semantic class-index map of shape ``(H, W)``.

    Returns
    -------
    NumpyImage
        RGB float image of shape ``(H, W, 3)``.
    """
    unknown_class = len(CLASS_COLORS) - 1
    safe_ids = np.where(
        (class_map >= 0) & (class_map < len(CLASS_COLORS)),
        class_map,
        unknown_class,
    )
    colour = CLASS_COLORS[safe_ids].astype(np.float32) / 255.0
    return cast(NumpyImage, colour)


@jaxtyped(typechecker=beartype)
def save_case_visualisations(
    image: NumpyImage,
    target_map: NumpyClassMap,
    prediction_map: NumpyClassMap,
    heatmap: NumpyHeatmap,
    heatmap_path: FilePath,
    overlay_path: FilePath,
    panel_path: FilePath,
    title: str,
    colormap: str,
    overlay_alpha: float,
) -> None:
    """Save raw heatmap, image overlay, and diagnostic four-panel figure.

    The panel keeps the input, GT, prediction, and Grad-CAM evidence together so
    a heatmap can be interpreted alongside the predicted segmentation. No
    spatial target contour is needed because the score averages the selected
    class's logits across every pixel of the model output.

    Steps
    -----
    1. Save the normalised heatmap as a standalone PNG.
    2. Save a transparent heatmap overlay on the resized RGB image.
    3. Save a four-panel input/GT/prediction/explanation diagnostic figure.

    Parameters
    ----------
    image : NumpyImage
        RGB input image of shape ``(H, W, 3)`` in ``[0, 1]``.
    target_map : NumpyClassMap
        Ground-truth semantic class map of shape ``(H, W)``.
    prediction_map : NumpyClassMap
        Predicted semantic class map of shape ``(H, W)``.
    heatmap : NumpyHeatmap
        Normalised Grad-CAM++ heatmap of shape ``(H, W)``.
    heatmap_path : FilePath
        Destination for the standalone heatmap PNG.
    overlay_path : FilePath
        Destination for the image-plus-heatmap PNG.
    panel_path : FilePath
        Destination for the diagnostic panel PNG.
    title : str
        Case and target description written on the panel.
    colormap : str
        Matplotlib colormap for heatmap rendering.
    overlay_alpha : float
        Heatmap opacity in overlay figures.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``overlay_alpha`` is outside ``[0, 1]``.
    """
    if overlay_alpha < 0.0 or overlay_alpha > 1.0:
        raise ValueError("overlay_alpha must be inside [0, 1].")

    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(heatmap_path, heatmap, cmap=colormap, vmin=0.0, vmax=1.0)

    overlay_figure, overlay_axis = plt.subplots(figsize=(10, 6))
    overlay_axis.imshow(image)
    overlay_axis.imshow(
        heatmap,
        cmap=colormap,
        alpha=overlay_alpha,
        vmin=0.0,
        vmax=1.0,
    )
    overlay_axis.set_title(title)
    overlay_axis.set_axis_off()
    overlay_figure.tight_layout()
    overlay_figure.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(overlay_figure)

    ground_truth_colour = colourise_class_map(target_map)
    prediction_colour = colourise_class_map(prediction_map)

    panel_figure, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(image)
    axes[0].set_title("Input")

    axes[1].imshow(image)
    axes[1].imshow(ground_truth_colour, alpha=0.5)
    axes[1].set_title("Ground truth")

    axes[2].imshow(image)
    axes[2].imshow(prediction_colour, alpha=0.5)
    axes[2].set_title("Prediction")

    axes[3].imshow(image)
    axes[3].imshow(
        heatmap,
        cmap=colormap,
        alpha=overlay_alpha,
        vmin=0.0,
        vmax=1.0,
    )
    axes[3].set_title("Grad-CAM++ explanation")

    for axis in axes:
        axis.set_axis_off()

    panel_figure.suptitle(title)
    panel_figure.tight_layout()
    panel_figure.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(panel_figure)


def optional_class_id(value: object) -> Optional[int]:
    """Convert a CSV class-id value into ``Optional[int]`` safely.

    Steps
    -----
    1. Return ``None`` for pandas missing values.
    2. Convert present numeric-like values through text to a stable integer id.

    Parameters
    ----------
    value : object
        Pandas cell containing an integer-like value or ``NaN``.

    Returns
    -------
    Optional[int]
        Integer class id, or ``None`` for missing values.
    """
    if pd.isna(value):
        return None
    return int(float(str(value)))


def resolve_target_class(specification: str) -> int:
    """Resolve the CLI target class name or numeric id to one model channel.

    Steps
    -----
    1. Interpret numeric values directly as semantic class ids.
    2. Resolve class names case-insensitively against ``CLASS_NAMES``.
    3. Validate the id against the configured class-name list.

    Parameters
    ----------
    specification : str
        Requested target class name (e.g. ``road``) or decimal class id.

    Returns
    -------
    int
        Valid semantic class id to explain in every selected case.

    Raises
    ------
    ValueError
        If the name is unknown or the id is outside the configured classes.
    """
    stripped = specification.strip()
    try:
        class_id = int(stripped)
    except ValueError:
        class_lookup = {str(name).casefold(): i for i, name in enumerate(CLASS_NAMES)}
        if stripped.casefold() not in class_lookup:
            raise ValueError(
                f"Unknown target class '{specification}'. "
                f"Valid names: {', '.join(CLASS_NAMES)}."
            ) from None
        class_id = class_lookup[stripped.casefold()]

    if not 0 <= class_id < len(CLASS_NAMES):
        raise ValueError(
            f"Target class id {class_id} is outside [0, {len(CLASS_NAMES) - 1}]."
        )
    return class_id


def expand_and_filter_cases(
    selected_cases: pd.DataFrame,
    checkpoint_names: Sequence[str],
    max_cases: Optional[int] = None,
) -> pd.DataFrame:
    """Expand cross-model selected cases and cap the number to explain.

    ``all_models`` rows from ``explain_choose.py`` are duplicated for each
    supplied checkpoint. Per-model rows are retained only if that checkpoint
    was supplied, while the original selection category and rank are preserved.

    Steps
    -----
    1. Require columns needed to identify, name, and load selected cases.
    2. Keep model-specific rows for the supplied checkpoints only.
    3. Expand ``all_models`` rows once for every supplied checkpoint.
    4. Preserve input ordering and optionally cap the rows with ``max_cases``.

    Parameters
    ----------
    selected_cases : pd.DataFrame
        Source ``selected_cases.csv`` rows, including model, class, and image paths.
    checkpoint_names : Sequence[str]
        Names from ``--checkpoint`` that determine which models receive cases.
    max_cases : Optional[int]
        Maximum number of expanded cases, or ``None`` to keep all cases.

    Returns
    -------
    pd.DataFrame
        Expanded case table, preserving source columns and adding
        ``source_row_index`` and ``source_model_name`` provenance columns.

    Raises
    ------
    ValueError
        If required columns are missing or ``max_cases`` is less than one.
    """
    required_columns = {
        "category", "model_name", "class_id", "image_index",
        "image_path", "mask_path", "rank",
    }
    missing = sorted(required_columns - set(selected_cases.columns))
    if missing:
        raise ValueError(
            "selected_cases.csv is missing required columns: " + ", ".join(missing)
        )
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be at least 1 when provided.")

    # Preserve column headers even when no rows match, so the exported CSV
    # still shows which case fields the selection stage produced.
    expanded_columns = list(selected_cases.columns)
    for column in ("source_row_index", "source_model_name"):
        if column not in expanded_columns:
            expanded_columns.append(column)

    checkpoint_set = set(checkpoint_names)
    expanded_rows: List[Dict[str, object]] = []
    for source_row_number, (_, row) in enumerate(selected_cases.iterrows()):
        source_model_name = str(row["model_name"])
        if source_model_name == "all_models":
            targets = checkpoint_names
        elif source_model_name in checkpoint_set:
            targets = [source_model_name]
        else:
            continue

        for target_model_name in targets:
            record = dict(row)
            record["source_row_index"] = source_row_number
            record["source_model_name"] = source_model_name
            record["model_name"] = target_model_name
            expanded_rows.append(record)

    frame = pd.DataFrame(expanded_rows, columns=expanded_columns).reset_index(drop=True)
    if max_cases is not None:
        frame = frame.head(max_cases).copy()
    return frame


def build_single_sample_dataset(
    image_path: FilePath,
    mask_path: FilePath,
) -> BDDSegmentationDataset:
    """Create a one-image dataset using the repository's inference preprocessing.

    Reusing ``BDDSegmentationDataset`` prevents Grad-CAM inputs from drifting
    from the evaluation pipeline: image scaling, RGB-label decoding, resize, and
    tensor conversion therefore match the model's ordinary test-time domain.

    Steps
    -----
    1. Validate that the selected image and mask files exist.
    2. Build a one-row dataframe with their resolved paths.
    3. Apply the same resize/ToTensor inference transform used by evaluation.

    Parameters
    ----------
    image_path : FilePath
        Resolved source RGB image path.
    mask_path : FilePath
        Resolved source semantic-mask path.

    Returns
    -------
    BDDSegmentationDataset
        One-sample dataset ready for direct indexing.

    Raises
    ------
    FileNotFoundError
        If either selected source file does not exist.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Selected image does not exist: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Selected mask does not exist: {mask_path}")

    frame = pd.DataFrame(
        {
            "image_paths": [str(image_path)],
            "mask_paths": [str(mask_path)],
        }
    )
    transform = A.Compose(
        [
            A.Resize(
                height=Configuration.IMAGE_HEIGHT,
                width=Configuration.IMAGE_WIDTH,
            ),
            ToTensorV2(),
        ]
    )
    return BDDSegmentationDataset(frame, transform=transform)


@jaxtyped(typechecker=beartype)
def load_single_sample(
    image_path: FilePath,
    mask_path: FilePath,
    device: torch.device,
) -> Tuple[BatchImage, ClassMap]:
    """Load one selected image/mask pair into the model's inference domain.

    Steps
    -----
    1. Build the repository's one-sample evaluation dataset.
    2. Add a batch dimension and move the RGB tensor to the target device.
    3. Collapse the one-hot semantic mask to a class-index map on the same device.

    Parameters
    ----------
    image_path : FilePath
        Resolved selected RGB image path.
    mask_path : FilePath
        Resolved selected semantic-mask path.
    device : torch.device
        Device used by the loaded model.

    Returns
    -------
    Tuple[BatchImage, ClassMap]
        Batched image tensor and ground-truth class-index map.
    """
    dataset = build_single_sample_dataset(image_path, mask_path)
    image, mask = dataset[0]
    batch_image = image.unsqueeze(0).to(device)
    class_map = mask.argmax(dim=0).to(device=device, dtype=torch.int64)
    return cast(BatchImage, batch_image), cast(ClassMap, class_map)


@jaxtyped(typechecker=beartype)
def generate_robust_input(
    model: nn.Module,
    image: BatchImage,
    target: ClassMap,
) -> BatchImage:
    """Perturb one dataset image with the repository's PGD robustness attack.

    The ground-truth class map is converted back to the one-hot batch layout
    expected by :func:`evaluate_robustness.pgd_attack`. The attack function is
    called without overriding its attack parameters so Grad-CAM robustness uses
    the same 10-step PGD defaults as the robustness-evaluation pipeline.

    Steps
    -----
    1. One-hot encode the unchanged ground-truth class map into a batch mask.
    2. Run the shared untargeted PGD implementation on the clean input image.
    3. Detach the adversarial image before subsequent inference/CAM operations.

    Parameters
    ----------
    model : nn.Module
        Loaded semantic-segmentation checkpoint to attack.
    image : BatchImage
        Clean dataset image with shape ``(1, 3, H, W)`` and values in ``[0, 1]``.
    target : ClassMap
        Unchanged ground-truth class-index map with shape ``(H, W)``.

    Returns
    -------
    BatchImage
        PGD-perturbed image with the same shape and valid value range as ``image``.
    """
    # PGD maximizes cross-entropy against a one-hot segmentation target, so rebuild
    # exactly the batched target representation consumed by evaluate_robustness.py.
    target_one_hot = F.one_hot(
        target,
        num_classes=Configuration.NUM_CLASSES,
    ).permute(2, 0, 1).float().unsqueeze(0)
    batch_target = cast(BatchMask, target_one_hot.to(device=image.device))

    # Reusing the shared attack implementation keeps epsilon, step size, iteration
    # count, clamping, and ignored-class handling identical to robustness evaluation.
    robust_image = pgd_attack(
        model=model,
        images=image,
        targets=batch_target,
    )
    return cast(BatchImage, robust_image.detach())


def case_output_paths(
    output_dir: FilePath,
    row: pd.Series,
    target_class: int,
    input_mode: str,
) -> Tuple[FilePath, FilePath, FilePath]:
    """Build deterministic output paths for one explanation case.

    Steps
    -----
    1. Insert the concrete input mode below the model directory.
    2. Keep clean and adversarial outputs separated to avoid collisions.
    3. Encode source-row/rank/image/selected-class/target-class in the filename.
    4. Return separate paths for heatmap, overlay, and diagnostic panel.

    Parameters
    ----------
    output_dir : FilePath
        Root directory receiving Grad-CAM artifacts.
    row : pd.Series
        Expanded selected-case row.
    target_class : int
        Actual model class explained by Grad-CAM.
    input_mode : str
        Concrete input variant being explained: ``clean`` or ``robust``.

    Returns
    -------
    Tuple[FilePath, FilePath, FilePath]
        Heatmap PNG path, overlay PNG path, and panel PNG path.

    Raises
    ------
    ValueError
        If ``input_mode`` is not ``clean`` or ``robust``.
    """
    if input_mode not in {"clean", "robust"}:
        raise ValueError("input_mode must be either 'clean' or 'robust'.")

    model_name = sanitise_filename(str(row["model_name"]))
    category = sanitise_filename(str(row["category"]))
    rank = int(float(row["rank"]))
    image_index = int(float(row["image_index"]))
    source_row_index = int(row["source_row_index"])
    selected_class = optional_class_id(row["class_id"])
    selected_token = "none" if selected_class is None else str(selected_class)

    basename = (
        f"row{source_row_index:04d}_rank{rank:02d}_img{image_index:04d}_"
        f"selected{selected_token}_target{target_class}"
    )
    # Namespace every artifact by its concrete input mode so --input-mode all
    # produces clean and robust trees side by side without overwriting either.
    destination = output_dir / model_name / input_mode / category
    return (
        destination / f"{basename}_heatmap.png",
        destination / f"{basename}_overlay.png",
        destination / f"{basename}_panel.png",
    )


def build_case_title(
    row: pd.Series,
    selected_class: Optional[int],
    target_class: int,
    input_mode: str,
) -> str:
    """Build a compact, self-describing title for one explanation figure.

    Steps
    -----
    1. Resolve selected and actual target class ids into readable names.
    2. Combine the model, input variant, selection metadata, globally averaged
       class-logit target, and fixed Grad-CAM++ method into a title.

    Parameters
    ----------
    row : pd.Series
        Expanded selected-case metadata.
    selected_class : Optional[int]
        Class associated with the original selection row.
    target_class : int
        Actual class whose logit was differentiated.
    input_mode : str
        Concrete input variant being explained: ``clean`` or ``robust``.

    Returns
    -------
    str
        Human-readable diagnostic title.
    """
    selected_name = "none"
    if selected_class is not None and 0 <= selected_class < len(CLASS_NAMES):
        selected_name = CLASS_NAMES[selected_class]
    target_name = (
        CLASS_NAMES[target_class]
        if 0 <= target_class < len(CLASS_NAMES)
        else f"class_{target_class}"
    )
    return (
        f"{row['model_name']} | input={input_mode} | {row['category']} | "
        f"selected={selected_name} | "
        f"target={target_name} | score=global_mean_logit | Grad-CAM++"
    )


def explain_cases_for_model(
    model: nn.Module,
    model_name: str,
    model_cases: pd.DataFrame,
    project_root: FilePath,
    output_dir: FilePath,
    device: torch.device,
    layer_scope: str,
    target_class: int,
    colormap: str,
    overlay_alpha: float,
    epsilon: float,
    input_mode: str,
    target_layer_path: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Generate Grad-CAM++ explanations for every selected case of one model.

    Every output pixel contributes to the mean target-class logit, regardless
    of its ground-truth or predicted label. Ground-truth masks are retained for
    visual diagnostics and the shared PGD attack in ``evaluate_robustness.py``.
    Grad-CAM++ runs once per image and input variant; individual selection rows
    retain their source metadata and separate figure paths.

    Steps
    -----
    1. Discover the model target layer used by the CAM library.
    2. Group selected rows by image/mask pair and load the clean dataset sample once.
    3. Build the requested clean/robust input variants, using shared PGD for robust.
    4. Obtain each variant's diagnostic prediction and global class-logit score.
    5. Generate each variant's Grad-CAM++ map once, save case-specific figures,
       and return the manifest rows.

    Parameters
    ----------
    model : nn.Module
        Loaded semantic-segmentation checkpoint.
    model_name : str
        Human-readable checkpoint label.
    model_cases : pd.DataFrame
        Filtered selected rows belonging to this model.
    project_root : FilePath
        Repository root for resolving relative image/mask paths.
    output_dir : FilePath
        Root directory receiving Grad-CAM outputs.
    device : torch.device
        Device used for model inference and CAM computation.
    layer_scope : str
        Automatic layer scope: ``decoder``, ``encoder``, or ``auto``.
    target_class : int
        Fixed semantic class channel whose logit is averaged across the entire
        model output map, including pixels of other classes.
    colormap : str
        Matplotlib heatmap colormap.
    overlay_alpha : float
        Heatmap opacity for image overlays.
    epsilon : float
        Threshold used to detect degenerate library CAM outputs.
    input_mode : str
        Requested inputs: ``clean``, ``robust``, or ``all``.
    target_layer_path : Optional[str], optional
        Explicit dotted target-layer path, or ``None`` to resolve the layer
        according to ``layer_scope``.

    Returns
    -------
    List[Dict[str, object]]
        Explanation-manifest rows describing all generated outputs.

    Raises
    ------
    RuntimeError
        If Grad-CAM++ fails or returns an unusable heatmap.
    TypeError
        If the segmentation model does not return a tensor.
    ValueError
        If ``input_mode`` is invalid or the requested target class is outside
        the model's output-channel range.
    FileNotFoundError
        If selected source files are missing.
    """
    if input_mode not in {"clean", "robust", "all"}:
        raise ValueError("input_mode must be one of: clean, robust, all.")

    model.eval()
    layer_name, target_layer = choose_target_layer(
        model,
        layer_scope=layer_scope,
        target_layer_path=target_layer_path,
    )
    print(f"  Target layer: {layer_name}")

    # The global target depends only on the requested class, so create it once
    # and reuse its identical score definition for every image and input mode.
    cam_target = MeanClassSegmentationTarget(target_class=target_class)
    manifest_rows: List[Dict[str, object]] = []
    grouping_columns = ["image_index", "image_path", "mask_path"]
    grouped = model_cases.groupby(grouping_columns, sort=False, dropna=False)

    for (image_index_value, image_path_value, mask_path_value), case_group in grouped:
        image_index = int(float(image_index_value))
        image_path = resolve_project_path(str(image_path_value), project_root)
        mask_path = resolve_project_path(str(mask_path_value), project_root)

        clean_image, target = load_single_sample(image_path, mask_path, device)

        # Ground truth is still needed for the diagnostic panel and PGD attack,
        # but does not define or restrict the global Grad-CAM++ target score.
        target_numpy = tensor_class_map_to_numpy(target)

        # Build robust input only when requested because PGD requires ten backward
        # passes and would otherwise add substantial cost to clean-only explanations.
        input_variants: List[Tuple[str, BatchImage]] = []
        if input_mode in {"clean", "all"}:
            input_variants.append(("clean", clean_image))
        if input_mode in {"robust", "all"}:
            input_variants.append(
                (
                    "robust",
                    generate_robust_input(
                        model=model,
                        image=clean_image,
                        target=target,
                    ),
                )
            )

        for concrete_input_mode, image in input_variants:
            # Inference supplies the diagnostic prediction and full-image mean
            # class logit; CAM performs its own gradient-enabled forward pass.
            with torch.no_grad():
                logits_raw = model(image)
            if not isinstance(logits_raw, Tensor):
                raise TypeError("Segmentation model forward output must be a Tensor.")
            logits = cast(Logits, logits_raw)
            prediction = cast(ClassMap, logits.argmax(dim=1)[0])

            # The requested semantic class must exist in this checkpoint's
            # output channels; never silently explain another class instead.
            if not 0 <= target_class < int(logits.shape[1]):
                raise ValueError(
                    f"Target class {target_class} is outside model '{model_name}' "
                    f"output range [0, {int(logits.shape[1]) - 1}]."
                )

            # All selected rows for this image share the same target and input.
            # Generate the expensive map only once and reuse it for their panels.
            target_score = cam_target(cast(SingleLogits, logits[0]))
            heatmap = generate_cam(
                model=model,
                target_layer=target_layer,
                image=image,
                target=cam_target,
                epsilon=epsilon,
            )
            input_numpy = tensor_image_to_numpy(image)
            prediction_numpy = tensor_class_map_to_numpy(prediction)

            # Persist the EXACT input used for the CAM, especially model-specific
            # PGD inputs. A later faithfulness run must not regenerate a random
            # attack and accidentally evaluate a different reference example.
            cpu_input = image.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
            input_digest = hashlib.sha256(cpu_input.tobytes()).hexdigest()
            source_digest = hashlib.sha256(
                f"{image_path.resolve()}|{mask_path.resolve()}".encode()
            ).hexdigest()[:12]
            shared_input_path = (
                output_dir / sanitise_filename(model_name) / concrete_input_mode /
                "_inputs" / f"img{image_index:05d}_{source_digest}.npy"
            )
            shared_input_path.parent.mkdir(parents=True, exist_ok=True)
            # Store once for the image/mode, even if selected under several
            # categories. Atomic replacement avoids corrupting a previous file.
            shared_input_temp = shared_input_path.with_suffix(".tmp.npy")
            np.save(shared_input_temp, cpu_input, allow_pickle=False)
            shared_input_temp.replace(shared_input_path)

            for _, row in case_group.iterrows():
                # Selection class/category are provenance only and do not
                # influence the spatially global Grad-CAM++ target score.
                selected_class = optional_class_id(row["class_id"])

                heatmap_path, overlay_path, panel_path = case_output_paths(
                    output_dir=output_dir,
                    row=row,
                    target_class=target_class,
                    input_mode=concrete_input_mode,
                )
                title = build_case_title(
                    row=row,
                    selected_class=selected_class,
                    target_class=target_class,
                    input_mode=concrete_input_mode,
                )

                # PNG heatmaps are color-rendered and lose numerical fidelity;
                # save the actual library-normalized spatial CAM for occlusion.
                heatmap_npy_path = heatmap_path.with_suffix(".npy")
                heatmap_npy_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(heatmap_npy_path, heatmap.astype(np.float32), allow_pickle=False)

                save_case_visualisations(
                    image=input_numpy,
                    target_map=target_numpy,
                    prediction_map=prediction_numpy,
                    heatmap=heatmap,
                    heatmap_path=heatmap_path,
                    overlay_path=overlay_path,
                    panel_path=panel_path,
                    title=title,
                    colormap=colormap,
                    overlay_alpha=overlay_alpha,
                )

                selected_class_name = ""
                if selected_class is not None and selected_class < len(CLASS_NAMES):
                    selected_class_name = CLASS_NAMES[selected_class]
                target_class_name = (
                    CLASS_NAMES[target_class]
                    if target_class < len(CLASS_NAMES)
                    else f"class_{target_class}"
                )

                # Recording the concrete input mode is essential for --input-mode
                # all because one source case deliberately produces two explanations.
                manifest_rows.append(
                    {
                        "source_row_index": int(row["source_row_index"]),
                        "source_model_name": str(row["source_model_name"]),
                        "model_name": model_name,
                        "input_mode": concrete_input_mode,
                        "category": str(row["category"]),
                        "rank": int(float(row["rank"])),
                        "image_index": image_index,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "selection_score_name": str(row.get("score_name", "")),
                        "selection_score": row.get("score", np.nan),
                        "selected_class_id": selected_class,
                        "selected_class_name": selected_class_name,
                        "target_class_id": target_class,
                        "target_class_name": target_class_name,
                        "target_score": float(target_score.detach().item()),
                        "target_layer": layer_name,
                        "method_used": "gradcam++",
                        "heatmap_path": str(heatmap_path),
                        "heatmap_npy_path": str(heatmap_npy_path),
                        "input_tensor_path": str(shared_input_path),
                        "input_sha256": input_digest,
                        "attack_provenance": (
                            "evaluate_robustness.pgd_attack defaults"
                            if concrete_input_mode == "robust"
                            else "none"
                        ),
                        "overlay_path": str(overlay_path),
                        "panel_path": str(panel_path),
                    }
                )

            # Release variant-specific forward results before moving to the next
            # input mode; CAM backward intermediates can consume substantial VRAM.
            del logits, prediction

        # Release per-image tensors before the next group because manifests can
        # contain hundreds of cases and robust mode adds an adversarial tensor too.
        del clean_image, target, input_variants, image

    return manifest_rows


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for selected-case Grad-CAM generation.

    Steps
    -----
    1. Accept the selected-case manifest and one or more named checkpoints.
    2. Choose a fixed target class (road by default) and optional case limit.
    3. Choose clean, PGD-robust, or combined input processing.
    4. Configure the feature layer and rendering options. The target class's
       logits are averaged over the entire spatial output, without a mask.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM++ heatmaps for semantic-segmentation cases selected "
            "by explain_choose.py. Invalid Grad-CAM++ maps raise errors."
        )
    )
    parser.add_argument(
        "--selected-cases",
        type=str,
        default="./model/xai_selection/selected_cases.csv",
        help="Path to selected_cases.csv produced by explain_choose.py.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help=(
            "Checkpoint to explain. Repeat for multiple models. Accepts PATH or "
            "NAME=PATH; names should match selected_cases.csv model_name values."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Repository root used to resolve project-relative image/mask paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./model/xai_gradcam",
        help="Directory receiving heatmaps, overlays, panels, and output manifest.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="road",
        help=(
            "Semantic class name or numeric id whose logits are averaged across "
            "the entire output image for every case (default: road)."
        ),
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help=(
            "Optional cap after expanding cases across checkpoints, "
            "useful for a smoke test."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=["clean", "robust", "all"],
        default="clean",
        help=(
            "Input variant to explain: clean dataset images, robust images perturbed "
            "with evaluate_robustness.py PGD defaults, or both. Defaults to clean."
        ),
    )
    parser.add_argument(
        "--layer-scope",
        choices=["decoder", "encoder", "auto"],
        default="decoder",
        help="Automatic target-layer search scope when --target-layer is omitted.",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default=None,
        help=(
            "Optional dotted module path, e.g. decoder.blocks.4.conv2.0. "
            "By default the decoder module is used for decoder scope; "
            "encoder scope uses its last convolution."
        ),
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap used for heatmap PNGs. Defaults to jet.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Heatmap opacity over the RGB image. Defaults to 0.5.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Threshold for detecting degenerate library CAMs. Defaults to 1e-8.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate clean and/or PGD-robust Grad-CAM++ heatmaps for selected cases.

    Steps
    -----
    1. Read selected cases, expand cross-model rows, and apply the case limit.
    2. Save the resulting rows to ``expanded_cases.csv`` before any XAI work.
    3. Load checkpoints and generate clean/robust inputs for the chosen class.
    4. Explain the mean selected-class logit across all model output pixels,
       requiring a usable Grad-CAM++ map without any method fallback.
    5. Append each completed checkpoint's metadata to ``gradcam_manifest.csv``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the selected-case CSV, checkpoint, image, or mask cannot be found.
    ValueError
        If options are invalid or no selected cases match the checkpoints.
    RuntimeError
        If Grad-CAM++ fails to generate a usable explanation.
    """
    args = parse_arguments()

    if args.overlay_alpha < 0.0 or args.overlay_alpha > 1.0:
        raise ValueError("--overlay-alpha must be inside [0, 1].")
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be greater than 0.")

    project_root = FilePath(args.project_root).expanduser().resolve()
    selected_cases_path = resolve_project_path(args.selected_cases, project_root)
    if not selected_cases_path.is_file():
        raise FileNotFoundError(
            f"Selected-case manifest does not exist: {selected_cases_path}"
        )

    checkpoint_specs = [parse_checkpoint_spec(spec) for spec in args.checkpoint]
    checkpoint_names = [name for name, _ in checkpoint_specs]
    if len(checkpoint_names) != len(set(checkpoint_names)):
        raise ValueError("Every --checkpoint must use a unique model name.")

    # Resolve checkpoint paths relative to the same project root used by the CSV
    # so command behavior is deterministic regardless of the shell's cwd.
    checkpoint_map: Dict[str, FilePath] = {}
    for model_name, checkpoint_path in checkpoint_specs:
        resolved_checkpoint = resolve_project_path(checkpoint_path, project_root)
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint for '{model_name}' does not exist: {resolved_checkpoint}"
            )
        checkpoint_map[model_name] = resolved_checkpoint

    target_class = resolve_target_class(args.target_class)
    # The dataset's ignored label is not a scored semantic category, so do not
    # use its output channel as the global class-logit explanation target.
    if target_class == 19:
        raise ValueError("The ignored ground-truth class cannot be the CAM target.")
    selected_cases = pd.read_csv(selected_cases_path)
    cases = expand_and_filter_cases(
        selected_cases=selected_cases,
        checkpoint_names=checkpoint_names,
        max_cases=args.max_cases,
    )

    # Write the exact expanded, capped case table before loading any checkpoints
    # or generating explanations so the inputs can be audited after a failure.
    output_dir = resolve_project_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "expanded_cases.csv"
    cases.to_csv(cases_path, index=False)
    if cases.empty:
        raise ValueError("No selected cases match the supplied checkpoints.")

    print("\n" + "=" * 72)
    print("GRAD-CAM++ SEMANTIC-SEGMENTATION EXPLANATION PIPELINE")
    print("=" * 72)
    print(f"Execution device : {Configuration.DEVICE}")
    print(f"Selected cases   : {selected_cases_path}")
    print(f"Cases to explain : {len(cases)}")
    print(f"Expanded cases   : {cases_path}")
    print(f"Target class     : {CLASS_NAMES[target_class]} ({target_class})")
    print(f"Input mode       : {args.input_mode}")
    print("Method           : Grad-CAM++")
    print("Target score     : mean class logit across all output pixels")
    print(f"Output directory : {output_dir}")

    manifest_path = output_dir / "gradcam_manifest.csv"
    all_manifest_rows: List[Dict[str, object]] = []
    for model_name in checkpoint_names:
        model_cases = cases[cases["model_name"] == model_name].copy()
        if model_cases.empty:
            continue

        checkpoint_path = checkpoint_map[model_name]
        print(
            f"\nExplaining {len(model_cases)} case(s) for '{model_name}' "
            f"from {checkpoint_path}..."
        )
        model = load_trained_model(str(checkpoint_path), Configuration.DEVICE)
        model_rows = explain_cases_for_model(
            model=model,
            model_name=model_name,
            model_cases=model_cases,
            project_root=project_root,
            output_dir=output_dir,
            device=Configuration.DEVICE,
            layer_scope=args.layer_scope,
            target_layer_path=args.target_layer,
            target_class=target_class,
            colormap=args.colormap,
            overlay_alpha=args.overlay_alpha,
            epsilon=args.epsilon,
            input_mode=args.input_mode,
        )
        all_manifest_rows.extend(model_rows)

        # Upsert manifest rows for reproducible reruns. The old append behavior
        # could write mismatched columns after new artifact fields were added and
        # produced duplicates when a checkpoint was processed twice.
        model_manifest = pd.DataFrame(model_rows)
        if not model_manifest.empty:
            if manifest_path.is_file() and manifest_path.stat().st_size > 0:
                previous = pd.read_csv(manifest_path)
                model_manifest = pd.concat(
                    [previous, model_manifest], ignore_index=True
                )
            model_manifest.drop_duplicates(
                subset=["source_row_index", "model_name", "input_mode", "target_class_id"],
                keep="last",
            ).to_csv(manifest_path, index=False)

        # Release the checkpoint before loading the next one. The explanation
        # workload uses backward graphs, so explicit cleanup avoids cumulative VRAM.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = pd.DataFrame(all_manifest_rows)

    print("\nGrad-CAM generation complete.")
    print(f"  Explanations generated : {len(manifest)}")
    print(f"  Manifest               : {manifest_path}")


if __name__ == "__main__":
    main()
