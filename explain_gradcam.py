"""Generate Grad-CAM++ explanations for selected semantic-segmentation cases.

The program consumes the ``selected_cases.csv`` manifest written by
``explain_choose.py`` and generates class/region-targeted Grad-CAM++ heatmaps for
the corresponding trained segmentation checkpoints. Grad-CAM is used as a
numerical fallback when a Grad-CAM++ map is undefined or degenerate.
"""

import argparse
import re
from pathlib import Path as FilePath
from types import TracebackType
from typing import Dict, List, Optional, Sequence, Tuple, Type, cast

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch import Tensor, nn

from evaluate_performance import (
    BDDSegmentationDataset,
    CLASS_COLORS,
    CLASS_NAMES,
    Configuration,
    load_trained_model,
)


BatchImage = Float[Tensor, "1 3 h w"]
Logits = Float[Tensor, "1 c h w"]
FeatureMap = Float[Tensor, "1 c h w"]
Scalar = Float[Tensor, ""]
ClassMap = Int64[Tensor, "h w"]
BooleanMask = Bool[Tensor, "h w"]
HeatmapTensor = Float[Tensor, "h w"]
NumpyImage = Float[np.ndarray, "h w 3"]
NumpyHeatmap = Float[np.ndarray, "h w"]
NumpyClassMap = Int64[np.ndarray, "h w"]
NumpyBooleanMask = Bool[np.ndarray, "h w"]


class ActivationCapture:
    """Capture the output activation of one target module during inference.

    Grad-CAM methods require an intermediate convolutional activation that still
    belongs to the autograd graph. A forward hook is used instead of changing
    the segmentation architecture, keeping explanation generation independent
    of the exact U-Net encoder used by a checkpoint.

    Steps
    -----
    1. Register a forward hook on the requested target module.
    2. Store its tensor output every time the model performs a forward pass.
    3. Remove the hook when leaving the context manager.

    Parameters
    ----------
    module : nn.Module
        Model module whose forward output should be captured.

    Attributes
    ----------
    activation : Optional[Tensor]
        Most recently captured tensor output, or ``None`` before a forward pass.

    Raises
    ------
    TypeError
        If the target module returns a non-tensor output.
    """

    def __init__(self, module: nn.Module) -> None:
        """Register the forward hook used to capture target-layer activations.

        Steps
        -----
        1. Initialise the capture slot to ``None`` before any forward pass.
        2. Register this instance's callback on the requested target module.

        Parameters
        ----------
        module : nn.Module
            Model layer whose output activation should be captured.
        """
        self.activation: Optional[Tensor] = None
        self._handle = module.register_forward_hook(self._capture)

    def _capture(
        self,
        module: nn.Module,
        inputs: Tuple[object, ...],
        output: object,
    ) -> None:
        """Store one hooked module output without detaching it from autograd.

        Parameters
        ----------
        module : nn.Module
            Module that triggered the hook. It is accepted for the PyTorch hook
            interface and is not otherwise inspected.
        inputs : Tuple[object, ...]
            Positional inputs supplied to the hooked module. They are retained
            only to satisfy the hook callback signature.
        output : object
            Hooked module output expected to be a tensor.

        Steps
        -----
        1. Validate that the hooked output is a tensor.
        2. Keep the graph-connected activation for later target differentiation.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``output`` is not a ``torch.Tensor``.
        """
        del module, inputs
        if not isinstance(output, Tensor):
            raise TypeError(
                "The selected Grad-CAM target layer must return a torch.Tensor."
            )
        self.activation = output

    def __enter__(self) -> "ActivationCapture":
        """Return this capture object after the hook has been registered.

        Steps
        -----
        1. Return the already-initialised capture object without changing the hook.

        Returns
        -------
        ActivationCapture
            Active capture object used inside the context manager.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Remove the forward hook when explanation generation is finished.

        Parameters
        ----------
        exc_type : object
            Exception type propagated by the context manager, if any.
        exc_value : object
            Exception instance propagated by the context manager, if any.
        traceback : object
            Traceback propagated by the context manager, if any.

        Steps
        -----
        1. Discard exception metadata because it does not affect hook cleanup.
        2. Remove the PyTorch forward hook exactly once.

        Returns
        -------
        None
        """
        del exc_type, exc_value, traceback
        self._handle.remove()


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
                    f"Cannot index component '{component}' inside {type(current).__name__}."
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


@jaxtyped(typechecker=beartype)
def semantic_boundary_mask(
    class_map: ClassMap,
    ignored_class: int,
    radius: int,
) -> BooleanMask:
    """Build a dilated semantic-boundary mask from a class-index image.

    This mirrors the case-selection definition so a boundary-failure heatmap is
    targeted at the same type of spatial region that caused the image to be
    selected in the first place.

    Steps
    -----
    1. Mark valid horizontal and vertical class transitions.
    2. Mark both sides of every transition for a symmetric contour.
    3. Dilate the contour by ``radius`` pixels with max pooling.

    Parameters
    ----------
    class_map : ClassMap
        Ground-truth class-index map of shape ``(H, W)``.
    ignored_class : int
        Semantic class excluded from explanation scoring.
    radius : int
        Boundary dilation radius in resized pixels.

    Returns
    -------
    BooleanMask
        Dilated semantic-boundary mask of shape ``(H, W)``.

    Raises
    ------
    ValueError
        If ``radius`` is negative.
    """
    if radius < 0:
        raise ValueError("radius must be greater than or equal to 0.")

    boundary = torch.zeros_like(class_map, dtype=torch.bool)
    valid = class_map != ignored_class

    horizontal = (
        (class_map[:, 1:] != class_map[:, :-1])
        & valid[:, 1:]
        & valid[:, :-1]
    )
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal

    vertical = (
        (class_map[1:, :] != class_map[:-1, :])
        & valid[1:, :]
        & valid[:-1, :]
    )
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical

    if radius == 0:
        return cast(BooleanMask, boundary)

    kernel_size = (2 * radius) + 1
    expanded = F.max_pool2d(
        boundary.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )
    return cast(BooleanMask, expanded.squeeze(0).squeeze(0).bool())


@jaxtyped(typechecker=beartype)
def class_pair_boundary_mask(
    class_map: ClassMap,
    first_class: int,
    second_class: int,
    ignored_class: int,
    radius: int,
) -> BooleanMask:
    """Build a dilated boundary mask for one semantic class pair.

    Road/sidewalk cases are selected using this narrower region so unrelated
    object boundaries do not determine which class receives the Grad-CAM target.

    Steps
    -----
    1. Detect adjacent pixels containing the requested class pair in either order.
    2. Mark both sides of each pair transition.
    3. Dilate the contour and retain pixels belonging to either requested class.

    Parameters
    ----------
    class_map : ClassMap
        Ground-truth class-index map of shape ``(H, W)``.
    first_class : int
        First semantic class id.
    second_class : int
        Second semantic class id.
    ignored_class : int
        Semantic class excluded from scoring.
    radius : int
        Boundary dilation radius in resized pixels.

    Returns
    -------
    BooleanMask
        Pair-specific boundary mask of shape ``(H, W)``.

    Raises
    ------
    ValueError
        If the requested class ids are equal or ``radius`` is negative.
    """
    if first_class == second_class:
        raise ValueError("first_class and second_class must be different.")
    if radius < 0:
        raise ValueError("radius must be greater than or equal to 0.")

    boundary = torch.zeros_like(class_map, dtype=torch.bool)
    valid = class_map != ignored_class

    left = class_map[:, :-1]
    right = class_map[:, 1:]
    horizontal = (
        (((left == first_class) & (right == second_class))
         | ((left == second_class) & (right == first_class)))
        & valid[:, :-1]
        & valid[:, 1:]
    )
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal

    top = class_map[:-1, :]
    bottom = class_map[1:, :]
    vertical = (
        (((top == first_class) & (bottom == second_class))
         | ((top == second_class) & (bottom == first_class)))
        & valid[:-1, :]
        & valid[1:, :]
    )
    boundary[:-1, :] |= vertical
    boundary[1:, :] |= vertical

    if radius > 0:
        kernel_size = (2 * radius) + 1
        boundary = F.max_pool2d(
            boundary.float().unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        ).squeeze(0).squeeze(0).bool()

    pair_pixels = (class_map == first_class) | (class_map == second_class)
    return cast(BooleanMask, boundary & pair_pixels)


@jaxtyped(typechecker=beartype)
def dominant_class_in_region(
    class_map: ClassMap,
    region: BooleanMask,
    num_output_classes: int,
    ignored_class: int,
) -> Optional[int]:
    """Return the most frequent non-ignored class inside a boolean region.

    Steps
    -----
    1. Extract class ids covered by ``region``.
    2. Remove ignored and out-of-range labels.
    3. Return the class with the largest pixel count, or ``None`` when empty.

    Parameters
    ----------
    class_map : ClassMap
        Semantic class-index map of shape ``(H, W)``.
    region : BooleanMask
        Spatial region from which class frequencies are measured.
    num_output_classes : int
        Number of class channels produced by the model.
    ignored_class : int
        Class id excluded from automatic target selection when it lies inside the
        output range; use ``-1`` to exclude no predicted class.

    Returns
    -------
    Optional[int]
        Dominant valid class id, or ``None`` if no eligible pixels exist.
    """
    values = class_map[region]
    if values.numel() == 0:
        return None

    valid = (values >= 0) & (values < num_output_classes) & (values != ignored_class)
    values = values[valid]
    if values.numel() == 0:
        return None

    counts = torch.bincount(values, minlength=num_output_classes)
    if 0 <= ignored_class < num_output_classes:
        counts[ignored_class] = 0
    if int(counts.sum().item()) == 0:
        return None
    return int(counts.argmax().item())


@jaxtyped(typechecker=beartype)
def fallback_class_region(
    prediction: ClassMap,
    target: ClassMap,
    target_class: int,
    ignored_class: int,
) -> Tuple[BooleanMask, str]:
    """Find a non-empty fallback region for a requested target class.

    The chosen category-specific region can occasionally be empty, for example
    when a nominal false-positive row changes after loading a different model.
    Falling back to GT, prediction, then all valid pixels keeps the script usable
    while recording the region actually explained in the output manifest.

    Steps
    -----
    1. Prefer ground-truth pixels of ``target_class``.
    2. Otherwise use predicted pixels of the target class.
    3. Otherwise use every non-ignored ground-truth pixel.

    Parameters
    ----------
    prediction : ClassMap
        Predicted class-index map of shape ``(H, W)``.
    target : ClassMap
        Ground-truth class-index map of shape ``(H, W)``.
    target_class : int
        Class whose logit will be explained.
    ignored_class : int
        Ground-truth class excluded from explanation regions.

    Returns
    -------
    Tuple[BooleanMask, str]
        Non-empty region and a description of the fallback source.

    Raises
    ------
    ValueError
        If the image contains no valid non-ignored target pixels at all.
    """
    gt_region = target == target_class
    if bool(gt_region.any().item()):
        return cast(BooleanMask, gt_region), "fallback_ground_truth_class"

    predicted_region = (prediction == target_class) & (target != ignored_class)
    if bool(predicted_region.any().item()):
        return cast(BooleanMask, predicted_region), "fallback_predicted_class"

    valid = target != ignored_class
    if bool(valid.any().item()):
        return cast(BooleanMask, valid), "fallback_all_valid_pixels"
    raise ValueError("The selected image contains no valid non-ignored pixels.")


@jaxtyped(typechecker=beartype)
def choose_target_class_and_region(
    prediction: ClassMap,
    target: ClassMap,
    category: str,
    selected_class: Optional[int],
    num_output_classes: int,
    ignored_class: int,
    boundary_radius: int,
    region_mode: str,
) -> Tuple[int, BooleanMask, str]:
    """Choose the semantic class and pixel region explained by one selected case.

    Automatic targeting makes each selection category answer a concrete question.
    False positives explain the selected class where it was predicted wrongly;
    false negatives explain the dominant competing prediction in the missed GT
    region; small-region cases explain the selected GT class; and classless
    failure cases explain the dominant wrong prediction inside the relevant error
    region. Successful classless cases explain the dominant true-positive class.

    Steps
    -----
    1. Honour an explicit region mode when the user requests GT/prediction/error.
    2. Otherwise apply category-specific targeting rules from ``selected_cases``.
    3. Fall back to a valid GT/prediction region if the preferred region is empty.

    Parameters
    ----------
    prediction : ClassMap
        Predicted class-index map of shape ``(H, W)``.
    target : ClassMap
        Ground-truth class-index map of shape ``(H, W)``.
    category : str
        Case-selection category from ``selected_cases.csv``.
    selected_class : Optional[int]
        Class id attached to the selection row, or ``None`` for image-level cases.
    num_output_classes : int
        Number of model output channels.
    ignored_class : int
        Ground-truth class excluded from evaluation/explanation targeting.
    boundary_radius : int
        Dilation radius for boundary-failure targeting.
    region_mode : str
        One of ``auto``, ``ground_truth``, ``prediction``, or ``error``.

    Returns
    -------
    Tuple[int, BooleanMask, str]
        Target class id, non-empty region mask, and region description.

    Raises
    ------
    ValueError
        If ``region_mode`` is invalid, a selected class is outside the model
        output range, or no valid target class/region can be constructed.
    """
    if region_mode not in {"auto", "ground_truth", "prediction", "error"}:
        raise ValueError(
            "region_mode must be one of: auto, ground_truth, prediction, error."
        )
    if selected_class is not None and not 0 <= selected_class < num_output_classes:
        raise ValueError(
            f"Selected class {selected_class} is outside model output range "
            f"[0, {num_output_classes - 1}]."
        )

    valid = target != ignored_class
    errors = (prediction != target) & valid

    # Explicit modes are useful for controlled experiments where all selected
    # cases should use the same semantic definition instead of category logic.
    if region_mode == "ground_truth":
        target_class = selected_class
        if target_class is None:
            target_class = dominant_class_in_region(
                target, valid, num_output_classes, ignored_class
            )
        if target_class is None:
            raise ValueError("Unable to choose a ground-truth target class.")
        region = target == target_class
        if bool(region.any().item()):
            return target_class, cast(BooleanMask, region), "ground_truth_class"
        fallback, fallback_name = fallback_class_region(
            prediction, target, target_class, ignored_class
        )
        return target_class, fallback, fallback_name

    if region_mode == "prediction":
        target_class = selected_class
        if target_class is None:
            target_class = dominant_class_in_region(
                prediction, valid, num_output_classes, -1
            )
        if target_class is None:
            raise ValueError("Unable to choose a predicted target class.")
        region = (prediction == target_class) & valid
        if bool(region.any().item()):
            return target_class, cast(BooleanMask, region), "predicted_class"
        fallback, fallback_name = fallback_class_region(
            prediction, target, target_class, ignored_class
        )
        return target_class, fallback, fallback_name

    if region_mode == "error":
        error_region = errors
        if selected_class is not None:
            selected_fp = errors & (prediction == selected_class)
            selected_fn = errors & (target == selected_class)
            if bool(selected_fp.any().item()):
                return selected_class, cast(BooleanMask, selected_fp), "selected_class_error"
            competitor = dominant_class_in_region(
                prediction, selected_fn, num_output_classes, -1
            )
            if competitor is not None:
                competitor_region = selected_fn & (prediction == competitor)
                return competitor, cast(BooleanMask, competitor_region), "selected_class_competitor"

        target_class = dominant_class_in_region(
            prediction, error_region, num_output_classes, -1
        )
        if target_class is not None:
            region = error_region & (prediction == target_class)
            return target_class, cast(BooleanMask, region), "dominant_prediction_error"

        # Perfect predictions have no error pixels, so use a true-positive class
        # rather than failing an otherwise valid explanation request.
        true_positive = prediction == target
        target_class = dominant_class_in_region(
            target, true_positive & valid, num_output_classes, ignored_class
        )
        if target_class is None:
            raise ValueError("Unable to choose an error-region fallback target.")
        region = true_positive & (target == target_class)
        return target_class, cast(BooleanMask, region), "error_mode_true_positive_fallback"

    # Class-specific categories keep the selected class where that class itself
    # is the phenomenon of interest, except FN cases where the competing class
    # answers why the requested semantic region was missed.
    if selected_class is not None:
        if category == "highest_fp_rate":
            fp_region = (prediction == selected_class) & (target != selected_class) & valid
            if bool(fp_region.any().item()):
                return selected_class, cast(BooleanMask, fp_region), "false_positive_region"

        if category == "highest_fn_rate":
            fn_region = (target == selected_class) & (prediction != selected_class) & valid
            competitor = dominant_class_in_region(
                prediction, fn_region, num_output_classes, -1
            )
            if competitor is not None:
                region = fn_region & (prediction == competitor)
                return competitor, cast(BooleanMask, region), "false_negative_competitor_region"
            if bool(fn_region.any().item()):
                return selected_class, cast(BooleanMask, fn_region), "false_negative_region"

        gt_region = target == selected_class
        if bool(gt_region.any().item()):
            return selected_class, cast(BooleanMask, gt_region), "ground_truth_selected_class"

        predicted_region = (prediction == selected_class) & valid
        if bool(predicted_region.any().item()):
            return selected_class, cast(BooleanMask, predicted_region), "predicted_selected_class"

        fallback, fallback_name = fallback_class_region(
            prediction, target, selected_class, ignored_class
        )
        return selected_class, fallback, fallback_name

    # Boundary categories explain the dominant wrong prediction inside the exact
    # boundary band responsible for selection rather than unrelated image errors.
    if category == "road_sidewalk_boundary_failure":
        boundary = class_pair_boundary_mask(
            target,
            first_class=0,
            second_class=1,
            ignored_class=ignored_class,
            radius=boundary_radius,
        )
        boundary_errors = boundary & errors
        target_class = dominant_class_in_region(
            prediction, boundary_errors, num_output_classes, -1
        )
        if target_class is not None:
            region = boundary_errors & (prediction == target_class)
            return target_class, cast(BooleanMask, region), "road_sidewalk_boundary_error"

    if category == "boundary_failure":
        boundary = semantic_boundary_mask(target, ignored_class, boundary_radius)
        boundary_errors = boundary & errors
        target_class = dominant_class_in_region(
            prediction, boundary_errors, num_output_classes, -1
        )
        if target_class is not None:
            region = boundary_errors & (prediction == target_class)
            return target_class, cast(BooleanMask, region), "semantic_boundary_error"

    failure_categories = {"low_iou", "shared_low_iou", "model_disagreement"}
    if category in failure_categories:
        target_class = dominant_class_in_region(
            prediction, errors, num_output_classes, -1
        )
        if target_class is not None:
            region = errors & (prediction == target_class)
            return target_class, cast(BooleanMask, region), "dominant_error_prediction"

    # Successful/typical image-level cases are best represented by the class with
    # the largest correctly segmented region, which yields a stable scalar target.
    true_positive = (prediction == target) & valid
    target_class = dominant_class_in_region(
        target, true_positive, num_output_classes, ignored_class
    )
    if target_class is not None:
        region = true_positive & (target == target_class)
        return target_class, cast(BooleanMask, region), "dominant_true_positive"

    # If no true positives exist, explain the most common non-ignored prediction
    # so even a catastrophic segmentation failure can still produce a heatmap.
    target_class = dominant_class_in_region(
        prediction, valid, num_output_classes, -1
    )
    if target_class is None:
        raise ValueError("Unable to choose any valid automatic Grad-CAM target class.")
    region = (prediction == target_class) & valid
    return target_class, cast(BooleanMask, region), "dominant_prediction_fallback"


@jaxtyped(typechecker=beartype)
def segmentation_target_score(
    logits: Logits,
    target_class: int,
    region: BooleanMask,
) -> Scalar:
    """Reduce dense segmentation logits to one scalar Grad-CAM target score.

    Grad-CAM requires a scalar output. Averaging the selected class logit only
    over the case-specific region preserves the semantic question encoded by the
    selection category while avoiding domination by unrelated image pixels.

    Steps
    -----
    1. Validate the requested class channel and non-empty spatial region.
    2. Extract target-class logits over the region.
    3. Return their mean as the scalar differentiation target.

    Parameters
    ----------
    logits : Logits
        Model output logits with shape ``(1, C, H, W)``.
    target_class : int
        Class channel whose evidence should be explained.
    region : BooleanMask
        Spatial mask of shape ``(H, W)`` defining the explained output region.

    Returns
    -------
    Tensor
        Scalar mean target-class logit.

    Raises
    ------
    ValueError
        If the class id is outside the logits range or the region is empty.
    """
    if target_class < 0 or target_class >= logits.shape[1]:
        raise ValueError(
            f"Target class {target_class} is outside logits range "
            f"[0, {logits.shape[1] - 1}]."
        )
    if not bool(region.any().item()):
        raise ValueError("Grad-CAM target region must contain at least one pixel.")

    class_logits = logits[0, target_class]
    return class_logits[region].mean()


@jaxtyped(typechecker=beartype)
def gradcam_plus_plus_map(
    activation: FeatureMap,
    gradient: FeatureMap,
    epsilon: float,
) -> Optional[HeatmapTensor]:
    """Compute a practical Grad-CAM++ map from activations and first gradients.

    This follows the commonly used Grad-CAM++ weighting form where squared and
    cubed first-order gradients approximate the per-location alpha coefficients.
    Positive gradients are then weighted and summed across channels. Returning
    ``None`` for non-finite or degenerate results triggers the requested Grad-CAM
    fallback instead of emitting a misleading blank heatmap.

    Steps
    -----
    1. Compute squared/cubed gradients and Grad-CAM++ alpha coefficients.
    2. Weight positive gradients spatially to obtain one weight per channel.
    3. Sum weighted activations and apply ReLU to retain positive evidence.
    4. Reject non-finite or effectively zero maps so Grad-CAM can be used instead.

    Parameters
    ----------
    activation : FeatureMap
        Target-layer activation with shape ``(1, C, H, W)``.
    gradient : FeatureMap
        Gradient of the scalar target score with respect to ``activation``.
    epsilon : float
        Numerical stability threshold used for denominator and map checks.

    Returns
    -------
    Optional[HeatmapTensor]
        Raw non-negative CAM at feature resolution, or ``None`` when unusable.

    Raises
    ------
    ValueError
        If activation and gradient shapes differ or batch size is not one.
    """
    if activation.shape != gradient.shape:
        raise ValueError(
            f"Activation/gradient shapes differ: {activation.shape} vs {gradient.shape}."
        )
    if activation.shape[0] != 1:
        raise ValueError("Grad-CAM++ generation requires batch size 1.")

    gradients_squared = gradient.pow(2)
    gradients_cubed = gradient.pow(3)
    activation_sum = activation.sum(dim=(2, 3), keepdim=True)
    denominator = (2.0 * gradients_squared) + (activation_sum * gradients_cubed)

    # Only divide where the denominator is numerically meaningful. Zero-alpha
    # locations contribute nothing instead of introducing NaN/Inf artifacts.
    safe_denominator = torch.where(
        denominator.abs() > epsilon,
        denominator,
        torch.ones_like(denominator),
    )
    alpha = gradients_squared / safe_denominator
    alpha = torch.where(
        denominator.abs() > epsilon,
        alpha,
        torch.zeros_like(alpha),
    )

    positive_gradient = F.relu(gradient)
    weights = (alpha * positive_gradient).sum(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1).squeeze(0))

    if not bool(torch.isfinite(cam).all().item()):
        return None
    if float(cam.max().item()) <= epsilon:
        return None
    return cast(HeatmapTensor, cam)


@jaxtyped(typechecker=beartype)
def gradcam_map(
    activation: FeatureMap,
    gradient: FeatureMap,
) -> HeatmapTensor:
    """Compute the standard Grad-CAM map used as the numerical fallback.

    Steps
    -----
    1. Global-average the target gradient into one weight per feature channel.
    2. Sum the weighted target-layer activations.
    3. Apply ReLU so the heatmap represents positive evidence for the target.

    Parameters
    ----------
    activation : FeatureMap
        Target-layer activation with shape ``(1, C, H, W)``.
    gradient : FeatureMap
        Gradient of the scalar target score with respect to ``activation``.

    Returns
    -------
    HeatmapTensor
        Raw non-negative Grad-CAM map at feature-map resolution.

    Raises
    ------
    ValueError
        If activation and gradient shapes differ or batch size is not one.
    """
    if activation.shape != gradient.shape:
        raise ValueError(
            f"Activation/gradient shapes differ: {activation.shape} vs {gradient.shape}."
        )
    if activation.shape[0] != 1:
        raise ValueError("Grad-CAM generation requires batch size 1.")

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1).squeeze(0))
    return cast(HeatmapTensor, cam)


@jaxtyped(typechecker=beartype)
def resize_and_normalise_cam(
    cam: HeatmapTensor,
    output_height: int,
    output_width: int,
    epsilon: float,
) -> NumpyHeatmap:
    """Resize a raw CAM to input resolution and normalise it to ``[0, 1]``.

    Steps
    -----
    1. Bilinearly resize the feature-resolution CAM to image resolution.
    2. Replace non-finite values with zeros before visualisation.
    3. Min-max normalise non-constant maps and return float32 NumPy data.

    Parameters
    ----------
    cam : HeatmapTensor
        Raw CAM of shape ``(h, w)``.
    output_height : int
        Desired image-space heatmap height.
    output_width : int
        Desired image-space heatmap width.
    epsilon : float
        Minimum value range required for min-max normalisation.

    Returns
    -------
    NumpyHeatmap
        Normalised heatmap of shape ``(output_height, output_width)``.

    Raises
    ------
    ValueError
        If either requested output dimension is less than one.
    """
    if output_height < 1 or output_width < 1:
        raise ValueError("CAM output dimensions must be positive.")

    resized = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    resized = torch.nan_to_num(resized, nan=0.0, posinf=0.0, neginf=0.0)

    minimum = resized.min()
    maximum = resized.max()
    value_range = maximum - minimum
    if float(value_range.item()) <= epsilon:
        normalised = torch.zeros_like(resized)
    else:
        normalised = (resized - minimum) / value_range

    array = normalised.detach().cpu().numpy().astype(np.float32, copy=False)
    return cast(NumpyHeatmap, array)


@jaxtyped(typechecker=beartype)
def generate_cam(
    score: Scalar,
    activation: FeatureMap,
    method: str,
    output_height: int,
    output_width: int,
    retain_graph: bool,
    epsilon: float,
) -> Tuple[NumpyHeatmap, str]:
    """Differentiate one segmentation target and generate its explanation map.

    ``gradcam++`` is attempted first when requested. The same first-order gradient
    is sufficient for both the practical Grad-CAM++ weighting and ordinary
    Grad-CAM, so a numerical fallback requires no second model forward pass.

    Steps
    -----
    1. Differentiate the scalar target score with respect to target activations.
    2. Build Grad-CAM++ when requested and numerically valid.
    3. Fall back to Grad-CAM otherwise.
    4. Resize and normalise the chosen map to the input image dimensions.

    Parameters
    ----------
    score : Tensor
        Scalar segmentation target score.
    activation : FeatureMap
        Captured target-layer activation with shape ``(1, C, h, w)``.
    method : str
        Requested method: ``gradcam++`` or ``gradcam``.
    output_height : int
        Input image height.
    output_width : int
        Input image width.
    retain_graph : bool
        Whether autograd should preserve the graph for additional cases from the
        same image forward pass.
    epsilon : float
        Numerical stability threshold.

    Returns
    -------
    Tuple[NumpyHeatmap, str]
        Normalised image-resolution heatmap and actual method used.

    Raises
    ------
    ValueError
        If the requested method is invalid or captured activation is not 4-D.
    RuntimeError
        If autograd cannot differentiate the target score from the activation.
    """
    if method not in {"gradcam++", "gradcam"}:
        raise ValueError("method must be either 'gradcam++' or 'gradcam'.")
    if activation.ndim != 4:
        raise ValueError(
            f"Target layer must produce a 4-D feature map, got {activation.shape}."
        )

    gradient = torch.autograd.grad(
        score,
        activation,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )[0]
    typed_gradient = cast(FeatureMap, gradient)

    method_used = "gradcam"
    raw_cam: HeatmapTensor
    if method == "gradcam++":
        plus_plus_cam = gradcam_plus_plus_map(activation, typed_gradient, epsilon)
        if plus_plus_cam is not None:
            raw_cam = plus_plus_cam
            method_used = "gradcam++"
        else:
            raw_cam = gradcam_map(activation, typed_gradient)
            method_used = "gradcam_fallback"
    else:
        raw_cam = gradcam_map(activation, typed_gradient)

    heatmap = resize_and_normalise_cam(
        raw_cam,
        output_height=output_height,
        output_width=output_width,
        epsilon=epsilon,
    )
    return heatmap, method_used


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
def tensor_boolean_mask_to_numpy(mask: BooleanMask) -> NumpyBooleanMask:
    """Convert a torch boolean region mask to a NumPy boolean array.

    Steps
    -----
    1. Detach the mask, move it to CPU, and preserve boolean semantics.

    Parameters
    ----------
    mask : BooleanMask
        Boolean tensor of shape ``(H, W)``.

    Returns
    -------
    NumpyBooleanMask
        CPU boolean array with shape ``(H, W)``.
    """
    array = mask.detach().cpu().numpy().astype(bool, copy=False)
    return cast(NumpyBooleanMask, array)


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
    target_region: NumpyBooleanMask,
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
    a heatmap cannot be interpreted without seeing whether the segmentation was
    actually correct. The target-region contour additionally documents exactly
    which output pixels were reduced into the scalar Grad-CAM score.

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
    target_region : NumpyBooleanMask
        Region used to construct the scalar explanation target.
    heatmap : NumpyHeatmap
        Normalised Grad-CAM heatmap of shape ``(H, W)``.
    heatmap_path : FilePath
        Destination for the standalone heatmap PNG.
    overlay_path : FilePath
        Destination for the image-plus-heatmap PNG.
    panel_path : FilePath
        Destination for the diagnostic panel PNG.
    title : str
        Explanation title written on the panel.
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
    if bool(target_region.any()):
        overlay_axis.contour(target_region.astype(np.uint8), levels=[0.5], linewidths=1.0)
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
    if bool(target_region.any()):
        axes[3].contour(target_region.astype(np.uint8), levels=[0.5], linewidths=1.0)
    axes[3].set_title("Grad-CAM explanation")

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


def resolve_class_filters(class_specs: Sequence[str]) -> List[int]:
    """Resolve optional class-name/id CLI filters against ``CLASS_NAMES``.

    Steps
    -----
    1. Interpret numeric values directly as class ids.
    2. Resolve text values case-insensitively through ``CLASS_NAMES``.
    3. Deduplicate ids while preserving user order.

    Parameters
    ----------
    class_specs : Sequence[str]
        Requested class ids or names.

    Returns
    -------
    List[int]
        Unique class ids to retain.

    Raises
    ------
    ValueError
        If a class id/name is not valid for the configured segmentation labels.
    """
    name_to_id = {name.lower(): index for index, name in enumerate(CLASS_NAMES)}
    resolved: List[int] = []
    for specification in class_specs:
        stripped = specification.strip()
        try:
            class_id = int(stripped)
        except ValueError:
            lookup = stripped.lower()
            if lookup not in name_to_id:
                raise ValueError(
                    f"Unknown class '{specification}'. Valid names: {', '.join(CLASS_NAMES)}."
                ) from None
            class_id = name_to_id[lookup]

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(
                f"Class id {class_id} is outside [0, {len(CLASS_NAMES) - 1}]."
            )
        if class_id not in resolved:
            resolved.append(class_id)
    return resolved


def expand_and_filter_cases(
    selected_cases: pd.DataFrame,
    checkpoint_names: Sequence[str],
    model_filters: Sequence[str],
    category_filters: Sequence[str],
    class_filters: Sequence[int],
    rank_max: Optional[int],
    max_cases: Optional[int],
) -> pd.DataFrame:
    """Expand cross-model rows and apply optional explanation filters.

    ``all_models`` rows from ``explain_choose.py`` describe cross-model cases.
    They are expanded once per supplied checkpoint so the same image can receive
    comparable explanations from every model actually provided to this script.

    Steps
    -----
    1. Keep model-specific rows only for supplied checkpoints.
    2. Expand ``all_models`` rows across every supplied checkpoint.
    3. Apply optional model/category/class/rank filters.
    4. Preserve stable CSV order and optionally cap total work with ``max_cases``.

    Parameters
    ----------
    selected_cases : pd.DataFrame
        Raw ``selected_cases.csv`` contents.
    checkpoint_names : Sequence[str]
        Model labels available through ``--checkpoint``.
    model_filters : Sequence[str]
        Optional subset of model labels requested by the user.
    category_filters : Sequence[str]
        Optional selection categories requested by the user.
    class_filters : Sequence[int]
        Optional class ids. Classless image-level rows are omitted when supplied.
    rank_max : Optional[int]
        Optional maximum category rank to retain.
    max_cases : Optional[int]
        Optional cap on total case rows after filtering.

    Returns
    -------
    pd.DataFrame
        Expanded and filtered case table with ``source_model_name`` retained.

    Raises
    ------
    ValueError
        If required manifest columns are missing, filters reference unavailable
        models/categories, or numeric limits are invalid.
    """
    required_columns = {
        "category",
        "model_name",
        "class_id",
        "image_index",
        "image_path",
        "mask_path",
        "rank",
    }
    missing = sorted(required_columns - set(selected_cases.columns))
    if missing:
        raise ValueError(
            "selected_cases.csv is missing required columns: " + ", ".join(missing)
        )
    if rank_max is not None and rank_max < 1:
        raise ValueError("rank_max must be at least 1 when provided.")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be at least 1 when provided.")

    checkpoint_set = set(checkpoint_names)
    requested_models = set(model_filters) if model_filters else checkpoint_set
    unavailable_models = requested_models - checkpoint_set
    if unavailable_models:
        raise ValueError(
            "--model requested checkpoint labels that were not supplied: "
            + ", ".join(sorted(unavailable_models))
        )

    available_categories = set(map(str, selected_cases["category"].dropna().unique()))
    missing_categories = set(category_filters) - available_categories
    if missing_categories:
        raise ValueError(
            "Unknown --category values: " + ", ".join(sorted(missing_categories))
        )

    expanded_rows: List[Dict[str, object]] = []
    for source_row_number, (_, row) in enumerate(selected_cases.iterrows()):
        source_model_name = str(row["model_name"])
        if source_model_name == "all_models":
            targets = [name for name in checkpoint_names if name in requested_models]
        elif source_model_name in checkpoint_set and source_model_name in requested_models:
            targets = [source_model_name]
        else:
            continue

        for target_model_name in targets:
            record = dict(row)
            record["source_row_index"] = source_row_number
            record["source_model_name"] = source_model_name
            record["model_name"] = target_model_name
            expanded_rows.append(record)

    frame = pd.DataFrame(expanded_rows)
    if frame.empty:
        return frame

    if category_filters:
        frame = frame[frame["category"].isin(category_filters)].copy()
    if class_filters:
        class_numeric = pd.to_numeric(frame["class_id"], errors="coerce")
        frame = frame[class_numeric.isin(class_filters)].copy()
    if rank_max is not None:
        rank_numeric = pd.to_numeric(frame["rank"], errors="coerce")
        frame = frame[rank_numeric <= rank_max].copy()

    frame = frame.reset_index(drop=True)
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


def case_output_paths(
    output_dir: FilePath,
    row: pd.Series,
    target_class: int,
) -> Tuple[FilePath, FilePath, FilePath]:
    """Build deterministic output paths for one explanation case.

    Steps
    -----
    1. Organise outputs by model then selection category.
    2. Encode source-row/rank/image/selected-class/target-class in the filename.
    3. Return separate paths for heatmap, overlay, and diagnostic panel.

    Parameters
    ----------
    output_dir : FilePath
        Root directory receiving Grad-CAM artifacts.
    row : pd.Series
        Expanded selected-case row.
    target_class : int
        Actual model class explained by Grad-CAM.

    Returns
    -------
    Tuple[FilePath, FilePath, FilePath]
        Heatmap PNG path, overlay PNG path, and panel PNG path.
    """
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
    destination = output_dir / model_name / category
    return (
        destination / f"{basename}_heatmap.png",
        destination / f"{basename}_overlay.png",
        destination / f"{basename}_panel.png",
    )


def build_case_title(
    row: pd.Series,
    selected_class: Optional[int],
    target_class: int,
    target_region_name: str,
    method_used: str,
) -> str:
    """Build a compact, self-describing title for one explanation figure.

    Steps
    -----
    1. Resolve selected and actual target class ids into readable names.
    2. Combine model, category, region, and method metadata into one title.

    Parameters
    ----------
    row : pd.Series
        Expanded selected-case metadata.
    selected_class : Optional[int]
        Class associated with the original selection row.
    target_class : int
        Actual class whose logit was differentiated.
    target_region_name : str
        Description of the pixel region used for the scalar target.
    method_used : str
        Grad-CAM method actually used after any fallback.

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
        f"{row['model_name']} | {row['category']} | selected={selected_name} | "
        f"target={target_name} | region={target_region_name} | {method_used}"
    )


def explain_cases_for_model(
    model: nn.Module,
    model_name: str,
    model_cases: pd.DataFrame,
    project_root: FilePath,
    output_dir: FilePath,
    device: torch.device,
    layer_scope: str,
    target_layer_path: Optional[str],
    method: str,
    region_mode: str,
    boundary_radius: int,
    colormap: str,
    overlay_alpha: float,
    epsilon: float,
) -> List[Dict[str, object]]:
    """Generate Grad-CAM explanations for every selected case of one model.

    Cases are grouped by image path so one forward pass can serve several
    selected categories/classes from the same image. Autograd retains that graph
    only until the final case in the group, reducing repeated inference without
    accumulating graphs across images.

    Steps
    -----
    1. Discover/register the model's target convolutional layer.
    2. Group selected rows by image/mask pair and run one differentiable forward.
    3. Choose a class/region target for each case and generate Grad-CAM++/fallback.
    4. Save figures and return one manifest row per explanation.

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
        Device used for forward/backward computation.
    layer_scope : str
        Automatic layer scope: ``decoder``, ``encoder``, or ``auto``.
    target_layer_path : Optional[str]
        Optional explicit dotted target-layer path.
    method : str
        Requested method: ``gradcam++`` or ``gradcam``.
    region_mode : str
        Target-region strategy: ``auto``, ``ground_truth``, ``prediction``, or
        ``error``.
    boundary_radius : int
        Boundary dilation radius used by boundary cases.
    colormap : str
        Matplotlib heatmap colormap.
    overlay_alpha : float
        Heatmap opacity for image overlays.
    epsilon : float
        Numerical stability threshold for CAM calculations.

    Returns
    -------
    List[Dict[str, object]]
        Explanation-manifest rows describing all generated outputs.

    Raises
    ------
    RuntimeError
        If the target-layer hook fails to capture a feature tensor.
    ValueError
        If a model output/target region is incompatible with explanation.
    FileNotFoundError
        If selected source files are missing.
    """
    model.eval()
    layer_name, target_layer = choose_target_layer(
        model,
        layer_scope=layer_scope,
        target_layer_path=target_layer_path,
    )
    print(f"  Target layer: {layer_name}")

    manifest_rows: List[Dict[str, object]] = []
    grouping_columns = ["image_index", "image_path", "mask_path"]

    with ActivationCapture(target_layer) as capture:
        grouped = model_cases.groupby(grouping_columns, sort=False, dropna=False)
        for (image_index_value, image_path_value, mask_path_value), case_group in grouped:
            image_index = int(float(image_index_value))
            image_path = resolve_project_path(str(image_path_value), project_root)
            mask_path = resolve_project_path(str(mask_path_value), project_root)

            image, target = load_single_sample(image_path, mask_path, device)

            # Enable gradients for the model graph even though the checkpoint is
            # in eval mode; only inference-time dropout/BN behavior is disabled.
            with torch.enable_grad():
                logits_raw = model(image)
                if not isinstance(logits_raw, Tensor):
                    raise TypeError("Segmentation model forward output must be a Tensor.")
                logits = cast(Logits, logits_raw)

                if capture.activation is None:
                    raise RuntimeError(
                        f"Target layer '{layer_name}' did not capture an activation."
                    )
                activation = cast(FeatureMap, capture.activation)
                prediction = cast(ClassMap, logits.argmax(dim=1)[0])

                case_rows = [row for _, row in case_group.iterrows()]
                for case_position, row in enumerate(case_rows):
                    selected_class = optional_class_id(row["class_id"])
                    target_class, target_region, target_region_name = (
                        choose_target_class_and_region(
                            prediction=prediction,
                            target=target,
                            category=str(row["category"]),
                            selected_class=selected_class,
                            num_output_classes=int(logits.shape[1]),
                            ignored_class=19,
                            boundary_radius=boundary_radius,
                            region_mode=region_mode,
                        )
                    )

                    target_score = segmentation_target_score(
                        logits=logits,
                        target_class=target_class,
                        region=target_region,
                    )
                    retain_graph = case_position < (len(case_rows) - 1)
                    heatmap, method_used = generate_cam(
                        score=target_score,
                        activation=activation,
                        method=method,
                        output_height=int(image.shape[2]),
                        output_width=int(image.shape[3]),
                        retain_graph=retain_graph,
                        epsilon=epsilon,
                    )

                    heatmap_path, overlay_path, panel_path = case_output_paths(
                        output_dir=output_dir,
                        row=row,
                        target_class=target_class,
                    )
                    title = build_case_title(
                        row=row,
                        selected_class=selected_class,
                        target_class=target_class,
                        target_region_name=target_region_name,
                        method_used=method_used,
                    )

                    input_numpy = tensor_image_to_numpy(image)
                    target_numpy = tensor_class_map_to_numpy(target)
                    prediction_numpy = tensor_class_map_to_numpy(prediction)
                    region_numpy = tensor_boolean_mask_to_numpy(target_region)
                    save_case_visualisations(
                        image=input_numpy,
                        target_map=target_numpy,
                        prediction_map=prediction_numpy,
                        target_region=region_numpy,
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

                    # The manifest captures both original selection metadata and
                    # the actual Grad-CAM target so every heatmap is auditable.
                    manifest_rows.append(
                        {
                            "source_row_index": int(row["source_row_index"]),
                            "source_model_name": str(row["source_model_name"]),
                            "model_name": model_name,
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
                            "target_region": target_region_name,
                            "target_region_pixels": int(target_region.sum().item()),
                            "target_score": float(target_score.detach().item()),
                            "target_layer": layer_name,
                            "requested_method": method,
                            "method_used": method_used,
                            "heatmap_path": str(heatmap_path),
                            "overlay_path": str(overlay_path),
                            "panel_path": str(panel_path),
                        }
                    )

            # Explicitly release per-image autograd tensors before the next image
            # because selected manifests can contain hundreds of cases.
            capture.activation = None
            del image, target, logits, prediction, activation

    return manifest_rows


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for selected-case Grad-CAM generation.

    Steps
    -----
    1. Accept the selected-case manifest and one or more named checkpoints.
    2. Configure optional model/category/class/rank filters to control workload.
    3. Configure target-layer, Grad-CAM method, target-region, and rendering options.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM++ heatmaps for semantic-segmentation cases selected "
            "by explain_choose.py, with automatic Grad-CAM fallback."
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
        "--model",
        action="append",
        default=[],
        help="Optional model-name filter. Repeat to keep multiple checkpoint labels.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Optional selected-case category filter. Repeat to keep multiple categories.",
    )
    parser.add_argument(
        "--target-class",
        action="append",
        default=[],
        help=(
            "Optional selected-class id/name filter. When supplied, classless image-level "
            "rows are omitted. Repeat to keep multiple classes."
        ),
    )
    parser.add_argument(
        "--rank-max",
        type=int,
        default=None,
        help="Optional maximum selection rank retained within each category.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional global cap after all filters, useful for a quick smoke test.",
    )
    parser.add_argument(
        "--method",
        choices=["gradcam++", "gradcam"],
        default="gradcam++",
        help="Requested explanation method. Grad-CAM++ falls back to Grad-CAM if degenerate.",
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
            "Optional explicit dotted module path, e.g. decoder.blocks.4.conv2.0. "
            "When omitted, the final convolution in --layer-scope is used."
        ),
    )
    parser.add_argument(
        "--region-mode",
        choices=["auto", "ground_truth", "prediction", "error"],
        default="auto",
        help="How output pixels are reduced into the scalar Grad-CAM target.",
    )
    parser.add_argument(
        "--boundary-radius",
        type=int,
        default=3,
        help="Boundary dilation radius for boundary-failure target regions. Defaults to 3.",
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
        help="Numerical stability threshold for Grad-CAM calculations. Defaults to 1e-8.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate Grad-CAM++ heatmaps for filtered selected semantic cases.

    Steps
    -----
    1. Load and filter ``selected_cases.csv`` against supplied checkpoint names.
    2. Expand cross-model cases so each supplied model explains the same image.
    3. Load one checkpoint at a time and generate category-aware Grad-CAM++ maps.
    4. Save all explanation metadata into ``gradcam_manifest.csv``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the selected-case CSV, checkpoint, image, or mask cannot be found.
    ValueError
        If filters/options are invalid or no selected cases remain.
    """
    args = parse_arguments()

    if args.boundary_radius < 0:
        raise ValueError("--boundary-radius must be greater than or equal to 0.")
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

    class_filters = resolve_class_filters(args.target_class)
    selected_cases = pd.read_csv(selected_cases_path)
    cases = expand_and_filter_cases(
        selected_cases=selected_cases,
        checkpoint_names=checkpoint_names,
        model_filters=args.model,
        category_filters=args.category,
        class_filters=class_filters,
        rank_max=args.rank_max,
        max_cases=args.max_cases,
    )
    if cases.empty:
        raise ValueError("No selected cases remain after applying the requested filters.")

    output_dir = resolve_project_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("GRAD-CAM++ SEMANTIC-SEGMENTATION EXPLANATION PIPELINE")
    print("=" * 72)
    print(f"Execution device : {Configuration.DEVICE}")
    print(f"Selected cases   : {selected_cases_path}")
    print(f"Cases to explain : {len(cases)}")
    print(f"Requested method : {args.method}")
    print(f"Region mode      : {args.region_mode}")
    print(f"Output directory : {output_dir}")

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
            method=args.method,
            region_mode=args.region_mode,
            boundary_radius=args.boundary_radius,
            colormap=args.colormap,
            overlay_alpha=args.overlay_alpha,
            epsilon=args.epsilon,
        )
        all_manifest_rows.extend(model_rows)

        # Release the checkpoint before loading the next one. The explanation
        # workload uses backward graphs, so explicit cleanup avoids cumulative VRAM.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = pd.DataFrame(all_manifest_rows)
    manifest_path = output_dir / "gradcam_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nGrad-CAM generation complete.")
    print(f"  Explanations generated : {len(manifest)}")
    print(f"  Manifest               : {manifest_path}")
    if not manifest.empty:
        method_counts = manifest["method_used"].value_counts().to_dict()
        print(f"  Methods used           : {method_counts}")


if __name__ == "__main__":
    main()
