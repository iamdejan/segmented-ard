"""Generate clean or PGD-robust Grad-CAM++ explanations for selected cases.

The program consumes the ``selected_cases.csv`` manifest written by
``explain_choose.py`` and generates class/region-targeted Grad-CAM++ heatmaps for
the corresponding trained segmentation checkpoints. Inputs can remain clean,
be perturbed with the PGD attack shared by ``evaluate_robustness.py``, or include
both variants in one run. Grad-CAM is used as a numerical fallback when a
Grad-CAM++ map is undefined or degenerate.
"""

import argparse
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
from jaxtyping import Bool, Float, Int64, jaxtyped
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
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
BooleanMask = Bool[Tensor, "h w"]
NumpyImage = Float[np.ndarray, "h w 3"]
NumpyCamBatch = Float[np.ndarray, "b h w"]
NumpyHeatmap = Float[np.ndarray, "h w"]
NumpyClassMap = Int64[np.ndarray, "h w"]
NumpyBooleanMask = Bool[np.ndarray, "h w"]


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


class MeanRegionSegmentationTarget:
    """Select the mean logit of one semantic class inside a spatial region.

    ``pytorch-grad-cam`` accepts callable model targets for non-classification
    tasks. This target preserves the repository's previous scoring semantics:
    explain the mean target-class logit over the category-specific region rather
    than changing the score definition to the library's built-in summed mask.

    Steps
    -----
    1. Validate and store the target class and non-empty boolean region.
    2. Move the stored region to the model-output device when called.
    3. Return the mean target-class logit covered by the region.

    Parameters
    ----------
    target_class : int
        Semantic class channel to explain.
    region : BooleanMask
        Boolean mask of shape ``(H, W)`` defining the explained output region.

    Raises
    ------
    ValueError
        If ``target_class`` is negative or ``region`` is empty.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self, target_class: int, region: BooleanMask) -> None:
        """Store one validated semantic-segmentation CAM target.

        Steps
        -----
        1. Reject negative class ids and empty target regions early.
        2. Detach and clone the region so later tensor reuse cannot mutate it.

        Parameters
        ----------
        target_class : int
            Semantic class channel to explain.
        region : BooleanMask
            Boolean target region of shape ``(H, W)``.

        Raises
        ------
        ValueError
            If ``target_class`` is negative or ``region`` is empty.
        """
        if target_class < 0:
            raise ValueError("Grad-CAM target class must be non-negative.")
        if not bool(region.any().item()):
            raise ValueError("Grad-CAM target region must contain at least one pixel.")

        # Keep the target independent from the image-selection tensors because the
        # CAM library performs a separate differentiable model forward pass.
        self.target_class = target_class
        self.region = cast(BooleanMask, region.detach().clone())

    @jaxtyped(typechecker=beartype)
    def __call__(self, model_output: SingleLogits) -> Scalar:
        """Reduce one segmentation output to the scalar score explained by CAM.

        Steps
        -----
        1. Validate the target channel and spatial size against model output.
        2. Move the boolean region to the model output device.
        3. Return the mean target-class logit inside that region.

        Parameters
        ----------
        model_output : SingleLogits
            Per-image segmentation logits with shape ``(C, H, W)``.

        Returns
        -------
        Scalar
            Mean target-class logit over the stored region.

        Raises
        ------
        ValueError
            If the class id is outside the model output range or the region shape
            differs from the output spatial dimensions.
        """
        if self.target_class >= model_output.shape[0]:
            raise ValueError(
                f"Target class {self.target_class} is outside logits range "
                f"[0, {model_output.shape[0] - 1}]."
            )
        if tuple(self.region.shape) != tuple(model_output.shape[1:]):
            raise ValueError(
                "Grad-CAM target region shape does not match model output: "
                f"{tuple(self.region.shape)} vs {tuple(model_output.shape[1:])}."
            )

        # Moving only the mask keeps the target compatible with whichever device
        # pytorch-grad-cam derives from the checkpoint parameters.
        region = self.region.to(device=model_output.device)
        score = model_output[self.target_class][region].mean()
        return cast(Scalar, score)


@jaxtyped(typechecker=beartype)
def run_cam_algorithm(
    cam: Callable[..., NumpyCamBatch],
    image: BatchImage,
    target: MeanRegionSegmentationTarget,
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
    target : MeanRegionSegmentationTarget
        Scalar semantic-segmentation target used by the CAM library.

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

    This check retains the script's previous Grad-CAM fallback policy without
    reimplementing either CAM algorithm. A Grad-CAM++ map that is entirely zero,
    non-finite, or below the configured threshold is retried with Grad-CAM.

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
    target: MeanRegionSegmentationTarget,
    method: str,
    epsilon: float,
) -> Tuple[NumpyHeatmap, str]:
    """Generate a Grad-CAM/Grad-CAM++ map with ``pytorch-grad-cam``.

    The established library now owns the algorithm implementation, including
    activation/gradient hooks, channel weighting, image-space resizing, and
    normalisation. The repository keeps only its existing policy of falling back
    from a degenerate Grad-CAM++ result to standard Grad-CAM.

    Steps
    -----
    1. Validate the requested method.
    2. Run the matching ``pytorch-grad-cam`` implementation on the target layer.
    3. When Grad-CAM++ is degenerate, retry the same target with Grad-CAM.
    4. Return standard Grad-CAM even when it is degenerate so one difficult case
       cannot abort the remaining explanation pipeline.

    Parameters
    ----------
    model : nn.Module
        Loaded semantic-segmentation model.
    target_layer : nn.Module
        Model layer whose activations are used by the CAM library.
    image : BatchImage
        Input image tensor with shape ``(1, 3, H, W)``.
    target : MeanRegionSegmentationTarget
        Class/region scalar target supplied to the CAM implementation.
    method : str
        Requested method: ``gradcam++`` or ``gradcam``.
    epsilon : float
        Threshold used only to detect degenerate library-normalised CAMs.

    Returns
    -------
    Tuple[NumpyHeatmap, str]
        Normalised image-resolution heatmap and actual method used.

    Raises
    ------
    ValueError
        If ``method`` is unsupported or ``epsilon`` is not positive.
    Notes
    -----
    A standard Grad-CAM map can legitimately become all-zero after the method's
    final ReLU when the selected layer contains no positive evidence for the
    requested class/region target. This is recorded in ``method_used`` instead of
    being treated as a fatal error, matching the resilience of the previous local
    implementation and allowing later cases/checkpoints to continue.
    """
    if method not in {"gradcam++", "gradcam"}:
        raise ValueError("method must be either 'gradcam++' or 'gradcam'.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than 0.")

    if method == "gradcam++":
        # Use the maintained library implementation instead of reproducing the
        # Grad-CAM++ alpha/gradient weighting equations in this repository.
        with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
            heatmap = run_cam_algorithm(cam=cam, image=image, target=target)
        if cam_is_usable(heatmap, epsilon):
            return heatmap, "gradcam++"

        # Preserve the previous numerical fallback behavior, but delegate the
        # fallback algorithm itself to the same well-established library.
        with GradCAM(model=model, target_layers=[target_layer]) as fallback_cam:
            fallback_heatmap = run_cam_algorithm(
                cam=fallback_cam,
                image=image,
                target=target,
            )
        if cam_is_usable(fallback_heatmap, epsilon):
            return fallback_heatmap, "gradcam_fallback"

        # A zero Grad-CAM is still a valid algorithm result: the final ReLU can
        # remove all evidence when every weighted activation is non-positive.
        # Do not let that numerical/semantic edge case terminate the checkpoint;
        # preserve the zero heatmap and make its status explicit in the manifest.
        return fallback_heatmap, "gradcam_fallback_degenerate"

    # Standard Grad-CAM is delegated directly to pytorch-grad-cam as well, so no
    # local gradient pooling or activation weighting remains in this script.
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        heatmap = run_cam_algorithm(cam=cam, image=image, target=target)
    if cam_is_usable(heatmap, epsilon):
        return heatmap, "gradcam"

    # Keep processing later cases rather than raising for an all-zero standard
    # Grad-CAM. The explicit method label keeps the condition auditable in CSV.
    return heatmap, "gradcam_degenerate"


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
            targets = [name for name in checkpoint_names if name in checkpoint_set]
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
    target_region_name: str,
    method_used: str,
    input_mode: str,
) -> str:
    """Build a compact, self-describing title for one explanation figure.

    Steps
    -----
    1. Resolve selected and actual target class ids into readable names.
    2. Combine model, input mode, category, region, and method metadata into one title.

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
    input_mode: str,
) -> List[Dict[str, object]]:
    """Generate Grad-CAM explanations for every selected case of one model.

    A cheap inference pass is reused per image and input variant to select the
    prediction-dependent class/region target. Robust inputs are generated with
    the shared PGD attack from ``evaluate_robustness.py`` while retaining the
    original dataset mask. Each selected case is then passed to ``pytorch-grad-cam``,
    which performs the differentiable forward/backward pass needed by its own CAM
    hooks and algorithm implementation.

    Steps
    -----
    1. Discover the model target layer used by the CAM library.
    2. Group selected rows by image/mask pair and load the clean dataset sample once.
    3. Build the requested clean/robust input variants, using shared PGD for robust.
    4. Run inference for each variant and choose each case's class/region target.
    5. Generate the requested library CAM, save figures, and return manifest rows.

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
        Threshold used to detect degenerate library CAM outputs.
    input_mode : str
        Requested inputs: ``clean``, ``robust``, or ``all``.

    Returns
    -------
    List[Dict[str, object]]
        Explanation-manifest rows describing all generated outputs.

    Raises
    ------
    RuntimeError
        If ``pytorch-grad-cam`` returns an unusable heatmap.
    TypeError
        If the segmentation model does not return a tensor.
    ValueError
        If ``input_mode`` is invalid or a model output/target region is incompatible
        with explanation.
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

    manifest_rows: List[Dict[str, object]] = []
    grouping_columns = ["image_index", "image_path", "mask_path"]
    grouped = model_cases.groupby(grouping_columns, sort=False, dropna=False)

    for (image_index_value, image_path_value, mask_path_value), case_group in grouped:
        image_index = int(float(image_index_value))
        image_path = resolve_project_path(str(image_path_value), project_root)
        mask_path = resolve_project_path(str(mask_path_value), project_root)

        clean_image, target = load_single_sample(image_path, mask_path, device)

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
            # Target selection depends on the prediction for the concrete input.
            # No graph is needed here because pytorch-grad-cam runs its own forward.
            with torch.no_grad():
                logits_raw = model(image)
            if not isinstance(logits_raw, Tensor):
                raise TypeError("Segmentation model forward output must be a Tensor.")
            logits = cast(Logits, logits_raw)
            prediction = cast(ClassMap, logits.argmax(dim=1)[0])

            for _, row in case_group.iterrows():
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

                # The custom target keeps the old mean-over-region semantics while
                # allowing the external library to own all CAM-specific computation.
                cam_target = MeanRegionSegmentationTarget(
                    target_class=target_class,
                    region=target_region,
                )
                target_score = cam_target(cast(SingleLogits, logits[0]))
                heatmap, method_used = generate_cam(
                    model=model,
                    target_layer=target_layer,
                    image=image,
                    target=cam_target,
                    method=method,
                    epsilon=epsilon,
                )

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
                    target_region_name=target_region_name,
                    method_used=method_used,
                    input_mode=concrete_input_mode,
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
    2. Configure optional model/category/class/rank filters to control workload.
    3. Choose clean, PGD-robust, or combined input processing.
    4. Configure target-layer, Grad-CAM method, target-region, and rendering options.

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
        "--input-mode",
        choices=["clean", "robust", "all"],
        default="clean",
        help=(
            "Input variant to explain: clean dataset images, robust images perturbed "
            "with evaluate_robustness.py PGD defaults, or both. Defaults to clean."
        ),
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
        help="Threshold for detecting degenerate library CAMs. Defaults to 1e-8.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate clean and/or PGD-robust Grad-CAM++ heatmaps for selected cases.

    Steps
    -----
    1. Load and filter ``selected_cases.csv`` against supplied checkpoint names.
    2. Expand cross-model cases so each supplied model explains the same image.
    3. Load one checkpoint at a time and generate the requested clean/robust inputs.
    4. Generate category-aware Grad-CAM++ maps for each concrete input variant.
    5. Append each completed checkpoint's metadata to ``gradcam_manifest.csv``.

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
    print(f"Input mode       : {args.input_mode}")
    print(f"Requested method : {args.method}")
    print(f"Region mode      : {args.region_mode}")
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
            method=args.method,
            region_mode=args.region_mode,
            boundary_radius=args.boundary_radius,
            colormap=args.colormap,
            overlay_alpha=args.overlay_alpha,
            epsilon=args.epsilon,
            input_mode=args.input_mode,
        )
        all_manifest_rows.extend(model_rows)

        # Persist each completed checkpoint immediately. If a later checkpoint
        # runs out of memory, the work that already finished remains recorded and
        # a subsequent invocation can safely append the remaining checkpoints.
        model_manifest = pd.DataFrame(model_rows)
        if not model_manifest.empty:
            manifest_exists = (
                manifest_path.is_file() and manifest_path.stat().st_size > 0
            )
            model_manifest.to_csv(
                manifest_path,
                mode="a",
                header=not manifest_exists,
                index=False,
            )

        # Release the checkpoint before loading the next one. The explanation
        # workload uses backward graphs, so explicit cleanup avoids cumulative VRAM.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = pd.DataFrame(all_manifest_rows)

    print("\nGrad-CAM generation complete.")
    print(f"  Explanations generated : {len(manifest)}")
    print(f"  Manifest               : {manifest_path}")
    if not manifest.empty:
        method_counts = manifest["method_used"].value_counts().to_dict()
        print(f"  Methods used           : {method_counts}")


if __name__ == "__main__":
    main()
