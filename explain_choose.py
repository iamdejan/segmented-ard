"""Select representative semantic-segmentation images for XAI analysis.

The program evaluates one or more trained checkpoints on the same BDD100k test
split used by ``evaluate_performance.py``. It stores per-image and per-class
statistics, then selects representative images for attribution and
counterfactual explanation experiments without requiring manual browsing.
"""

import argparse
import math

import shutil
from pathlib import Path as FilePath
from typing import Dict, List, Optional, Sequence, Tuple, cast

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch import Tensor, nn
from torch.utils.data import DataLoader

from evaluate_performance import (
    BDDSegmentationDataset,
    CLASS_NAMES,
    Configuration,
    load_dataset_from_files,
    load_trained_model,
)


Logits = Float[Tensor, "b c h w"]
ClassMap = Int64[Tensor, "h w"]
BooleanMask = Bool[Tensor, "h w"]


@jaxtyped(typechecker=beartype)
def semantic_boundary_mask(
    class_map: ClassMap,
    ignored_class: int,
    radius: int,
) -> BooleanMask:
    """Build a dilated semantic-boundary mask from a class-index map.

    A pixel is treated as a semantic boundary when one of its four direct
    neighbours has a different valid class. The boundary is then dilated so XAI
    case selection can capture mistakes close to class transitions rather than
    requiring an error on the exact one-pixel contour.

    Steps
    -----
    1. Compare horizontally and vertically adjacent valid pixels.
    2. Mark both sides of every class transition as boundary pixels.
    3. Dilate the boundary with max pooling by ``radius`` pixels.

    Parameters
    ----------
    class_map : ClassMap
        Ground-truth class-index tensor of shape ``(H, W)``.
    ignored_class : int
        Class id excluded from boundary construction.
    radius : int
        Number of pixels by which to dilate the one-pixel semantic boundary.

    Returns
    -------
    BooleanMask
        Boolean boundary mask of shape ``(H, W)``.

    Raises
    ------
    ValueError
        If ``radius`` is negative.
    """
    if radius < 0:
        raise ValueError("radius must be greater than or equal to 0.")

    # Start with an empty mask because only genuine class transitions should be
    # scored as boundaries; image borders alone do not represent class errors.
    boundary = torch.zeros_like(class_map, dtype=torch.bool)
    valid = class_map != ignored_class

    # Mark both pixels participating in each horizontal transition. Keeping both
    # sides makes the score symmetric with respect to which class was predicted.
    horizontal_difference = (
        (class_map[:, 1:] != class_map[:, :-1])
        & valid[:, 1:]
        & valid[:, :-1]
    )
    boundary[:, 1:] |= horizontal_difference
    boundary[:, :-1] |= horizontal_difference

    # Repeat the same transition test vertically so all four-neighbour semantic
    # edges are represented before the optional dilation step.
    vertical_difference = (
        (class_map[1:, :] != class_map[:-1, :])
        & valid[1:, :]
        & valid[:-1, :]
    )
    boundary[1:, :] |= vertical_difference
    boundary[:-1, :] |= vertical_difference

    if radius == 0:
        return cast(BooleanMask, boundary)

    # Max pooling is a dependency-free morphological dilation. The odd kernel
    # preserves output size while expanding the boundary by exactly ``radius``.
    kernel_size = (2 * radius) + 1
    dilated = F.max_pool2d(
        boundary.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )
    return cast(BooleanMask, dilated.squeeze(0).squeeze(0).bool())


@jaxtyped(typechecker=beartype)
def class_pair_boundary_mask(
    class_map: ClassMap,
    first_class: int,
    second_class: int,
    ignored_class: int,
    radius: int,
) -> BooleanMask:
    """Build a dilated boundary mask for one semantic class pair.

    This specialised boundary is useful for known confusion pairs such as road
    and sidewalk. Only direct transitions between the two requested classes are
    retained, preventing unrelated object boundaries from dominating the score.

    Steps
    -----
    1. Find horizontal and vertical neighbours equal to the requested pair.
    2. Mark pixels on both sides of each pair-specific transition.
    3. Dilate the resulting contour by ``radius`` pixels.

    Parameters
    ----------
    class_map : ClassMap
        Ground-truth class-index tensor of shape ``(H, W)``.
    first_class : int
        First class id in the confusion pair.
    second_class : int
        Second class id in the confusion pair.
    ignored_class : int
        Class id excluded from scoring.
    radius : int
        Number of pixels by which to dilate the pair boundary.

    Returns
    -------
    BooleanMask
        Boolean pair-boundary mask of shape ``(H, W)``.

    Raises
    ------
    ValueError
        If the class ids are equal or ``radius`` is negative.
    """
    if first_class == second_class:
        raise ValueError("first_class and second_class must be different.")
    if radius < 0:
        raise ValueError("radius must be greater than or equal to 0.")

    boundary = torch.zeros_like(class_map, dtype=torch.bool)
    valid = class_map != ignored_class

    # Check either ordering of the class pair because adjacency is undirected.
    left = class_map[:, :-1]
    right = class_map[:, 1:]
    horizontal_pair = (
        (((left == first_class) & (right == second_class))
         | ((left == second_class) & (right == first_class)))
        & valid[:, :-1]
        & valid[:, 1:]
    )
    boundary[:, :-1] |= horizontal_pair
    boundary[:, 1:] |= horizontal_pair

    top = class_map[:-1, :]
    bottom = class_map[1:, :]
    vertical_pair = (
        (((top == first_class) & (bottom == second_class))
         | ((top == second_class) & (bottom == first_class)))
        & valid[:-1, :]
        & valid[1:, :]
    )
    boundary[:-1, :] |= vertical_pair
    boundary[1:, :] |= vertical_pair

    if radius == 0:
        return cast(BooleanMask, boundary)

    kernel_size = (2 * radius) + 1
    dilated = F.max_pool2d(
        boundary.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )
    expanded = dilated.squeeze(0).squeeze(0).bool()

    # Keep the dilated band centred on the requested pair but score only pixels
    # belonging to that pair. Otherwise a nearby third class could enter the
    # dilated band and inflate the pair-specific error rate.
    pair_pixels = (class_map == first_class) | (class_map == second_class)
    return cast(BooleanMask, expanded & pair_pixels)


@jaxtyped(typechecker=beartype)
def masked_error_rate(
    prediction: ClassMap,
    target: ClassMap,
    region: BooleanMask,
    ignored_class: int,
) -> float:
    """Compute the misclassification rate inside a selected spatial region.

    Steps
    -----
    1. Remove ignored-class target pixels from the supplied region.
    2. Count prediction/target disagreements inside the remaining region.
    3. Return ``NaN`` when the region contains no valid pixels.

    Parameters
    ----------
    prediction : ClassMap
        Predicted class-index tensor of shape ``(H, W)``.
    target : ClassMap
        Ground-truth class-index tensor of shape ``(H, W)``.
    region : BooleanMask
        Boolean mask defining the pixels to score.
    ignored_class : int
        Ground-truth class id excluded from the calculation.

    Returns
    -------
    float
        Fraction of valid region pixels that are incorrect, or ``NaN`` when the
        region is empty.
    """
    valid_region = region & (target != ignored_class)
    region_size = int(valid_region.sum().item())
    if region_size == 0:
        return float("nan")

    errors = ((prediction != target) & valid_region).sum().item()
    return float(errors / region_size)


def safe_rate(numerator: int, denominator: int) -> float:
    """Return a division result while preserving undefined segmentation rates.

    Steps
    -----
    1. Return ``NaN`` when the denominator is zero.
    2. Otherwise divide the integer counts as floating-point values.

    Parameters
    ----------
    numerator : int
        Count forming the numerator of a metric.
    denominator : int
        Count forming the denominator of a metric.

    Returns
    -------
    float
        Ratio of ``numerator`` to ``denominator``, or ``NaN`` if undefined.
    """
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


@jaxtyped(typechecker=beartype)
def score_single_image(
    prediction: ClassMap,
    target: ClassMap,
    image_index: int,
    image_path: str,
    mask_path: str,
    model_name: str,
    num_classes: int,
    ignored_class: int,
    boundary_radius: int,
    road_class: int,
    sidewalk_class: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Compute per-image and per-class statistics used for XAI case selection.

    The function intentionally stores raw confusion counts alongside derived
    rates. This makes each later selection auditable and avoids rerunning the
    model when a different ranking criterion is desired.

    Steps
    -----
    1. Build the valid-pixel mask and dataset-level image accuracy statistics.
    2. Compute class-specific TP/FP/FN, IoU, FP/FN rates, and region area.
    3. Compute generic and road/sidewalk boundary error rates.
    4. Return one image row and one row per evaluated semantic class.

    Parameters
    ----------
    prediction : ClassMap
        Predicted class-index tensor of shape ``(H, W)``.
    target : ClassMap
        Ground-truth class-index tensor of shape ``(H, W)``.
    image_index : int
        Stable row index of the image in the deterministic test split.
    image_path : str
        Filesystem path to the original RGB image.
    mask_path : str
        Filesystem path to the corresponding color-label mask.
    model_name : str
        Human-readable checkpoint identifier.
    num_classes : int
        Number of scored classes, excluding the ignored class.
    ignored_class : int
        Class id omitted from performance scoring.
    boundary_radius : int
        Dilation radius used for boundary-error selection.
    road_class : int
        Class id representing road.
    sidewalk_class : int
        Class id representing sidewalk.

    Returns
    -------
    Tuple[Dict[str, object], List[Dict[str, object]]]
        Image-level metric row and class-level metric rows.
    """
    valid = target != ignored_class
    valid_pixels = int(valid.sum().item())
    correct_pixels = int(((prediction == target) & valid).sum().item())
    pixel_accuracy = safe_rate(correct_pixels, valid_pixels)

    class_rows: List[Dict[str, object]] = []
    class_ious: List[float] = []

    # Compute class statistics explicitly so the saved table contains the exact
    # evidence used later for FP/FN and small-region case selection.
    for class_id in range(num_classes):
        predicted_class = prediction == class_id
        target_class = target == class_id

        tp = int((predicted_class & target_class & valid).sum().item())
        fp = int((predicted_class & ~target_class & valid).sum().item())
        fn = int((~predicted_class & target_class & valid).sum().item())
        gt_area = tp + fn
        pred_area = tp + fp
        union = tp + fp + fn

        class_iou = safe_rate(tp, union)
        fp_rate = safe_rate(fp, pred_area)
        fn_rate = safe_rate(fn, gt_area)
        gt_area_ratio = safe_rate(gt_area, valid_pixels)

        if not math.isnan(class_iou):
            class_ious.append(class_iou)

        class_rows.append(
            {
                "model_name": model_name,
                "image_index": image_index,
                "image_path": image_path,
                "mask_path": mask_path,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "tp_pixels": tp,
                "fp_pixels": fp,
                "fn_pixels": fn,
                "gt_area_pixels": gt_area,
                "pred_area_pixels": pred_area,
                "gt_area_ratio": gt_area_ratio,
                "class_iou": class_iou,
                "fp_rate": fp_rate,
                "fn_rate": fn_rate,
            }
        )

    # Macro-average only classes with a non-empty GT/prediction union. Empty
    # classes are undefined for that image and should not artificially lower it.
    mean_iou = float(np.mean(class_ious)) if class_ious else float("nan")

    boundary = semantic_boundary_mask(target, ignored_class, boundary_radius)
    boundary_error = masked_error_rate(prediction, target, boundary, ignored_class)

    road_sidewalk_boundary = class_pair_boundary_mask(
        target,
        road_class,
        sidewalk_class,
        ignored_class,
        boundary_radius,
    )
    road_sidewalk_boundary_error = masked_error_rate(
        prediction,
        target,
        road_sidewalk_boundary,
        ignored_class,
    )

    image_row: Dict[str, object] = {
        "model_name": model_name,
        "image_index": image_index,
        "image_path": image_path,
        "mask_path": mask_path,
        "valid_pixels": valid_pixels,
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "boundary_error_rate": boundary_error,
        "road_sidewalk_boundary_error_rate": road_sidewalk_boundary_error,
        "boundary_pixels": int(boundary.sum().item()),
        "road_sidewalk_boundary_pixels": int(road_sidewalk_boundary.sum().item()),
    }
    return image_row, class_rows


def parse_checkpoint_spec(specification: str) -> Tuple[str, str]:
    """Parse a checkpoint CLI value into a display name and filesystem path.

    Checkpoints can be supplied as ``NAME=PATH`` for explicit labels or as a
    plain path, in which case the filename stem is used as the model name.

    Steps
    -----
    1. Split ``NAME=PATH`` values at the first equals sign when present.
    2. Fall back to the checkpoint filename stem for plain path values.
    3. Validate that neither the name nor path is empty.

    Parameters
    ----------
    specification : str
        Checkpoint specification from the command line.

    Returns
    -------
    Tuple[str, str]
        ``(model_name, checkpoint_path)``.

    Raises
    ------
    ValueError
        If the specification contains an empty model name or path.
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


def resolve_target_classes(class_specs: Sequence[str], num_classes: int) -> List[int]:
    """Resolve optional class ids/names into unique numeric class ids.

    When no class filter is provided, every scored class is returned. This keeps
    the default selection exhaustive while still allowing focused XAI studies.

    Steps
    -----
    1. Return all classes when ``class_specs`` is empty.
    2. Resolve each value as either an integer id or case-insensitive class name.
    3. Deduplicate ids while preserving their requested order.

    Parameters
    ----------
    class_specs : Sequence[str]
        Class ids or names supplied by ``--target-class``.
    num_classes : int
        Number of valid scored classes.

    Returns
    -------
    List[int]
        Unique resolved class ids.

    Raises
    ------
    ValueError
        If a class id is outside the scored range or a class name is unknown.
    """
    if not class_specs:
        return list(range(num_classes))

    name_to_id = {
        name.lower(): class_id
        for class_id, name in enumerate(CLASS_NAMES[:num_classes])
    }
    resolved: List[int] = []
    for specification in class_specs:
        stripped = specification.strip()
        try:
            class_id = int(stripped)
        except ValueError:
            lookup = stripped.lower()
            if lookup not in name_to_id:
                valid_names = ", ".join(CLASS_NAMES[:num_classes])
                raise ValueError(
                    f"Unknown class '{specification}'. Valid names: {valid_names}."
                ) from None
            class_id = name_to_id[lookup]

        if class_id < 0 or class_id >= num_classes:
            raise ValueError(
                f"Class id {class_id} is outside the scored range [0, {num_classes - 1}]."
            )
        if class_id not in resolved:
            resolved.append(class_id)

    return resolved


def append_ranked_rows(
    selections: List[Dict[str, object]],
    frame: pd.DataFrame,
    category: str,
    score_column: str,
    top_k: int,
    ascending: bool,
    model_name: str,
    class_id: Optional[int] = None,
    notes: str = "",
) -> None:
    """Append the highest-ranked rows from a metric frame to the selection list.

    Steps
    -----
    1. Drop rows whose ranking score is undefined.
    2. Sort by the requested metric and keep at most ``top_k`` cases.
    3. Add a compact selection record retaining image paths and source score.

    Parameters
    ----------
    selections : List[Dict[str, object]]
        Mutable output list receiving selection records.
    frame : pd.DataFrame
        Candidate metric rows containing image and mask paths.
    category : str
        Human-readable selection category.
    score_column : str
        Column used for ranking.
    top_k : int
        Maximum number of rows to append.
    ascending : bool
        Whether smaller metric values rank ahead of larger ones.
    model_name : str
        Model label written into the selection manifest.
    class_id : Optional[int], optional
        Semantic class id for class-specific selections. Defaults to ``None``.
    notes : str, optional
        Explanation of the selection criterion. Defaults to an empty string.

    Returns
    -------
    None
        The function mutates ``selections`` in place.
    """
    if frame.empty:
        return

    ranked = frame.dropna(subset=[score_column]).sort_values(
        score_column,
        ascending=ascending,
        kind="stable",
    ).head(top_k)

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        selected_class_id = class_id
        class_name = ""
        if selected_class_id is not None:
            class_name = CLASS_NAMES[selected_class_id]

        selections.append(
            {
                "category": category,
                "model_name": model_name,
                "class_id": selected_class_id,
                "class_name": class_name,
                "image_index": int(row["image_index"]),
                "image_path": str(row["image_path"]),
                "mask_path": str(row["mask_path"]),
                "score_name": score_column,
                "score": float(row[score_column]),
                "rank": rank,
                "notes": notes,
            }
        )


def select_image_level_cases(
    image_scores: pd.DataFrame,
    top_k: int,
) -> List[Dict[str, object]]:
    """Select high, typical, low, and boundary-focused cases for each model.

    Typical cases are ranked by absolute distance from the model's median image
    mIoU, avoiding an arbitrary fixed IoU interval that may not fit every model.

    Steps
    -----
    1. Split image statistics by model.
    2. Rank high/low mIoU and largest boundary-error cases.
    3. Rank typical cases by proximity to the model-specific median mIoU.

    Parameters
    ----------
    image_scores : pd.DataFrame
        Per-image metric table across one or more models.
    top_k : int
        Number of examples to retain per selection category.

    Returns
    -------
    List[Dict[str, object]]
        Selection-manifest rows.
    """
    selections: List[Dict[str, object]] = []

    for model_name, model_frame in image_scores.groupby("model_name", sort=False):
        model_label = str(model_name)
        append_ranked_rows(
            selections,
            model_frame,
            "high_iou",
            "mean_iou",
            top_k,
            False,
            model_label,
            notes="Highest per-image macro IoU.",
        )
        append_ranked_rows(
            selections,
            model_frame,
            "low_iou",
            "mean_iou",
            top_k,
            True,
            model_label,
            notes="Lowest per-image macro IoU.",
        )
        append_ranked_rows(
            selections,
            model_frame,
            "boundary_failure",
            "boundary_error_rate",
            top_k,
            False,
            model_label,
            notes="Highest prediction error near any ground-truth semantic boundary.",
        )
        append_ranked_rows(
            selections,
            model_frame,
            "road_sidewalk_boundary_failure",
            "road_sidewalk_boundary_error_rate",
            top_k,
            False,
            model_label,
            notes="Highest error near the ground-truth road/sidewalk boundary.",
        )

        valid_iou = model_frame.dropna(subset=["mean_iou"]).copy()
        if not valid_iou.empty:
            median_iou = float(valid_iou["mean_iou"].median())
            valid_iou["distance_from_median_iou"] = (
                valid_iou["mean_iou"] - median_iou
            ).abs()
            append_ranked_rows(
                selections,
                valid_iou,
                "typical_iou",
                "distance_from_median_iou",
                top_k,
                True,
                model_label,
                notes=f"Per-image IoU closest to model median ({median_iou:.6f}).",
            )

    return selections


def select_class_level_cases(
    class_scores: pd.DataFrame,
    target_classes: Sequence[int],
    top_k: int,
    small_region_quantile: float,
) -> List[Dict[str, object]]:
    """Select FP, FN, and small-region success/failure examples by class.

    Semantic masks do not identify object instances, so "small object" is
    intentionally defined as a small total ground-truth class region within an
    image. The area threshold is estimated separately for every model/class from
    the requested quantile of non-empty ground-truth regions.

    Steps
    -----
    1. Filter rows to each requested model/class combination.
    2. Select largest false-positive and false-negative rates.
    3. Estimate a class-specific small-region area threshold.
    4. Within small regions, select best and worst class-IoU cases.

    Parameters
    ----------
    class_scores : pd.DataFrame
        Per-image, per-class segmentation statistics.
    target_classes : Sequence[int]
        Class ids for which class-specific examples should be selected.
    top_k : int
        Number of examples retained per category.
    small_region_quantile : float
        Quantile in ``(0, 1)`` defining small class regions by GT pixel area.

    Returns
    -------
    List[Dict[str, object]]
        Selection-manifest rows.

    Raises
    ------
    ValueError
        If ``small_region_quantile`` is not strictly between zero and one.
    """
    if small_region_quantile <= 0.0 or small_region_quantile >= 1.0:
        raise ValueError("small_region_quantile must be strictly between 0 and 1.")

    selections: List[Dict[str, object]] = []
    for model_name, model_frame in class_scores.groupby("model_name", sort=False):
        model_label = str(model_name)
        for class_id in target_classes:
            class_frame = model_frame[model_frame["class_id"] == class_id].copy()
            if class_frame.empty:
                continue

            # FP rate is meaningful only when the model predicted at least one
            # pixel of the class; undefined rows remain NaN and are dropped.
            append_ranked_rows(
                selections,
                class_frame,
                "highest_fp_rate",
                "fp_rate",
                top_k,
                False,
                model_label,
                class_id,
                notes="Largest fraction of predicted class pixels that are false positives.",
            )

            # FN rate is meaningful only when the class exists in ground truth,
            # making this a direct selector for missed semantic regions.
            append_ranked_rows(
                selections,
                class_frame,
                "highest_fn_rate",
                "fn_rate",
                top_k,
                False,
                model_label,
                class_id,
                notes="Largest fraction of ground-truth class pixels missed by the model.",
            )

            present = class_frame[class_frame["gt_area_pixels"] > 0].copy()
            if present.empty:
                continue

            threshold = float(
                present["gt_area_pixels"].quantile(small_region_quantile)
            )
            small_regions = present[present["gt_area_pixels"] <= threshold].copy()
            area_note = (
                "Small semantic class region defined by the "
                f"{small_region_quantile:.0%} GT-area quantile for this model/class "
                f"(<= {threshold:.1f} resized pixels)."
            )
            append_ranked_rows(
                selections,
                small_regions,
                "small_region_failure",
                "class_iou",
                top_k,
                True,
                model_label,
                class_id,
                notes=area_note + " Lowest class IoU among those regions.",
            )
            append_ranked_rows(
                selections,
                small_regions,
                "small_region_success",
                "class_iou",
                top_k,
                False,
                model_label,
                class_id,
                notes=area_note + " Highest class IoU among those regions.",
            )

    return selections


def select_multi_model_cases(
    image_scores: pd.DataFrame,
    top_k: int,
) -> List[Dict[str, object]]:
    """Select shared successes/failures and largest disagreements across models.

    This selection is only meaningful when at least two checkpoints were scored.
    It compares exactly the same test images so later XAI figures can attribute
    behavioral differences to models rather than different sample choices.

    Steps
    -----
    1. Pivot per-image mIoU into one column per model.
    2. Compute the across-model mean and mIoU range for every image.
    3. Rank shared successes, shared failures, and largest disagreements.

    Parameters
    ----------
    image_scores : pd.DataFrame
        Per-image metric table containing multiple model names.
    top_k : int
        Number of examples retained per category.

    Returns
    -------
    List[Dict[str, object]]
        Cross-model selection-manifest rows, or an empty list for one model.
    """
    model_names = image_scores["model_name"].drop_duplicates().tolist()
    if len(model_names) < 2:
        return []

    metadata = image_scores.drop_duplicates(subset=["image_index"])[
        ["image_index", "image_path", "mask_path"]
    ].set_index("image_index")
    pivot = image_scores.pivot(
        index="image_index",
        columns="model_name",
        values="mean_iou",
    )
    complete = pivot.dropna(axis=0, how="any").copy()
    if complete.empty:
        return []

    complete["mean_across_models"] = complete.mean(axis=1)
    model_columns = [column for column in complete.columns if column != "mean_across_models"]
    complete["iou_range_across_models"] = (
        complete[model_columns].max(axis=1) - complete[model_columns].min(axis=1)
    )
    complete = complete.join(metadata, how="left").reset_index()

    selections: List[Dict[str, object]] = []
    append_ranked_rows(
        selections,
        complete,
        "shared_high_iou",
        "mean_across_models",
        top_k,
        False,
        "all_models",
        notes="Highest mean per-image IoU across all supplied checkpoints.",
    )
    append_ranked_rows(
        selections,
        complete,
        "shared_low_iou",
        "mean_across_models",
        top_k,
        True,
        "all_models",
        notes="Lowest mean per-image IoU across all supplied checkpoints.",
    )
    append_ranked_rows(
        selections,
        complete,
        "model_disagreement",
        "iou_range_across_models",
        top_k,
        False,
        "all_models",
        notes="Largest max-minus-min per-image IoU gap across supplied checkpoints.",
    )
    return selections


def build_unique_image_manifest(selections: pd.DataFrame) -> pd.DataFrame:
    """Collapse category-level selections into one row per unique image.

    Steps
    -----
    1. Group selected rows by stable image identity and source paths.
    2. Join all categories and model labels associated with each image.
    3. Count category occurrences so highly informative images are easy to spot.

    Parameters
    ----------
    selections : pd.DataFrame
        Full category-level selection manifest.

    Returns
    -------
    pd.DataFrame
        One row per unique selected image with aggregated category metadata.
    """
    if selections.empty:
        return pd.DataFrame(
            columns=[
                "image_index",
                "image_path",
                "mask_path",
                "selection_count",
                "categories",
                "models",
            ]
        )

    grouped = selections.groupby(
        ["image_index", "image_path", "mask_path"],
        sort=True,
        as_index=False,
    ).agg(
        selection_count=("category", "size"),
        categories=("category", lambda values: ";".join(sorted(set(map(str, values))))),
        models=("model_name", lambda values: ";".join(sorted(set(map(str, values))))),
    )
    return cast(pd.DataFrame, grouped)


def copy_selected_files(
    unique_images: pd.DataFrame,
    destination: str,
) -> None:
    """Copy unique selected images and masks into an optional review directory.

    The CSV paths remain the canonical source of truth. Copies are organised in
    separate ``images`` and ``masks`` folders only to make manual XAI review more
    convenient without altering the original dataset.

    Steps
    -----
    1. Create destination subdirectories for images and masks.
    2. Copy every unique selected image and mask while preserving filenames.
    3. Reject filename collisions originating from different source paths.

    Parameters
    ----------
    unique_images : pd.DataFrame
        Unique-image selection manifest.
    destination : str
        Root directory receiving ``images`` and ``masks`` subdirectories.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If a selected source image or mask no longer exists.
    FileExistsError
        If two different source files would overwrite the same destination name.
    """
    image_dir = FilePath(destination) / "images"
    mask_dir = FilePath(destination) / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    copied_sources: Dict[str, str] = {}
    for _, row in unique_images.iterrows():
        for source_value, target_dir in (
            (str(row["image_path"]), image_dir),
            (str(row["mask_path"]), mask_dir),
        ):
            source = FilePath(source_value)
            if not source.is_file():
                raise FileNotFoundError(f"Selected source file does not exist: {source}")

            target = target_dir / source.name
            target_key = str(target.resolve())
            previous_source = copied_sources.get(target_key)
            if previous_source is not None and previous_source != str(source.resolve()):
                raise FileExistsError(
                    f"Destination collision for '{target.name}' from '{previous_source}' "
                    f"and '{source}'."
                )

            if previous_source is None:
                shutil.copy2(source, target)
                copied_sources[target_key] = str(source.resolve())


def evaluate_checkpoint(
    model: nn.Module,
    dataloader: DataLoader[Tuple[Tensor, Tensor]],
    test_df: pd.DataFrame,
    model_name: str,
    device: torch.device,
    num_classes: int,
    ignored_class: int,
    boundary_radius: int,
    road_class: int,
    sidewalk_class: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Run one inference pass and collect all case-selection statistics.

    The DataLoader must be deterministic (``shuffle=False``). Batch offsets are
    then mapped directly back to ``test_df`` rows, avoiding any need to modify
    the existing dataset class merely to return file paths.

    Steps
    -----
    1. Run the checkpoint in evaluation/inference mode over each test batch.
    2. Collapse logits and one-hot targets to class-index maps.
    3. Map each batch item back to its deterministic test-set file paths.
    4. Compute and accumulate per-image and per-class selection statistics.

    Parameters
    ----------
    model : nn.Module
        Trained semantic-segmentation checkpoint.
    dataloader : DataLoader[Tuple[Tensor, Tensor]]
        Deterministically ordered test DataLoader.
    test_df : pd.DataFrame
        Test split containing ``image_paths`` and ``mask_paths``.
    model_name : str
        Human-readable checkpoint label.
    device : torch.device
        Device used for inference.
    num_classes : int
        Number of scored semantic classes.
    ignored_class : int
        Class id excluded from performance scoring.
    boundary_radius : int
        Pixel radius for boundary-error regions.
    road_class : int
        Road class id for pair-boundary analysis.
    sidewalk_class : int
        Sidewalk class id for pair-boundary analysis.

    Returns
    -------
    Tuple[List[Dict[str, object]], List[Dict[str, object]]]
        Image-level rows and class-level rows for the checkpoint.

    Raises
    ------
    RuntimeError
        If the deterministic DataLoader yields a different number of samples
        than ``test_df`` contains.
    """
    model.eval()
    image_rows: List[Dict[str, object]] = []
    class_rows: List[Dict[str, object]] = []
    sample_offset = 0

    with torch.inference_mode():
        for images, masks in dataloader:
            # Inference uses the exact resized tensors produced by the existing
            # dataset so selection scores match the model's evaluation domain.
            images = images.to(device)
            masks = masks.to(device)
            logits = cast(Logits, model(images))
            predictions = logits.argmax(dim=1)
            targets = masks.argmax(dim=1)

            batch_size = int(images.shape[0])
            for batch_index in range(batch_size):
                image_index = sample_offset + batch_index
                if image_index >= len(test_df):
                    raise RuntimeError(
                        "DataLoader yielded more samples than the deterministic test frame."
                    )

                row = test_df.iloc[image_index]
                image_row, per_class_rows = score_single_image(
                    cast(ClassMap, predictions[batch_index]),
                    cast(ClassMap, targets[batch_index]),
                    image_index,
                    str(row["image_paths"]),
                    str(row["mask_paths"]),
                    model_name,
                    num_classes,
                    ignored_class,
                    boundary_radius,
                    road_class,
                    sidewalk_class,
                )
                image_rows.append(image_row)
                class_rows.extend(per_class_rows)

            sample_offset += batch_size

    if sample_offset != len(test_df):
        raise RuntimeError(
            f"DataLoader yielded {sample_offset} samples, but test_df contains {len(test_df)}."
        )

    return image_rows, class_rows


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for automated XAI image selection.

    Steps
    -----
    1. Accept one or more checkpoint specifications as ``PATH`` or ``NAME=PATH``.
    2. Configure evaluation batch size and boundary/small-region criteria.
    3. Configure output paths and optional copying of selected source files.

    Returns
    -------
    argparse.Namespace
        Parsed command-line values.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score BDD100k test images and select representative cases for "
            "semantic-segmentation XAI."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help=(
            "Checkpoint to evaluate. Repeat for multiple models. Accepts PATH or "
            "NAME=PATH, e.g. --checkpoint teacher=model/teacher.pt."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Configuration.BATCH_SIZE,
        help="Inference batch size. Defaults to the evaluation configuration value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of cases retained per category. Defaults to 1.",
    )
    parser.add_argument(
        "--target-class",
        action="append",
        default=[],
        help=(
            "Class id or name for class-specific FP/FN/small-region selection. "
            "Repeat as needed. Defaults to all scored classes."
        ),
    )
    parser.add_argument(
        "--boundary-radius",
        type=int,
        default=3,
        help="Boundary dilation radius in resized pixels. Defaults to 3.",
    )
    parser.add_argument(
        "--small-region-quantile",
        type=float,
        default=0.25,
        help=(
            "GT-area quantile defining a small semantic class region per model/class. "
            "Defaults to 0.25."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./model/xai_selection",
        help="Directory receiving score and selection CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate checkpoints and write reproducible XAI image-selection manifests.

    Steps
    -----
    1. Recreate the held-out BDD100k test split and deterministic inference loader.
    2. Evaluate each supplied checkpoint once and save raw per-image/class scores.
    3. Select image-level, class-level, and optional cross-model representative cases.
    4. Save category-level and unique-image manifests for downstream XAI scripts.
    5. Optionally copy selected source files into a convenient review directory.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If CLI selection parameters or class/checkpoint specifications are invalid.
    """
    args = parse_arguments()

    # Fail early on invalid controls so a long inference pass cannot finish only
    # to discover that the requested selection configuration was impossible.
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if args.boundary_radius < 0:
        raise ValueError("--boundary-radius must be greater than or equal to 0.")
    if args.small_region_quantile <= 0.0 or args.small_region_quantile >= 1.0:
        raise ValueError("--small-region-quantile must be strictly between 0 and 1.")

    num_classes = 19
    ignored_class = 19
    road_class = 0
    sidewalk_class = 1
    target_classes = resolve_target_classes(args.target_class, num_classes)

    checkpoint_specs = [parse_checkpoint_spec(spec) for spec in args.checkpoint]
    checkpoint_names = [name for name, _ in checkpoint_specs]
    if len(checkpoint_names) != len(set(checkpoint_names)):
        raise ValueError("Every --checkpoint must have a unique model name.")

    output_dir = FilePath(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("XAI IMAGE-SELECTION PIPELINE")
    print("=" * 72)
    print(f"Execution device: {Configuration.DEVICE}")
    print(f"Selection output: {output_dir}")

    # Reuse the exact split function from performance evaluation so selected XAI
    # examples are guaranteed to come from the held-out test partition.
    print("Loading deterministic BDD100k test split...")
    _, _, test_df = load_dataset_from_files()
    test_df = test_df.reset_index(drop=True)
    print(f"Loaded {len(test_df)} test images.")

    inference_transforms = A.Compose(
        [
            A.Resize(
                height=Configuration.IMAGE_HEIGHT,
                width=Configuration.IMAGE_WIDTH,
            ),
            ToTensorV2(),
        ]
    )
    test_dataset = BDDSegmentationDataset(test_df, transform=inference_transforms)
    test_loader: DataLoader[Tuple[Tensor, Tensor]] = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=Configuration.NUM_WORKERS,
    )

    all_image_rows: List[Dict[str, object]] = []
    all_class_rows: List[Dict[str, object]] = []

    for model_name, checkpoint_path in checkpoint_specs:
        print(f"\nScoring model '{model_name}' from {checkpoint_path}...")
        model = load_trained_model(checkpoint_path, Configuration.DEVICE)
        image_rows, class_rows = evaluate_checkpoint(
            model,
            test_loader,
            test_df,
            model_name,
            Configuration.DEVICE,
            num_classes,
            ignored_class,
            args.boundary_radius,
            road_class,
            sidewalk_class,
        )
        all_image_rows.extend(image_rows)
        all_class_rows.extend(class_rows)

        # Release the current checkpoint before loading the next one so comparing
        # multiple models does not make GPU memory grow with checkpoint count.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    image_scores = pd.DataFrame(all_image_rows)
    class_scores = pd.DataFrame(all_class_rows)

    image_scores_path = output_dir / "image_scores.csv"
    class_scores_path = output_dir / "class_scores.csv"
    image_scores.to_csv(image_scores_path, index=False)
    class_scores.to_csv(class_scores_path, index=False)

    # Selection happens entirely from saved scalar statistics, so changing top-k
    # logic later never requires another expensive model inference pass.
    selections = select_image_level_cases(image_scores, args.top_k)
    selections.extend(
        select_class_level_cases(
            class_scores,
            target_classes,
            args.top_k,
            args.small_region_quantile,
        )
    )
    selections.extend(select_multi_model_cases(image_scores, args.top_k))

    selection_frame = pd.DataFrame(selections)
    selected_cases_path = output_dir / "selected_cases.csv"
    selection_frame.to_csv(selected_cases_path, index=False)

    unique_images = build_unique_image_manifest(selection_frame)
    unique_images_path = output_dir / "selected_unique_images.csv"
    unique_images.to_csv(unique_images_path, index=False)

    if args.copy_selected:
        copy_destination = str(output_dir / "selected_files")
        copy_selected_files(unique_images, copy_destination)
        print(f"Copied unique selected files to: {copy_destination}")

    print("\nSelection complete.")
    print(f"  Per-image scores       : {image_scores_path}")
    print(f"  Per-class scores       : {class_scores_path}")
    print(f"  Category selections    : {selected_cases_path}")
    print(f"  Unique selected images : {unique_images_path}")
    print(f"  Unique images selected : {len(unique_images)}")


if __name__ == "__main__":
    main()
