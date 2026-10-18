"""Select representative BDD100k test cases for later XAI analysis.

This module performs one lightweight inference pass over the deterministic test
split used by ``evaluate_performance.py``. It records per-image segmentation
statistics, writes them to CSV, and automatically selects representative images
covering strong predictions and several common failure modes.
"""

import argparse
import os
from typing import Dict, List, Tuple, cast

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch import nn
from torch.utils.data import DataLoader

from evaluate_performance import (
    BDDSegmentationDataset,
    CLASS_NAMES,
    Configuration,
    ImageTensor,
    MaskTensor,
    forward,
    load_dataset_from_files,
    load_trained_model,
)


# Shape aliases make the image-level metric helpers fail early when a caller
# accidentally supplies a batched tensor or an array with the wrong layout.
ClassIndexArray = Int64[np.ndarray, "h w"]
BinaryMask = Bool[np.ndarray, "h w"]
ClassIoUArray = Float[np.ndarray, "classes"]

# The main evaluation script excludes the final "unknown" class from metrics.
# Reusing the same convention keeps representative-case ranking comparable.
IGNORED_CLASS_ID = 19
NUM_SCORED_CLASSES = 19

# These class ids correspond to the three explicit class columns requested for
# the per-image CSV. The complete class list remains available via CLASS_NAMES.
ROAD_CLASS_ID = 0
SIDEWALK_CLASS_ID = 1
PERSON_CLASS_ID = 11


@jaxtyped(typechecker=beartype)
def compute_class_iou(
    prediction: ClassIndexArray,
    target: ClassIndexArray,
    valid_mask: BinaryMask,
    num_classes: int,
) -> ClassIoUArray:
    """Compute per-class IoU for one semantic-segmentation image.

    Classes with neither ground-truth nor predicted pixels have an undefined IoU
    for that image and are stored as ``NaN``. This prevents absent classes from
    artificially improving the per-image mean IoU.

    Steps
    -----
    1. Restrict all counts to pixels allowed by ``valid_mask``.
    2. Compute true-positive, false-positive, and false-negative counts per class.
    3. Return IoU for classes with a non-zero union and ``NaN`` otherwise.

    Parameters
    ----------
    prediction : np.ndarray
        Predicted class ids with shape ``(H, W)``.
    target : np.ndarray
        Ground-truth class ids with shape ``(H, W)``.
    valid_mask : np.ndarray
        Boolean mask indicating pixels that should contribute to metrics.
    num_classes : int
        Number of scored classes, excluding the ignored class.

    Returns
    -------
    np.ndarray
        Float array of shape ``(num_classes,)`` containing per-class IoU values.

    Raises
    ------
    ValueError
        If ``num_classes`` is not positive.
    """
    # Reject invalid class counts explicitly because an empty IoU vector would
    # make the downstream mean-IoU calculation ambiguous.
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    # Start with NaNs so classes absent from both prediction and target are
    # excluded naturally by np.nanmean instead of being treated as perfect.
    class_iou = np.full(num_classes, np.nan, dtype=np.float64)

    # Compute one confusion triplet per class. This loop is small (19 classes),
    # while each NumPy boolean expression remains vectorized over all pixels.
    for class_id in range(num_classes):
        pred_class = prediction == class_id
        target_class = target == class_id

        true_positive = np.count_nonzero(pred_class & target_class & valid_mask)
        false_positive = np.count_nonzero(pred_class & ~target_class & valid_mask)
        false_negative = np.count_nonzero(~pred_class & target_class & valid_mask)
        union = true_positive + false_positive + false_negative

        # Leave IoU undefined when the class does not occur in either map.
        if union > 0:
            class_iou[class_id] = true_positive / union

    return class_iou


@jaxtyped(typechecker=beartype)
def build_boundary_band(
    target: ClassIndexArray,
    valid_mask: BinaryMask,
    radius: int,
) -> BinaryMask:
    """Build a dilated band around ground-truth semantic boundaries.

    A semantic boundary is marked when two horizontally or vertically adjacent
    valid pixels have different ground-truth class ids. The thin boundary is then
    dilated so nearby localization errors are counted as boundary errors rather
    than only errors on the exact one-pixel transition.

    Steps
    -----
    1. Detect horizontal and vertical class transitions between valid pixels.
    2. Mark both pixels adjacent to every detected transition.
    3. Dilate the boundary by ``radius`` pixels using an elliptical kernel.

    Parameters
    ----------
    target : np.ndarray
        Ground-truth class ids with shape ``(H, W)``.
    valid_mask : np.ndarray
        Boolean mask identifying pixels that are included in evaluation.
    radius : int
        Dilation radius in pixels. A value of ``0`` keeps the one-pixel boundary.

    Returns
    -------
    np.ndarray
        Boolean boundary-band mask with shape ``(H, W)``.

    Raises
    ------
    ValueError
        If ``radius`` is negative.
    """
    # A negative radius has no geometric meaning, so fail before constructing an
    # invalid OpenCV kernel.
    if radius < 0:
        raise ValueError("radius must be non-negative.")

    # Mark both sides of every valid horizontal class transition. Keeping both
    # pixels makes the boundary symmetric with respect to the two neighboring classes.
    boundary = np.zeros_like(valid_mask, dtype=np.bool_)
    horizontal_change = (
        (target[:, 1:] != target[:, :-1])
        & valid_mask[:, 1:]
        & valid_mask[:, :-1]
    )
    boundary[:, 1:] |= horizontal_change
    boundary[:, :-1] |= horizontal_change

    # Repeat the same operation vertically so the boundary mask covers all
    # four-connected semantic transitions.
    vertical_change = (
        (target[1:, :] != target[:-1, :])
        & valid_mask[1:, :]
        & valid_mask[:-1, :]
    )
    boundary[1:, :] |= vertical_change
    boundary[:-1, :] |= vertical_change

    # Radius zero is useful for debugging and avoids an unnecessary OpenCV call.
    if radius == 0:
        return boundary

    # Dilate into a local boundary neighborhood. An elliptical kernel reduces
    # directional bias compared with a purely square neighborhood.
    kernel_size = (2 * radius) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    dilated = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1)
    return cast(BinaryMask, dilated.astype(np.bool_))


@jaxtyped(typechecker=beartype)
def compute_boundary_metrics(
    prediction: ClassIndexArray,
    target: ClassIndexArray,
    valid_mask: BinaryMask,
    radius: int,
) -> Tuple[float, int, float, int]:
    """Measure how strongly an image's segmentation errors occur near boundaries.

    Two complementary values are returned. ``boundary_error_rate`` measures the
    fraction of pixels in the ground-truth boundary band that are misclassified,
    while ``boundary_error_concentration`` measures what fraction of all image
    errors lie inside that band.

    Steps
    -----
    1. Construct the dilated ground-truth boundary band.
    2. Count all valid misclassified pixels and boundary-band misclassifications.
    3. Normalize by boundary area and total error count respectively.

    Parameters
    ----------
    prediction : np.ndarray
        Predicted class ids with shape ``(H, W)``.
    target : np.ndarray
        Ground-truth class ids with shape ``(H, W)``.
    valid_mask : np.ndarray
        Boolean mask identifying pixels included in evaluation.
    radius : int
        Boundary dilation radius in pixels.

    Returns
    -------
    tuple of float, int, float, int
        ``(boundary_error_rate, boundary_error_pixels,
        boundary_error_concentration, total_error_pixels)``.

    Raises
    ------
    ValueError
        If ``radius`` is negative, propagated from :func:`build_boundary_band`.
    """
    # The boundary band is based only on ground truth so the diagnostic measures
    # prediction quality around a stable reference contour.
    boundary_band = build_boundary_band(target, valid_mask, radius)
    error_mask = (prediction != target) & valid_mask

    # Count errors inside the band as well as total errors so both local severity
    # and concentration can be reported without another image pass.
    boundary_valid_pixels = int(np.count_nonzero(boundary_band & valid_mask))
    boundary_error_pixels = int(np.count_nonzero(error_mask & boundary_band))
    total_error_pixels = int(np.count_nonzero(error_mask))

    # Zero denominators can occur on all-one-class masks or perfectly predicted
    # images. Returning zero keeps sorting deterministic and avoids NaNs here.
    boundary_error_rate = (
        boundary_error_pixels / boundary_valid_pixels
        if boundary_valid_pixels > 0
        else 0.0
    )
    boundary_error_concentration = (
        boundary_error_pixels / total_error_pixels
        if total_error_pixels > 0
        else 0.0
    )

    return (
        float(boundary_error_rate),
        boundary_error_pixels,
        float(boundary_error_concentration),
        total_error_pixels,
    )


@jaxtyped(typechecker=beartype)
def count_small_objects(
    target_class_mask: BinaryMask,
    max_object_pixels: int,
) -> int:
    """Count small connected ground-truth regions for the selected target class.

    Semantic masks do not contain instance ids, so connected components are used
    as an operational approximation of separate objects. Any 8-connected region
    with area at most ``max_object_pixels`` is counted as a small object.

    Steps
    -----
    1. Convert the boolean target-class mask to an 8-bit image.
    2. Label 8-connected foreground components with OpenCV.
    3. Count non-background components whose area is below the threshold.

    Parameters
    ----------
    target_class_mask : np.ndarray
        Boolean mask for one semantic class with shape ``(H, W)``.
    max_object_pixels : int
        Maximum connected-component area, in resized-image pixels, considered small.

    Returns
    -------
    int
        Number of connected components whose area is in ``[1, max_object_pixels]``.

    Raises
    ------
    ValueError
        If ``max_object_pixels`` is not positive.
    """
    # The threshold must describe at least one pixel to define a meaningful
    # small-object region.
    if max_object_pixels <= 0:
        raise ValueError("max_object_pixels must be positive.")

    # Empty masks contain no connected foreground objects, and short-circuiting
    # avoids paying the component-labeling cost for the common absent-class case.
    if not np.any(target_class_mask):
        return 0

    # OpenCV label 0 is background. The stats table stores component area in the
    # CC_STAT_AREA column, so rows 1..N contain the semantic foreground regions.
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        target_class_mask.astype(np.uint8),
        connectivity=8,
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.count_nonzero(component_areas <= max_object_pixels))


def boundary_error_level(boundary_error_rate: float) -> str:
    """Convert a numeric boundary error rate into a readable severity label.

    The labels are intended for quick CSV inspection only; automatic case
    selection uses the underlying numeric measurements rather than these bins.

    Steps
    -----
    1. Label rates below 10% as ``low``.
    2. Label rates below 25% as ``medium``.
    3. Label all larger rates as ``high``.

    Parameters
    ----------
    boundary_error_rate : float
        Fraction of valid boundary-band pixels that are misclassified.

    Returns
    -------
    str
        One of ``"low"``, ``"medium"``, or ``"high"``.
    """
    # Human-readable bins make the output resemble the requested example table,
    # while the numeric rate is preserved in its own column for analysis.
    if boundary_error_rate < 0.10:
        return "low"
    if boundary_error_rate < 0.25:
        return "medium"
    return "high"


def resolve_target_class(target_class: str) -> int:
    """Resolve a class name or numeric id to a scored BDD100k class id.

    Class names are matched case-insensitively, with underscores and hyphens
    treated as spaces. The ignored ``unknown`` class is intentionally rejected.

    Steps
    -----
    1. Accept integer-like strings directly when they fall in ``0..18``.
    2. Normalize a textual class name and compare it with ``CLASS_NAMES``.
    3. Raise an informative error when the value cannot be resolved.

    Parameters
    ----------
    target_class : str
        Class name such as ``"person"`` or numeric id such as ``"11"``.

    Returns
    -------
    int
        Resolved class id in the scored range ``0..18``.

    Raises
    ------
    ValueError
        If the class name/id is unknown or refers to the ignored class.
    """
    # Numeric ids are convenient when callers already know the palette index.
    normalized = target_class.strip().lower()
    if normalized.lstrip("+-").isdigit():
        class_id = int(normalized)
        if 0 <= class_id < NUM_SCORED_CLASSES:
            return class_id
        raise ValueError(
            f"Target class id must be in [0, {NUM_SCORED_CLASSES - 1}], got {class_id}."
        )

    # Normalize common CLI spellings so names such as traffic_light and
    # traffic-light both match the human-readable CLASS_NAMES list.
    normalized = " ".join(normalized.replace("_", " ").replace("-", " ").split())
    for class_id, class_name in enumerate(CLASS_NAMES[:NUM_SCORED_CLASSES]):
        if normalized == class_name.lower():
            return class_id

    valid_names = ", ".join(CLASS_NAMES[:NUM_SCORED_CLASSES])
    raise ValueError(
        f"Unknown target class '{target_class}'. Use an id in 0..18 or one of: {valid_names}."
    )


@jaxtyped(typechecker=beartype)
def safe_iou_value(class_iou: ClassIoUArray, class_id: int) -> float:
    """Return one class IoU as a Python float while preserving missing values.

    Steps
    -----
    1. Read the requested class entry from the IoU array.
    2. Convert finite values to ``float`` and absent-class ``NaN`` to ``np.nan``.

    Parameters
    ----------
    class_iou : np.ndarray
        Per-class IoU array.
    class_id : int
        Class index to retrieve.

    Returns
    -------
    float
        IoU value or ``NaN`` when the class is absent from both maps.

    Raises
    ------
    IndexError
        If ``class_id`` lies outside ``class_iou``.
    """
    # Preserve NaN instead of replacing it with zero, because class absence and a
    # genuine zero-IoU failure are analytically different situations.
    return float(class_iou[class_id])


def evaluate_per_image(
    model: nn.Module,
    dataloader: DataLoader[Tuple[ImageTensor, MaskTensor]],
    test_df: pd.DataFrame,
    device: torch.device,
    target_class_id: int,
    boundary_radius: int,
    small_object_max_pixels: int,
) -> pd.DataFrame:
    """Run one inference pass and collect per-image segmentation diagnostics.

    This function keeps the batch-efficient model forward pass from normal
    evaluation, but reduces metrics separately for every image instead of
    accumulating confusion counts over the whole dataset.

    Steps
    -----
    1. Run each test batch once under ``torch.inference_mode``.
    2. Convert predictions and one-hot targets to per-pixel class ids.
    3. Compute per-image IoU, target-class FP/FN, boundary, and small-object metrics.
    4. Associate each result with the corresponding deterministic test-set paths.
    5. Return one DataFrame row per image.

    Parameters
    ----------
    model : torch.nn.Module
        Trained semantic-segmentation model.
    dataloader : torch.utils.data.DataLoader
        Deterministic test loader with ``shuffle=False``.
    test_df : pandas.DataFrame
        Test-set rows containing ``image_paths`` and ``mask_paths``.
    device : torch.device
        Device used for inference.
    target_class_id : int
        Scored class id used for FP/FN and small-object analysis.
    boundary_radius : int
        Dilation radius used to define the ground-truth boundary band.
    small_object_max_pixels : int
        Maximum connected-component area counted as a small target object.

    Returns
    -------
    pandas.DataFrame
        Per-image metric table suitable for CSV export and case selection.

    Raises
    ------
    ValueError
        If loader ordering yields a different sample count than ``test_df`` or
        if no rows are produced.
    """
    # Inference mode disables dropout/batch-norm updates and avoids gradient
    # storage, keeping this selection pass lightweight.
    model.eval()
    rows: List[Dict[str, object]] = []
    row_offset = 0

    with torch.inference_mode():
        for images, masks in dataloader:
            # Move only the current batch to the execution device so memory use
            # remains equivalent to the normal batched evaluation pipeline.
            images = images.to(device)
            masks = masks.to(device)

            logits = forward(model, images)
            predictions = logits.argmax(dim=1)
            targets = masks.argmax(dim=1)
            batch_size = int(predictions.shape[0])

            # Path lookup relies on DataLoader order, so verify the batch cannot
            # overrun the deterministic test frame before computing statistics.
            if row_offset + batch_size > len(test_df):
                raise ValueError(
                    "The dataloader yielded more samples than are present in test_df."
                )

            # Each batch item gets independent statistics while sharing the same
            # model forward pass, which avoids per-image model invocations.
            for batch_index in range(batch_size):
                prediction = cast(
                    ClassIndexArray,
                    predictions[batch_index].detach().cpu().numpy().astype(np.int64),
                )
                target = cast(
                    ClassIndexArray,
                    targets[batch_index].detach().cpu().numpy().astype(np.int64),
                )

                # Unknown ground-truth pixels are excluded exactly as in the
                # reference evaluator. Predicted unknown pixels on valid GT remain errors.
                valid_mask = cast(BinaryMask, target != IGNORED_CLASS_ID)
                valid_pixels = int(np.count_nonzero(valid_mask))
                correct_pixels = int(np.count_nonzero((prediction == target) & valid_mask))
                pixel_accuracy = correct_pixels / valid_pixels if valid_pixels > 0 else 0.0

                class_iou = compute_class_iou(
                    prediction,
                    target,
                    valid_mask,
                    NUM_SCORED_CLASSES,
                )
                finite_iou = class_iou[np.isfinite(class_iou)]
                mean_iou = float(np.mean(finite_iou)) if finite_iou.size > 0 else 0.0

                # FP/FN are deliberately target-class-specific so the failure
                # categories correspond to an interpretable semantic class.
                pred_target = prediction == target_class_id
                gt_target = target == target_class_id
                fp_pixels = int(np.count_nonzero(pred_target & ~gt_target & valid_mask))
                fn_pixels = int(np.count_nonzero(~pred_target & gt_target & valid_mask))
                target_gt_pixels = int(np.count_nonzero(gt_target & valid_mask))

                # Boundary statistics quantify both local boundary difficulty and
                # whether the image's total error mass is concentrated there.
                (
                    boundary_error_rate,
                    boundary_error_pixels,
                    boundary_error_concentration,
                    total_error_pixels,
                ) = compute_boundary_metrics(
                    prediction,
                    target,
                    valid_mask,
                    boundary_radius,
                )
                boundary_failure_score = (
                    boundary_error_rate * boundary_error_concentration
                )

                # Connected semantic regions approximate small objects because
                # BDD100k semantic masks do not provide instance ids here.
                small_object_count = count_small_objects(
                    cast(BinaryMask, gt_target & valid_mask),
                    small_object_max_pixels,
                )

                # Recover the image/mask paths by deterministic row order. Resetting
                # test_df's index before this function makes iloc alignment explicit.
                data_row = test_df.iloc[row_offset + batch_index]
                image_path = str(data_row["image_paths"])
                mask_path = str(data_row["mask_paths"])
                image_name = os.path.basename(image_path)

                rows.append(
                    {
                        "image": image_name,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "mean_iou": mean_iou,
                        "road_iou": safe_iou_value(class_iou, ROAD_CLASS_ID),
                        "sidewalk_iou": safe_iou_value(class_iou, SIDEWALK_CLASS_ID),
                        "person_iou": safe_iou_value(class_iou, PERSON_CLASS_ID),
                        "target_class_iou": safe_iou_value(class_iou, target_class_id),
                        "pixel_accuracy": float(pixel_accuracy),
                        "target_class": CLASS_NAMES[target_class_id],
                        "target_gt_pixels": target_gt_pixels,
                        "fp_pixels": fp_pixels,
                        "fn_pixels": fn_pixels,
                        "boundary_error": boundary_error_rate,
                        "boundary_error_level": boundary_error_level(boundary_error_rate),
                        "boundary_error_pixels": boundary_error_pixels,
                        "boundary_error_concentration": boundary_error_concentration,
                        "boundary_failure_score": float(boundary_failure_score),
                        "error_pixels": total_error_pixels,
                        "small_object_count": small_object_count,
                    }
                )

            row_offset += batch_size

    # A mismatch would silently associate metrics with the wrong paths, which is
    # unacceptable because the selected CSV is intended to drive later XAI runs.
    if row_offset != len(test_df):
        raise ValueError(
            f"The dataloader yielded {row_offset} samples, but test_df has {len(test_df)} rows."
        )
    if not rows:
        raise ValueError("The test dataloader yielded no images.")

    return pd.DataFrame(rows)


def select_representative_cases(
    metrics_df: pd.DataFrame,
    cases_per_category: int,
) -> pd.DataFrame:
    """Select representative images for each requested XAI case category.

    Every category is ranked independently, so an image may legitimately appear
    in multiple categories when it exemplifies more than one behavior. This is
    retained in the output rather than silently discarding useful overlaps.

    Steps
    -----
    1. Rank correct predictions by pixel accuracy and strong/weak cases by mIoU.
    2. Rank false-positive and false-negative cases by target-class pixel counts.
    3. Rank small-object cases by target-class area, preferring images with small regions.
    4. Rank boundary failures using both boundary error rate and error concentration.
    5. Concatenate the selected rows with category, rank, and selection score.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Per-image statistics returned by :func:`evaluate_per_image`.
    cases_per_category : int
        Maximum number of images to select in each category.

    Returns
    -------
    pandas.DataFrame
        Selected case table containing category metadata and image/mask paths.

    Raises
    ------
    ValueError
        If ``cases_per_category`` is not positive or required columns are missing.
    """
    # A positive category size is required for every ranking operation below.
    if cases_per_category <= 0:
        raise ValueError("cases_per_category must be positive.")

    required_columns = {
        "image",
        "image_path",
        "mask_path",
        "mean_iou",
        "target_class_iou",
        "pixel_accuracy",
        "fp_pixels",
        "fn_pixels",
        "target_gt_pixels",
        "small_object_count",
        "boundary_failure_score",
        "boundary_error",
        "boundary_error_concentration",
        "error_pixels",
    }
    missing_columns = sorted(required_columns.difference(metrics_df.columns))
    if missing_columns:
        raise ValueError(
            "metrics_df is missing required columns: " + ", ".join(missing_columns)
        )

    selected_frames: List[pd.DataFrame] = []

    def append_selection(
        category: str,
        frame: pd.DataFrame,
        score_column: str,
        ascending: bool,
    ) -> None:
        """Append one ranked category to the outer selection list.

        Steps
        -----
        1. Sort the candidate frame by the category's score column.
        2. Keep at most ``cases_per_category`` rows.
        3. Add stable category/rank/score fields for CSV export.

        Parameters
        ----------
        category : str
            Human-readable selection category.
        frame : pandas.DataFrame
            Candidate rows eligible for this category.
        score_column : str
            Column used as the primary ranking score.
        ascending : bool
            Whether smaller values should rank ahead of larger values.

        Returns
        -------
        None
            The selected frame is appended to the enclosing ``selected_frames`` list.
        """
        # Skip empty candidate sets rather than manufacturing placeholder rows.
        if frame.empty:
            return

        ranked = frame.sort_values(
            by=score_column,
            ascending=ascending,
            kind="stable",
        ).head(cases_per_category).copy()
        ranked.insert(0, "category", category)
        ranked.insert(1, "rank", np.arange(1, len(ranked) + 1, dtype=np.int64))
        ranked.insert(2, "selection_score", ranked[score_column].astype(float))
        selected_frames.append(ranked)

    # "Correct" is operationalized as highest per-image pixel accuracy, which is
    # distinct from class-balanced mean IoU and therefore captures another notion
    # of a globally strong segmentation.
    append_selection(
        "correct_prediction",
        metrics_df,
        "pixel_accuracy",
        ascending=False,
    )

    # Mean IoU gives class-balanced high- and low-quality examples independent of
    # the dominant background/road pixel mass.
    append_selection(
        "high_iou",
        metrics_df,
        "mean_iou",
        ascending=False,
    )
    append_selection(
        "low_iou_failure",
        metrics_df,
        "mean_iou",
        ascending=True,
    )

    # Require a positive FP/FN count so these categories never contain examples
    # that do not actually exhibit the corresponding target-class error mode.
    append_selection(
        "false_positive",
        metrics_df.loc[metrics_df["fp_pixels"] > 0],
        "fp_pixels",
        ascending=False,
    )
    append_selection(
        "false_negative",
        metrics_df.loc[metrics_df["fn_pixels"] > 0],
        "fn_pixels",
        ascending=False,
    )

    # Small-object examples must contain the target class and at least one small
    # connected component. Ranking by total target pixels prioritizes images where
    # that class occupies the least image area, as requested.
    small_candidates = metrics_df.loc[
        (metrics_df["target_gt_pixels"] > 0)
        & (metrics_df["small_object_count"] > 0)
    ]
    append_selection(
        "small_object",
        small_candidates,
        "target_gt_pixels",
        ascending=True,
    )

    # The product score is high only when boundary pixels are error-prone and a
    # large share of the image's errors are concentrated around those boundaries.
    boundary_candidates = metrics_df.loc[metrics_df["error_pixels"] > 0]
    append_selection(
        "boundary_failure",
        boundary_candidates,
        "boundary_failure_score",
        ascending=False,
    )

    # If every specialized candidate frame is empty, fail loudly instead of
    # emitting an unusable case CSV without explanation.
    if not selected_frames:
        raise ValueError("No representative cases could be selected from metrics_df.")

    selected = pd.concat(selected_frames, ignore_index=True)

    # Keep the paths prominent because this CSV is the hand-off artifact for XAI,
    # while retaining key scores so later analysis can explain why each row was chosen.
    output_columns = [
        "category",
        "rank",
        "selection_score",
        "image",
        "image_path",
        "mask_path",
        "mean_iou",
        "pixel_accuracy",
        "fp_pixels",
        "fn_pixels",
        "target_gt_pixels",
        "small_object_count",
        "boundary_error",
        "boundary_error_concentration",
        "boundary_failure_score",
    ]
    return selected[output_columns]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for representative-case selection.

    Steps
    -----
    1. Require a trained model checkpoint path, matching ``evaluate_performance.py``.
    2. Accept optional inference, ranking, boundary, and small-object parameters.
    3. Return the parsed argument namespace.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    # The checkpoint is mandatory because all selection metrics depend on model
    # predictions. Remaining flags have reproducible defaults for routine use.
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the BDD100k test split per image and automatically select "
            "representative cases for later XAI analysis."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the PyTorch checkpoint file (.pt) to explain.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Configuration.BATCH_SIZE,
        help="Inference batch size. Defaults to the evaluation configuration value.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="person",
        help=(
            "Class name or id used for FP/FN and small-object selection. "
            "Defaults to 'person'."
        ),
    )
    parser.add_argument(
        "--cases-per-category",
        type=int,
        default=10,
        help="Maximum selected images per category. Defaults to 10.",
    )
    parser.add_argument(
        "--boundary-radius",
        type=int,
        default=3,
        help="Boundary-band dilation radius in resized-image pixels. Defaults to 3.",
    )
    parser.add_argument(
        "--small-object-max-pixels",
        type=int,
        default=500,
        help=(
            "Maximum connected target-class region area counted as a small object. "
            "Defaults to 500 resized-image pixels."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./model/explanations",
        help="Directory for per-image and selected-case CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run per-image evaluation and export representative XAI case selections.

    Steps
    -----
    1. Parse CLI arguments and resolve the target semantic class.
    2. Load the trained checkpoint and deterministic BDD100k test split.
    3. Run one batched inference pass while computing metrics per image.
    4. Save the complete per-image metric table to CSV.
    5. Select representative cases and save their image/mask paths to a second CSV.

    Returns
    -------
    None
        Results are written to CSV files and summarized on standard output.

    Raises
    ------
    ValueError
        If CLI numeric values are invalid, the target class cannot be resolved,
        or evaluation/selection produces no usable rows.
    FileNotFoundError
        If the checkpoint or dataset files required by the imported evaluator are missing.
    """
    args = parse_arguments()

    # Validate simple numeric CLI values before loading the potentially large model
    # so incorrect arguments fail quickly and with focused error messages.
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.cases_per_category <= 0:
        raise ValueError("--cases-per-category must be positive.")
    if args.boundary_radius < 0:
        raise ValueError("--boundary-radius must be non-negative.")
    if args.small_object_max_pixels <= 0:
        raise ValueError("--small-object-max-pixels must be positive.")

    target_class_id = resolve_target_class(args.target_class)

    print("\n" + "=" * 68)
    print("REPRESENTATIVE XAI CASE SELECTION")
    print("=" * 68)
    print(f"Loading checkpoint from: {args.checkpoint}")

    # Reuse the checkpoint loader from evaluate_performance.py so serialization,
    # device transfer, and eval-mode behavior remain identical.
    model = load_trained_model(args.checkpoint, Configuration.DEVICE)
    print(f"Target execution device: {Configuration.DEVICE}")
    print(
        f"Target class for FP/FN and small-object analysis: "
        f"{target_class_id} ({CLASS_NAMES[target_class_id]})"
    )

    # Recreate the same held-out test split as training/performance evaluation,
    # then reset the index because path association uses deterministic iloc order.
    print("Loading BDD100k test split...")
    _, _, test_df = load_dataset_from_files()
    test_df = test_df.reset_index(drop=True)
    print(f"Loaded {len(test_df)} test samples.")

    # Match the reference evaluator's resize/tensor transform so per-image metrics
    # are measured on exactly the same model input resolution.
    inference_transforms = A.Compose(
        [
            A.Resize(
                height=Configuration.IMAGE_HEIGHT,
                width=Configuration.IMAGE_WIDTH,
            ),
            ToTensorV2(),
        ]
    )
    test_ds = BDDSegmentationDataset(test_df, transform=inference_transforms)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=Configuration.NUM_WORKERS,
    )

    # One batched pass is sufficient: all ranking statistics are reduced per image
    # immediately instead of rerunning inference for every candidate category.
    print("Running one per-image inference pass...")
    metrics_df = evaluate_per_image(
        model=model,
        dataloader=test_loader,
        test_df=test_df,
        device=Configuration.DEVICE,
        target_class_id=target_class_id,
        boundary_radius=args.boundary_radius,
        small_object_max_pixels=args.small_object_max_pixels,
    )

    # Create the output directory once and write the complete diagnostic table
    # before selection so every chosen case can be audited against source metrics.
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "per_image_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Rank each requested category from the same metric table, then export a compact
    # hand-off CSV containing the image and ground-truth mask paths for later XAI.
    selected_df = select_representative_cases(
        metrics_df=metrics_df,
        cases_per_category=args.cases_per_category,
    )
    selected_path = os.path.join(args.output_dir, "representative_cases.csv")
    selected_df.to_csv(selected_path, index=False)

    print(f"Saved per-image metrics to: {metrics_path}")
    print(f"Saved representative cases to: {selected_path}")
    print("Selected rows by category:")
    category_counts = selected_df.groupby("category", sort=False).size()
    for category, count in category_counts.items():
        print(f"  {category:<20} : {int(count)}")


if __name__ == "__main__":
    main()

