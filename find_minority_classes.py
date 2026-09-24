import os
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from beartype import beartype
from jaxtyping import Float64, UInt8, jaxtyped


# BDD100k color-label palette. The row index is the class id (0-19).
# Predicted class maps are coloured with this same palette so they render
# consistently with ground-truth color labels.
CLASS_COLORS: np.ndarray = np.array([
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


# Human-readable names for each palette row, kept in lock-step with CLASS_COLORS.
CLASS_NAMES: List[str] = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle", "unknown",
]

# Boundary-critical classes that are forced into the minority set regardless of
# their pixel prevalence. Sidewalk is common in urban frames, so a pure
# frequency threshold can miss it, but the road/sidewalk boundary is exactly
# where the model over-predicts road and needs corrective oversampling.
BOUNDARY_CLASS_IDS: List[int] = [1]  # Sidewalk


# Shape aliases enforced at runtime by ``jaxtyped`` + ``beartype``. Each alias
# pins the rank and dtype of an array crossing a decorated function boundary, so
# a shape mismatch raises a ``TypeCheckError`` instead of a confusing
# downstream indexing or broadcasting error.
# ``h``/``w`` are the spatial dimensions of a mask, and ``c`` is the number of
# classes.
NumpyColorLabel = UInt8[np.ndarray, "h w 3"]  # RGB color-label, channels-last
ClassIndexArray = UInt8[np.ndarray, "h w"]  # per-pixel class-id map
ClassCounts = Float64[np.ndarray, "c"]  # per-class counts, 1-D
SampleWeightsArray = Float64[np.ndarray, "_"]  # per-sample weights, 1-D


class ImagePath:
    """Default directory paths for BDD100k masks and cached class weights."""

    BASE: str = "./data/bdd100k"
    CLASS_WEIGHTS_PATH: str = "./data/class_weights.npy"


@jaxtyped(typechecker=beartype)
def color_label_to_class_index(label: NumpyColorLabel) -> ClassIndexArray:
    """Map an RGB color-label image to a per-pixel class-index map.

    BDD100k stores segmentation masks as RGB PNGs whose colours are exactly
    the entries of ``CLASS_COLORS``. Semantic segmentation needs the class id
    per pixel (shape ``(H, W)``) rather than the RGB representation (shape
    ``(H, W, 3)``), so this conversion must happen before mask statistics
    can be aggregated.

    Steps
    -----
    1. Initialise the output array with the id of the last palette entry so
       any unknown colour degrades to Unknown rather than an invalid index.
    2. For each palette colour, match RGB values across the entire image and
       assign the corresponding integer class index.

    Parameters
    ----------
    label : NumpyColorLabel
        RGB color-label array of shape ``(H, W, 3)`` with uint8 values.

    Returns
    -------
    ClassIndexArray
        Class-index array of shape ``(H, W)`` and dtype ``uint8``, whose values
        are in ``[0, NUM_CLASSES)``. The spatial dimensions match ``label``.
    """
    # Default to the last class id so unknown colours fall back gracefully
    # instead of indexing the palette out of bounds later.
    class_ids = np.full(label.shape[:2], CLASS_COLORS.shape[0] - 1, dtype=np.uint8)

    # Match each palette colour via exact RGB equality. This stays fast because
    # every comparison operates on the whole image at once.
    for class_id, (red, green, blue) in enumerate(CLASS_COLORS):
        match = (
            (label[..., 0] == red)
            & (label[..., 1] == green)
            & (label[..., 2] == blue)
        )
        class_ids[match] = class_id

    return class_ids


@jaxtyped(typechecker=beartype)
def compute_class_statistics(
    df: pd.DataFrame,
    num_classes: int = 20,
) -> Tuple[ClassCounts, ClassCounts, List[Set[int]]]:
    """Scan every train mask and aggregate per-class statistics.

    The bias correction needs two pieces of information: which classes are
    under-represented (decided from pixel prevalence) and how to weight each
    image (decided from image-level occurrence). Both are produced by a single
    pass over the masks, so the slow PNG decoding happens only once and is kept
    independent of the online augmentation pipeline.

    Steps
    -----
    1. Load each mask in ``df`` and collapse its RGB color-label to a per-pixel
       class-index map via :func:`color_label_to_class_index`.
    2. Record the set of class ids present per mask and, for each present class,
       add one to its occurrence count and its pixel area to its pixel count.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame carrying the ``mask_paths`` column used to locate each mask.
    num_classes : int, optional
        Total number of classes including the background/unknown class.
        Defaults to ``20``.

    Returns
    -------
    Tuple[ClassCounts, ClassCounts, List[Set[int]]]
        ``(pixel_counts, occurrence_counts, present_classes)``. ``pixel_counts``
        and ``occurrence_counts`` are float64 arrays of shape ``(num_classes,)``;
        ``present_classes[i]`` is the set of class ids present in mask ``i``.
    """
    mask_paths: List[str] = df["mask_paths"].to_list()

    pixel_counts = np.zeros(num_classes, dtype=np.float64)
    occurrence_counts = np.zeros(num_classes, dtype=np.float64)
    present_classes: List[Set[int]] = []

    for mask_path in mask_paths:
        mask_pil = Image.open(mask_path).convert("RGB")
        class_index = color_label_to_class_index(np.array(mask_pil))

        # ``return_counts`` yields the pixel area per class in one pass, which
        # feeds both the occurrence count (presence) and the pixel count
        # (prevalence) used downstream.
        class_ids, areas = np.unique(class_index, return_counts=True)
        present_classes.append(set(class_ids.tolist()))
        for class_id, area in zip(class_ids, areas, strict=True):
            occurrence_counts[class_id] += 1
            pixel_counts[class_id] += float(area)

    return pixel_counts, occurrence_counts, present_classes


@jaxtyped(typechecker=beartype)
def find_minority_classes(
    pixel_counts: ClassCounts,
    method: str = "relative_to_max",
    threshold: float = 0.05,
    num_classes: int = 20,
) -> List[int]:
    """Derive under-represented class ids from pixel-prevalence statistics.

    Instead of hardcoding which classes are rare, this function ranks classes by
    the fraction of pixels they occupy and flags everything below a relative
    cutoff. Pixel prevalence (rather than image occurrence) is the right signal
    for segmentation because a bias is caused by class area, not mere presence:
    road dominates the bottom half of highway frames precisely because it
    occupies the largest area.

    Steps
    -----
    1. Normalise ``pixel_counts`` by the total pixel count to obtain per-class
       prevalence fractions.
    2. Pick a cutoff fraction using ``method``: the median prevalence, or a
       ``threshold`` fraction of the most prevalent class.
    3. Return every class id below the cutoff, excluding the ``Unknown`` class.

    Parameters
    ----------
    pixel_counts : ClassCounts
        Per-class pixel counts of shape ``(num_classes,)``.
    method : str, optional
        ``"relative_to_max"`` uses ``threshold * max(prevalence)`` as the
        cutoff; ``"below_median"`` uses the median prevalence. Defaults to
        ``"relative_to_max"``.
    threshold : float, optional
        Fraction of the most prevalent class to use as cutoff when
        ``method == "relative_to_max"``. Defaults to ``0.05``.
    num_classes : int, optional
        Total number of semantic classes. Defaults to ``20``.

    Returns
    -------
    List[int]
        Class ids whose prevalence falls below the cutoff, sorted ascending.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"relative_to_max"`` or ``"below_median"``.
    """
    total_pixels = pixel_counts.sum()
    fractions = pixel_counts / total_pixels

    if method == "below_median":
        cutoff = float(np.median(fractions))
    elif method == "relative_to_max":
        cutoff = float(fractions.max()) * threshold
    else:
        raise ValueError(
            f"Unknown minority selection method: {method!r}. "
            "Use 'relative_to_max' or 'below_median'."
        )

    # The last class (Unknown) is excluded: it is not a real object class and
    # would otherwise always be flagged as rare.
    minority_ids = [
        int(class_id)
        for class_id in range(num_classes - 1)
        if fractions[class_id] < cutoff
    ]
    return minority_ids


@jaxtyped(typechecker=beartype)
def compute_sample_weights(
    present_classes: List[Set[int]],
    occurrence_counts: ClassCounts,
    minority_class_ids: List[int],
    eps: float = 1e-6,
) -> SampleWeightsArray:
    """Build per-sample weights that oversample minority-class images.

    Each sample starts with a base weight of ``1.0`` and gains an inverse-
    frequency boost for every minority class it contains. A class that appears
    in few masks therefore contributes a larger boost, which
    :class:`torch.utils.data.WeightedRandomSampler` uses to draw minority-class
    images more often while still keeping majority images in the mix.

    Steps
    -----
    1. Start from a unit weight for every sample.
    2. For each sample, add ``num_masks / occurrence_count`` for every minority
       class present in that sample's mask.

    Parameters
    ----------
    present_classes : List[Set[int]]
        Per-sample set of class ids present in each mask.
    occurrence_counts : ClassCounts
        Per-class image occurrence counts of shape ``(num_classes,)``.
    minority_class_ids : List[int]
        Class ids to oversample.
    eps : float, optional
        Small constant guarding against a zero occurrence count. Defaults to
        ``1e-6``.

    Returns
    -------
    SampleWeightsArray
        Float64 array of shape ``(num_samples,)``; higher values correspond to
        samples the sampler should draw more frequently.
    """
    num_masks = len(present_classes)

    # A class that appears in few masks gets a large boost, while an
    # always-present class gets roughly ``num_masks / num_masks == 1``. The base
    # weight of ``1.0`` keeps highway-only frames in the mix so the model still
    # sees the majority class, just not exclusively.
    weights = np.ones(num_masks, dtype=np.float64)
    for sample_index, present in enumerate(present_classes):
        for class_id in minority_class_ids:
            if class_id in present:
                count = occurrence_counts[class_id]
                weights[sample_index] += num_masks / (count + eps)

    return weights


@jaxtyped(typechecker=beartype)
def calculate_class_weights(
    df: pd.DataFrame,
    num_classes: int = 20,
    boundary_class_ids: Optional[List[int]] = None,
    method: str = "relative_to_max",
    threshold: float = 0.05,
    eps: float = 1e-6,
) -> SampleWeightsArray:
    """Compute sampling weights for all masks in a dataset based on minority classes.

    Aggregates pixel and occurrence statistics across all masks, derives the
    under-represented minority classes based on relative pixel prevalence,
    merges any mandatory boundary classes, and generates inverse-frequency
    sample weights.

    Steps
    -----
    1. Aggregate per-class pixel counts, occurrence counts, and per-mask class sets.
    2. Identify under-represented minority class IDs below the prevalence threshold.
    3. Merge mandatory boundary-critical class IDs into the minority set.
    4. Log the selected minority classes along with their relative pixel prevalence.
    5. Compute inverse-frequency sampling weights for every sample in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame carrying the ``mask_paths`` column for masks to scan.
    num_classes : int, optional
        Total number of semantic classes. Defaults to ``20``.
    boundary_class_ids : Optional[List[int]], optional
        Class IDs forced into the minority set regardless of prevalence.
        Defaults to ``None`` (which uses ``BOUNDARY_CLASS_IDS``).
    method : str, optional
        Cutoff method for minority selection. Defaults to ``"relative_to_max"``.
    threshold : float, optional
        Prevalence cutoff fraction. Defaults to ``0.05``.
    eps : float, optional
        Epsilon added to denominator to avoid division by zero. Defaults to ``1e-6``.

    Returns
    -------
    SampleWeightsArray
        Float64 array of shape ``(num_samples,)`` with per-sample sampling weights.

    Raises
    ------
    ValueError
        If ``method`` is unrecognized by :func:`find_minority_classes`.
    """
    # Default boundary classes to module constant if None specified
    if boundary_class_ids is None:
        boundary_class_ids = BOUNDARY_CLASS_IDS

    # Step 1: Scan masks to compute pixel area and occurrence frequency per class
    pixel_counts, occurrence_counts, present_classes = compute_class_statistics(
        df=df, num_classes=num_classes
    )

    # Step 2: Automatically identify minority classes based on area prevalence
    minority_ids = find_minority_classes(
        pixel_counts=pixel_counts,
        method=method,
        threshold=threshold,
        num_classes=num_classes,
    )

    # Step 3: Forcibly include boundary-critical classes (such as sidewalk) that
    # are spatially common but critical for error prevention along object edges
    merged_minority_ids = sorted(set(minority_ids) | set(boundary_class_ids))

    # Step 4: Log minority class details and their pixel fractions for visibility
    total_pixels = pixel_counts.sum()
    print("Oversampling the following minority classes:")
    for class_id in merged_minority_ids:
        prevalence = 100.0 * pixel_counts[class_id] / total_pixels
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
        print(f"  {class_id:>2} {class_name:<14} ({prevalence:6.3f}% of pixels)")

    # Step 5: Derive sample weights boosted by inverse frequency of contained minority classes
    sample_weights = compute_sample_weights(
        present_classes=present_classes,
        occurrence_counts=occurrence_counts,
        minority_class_ids=merged_minority_ids,
        eps=eps,
    )

    return sample_weights


@jaxtyped(typechecker=beartype)
def calculate_and_save_class_weights(
    df: pd.DataFrame,
    output_path: str,
    num_classes: int = 20,
    boundary_class_ids: Optional[List[int]] = None,
    method: str = "relative_to_max",
    threshold: float = 0.05,
    eps: float = 1e-6,
) -> SampleWeightsArray:
    """Calculate class-based sample weights and persist them into an NPY file.

    Computes the sample weights from mask prevalence and saves the resulting
    NumPy array to ``output_path`` so subsequent training runs can load it directly.

    Steps
    -----
    1. Calculate sample weights using :func:`calculate_class_weights`.
    2. Ensure the parent directory of ``output_path`` exists.
    3. Persist the NumPy array to ``output_path`` using ``np.save``.
    4. Return the computed weights array.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame containing the ``mask_paths`` column.
    output_path : str
        Destination file path where the .npy file will be written.
    num_classes : int, optional
        Total number of semantic classes. Defaults to ``20``.
    boundary_class_ids : Optional[List[int]], optional
        Class IDs forced into the minority set regardless of prevalence.
        Defaults to ``None``.
    method : str, optional
        Cutoff method for minority selection. Defaults to ``"relative_to_max"``.
    threshold : float, optional
        Prevalence cutoff fraction. Defaults to ``0.05``.
    eps : float, optional
        Epsilon to guard against division by zero. Defaults to ``1e-6``.

    Returns
    -------
    SampleWeightsArray
        Float64 array of shape ``(num_samples,)``.

    Raises
    ------
    OSError
        If directory creation or file writing fails.
    ValueError
        If ``method`` is unrecognized.
    """
    # Compute the sample weights using mask prevalence statistics
    sample_weights = calculate_class_weights(
        df=df,
        num_classes=num_classes,
        boundary_class_ids=boundary_class_ids,
        method=method,
        threshold=threshold,
        eps=eps,
    )

    # Ensure the parent directory exists before attempting to write the file
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Persist the weights array to disk
    np.save(output_path, sample_weights)
    print(f"Class weights successfully saved to '{output_path}' (shape: {sample_weights.shape}).")

    return sample_weights


@jaxtyped(typechecker=beartype)
def load_or_compute_class_weights(
    df: pd.DataFrame,
    weights_path: str,
    num_classes: int = 20,
    boundary_class_ids: Optional[List[int]] = None,
    method: str = "relative_to_max",
    threshold: float = 0.05,
    eps: float = 1e-6,
) -> SampleWeightsArray:
    """Load precomputed class weights from disk or compute and persist them.

    Inspects whether the designated NPY file exists. If it exists, the weights
    are loaded immediately, bypassing the expensive mask reading loop. If the
    file does not exist, the weights are calculated, saved to ``weights_path``,
    and returned.

    Steps
    -----
    1. Check if ``weights_path`` exists on disk.
    2. If found, load and return the weights using ``np.load``.
    3. If not found, invoke :func:`calculate_and_save_class_weights` to compute,
       save, and return the weights.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame containing the ``mask_paths`` column.
    weights_path : str
        Path to the .npy file containing serialized weights.
    num_classes : int, optional
        Total number of semantic classes. Defaults to ``20``.
    boundary_class_ids : Optional[List[int]], optional
        Class IDs forced into the minority set. Defaults to ``None``.
    method : str, optional
        Cutoff method for minority selection. Defaults to ``"relative_to_max"``.
    threshold : float, optional
        Prevalence cutoff fraction. Defaults to ``0.05``.
    eps : float, optional
        Epsilon to guard against division by zero. Defaults to ``1e-6``.

    Returns
    -------
    SampleWeightsArray
        Float64 array of shape ``(num_samples,)``.
    """
    # Check for cached weights file to bypass mask image decoding
    if os.path.exists(weights_path):
        print(f"Reusing existing class weights from '{weights_path}'...")
        weights: np.ndarray = np.load(weights_path)
        return weights

    # File does not exist: compute from scratch and serialize
    print(
        f"Class weights file '{weights_path}' not found. "
        "Calculating class weights from training masks and saving to disk..."
    )
    return calculate_and_save_class_weights(
        df=df,
        output_path=weights_path,
        num_classes=num_classes,
        boundary_class_ids=boundary_class_ids,
        method=method,
        threshold=threshold,
        eps=eps,
    )


def main() -> None:
    """Scan training masks, calculate sample weights for minority classes, and save to NPY.

    Steps
    -----
    1. Load dataset splits using the project's standard preprocessing logic.
    2. Calculate sample weights and serialize them to the specified NPY file.
    """

    # Import Configuration and dataset loader locally to prevent circular module imports
    from train_teacher import Configuration, load_dataset_from_files

    print("Loading BDD100k training dataset splits...")
    train_df, _, _ = load_dataset_from_files()

    print(f"Calculating and persisting class weights for {len(train_df)} training samples...")
    weights = calculate_and_save_class_weights(
        df=train_df,
        output_path=ImagePath.CLASS_WEIGHTS_PATH,
        num_classes=Configuration.NUM_CLASSES,
        boundary_class_ids=BOUNDARY_CLASS_IDS,
        method="relative_to_max",
        threshold=0.05,
    )
    print(f"Execution complete. Output shape: {weights.shape}, file: '{ImagePath.CLASS_WEIGHTS_PATH}'.")


if __name__ == "__main__":
    main()
