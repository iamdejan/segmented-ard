"""Analyze the BDD100K 10K/100K intersection for evaluator condition pairs.

This script regenerates ``weather_timeofday_pair_data.csv`` from the exact
held-out segmentation population used by the performance and robustness
evaluators. The resulting pair counts therefore represent evaluation samples,
not the official BDD100K 10K ``test`` split.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


class Configuration:
    """Configuration shared with the segmentation evaluation split.

    The constants intentionally mirror the evaluator configuration because the
    pair-count CSV must be derived from the same deterministic held-out split.
    """

    SEED = 768
    TEST_SIZE = 0.2
    SOURCE_IMAGE_WIDTH = 1280
    SOURCE_IMAGE_HEIGHT = 720


class DatasetPath:
    """Filesystem locations used to reconstruct the evaluation population."""

    BASE = Path("./data/bdd100k")
    SEGMENTATION_MASK_TRAIN_PATH = BASE / "segmentation_maps/color_labels/train"
    IMAGE_TRAIN_PATH = BASE / "images_10k/train"
    INTERSECTION_CSV = Path(
        "./data/bdd100k_10k_images_with_100k_metadata_intersection.csv"
    )
    OUTPUT_PAIR_CSV = Path("./data/weather_timeofday_pair_data.csv")


def image_id_from_path(image_path: str) -> str:
    """Extract a BDD100K image ID from an image path.

    Steps
    -----
    1. Take the basename of the supplied path.
    2. Remove the filename extension and return the remaining BDD image ID.

    Parameters
    ----------
    image_path : str
        Path to a BDD100K image file.

    Returns
    -------
    str
        Filename stem used as the BDD100K image ID.
    """
    return os.path.splitext(os.path.basename(image_path))[0]


def find_train_image_path_from_mask(mask_path: str) -> str:
    """Derive the 10K training image path corresponding to a color mask.

    Steps
    -----
    1. Extract the BDD image ID from the ``*_train_color.png`` mask filename.
    2. Construct the matching JPEG path under the 10K training image directory.

    Parameters
    ----------
    mask_path : str
        Path to a BDD100K training color-label mask.

    Returns
    -------
    str
        Path to the corresponding BDD100K 10K training JPEG.
    """
    mask_filename = os.path.basename(mask_path)
    image_id = mask_filename.split("_")[0]
    return str(DatasetPath.IMAGE_TRAIN_PATH / f"{image_id}.jpg")


def is_expected_resolution(file_path: str) -> bool:
    """Check whether an image or mask has the evaluator's expected source size.

    Steps
    -----
    1. Open the file with Pillow without decoding it into a NumPy array.
    2. Compare its width and height with the 1280x720 BDD100K source size.

    Parameters
    ----------
    file_path : str
        Image or mask path to inspect.

    Returns
    -------
    bool
        ``True`` when the file is 1280x720, otherwise ``False``.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    OSError
        If Pillow cannot open the supplied file.
    """
    with Image.open(file_path) as image:
        width, height = image.size

    return (
        width == Configuration.SOURCE_IMAGE_WIDTH
        and height == Configuration.SOURCE_IMAGE_HEIGHT
    )


def load_segmentation_train_pool() -> pd.DataFrame:
    """Build the valid segmentation training pool used before the 80/20 split.

    This reproduces the evaluator's source population while avoiding mutation of
    lists during iteration. File ordering from ``glob.glob`` is preserved so the
    subsequent seeded ``train_test_split`` receives the same ordered population.

    Steps
    -----
    1. Collect training color-label masks in the same directory used by the evaluator.
    2. Skip masks whose source resolution is not 1280x720.
    3. Derive each corresponding 10K training image path.
    4. Skip image/mask pairs whose image resolution is not 1280x720.
    5. Return the remaining image and mask paths in their original glob order.

    Returns
    -------
    pd.DataFrame
        Valid segmentation training rows with ``image_paths`` and ``mask_paths``.

    Raises
    ------
    FileNotFoundError
        If a mask's corresponding JPEG does not exist.
    ValueError
        If no valid segmentation training pairs remain after validation.
    OSError
        If Pillow cannot open a candidate image or mask.
    """
    mask_pattern = str(DatasetPath.SEGMENTATION_MASK_TRAIN_PATH / "*.png")
    train_mask_paths = glob.glob(mask_pattern)

    # Keep explicit lists so row ordering matches the discovered mask ordering;
    # train_test_split is deterministic only when it receives the same input order.
    valid_image_paths: List[str] = []
    valid_mask_paths: List[str] = []
    problematic_masks: List[str] = []
    problematic_images: List[str] = []

    for mask_path in train_mask_paths:
        # Invalid-size masks are excluded because the base evaluator excludes
        # them before constructing its train/test pool.
        if not is_expected_resolution(mask_path):
            problematic_masks.append(mask_path)
            continue

        image_path = find_train_image_path_from_mask(mask_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Training image corresponding to mask was not found: {image_path}"
            )

        # The held-out split must be built from the same valid image/mask pairs
        # that can actually be consumed by the segmentation evaluator.
        if not is_expected_resolution(image_path):
            problematic_images.append(image_path)
            continue

        valid_image_paths.append(image_path)
        valid_mask_paths.append(mask_path)

    print(f"Problematic masks: {problematic_masks}")
    print(f"Problematic images: {problematic_images}")

    if not valid_image_paths:
        raise ValueError("No valid segmentation training image/mask pairs were found.")

    return pd.DataFrame(
        {
            "image_paths": valid_image_paths,
            "mask_paths": valid_mask_paths,
        }
    )


def build_held_out_evaluation_split(train_pool_df: pd.DataFrame) -> pd.DataFrame:
    """Create the deterministic held-out segmentation split used by evaluators.

    Steps
    -----
    1. Apply the same 80/20 ``train_test_split`` used by the evaluation scripts.
    2. Keep the 20% held-out portion only.
    3. Derive a BDD image ID from every held-out JPEG path for metadata joining.

    Parameters
    ----------
    train_pool_df : pd.DataFrame
        Valid segmentation training population containing image and mask paths.

    Returns
    -------
    pd.DataFrame
        Held-out segmentation rows with an added ``id`` column.

    Raises
    ------
    ValueError
        If the supplied training pool is empty.
    """
    if train_pool_df.empty:
        raise ValueError("Cannot create a held-out split from an empty training pool.")

    # The first returned DataFrame is intentionally unused: only the held-out
    # 20% subset is relevant for performance/robustness evaluation counts.
    _, held_out_df = train_test_split(
        train_pool_df,
        test_size=Configuration.TEST_SIZE,
        random_state=Configuration.SEED,
    )
    held_out_df = held_out_df.copy()
    held_out_df["id"] = held_out_df["image_paths"].map(image_id_from_path)
    return held_out_df.reset_index(drop=True)


def load_intersection_metadata(intersection_csv_path: Path) -> pd.DataFrame:
    """Load and validate the image-level 10K/100K metadata intersection.

    Steps
    -----
    1. Read the CSV generated by ``find_100k_10k_intersection.py``.
    2. Require the image ID, environment attributes, image split, and image path.
    3. Normalize string fields and reject missing or duplicate image IDs.
    4. Verify that every metadata image path basename agrees with its declared ID.

    Parameters
    ----------
    intersection_csv_path : pathlib.Path
        Path to the 10K-image/100K-metadata intersection CSV.

    Returns
    -------
    pd.DataFrame
        Validated metadata rows with ``timeofday`` renamed to ``time_of_day``.

    Raises
    ------
    FileNotFoundError
        If the intersection CSV does not exist.
    ValueError
        If required columns or values are missing, IDs are duplicated, or an
        image path does not match its declared ID.
    """
    if not intersection_csv_path.is_file():
        raise FileNotFoundError(
            f"Intersection CSV not found: {intersection_csv_path}"
        )

    intersection_df = pd.read_csv(intersection_csv_path, dtype=str)
    required_columns = {"id", "weather", "timeofday", "image_split", "image_path"}
    missing_columns = required_columns.difference(intersection_df.columns)
    if missing_columns:
        raise ValueError(
            "Intersection CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    intersection_df = intersection_df[
        ["id", "weather", "timeofday", "image_split", "image_path"]
    ].copy()
    if intersection_df.isna().any().any():
        raise ValueError("Intersection CSV contains missing required values.")

    # Strip only accidental surrounding whitespace; category spelling remains
    # unchanged because the evaluators perform exact weather/time-of-day matching.
    for column in ["id", "weather", "timeofday", "image_split", "image_path"]:
        intersection_df[column] = intersection_df[column].str.strip()
        if (intersection_df[column] == "").any():
            raise ValueError(
                f"Intersection CSV contains an empty value in column '{column}'."
            )

    duplicate_ids = intersection_df["id"].duplicated(keep=False)
    if duplicate_ids.any():
        duplicate_examples = (
            intersection_df.loc[duplicate_ids, "id"].drop_duplicates().head(10).tolist()
        )
        raise ValueError(
            "Intersection CSV contains duplicate image IDs, e.g. "
            f"{duplicate_examples}"
        )

    metadata_ids = intersection_df["image_path"].map(image_id_from_path)
    mismatch_mask = metadata_ids != intersection_df["id"]
    if mismatch_mask.any():
        mismatch_rows = intersection_df.loc[
            mismatch_mask, ["id", "image_path"]
        ].head(10)
        raise ValueError(
            "Intersection CSV contains image_path/id mismatches, e.g. "
            f"{mismatch_rows.to_dict(orient='records')}"
        )

    return intersection_df.rename(columns={"timeofday": "time_of_day"})


def attach_evaluation_metadata(
    held_out_df: pd.DataFrame,
    intersection_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach weather/time-of-day metadata to the held-out evaluation rows.

    Steps
    -----
    1. Inner-join held-out segmentation rows to the intersection CSV by image ID.
    2. Require every matched row to belong to the 10K training image split.
    3. Verify the held-out and metadata image basenames identify the same image.
    4. Remove ``undefined`` environment labels because they are not evaluable pairs.

    Parameters
    ----------
    held_out_df : pd.DataFrame
        Deterministic 20% held-out segmentation rows with an ``id`` column.
    intersection_df : pd.DataFrame
        Validated image-level metadata intersection rows.

    Returns
    -------
    pd.DataFrame
        Held-out rows with valid weather and time-of-day metadata.

    Raises
    ------
    ValueError
        If no held-out rows have metadata, a matched row is not a 10K training
        image, image IDs disagree, or no defined environment rows remain.
    """
    annotated_df = held_out_df.merge(
        intersection_df,
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if annotated_df.empty:
        raise ValueError(
            "No held-out segmentation images were found in the metadata intersection."
        )

    # Evaluator holdout rows originate from images_10k/train, so a metadata row
    # declaring another image split indicates a stale or inconsistent intersection.
    wrong_split_mask = annotated_df["image_split"] != "train"
    if wrong_split_mask.any():
        wrong_rows = annotated_df.loc[
            wrong_split_mask, ["id", "image_split"]
        ].head(10)
        raise ValueError(
            "Held-out rows must map to the 10K training split, but found: "
            f"{wrong_rows.to_dict(orient='records')}"
        )

    held_out_ids = annotated_df["image_paths"].map(image_id_from_path)
    metadata_ids = annotated_df["image_path"].map(image_id_from_path)
    mismatch_mask = held_out_ids != metadata_ids
    if mismatch_mask.any():
        mismatch_rows = annotated_df.loc[
            mismatch_mask, ["id", "image_paths", "image_path"]
        ].head(10)
        raise ValueError(
            "Held-out and metadata image paths disagree, e.g. "
            f"{mismatch_rows.to_dict(orient='records')}"
        )

    # Undefined attributes cannot form a meaningful environment condition and
    # are excluded identically before pair counts are generated.
    defined_df = annotated_df.loc[
        (annotated_df["weather"] != "undefined")
        & (annotated_df["time_of_day"] != "undefined")
    ].copy()
    if defined_df.empty:
        raise ValueError(
            "No held-out metadata rows have defined weather and time_of_day values."
        )

    return defined_df.reset_index(drop=True)


def summarize_condition_pairs(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Count weather/time-of-day pairs in the actual evaluation population.

    Steps
    -----
    1. Group the held-out metadata rows by weather and time of day.
    2. Count rows in every observed pair.
    3. Sort pairs by descending count for stable, readable output.

    Parameters
    ----------
    annotated_df : pd.DataFrame
        Held-out evaluation rows with defined environment metadata.

    Returns
    -------
    pd.DataFrame
        Columns ``weather``, ``time_of_day``, and ``count`` where every count is
        the exact number of samples the pair-wise evaluators will process.

    Raises
    ------
    ValueError
        If the annotated evaluation population is empty.
    """
    if annotated_df.empty:
        raise ValueError("Cannot summarize condition pairs from an empty DataFrame.")

    # ``groupby(...).size()`` emits only observed pairs, so zero-count Cartesian
    # combinations cannot be mistaken for requested evaluation conditions.
    pair_df = (
        annotated_df.groupby(["weather", "time_of_day"], sort=False)
        .size()
        .reset_index(name="count")
        .sort_values(
            by=["count", "weather", "time_of_day"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    pair_df["count"] = pair_df["count"].astype(int)
    return pair_df


def pair_count_map(pair_df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """Convert the pair table into a typed mapping for concise console output.

    Steps
    -----
    1. Iterate through pair rows.
    2. Map each ``(weather, time_of_day)`` tuple to its integer count.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Pair summary containing ``weather``, ``time_of_day``, and ``count``.

    Returns
    -------
    Dict[Tuple[str, str], int]
        Mapping from condition pair to evaluation sample count.
    """
    pair_counts: Dict[Tuple[str, str], int] = {}
    for _, row in pair_df.iterrows():
        pair_counts[(str(row["weather"]), str(row["time_of_day"]))] = int(
            row["count"]
        )
    return pair_counts


def main() -> None:
    """Generate pair counts from the exact held-out evaluation population.

    Steps
    -----
    1. Reconstruct the valid segmentation training pool.
    2. Apply the evaluator's deterministic 80/20 split with seed 768.
    3. Join held-out image IDs to the 10K/100K metadata intersection.
    4. Remove undefined environment labels.
    5. Count observed weather/time-of-day pairs and write the pair CSV.

    Returns
    -------
    None
        Results are written to ``./data/weather_timeofday_pair_data.csv``.

    Raises
    ------
    FileNotFoundError
        If required dataset/intersection files are missing.
    ValueError
        If the dataset, metadata, or resulting evaluation population is invalid.
    OSError
        If an image/mask cannot be opened or the output CSV cannot be written.
    """
    print("Reconstructing held-out segmentation evaluation split...")
    train_pool_df = load_segmentation_train_pool()
    held_out_df = build_held_out_evaluation_split(train_pool_df)
    print(f"Held-out segmentation rows: {len(held_out_df)}")

    print("Loading 10K/100K metadata intersection...")
    intersection_df = load_intersection_metadata(DatasetPath.INTERSECTION_CSV)
    annotated_df = attach_evaluation_metadata(held_out_df, intersection_df)
    print(
        "Held-out rows with defined weather/time-of-day metadata: "
        f"{len(annotated_df)}"
    )

    pair_df = summarize_condition_pairs(annotated_df)
    print("Evaluation pair counts:")
    for (weather, time_of_day), count in pair_count_map(pair_df).items():
        print(f"  {weather}/{time_of_day}: {count}")

    # Suppress the DataFrame index because it has no semantic meaning and the
    # evaluators expect only the three explicit pair-specification columns.
    DatasetPath.OUTPUT_PAIR_CSV.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(DatasetPath.OUTPUT_PAIR_CSV, index=False)
    print(f"Pair CSV written to: {DatasetPath.OUTPUT_PAIR_CSV}")


if __name__ == "__main__":
    main()
