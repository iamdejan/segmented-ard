from pathlib import Path
import csv
import json

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

LABELS_BASE = Path("./data/bdd100k/labels/100k")
IMAGES_BASE = Path("./data/bdd100k/images_10k")

OUTPUT_CSV = Path("./data/bdd100k_10k_images_with_100k_metadata_intersection.csv")

SPLITS = ["train", "test", "val"]


def collect_files(base_dir: Path, extension: str) -> dict[str, list[tuple[str, Path]]]:
    """Collect files grouped by filename stem (BDD100K image/label ID)."""
    files_by_id: dict[str, list[tuple[str, Path]]] = {}

    for split in SPLITS:
        split_dir = base_dir / split

        if not split_dir.exists():
            print(f"Warning: directory does not exist: {split_dir}")
            continue

        for file_path in split_dir.glob(f"*{extension}"):
            file_id = file_path.stem
            files_by_id.setdefault(file_id, []).append((split, file_path))

    return files_by_id


def load_environment_metadata(label_path: Path, expected_id: str) -> tuple[str, str]:
    """Read and strictly validate weather and time-of-day metadata from one JSON file.

    According to the supplied schema, both values live at:

        metadata["attributes"]["weather"]
        metadata["attributes"]["timeofday"]

    This function intentionally raises an error instead of substituting guessed or
    default values when an intersecting metadata file is malformed or incomplete.
    """
    try:
        with label_path.open("r", encoding="utf-8") as json_file:
            metadata = json.load(json_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in metadata file: {label_path}") from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata root must be a JSON object: {label_path}")

    name = metadata.get("name")
    if not isinstance(name, str):
        raise ValueError(f"Missing or non-string 'name' field: {label_path}")

    if name != expected_id:
        raise ValueError(
            f"Metadata ID mismatch in {label_path}: filename ID is "
            f"'{expected_id}', but JSON 'name' is '{name}'."
        )

    attributes = metadata.get("attributes")
    if not isinstance(attributes, dict):
        raise ValueError(f"Missing or non-object 'attributes' field: {label_path}")

    weather = attributes.get("weather")
    timeofday = attributes.get("timeofday")

    if not isinstance(weather, str) or not weather:
        raise ValueError(
            f"Missing, empty, or non-string 'attributes.weather': {label_path}"
        )

    if not isinstance(timeofday, str) or not timeofday:
        raise ValueError(
            f"Missing, empty, or non-string 'attributes.timeofday': {label_path}"
        )

    return weather, timeofday


def main() -> None:
    print("Scanning BDD100K files...")

    # Example:
    # cabc30fc-e7726578.json
    # -> ID = cabc30fc-e7726578
    labels = collect_files(LABELS_BASE, ".json")

    # Example:
    # ac6d4f42-00000000.jpg
    # -> ID = ac6d4f42-00000000
    images = collect_files(IMAGES_BASE, ".jpg")

    print(f"100K label IDs found : {len(labels)}")
    print(f"10K image IDs found  : {len(images)}")

    # IDs that exist in BOTH collections, regardless of split.
    intersection_ids = sorted(set(labels.keys()) & set(images.keys()))

    print(f"Intersection IDs      : {len(intersection_ids)}")

    # -----------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "id",
                "weather",
                "timeofday",
                "label_split",
                "image_split",
                "same_split",
                "label_path",
                "image_path",
            ],
        )

        writer.writeheader()

        for file_id in intersection_ids:
            label_entries = labels[file_id]
            image_entries = images[file_id]

            # Usually there is only one label/image per ID. If duplicate IDs exist
            # across splits, emit every label-image pairing rather than discarding
            # data silently.
            for label_split, label_path in label_entries:
                weather, timeofday = load_environment_metadata(label_path, file_id)

                for image_split, image_path in image_entries:
                    writer.writerow(
                        {
                            "id": file_id,
                            "weather": weather,
                            "timeofday": timeofday,
                            "label_split": label_split,
                            "image_split": image_split,
                            "same_split": label_split == image_split,
                            "label_path": str(label_path),
                            "image_path": str(image_path),
                        }
                    )
                    rows_written += 1

    print(f"CSV rows written      : {rows_written}")
    print(f"CSV written to        : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
