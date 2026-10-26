from pathlib import Path
import csv

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

LABELS_BASE = Path("./data/bdd100k/labels/100k")
IMAGES_BASE = Path("./data/bdd100k/images_10k")

OUTPUT_CSV = Path("./data/bdd100k_10k_images_with_100k_metadata_intersection.csv")

SPLITS = ["train", "test", "val"]


def collect_files(base_dir: Path, extension: str) -> dict[str, list[tuple[str, Path]]]:
    """
    Collect files grouped by ID.

    Returns
    -------
    dict
        {
            "cabc30fc-e7726578": [
                ("train", Path(".../cabc30fc-e7726578.json"))
            ],
            ...
        }
    """
    files_by_id: dict[str, list[tuple[str, Path]]] = {}

    for split in SPLITS:
        split_dir = base_dir / split

        if not split_dir.exists():
            print(f"Warning: directory does not exist: {split_dir}")
            continue

        for file_path in split_dir.glob(f"*{extension}"):
            file_id = file_path.stem

            files_by_id.setdefault(file_id, []).append(
                (split, file_path)
            )

    return files_by_id


def main() -> None:
    print("Scanning BDD100K files...")

    # Example:
    # cabc30fc-e7726578.json
    # -> ID = cabc30fc-e7726578
    labels = collect_files(
        LABELS_BASE,
        ".json",
    )

    # Example:
    # ac6d4f42-00000000.jpg
    # -> ID = ac6d4f42-00000000
    images = collect_files(
        IMAGES_BASE,
        ".jpg",
    )

    print(f"100K label IDs found : {len(labels)}")
    print(f"10K image IDs found  : {len(images)}")

    # IDs that exist in BOTH collections.
    intersection_ids = sorted(
        set(labels.keys()) & set(images.keys())
    )

    print(f"Intersection IDs      : {len(intersection_ids)}")

    # -----------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "id",
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

            # Usually there should only be one label/image per ID,
            # but this handles duplicates across splits safely.
            for label_split, label_path in label_entries:
                for image_split, image_path in image_entries:
                    writer.writerow({
                        "id": file_id,
                        "label_split": label_split,
                        "image_split": image_split,
                        "same_split": label_split == image_split,
                        "label_path": str(label_path),
                        "image_path": str(image_path),
                    })

    print()
    print(f"CSV written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
