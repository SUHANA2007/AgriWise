
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


# ============================================================
# LEAFGUARD - DUPLICATE-FREE DATASET SPLIT REBUILD
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "leafguard"
    / "Plant_leave_diseases_dataset_without_augmentation"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "leafguard"
)

BACKGROUND_FOLDER = "Background_without_leaves"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def calculate_hash(image_path):

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def save_csv(filename, records):

    output_file = OUTPUT_DIR / filename

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "class_name",
                "class_index"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"Saved: {output_file}")


def main():

    print("=" * 65)
    print("LEAFGUARD - DUPLICATE-FREE SPLIT REBUILD")
    print("=" * 65)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_folders = sorted(
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
        and folder.name != BACKGROUND_FOLDER
    )

    if len(class_folders) != 38:
        raise ValueError(
            f"Expected 38 classes, found {len(class_folders)}"
        )

    class_mapping = {
        folder.name: index
        for index, folder in enumerate(class_folders)
    }

    # --------------------------------------------------------
    # 1. Calculate hashes and group identical images
    # --------------------------------------------------------

    hash_groups = defaultdict(list)

    print("\nChecking original images and calculating hashes...")

    total_images = 0

    for folder in class_folders:

        images = sorted(
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in images:

            image_hash = calculate_hash(image_path)

            hash_groups[image_hash].append({
                "image_path": image_path,
                "class_name": folder.name
            })

            total_images += 1

    print(f"Original images checked: {total_images:,}")

    # --------------------------------------------------------
    # 2. Check for different-class label conflicts
    # --------------------------------------------------------

    for image_hash, records in hash_groups.items():

        labels = {
            record["class_name"]
            for record in records
        }

        if len(labels) > 1:

            print("\nLABEL CONFLICT DETECTED")

            for record in records:
                print(
                    record["class_name"],
                    record["image_path"]
                )

            raise ValueError(
                "Identical image content has different class labels. "
                "Resolve this before rebuilding."
            )

    # --------------------------------------------------------
    # 3. Keep one canonical copy per unique image
    # --------------------------------------------------------

    unique_by_class = defaultdict(list)
    duplicate_report = []

    for image_hash, records in hash_groups.items():

        records = sorted(
            records,
            key=lambda record: str(record["image_path"])
        )

        canonical = records[0]

        unique_by_class[canonical["class_name"]].append(
            canonical["image_path"]
        )

        for duplicate in records[1:]:

            duplicate_report.append({
                "sha256": image_hash,
                "class_name": canonical["class_name"],
                "kept_image": canonical["image_path"].relative_to(
                    PROJECT_DIR
                ).as_posix(),
                "excluded_duplicate": duplicate[
                    "image_path"
                ].relative_to(PROJECT_DIR).as_posix()
            })

    unique_total = sum(
        len(images)
        for images in unique_by_class.values()
    )

    print(f"Unique images retained: {unique_total:,}")
    print(f"Extra copies excluded: {len(duplicate_report)}")

    # --------------------------------------------------------
    # 4. Save duplicate resolution report
    # --------------------------------------------------------

    report_file = OUTPUT_DIR / "duplicate_resolution_report.csv"

    with open(
        report_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sha256",
                "class_name",
                "kept_image",
                "excluded_duplicate"
            ]
        )

        writer.writeheader()
        writer.writerows(duplicate_report)

    print(f"Saved: {report_file}")

    # --------------------------------------------------------
    # 5. Create new stratified splits
    # --------------------------------------------------------

    rng = random.Random(RANDOM_SEED)

    train_records = []
    validation_records = []
    test_records = []

    split_summary = []

    for class_name in class_mapping:

        images = sorted(unique_by_class[class_name])

        rng.shuffle(images)

        total = len(images)

        test_count = round(total * TEST_RATIO)
        validation_count = round(total * VALIDATION_RATIO)

        train_count = total - validation_count - test_count

        train_images = images[:train_count]

        validation_images = images[
            train_count:train_count + validation_count
        ]

        test_images = images[
            train_count + validation_count:
        ]

        class_index = class_mapping[class_name]

        def create_records(image_list):

            return [
                {
                    "image_path": image.relative_to(
                        PROJECT_DIR
                    ).as_posix(),
                    "class_name": class_name,
                    "class_index": class_index
                }
                for image in image_list
            ]

        train_records.extend(create_records(train_images))

        validation_records.extend(
            create_records(validation_images)
        )

        test_records.extend(create_records(test_images))

        split_summary.append({
            "class_name": class_name,
            "total_images": total,
            "training_images": len(train_images),
            "validation_images": len(validation_images),
            "testing_images": len(test_images)
        })

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    rng.shuffle(test_records)

    # --------------------------------------------------------
    # 6. Save all new split files
    # --------------------------------------------------------

    save_csv("train.csv", train_records)
    save_csv("validation.csv", validation_records)
    save_csv("test.csv", test_records)

    # --------------------------------------------------------
    # 7. Save class mapping
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR / "class_mapping.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(class_mapping, file, indent=4)

    # --------------------------------------------------------
    # 8. Save split summary
    # --------------------------------------------------------

    summary_file = OUTPUT_DIR / "split_summary.csv"

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_name",
                "total_images",
                "training_images",
                "validation_images",
                "testing_images"
            ]
        )

        writer.writeheader()
        writer.writerows(split_summary)

    # --------------------------------------------------------
    # 9. Final checks
    # --------------------------------------------------------

    split_total = (
        len(train_records)
        + len(validation_records)
        + len(test_records)
    )

    if split_total != unique_total:
        raise RuntimeError("Split total verification failed.")

    print("\n" + "=" * 65)
    print("NEW DATASET SPLIT SUMMARY")
    print("=" * 65)

    print(f"Classes: {len(class_mapping)}")
    print(f"Original images: {total_images:,}")
    print(f"Unique images: {unique_total:,}")
    print(f"Excluded duplicate copies: {len(duplicate_report)}")

    print(f"Training: {len(train_records):,}")
    print(f"Validation: {len(validation_records):,}")
    print(f"Testing: {len(test_records):,}")

    print("\nOriginal images were not modified.")
    print("New duplicate-free splits created successfully.")

    print("=" * 65)


if __name__ == "__main__":
    main()
