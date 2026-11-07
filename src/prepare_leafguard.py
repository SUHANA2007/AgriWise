
import csv
import json
import random
from collections import Counter
from pathlib import Path


# ============================================================
# LEAFGUARD - DATASET PREPARATION
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

EXPECTED_CLASSES = 38
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():

    print("=" * 65)
    print("LEAFGUARD - DATASET PREPARATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Check dataset directory
    # --------------------------------------------------------

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDataset directory:")
    print(DATASET_DIR)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # 2. Identify the 38 plant classes
    # --------------------------------------------------------

    class_folders = sorted(
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
        and folder.name != BACKGROUND_FOLDER
    )

    if len(class_folders) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"but found {len(class_folders)}."
        )

    class_mapping = {
        folder.name: index
        for index, folder in enumerate(class_folders)
    }

    print(f"\nPlant classes found: {len(class_folders)}")
    print(f"Background folder excluded: {BACKGROUND_FOLDER}")

    # --------------------------------------------------------
    # 3. Collect image paths class by class
    # --------------------------------------------------------

    class_images = {}

    for folder in class_folders:

        images = sorted(
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) < 3:
            raise ValueError(
                f"Not enough images in class: {folder.name}"
            )

        class_images[folder.name] = images

    total_images = sum(
        len(images)
        for images in class_images.values()
    )

    print(f"Total leaf images found: {total_images:,}")

    # --------------------------------------------------------
    # 4. Create stratified train / validation / test splits
    # --------------------------------------------------------

    train_records = []
    validation_records = []
    test_records = []

    rng = random.Random(RANDOM_SEED)

    for class_name, images in class_images.items():

        shuffled_images = images.copy()
        rng.shuffle(shuffled_images)

        total = len(shuffled_images)

        test_count = round(total * TEST_RATIO)
        validation_count = round(total * VALIDATION_RATIO)

        train_count = total - validation_count - test_count

        train_images = shuffled_images[:train_count]

        validation_images = shuffled_images[
            train_count:train_count + validation_count
        ]

        test_images = shuffled_images[
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

    # --------------------------------------------------------
    # 5. Shuffle records within each split
    # --------------------------------------------------------

    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    rng.shuffle(test_records)

    # --------------------------------------------------------
    # 6. Save CSV files
    # --------------------------------------------------------

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

    save_csv("train.csv", train_records)
    save_csv("validation.csv", validation_records)
    save_csv("test.csv", test_records)

    # --------------------------------------------------------
    # 7. Save class mapping
    # --------------------------------------------------------

    mapping_file = OUTPUT_DIR / "class_mapping.json"

    with open(
        mapping_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_mapping,
            file,
            indent=4
        )

    print(f"Saved: {mapping_file}")

    # --------------------------------------------------------
    # 8. Save class-wise split summary
    # --------------------------------------------------------

    train_counts = Counter(
        record["class_name"]
        for record in train_records
    )

    validation_counts = Counter(
        record["class_name"]
        for record in validation_records
    )

    test_counts = Counter(
        record["class_name"]
        for record in test_records
    )

    summary_file = OUTPUT_DIR / "split_summary.csv"

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "class_name",
            "total_images",
            "training_images",
            "validation_images",
            "testing_images"
        ])

        for class_name in class_mapping:

            writer.writerow([
                class_name,
                len(class_images[class_name]),
                train_counts[class_name],
                validation_counts[class_name],
                test_counts[class_name]
            ])

    print(f"Saved: {summary_file}")

    # --------------------------------------------------------
    # 9. Final verification
    # --------------------------------------------------------

    split_total = (
        len(train_records)
        + len(validation_records)
        + len(test_records)
    )

    if split_total != total_images:
        raise RuntimeError(
            "Split verification failed: image counts do not match."
        )

    print("\n" + "=" * 65)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 65)

    print(f"Classes:             {len(class_mapping)}")
    print(f"Total images:        {total_images:,}")

    print(
        f"Training images:     {len(train_records):,} "
        f"({len(train_records) / total_images * 100:.2f}%)"
    )

    print(
        f"Validation images:   {len(validation_records):,} "
        f"({len(validation_records) / total_images * 100:.2f}%)"
    )

    print(
        f"Testing images:      {len(test_records):,} "
        f"({len(test_records) / total_images * 100:.2f}%)"
    )

    print("\nAll image paths assigned exactly once.")
    print("Original dataset remains unchanged.")

    print("\nLEAFGUARD DATASET PREPARATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
