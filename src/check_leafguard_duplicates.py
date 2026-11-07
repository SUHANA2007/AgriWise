
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "leafguard"

SPLIT_FILES = [
    "train.csv",
    "validation.csv",
    "test.csv"
]

REPORT_FILE = DATA_DIR / "duplicate_image_report.csv"


def calculate_hash(image_path):
    """Calculate SHA-256 hash without loading the whole image into RAM."""

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():

    print("=" * 65)
    print("LEAFGUARD - EXACT DUPLICATE IMAGE AUDIT")
    print("=" * 65)

    hash_records = defaultdict(list)
    total_images = 0

    # Read all three splits
    for split_file in SPLIT_FILES:

        csv_path = DATA_DIR / split_file

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        split_name = split_file.replace(".csv", "")

        with open(csv_path, newline="", encoding="utf-8") as file:
            records = list(csv.DictReader(file))

        print(f"\nChecking {split_name}: {len(records):,} images")

        for index, record in enumerate(records, start=1):

            relative_path = record["image_path"]
            image_path = PROJECT_DIR / relative_path

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            image_hash = calculate_hash(image_path)

            hash_records[image_hash].append({
                "split": split_name,
                "image_path": relative_path,
                "class_name": record["class_name"]
            })

            total_images += 1

            if index % 5000 == 0:
                print(f"  Processed: {index:,}/{len(records):,}")

    # Identify duplicate groups
    duplicate_groups = {
        image_hash: records
        for image_hash, records in hash_records.items()
        if len(records) > 1
    }

    cross_split_groups = {
        image_hash: records
        for image_hash, records in duplicate_groups.items()
        if len({record["split"] for record in records}) > 1
    }

    # Save duplicate report
    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sha256",
                "split",
                "image_path",
                "class_name"
            ]
        )

        writer.writeheader()

        for image_hash, records in duplicate_groups.items():

            for record in records:

                writer.writerow({
                    "sha256": image_hash,
                    "split": record["split"],
                    "image_path": record["image_path"],
                    "class_name": record["class_name"]
                })

    # Final summary
    duplicate_file_count = sum(
        len(records)
        for records in duplicate_groups.values()
    )

    print("\n" + "=" * 65)
    print("DUPLICATE AUDIT SUMMARY")
    print("=" * 65)

    print(f"Total images checked: {total_images:,}")
    print(f"Unique image contents: {len(hash_records):,}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Files in duplicate groups: {duplicate_file_count}")
    print(f"Cross-split duplicate groups: {len(cross_split_groups)}")

    print(f"\nReport saved: {REPORT_FILE}")

    if cross_split_groups:
        print("\nWARNING: Exact duplicate images exist across splits.")
        print("We must resolve these before model training.")
    else:
        print("\nNo exact duplicate image content across splits.")

    print("\nDUPLICATE AUDIT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
