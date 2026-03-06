#!/usr/bin/env python3
import argparse
import os
import shutil

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for Leafy")
    parser.add_argument("--source-data-dir", default="./data", help="Folder with one subdirectory per class")
    parser.add_argument("--output-dir", default="./new_data", help="Destination for the split dataset")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=0,
        help="Skip classes with fewer than this many source images",
    )
    return parser.parse_args()


def copy_files(file_list, source_dir, destination_dir, class_name):
    for file_name in file_list:
        source_file = os.path.join(source_dir, file_name)
        dest_file = os.path.join(destination_dir, class_name, file_name)
        shutil.copy2(source_file, dest_file)


def main():
    args = parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    if os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} already exists. Deleting it.")
        shutil.rmtree(args.output_dir)

    print("Creating new output directory structure...")
    train_path = os.path.join(args.output_dir, "train")
    val_path = os.path.join(args.output_dir, "val")
    test_path = os.path.join(args.output_dir, "test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    disease_classes = sorted(
        d for d in os.listdir(args.source_data_dir)
        if not d.startswith(".") and os.path.isdir(os.path.join(args.source_data_dir, d))
    )
    print(f"Found {len(disease_classes)} classes.")
    if args.min_images_per_class > 0:
        print(f"Skipping classes with fewer than {args.min_images_per_class} images.")

    kept_classes = 0
    skipped_classes = []

    for disease_class in disease_classes:
        print(f"Processing class: {disease_class}")

        class_dir = os.path.join(args.source_data_dir, disease_class)
        images = sorted(
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        )

        if not images:
            print(f"  [WARNING] No images found for class '{disease_class}'. Skipping.")
            skipped_classes.append((disease_class, 0, "empty"))
            continue

        if len(images) < args.min_images_per_class:
            print(
                f"  [SKIP] Only {len(images)} images for '{disease_class}' "
                f"(threshold: {args.min_images_per_class})."
            )
            skipped_classes.append((disease_class, len(images), "below_threshold"))
            continue

        os.makedirs(os.path.join(train_path, disease_class), exist_ok=True)
        os.makedirs(os.path.join(val_path, disease_class), exist_ok=True)
        os.makedirs(os.path.join(test_path, disease_class), exist_ok=True)

        train_images, val_test_images = train_test_split(
            images,
            test_size=(1.0 - args.train_ratio),
            random_state=args.seed,
        )

        relative_test_size = args.val_ratio / (1.0 - args.train_ratio)
        if len(val_test_images) < 2:
            print(
                f"  [WARNING] Not enough images for class '{disease_class}' "
                "to create validation/test splits. Placing all in training."
            )
            train_images = images
            val_images = []
            test_images = []
        else:
            val_images, test_images = train_test_split(
                val_test_images,
                test_size=relative_test_size,
                random_state=args.seed,
            )

        copy_files(train_images, class_dir, train_path, disease_class)
        copy_files(val_images, class_dir, val_path, disease_class)
        copy_files(test_images, class_dir, test_path, disease_class)
        kept_classes += 1

    print("\nData splitting complete!")
    print(f"Data has been split into train/val/test in: {args.output_dir}")
    print(f"Classes kept: {kept_classes}")
    print(f"Classes skipped: {len(skipped_classes)}")

    if skipped_classes:
        print("\nSkipped classes:")
        for class_name, count, reason in skipped_classes:
            print(f"  {class_name}: {count} images ({reason})")


if __name__ == "__main__":
    main()
