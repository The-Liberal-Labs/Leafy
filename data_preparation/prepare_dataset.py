#!/usr/bin/env python3
"""Create train/val/test splits from a class-folder dataset."""

import argparse
import json
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for Leafy"
    )
    parser.add_argument(
        "--source-data-dir",
        default="./data",
        help="Folder with one subdirectory per class",
    )
    parser.add_argument(
        "--output-dir", default="./new_data", help="Destination for the split dataset"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.80, help="Training split ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.10, help="Validation split ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
    )
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=80,
        help="Skip classes with fewer than this many source images",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output directory before writing",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to save split summary as JSON",
    )
    return parser.parse_args()


def copy_files(file_list, source_dir: Path, destination_dir: Path, class_name: str):
    for file_name in file_list:
        shutil.copy2(source_dir / file_name, destination_dir / class_name / file_name)


def class_file_list(class_dir: Path):
    return sorted(
        file_path.name for file_path in class_dir.iterdir() if file_path.is_file()
    )


def main():
    args = parse_args()
    source_dir = Path(args.source_data_dir)
    output_dir = Path(args.output_dir)
    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    train_path = output_dir / "train"
    val_path = output_dir / "val"
    test_path = output_dir / "test"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    split_summary = {"kept_classes": {}, "skipped_classes": {}}
    class_dirs = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_names = class_file_list(class_dir)
        image_count = len(image_names)

        if image_count == 0:
            split_summary["skipped_classes"][class_name] = {
                "count": 0,
                "reason": "empty",
            }
            continue
        if image_count < args.min_images_per_class:
            split_summary["skipped_classes"][class_name] = {
                "count": image_count,
                "reason": f"below_threshold_{args.min_images_per_class}",
            }
            continue

        (train_path / class_name).mkdir(parents=True, exist_ok=True)
        (val_path / class_name).mkdir(parents=True, exist_ok=True)
        (test_path / class_name).mkdir(parents=True, exist_ok=True)

        train_images, val_test_images = train_test_split(
            image_names,
            test_size=(1.0 - args.train_ratio),
            random_state=args.seed,
        )

        relative_test_ratio = args.val_ratio / (1.0 - args.train_ratio)
        if len(val_test_images) < 2:
            val_images = []
            test_images = []
            train_images = image_names
        else:
            val_images, test_images = train_test_split(
                val_test_images,
                test_size=relative_test_ratio,
                random_state=args.seed,
            )

        copy_files(train_images, class_dir, train_path, class_name)
        copy_files(val_images, class_dir, val_path, class_name)
        copy_files(test_images, class_dir, test_path, class_name)

        split_summary["kept_classes"][class_name] = {
            "source": image_count,
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
        }

    print(f"Created split dataset at: {output_dir}")
    print(f"Classes kept:    {len(split_summary['kept_classes'])}")
    print(f"Classes skipped: {len(split_summary['skipped_classes'])}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
        print(f"Split summary saved to: {report_path}")


if __name__ == "__main__":
    main()
