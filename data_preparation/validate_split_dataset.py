#!/usr/bin/env python3
"""Validate an existing train/val/test class-folder dataset."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Leafy data_split dataset")
    parser.add_argument("--data-dir", default="./data_split", help="Split dataset root")
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write split_summary.json and dataset_fingerprint.json into data-dir",
    )
    return parser.parse_args()


def image_files(class_dir: Path):
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_split_counts(data_dir: Path):
    split_counts = {}
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        split_counts[split] = {
            class_dir.name: len(image_files(class_dir))
            for class_dir in sorted(split_dir.iterdir())
            if class_dir.is_dir()
        }
    return split_counts


def find_cross_split_duplicate_hashes(data_dir: Path):
    hash_splits = defaultdict(set)
    for split in SPLITS:
        for path in (data_dir / split).rglob("*"):
            if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
                hash_splits[file_sha256(path)].add(split)
    return {
        digest: sorted(splits)
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    }


def build_summary(data_dir: Path):
    split_counts = collect_split_counts(data_dir)
    class_sets = {split: set(counts) for split, counts in split_counts.items()}
    expected_classes = class_sets["train"]
    errors = []

    for split, classes in class_sets.items():
        missing = expected_classes - classes
        extra = classes - expected_classes
        if missing:
            errors.append(f"{split} missing classes: {sorted(missing)}")
        if extra:
            errors.append(f"{split} extra classes: {sorted(extra)}")

    kept_classes = {}
    for class_name in sorted(expected_classes):
        per_split = {split: split_counts[split].get(class_name, 0) for split in SPLITS}
        for split, count in per_split.items():
            if count <= 0:
                errors.append(f"{class_name} has empty {split} split")
        kept_classes[class_name] = {
            "train": per_split["train"],
            "val": per_split["val"],
            "test": per_split["test"],
            "total": sum(per_split.values()),
        }

    cross_split_duplicates = find_cross_split_duplicate_hashes(data_dir)
    if cross_split_duplicates:
        errors.append(
            f"{len(cross_split_duplicates)} exact duplicate hash groups span multiple splits"
        )

    class_count_payload = {
        name: item["total"] for name, item in sorted(kept_classes.items())
    }
    encoded_counts = json.dumps(class_count_payload, sort_keys=True).encode("utf-8")
    fingerprint = {
        "num_classes": len(kept_classes),
        "total_images": sum(class_count_payload.values()),
        "class_counts_sha256": hashlib.sha256(encoded_counts).hexdigest(),
        "primary_data_dir": str(data_dir),
        "split_dir": str(data_dir),
        "splits": list(SPLITS),
    }
    summary = {
        "primary_data_dir": str(data_dir),
        "num_classes": len(kept_classes),
        "total_images": fingerprint["total_images"],
        "split_totals": {
            split: sum(split_counts[split].values()) for split in SPLITS
        },
        "kept_classes": kept_classes,
        "cross_split_duplicate_hashes": cross_split_duplicates,
        "validation": {"ok": not errors, "errors": errors},
        "dataset_fingerprint": fingerprint,
    }
    return summary


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    summary = build_summary(data_dir)
    print(f"Dataset: {data_dir}")
    print(f"Classes: {summary['num_classes']}")
    print(f"Images:  {summary['total_images']:,}")
    print(
        "Splits:  "
        + ", ".join(
            f"{split}={count:,}" for split, count in summary["split_totals"].items()
        )
    )
    print(
        f"Cross-split exact duplicate groups: {len(summary['cross_split_duplicate_hashes'])}"
    )

    if args.write_summary:
        (data_dir / "split_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (data_dir / "dataset_fingerprint.json").write_text(
            json.dumps(summary["dataset_fingerprint"], indent=2), encoding="utf-8"
        )
        (data_dir / "class_names.json").write_text(
            json.dumps(sorted(summary["kept_classes"]), indent=2), encoding="utf-8"
        )
        print(f"Updated: {data_dir / 'split_summary.json'}")
        print(f"Updated: {data_dir / 'dataset_fingerprint.json'}")
        print(f"Updated: {data_dir / 'class_names.json'}")

    if not summary["validation"]["ok"]:
        print("Validation failed:")
        for error in summary["validation"]["errors"]:
            print(f"  - {error}")
        raise SystemExit(1)
    print("Validation: ok")


if __name__ == "__main__":
    main()
