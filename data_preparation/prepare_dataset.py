#!/usr/bin/env python3
"""Create validated train/val/test splits from a class-folder dataset."""

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for Leafy"
    )
    parser.add_argument(
        "--source-data-dir",
        default="./class_folder_data",
        help="Folder with one subdirectory per class",
    )
    parser.add_argument(
        "--output-dir", default="./data_split", help="Destination for split dataset"
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
        help="Skip classes with fewer than this many images",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output directory before writing",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional extra path to save split summary as JSON",
    )
    parser.add_argument(
        "--no-hash-groups",
        action="store_true",
        help="Disable exact duplicate SHA-256 grouping before splitting",
    )
    return parser.parse_args()


def copy_files(file_list, source_dir: Path, destination_dir: Path, class_name: str):
    for file_name in file_list:
        shutil.copy2(source_dir / file_name, destination_dir / class_name / file_name)


def class_file_list(class_dir: Path):
    return sorted(
        file_path.name for file_path in class_dir.iterdir() if file_path.is_file()
    )


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_groups(class_dir: Path, image_names, use_hash_groups=True):
    if not use_hash_groups:
        return [{"group_id": name, "files": [name], "sha256": None} for name in image_names]

    by_hash = defaultdict(list)
    for image_name in image_names:
        by_hash[file_sha256(class_dir / image_name)].append(image_name)
    return [
        {"group_id": sha256, "files": sorted(names), "sha256": sha256}
        for sha256, names in sorted(by_hash.items())
    ]


def flatten_groups(groups):
    return [file_name for group in groups for file_name in group["files"]]


def split_groups(groups, train_ratio, val_ratio, seed, global_hash_to_split=None):
    if len(groups) < 3:
        return groups, [], []

    fixed = {"train": [], "val": [], "test": []}
    unassigned = []
    global_hash_to_split = global_hash_to_split if global_hash_to_split is not None else {}

    for group in groups:
        sha256 = group.get("sha256")
        if sha256 is not None and sha256 in global_hash_to_split:
            fixed[global_hash_to_split[sha256]].append(group)
        else:
            unassigned.append(group)

    shuffled = list(unassigned)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total == 0:
        return fixed["train"], fixed["val"], fixed["test"]
    if total < 3:
        for index, group in enumerate(shuffled):
            split_name = ("train", "val", "test")[index % 3]
            fixed[split_name].append(group)
            if group.get("sha256") is not None:
                global_hash_to_split[group["sha256"]] = split_name
        return fixed["train"], fixed["val"], fixed["test"]

    train_count = max(1, int(round(total * train_ratio)))
    val_count = max(1, int(round(total * val_ratio)))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        train_count = max(1, total - val_count - test_count)

    split_parts = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }
    for split_name, split_groups_for_name in split_parts.items():
        for group in split_groups_for_name:
            if group.get("sha256") is not None:
                global_hash_to_split[group["sha256"]] = split_name
        fixed[split_name].extend(split_groups_for_name)
    return fixed["train"], fixed["val"], fixed["test"]


def split_class_counts(split_root: Path, split_name: str):
    root = split_root / split_name
    if not root.exists():
        return {}
    return {
        class_dir.name: len(class_file_list(class_dir))
        for class_dir in sorted(root.iterdir())
        if class_dir.is_dir()
    }


def validate_split(output_dir: Path, kept_summary):
    split_counts = {
        "train": split_class_counts(output_dir, "train"),
        "val": split_class_counts(output_dir, "val"),
        "test": split_class_counts(output_dir, "test"),
    }
    kept_classes = set(kept_summary)
    errors = []

    for split_name, counts in split_counts.items():
        missing = kept_classes - set(counts)
        extra = set(counts) - kept_classes
        if missing:
            errors.append(f"{split_name} missing classes: {sorted(missing)[:10]}")
        if extra:
            errors.append(f"{split_name} extra classes: {sorted(extra)[:10]}")

    for class_name, expected in kept_summary.items():
        actual_total = sum(
            split_counts[split].get(class_name, 0) for split in split_counts
        )
        if actual_total != expected["source"]:
            errors.append(
                f"{class_name} copied {actual_total}, expected {expected['source']}"
            )
        for split_name in split_counts:
            if split_counts[split_name].get(class_name, 0) <= 0:
                errors.append(f"{class_name} has empty {split_name} split")

    return {"ok": not errors, "errors": errors, "split_counts": split_counts}


def find_cross_split_duplicate_hashes(output_dir: Path):
    hash_splits = defaultdict(set)
    for split_name in ("train", "val", "test"):
        for path in (output_dir / split_name).rglob("*"):
            if path.is_file():
                hash_splits[file_sha256(path)].add(split_name)
    return {
        digest: sorted(splits)
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    }


def build_fingerprint(source_dir, output_dir, split_summary, args):
    payload = {
        name: item["source"]
        for name, item in sorted(split_summary["kept_classes"].items())
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return {
        "num_classes": len(payload),
        "total_images": sum(payload.values()),
        "class_counts_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_data_dir": str(source_dir),
        "split_dir": str(output_dir),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "min_images_per_class": args.min_images_per_class,
    }


def main():
    args = parse_args()
    source_dir = Path(args.source_data_dir)
    output_dir = Path(args.output_dir)
    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    if args.train_ratio <= 0 or args.val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must all be positive")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)

    train_path = output_dir / "train"
    val_path = output_dir / "val"
    test_path = output_dir / "test"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    split_summary = {
        "source_data_dir": str(source_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": test_ratio,
        },
        "hash_grouping": not args.no_hash_groups,
        "kept_classes": {},
        "skipped_classes": {},
        "duplicate_hash_groups": {},
    }
    class_dirs = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    global_hash_to_split = {}

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

        groups = build_file_groups(
            class_dir, image_names, use_hash_groups=not args.no_hash_groups
        )
        duplicate_groups = [group for group in groups if len(group["files"]) > 1]
        if duplicate_groups:
            split_summary["duplicate_hash_groups"][class_name] = [
                {
                    "sha256": group["sha256"],
                    "count": len(group["files"]),
                    "files": group["files"],
                }
                for group in duplicate_groups
            ]

        train_groups, val_groups, test_groups = split_groups(
            groups,
            args.train_ratio,
            args.val_ratio,
            args.seed,
            global_hash_to_split=global_hash_to_split,
        )
        train_images = flatten_groups(train_groups)
        val_images = flatten_groups(val_groups)
        test_images = flatten_groups(test_groups)

        for split_path in (train_path, val_path, test_path):
            (split_path / class_name).mkdir(parents=True, exist_ok=True)

        copy_files(train_images, class_dir, train_path, class_name)
        copy_files(val_images, class_dir, val_path, class_name)
        copy_files(test_images, class_dir, test_path, class_name)

        split_summary["kept_classes"][class_name] = {
            "source": image_count,
            "groups": len(groups),
            "duplicate_groups": len(duplicate_groups),
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
        }

    validation = validate_split(output_dir, split_summary["kept_classes"])
    cross_split_duplicates = (
        {} if args.no_hash_groups else find_cross_split_duplicate_hashes(output_dir)
    )
    split_summary["cross_split_duplicate_hashes"] = cross_split_duplicates
    if cross_split_duplicates:
        validation["ok"] = False
        validation["errors"].append(
            f"{len(cross_split_duplicates)} exact duplicate hash groups span multiple splits"
        )
    split_summary["validation"] = validation
    split_summary["dataset_fingerprint"] = build_fingerprint(
        source_dir, output_dir, split_summary, args
    )

    summary_path = output_dir / "split_summary.json"
    fingerprint_path = output_dir / "dataset_fingerprint.json"
    summary_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    fingerprint_path.write_text(
        json.dumps(split_summary["dataset_fingerprint"], indent=2),
        encoding="utf-8",
    )

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
        print(f"Split summary also saved to: {report_path}")

    print(f"Created split dataset at: {output_dir}")
    print(f"Classes kept:    {len(split_summary['kept_classes'])}")
    print(f"Classes skipped: {len(split_summary['skipped_classes'])}")
    print(f"Summary saved:   {summary_path}")
    print(f"Fingerprint:     {fingerprint_path}")

    if not validation["ok"]:
        print("Split validation failed:")
        for error in validation["errors"]:
            print(f"  - {error}")
        raise SystemExit(1)
    print("Split validation: ok")


if __name__ == "__main__":
    main()
