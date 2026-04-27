#!/usr/bin/env python3
"""Create a curated raw dataset with merged duplicates and dropped weak classes."""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curate Leafy raw dataset into a reproducible clean taxonomy"
    )
    parser.add_argument(
        "--source-data-dir", default="./data", help="Raw class-folder dataset"
    )
    parser.add_argument(
        "--output-dir", default="./data_curated", help="Curated output directory"
    )
    parser.add_argument(
        "--config",
        default="./configs/dataset_curation/default_leafy_curation.json",
        help="JSON file with merge/drop rules",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output directory before writing",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without copying files"
    )
    return parser.parse_args()


def load_config(config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("merge_into", {}), set(config.get("drop_classes", []))


def unique_destination_name(destination_dir: Path, source_name: str, file_name: str):
    destination = destination_dir / file_name
    if not destination.exists():
        return destination
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    return destination_dir / f"{source_name}_{stem}{suffix}"


def main():
    args = parse_args()
    source_dir = Path(args.source_data_dir)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Curation config not found: {config_path}")

    merge_into, drop_classes = load_config(config_path)

    if output_dir.exists() and not args.dry_run:
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "kept_classes": set(),
        "dropped_classes": [],
        "merged_classes": [],
        "copied_images": 0,
    }

    for class_dir in sorted(
        path
        for path in source_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ):
        source_name = class_dir.name
        if source_name in drop_classes:
            summary["dropped_classes"].append(source_name)
            continue

        target_name = merge_into.get(source_name, source_name)
        if target_name != source_name:
            summary["merged_classes"].append({"from": source_name, "to": target_name})
        summary["kept_classes"].add(target_name)

        image_files = sorted(
            file_path for file_path in class_dir.iterdir() if file_path.is_file()
        )
        if args.dry_run:
            print(f"{source_name} -> {target_name} ({len(image_files)} images)")
            continue

        destination_dir = output_dir / target_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for image_file in image_files:
            destination_path = unique_destination_name(
                destination_dir, source_name, image_file.name
            )
            shutil.copy2(image_file, destination_path)
            summary["copied_images"] += 1

    kept_classes = sorted(summary["kept_classes"])
    print(f"Curated classes: {len(kept_classes)}")
    print(f"Dropped classes: {len(summary['dropped_classes'])}")
    print(f"Merged aliases:  {len(summary['merged_classes'])}")
    if not args.dry_run:
        print(f"Copied images:   {summary['copied_images']:,}")
        report_path = output_dir / "curation_summary.json"
        report_payload = {
            "kept_classes": kept_classes,
            "dropped_classes": sorted(summary["dropped_classes"]),
            "merged_classes": summary["merged_classes"],
            "copied_images": summary["copied_images"],
        }
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"Curation summary saved to: {report_path}")


if __name__ == "__main__":
    main()
