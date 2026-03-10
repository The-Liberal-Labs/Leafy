#!/usr/bin/env python3
"""
Merge duplicate class folders in the raw dataset before re-splitting.

Duplicate pairs identified from classification report analysis:
  1. Bell_pepper___bacterial_spot + Pepper_bell__bacterial_spot → Pepper_bell___bacterial_spot
  2. Bell_pepper___healthy + Pepper_bell__healthy              → Pepper_bell___healthy
  3. Tomato___leaf_curl + Tomato__yellow_leaf_curl_virus        → Tomato___leaf_curl

Usage:
    python merge_duplicates.py                        # Default: ./data
    python merge_duplicates.py --data-dir /path/to/data
    python merge_duplicates.py --dry-run              # Preview only, no changes
"""

import argparse
import os
import shutil
from pathlib import Path


# (source_to_merge, target_to_keep)
# All images from source are moved into target, then source is deleted.
MERGE_PAIRS = [
    ("Bell_pepper___bacterial_spot", "Pepper_bell___bacterial_spot"),
    ("Bell_pepper___healthy", "Pepper_bell___healthy"),
    ("Tomato__yellow_leaf_curl_virus", "Tomato___leaf_curl"),
]


def merge_class(data_dir: Path, source_name: str, target_name: str, dry_run: bool):
    """Move all images from source_name into target_name, handling filename collisions."""
    source_dir = data_dir / source_name
    target_dir = data_dir / target_name

    if not source_dir.exists():
        print(f"  ⚠️  Source '{source_name}' not found — skipping")
        return 0

    if not target_dir.exists():
        if dry_run:
            print(f"  [DRY RUN] Would create '{target_name}/'")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"  📁 Created '{target_name}/'")

    source_files = [f for f in source_dir.iterdir() if f.is_file()]
    target_files = {f.name for f in target_dir.iterdir() if f.is_file()} if target_dir.exists() else set()
    moved = 0

    for src_file in source_files:
        dest_name = src_file.name
        # Handle filename collisions by prefixing with source class name
        if dest_name in target_files:
            stem = src_file.stem
            suffix = src_file.suffix
            dest_name = f"{source_name}_{stem}{suffix}"

        dest_path = target_dir / dest_name

        if dry_run:
            collision_note = " (renamed)" if dest_name != src_file.name else ""
            # Only print a few to avoid flooding
            if moved < 3:
                print(f"    [DRY RUN] {src_file.name} → {target_name}/{dest_name}{collision_note}")
            elif moved == 3:
                print(f"    ... and {len(source_files) - 3} more files")
        else:
            shutil.move(str(src_file), str(dest_path))
        moved += 1
        target_files.add(dest_name)

    # Remove empty source directory
    if not dry_run:
        if source_dir.exists() and not any(source_dir.iterdir()):
            source_dir.rmdir()
            print(f"  🗑️  Removed empty '{source_name}/'")

    return moved


def main():
    parser = argparse.ArgumentParser(description="Merge duplicate class folders")
    parser.add_argument("--data-dir", default="./data", help="Path to raw data directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    print(f"\n{'═' * 60}")
    print(f"  🔄 MERGING DUPLICATE CLASSES")
    if args.dry_run:
        print(f"  ⚠️  DRY RUN — no files will be modified")
    print(f"{'═' * 60}")
    print(f"  Data dir: {data_dir.resolve()}\n")

    total_moved = 0
    for source_name, target_name in MERGE_PAIRS:
        source_dir = data_dir / source_name
        target_dir = data_dir / target_name
        source_count = len([f for f in source_dir.iterdir() if f.is_file()]) if source_dir.exists() else 0
        target_count = len([f for f in target_dir.iterdir() if f.is_file()]) if target_dir.exists() else 0

        print(f"  📋 {source_name} ({source_count} imgs) → {target_name} ({target_count} imgs)")
        moved = merge_class(data_dir, source_name, target_name, args.dry_run)
        if moved:
            new_count = source_count + target_count
            print(f"     ✅ Merged {moved} images → {target_name} now has {new_count} images\n")
        total_moved += moved

    # Summary
    remaining_classes = len([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"\n{'─' * 60}")
    print(f"  Total images moved: {total_moved}")
    print(f"  Remaining classes:  {remaining_classes}")
    print(f"{'─' * 60}")

    if not args.dry_run:
        print(f"\n  ✅ Done! Now re-run data splitting:")
        print(f"     python prepare_data.py --source-data-dir {data_dir} --output-dir ./new_data --min-images-per-class 50")


if __name__ == "__main__":
    main()
