#!/usr/bin/env python3
"""Report exact duplicate image hashes in a class-folder dataset."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Find duplicate image files")
    parser.add_argument("--data-dir", default="./data_split", help="Dataset root")
    parser.add_argument(
        "--report-json",
        default="./reports/duplicate_report.json",
        help="Path to write duplicate report JSON",
    )
    parser.add_argument(
        "--include-perceptual",
        action="store_true",
        help="Also compute average perceptual hashes with PIL",
    )
    return parser.parse_args()


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, hash_size=8):
    with Image.open(path) as image:
        image = image.convert("L").resize((hash_size, hash_size))
        pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)
    bits = ["1" if pixel > avg else "0" for pixel in pixels]
    return f"{int(''.join(bits), 2):0{hash_size * hash_size // 4}x}"


def gather_images(root: Path):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def duplicate_groups(index):
    return {
        digest: entries
        for digest, entries in sorted(index.items())
        if len(entries) > 1
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    images = gather_images(data_dir)
    exact_index = defaultdict(list)
    perceptual_index = defaultdict(list)

    for path in tqdm(images, desc="Hashing images"):
        relative = str(path.relative_to(data_dir))
        exact_index[file_sha256(path)].append(relative)
        if args.include_perceptual:
            try:
                perceptual_index[average_hash(path)].append(relative)
            except Exception as exc:
                perceptual_index[f"error:{exc}"].append(relative)

    exact_duplicates = duplicate_groups(exact_index)
    perceptual_duplicates = (
        duplicate_groups(perceptual_index) if args.include_perceptual else {}
    )
    report = {
        "data_dir": str(data_dir),
        "total_images": len(images),
        "exact_duplicate_group_count": len(exact_duplicates),
        "exact_duplicate_file_count": sum(len(v) for v in exact_duplicates.values()),
        "perceptual_duplicate_group_count": len(perceptual_duplicates),
        "perceptual_duplicate_file_count": sum(
            len(v) for v in perceptual_duplicates.values()
        ),
        "exact_duplicates": exact_duplicates,
        "perceptual_duplicates": perceptual_duplicates,
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Images: {len(images):,}")
    print(f"Exact duplicate groups: {len(exact_duplicates):,}")
    if args.include_perceptual:
        print(f"Perceptual duplicate groups: {len(perceptual_duplicates):,}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
