#!/usr/bin/env python3
"""Validate images and optionally delete corrupted files."""

import argparse
import concurrent.futures
from pathlib import Path

from PIL import Image
from tqdm import tqdm


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_FORMATS = {"jpeg", "png", "bmp", "webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate and clean image datasets")
    parser.add_argument(
        "--data-dir", default="./data_split", help="Dataset root directory"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Override worker count"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete invalid files instead of dry-run only",
    )
    return parser.parse_args()


def verify_image(file_path: Path):
    try:
        with Image.open(file_path) as image:
            image.load()
            if (image.format or "").lower() not in VALID_FORMATS:
                return str(file_path), f"unsupported format: {image.format}"
    except Exception as exc:  # pragma: no cover - PIL error types vary
        return str(file_path), str(exc)
    return None


def gather_image_paths(root_dir: Path):
    return [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    ]


def main():
    args = parse_args()
    root_dir = Path(args.data_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    image_paths = gather_image_paths(root_dir)
    print(f"Scanning {len(image_paths):,} images under {root_dir.resolve()}")

    corrupted = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(verify_image, image_paths)
        for result in tqdm(results, total=len(image_paths), desc="Verifying images"):
            if result is not None:
                corrupted.append(result)

    if not corrupted:
        print("No corrupted or unsupported files detected.")
        return

    print(f"Found {len(corrupted)} invalid files.")
    for file_path, reason in corrupted[:20]:
        print(f"  {file_path} -> {reason}")
    if len(corrupted) > 20:
        print(f"  ... and {len(corrupted) - 20} more")

    if not args.delete:
        print("Dry run only. Re-run with --delete to remove invalid files.")
        return

    for file_path, _ in corrupted:
        Path(file_path).unlink(missing_ok=True)
    print(f"Deleted {len(corrupted)} invalid files.")


if __name__ == "__main__":
    main()
