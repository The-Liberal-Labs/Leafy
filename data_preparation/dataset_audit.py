#!/usr/bin/env python3
"""Audit Leafy dataset class counts and likely duplicate labels."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Audit raw or split Leafy datasets")
    parser.add_argument(
        "--data-dir", default="./data", help="Directory with one subdirectory per class"
    )
    parser.add_argument(
        "--top-k", type=int, default=15, help="How many high/low classes to print"
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to save the audit report as JSON",
    )
    return parser.parse_args()


def count_images(class_dir: Path):
    return sum(1 for file_path in class_dir.iterdir() if file_path.is_file())


def normalize_class_name(name: str):
    normalized = name.lower().strip()
    normalized = normalized.replace("gauva", "guava")
    normalized = normalized.replace("sugercane", "sugarcane")
    normalized = normalized.replace("bell_pepper", "pepper_bell")
    normalized = normalized.replace("yellow_leaf_curl_virus", "leaf_curl")
    normalized = normalized.replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized


def collect_class_counts(data_dir: Path):
    return {
        class_dir.name: count_images(class_dir)
        for class_dir in sorted(data_dir.iterdir())
        if class_dir.is_dir() and not class_dir.name.startswith(".")
    }


def build_duplicate_groups(class_counts):
    grouped = defaultdict(list)
    for class_name, count in class_counts.items():
        grouped[normalize_class_name(class_name)].append(
            {"name": class_name, "count": count}
        )
    return {
        normalized: sorted(entries, key=lambda item: item["name"])
        for normalized, entries in grouped.items()
        if len(entries) > 1
    }


def build_report(class_counts, top_k):
    counts = list(class_counts.values())
    sorted_counts = sorted(class_counts.items(), key=lambda item: item[1])
    return {
        "num_classes": len(class_counts),
        "total_images": sum(counts),
        "min_count": min(counts) if counts else 0,
        "max_count": max(counts) if counts else 0,
        "mean_count": (sum(counts) / len(counts)) if counts else 0,
        "imbalance_ratio": (max(counts) / max(min(counts), 1)) if counts else 0,
        "bottom_classes": [
            {"name": name, "count": count} for name, count in sorted_counts[:top_k]
        ],
        "top_classes": [
            {"name": name, "count": count} for name, count in sorted_counts[-top_k:]
        ],
        "duplicate_groups": build_duplicate_groups(class_counts),
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_counts = collect_class_counts(data_dir)
    if not class_counts:
        raise RuntimeError(f"No class folders found under: {data_dir}")

    report = build_report(class_counts, args.top_k)

    print(f"\nDataset: {data_dir.resolve()}")
    print(f"Classes: {report['num_classes']}")
    print(f"Images:  {report['total_images']:,}")
    print(f"Min/Max: {report['min_count']} / {report['max_count']}")
    print(f"Imbalance ratio: {report['imbalance_ratio']:.1f}x")

    print("\nLowest-count classes:")
    for item in report["bottom_classes"]:
        print(f"  {item['count']:>5}  {item['name']}")

    print("\nHighest-count classes:")
    for item in report["top_classes"]:
        print(f"  {item['count']:>5}  {item['name']}")

    if report["duplicate_groups"]:
        print("\nLikely duplicate label groups:")
        for normalized_name, entries in sorted(report["duplicate_groups"].items()):
            total = sum(entry["count"] for entry in entries)
            print(f"  {normalized_name} (total={total})")
            for entry in entries:
                print(f"    - {entry['name']}: {entry['count']}")
    else:
        print("\nNo normalized duplicate label groups detected.")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nAudit report saved to: {report_path}")


if __name__ == "__main__":
    main()
