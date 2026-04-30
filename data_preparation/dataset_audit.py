#!/usr/bin/env python3
"""Audit Leafy dataset class counts, support buckets, and duplicate labels."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Leafy class-folder datasets")
    parser.add_argument(
        "--data-dir",
        default="./data_split",
        help="Class-folder dataset or split dataset root",
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


def canonical_class_name(name: str):
    cleaned = name.strip().replace(" ", "_")
    cleaned = cleaned.replace("Gauva", "Guava").replace("Sugercane", "Sugarcane")
    cleaned = cleaned.replace("Bell_pepper", "Pepper_bell")

    if "___" in cleaned:
        species, condition = cleaned.split("___", 1)
    elif "__" in cleaned:
        species, condition = cleaned.split("__", 1)
    elif "_" in cleaned:
        species, condition = cleaned.split("_", 1)
    else:
        species, condition = cleaned, "unknown"

    species = "_".join(part[:1].upper() + part[1:] for part in species.split("_") if part)
    condition = re.sub(r"_+", "_", condition.lower()).strip("_")
    return f"{species}___{condition}"


def collect_class_counts(data_dir: Path):
    split_dirs = [data_dir / split for split in ("train", "val", "test")]
    if all(path.exists() and path.is_dir() for path in split_dirs):
        counts = defaultdict(int)
        for split_dir in split_dirs:
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir() and not class_dir.name.startswith("."):
                    counts[class_dir.name] += count_images(class_dir)
        return dict(sorted(counts.items()))

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


def build_canonical_duplicate_groups(class_counts):
    grouped = defaultdict(list)
    for class_name, count in class_counts.items():
        grouped[canonical_class_name(class_name)].append(
            {"name": class_name, "count": count}
        )
    return {
        canonical: sorted(entries, key=lambda item: item["name"])
        for canonical, entries in grouped.items()
        if len(entries) > 1
    }


def build_support_buckets(class_counts):
    return {
        "under_200": [
            {"name": name, "count": count}
            for name, count in sorted(class_counts.items(), key=lambda item: item[1])
            if count < 200
        ],
        "low_200_499": [
            {"name": name, "count": count}
            for name, count in sorted(class_counts.items(), key=lambda item: item[1])
            if 200 <= count < 500
        ],
        "normal_500_plus": [
            {"name": name, "count": count}
            for name, count in sorted(class_counts.items(), key=lambda item: item[1])
            if count >= 500
        ],
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
        "canonical_duplicate_groups": build_canonical_duplicate_groups(class_counts),
        "support_buckets": build_support_buckets(class_counts),
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
    print(
        "Support buckets: "
        f"<200={len(report['support_buckets']['under_200'])}, "
        f"200-499={len(report['support_buckets']['low_200_499'])}, "
        f"500+={len(report['support_buckets']['normal_500_plus'])}"
    )

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

    if report["canonical_duplicate_groups"]:
        print("\nCanonical duplicate label groups:")
        for canonical_name, entries in sorted(report["canonical_duplicate_groups"].items()):
            total = sum(entry["count"] for entry in entries)
            print(f"  {canonical_name} (total={total})")
            for entry in entries:
                print(f"    - {entry['name']}: {entry['count']}")
    else:
        print("No canonical duplicate label groups detected.")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nAudit report saved to: {report_path}")


if __name__ == "__main__":
    main()
