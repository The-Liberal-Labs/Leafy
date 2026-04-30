import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def write_images(class_dir: Path, count: int):
    class_dir.mkdir(parents=True, exist_ok=True)
    offset = sum(ord(char) for char in class_dir.name) % 255
    for index in range(count):
        image = Image.new(
            "RGB",
            (12, 12),
            color=((index * 3) % 255, offset, (offset * 2) % 255),
        )
        image.save(class_dir / f"{index:03d}.jpg")


def run_command(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def split_counts(split_dir: Path, class_name: str):
    return {
        split: len(list((split_dir / split / class_name).iterdir()))
        for split in ("train", "val", "test")
    }


def test_prepare_dataset_ratios_and_validation(tmp_path):
    source = tmp_path / "source"
    for class_name in ("Apple___healthy", "Corn___rust", "Rice___blast"):
        write_images(source / class_name, 20)

    ratios = [
        ("0.80", "0.10", {"train": 16, "val": 2, "test": 2}),
        ("0.70", "0.15", {"train": 14, "val": 3, "test": 3}),
        ("0.70", "0.20", {"train": 14, "val": 4, "test": 2}),
        ("0.60", "0.20", {"train": 12, "val": 4, "test": 4}),
    ]

    for train_ratio, val_ratio, expected in ratios:
        output = tmp_path / f"split_{train_ratio}_{val_ratio}"
        run_command(
            "data_preparation/prepare_dataset.py",
            "--source-data-dir",
            str(source),
            "--output-dir",
            str(output),
            "--train-ratio",
            train_ratio,
            "--val-ratio",
            val_ratio,
            "--min-images-per-class",
            "1",
            "--no-hash-groups",
        )
        summary = json.loads((output / "split_summary.json").read_text())
        assert summary["validation"]["ok"] is True
        assert split_counts(output, "Apple___healthy") == expected


def test_validate_split_dataset_writes_primary_summary(tmp_path):
    source = tmp_path / "source"
    for class_name in ("Apple___healthy", "Corn___rust", "Rice___blast"):
        write_images(source / class_name, 20)
    output = tmp_path / "split"
    run_command(
        "data_preparation/prepare_dataset.py",
        "--source-data-dir",
        str(source),
        "--output-dir",
        str(output),
        "--min-images-per-class",
        "1",
        "--no-hash-groups",
    )

    run_command(
        "data_preparation/validate_split_dataset.py",
        "--data-dir",
        str(output),
        "--write-summary",
    )

    summary = json.loads((output / "split_summary.json").read_text())
    fingerprint = json.loads((output / "dataset_fingerprint.json").read_text())
    assert summary["validation"]["ok"] is True
    assert summary["primary_data_dir"] == str(output)
    assert fingerprint["primary_data_dir"] == str(output)
