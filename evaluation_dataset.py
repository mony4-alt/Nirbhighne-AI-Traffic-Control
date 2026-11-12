"""Create a portable, validation-only YOLO dataset for reliable evaluation."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}


def _resolve_path(base_dir: Path, root: str, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(root) if root else base_dir) / path


def _labels_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


def prepare_validation_dataset(source_yaml: str, destination_dir: str) -> str:
    """Copy validation data and convert HEIC/HEIF images to JPEG.

    Ultralytics can otherwise silently mark HEIC files as corrupt.  The output
    contains only what validation needs: ``images/val``, ``labels/val``, and a
    portable relative ``data.yaml``.
    """
    source_path = Path(source_yaml).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    root_value = str(data.get("path", "") or "")
    # Resolve ``path`` relative to the YAML file.  The previous implementation
    # passed the already-resolved directory as the base for ``val`` even when
    # ``path`` was empty, which turned ``../valid/images`` into
    # ``<project>/valid/images`` instead of the dataset's sibling directory.
    root_dir = _resolve_path(source_path.parent, "", root_value).resolve() if root_value else source_path.parent.resolve()
    val_value = data.get("val")
    if not val_value:
        raise ValueError(f"No 'val:' entry found in {source_path}")
    images_dir = _resolve_path(source_path.parent, str(root_dir) if root_value else "", str(val_value)).resolve()
    labels_dir = _labels_dir(images_dir)
    if not images_dir.is_dir():
        raise ValueError(f"Validation images directory not found: {images_dir}")

    destination = Path(destination_dir).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    output_images = destination / "valid" / "images"
    output_labels = destination / "valid" / "labels"
    output_images.mkdir(parents=True)
    output_labels.mkdir(parents=True)

    converted = 0
    copied = 0
    for source_image in images_dir.iterdir():
        if not source_image.is_file() or source_image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        target_stem = source_image.stem
        if source_image.suffix.lower() in {".heic", ".heif"}:
            import pi_heif
            pi_heif.register_heif_opener()
            target_image = output_images / f"{target_stem}.jpg"
            with Image.open(source_image) as image:
                image.convert("RGB").save(target_image, "JPEG", quality=95)
            converted += 1
        else:
            target_image = output_images / source_image.name
            shutil.copy2(source_image, target_image)
            copied += 1

        source_label = labels_dir / f"{source_image.stem}.txt"
        if source_label.is_file():
            shutil.copy2(source_label, output_labels / f"{target_stem}.txt")

    names = data.get("names", [])
    nc = data.get("nc", len(names))
    manifest = destination / "data.yaml"
    manifest.write_text(
        "# Auto-generated portable validation dataset\n"
        "path: .\n"
        "train: valid/images\n"
        "val: valid/images\n"
        "test: valid/images\n"
        f"nc: {nc}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )
    if copied + converted == 0:
        raise ValueError(f"No supported validation images found in {images_dir}")
    return str(manifest)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python evaluation_dataset.py <source-data.yaml> <destination-folder>")
    result = prepare_validation_dataset(sys.argv[1], sys.argv[2])
    print(f"Portable evaluation data ready: {result}")
