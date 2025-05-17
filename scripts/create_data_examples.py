#!/usr/bin/env python3
"""
create_data_examples.py

Build a lightweight `data_examples/` tree that holds a sample of images
(20 per sub-folder) from the full `data/` directory, leaving the originals
untouched.

Run from the repo root:
    python create_data_examples.py
"""

import itertools
import shutil
from pathlib import Path
from typing import Iterable

KEEP_PER_FOLDER = 20                       # adjust if you want a different sample size
SRC_ROOT = Path("data")
DEST_ROOT = Path("data_examples")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


def iter_images(folder: Path) -> Iterable[Path]:
    """Yield image files in `folder`, sorted by name."""
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS),
        key=lambda p: p.name.lower(),
    )


def main() -> None:
    if not SRC_ROOT.exists():
        raise SystemExit(f"Source folder {SRC_ROOT} not found. Run from repo root.")

    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for subdir in sorted([p for p in SRC_ROOT.iterdir() if p.is_dir()]):
        dest_sub = DEST_ROOT / subdir.name
        dest_sub.mkdir(parents=True, exist_ok=True)

        keepers = itertools.islice(iter_images(subdir), KEEP_PER_FOLDER)
        count = 0
        for img_path in keepers:
            shutil.copy2(img_path, dest_sub / img_path.name)
            count += 1
            total_copied += 1

        print(f"✓  {subdir.name}: copied {count} image(s)")

    print(f"\nDone. Total images copied: {total_copied}")
    print(f"Sample set available at: {DEST_ROOT.resolve()}")


if __name__ == "__main__":
    main()
