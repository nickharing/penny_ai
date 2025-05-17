# dataset.py
# Purpose: Define PennyDataset and related data loading utilities
# Author: Nick Haring
# Date: 2025-05-17
# Dependencies: PIL, torch, torchvision, logging

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

#---***---*** Paths ***---***---
# (Paths configured via config_utils.py)
#---***---*** End Paths ***---***---

logger = logging.getLogger(__name__)

class PennyDataset(Dataset):
    """
    Dataset for Penny classification tasks.

    Args:
        data_root: Path to root directory containing images.
        metadata_json_path: Path to JSON with metadata about each image.
        roi_json_path: Path to JSON with ROI coordinates (optional).
        transform: torchvision transforms to apply.
    """
    def __init__(
        self,
        data_root: Path,
        metadata_json_path: Path,
        roi_json_path: Optional[Path] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_root = Path(data_root)
        self.metadata_json_path = Path(metadata_json_path)
        self.roi_json_path = Path(roi_json_path) if roi_json_path else None
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Load metadata
        with open(self.metadata_json_path, 'r') as f:
            self.metadata = json.load(f)

        # Optionally load ROI coords
        self.rois = {}
        if self.roi_json_path and self.roi_json_path.exists():
            with open(self.roi_json_path, 'r') as f:
                self.rois = json.load(f)

        # Build list of samples (image path, label)
        samples: List[Tuple[Path, int]] = []
        missing_count = 0
        for entry in self.metadata:
            fname = entry.get('filename')
            label = entry.get('label')
            img_path = self.data_root / fname
            if not img_path.exists():
                logger.warning(f"Missing image file, skipping: {img_path}")
                missing_count += 1
                continue
            samples.append((img_path, label))

        if missing_count > 0:
            logger.info(f"Skipped {missing_count} missing files; {len(samples)} samples remain.")
        if len(samples) == 0:
            raise FileNotFoundError(f"No valid image files found in {self.data_root}")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
