# src/dataset.py
# Purpose: PennyDataset supporting type filtering and ROI cropping
# Author:  <Your Name>
# Date:    2025-05-16
# Dependencies: pillow, torch

import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from .config_utils import TrainingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


class PennyDataset(Dataset):
    SPLIT_RATIOS: Tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __init__(
        self,
        cfg: TrainingConfig,
        image_transform: Optional[Callable] = None,
        subset: str = "train",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.subset = subset.lower()
        self.transform = image_transform or (lambda x: x)

        if self.subset not in {"train", "val", "test"}:
            raise ValueError("subset must be one of {'train','val','test'}")

        # Load full metadata
        logger.info("Loading metadata from %s", cfg.metadata_json_path)
        with open(cfg.metadata_json_path, "r", encoding="utf-8") as fh:
            all_meta: List[Dict] = json.load(fh)

        # Filter by `type` field depending on model_type
        if cfg.model_type == "side":
            meta = [m for m in all_meta if m.get("type") == "penny"]
        elif cfg.model_type in {"date", "mint"}:
            meta = [m for m in all_meta if m.get("type") == "roi"]
        else:
            meta = all_meta

        # Shuffle and split
        random.seed(cfg.seed)
        random.shuffle(meta)
        n = len(meta)
        n_train = int(self.SPLIT_RATIOS[0] * n)
        n_val = int(self.SPLIT_RATIOS[1] * n)
        splits = {
            "train": meta[:n_train],
            "val": meta[n_train : n_train + n_val],
            "test": meta[n_train + n_val :],
        }
        self.items = splits[self.subset]

        # Load ROI definitions for ROI‐based models
        if cfg.model_type in {"date", "mint"}:
            logger.info("Loading ROI definitions from %s", cfg.roi_json_path)
            try:
                with open(cfg.roi_json_path, "r", encoding="utf-8") as fh:
                    self.roi_defs: Dict[str, List[int]] = json.load(fh)
            except Exception as e:
                logger.warning("Could not load ROI JSON: %s", e)
                self.roi_defs = {}
        else:
            self.roi_defs = {}

        # Build label ↔ index map
        self._build_label_map()

    def _build_label_map(self) -> None:
        """Populate cfg.label_map based on cfg.model_type."""
        if self.cfg.model_type == "side":
            labels = sorted({item.get("side", "unknown") for item in self.items})
        elif self.cfg.model_type == "date":
            labels = sorted({item.get("year", "unknown") for item in self.items})
        elif self.cfg.model_type == "mint":
            labels = sorted({item.get("mint", "unknown") for item in self.items})
        else:
            labels = sorted({item.get(self.cfg.model_type, "unknown") for item in self.items})

        idx2lbl = {i: lbl for i, lbl in enumerate(labels)}
        self.label_to_idx = {v: k for k, v in idx2lbl.items()}

        if not self.cfg.label_map:
            self.cfg.label_map.update(idx2lbl)
            logger.info("Label map built: %s", idx2lbl)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample = self.items[idx]
        img_path = Path(self.cfg.data_root) / sample["filename"]
        image = Image.open(img_path).convert("RGB")

        # For ROI‐based models, crop to the region
        if self.cfg.model_type in {"date", "mint"}:
            coords = self.roi_defs.get(sample["filename"])
            if coords:
                x, y, w, h = coords
                image = image.crop((x, y, x + w, y + h))

        image = self.transform(image)

        # Pick label string
        if self.cfg.model_type == "side":
            label_str = sample.get("side", "unknown")
        elif self.cfg.model_type == "date":
            label_str = sample.get("year", "unknown")
        elif self.cfg.model_type == "mint":
            label_str = sample.get("mint", "unknown")
        else:
            label_str = sample.get(self.cfg.model_type, "unknown")

        label_id = self.label_to_idx.get(label_str, 0)
        return image, label_id
