# src/dataset.py
# Purpose: Define PennyDataset with optional ROI cropping
# Author:  <Your Name>
# Date:    2025-05-16
# Dependencies: pillow, torch

import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from .config_utils import TrainingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class PennyDataset(Dataset):
    SPLIT_RATIOS = (0.8, 0.1, 0.1)

    def __init__(
        self,
        cfg: TrainingConfig,
        image_transform: Optional[Callable] = None,
        subset: str = "train",
    ) -> None:
        super().__init__()
        self.cfg       = cfg
        self.subset    = subset.lower()
        self.transform = image_transform or (lambda x: x)

        if self.subset not in {"train", "val", "test"}:
            raise ValueError("subset must be one of {'train','val','test'}")

        logger.info("Loading metadata from %s", cfg.metadata_json_path)
        with open(cfg.metadata_json_path, "r", encoding="utf-8") as fh:
            self.metadata: List[Dict] = json.load(fh)

        random.seed(cfg.seed)
        random.shuffle(self.metadata)
        n = len(self.metadata)
        n_train = int(self.SPLIT_RATIOS[0] * n)
        n_val   = int(self.SPLIT_RATIOS[1] * n)
        splits = {
            "train": self.metadata[:n_train],
            "val":   self.metadata[n_train : n_train + n_val],
            "test":  self.metadata[n_train + n_val :],
        }
        self.items = splits[self.subset]

        # Load ROI definitions only for date & mint
        if cfg.model_type in {"date", "mint"}:
            logger.info("Loading ROI definitions from %s", cfg.roi_json_path)
            with open(cfg.roi_json_path, "r", encoding="utf-8") as fh:
                self.roi_defs: Dict[str, List[int]] = json.load(fh)
        else:
            self.roi_defs = {}

        self._build_label_map()

    def _build_label_map(self) -> None:
        labels = sorted({item[self.cfg.model_type] for item in self.metadata})
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

        # ROI cropping for date & mint
        if self.cfg.model_type in {"date", "mint"}:
            coords = self.roi_defs.get(sample["filename"])
            if coords:
                x, y, w, h = coords
                image = image.crop((x, y, x + w, y + h))

        image = self.transform(image)
        label_str = sample[self.cfg.model_type]
        label_id  = self.label_to_idx[label_str]
        return image, label_id
