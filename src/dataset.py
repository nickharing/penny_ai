# src/dataset.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, List, Dict, Optional

from PIL import Image
from torch.utils.data import Dataset

from .config_utils import TrainingConfig


class PennyDataset(Dataset):
    SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train / val / test

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

        # load metadata
        with open(self.cfg.metadata_json_path, "r", encoding="utf-8") as fh:
            self.metadata: List[Dict] = json.load(fh)

        # split
        random.seed(self.cfg.seed)
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

        # build + store label map
        self._build_label_map()

    def _build_label_map(self) -> None:
        labels = sorted({item[self.cfg.model_type] for item in self.metadata})
        idx2lbl = {i: lbl for i, lbl in enumerate(labels)}
        self.label_to_idx = {v: k for k, v in idx2lbl.items()}

        # only first call populates the config
        if not self.cfg.label_map:
            self.cfg.label_map.update(idx2lbl)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample   = self.items[idx]
        img_path = Path(self.cfg.data_root) / sample["filename"]
        image    = Image.open(img_path).convert("RGB")

        lbl_str  = sample[self.cfg.model_type]
        lbl_id   = self.label_to_idx[lbl_str]
        return self.transform(image), lbl_id
