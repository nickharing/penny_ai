from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

from PIL import Image
from torch.utils.data import Dataset

from .config_utils import TrainingConfig


class PennyDataset(Dataset):
    """
    Simple dataset wrapper that consumes the project TrainingConfig so the
    constructor never has to change when we add new fields.
    """

    SPLIT_RATIOS: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train / val / test

    def __init__(
        self,
        cfg: TrainingConfig,
        image_transform: Optional[Callable] = None,
        subset: str = "train",
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.subset = subset.lower()
        self.transform = image_transform or (lambda x: x)

        if self.subset not in {"train", "val", "test"}:
            raise ValueError("subset must be one of {'train','val','test'}")

        # ----------------------------------------------------------- load meta
        with open(cfg.metadata_json_path, "r", encoding="utf-8") as fh:
            self.metadata: List[Dict] = json.load(fh)

        # --------------------------------------------------------- train/val/test split
        random.seed(seed)
        random.shuffle(self.metadata)

        n_total = len(self.metadata)
        n_train = int(self.SPLIT_RATIOS[0] * n_total)
        n_val = int(self.SPLIT_RATIOS[1] * n_total)

        splits = {
            "train": self.metadata[:n_train],
            "val": self.metadata[n_train : n_train + n_val],
            "test": self.metadata[n_train + n_val :],
        }
        self.items = splits[self.subset]

        # --------------------------------------------------------- label map
        self._build_label_map()

    # --------------------------------------------------------------------- #

    def _build_label_map(self) -> None:
        """Create integer ↔ string mapping once, store on cfg."""
        labels = sorted({item[self.cfg.model_type] for item in self.metadata})
        idx_to_label = {idx: lbl for idx, lbl in enumerate(labels)}
        label_to_idx = {lbl: idx for idx, lbl in idx_to_label.items()}
        self.label_to_idx = label_to_idx

        # fill cfg.label_map only on first dataset instantiation
        if not self.cfg.label_map:
            self.cfg.label_map.update(idx_to_label)

    # --------------------------------------------------------------------- #
    #  Dataset protocol
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample = self.items[idx]
        img_path = Path(self.cfg.data_root) / sample["filename"]
        image = Image.open(img_path).convert("RGB")

        label_str = sample[self.cfg.model_type]
        label_id = self.label_to_idx[label_str]

        return self.transform(image), label_id
