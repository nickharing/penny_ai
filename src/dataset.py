# src/dataset.py
# Purpose: Dataset for penny image classification tasks using ModelConfig
# Author: <Your Name>
# Date: YYYY-MM-DD
# Dependencies: torch, torchvision, PIL, OpenCV, numpy, config_utils

import json
import logging
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.config_utils import load_config, ModelConfig

logger = logging.getLogger(__name__)

class PennyDataset(Dataset):
    """
    Dataset for different penny classification tasks: side, date, mint, orientation, reverse_type.
    Applies preprocessing (CLAHE, median filter, threshold), ROI cropping, augmentations, and returns (image, label).
    """
    def __init__(self, cfg: ModelConfig, model: str, subset: str = 'train'):
        self.cfg = cfg
        self.model = model
        self.subset = subset
        self.image_mode = 'L'  # grayscale

        # Build metadata filter
        with open(cfg.paths.metadata, 'r') as f:
            all_meta = json.load(f)
        if model == 'side':
            self.items = [m for m in all_meta if m.get('type') == 'penny']
        elif model == 'date':
            self.items = [m for m in all_meta if m.get('type') == 'roi' and m.get('roi_type') == 'date']
        elif model == 'mint':
            self.items = [m for m in all_meta if m.get('type') == 'roi' and m.get('roi_type') == 'mint']
        elif model == 'orientation':
            self.items = [m for m in all_meta if m.get('type') == 'penny']
        elif model == 'reverse_type':
            self.items = [m for m in all_meta if m.get('type') == 'penny' and m.get('side') == 'reverse']
        else:
            logger.warning(f"Unknown model type '{model}' for filtering. Using all items.")
            self.items = all_meta

        # Build file map
        self.file_map = {}
        mapping = {
            'side': ['obverse', 'reverse'],
            'date': ['date'],
            'mint': ['mint_mark'],
            'orientation': ['obverse', 'reverse'],
            'reverse_type': ['reverse']
        }
        subfolders = mapping.get(model, [])
        for sf in subfolders:
            base = Path(cfg.paths.data_root) / sf
            if base.is_dir():
                for ext in ['*.jpg','*.jpeg','*.png','*.bmp','*.gif']:
                    for p in base.rglob(ext):
                        self.file_map[p.name] = p
            else:
                logger.warning(f"Missing data subfolder for model '{model}': {base}")

        # Build label map
        label_field = {
            'side': 'side',
            'date': 'year',
            'mint': 'mint',
            'orientation': 'angle',
            'reverse_type': 'reverse_type'
        }.get(model)
        if label_field is None:
            raise ValueError(f"No label field for model '{model}'")
        # extract label strings
        label_vals = set()
        for m in self.items:
            if model == 'orientation':
                try:
                    angle = float(m.get('angle',0))
                    bin_idx = int(angle//3)%120
                    label_vals.add(str(bin_idx))
                except:
                    continue
            else:
                val = m.get(label_field)
                if val is not None:
                    label_vals.add(str(val))
        self.labels = sorted(label_vals)
        self.label_to_idx = {v:i for i,v in enumerate(self.labels)}

        # ROI definitions for date/mint
        roi_path = Path(cfg.paths.roi)
        self.roi_defs = json.loads(roi_path.read_text()) if roi_path.exists() else {}

        # Compose transforms: applied after preprocessing & ROI crop
        tfms = []
        if cfg.apply_augmentations and subset=='train':
            if cfg.random_horizontal_flip:
                tfms.append(transforms.RandomHorizontalFlip())
            if cfg.rotation_degrees:
                tfms.append(transforms.RandomRotation(cfg.rotation_degrees))
            # other augmentations to be added here
        tfms.extend([
            transforms.Resize(tuple(cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalization, cfg.std)
        ])
        self.transform = transforms.Compose(tfms)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]
        fname = sample.get('filename')
        img_path = self.file_map.get(Path(fname).name)
        if img_path is None:
            raise FileNotFoundError(f"Image '{fname}' not found in file_map")

        # Load grayscale PIL image
        pil = Image.open(img_path).convert(self.image_mode)

        # Preprocessing (CLAHE, median, threshold)
        img_np = np.array(pil)
        if self.cfg.preprocessing.get('apply_clahe', False):
            clahe = cv2.createCLAHE(
                clipLimit=self.cfg.preprocessing.get('clahe_clip_limit',1.5),
                tileGridSize=tuple(self.cfg.preprocessing.get('clahe_tile_grid_size',[8,8]))
            )
            img_np = clahe.apply(img_np)
        if self.cfg.preprocessing.get('apply_median_filter', False):
            k = self.cfg.preprocessing.get('median_filter_kernel_size',3)
            k += (1 - k%2)
            img_np = cv2.medianBlur(img_np, k)
        if self.cfg.preprocessing.get('apply_adaptive_threshold', False):
            b = self.cfg.preprocessing.get('adaptive_threshold_block_size',13)
            b += (1 - b%2)
            C = self.cfg.preprocessing.get('adaptive_threshold_C',3)
            img_np = cv2.adaptiveThreshold(
                img_np,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,b,C
            )
        pil = Image.fromarray(img_np)

        # ROI crop for date/mint
        if self.model in ['date','mint']:
            key = Path(fname).name
            coords = self.roi_defs.get(key)
            if coords:
                x,y,w,h = coords
                pad = self.cfg.roi_padding.get(self.model, {})
                top = int(h*pad.get('top',0))
                bot = int(h*pad.get('bottom',0))
                left = int(w*pad.get('left',0))
                right = int(w*pad.get('right',0))
                x0,x1 = max(0,x-left), min(pil.width, x+w+right)
                y0,y1 = max(0,y-top),  min(pil.height, y+h+bot)
                pil = pil.crop((x0,y0,x1,y1))

        # Transform
        img = self.transform(pil)

        # Label
        if self.model == 'orientation':
            angle = float(sample.get('angle',0))
            label = str(int(angle//3)%120)
        else:
            label = str(sample.get({
                'side':'side','date':'year','mint':'mint',
                'reverse_type':'reverse_type'
            }[self.model]))
        label_idx = self.label_to_idx.get(label)
        if label_idx is None:
            raise KeyError(f"Label '{label}' not in map for model '{self.model}'")

        return img, label_idx
