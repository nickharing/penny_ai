# dataset.py
# Purpose: Dataset for penny image classification tasks using ModelConfig
import json, logging, random
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .config_utils import load_config, ModelConfig

logger = logging.getLogger(__name__)

class PennyDataset(Dataset):
    def __init__(self, cfg: ModelConfig, model: str, subset: str = 'train'):
        self.cfg = cfg
        self.model = model
        self.subset = subset
        # build transform
        tfms = [transforms.Resize(tuple(cfg.image_size)), transforms.ToTensor()]
        tfms.append(transforms.Normalize(cfg.normalization, cfg.std))
        if cfg.apply_augmentations and subset=='train':
            aug = []
            if cfg.random_horizontal_flip: aug.append(transforms.RandomHorizontalFlip())
            if cfg.rotation_degrees: aug.append(transforms.RandomRotation(cfg.rotation_degrees))
            tfms = aug + tfms
        self.transform = transforms.Compose(tfms)
        # load metadata
        with open(cfg.paths.metadata, 'r') as f: all_meta = json.load(f)
        # filter by model type
        if model=='side': items=[m for m in all_meta if m['type']=='penny']
        elif model in ['date','mint']: items=[m for m in all_meta if m['type']=='roi']
        else: # Potentially handle unknown model type or use all_meta as a fallback
            logger.warning(f"Unknown model type '{model}' for filtering. Using all metadata items.")
            items=all_meta

        if not items:
            logger.warning(f"No items found after filtering for model '{model}' and type. Dataset will be empty.")
            self.items = []
            self.label_to_idx = {}
            self.roi_definitions = {} # Ensure attribute exists even if empty
            return

        random.seed(cfg.seed); random.shuffle(items)
        # split
        n=len(items); t=int(0.8*n); v=int(0.1*n)
        splits={'train':items[:t],'val':items[t:t+v],'test':items[t+v:]}
        self.items=splits[subset]
        # build file map
        self.file_map = {}
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']: # Add relevant image extensions
            for p in Path(cfg.paths.data_root).rglob(ext):
                self.file_map[p.name] = p
        # label map
        labels=sorted({m.get(model,'unknown') for m in self.items})
        self.label_to_idx={lbl:i for i,lbl in enumerate(labels)}
        # Load ROI definitions once if needed
        self.roi_definitions = {}
        if self.model in ['date', 'mint'] and Path(self.cfg.paths.roi).exists():
            try:
                with open(self.cfg.paths.roi, 'r') as f:
                    self.roi_definitions = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode ROI JSON from {self.cfg.paths.roi}")
            except FileNotFoundError:
                 logger.error(f"ROI file not found at {self.cfg.paths.roi} though model type suggests it's needed.")
        elif self.model in ['date', 'mint'] and not Path(self.cfg.paths.roi).exists():
            logger.warning(f"ROI file {self.cfg.paths.roi} not found, but model type is {self.model}. ROI cropping will be skipped.")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]
        fname_meta = sample['filename'] # Filename from metadata, e.g., "penny_reverse_w_0098 - Copy.jpg"

        # Attempt 1: Direct lookup in file_map using the exact basename from metadata
        # This assumes fname_meta is a basename, which aligns with "metadata formatting will not change"
        # if that implies filenames in metadata don't have subpaths.
        img_path = self.file_map.get(Path(fname_meta).name)

        # Attempt 2: If not found and metadata filename contains " - Copy.jpg",
        # try looking up the version without " - Copy.jpg"
        if not img_path and " - Copy.jpg" in fname_meta:
            fname_stripped = Path(fname_meta).name.replace(" - Copy.jpg", ".jpg")
            img_path = self.file_map.get(fname_stripped)
            if img_path:
                logger.debug(f"Found image using stripped name '{fname_stripped}' for metadata entry '{fname_meta}'")
        # Attempt 2b: Handle ".png" as well for " - Copy.png"
        elif not img_path and " - Copy.png" in fname_meta: # Use elif to avoid re-stripping if .jpg version was already tried
            fname_stripped = Path(fname_meta).name.replace(" - Copy.png", ".png")
            img_path = self.file_map.get(fname_stripped)
            if img_path:
                logger.debug(f"Found image using stripped name '{fname_stripped}' for metadata entry '{fname_meta}'")

        if not img_path or not img_path.exists():
            error_msg_detail = f"metadata filename: '{fname_meta}'. Tried direct map lookup for '{Path(fname_meta).name}'"
            if " - Copy" in fname_meta: error_msg_detail += f" and stripped version '{fname_meta.replace(' - Copy.jpg', '.jpg').replace(' - Copy.png', '.png')}'."
            logger.error(f"Image file not found for metadata entry: {sample}. {error_msg_detail}")
            raise FileNotFoundError(f"Image not found. {error_msg_detail}")

        image=Image.open(img_path).convert('RGB')
        # ROI crop
        if self.model in ['date','mint']:
            # Use the same logic for the ROI key as for finding the image in file_map
            # if metadata filenames might have " - Copy" but ROI keys don't.
            roi_key = Path(fname_meta).name 
            if " - Copy.jpg" in roi_key:
                roi_key = roi_key.replace(" - Copy.jpg", ".jpg")
            elif " - Copy.png" in roi_key:
                roi_key = roi_key.replace(" - Copy.png", ".png")

            coords=self.roi_definitions.get(roi_key)
            if coords: x,y,w,h=coords; image=image.crop((x,y,x+w,y+h))
            elif self.roi_definitions: # Only warn if ROI defs were loaded but this specific image's ROI is missing
                logger.warning(f"ROI coordinates not found for '{roi_key}' in loaded ROI definitions. Full image used.")

        img=self.transform(image)
        label_str = sample.get(self.model) # Simplified label retrieval
        label_idx = self.label_to_idx.get(label_str)
        if label_idx is None:
            logger.error(f"Label '{label_str}' for model '{self.model}' in sample '{fname_meta}' not found in label_to_idx. Mapped labels: {self.label_to_idx}")
            raise KeyError(f"Label '{label_str}' not found in label map. Check metadata and dataset initialization.")
        return img, label_idx
