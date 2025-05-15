# src/dataset.py
# Implements the PennyDataset class for loading and preprocessing penny images.

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image, ImageOps # ImageOps for padding if needed, though crop handles it
from torch.utils.data import Dataset
from torchvision import transforms as T
# from sklearn.model_selection import train_test_split # Not needed due to UID hashing

class PennyDataset(Dataset):
    """
    A dataset class for loading penny images and their corresponding labels
    for various classification tasks (side, orientation, date, mint).

    The dataset handles:
    - Loading images from file paths.
    - Parsing metadata from JSON.
    - Applying model-specific transformations and augmentations.
    - Extracting ROIs for date and mint tasks.
    - Generating labels based on the specified model_type.
    - Splitting data into train, validation, and test sets using UID hashing.
    """

    def __init__(self,
                 data_root: Union[str, Path],
                 metadata_json_path: Union[str, Path],
                 roi_json_path: Union[str, Path],
                 model_type: str,
                 image_transform: Optional[Callable] = None,
                 subset: str = "train", # "train", "val", or "test"
                 # seed is not directly used by UID hashing but good for other randomness if any
                 seed: int = 42): 
        """
        Initializes the PennyDataset.

        Args:
            data_root (Union[str, Path]): Path to the root directory of the image data
                                          (e.g., "data/" or "data_examples/").
            metadata_json_path (Union[str, Path]): Path to the metadata JSON file.
            roi_json_path (Union[str, Path]): Path to the ROI coordinates JSON file.
            model_type (str): The type of model this dataset is for.
                              Expected values: "side", "orientation", "date", "mint".
            image_transform (Optional[Callable]): Torchvision transforms to be applied to images.
            subset (str): Specifies the dataset subset to load: "train", "val", or "test".
            seed (int): Random seed, primarily for reproducibility if any operations
                        not covered by UID hashing require it.
        """
        self.data_root = Path(data_root)
        self.metadata_json_path = Path(metadata_json_path)
        self.roi_json_path = Path(roi_json_path)
        self.model_type = model_type.lower()
        self.image_transform = image_transform
        self.subset = subset.lower()
        self.seed = seed # Kept for consistency, though UID hashing is deterministic

        if self.subset not in ["train", "val", "test"]:
            raise ValueError(f"subset must be one of 'train', 'val', or 'test', got {subset}")

        self._load_metadata()
        self._load_roi_coordinates() # Load only if needed
        self._prepare_data_splits()

        # Define base label maps (integer to string)
        # These are primarily for creating the label_map.json output
        self._base_label_maps: Dict[str, Dict[int, str]] = {
            "side": {0: "obverse", 1: "memorial", 2: "shield", 3: "wheat", 4: "bicentennial"},
            "mint": {0: "D", 1: "S", 2: "P", 3: "nomint"},
            "date": {i: str(1909 + i) for i in range(2024 - 1909 + 1)},
            "orientation": {i: f"{i*3}-{(i*3)+2}" for i in range(120)},
        }
        self.current_label_map = self._get_label_map()

    def _load_metadata(self):
        """Loads image metadata from the JSON file."""
        try:
            with open(self.metadata_json_path, 'r') as f:
                metadata_content = json.load(f)
            # Ensure metadata is a list of dicts, where each dict has a 'uid'
            if isinstance(metadata_content, dict): # If it's a dict of UID -> details
                 self.metadata = [{"uid": k, **v} for k, v in metadata_content.items()]
            elif isinstance(metadata_content, list): # If it's already a list of items
                self.metadata = metadata_content
            else:
                raise ValueError("Metadata JSON format not recognized. Expected list of items or dict of UID to items.")

        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata JSON file not found: {self.metadata_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from: {self.metadata_json_path}")

    def _load_roi_coordinates(self):
        """Loads ROI coordinates from the JSON file if model_type requires it."""
        if self.model_type in ["date", "mint"]:
            try:
                with open(self.roi_json_path, 'r') as f:
                    self.roi_coordinates = json.load(f) # Expected dict: {UID_stem: {roi_key: coords}}
            except FileNotFoundError:
                raise FileNotFoundError(f"ROI JSON file not found: {self.roi_json_path}")
            except json.JSONDecodeError:
                raise ValueError(f"Error decoding JSON from: {self.roi_json_path}")
        else:
            self.roi_coordinates = {} # Not needed for side/orientation

    def _prepare_data_splits(self):
        """
        Filters metadata for the current model_type and splits into train/val/test
        based on UID hashing.
        """
        # Filter data relevant to the model type
        relevant_samples = []
        for item in self.metadata:
            uid = item.get("uid")
            if not uid:
                # print(f"Warning: Skipping item without UID: {item}")
                continue # Skip items without a UID

            if self.model_type == "orientation":
                # For 'orientation', only use 'obverse' images that have an 'angle'
                if item.get("side") == "obverse" and "angle" in item:
                    relevant_samples.append(item)
            elif self._is_valid_sample(item): # General validity check for other types
                relevant_samples.append(item)
        
        self.all_samples = relevant_samples # Store all relevant samples before splitting

        train_items, val_test_items = [], []
        for item in self.all_samples:
            uid = item["uid"] # Assumed to exist due to previous check
            hash_object = hashlib.sha256(uid.encode('utf-8'))
            hex_dig = hash_object.hexdigest()
            hash_val = int(hex_dig, 16)
            bucket = hash_val % 100 # Assign to one of 100 buckets (0-99)

            # 80% for training (buckets 0-79)
            if bucket < 80:
                train_items.append(item)
            # Remaining 20% for validation and test (buckets 80-99)
            else:
                val_test_items.append(item)
        
        val_items, test_items = [], []
        for item in val_test_items:
            uid = item["uid"]
            # Use a different salt for the second split to ensure different distribution
            hash_object = hashlib.sha256((uid + "_val_test_split_salt").encode('utf-8'))
            hex_dig = hash_object.hexdigest()
            hash_val = int(hex_dig, 16)
            bucket = hash_val % 2 # Assign to one of 2 buckets (0 or 1) for 50/50 split

            # 10% for validation (50% of the 20%)
            if bucket == 0:
                val_items.append(item)
            # 10% for test (remaining 50% of the 20%)
            else:
                test_items.append(item)

        if self.subset == "train":
            self.samples = train_items
        elif self.subset == "val":
            self.samples = val_items
        elif self.subset == "test":
            self.samples = test_items
        else:
            # This case should be caught by the __init__ check
            self.samples = []

        if not self.samples and len(self.all_samples) > 0 : # Only warn if there were relevant samples
             print(f"Warning: No samples assigned to subset '{self.subset}' for model_type '{self.model_type}'. "
                   f"Total relevant samples: {len(self.all_samples)}. "
                   f"Train attempt: {len(train_items)}, Val attempt: {len(val_items)}, Test attempt: {len(test_items)}")

    def _is_valid_sample(self, item: Dict[str, Any]) -> bool:
        """Checks if a metadata item is valid for the current model_type, excluding orientation."""
        if self.model_type == "side":
            # Must have 'side'. If 'side' is 'reverse', must also have 'type'.
            has_side = "side" in item
            is_obverse = item.get("side") == "obverse"
            is_reverse_with_type = item.get("side") == "reverse" and "type" in item
            return has_side and (is_obverse or is_reverse_with_type)
        # 'orientation' is handled separately in _prepare_data_splits for its specific 'obverse' & 'angle' criteria
        elif self.model_type == "date":
            # Must be 'obverse' and have 'year'
            return item.get("side") == "obverse" and "year" in item
        elif self.model_type == "mint":
            # Must be 'obverse' and have 'mint_mark'
            return item.get("side") == "obverse" and "mint_mark" in item
        return False # Should not be reached if model_type is valid

    def _get_label_map(self) -> Dict[int, str]:
        """Retrieves the predefined label map for the current model type."""
        return self._base_label_maps.get(self.model_type, {})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves an image and its label for the given index.
        Handles ROI cropping and label generation based on model_type.
        """
        sample_info = self.samples[idx]
        image_uid = sample_info["uid"] # UID is the filename, e.g., "UID0001.jpg"
        
        # Image path is directly under data_root
        img_path = self.data_root / image_uid
        if not img_path.exists():
            # This indicates an issue with data setup or metadata UIDs not matching filenames
            raise FileNotFoundError(f"Image not found: {img_path}. UID: {image_uid}")

        image = Image.open(img_path).convert("RGB")

        # ROI Cropping for 'date' and 'mint'
        if self.model_type in ["date", "mint"]:
            roi_key_map = {"date": "date_roi", "mint": "mint_mark_roi"}
            roi_data_key = roi_key_map[self.model_type]
            
            # UID in roi_coordinates.json is filename stem (e.g., "UID0001")
            image_uid_stem = Path(image_uid).stem 
            
            if image_uid_stem not in self.roi_coordinates:
                raise ValueError(f"ROI coordinates not found for image stem: {image_uid_stem} "
                                 f"in {self.roi_json_path}. Available keys: {list(self.roi_coordinates.keys())[:5]}...")

            roi_entry = self.roi_coordinates[image_uid_stem]
            if roi_data_key not in roi_entry:
                raise ValueError(f"ROI key '{roi_data_key}' not found for {image_uid_stem}. "
                                 f"Available ROI keys: {list(roi_entry.keys())}")

            coords = roi_entry[roi_data_key]
            x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]
            
            # Apply 5% padding to width and height
            padding_w = int(w * 0.05)
            padding_h = int(h * 0.05)

            # Calculate padded box coordinates, ensuring they are within image bounds
            x1 = max(0, x - padding_w)
            y1 = max(0, y - padding_h)
            x2 = min(image.width, x + w + padding_w)
            y2 = min(image.height, y + h + padding_h)
            
            # Ensure the box has a positive area
            if x1 >= x2 or y1 >= y2:
                 raise ValueError(f"Invalid ROI crop box for {image_uid_stem} after padding: ({x1},{y1},{x2},{y2}). Original: {coords}")
            image = image.crop((x1, y1, x2, y2))

        # Apply transformations
        if self.image_transform:
            image_tensor = self.image_transform(image)
        else:
            # Fallback minimal transform if none provided (e.g., for raw image viewing)
            image_tensor = T.ToTensor()(image)

        # Label extraction
        label = -1 # Default invalid label
        if self.model_type == "side":
            side_str = sample_info.get("side")
            type_str = sample_info.get("type") # e.g., "memorial", "shield", etc. for reverse
            if side_str == "obverse":
                label = 0 # "obverse"
            elif side_str == "reverse":
                if type_str == "memorial": label = 1
                elif type_str == "shield": label = 2
                elif type_str == "wheat": label = 3
                elif type_str == "bicentennial": label = 4
                else: raise ValueError(f"Unknown reverse type '{type_str}' for UID {image_uid}")
            else: raise ValueError(f"Unknown side '{side_str}' for UID {image_uid}")

        elif self.model_type == "orientation":
            angle = sample_info.get("angle")
            if angle is None: raise ValueError(f"Missing 'angle' for orientation, UID {image_uid}")
            label = int(float(angle) / 3)
            label = min(max(0, label), 119) # Clamp to [0, 119]

        elif self.model_type == "date":
            year = sample_info.get("year")
            if year is None: raise ValueError(f"Missing 'year' for date, UID {image_uid}")
            label = int(year) - 1909
            if not (0 <= label < self._base_label_maps["date"].__len__()):
                raise ValueError(f"Year {year} (label {label}) out of range [1909-2024] for UID {image_uid}")

        elif self.model_type == "mint":
            mint_str = sample_info.get("mint_mark")
            if mint_str == "D": label = 0
            elif mint_str == "S": label = 1
            elif mint_str == "P": label = 2
            elif mint_str in ["no_mint", "nomint"]: label = 3
            else: raise ValueError(f"Unknown mint_mark '{mint_str}' for UID {image_uid}")
        
        if label == -1:
            raise ValueError(f"Failed to determine label for UID {image_uid}, model_type {self.model_type}")

        return image_tensor, label


def get_transforms(model_type: str, subset: str = "train", img_size: Tuple[int, int] = (224, 224)) -> Callable:
    """
    Builds and returns the appropriate torchvision transforms pipeline.

    Args:
        model_type (str): 'side', 'orientation', 'date', 'mint'.
        subset (str): 'train' or 'val'/'test'.
        img_size (Tuple[int, int]): Target image size [height, width].
    
    Returns:
        Callable: A torchvision.transforms.Compose object.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Base transforms for val/test (and end of train pipeline)
    val_test_transforms = [
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]

    if subset == "train":
        # Augmentations for training
        train_augs = [
            T.RandomRotation(degrees=3),
            # RandomResizedCrop might change aspect ratio if ratio is not (1.0, 1.0)
            # Spec: ratio = (1:1) -> (1.0, 1.0)
            T.RandomResizedCrop(size=img_size[0], scale=(0.9, 1.0), ratio=(1.0, 1.0)),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05),
            # Apply GaussianBlur with 25% probability
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.25),
        ]
        # RandomHorizontalFlip is NOT for orientation
        if model_type != "orientation":
            train_augs.append(T.RandomHorizontalFlip(p=0.5))
        
        return T.Compose(train_augs + val_test_transforms)
    else: # For "val" or "test"
        return T.Compose(val_test_transforms)


if __name__ == '__main__':
    # Example of how to use the PennyDataset.
    # This assumes the script is in 'src/', and 'data_examples/', 'metadata/' are at the project root.
    print("Running PennyDataset example...")

    try:
        # Construct paths relative to the project root.
        # Assumes this script 'dataset.py' is in 'penny_ai/src/'.
        # So, Path(__file__).parent is 'penny_ai/src/', and .parent.parent is 'penny_ai/'.
        project_root = Path(__file__).resolve().parent.parent
    except NameError: # __file__ is not defined in some interactive environments (e.g. notebook)
        project_root = Path(".").resolve() # Assume current working directory is project root

    example_data_root = project_root / "data_examples"
    example_metadata_path = project_root / "metadata" / "metadata.json"
    example_roi_path = project_root / "metadata" / "roi_coordinates.json"

    print(f"Using Project Root: {project_root}")
    print(f"Looking for example data in: {example_data_root}")
    print(f"Looking for metadata in: {example_metadata_path}")
    print(f"Looking for ROI data in: {example_roi_path}")

    if not example_data_root.exists():
        print(f"ERROR: Example data root not found: {example_data_root}")
        print("Please ensure 'data_examples/' directory exists at the project root.")
        exit()
    if not example_metadata_path.exists():
        print(f"ERROR: Example metadata JSON not found: {example_metadata_path}")
        print("Please ensure 'metadata/metadata.json' exists.")
        exit()
    if not example_roi_path.exists():
        print(f"ERROR: Example ROI JSON not found: {example_roi_path}")
        print("Please ensure 'metadata/roi_coordinates.json' exists.")
        exit()

    model_types_to_test = ["side", "orientation", "date", "mint"]
    subsets_to_test = ["train", "val", "test"]

    for mt in model_types_to_test:
        print(f"\n--- Testing Model Type: {mt} ---")
        label_map_for_type = PennyDataset._base_label_maps.get(mt, {}) # Access static member for example
        num_expected_classes = len(label_map_for_type)

        for sub in subsets_to_test:
            print(f"  Subset: {sub}")
            try:
                transforms = get_transforms(model_type=mt, subset=sub)
                dataset = PennyDataset(
                    data_root=example_data_root,
                    metadata_json_path=example_metadata_path,
                    roi_json_path=example_roi_path,
                    model_type=mt,
                    image_transform=transforms,
                    subset=sub
                )
                print(f"    Number of samples: {len(dataset)}")
                
                if len(dataset) > 0:
                    img, label = dataset[0]
                    print(f"      Sample 0 - Image shape: {img.shape}, Label: {label}")
                    
                    # Verify label is within expected range for the model type
                    assert 0 <= label < num_expected_classes, \
                        f"Label {label} out of range for {mt} (expected < {num_expected_classes})"
                    
                    # Display string label from the map
                    string_label = dataset.current_label_map.get(label, "Label not in map")
                    print(f"      Sample 0 - String Label: '{string_label}'")
                elif len(dataset.all_samples) == 0:
                     print(f"    No relevant samples found for model type '{mt}' in metadata.")
                else:
                    print(f"    No samples assigned to this subset '{sub}' by UID hashing (total relevant: {len(dataset.all_samples)}).")

            except Exception as e:
                print(f"    ERROR testing {mt} - {sub}: {e}")
                import traceback
                traceback.print_exc()
