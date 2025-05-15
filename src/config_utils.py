# src/config_utils.py
# Utilities for loading and managing training configurations from a master YAML file.

import yaml
import os
from pathlib import Path # Added Path for DEFAULT_CFG
from dataclasses import dataclass, field
from typing import List, Optional

# Default configuration file path as per the hand-off document
DEFAULT_CFG_PATH = Path("configs/training_config.yaml")

@dataclass
class TrainingConfig:
    """
    Dataclass to hold all training configuration parameters.

    Attributes:
        data_root (str): Root directory for image data.
        metadata_json_path (str): Path to the metadata JSON file.
        roi_json_path (str): Path to the ROI coordinates JSON file.
        output_dir (str): Base directory for saving exported ONNX models for the specific model type.
        model_type (str): Type of model to train ('date', 'mint', 'orientation', 'side'). This is crucial
                          and should match the key used to load the config section.
        backbone (str): Name of the backbone model architecture to use.
        img_size (List[int]): Target image size [height, width] for model input.
        num_classes (Optional[int]): Number of output classes. Auto-determined in __post_init__.
        num_epochs (int): Total number of training epochs.
        batch_size (int): Batch size for training and validation.
        lr (float): Initial learning rate.
        weight_decay (float): Weight decay for the optimizer.
        optimizer_betas (List[float]): Beta values for AdamW optimizer.
        lr_scheduler_T_max_epochs (Optional[int]): T_max for CosineAnnealingLR.
                                                  Defaults to num_epochs - lr_warmup_epochs.
        lr_scheduler_eta_min (float): Minimum learning rate for CosineAnnealingLR.
        lr_warmup_epochs (int): Number of linear warmup epochs.
        quantize (bool): Whether to perform post-training quantization (INT8).
        qat (bool): Whether to perform Quantization Aware Training (future stub).
        seed (int): Random seed for reproducibility.
        device (Optional[str]): Device for training ('cuda', 'cpu', 'cuda:0').
                                Auto-detects if None.
        num_workers (int): Number of DataLoader worker processes.
        checkpoint_dir (Optional[str]): Directory to save training checkpoints. Auto-derived.
        log_dir (Optional[str]): Directory to save training logs. Auto-derived.
        label_map_dir (Optional[str]): Directory to save label maps. Auto-derived.
    """
    # Data paths
    data_root: str = "data/"
    metadata_json_path: str = "metadata/metadata.json"
    roi_json_path: str = "metadata/roi_coordinates.json"
    output_dir: str = "output/models/default_model" # Will be specific to model_type from YAML

    # Model selection and architecture
    model_type: str # This will be set from the specific config section
    backbone: str = "coin_clip_vit_b32"
    img_size: List[int] = field(default_factory=lambda: [224, 224])
    num_classes: Optional[int] = None

    # Training hyperparameters
    num_epochs: int = 30
    batch_size: int = 64
    lr: float = 0.0003
    weight_decay: float = 0.05
    optimizer_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

    # LR Scheduler (CosineAnnealingLR)
    lr_scheduler_T_max_epochs: Optional[int] = None
    lr_scheduler_eta_min: float = 0.000001 # η_min in spec
    lr_warmup_epochs: int = 1

    # ONNX Export
    quantize: bool = True
    qat: bool = False # Stub for future QAT

    # Miscellaneous
    seed: int = 42
    device: Optional[str] = None # e.g., 'cuda', 'cpu', 'cuda:0'. Auto-detect if None.
    num_workers: int = 4

    # Derived paths (not directly set from YAML, but useful internally)
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    label_map_dir: Optional[str] = None


    def __post_init__(self):
        """
        Post-initialization checks and derivations.
        Sets num_classes, adjusts scheduler T_max, and defines output sub-directories.
        """
        valid_model_types = ["date", "mint", "orientation", "side"]
        if self.model_type not in valid_model_types:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}. "
                f"Must be one of {valid_model_types}."
            )

        # Determine num_classes based on model_type
        if self.model_type == "date":
            self.num_classes = 2024 - 1909 + 1 # 116 classes
        elif self.model_type == "mint":
            self.num_classes = 4 # "D", "S", "P", "nomint"
        elif self.model_type == "orientation":
            self.num_classes = 120 # 360 degrees / 3 degree bins
        elif self.model_type == "side":
            self.num_classes = 5 # {obverse, memorial, shield, wheat, bicentennial}
        else:
            # This case should not be reached due to the check above
            raise ValueError(f"Unknown model_type for num_classes: {self.model_type}")

        # Adjust T_max for cosine annealing scheduler if using warmup
        if self.lr_scheduler_T_max_epochs is None: # If not specified in YAML
            self.lr_scheduler_T_max_epochs = self.num_epochs - self.lr_warmup_epochs
            if self.lr_scheduler_T_max_epochs <= 0:
                # Ensure T_max is at least 1 if warmup is >= num_epochs
                self.lr_scheduler_T_max_epochs = 1
        
        # Define checkpoint, log, and label_map directories based on a common structure
        # output_dir from YAML (e.g., 'models/date') is for final ONNX models.
        project_root_output_base = Path("output")
        self.checkpoint_dir = project_root_output_base / "checkpoints"
        self.log_dir = project_root_output_base / "training_logs"
        self.label_map_dir = project_root_output_base / "label_maps"


def load_config(model_config_key: str, config_path: Path = DEFAULT_CFG_PATH) -> TrainingConfig:
    """
    Loads a specific model's training configuration from the master YAML file.

    Args:
        model_config_key (str): The top-level key in the YAML file corresponding to the
                                desired model's configuration (e.g., "date", "mint", "side").
        config_path (Path): Path to the master YAML configuration file.
                            Defaults to DEFAULT_CFG_PATH.

    Returns:
        TrainingConfig: An object populated with settings for the specified model.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If there's an error parsing YAML, the model_config_key is not found,
                    or essential keys like 'model_type' are missing or mismatched
                    in the specified section.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Master configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            full_config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing master YAML configuration file: {config_path}\n{e}")

    if full_config_dict is None:
        raise ValueError(f"Master configuration file is empty or invalid: {config_path}")

    if model_config_key not in full_config_dict:
        raise ValueError(
            f"Configuration key '{model_config_key}' not found in "
            f"master config file: {config_path}. Available keys: {list(full_config_dict.keys())}"
        )

    specific_model_config_dict = full_config_dict[model_config_key]

    if not isinstance(specific_model_config_dict, dict):
        raise ValueError(
            f"The section '{model_config_key}' in {config_path} is not a valid "
            "configuration dictionary."
        )

    # Ensure 'model_type' is present in the specific config section and matches the key.
    # This is crucial because TrainingConfig.__post_init__ uses self.model_type.
    if 'model_type' not in specific_model_config_dict:
        # If not present, infer it from the model_config_key (which is the section name)
        specific_model_config_dict['model_type'] = model_config_key
    elif specific_model_config_dict['model_type'] != model_config_key:
        # If present but mismatched, raise an error for clarity.
        raise ValueError(
            f"Mismatch in configuration section '{model_config_key}': "
            f"Section key implies model_type '{model_config_key}', but YAML specifies "
            f"'model_type: {specific_model_config_dict['model_type']}'. "
            "Ensure consistency or remove 'model_type' from the YAML section "
            "to allow it to be inferred from the section key."
        )
    
    try:
        # Unpack the specific model's configuration dictionary into TrainingConfig
        config = TrainingConfig(**specific_model_config_dict)
    except TypeError as e:
        # This can happen if YAML has unexpected keys or misses keys that don't have defaults
        # in TrainingConfig.
        raise ValueError(
            f"Error creating TrainingConfig for '{model_config_key}' from YAML. "
            f"Check for missing or misspelled keys in section '{model_config_key}' of {config_path}. "
            f"Details: {e}"
        )
    return config
