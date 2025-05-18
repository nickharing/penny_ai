# config_utils.py
# Purpose: Load and merge training configuration for different coin classification models
# Author: Nick Haring
# Date: 2025-05-17
# Dependencies: yaml, dataclasses, pathlib, typing, torch

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "training_config.yaml"

@dataclass
class PathsConfig:
    data_root: Path
    metadata: Path
    roi: Path
    models: Path
    logs: Path
    exports: Path

@dataclass
class DataDefaults:
    normalization: Dict[str, List[float]]
    roi_padding: Dict[str, Dict[str, float]]

@dataclass
class AugmentationDefaults:
    apply_augmentations: bool
    random_horizontal_flip: bool
    rotation_degrees: float
    translate: List[float]
    scale: List[float]
    shear: float
    random_erasing_prob: float
    random_erasing_scale: List[float]
    random_erasing_ratio: List[float]
    gaussian_noise_std: float

@dataclass
class TrainingDefaults:
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    scheduler: str
    warmup_epochs: int
    early_stopping: Dict[str, Any]
    save_every_n_epochs: int

@dataclass
class ExecutionDefaults:
    seed: int
    device: torch.device
    num_workers: int

@dataclass
class ModelConfig:
    model_type: str # Added to store the name of the model configuration
    # Paths
    paths: PathsConfig
    # Data settings
    image_size: List[int]
    normalization: List[float]
    std: List[float]
    roi_padding: Dict[str, Dict[str, float]]
    # Augmentations
    apply_augmentations: bool
    random_horizontal_flip: bool
    rotation_degrees: float
    translate: List[float]
    scale: List[float]
    shear: float
    random_erasing_prob: float
    random_erasing_scale: List[float]
    random_erasing_ratio: List[float]
    gaussian_noise_std: float
    # Training
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    scheduler: str
    warmup_epochs: int
    early_stopping: Dict[str, Any]
    save_every_n_epochs: int

    # Execution
    seed: int
    device: torch.device
    num_workers: int


def load_config(model: str) -> ModelConfig:
    """
    Load configuration for a given model by merging global defaults with per-model overrides.
    Args:
        model: One of the keys under 'models' in the YAML (e.g., 'side', 'date', etc.).
    Returns:
        ModelConfig with all parameters resolved.
    """
    cfg_raw = yaml.safe_load(CONFIG_PATH.read_text())
    # Paths
    paths_raw = cfg_raw.get("paths", {})
    paths = PathsConfig(
        data_root=Path(paths_raw["data_root"]).resolve(),
        metadata=Path(paths_raw["metadata"]).resolve(),
        roi=Path(paths_raw["roi"]).resolve(),
        models=Path(paths_raw["models"]).resolve(),
        logs=Path(paths_raw["logs"]).resolve(),
        exports=Path(paths_raw["exports"]).resolve(),
    )
    # Global defaults
    defaults = cfg_raw.get("defaults", {})
    data_def = defaults.get("data", {})
    aug_def = defaults.get("augmentation", {})
    train_def = defaults.get("training", {})
    exec_def = defaults.get("execution", {})
    # Per-model overrides
    model_raw = cfg_raw.get("models", {}).get(model, {})
    # Image size
    image_size = model_raw.get("data", {}).get("image_size")
    # Training overrides
    tr_ovr = model_raw.get("training", {})
    # Augmentation overrides
    aug_ovr = model_raw.get("augmentation", {})
    # Execution: use global

    # Resolve device
    dev_cfg = exec_def.get("device", "auto")
    if dev_cfg in ("auto", None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_cfg)

    return ModelConfig(
        model_type=model, # Store the model type
        paths=paths,
        image_size=image_size if image_size else data_def.get("image_size", []),
        normalization=data_def.get("normalization", {}).get("mean", []),
        std=data_def.get("normalization", {}).get("std", []),
        roi_padding=data_def.get("roi_padding", {}),
        apply_augmentations=aug_ovr.get("apply_augmentations", aug_def.get("apply_augmentations", False)),
        random_horizontal_flip=aug_ovr.get("random_horizontal_flip", aug_def.get("random_horizontal_flip", False)),
        rotation_degrees=aug_ovr.get("rotation_degrees", aug_def.get("rotation_degrees", 0)),
        translate=aug_ovr.get("translate", aug_def.get("translate", [0,0])),
        scale=aug_ovr.get("scale", aug_def.get("scale", [1,1])),
        shear=aug_ovr.get("shear", aug_def.get("shear", 0)),
        random_erasing_prob=aug_ovr.get("random_erasing_prob", aug_def.get("random_erasing_prob", 0)),
        random_erasing_scale=aug_ovr.get("random_erasing_scale", aug_def.get("random_erasing_scale", [0,0])),
        random_erasing_ratio=aug_ovr.get("random_erasing_ratio", aug_def.get("random_erasing_ratio", [1,1])),
        gaussian_noise_std=aug_ovr.get("gaussian_noise_std", aug_def.get("gaussian_noise_std", 0)),
        epochs=tr_ovr.get("epochs", train_def.get("epochs", 1)),
        batch_size=tr_ovr.get("batch_size", train_def.get("batch_size", 1)),
        learning_rate=tr_ovr.get("learning_rate", train_def.get("learning_rate", 1e-3)),
        optimizer=tr_ovr.get("optimizer", train_def.get("optimizer", "adam")),
        weight_decay=tr_ovr.get("weight_decay", train_def.get("weight_decay", 0)),
        scheduler=tr_ovr.get("scheduler", train_def.get("scheduler", None)),
        warmup_epochs=tr_ovr.get("warmup_epochs", train_def.get("warmup_epochs", 0)),
        early_stopping=tr_ovr.get("early_stopping", train_def.get("early_stopping", {})),
        save_every_n_epochs=tr_ovr.get("save_every_n_epochs", train_def.get("save_every_n_epochs", 1)),
        seed=exec_def.get("seed", 0),
        device=device,
        num_workers=exec_def.get("num_workers", 0),
    )
