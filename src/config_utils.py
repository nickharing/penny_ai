# src/config_utils.py
# Purpose: Load training configuration and provide TrainingConfig dataclass
# Author:  <Your Name>
# Date:    2025-05-16
# Dependencies: pyyaml

import logging
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# --------------------------------------------------------------------------- #
PROJECT_ROOT     = Path(__file__).resolve().parents[1]
DEFAULT_CFG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"

# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    # Data paths
    data_root:          Path
    metadata_json_path: Path
    roi_json_path:      Path

    # Model selection & architecture
    model_type:         str                 = "side"
    backbone:           str                 = "coin_clip_vit_b32"

    # Preprocessing
    img_size:           Tuple[int, int]     = (224, 224)

    # Training hyperparameters
    epochs:             int                 = 20
    batch_size:         int                 = 32
    learning_rate:      float               = 1e-4
    weight_decay:       float               = 0.0
    optimizer_betas:    Tuple[float, float] = (0.9, 0.999)

    # LR Scheduler
    lr_scheduler_eta_min:     float          = 0.0
    lr_warmup_epochs:         int            = 0
    lr_scheduler_T_max_epochs: Optional[int] = None

    # Output & quantization
    output_dir:         Path                = Path("output/models")
    quantize:           bool                = False
    qat:                bool                = False

    # Misc
    seed:               int                 = 42
    device:             Optional[str]       = None
    num_workers:        int                 = 4

    # Derived (not from YAML)
    project_root:       Path                = PROJECT_ROOT
    checkpoint_dir:     Path                = field(init=False)
    label_map_dir:      Path                = field(init=False)
    label_map:          Dict[int, str]      = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Resolve and create directories
        self.data_root          = (self.project_root / self.data_root).expanduser().resolve()
        self.metadata_json_path = (self.project_root / self.metadata_json_path).expanduser().resolve()
        self.roi_json_path      = (self.project_root / self.roi_json_path).expanduser().resolve()
        self.output_dir         = (self.project_root / self.output_dir).expanduser().resolve()

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.label_map_dir  = self.output_dir / "label_maps"
        for d in (self.output_dir, self.checkpoint_dir, self.label_map_dir):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Configuration initialized with output_dir=%s", self.output_dir)

def load_config(
    model_key:  str,
    *,
    config_path: Path = DEFAULT_CFG_PATH
) -> TrainingConfig:
    """
    Load the YAML, select section `model_key`, normalize fields,
    and return a TrainingConfig instance.
    """
    logger.info("Loading config for model_key=%s from %s", model_key, config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        full_cfg: Dict[str, Any] = yaml.safe_load(fh)

    if model_key not in full_cfg:
        raise KeyError(f"model_key '{model_key}' not found in {config_path}")

    cfg_dict = full_cfg[model_key].copy()

    # require `learning_rate` (no alias), convert lists → tuples
    if "img_size" in cfg_dict and isinstance(cfg_dict["img_size"], list):
        cfg_dict["img_size"] = tuple(cfg_dict["img_size"])
    if "optimizer_betas" in cfg_dict and isinstance(cfg_dict["optimizer_betas"], list):
        cfg_dict["optimizer_betas"] = tuple(cfg_dict["optimizer_betas"])

    # convert path-like strings to Path
    for key in ("data_root", "metadata_json_path", "roi_json_path", "output_dir"):
        if key in cfg_dict:
            cfg_dict[key] = Path(cfg_dict[key])

    return TrainingConfig(**cfg_dict)
