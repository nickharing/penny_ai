# src/config_utils.py
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --------------------------------------------------------------------------- #
#  Globals
# --------------------------------------------------------------------------- #
PROJECT_ROOT     = Path(__file__).resolve().parents[1]
DEFAULT_CFG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"


# --------------------------------------------------------------------------- #
#  Dataclass
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    # Data paths
    data_root:            Path
    metadata_json_path:   Path
    roi_json_path:        Path

    # Model selection & architecture
    model_type:           str                  = "side"
    backbone:             str                  = "coin_clip_vit_b32"

    # Preprocessing
    img_size:             Tuple[int, int]      = (224, 224)

    # Training hyperparameters
    epochs:               int                  = 20
    batch_size:           int                  = 32
    learning_rate:        float                = 1e-4
    weight_decay:         float                = 0.0
    optimizer_betas:      Tuple[float, float]  = (0.9, 0.999)

    # LR Scheduler
    lr_scheduler_eta_min: float                = 0.0
    lr_warmup_epochs:     int                  = 0
    lr_scheduler_T_max_epochs: Optional[int]   = None

    # Output & quantization
    output_dir:           Path                 = Path("output/models")
    quantize:             bool                 = False
    qat:                  bool                 = False

    # Misc
    seed:                 int                  = 42
    device:               Optional[str]        = None
    num_workers:          int                  = 4

    # Derived (not from YAML)
    project_root:         Path                 = PROJECT_ROOT
    checkpoint_dir:       Path                 = field(init=False)
    label_map_dir:        Path                 = field(init=False)
    label_map:            Dict[int, str]       = field(default_factory=dict)

    def __post_init__(self) -> None:
        # resolve all relative paths against project root
        self.data_root          = (self.project_root / self.data_root).expanduser().resolve()
        self.metadata_json_path = (self.project_root / self.metadata_json_path).expanduser().resolve()
        self.roi_json_path      = (self.project_root / self.roi_json_path).expanduser().resolve()
        self.output_dir         = (self.project_root / self.output_dir).expanduser().resolve()

        # set up output subfolders
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.label_map_dir  = self.output_dir / "label_maps"
        for d in (self.output_dir, self.checkpoint_dir, self.label_map_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(
    model_key:  str,
    *,
    config_path: Path = DEFAULT_CFG_PATH
) -> TrainingConfig:
    """
    Load the YAML, pick the top-level key `model_key`, normalize
    and alias fields, then return a TrainingConfig.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        full_cfg: Dict[str, Any] = yaml.safe_load(fh)

    if model_key not in full_cfg:
        raise KeyError(f"model_key '{model_key}' not found in {config_path}")

    cfg_dict = full_cfg[model_key].copy()

    # alias 'lr' → 'learning_rate'
    if "lr" in cfg_dict and "learning_rate" not in cfg_dict:
        cfg_dict["learning_rate"] = cfg_dict.pop("lr")

    # list → tuple for img_size & optimizer_betas
    if "img_size" in cfg_dict and isinstance(cfg_dict["img_size"], list):
        cfg_dict["img_size"] = tuple(cfg_dict["img_size"])
    if "optimizer_betas" in cfg_dict and isinstance(cfg_dict["optimizer_betas"], list):
        cfg_dict["optimizer_betas"] = tuple(cfg_dict["optimizer_betas"])

    # Path fields
    for key in ("data_root", "metadata_json_path", "roi_json_path", "output_dir"):
        if key in cfg_dict:
            cfg_dict[key] = Path(cfg_dict[key])

    return TrainingConfig(**cfg_dict)
