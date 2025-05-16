from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


# --------------------------------------------------------------------------- #
#  Globals
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"


# --------------------------------------------------------------------------- #
#  Dataclass
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    # ------------------------------------------------------------------ data
    data_root: Path
    metadata_json_path: Path
    roi_json_path: Path

    # ------------------------------------------------------------------ model
    model_type: str = "side"          # e.g. side, date, mint
    backbone: str = "coin_clip_vit_b32"

    # ---------------------------------------------------------------- params
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    num_workers: int = 4

    # ---------------------------------------------------------------- output
    project_root: Path = PROJECT_ROOT
    output_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    label_map_dir: Path = field(init=False)

    # ---------------------------------------------------------------- runtime
    label_map: Dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.output_dir = self.project_root / "output"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.label_map_dir = self.output_dir / "label_maps"

        # ensure paths exist
        for p in (self.output_dir, self.checkpoint_dir, self.label_map_dir):
            p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def load_config(model_key: str, *, config_path: Path = DEFAULT_CFG_PATH) -> TrainingConfig:
    """Load the YAML and return a populated `TrainingConfig`."""
    with open(config_path, "r", encoding="utf-8") as fh:
        raw_yaml: Dict[str, Any] = yaml.safe_load(fh)

    if model_key not in raw_yaml:
        raise KeyError(f"model_key '{model_key}' not found in {config_path}")

    cfg_dict = raw_yaml[model_key]
    # Expand any relative paths
    for key in ("data_root", "metadata_json_path", "roi_json_path"):
        cfg_dict[key] = Path(cfg_dict[key]).expanduser().resolve()

    return TrainingConfig(**cfg_dict)
