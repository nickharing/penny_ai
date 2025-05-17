# tests/test_training.py
# Purpose: End-to-end smoke-test on real data_examples using metadata.json
# Author:  <Your Name>
# Date:    2025-05-16
# Dependencies: pytest, torch

import logging
from pathlib import Path

import pytest
import torch

from src.config_utils import load_config
from src.train_model   import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def test_one_epoch_with_real_data(tmp_path: Path):
    """
    Run a single training epoch on the actual data_examples set,
    using the committed metadata/metadata.json and roi_coordinates.json.
    """
    repo_root = Path(__file__).resolve().parents[1]

    # Load config & override to point at real assets
    cfg = load_config("side")
    cfg.data_root          = repo_root / "data_examples"
    cfg.metadata_json_path = repo_root / "metadata" / "metadata.json"
    cfg.roi_json_path      = repo_root / "metadata" / "roi_coordinates.json"
    cfg.output_dir         = tmp_path / "output"

    # Minimal training
    cfg.epochs     = 1
    cfg.batch_size = 8

    logger.info("Running real-data smoke test with cfg: %s", cfg)
    run_training(cfg, torch.device("cpu"))

    # Check that outputs were created
    fp32_onnx = cfg.output_dir / "onnx" / "side_fp32.onnx"
    ckpt      = cfg.checkpoint_dir / "side_fp32.pt"

    assert fp32_onnx.exists(), f"Expected ONNX at {fp32_onnx}"
    assert ckpt.exists(),      f"Expected checkpoint at {ckpt}"

    logger.info("Real-data smoke test successful: %s and %s", ckpt, fp32_onnx)
