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
    # repo_root = Path(__file__).resolve().parents[1] # No longer needed

    # Load config using the 'test_paths' section for inputs
    # The use_test_paths flag was removed from load_config, so we ensure test_paths are set in training_config.yaml
    # and then override output paths here for test isolation.
    cfg = load_config("side", use_test_paths=True) # Ensures cfg.paths.data_root etc. point to test data

    # Override output paths to use the temporary directory provided by pytest
    cfg.paths.models = tmp_path / "models_output"
    cfg.paths.exports = tmp_path / "exports_output"
    # cfg.paths.logs can also be overridden if logs were written by run_training

    # Minimal training
    cfg.epochs     = 1
    cfg.batch_size = 8

    logger.info("Running real-data smoke test with cfg: %s", cfg)
    run_training(cfg, torch.device("cpu"))

    # Check that outputs were created
    # Paths are now derived from cfg.paths as used in run_training
    onnx_output_dir = Path(cfg.paths.exports) / "onnx"
    checkpoint_output_dir = Path(cfg.paths.models) / "checkpoints"
    fp32_onnx = onnx_output_dir / "side_fp32.onnx"
    ckpt      = checkpoint_output_dir / "side_fp32.pt"

    assert fp32_onnx.exists(), f"Expected ONNX at {fp32_onnx}"
    assert ckpt.exists(),      f"Expected checkpoint at {ckpt}"

    logger.info("Real-data smoke test successful: %s and %s", ckpt, fp32_onnx)
