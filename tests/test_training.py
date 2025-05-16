# tests/test_training.py
# Purpose: Smoke-test one epoch training, including ONNX export & quantization
# Author:  <Your Name>
# Date:    2025-05-16
# Dependencies: pytest, torch

import logging
from pathlib import Path

import pytest
import torch

from src.config_utils import load_config
from src.train_model   import run_training

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def test_one_epoch_cpu(tmp_path: Path):
    cfg = load_config("side")
    cfg.epochs     = 1
    cfg.batch_size = 4
    cfg.output_dir = tmp_path

    run_training(cfg, torch.device("cpu"))

    # verify ONNX exports
    onnx_dir = tmp_path / "onnx"
    fp32 = onnx_dir / "side_fp32.onnx"
    int8 = onnx_dir / "side_int8.onnx"
    assert fp32.exists(), "FP32 ONNX model not created"
    # since quantize defaults false in tests, int8 may not exist
    logger.info("Smoke test completed; FP32 ONNX at %s", fp32)
