"""
One-epoch smoke test used in CI.  Runs CPU-only, no pretrained weights
to avoid HF download.
"""

import torch

from src.config_utils import load_config
from src.train_model import run_training


def test_one_epoch_cpu(tmp_path):
    cfg = load_config("side")
    cfg.epochs = 1
    cfg.batch_size = 4
    cfg.output_dir = tmp_path   # write artefacts under pytest temp dir

    run_training(cfg, torch.device("cpu"))
