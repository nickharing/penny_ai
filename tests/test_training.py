# tests/test_training.py
from src.config_utils import load_config
from src.train_model   import run_training

import torch


def test_one_epoch_cpu(tmp_path):
    cfg = load_config("side")
    cfg.epochs     = 1
    cfg.batch_size = 4
    cfg.output_dir = tmp_path      # write outputs under pytest's folder
    run_training(cfg, torch.device("cpu"))
