# src/train_model.py
# Purpose: Train classifier head on CoinClip ViT-B/32 and export ONNX
# Author:  <Your Name>
# Date:    2025-05-16
# Dependencies: torch, torchvision, coin_clip, onnx, onnxruntime-tools

import argparse
import logging
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import onnx
from onnxruntime_tools import quantization

from coin_clip import CoinClip
from .config_utils import load_config, TrainingConfig
from .dataset      import PennyDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def clip_vit_b32(device: str = "cpu") -> nn.Module:
    wrapper = CoinClip(
        model_name="breezedeus/coin-clip-vit-base-patch32",
        device=device,
    )
    logger.info("Loaded CoinClip backbone on %s", device)
    return wrapper.model.visual


def build_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
    ])
    ds_train = PennyDataset(cfg, image_transform=tfm, subset="train")
    ds_val   = PennyDataset(cfg, image_transform=tfm, subset="val")

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return dl_train, dl_val


def export_onnx(model: nn.Module, dummy_input: torch.Tensor, path: Path):
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    logger.info("ONNX model saved to %s", path)


def quantize_int8(fp32_path: Path, int8_path: Path):
    quantization.quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=quantization.QuantType.QInt8,
    )
    logger.info("INT8 quantized model saved to %s", int8_path)


def run_training(cfg: TrainingConfig, device: torch.device) -> None:
    logger.info("Starting training: %s", cfg)
    # reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    n_classes = len(cfg.label_map)
    backbone  = clip_vit_b32(device=str(device))
    head      = nn.Linear(backbone.output_dim, n_classes)
    model     = nn.Sequential(backbone, head).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.optimizer_betas,
        weight_decay=cfg.weight_decay,
    )

    T_max = cfg.lr_scheduler_T_max_epochs or max(1, cfg.epochs - cfg.lr_warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=cfg.lr_scheduler_eta_min
    )

    criterion = nn.CrossEntropyLoss()
    dl_train, dl_val = build_dataloaders(cfg)

    for epoch in range(cfg.epochs):
        if epoch < cfg.lr_warmup_epochs:
            factor = (epoch + 1) / cfg.lr_warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate * factor

        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                preds = model(xb.to(device)).argmax(dim=1)
                correct += (preds == yb.to(device)).sum().item()
                total   += yb.size(0)
        acc = correct / total if total else 0.0
        logger.info("Epoch %d/%d — val acc: %.3f", epoch+1, cfg.epochs, acc)

    # save PyTorch checkpoint
    ckpt_dir = cfg.checkpoint_dir
    pth_path = ckpt_dir / f"{cfg.model_type}_fp32.pt"
    torch.save(model.state_dict(), pth_path)
    logger.info("Saved PyTorch checkpoint to %s", pth_path)

    # ONNX export
    onnx_dir = cfg.output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, cfg.img_size[0], cfg.img_size[1], device=device)
    fp32_path = onnx_dir / f"{cfg.model_type}_fp32.onnx"
    export_onnx(model, dummy, fp32_path)

    int8_path = onnx_dir / f"{cfg.model_type}_int8.onnx"
    if cfg.quantize:
        quantize_int8(fp32_path, int8_path)
    else:
        logger.info("Skipping quantization; set quantize: true to enable")

    if cfg.qat:
        logger.warning("QAT requested but not yet implemented")

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",      default="side")
    parser.add_argument("--epochs",   type=int)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--device",    choices=["cpu", "cuda"])
    parser.add_argument("--cfg_path",  type=str)
    args = parser.parse_args(argv)

    if args.cfg_path:
        cfg = load_config(args.model, config_path=Path(args.cfg_path))
    else:
        cfg = load_config(args.model)

    if args.epochs:     cfg.epochs     = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.device:     cfg.device     = args.device

    use_dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_training(cfg, torch.device(use_dev))

if __name__ == "__main__":
    main()
