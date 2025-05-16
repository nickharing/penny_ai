# src/train_model.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from coin_clip import CoinClip
from .config_utils import load_config, TrainingConfig
from .dataset      import PennyDataset


def clip_vit_b32(device: str = "cpu") -> nn.Module:
    """Return ViT-B/32 visual backbone from CoinClip."""
    wrapper = CoinClip(
        model_name="breezedeus/coin-clip-vit-base-patch32",
        device=device,
    )
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
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return dl_train, dl_val


def run_training(cfg: TrainingConfig, device: torch.device) -> None:
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

    # LR Scheduler (cosine with optional warmup)
    T_max = cfg.lr_scheduler_T_max_epochs or max(1, cfg.epochs - cfg.lr_warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=cfg.lr_scheduler_eta_min
    )

    criterion = nn.CrossEntropyLoss()
    dl_train, dl_val = build_dataloaders(cfg)

    for epoch in range(cfg.epochs):
        # linear warmup on LR
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

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                preds = model(xb.to(device)).argmax(dim=1)
                correct += (preds == yb.to(device)).sum().item()
                total   += yb.size(0)
        acc = correct / total if total else 0.0
        print(f"Epoch {epoch+1}/{cfg.epochs} — val acc: {acc:.3f}", flush=True)

    # save FP32 checkpoint
    model_dir = cfg.output_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / f"{cfg.model_type}_fp32.pt")
    print(f"Saved FP32 → {model_dir}", flush=True)

    # quantization / QAT stubs
    if cfg.quantize:
        print("Quantization requested but not implemented here.", flush=True)
    if cfg.qat:
        print("QAT requested but not implemented here.", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model",      default="side")
    p.add_argument("--epochs",   type=int)
    p.add_argument("--batch_size",type=int)
    p.add_argument("--device",    choices=["cpu", "cuda"])
    p.add_argument("--cfg_path",  type=str)
    args = p.parse_args(argv)

    # load YAML
    if args.cfg_path:
        cfg = load_config(args.model, config_path=Path(args.cfg_path))
    else:
        cfg = load_config(args.model)

    # override CLI params
    if args.epochs:     cfg.epochs     = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.device:     cfg.device     = args.device

    # decide device
    use_dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_training(cfg, torch.device(use_dev))


if __name__ == "__main__":
    main()
