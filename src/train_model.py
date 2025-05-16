"""
Minimal training script that relies on CoinClip's ViT-B/32 visual
backbone and a single-layer classifier head.  One-epoch smoke-tests
run in CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data  import DataLoader
from torchvision       import transforms

from coin_clip import CoinClip
from .config_utils import load_config, TrainingConfig
from .dataset      import PennyDataset


def clip_vit_b32(device: str = "cpu") -> nn.Module:
    """Return the visual encoder from CoinClip (512-dim output)."""
    wrapper = CoinClip(
        model_name="breezedeus/coin-clip-vit-base-patch32",
        device=device,
    )
    return wrapper.model.visual


def build_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    # Use cfg.img_size instead of hard-coded 224
    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
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


def run_training(cfg: TrainingConfig, device: torch.device) -> None:
    n_classes = len(cfg.label_map)
    backbone  = clip_vit_b32(device=device)
    head      = nn.Linear(backbone.output_dim, n_classes)
    model     = nn.Sequential(backbone, head).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), learning_rate=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    dl_train, dl_val = build_dataloaders(cfg)

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    # save final FP32 weights
    out_dir = cfg.output_dir / "models"
    out_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), out_dir / f"{cfg.model_type}_fp32.pt")
    print(f"Saved → {out_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model",      default="side")
    p.add_argument("--epochs",   type=int)
    p.add_argument("--batch_size",type=int)
    p.add_argument("--device",    default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--cfg_path",  default=None)
    args = p.parse_args(argv)

    cfg_path = Path(args.cfg_path) if args.cfg_path else None
    cfg = load_config(args.model, config_path=cfg_path or Path())
    if args.epochs:    cfg.epochs     = args.epochs
    if args.batch_size:cfg.batch_size = args.batch_size

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_training(cfg, device)


if __name__ == "__main__":
    main()
