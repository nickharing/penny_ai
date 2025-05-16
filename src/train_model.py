"""
Minimal training script that relies on CoinClip's ViT-B/32 visual
backbone and a single-layer classifier head.  One epoch smoke-tests
run in CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from coin_clip import CoinClip

from .config_utils import load_config, TrainingConfig
from .dataset import PennyDataset


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def clip_vit_b32(device: str = "cpu") -> nn.Module:
    """Return *just* the visual encoder from CoinClip."""
    wrapper = CoinClip(
        model_name="breezedeus/coin-clip-vit-base-patch32",
        device=device,
    )
    return wrapper.model.visual  # 512-dim output


def build_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    ds_train = PennyDataset(cfg, image_transform=tfm, subset="train")
    ds_val = PennyDataset(cfg, image_transform=tfm, subset="val")

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


# --------------------------------------------------------------------------- #
#  Training loop (single-GPU / CPU, bare-bones)
# --------------------------------------------------------------------------- #
def run_training(cfg: TrainingConfig, device: torch.device) -> None:
    n_classes = len(cfg.label_map)
    backbone = clip_vit_b32(device=device)
    clf_head = nn.Linear(backbone.output_dim, n_classes)

    model = nn.Sequential(backbone, clf_head).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    dl_train, dl_val = build_dataloaders(cfg)

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # --------------------- quick val
        model.eval()
        n_correct = n_total = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                n_correct += (preds == yb).sum().item()
                n_total += yb.size(0)

        acc = n_correct / n_total if n_total else 0.0
        print(f"Epoch {epoch+1}/{cfg.epochs} — val acc: {acc:.3f}", flush=True)

    # ---------------------------------------------------------------- save
    out_dir = cfg.output_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{cfg.model_type}_fp32.pt")
    print(f"Saved weights → {out_dir}", flush=True)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="side", help="model key in YAML")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--cfg_path", default=None, help="override YAML path")
    args = parser.parse_args(argv)

    cfg_path = Path(args.cfg_path) if args.cfg_path else None
    cfg = load_config(args.model, config_path=cfg_path or Path())
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_training(cfg, device)


if __name__ == "__main__":
    main()
