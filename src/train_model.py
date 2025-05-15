#!/usr/bin/env python3
# src/train_model.py
# Purpose: Train penny-classification models (side, orientation, date, mint)
#          and export FP32 + INT8 ONNX.
#
# Works with coin-clip==0.1  (PyPI). That build exposes CoinClip but not
# coin_clip.models. We wrap CoinClip to return the ViT-B/32 backbone.
#
# Author: <Your Name>
# Date: 2025-05-15

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------#
# third-party deps
# -----------------------------------------------------------------------------#
try:
    from coin_clip import CoinClip
except ModuleNotFoundError as e:  # pragma: no cover
    raise RuntimeError(
        "coin_clip==0.1 not found. Install with:\n"
        "    pip install coin-clip==0.1"
    ) from e

from onnxruntime.quantization import quantize_dynamic, QuantType

# project modules
from config_utils import TrainingConfig, load_config, DEFAULT_CFG_PATH
from dataset import PennyDataset

# -----------------------------------------------------------------------------#
# backbone registry (ViT-B/32 only for now)
# -----------------------------------------------------------------------------#
def clip_vit_b32(pretrained: bool = True) -> nn.Module:
    """
    Wrap CoinClip (0.1) to expose the raw ViT-B/32 vision backbone.

    pretrained=True downloads 'breezedeus/coin-clip-vit-base-patch32' weights
    the first time and caches them under ~/.cache/clip.
    """
    clip_wrapper = CoinClip(
        model_name="breezedeus/coin-clip-vit-base-patch32",
        device="cpu",           # we'll move to correct device later
        cache_dir=None,         # default HF cache
    )
    return clip_wrapper.model.visual  # torch.nn.Module with .output_dim == 512


BACKBONES: Dict[str, Dict[str, object]] = {
    "coin_clip_vit_b32": dict(factory=clip_vit_b32, embed_dim=512),
}

# -----------------------------------------------------------------------------#
# util helpers
# -----------------------------------------------------------------------------#
LOG_DIR = Path("output/training_logs")
CKPT_DIR = Path("output/checkpoints")
LABEL_DIR = Path("output/label_maps")
MODEL_DIR = Path("models")


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def make_head(in_dim: int, num_classes: int) -> nn.Module:
    return nn.Linear(in_dim, num_classes)


def make_imx500_compatible(model: nn.Module) -> nn.Module:  # placeholder
    logging.warning(
        "make_imx500_compatible() is currently a no-op. "
        "Models may contain IMX-unsupported ops."
    )
    return model


# -----------------------------------------------------------------------------#
# Trainer
# -----------------------------------------------------------------------------#
class Trainer:
    def __init__(self, cfg: TrainingConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.loaders = self._build_dataloaders()
        self.model = self._build_model().to(device)
        self.scaler = GradScaler(enabled=(device.type == "cuda"))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.num_epochs, eta_min=1e-6
        )

        self.start_epoch = 0
        self.best_f1 = 0.0

        LABEL_DIR.mkdir(parents=True, exist_ok=True)
        (LABEL_DIR / f"{cfg.model_type}_label_map.json").write_text(
            json.dumps(cfg.label_map, indent=2)
        )

        if cfg.resume_path:
            self._load_checkpoint(cfg.resume_path)

    # ------------------------------------------------------------------ #
    # Data / model builders
    # ------------------------------------------------------------------ #
    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        loaders = {}
        for split in ("train", "val", "test"):
            ds = PennyDataset(self.cfg, split=split)
            loaders[split] = DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                shuffle=(split == "train"),
                num_workers=self.cfg.num_workers,
                pin_memory=(self.device.type == "cuda"),
            )
        logging.info(
            "Dataset sizes — train:%d  val:%d  test:%d",
            len(loaders["train"].dataset),
            len(loaders["val"].dataset),
            len(loaders["test"].dataset),
        )
        return loaders

    def _build_model(self) -> nn.Module:
        bb_cfg = BACKBONES[self.cfg.backbone]
        backbone = bb_cfg["factory"](pretrained=True)
        head = make_head(bb_cfg["embed_dim"], self.cfg.num_classes)
        model = nn.Sequential(backbone, nn.Flatten(1), head)
        logging.info(
            "Model: ViT-B/32 backbone → Linear(%d → %d)",
            bb_cfg["embed_dim"],
            self.cfg.num_classes,
        )
        return model

    # ------------------------------------------------------------------ #
    # Train / val loops
    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.num_epochs):
            self._train_epoch(epoch)
            val = self._eval_split("val")
            self.scheduler.step()

            if val["f1"] > self.best_f1:
                self.best_f1 = val["f1"]
                self._save_ckpt(epoch, best=True)
            self._save_ckpt(epoch, best=False)

        test = self._eval_split("test")
        logging.info("TEST metrics: %s", test)
        self._export_onnx()

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0

        for x, y in self.loaders["train"]:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(self.device.type == "cuda")):
                out = self.model(x)
                loss = self.criterion(out, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            run_loss += loss.item() * x.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
            run_total += x.size(0)

        logging.info(
            "Epoch %d/%d — train loss %.4f acc %.3f",
            epoch + 1,
            self.cfg.num_epochs,
            run_loss / run_total,
            run_correct / run_total,
        )

    @torch.no_grad()
    def _eval_split(self, split: str) -> Dict[str, float]:
        self.model.eval()
        loss_sum, y_true, y_pred = 0.0, [], []
        for x, y in self.loaders[split]:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            loss_sum += self.criterion(out, y).item() * x.size(0)
            y_true.extend(y.cpu())
            y_pred.extend(out.argmax(1).cpu())

        loss = loss_sum / len(self.loaders[split].dataset)
        acc = (torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item()
        f1 = f1_score(y_true, y_pred, average="macro")
        logging.info("%s — loss %.4f acc %.3f f1 %.3f", split.upper(), loss, acc, f1)
        return {"loss": loss, "acc": acc, "f1": f1}

    # ------------------------------------------------------------------ #
    # Checkpoints / export
    # ------------------------------------------------------------------ #
    def _save_ckpt(self, epoch: int, *, best: bool) -> None:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        tag = "best" if best else "last"
        path = CKPT_DIR / f"{self.cfg.model_type}_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_f1": self.best_f1,
                "cfg": self.cfg.to_dict(),
            },
            path,
        )

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_f1 = ckpt["best_f1"]
        logging.info("Resumed from %s (epoch %d)", path, ckpt["epoch"])

    def _export_onnx(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        ts = timestamp()
        fp32 = MODEL_DIR / f"{self.cfg.model_type}_{ts}_fp32.onnx"
        int8 = MODEL_DIR / f"{self.cfg.model_type}_{ts}_int8.onnx"

        dummy = torch.randn(1, 3, *self.cfg.img_size)
        model_cpu = make_imx500_compatible(self.model).to("cpu").eval()

        logging.info("Exporting FP32 ONNX to %s", fp32)
        torch.onnx.export(
            model_cpu,
            dummy,
            fp32,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        onnx_model = torch.onnx.load(fp32)
        onnx_model.metadata_props.add(
            key="label_map", value=json.dumps(self.cfg.label_map)
        )
        torch.onnx.save(onnx_model, fp32)

        logging.info("Quantizing dynamic INT8 → %s", int8)
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)

    # ------------------------------------------------------------------ #
    # Data helpers
    # ------------------------------------------------------------------ #
    def _build_dataloaders(self):  # left here to satisfy type checker above
        raise NotImplementedError


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Penny AI trainer")
    p.add_argument("-m", "--model", required=True,
                   choices=["side", "orientation", "date", "mint"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--device")
    p.add_argument("--resume", type=Path)
    p.add_argument("--config_file", type=Path)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--data_examples", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_cli()

    cfg_path = args.config_file or DEFAULT_CFG_PATH
    cfg = load_config(cfg_path, model_type=args.model)

    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.device:
        cfg.device_override = args.device
    if args.resume:
        cfg.resume_path = args.resume
    if args.data_examples:
        cfg.data_root = Path("data_examples")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{cfg.model_type}_{timestamp()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )

    device = torch.device(cfg.device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = Trainer(cfg, device)
    trainer.fit()


if __name__ == "__main__":
    main()
