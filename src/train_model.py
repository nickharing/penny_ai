# train_model.py
# Purpose: Training loop for penny AI models using pre-loaded TrainingConfig
# Author: Nick Haring
# Date: 2025-05-17

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Using Hugging Face's transformers library for CLIP
from transformers import CLIPImageProcessor, CLIPVisionModel

from .dataset import PennyDataset
# Remove old problematic import if it's still present
# from .coin_clip_wrapper import CoinClipWrapper 


class CoinClipVitB32(nn.Module):
    """
    Thin wrapper around CLIP's ViT-B/32 vision encoder using Hugging Face transformers.
    """
    def __init__(self, device):
        """
        Initialize your Coin CLIP ViT-B/32 backbone model.
        Args:
            device: The torch device (e.g., 'cuda', 'cpu') to move the model to.
        """
        super().__init__()
        self.device = device

        # Processor for pixel normalization, if needed elsewhere or for reference
        # The actual preprocessing (resize, normalize) is handled by PennyDataset
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Vision model (encoder only)
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.to(device) # model should be moved to device

        # Expose feature dimension for head size
        self.output_dim = self.model.config.hidden_size # For ViT-B/32 this is 768

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values.to(self.device))
        return outputs.pooler_output


def run_training(cfg, device):
    """
    Run training using provided cfg and device.

    Args:
        cfg: TrainingConfig object with fields like data_root, metadata_json_path,
             roi_json_path, output_dir, epochs, batch_size, learning_rate, etc.
        device: torch.device('cpu') or torch.device('cuda')
    """
    # Ensure the cfg object uses the device specified for this run,
    # overriding any device set during config loading if they differ.
    current_run_device = device 
    cfg.device = current_run_device # Modifies cfg, ensure this is intended scope

    # Prepare output directories
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir = checkpoint_dir

    # Build data loaders
    ds_train = PennyDataset(cfg, model=cfg.model_type, subset='train')
    ds_val = PennyDataset(cfg, model=cfg.model_type, subset='val')
    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # Initialize model: backbone + head
    backbone = CoinClipVitB32(current_run_device) # Use the device for this run
    n_classes = len(ds_train.label_to_idx)
    head = nn.Linear(backbone.output_dim, n_classes)
    model = nn.Sequential(backbone, head).to(current_run_device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # Single-epoch training loop
    model.train()
    for xb, yb in dl_train:
        xb, yb = xb.to(current_run_device), yb.to(current_run_device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = nn.CrossEntropyLoss()(preds, yb)
        loss.backward()
        optimizer.step()

    # Save PyTorch checkpoint
    ckpt_path = checkpoint_dir / f"{cfg.model_type}_fp32.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Export ONNX
    dummy = torch.randn(1, 3, cfg.image_size[0], cfg.image_size[1], device=current_run_device)
    onnx_path = onnx_dir / f"{cfg.model_type}_fp32.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=16
    )


if __name__ == '__main__':
    from .config_utils import load_config
    # The 'side' model configuration is loaded. cfg.device will be set by load_config.
    config = load_config(model='side') 
    # run_training will use the device specified in the config object.
    run_training(cfg=config, device=config.device)