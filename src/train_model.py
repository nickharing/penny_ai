# src/train_model.py
# Main script for training penny classification models.

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage # For ONNX INT8 quantization
from sklearn.metrics import accuracy_score, f1_score

# Attempt to import coin_clip, provide a clear error if not found
try:
    from coin_clip.models import clip_vit_b32
except ImportError:
    print("ERROR: coin_clip library not found or clip_vit_b32 not available.")
    print("Please ensure coin_clip is installed correctly, e.g., from the specified git repository:")
    print("  pip install git+https://github.com/nickharing/coin_clip.git")
    sys.exit(1)

# ONNX related imports
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType, CalibrationDataReader

# Local application/library specific imports
from config_utils import TrainingConfig, load_config, DEFAULT_CFG_PATH
from dataset import PennyDataset, get_transforms

# --- Global Variables & Constants ---
LOGGER = logging.getLogger(__name__) # Main logger for this script

# --- Model Definition & Registry ---
BACKBONE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_backbone(name: str) -> Callable:
    """Decorator to register a backbone model constructor."""
    def decorator(func: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        BACKBONE_REGISTRY[name] = func
        return func
    return decorator

@register_backbone("coin_clip_vit_b32")
def get_coin_clip_vit_b32_backbone(pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Loads the Coin-CLIP ViT-B/32 model.
    The actual coin_clip.models.clip_vit_b32 might not take a 'pretrained' flag
    in the same way as torchvision models. It's assumed to be pre-trained.
    We might need to adapt this if the actual API is different.
    For now, we assume it returns the feature extractor part.
    """
    # Assuming clip_vit_b32() returns a model that can be used as a backbone
    # and its output features are 512-dimensional as per spec.
    # If it returns (image_encoder, text_encoder, logit_scale), we need image_encoder.
    # This part needs verification against the actual coin_clip API.
    # For now, let's assume a simplified direct load.
    model = clip_vit_b32(**kwargs) # Pass any relevant kwargs
    # If 'model' is a tuple (image_encoder, text_encoder, ...), take the image_encoder
    if isinstance(model, tuple):
        image_encoder = model[0]
        # We need to ensure this image_encoder is a nn.Module that outputs features
        # and doesn't include the final projection for CLIP's contrastive loss if we
        # only want the visual backbone features.
        # This is a placeholder, actual adaptation might be needed.
        # For ViT-B/32, the output before projection is usually 768, but spec says 512.
        # This implies coin_clip_vit_b32 might already give a 512-dim embedding.
        return image_encoder
    return model


class PennyClassifier(nn.Module):
    """
    Combines a backbone with a classification head.
    """
    def __init__(self, backbone_name: str, num_classes: int, backbone_kwargs: Optional[Dict] = None):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}
        
        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(BACKBONE_REGISTRY.keys())}")
        
        self.backbone = BACKBONE_REGISTRY[backbone_name](**backbone_kwargs)
        
        # Determine backbone output features. Spec says 512 for coin_clip_vit_b32.
        # This might need to be dynamic if other backbones are added.
        # For now, hardcoding based on spec for the initial backbone.
        backbone_out_features = 512
        if backbone_name != "coin_clip_vit_b32":
            LOGGER.warning(f"Backbone output features assumed to be {backbone_out_features}. "
                           f"Verify for backbone '{backbone_name}'.")

        # Classification head (single linear layer as per spec)
        # TODO: Keep placeholder to swap for 2-layer MLP.
        self.head = nn.Linear(backbone_out_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # The output of ViT-B/32 from CLIP is typically the [CLS] token embedding.
        # Ensure `features` is the correct tensor [batch_size, feature_dim].
        # If `self.backbone` is the full CLIP image encoder, it might already give this.
        # If `features` has an extra dimension (e.g. sequence length for transformers),
        # it might need specific handling (e.g., taking the first token's features for CLS).
        # For ViT, output is often [batch, num_tokens, embed_dim].
        # If coin_clip_vit_b32 returns [batch, embed_dim] directly (e.g. pooled output or CLS token), this is fine.
        # Otherwise, if it's [batch, num_patches + 1, embed_dim], we might need features[:, 0] for CLS token.
        # Assuming coin_clip_vit_b32 output is directly [batch, 512]
        return self.head(features)

# --- IMX500 Compatibility ---
def make_imx500_compatible(model: torch.nn.Module) -> torch.nn.Module:
    """
    Placeholder function to replace or fuse unsupported ONNX ops for IMX500.
    TODO: Complete when operator list finalised (see onnx_packaging.md).
    For now, it's a no-op.
    """
    LOGGER.info("Applying IMX500 compatibility (currently a placeholder).")
    # Example (conceptual):
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.LayerNorm):
    #         # Replace LayerNorm with InstanceNorm or other compatible sequence
    #         LOGGER.info(f"Replacing LayerNorm '{name}' (not actually implemented yet).")
    return model

# --- Logging Setup ---
def setup_logging(log_dir: Path, run_id: str, level: int = logging.INFO) -> None:
    """Configures logging to console and a rotating file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{run_id}.log"

    # Basic configuration
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout) # Log to console
        ]
    )

    # File handler with rotation (e.g., new file per run, or could be size/time based)
    # For simplicity, one log file per run_id.
    file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite for new run
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    
    # Add file handler to the root logger
    logging.getLogger().addHandler(file_handler)
    
    LOGGER.info(f"Logging initialized. Log file: {log_file_path}")

# --- Seeding ---
def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    LOGGER.info(f"Random seed set to {seed}")


# --- Training & Evaluation Functions ---
def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    scaler: Optional[GradScaler] = None, # For AMP
                    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, # For warmup
                    epoch: int = 0, # For warmup scheduler step
                    warmup_epochs: int = 0 # For warmup scheduler step
                    ) -> float:
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        if scaler: # AMP enabled
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # AMP disabled or CPU
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Step the LR scheduler for warmup phase (if applicable, per iteration)
        # This is typical for linear warmup that happens per step, not per epoch.
        # However, the spec implies a 1-epoch linear warmup, then CosineAnnealing.
        # If warmup is per-epoch, scheduler.step() is called after the epoch.
        # If warmup is per-step for the first epoch:
        if epoch < warmup_epochs and lr_scheduler and hasattr(lr_scheduler, 'is_warmup_scheduler'):
             lr_scheduler.step()


    avg_loss = total_loss / len(dataloader)
    # accuracy = accuracy_score(all_labels, all_preds) # Not strictly needed for train epoch log
    # LOGGER.debug(f"Train Epoch: Avg Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    return avg_loss


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, float]:
    """Evaluates the model on the given dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            if device.type == 'cuda' and torch.cuda.is_available(): # AMP for evaluation if desired
                 with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    # Use zero_division=0.0 for f1_score to handle cases with no true positives for a class
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0.0)
    
    return avg_loss, accuracy, f1

# --- Checkpoint & Model Saving/Loading ---
def save_checkpoint(state: Dict[str, Any], filepath: Path) -> None:
    """Saves model and training state to a file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    LOGGER.info(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: Path, model: nn.Module, optimizer: Optional[optim.Optimizer] = None,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
    """Loads model and training state from a file."""
    if not filepath.exists():
        LOGGER.error(f"Checkpoint file not found: {filepath}")
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage) # Load to CPU first
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    LOGGER.info(f"Checkpoint loaded from {filepath}")
    return checkpoint

def save_label_map(label_map: Dict[int, str], filepath: Path) -> None:
    """Saves the label map (int -> string) to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Convert keys to strings for JSON compatibility if they are integers
    string_key_label_map = {str(k): v for k, v in label_map.items()}
    with open(filepath, 'w') as f:
        json.dump(string_key_label_map, f, indent=2)
    LOGGER.info(f"Label map saved to {filepath}")

# --- ONNX Export ---
class OnnxPTQDataReader(CalibrationDataReader):
    """Calibration data reader for ONNX INT8 quantization."""
    def __init__(self, dataloader: DataLoader, device: torch.device, input_name: str):
        self.dataloader = dataloader
        self.device = device
        self.input_name = input_name
        self.data_iter = iter(dataloader)
        self.to_pil = ToPILImage()

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        try:
            images, _ = next(self.data_iter)
            # Preprocess images like in training/validation if needed,
            # but usually calibration uses already transformed tensors.
            # The input to the ONNX model is expected to be a tensor.
            # For quantization, it might expect specific input format (e.g. NHWC).
            # For now, assume images are already processed tensors.
            images_np = images.cpu().numpy()
            return {self.input_name: images_np}
        except StopIteration:
            return None # End of data

def export_onnx_fp32(model: nn.Module, config: TrainingConfig, timestamp: str, label_map: Dict[int, str]) -> Path:
    """Exports the model to FP32 ONNX format."""
    model.eval() # Ensure model is in eval mode
    
    # Apply IMX500 compatibility transformations (currently a stub)
    compatible_model = make_imx500_compatible(model)

    dummy_input_size = [1] + [3] + config.img_size # Batch_size=1, Channels=3, H, W
    dummy_input = torch.randn(dummy_input_size, device='cpu') # Export on CPU

    onnx_filename = f"{config.model_type}_{timestamp}_fp32.onnx"
    onnx_filepath = Path(config.output_dir) / onnx_filename
    onnx_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Embed label map into ONNX model metadata
    # Keys must be strings. Values also strings.
    string_label_map = {str(k): str(v) for k, v in label_map.items()}
    metadata_props = {"label_map": json.dumps(string_label_map)}

    torch.onnx.export(
        compatible_model.to('cpu'), # Move model to CPU for export
        dummy_input,
        str(onnx_filepath),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=15, # As per spec (15-17)
        # metadata_props=metadata_props # Not a direct arg for torch.onnx.export
                                       # Metadata needs to be added post-export
    )
    LOGGER.info(f"FP32 ONNX model exported to {onnx_filepath}")

    # Add metadata to the exported ONNX model
    onnx_model = onnx.load(str(onnx_filepath))
    for key, value in string_label_map.items():
         meta = onnx_model.metadata_props.add()
         meta.key = f"label_{key}" # e.g. label_0, label_1
         meta.value = value
    # Could also store the full JSON string if preferred:
    # meta_json = onnx_model.metadata_props.add()
    # meta_json.key = "label_map_json"
    # meta_json.value = json.dumps(string_label_map)

    onnx.save(onnx_model, str(onnx_filepath))
    LOGGER.info(f"Metadata (label map) added to FP32 ONNX model: {onnx_filepath}")
    
    # Verify the model
    onnx.checker.check_model(str(onnx_filepath))
    LOGGER.info(f"FP32 ONNX model checked successfully: {onnx_filepath}")
    return onnx_filepath


def export_onnx_int8(fp32_onnx_path: Path, config: TrainingConfig, timestamp: str,
                       calibration_dataloader: DataLoader, device: torch.device) -> Path:
    """Exports the model to INT8 ONNX format using Post-Training Quantization (Static)."""
    int8_filename = f"{config.model_type}_{timestamp}_int8.onnx"
    int8_filepath = Path(config.output_dir) / int8_filename
    int8_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Get input name from the FP32 model
    try:
        onnx_model = onnx.load(str(fp32_onnx_path))
        input_name = onnx_model.graph.input[0].name
    except Exception as e:
        LOGGER.error(f"Failed to get input name from ONNX model {fp32_onnx_path}: {e}")
        # Fallback if parsing fails, though this is crucial
        input_name = 'input' # Default if unable to parse

    LOGGER.info(f"Starting INT8 quantization. Input name: {input_name}")
    
    # Create calibration data reader
    # The dataloader should provide data in the format expected by the model input
    calibration_dr = OnnxPTQDataReader(calibration_dataloader, device, input_name)

    try:
        onnxruntime.quantization.quantize_static(
            model_input=fp32_onnx_path,
            model_output=int8_filepath,
            calibration_data_reader=calibration_dr,
            quant_format=onnxruntime.quantization.QuantFormat.QDQ, # QOperator or QDQ
            activation_type=QuantType.QInt8, # Or QUInt8
            weight_type=QuantType.QInt8,     # Or QUInt8
            # per_channel=False, # or True, depending on desired granularity
            # nodes_to_quantize=None, # Specify particular nodes if needed
            # nodes_to_exclude=None,  # Exclude nodes problematic for quantization
            # optimize_model=True # onnxruntime performs some optimizations
        )
        LOGGER.info(f"INT8 ONNX model (static quantization) exported to {int8_filepath}")
        
        # Verify the INT8 model
        onnx.checker.check_model(str(int8_filepath))
        LOGGER.info(f"INT8 ONNX model checked successfully: {int8_filepath}")

    except Exception as e:
        LOGGER.error(f"Error during INT8 static quantization: {e}")
        LOGGER.error("INT8 model export failed. Check calibration data and model compatibility.")
        # Fallback: Dynamic quantization (less accurate, no calibration needed)
        # LOGGER.info("Attempting dynamic quantization as a fallback...")
        # try:
        #     onnxruntime.quantization.quantize_dynamic(
        #         model_input=fp32_onnx_path,
        #         model_output=int8_filepath,
        #         weight_type=QuantType.QInt8 # Or QUInt8
        #     )
        #     LOGGER.info(f"INT8 ONNX model (dynamic quantization) exported to {int8_filepath}")
        # except Exception as e_dyn:
        #     LOGGER.error(f"Dynamic quantization also failed: {e_dyn}")
        #     return None # Indicate failure
        return None # Indicate failure if static quantization fails

    return int8_filepath

# --- Main Training Orchestration ---
def run_training(config: TrainingConfig, resume_path: Optional[Path] = None, cli_device: Optional[str] = None) -> None:
    """
    Main function to run the training and evaluation pipeline.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    setup_logging(config.log_dir, f"{config.model_type}_{run_id}")
    set_seed(config.seed)

    # Determine device
    if cli_device:
        device_str = cli_device
    elif config.device: # From YAML
        device_str = config.device
    else: # Auto-detect
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning(f"CUDA specified ('{device_str}') but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    LOGGER.info(f"Using device: {device}")

    # Create datasets and dataloaders
    LOGGER.info("Loading datasets...")
    train_transform = get_transforms(model_type=config.model_type, subset="train", img_size=tuple(config.img_size))
    val_test_transform = get_transforms(model_type=config.model_type, subset="val", img_size=tuple(config.img_size))

    data_root_path = Path(config.data_root) # Use full dataset by default
    # For unit tests, this path might be overridden to data_examples/
    # This logic should be handled by the test script itself when calling run_training or by CLI arg.
    # For now, assume config.data_root is correct for the run.

    train_dataset = PennyDataset(
        data_root=data_root_path, metadata_json_path=config.metadata_json_path, roi_json_path=config.roi_json_path,
        model_type=config.model_type, image_transform=train_transform, subset="train", seed=config.seed
    )
    val_dataset = PennyDataset(
        data_root=data_root_path, metadata_json_path=config.metadata_json_path, roi_json_path=config.roi_json_path,
        model_type=config.model_type, image_transform=val_test_transform, subset="val", seed=config.seed
    )
    test_dataset = PennyDataset( # Test dataset for final evaluation (optional during main training loop)
        data_root=data_root_path, metadata_json_path=config.metadata_json_path, roi_json_path=config.roi_json_path,
        model_type=config.model_type, image_transform=val_test_transform, subset="test", seed=config.seed
    )
    
    # Log dataset sizes
    LOGGER.info(f"Loaded PennyDataset: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test samples")
    if len(train_dataset) == 0:
        LOGGER.error("Training dataset is empty. Please check data paths and metadata.")
        return
    if len(val_dataset) == 0:
        LOGGER.warning("Validation dataset is empty. Model selection might be suboptimal.")


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    # Model
    LOGGER.info(f"Initializing model: {config.backbone} with {config.num_classes} classes for model_type '{config.model_type}'.")
    model = PennyClassifier(backbone_name=config.backbone, num_classes=config.num_classes)
    model.to(device)
    LOGGER.info(f"Model head: {model.head}")


    # Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=tuple(config.optimizer_betas), weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # LR Scheduler: 1-epoch linear warm-up then CosineAnnealingLR
    # The warmup part can be implemented manually or using a chained scheduler.
    # For simplicity, manual step or a LambdaLR for warmup phase.
    # CosineAnnealingLR T_max should be num_epochs - warmup_epochs.
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.lr_scheduler_T_max_epochs, eta_min=config.lr_scheduler_eta_min
    )

    warmup_scheduler = None
    if config.lr_warmup_epochs > 0:
        # Example of a simple linear warmup using LambdaLR
        # This warms up from a very small fraction of LR to the full LR over warmup_epochs
        warmup_factor = 1.0 / (config.lr_warmup_epochs * len(train_loader)) # if per step
        # If per epoch warmup:
        # warmup_lambda = lambda epoch: min(1.0, (epoch + 1) / config.lr_warmup_epochs) if epoch < config.lr_warmup_epochs else 1.0
        # For step-wise warmup, it's more complex with LambdaLR or needs manual handling.
        # Let's assume a simpler epoch-based warmup logic for now, or step inside epoch loop.
        # For now, CosineAnnealingLR starts after warmup.
        # The actual LR adjustment for warmup will be handled in the epoch loop or by a ChainedScheduler.
        # A simple approach: set initial LR low, then ramp up for first epoch(s).
        # For this implementation, we'll adjust LR manually for warmup phase in the loop,
        # and main_scheduler will take over after warmup.
        pass # Manual warmup logic will be in the training loop or use a proper warmup scheduler.


    # AMP GradScaler
    scaler = GradScaler() if device.type == 'cuda' and torch.cuda.is_available() else None
    if scaler:
        LOGGER.info("Using Automatic Mixed Precision (AMP).")

    # Checkpoint loading
    start_epoch = 0
    best_val_metric = -float('inf') # For F1-score or accuracy

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True) # Ensure checkpoint dir exists
    last_ckpt_path = config.checkpoint_dir / f"{config.model_type}_last.pt"
    best_ckpt_path = config.checkpoint_dir / f"{config.model_type}_best.pt"

    if resume_path:
        if resume_path.exists():
            LOGGER.info(f"Resuming training from checkpoint: {resume_path}")
            checkpoint = load_checkpoint(resume_path, model, optimizer, main_scheduler) # Pass main_scheduler
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_metric = checkpoint.get('best_metric', -float('inf'))
            # Ensure scaler state is loaded if it exists and scaler is active
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            LOGGER.info(f"Resumed from epoch {start_epoch-1}. Best val metric: {best_val_metric:.4f}")
        else:
            LOGGER.warning(f"Resume checkpoint not found: {resume_path}. Starting from scratch.")
    elif last_ckpt_path.exists(): # Optionally auto-resume from _last.pt if no --resume flag
        LOGGER.info(f"Found last checkpoint: {last_ckpt_path}. Resuming training.")
        checkpoint = load_checkpoint(last_ckpt_path, model, optimizer, main_scheduler)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_metric = checkpoint.get('best_metric', -float('inf'))
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        LOGGER.info(f"Resumed from epoch {start_epoch-1}. Best val metric: {best_val_metric:.4f}")


    # Training loop
    LOGGER.info(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start_time = time.time()
        
        # Manual LR warmup for the first N epochs if lr_warmup_epochs > 0
        current_lr = optimizer.param_groups[0]['lr']
        if epoch < config.lr_warmup_epochs:
            # Simple linear warmup from a small fraction to target LR
            # This is an epoch-wise warmup.
            # For a more standard linear warmup, you might start optimizer LR very low,
            # then increment it each step for the first epoch.
            # Here, we assume CosineAnnealing starts after warmup_epochs.
            # If using a ChainedScheduler, this would be handled automatically.
            # For this example, let's adjust LR directly if in warmup.
            # Target LR is config.lr. Start from, say, config.lr / 10.
            initial_warmup_lr = config.lr / 10.0
            if epoch == 0 and start_epoch == 0 : # Only set very low LR if truly starting warmup
                 for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_warmup_lr
            
            # Linearly increase LR towards config.lr
            if config.lr_warmup_epochs > 0:
                progress = (epoch + 1) / config.lr_warmup_epochs
                new_lr = initial_warmup_lr + (config.lr - initial_warmup_lr) * progress
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            current_lr = optimizer.param_groups[0]['lr']
            LOGGER.info(f"Epoch {epoch+1}/{config.num_epochs} (Warmup) - LR: {current_lr:.2e}")
        else:
            # After warmup, the main scheduler takes over.
            # main_scheduler.step() is typically called after optimizer.step() or at end of epoch.
            # For CosineAnnealing, usually end of epoch.
            LOGGER.info(f"Epoch {epoch+1}/{config.num_epochs} - LR: {current_lr:.2e}")


        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler,
                                     epoch=epoch, warmup_epochs=config.lr_warmup_epochs) # Pass epoch info for schedulers
        
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start_time
        LOGGER.info(f"Epoch {epoch+1}/{config.num_epochs} Summary: "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}, Val F1 (Macro): {val_f1:.4f}, "
                    f"Duration: {epoch_duration:.2f}s")

        # Step the main LR scheduler after warmup phase
        if epoch >= config.lr_warmup_epochs:
            main_scheduler.step()

        # Save checkpoint (_last.pt)
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': main_scheduler.state_dict(), # Save main scheduler state
            'best_metric': best_val_metric,
            'config': config.__dict__, # Save config for reference
            'label_map': train_dataset.current_label_map # Save the label map used
        }
        if scaler:
            checkpoint_state['scaler_state_dict'] = scaler.state_dict()
        save_checkpoint(checkpoint_state, last_ckpt_path)

        # Save best model based on validation F1 score
        if val_f1 > best_val_metric:
            best_val_metric = val_f1
            save_checkpoint(checkpoint_state, best_ckpt_path)
            LOGGER.info(f"New best model saved with Val F1: {best_val_metric:.4f} at epoch {epoch+1}")
        
        # TODO: Placeholder for early-stopping logic

    LOGGER.info("Training finished.")

    # Load best model for final export and testing
    LOGGER.info(f"Loading best model from {best_ckpt_path} for final export.")
    if best_ckpt_path.exists():
        load_checkpoint(best_ckpt_path, model) # Just load model weights for export
    else:
        LOGGER.warning("Best checkpoint not found. Using last model state for export.")
        # Model is already in its last state if best_ckpt_path doesn't exist (e.g. 1 epoch run)

    # Save label map
    label_map_path = config.label_map_dir / f"{config.model_type}_label_map.json"
    save_label_map(train_dataset.current_label_map, label_map_path)

    # Export to ONNX
    LOGGER.info("Exporting model to ONNX...")
    fp32_onnx_path = export_onnx_fp32(model, config, run_id, train_dataset.current_label_map)
    
    if config.quantize and fp32_onnx_path:
        # Use a subset of the training data (or validation data) for calibration
        # Ensure this dataloader doesn't have shuffle=True if order matters for calibration_dr
        # And uses val_test_transform (no augmentations)
        calibration_dataset = PennyDataset(
            data_root=data_root_path, metadata_json_path=config.metadata_json_path, roi_json_path=config.roi_json_path,
            model_type=config.model_type, image_transform=val_test_transform, subset="train", seed=config.seed
        )
        # Limit calibration data size (e.g., 100-200 samples)
        calib_subset_indices = random.sample(range(len(calibration_dataset)), min(len(calibration_dataset), 128)) # e.g. 128 samples
        calib_subset = torch.utils.data.Subset(calibration_dataset, calib_subset_indices)

        calibration_loader = DataLoader(calib_subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        if len(calibration_loader) == 0:
            LOGGER.warning("Calibration dataset for INT8 quantization is empty. Skipping INT8 export.")
        else:
            export_onnx_int8(fp32_onnx_path, config, run_id, calibration_loader, device) # Use CPU for calibration if model moved
    
    LOGGER.info("ONNX export process finished.")

    # Final evaluation on test set with the best model (optional)
    if len(test_loader.dataset) > 0:
        LOGGER.info("Evaluating best model on the test set...")
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
        LOGGER.info(f"Test Set Performance: Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1 (Macro): {test_f1:.4f}")
    else:
        LOGGER.info("Test dataset is empty. Skipping final test set evaluation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Penny Classification Models.")
    parser.add_argument("-m", "--model_key", type=str, required=True,
                        choices=["side", "orientation", "date", "mint"],
                        help="Model type/key to train (must match a key in training_config.yaml).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs from YAML.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume training from.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g., 'cpu', 'cuda', 'cuda:0'). Auto-detects if not set.")
    parser.add_argument("--config_file", type=str, default=str(DEFAULT_CFG_PATH),
                        help=f"Path to the master YAML configuration file. Default: {DEFAULT_CFG_PATH}")
    # For testing on data_examples
    parser.add_argument("--data_examples", action="store_true",
                        help="If set, uses 'data_examples/' as data_root, overriding YAML.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from YAML (useful for small test runs).")


    args = parser.parse_args()

    try:
        # Load configuration for the specified model_key
        cfg = load_config(model_config_key=args.model_key, config_path=Path(args.config_file))

        # Override config with CLI arguments if provided
        if args.epochs is not None:
            cfg.num_epochs = args.epochs
        if args.batch_size is not None: # Allow overriding batch_size for sanity command
            cfg.batch_size = args.batch_size
        
        # Handle data_root override for unit tests/CI
        if args.data_examples:
            cfg.data_root = "data_examples/"
            # Potentially reduce num_workers for smaller dataset if on CI
            # cfg.num_workers = 0 # Or a smaller number

        resume_checkpoint_path = Path(args.resume) if args.resume else None
        
        run_training(config=cfg, resume_path=resume_checkpoint_path, cli_device=args.device)

    except FileNotFoundError as e:
        LOGGER.error(f"Configuration or data file not found: {e}")
        sys.exit(1)
    except ValueError as e:
        LOGGER.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e: # Catch any other exceptions
        LOGGER.error(f"An unexpected error occurred: {e}", exc_info=True) # Log traceback
        sys.exit(1)
