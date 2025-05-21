# Model Training Module (`src/train_model.py`)

## Overview

The Model Training module is responsible for training various classification models required for the Penny AI project. It uses a unified script, `src/train_model.py`, to train models for identifying penny side, orientation, date, and mint mark. The script leverages a master YAML configuration file for most settings, with minimal command-line overrides.

## Prerequisites

* Python 3.11+ environment.
* All dependencies listed in `requirements.txt` installed (e.g., `pip install -r requirements.txt`). This includes PyTorch, ONNX, ONNXRuntime, scikit-learn, PyYAML, and the custom `coin-clip` library.
* The project dataset (`data/` or `data_examples/` for testing).
* Metadata files (`metadata/metadata.json`, `metadata/roi_coordinates.json`).
* The master configuration file (`configs/training_config.yaml`).

## Configuration

The primary configuration for training is managed through a master YAML file, typically `configs/training_config.yaml`. This file contains sections for each model type (`side`, `orientation`, `date`, `mint`), allowing for tailored hyperparameters and settings.

The `src/config_utils.py` script contains the `TrainingConfig` dataclass and `load_config` function, which parses the specified section of this master YAML file. The path to this master configuration file is hardcoded in `src/train_model.py` as `DEFAULT_CFG_PATH = Path("configs/training_config.yaml")` but can be overridden via the `--config_file` CLI argument if needed for special cases.

### Key Configuration Parameters (within each model type section in YAML):

* **Data Paths**:
    * `data_root`: Path to the image dataset (e.g., "data/" or "data_examples/").
    * `metadata_json_path`: Path to `metadata.json`.
    * `roi_json_path`: Path to `roi_coordinates.json`.
* **Model Architecture**:
    * `model_type`: Specifies the model being trained (e.g., "side", "date"). Must match the section key.
    * `backbone`: Backbone architecture (e.g., "coin_clip_vit_b32").
    * `img_size`: Input image dimensions (e.g., `[224, 224]`).
* **Training Hyperparameters**:
    * `num_epochs`: Total training epochs.
    * `batch_size`: Batch size for dataloaders.
    * `lr`: Initial learning rate for AdamW.
    * `weight_decay`: Weight decay for AdamW.
    * `optimizer_betas`: Betas for AdamW.
    * `lr_scheduler_eta_min`: Minimum learning rate for CosineAnnealingLR.
    * `lr_warmup_epochs`: Number of linear warmup epochs.
* **Output & Export**:
    * `output_dir`: Directory to save final ONNX models (e.g., "models/side").
    * `quantize`: Boolean flag to enable INT8 PTQ ONNX export.
    * `qat`: Boolean flag (stub for future Quantization Aware Training).
* **Miscellaneous**:
    * `seed`: Random seed for reproducibility.
    * `device`: Preferred training device ("cuda", "cpu", null for auto-detect).
    * `num_workers`: Number of workers for DataLoader.

## Usage (Command-Line Interface)

The training script `src/train_model.py` is run from the command line.

**Synopsis:**

```bash
python -m src.train_model -m <model_key> [OPTIONS]
Required Argument:-m <model_key>, --model_key <model_key>Specifies the type of model to train.Choices: side, orientation, date, mint.This key must correspond to a top-level section in the training_config.yaml file.Optional Arguments:--epochs <number>Overrides the num_epochs value specified in the YAML configuration.--resume <path_to_checkpoint.pt>Path to a checkpoint file (.pt) to resume training from.Restores model weights, optimizer state, scheduler state, epoch number, and best validation metric.--device <device_name>Overrides the training device specified in the YAML or auto-detected.Examples: cpu, cuda, cuda:0, cuda:1.--config_file <path_to_yaml>Overrides the default path to the master configuration YAML file (configs/training_config.yaml).--data_examplesA flag that, if present, forces data_root to be "data_examples/". Useful for quick tests and CI.--batch_size <number>Overrides the batch_size from the YAML configuration. Useful for small test runs or memory-constrained environments.Model Types and SpecificsThe script supports training the following models, selected via the -m or --model_key flag:side:Task: Classifies the penny side into one of 5 categories.Classes (5): obverse, memorial, shield, wheat, bicentennial.Data Input: Full cropped penny image.Label Derivation: If metadata.side == "obverse", label is "obverse". If metadata.side == "reverse", label is derived from metadata.type.orientation:Task: Classifies the orientation of an obverse penny image.Classes (120): 3-degree bins covering 0-359 degrees (label = int(angle/3)).Data Input: Full cropped penny image (obverse side only).Note: RandomHorizontalFlip augmentation is disabled for this model.date:Task: Classifies the year on an obverse penny.Classes (116): Years from 1909 to 2024 inclusive (label = year - 1909).Data Input: ROI crop of the date region, extracted on-the-fly with 5% padding.mint:Task: Classifies the mint mark on an obverse penny.Classes (4): D, S, P, nomint.Data Input: ROI crop of the mint mark region, extracted on-the-fly with 5% padding.Output FilesThe training script generates several output files:ONNX Models: Saved in the directory specified by output_dir in the YAML config (e.g., models/<model_type>/).{model_type}_{YYYYMMDD-HHMMSS}_fp32.onnx: Full precision (FP32) ONNX model. Contains embedded label map metadata.{model_type}_{YYYYMMDD-HHMMSS}_int8.onnx: Quantized (INT8) ONNX model using Post-Training Quantization (PTQ), if quantize: true.Checkpoints: Saved in output/checkpoints/.{model_type}_last.pt: Checkpoint from the last completed epoch.{model_type}_best.pt: Checkpoint corresponding to the epoch with the best validation macro F1-score.Checkpoints include model state, optimizer state, scheduler state, epoch number, best validation metric, the configuration used, and the label map.Training Logs: Saved in output/training_logs/.{model_type}_{YYYYMMDD-HHMMSS}.log: Detailed log file for the training run, containing console output.Label Maps: Saved in output/label_maps/.{model_type}_label_map.json: A JSON file mapping the integer labels used by the model to their human-readable string representations (e.g., {"0": "obverse", "1": "memorial", ...}).Example Sanity CommandTo run a quick test of the side model for one epoch on the CPU using the data_examples/ dataset:python -m src.train_model -m side --epochs 1 --batch_size 8 --device cpu --data_examples
Expected console output snippet:Loaded PennyDataset: >0 train / >0 val / >0 test
Model head: Linear(in_features=512, out_features=5)
Epoch 1/1 Summary: Train Loss: ..., Val Loss: ..., Val Acc: ..., Val F1 (Macro): ..., Duration: ...s
Exported: models/side/side_YYYYMMDD-HHMMSS_fp32.onnx
Exported: models/side/side_YYYYMMDD-HHMMSS_int8.onnx
(Note: >0 indicates that some samples should be loaded, actual numbers depend on data_examples/ content and UID hashing. The ONNX paths will be relative to the project root, and the directory models/side is from the YAML output_dir.)IMX500 CompatibilityThe script includes a placeholder function make_imx500_compatible(model) which is called before ONNX export. This function is intended to house future modifications to the PyTorch model (e.g., replacing LayerNorm) to ensure the exported ONNX graph