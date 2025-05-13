# Penny Vision Project

A central index for all modules and supporting documents.

---

## Modules

* [Model Training](model_training.md)
  Train and validate CNN/embedding pipelines on the PC; output INT8/FP32 ONNX models.
* [Export Packaging](onnx_packaging.md)
  Convert, quantize, strip unsupported ops, and bundle ONNX model for deployment.
* [RPI Deployment](deploy_inference.md)
  Install runtime on Raspberry Pi, unpack model, and run live inference at ≥30 fps.
* [Data Extraction](roi_extraction.md)
  Capture raw obverse images; extract date, mint, and liberty ROIs; save processed data.
* [Monitoring Retraining](monitor_retrain.md)
  Log inference results, collect edge-case failures, and trigger the retraining pipeline.

---

## Supporting Documents

* [System Info](system_info.md)
  Hardware snapshot and core environment details (OS, CPU, memory, Python packages).
* [Coding Conventions](coding_conventions.md)
  Project-wide style rules: filenames, headers, docstrings, imports, logging, paths, tests.

---

## Folder Structure

```
/README.md                    <- This master README
                    
/data/                        <- images
  /raw_data/		      <- Raw penny images
  /obverse/                   <- Obverse ROI images
  /reverse/                   <- Reverse ROI images
  /date/                      <- Date ROI images
  /mint_marks/                <- Mint mark ROI images
  /liberty/                   <- Liberty ROI images
/scripts/                     <- Utility scripts (renaming, JSON generation)
/src/                         <- Core training & deployment code
/models/                      <- Trained model artifacts
  /orientation_classifier/    <- Side/orientation detection, rotation, normalization
  /date_classifier/           <- Date recognition models
  /mint_classifier/           <- Mint mark models
/configs/                     <- YAML/JSON configuration files
/metadata/                    <- Label metadata and JSON schemas
/output/                      <- Logs, visualizations, results
```

---

## How to Use

1. Click a module link to open its standalone document.
2. Follow that module’s instructions and produce the defined outputs.
3. Refer to **Supporting Documents** for environment setup and coding standards.
