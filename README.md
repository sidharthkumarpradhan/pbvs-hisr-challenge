# PBVS 2026 — Hyperspectral Image Super-Resolution (HISR) Challenge

> **Goal:** Rank #1 on the [PBVS 2026 HISR Challenge](https://www.codabench.org/competitions/12418/) leaderboard.

## Overview

This repository contains a fully automated, end-to-end pipeline for the **PBVS 2026 Hyperspectral Image Super-Resolution (HISR)** challenge:

- **Task:** ×4 spatial upscaling of hyperspectral images (224 spectral bands, SWIR 900–1700 nm)
- **Metrics:** PSNR ↑ (spatial quality) + SAM ↓ (spectral fidelity)
- **Model:** HISRNet — 3D spectral convolutions + 20 EDSR residual blocks + SE channel attention + PixelShuffle ×4
- **Submission format:** `submission.zip` containing `x4/HR_XX.h5` files

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/sidharthkumarpradhan/pbvs-hisr-challenge.git
cd pbvs-hisr-challenge

# 2. Set up the environment
bash setup_env.sh

# 3. Run the full pipeline (download → train → evaluate → submit)
python main.py
```

That's it. `python main.py` handles everything automatically.

## What `python main.py` does

| Step | Description |
|------|-------------|
| 1 | Downloads the ~8.6 GB dataset from Zenodo (resumable) |
| 2 | Splits into 55 train / 10 validation scenes |
| 3 | Trains HISRNet with cosine LR schedule, saves best checkpoint by PSNR |
| 4 | Evaluates on validation set using competition protocol (6 patches/scene) |
| 5 | Generates tiled predictions for all test scenes with 8-fold TTA |
| 6 | Packages `submission.zip` in the exact required format |

## Project Structure

```
pbvs-hisr-challenge/
├── main.py                  # Entry point — runs full pipeline
├── config.yaml              # All hyperparameters and paths
├── setup_env.sh             # Install dependencies
├── Dockerfile               # Reproducible GPU container
├── environment.yml          # Conda environment spec
└── src/
    ├── data/
    │   ├── dataset.py       # PyTorch Dataset classes
    │   └── downloader.py    # Zenodo download + train/val split
    ├── models/
    │   └── hsr_net.py       # HISRNet, Bicubic baseline, model registry
    ├── training/
    │   ├── trainer.py       # Training loop, checkpointing, validation
    │   └── losses.py        # L1 + SAM combined loss
    ├── evaluation/
    │   ├── evaluator.py     # Validation pipeline, patch extraction
    │   └── metrics.py       # PSNR, SAM, SSIM computation
    ├── inference/
    │   ├── predict.py       # Tiled inference with 8-fold TTA
    │   └── submit.py        # ZIP submission creation + verification
    └── utils/
        └── io.py            # HDF5 read/write, normalization helpers
```

## Model: HISRNet (7.77M parameters)

- **3D spectral convolutions** — captures correlations across all 224 bands simultaneously
- **20 EDSR residual blocks** with squeeze-and-excitation channel attention
- **PixelShuffle ×4 upsampling** (two ×2 stages)
- **Combined L1 + SAM loss** — optimizes both spatial quality and spectral fidelity
- **Test-time augmentation** — 8-fold TTA (flips) for better inference

## Tuning via `config.yaml`

Update `config.yaml` and retrain to chase rank #1:

```yaml
model:
  name: "HISRNet"      # or "Bicubic" for baseline
  num_features: 128    # increase for better quality (more VRAM)
  num_blocks: 20       # more blocks = better PSNR

training:
  sam_weight: 0.1      # increase to lower SAM (spectral angle)
  epochs: 300

inference:
  use_tta: true        # 8-fold test-time augmentation
```

## Docker

```bash
docker build -t hisr-challenge .
docker run --gpus all -v $(pwd)/data:/workspace/data hisr-challenge python main.py
```

## Requirements

- Python 3.10+
- PyTorch 2.1+ with CUDA 12.1
- GPU: A100 / V100 recommended (≥16 GB VRAM)

## Dataset

- **Training + Validation:** [Zenodo record 18171738](https://zenodo.org/records/18171738) (~8.6 GB, auto-downloaded)
- **Test set (LR only):** Download manually from the competition page and place in `data/test/`

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this codebase, please cite the PBVS 2026 HISR challenge and this repository.
