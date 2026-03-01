#!/usr/bin/env python3
"""PBVS 2026 - Hyperspectral Image Super-Resolution (HISR) Challenge.

Automated pipeline:
    python main.py                    # Full pipeline (download → train → evaluate → predict → submit)
    python main.py --config myconfig.yaml
    python main.py --skip-download    # If data is already present
    python main.py --skip-train       # Use existing checkpoint for prediction
    python main.py --eval-only        # Only run validation evaluation
    python main.py --predict-only     # Only generate predictions and submission
    python main.py --verify-only      # Only verify existing submission ZIP
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Configuration loaded from {config_path}")
    return cfg


def get_best_checkpoint(checkpoint_dir: str = "checkpoints") -> Path | None:
    ckpt_dir = Path(checkpoint_dir)
    best = ckpt_dir / "best_model.pth"
    last = ckpt_dir / "last_model.pth"
    if best.exists():
        return best
    if last.exists():
        return last
    # Look for any .pth file
    pths = sorted(ckpt_dir.glob("*.pth"))
    return pths[-1] if pths else None


def step_download(cfg: dict, args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Dataset Download")
    print("=" * 60)
    from src.data.downloader import check_or_download_dataset, verify_test_data
    check_or_download_dataset(cfg)
    verify_test_data(Path(cfg["data"]["test_dir"]))


def step_train(cfg: dict, args: argparse.Namespace) -> Path:
    print("\n" + "=" * 60)
    print("STEP 2: Model Training")
    print("=" * 60)
    from src.training.trainer import train
    best_ckpt = train(cfg, checkpoint_dir="checkpoints")
    print(f"Training complete. Best checkpoint: {best_ckpt}")
    return best_ckpt


def step_evaluate(cfg: dict, checkpoint_path: Path) -> dict:
    print("\n" + "=" * 60)
    print("STEP 3: Evaluation on Validation Set")
    print("=" * 60)
    import torch
    from src.evaluation.evaluator import evaluate_model, load_model_from_checkpoint

    if not checkpoint_path or not checkpoint_path.exists():
        print("No checkpoint found, skipping evaluation.")
        return {}

    device = _get_device()
    model = load_model_from_checkpoint(str(checkpoint_path), cfg, device)
    metrics = evaluate_model(model,
                             cfg["data"]["val_dir"],
                             device,
                             scale=cfg["data"]["scale"])

    print(f"\nValidation Results:")
    print(
        f"  PSNR: {metrics.get('psnr_mean', 0):.4f} ± {metrics.get('psnr_std', 0):.4f} dB"
    )
    print(
        f"  SAM:  {metrics.get('sam_mean', 0):.6f} ± {metrics.get('sam_std', 0):.6f}°"
    )

    results_path = Path("checkpoints/validation_results.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {results_path}")
    return metrics


def step_predict(cfg: dict, checkpoint_path: Path) -> list[Path]:
    print("\n" + "=" * 60)
    print("STEP 4: Generate Test Predictions")
    print("=" * 60)
    import torch
    from src.evaluation.evaluator import load_model_from_checkpoint
    from src.inference.predict import predict_all_test_scenes

    test_dir = Path(cfg["data"]["test_dir"])
    if not list(test_dir.glob("*.h5")):
        print(f"No test files found in {test_dir}.")
        print(
            "Download the test set from the competition page and place .h5 files there."
        )
        return []

    device = _get_device()
    model = load_model_from_checkpoint(str(checkpoint_path), cfg, device)

    infer_cfg = cfg.get("inference", {})
    output_paths = predict_all_test_scenes(
        model=model,
        test_dir=test_dir,
        output_dir=cfg["submission"]["output_dir"],
        device=device,
        scale=cfg["data"]["scale"],
        tile_size=infer_cfg.get("tile_size", 200),
        overlap=infer_cfg.get("overlap", 32),
        use_tta=infer_cfg.get("use_tta", True),
    )
    return output_paths


def step_submit(cfg: dict) -> Path | None:
    print("\n" + "=" * 60)
    print("STEP 5: Package Submission")
    print("=" * 60)
    from src.inference.submit import create_submission_zip, verify_submission

    pred_dir = Path(cfg["submission"]["output_dir"])
    if not list(pred_dir.glob("HR_*.h5")):
        print(
            f"No prediction files found in {pred_dir}. Run prediction step first."
        )
        return None

    zip_path = Path(cfg["submission"]["zip_name"])
    create_submission_zip(pred_dir, zip_path, scale=cfg["data"]["scale"])
    verify_submission(zip_path, scale=cfg["data"]["scale"])
    print(f"\nREADY TO SUBMIT: {zip_path.absolute()}")
    return zip_path


def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║   PBVS 2026 - Hyperspectral Image Super-Resolution (HISR)   ║
║   Automated Training & Submission Pipeline                   ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="HISR Challenge Pipeline")
    parser.add_argument("--config",
                        default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--skip-download",
                        action="store_true",
                        help="Skip dataset download")
    parser.add_argument("--skip-train",
                        action="store_true",
                        help="Skip training, use existing checkpoint")
    parser.add_argument("--eval-only",
                        action="store_true",
                        help="Only run validation evaluation")
    parser.add_argument("--predict-only",
                        action="store_true",
                        help="Only generate predictions")
    parser.add_argument("--verify-only",
                        action="store_true",
                        help="Only verify submission ZIP")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=None,
                        help="Explicit checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t_start = time.time()

    # Ensure directories exist
    for key in ["data_dir", "train_dir", "val_dir", "test_dir"]:
        Path(cfg["data"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["submission"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # --- VERIFY ONLY ---
    if args.verify_only:
        from src.inference.submit import verify_submission
        zip_path = Path(cfg["submission"]["zip_name"])
        verify_submission(zip_path, scale=cfg["data"]["scale"])
        return

    # --- PREDICT ONLY ---
    if args.predict_only:
        ckpt = Path(
            args.checkpoint) if args.checkpoint else get_best_checkpoint()
        if not ckpt:
            print("No checkpoint found. Train the model first.")
            sys.exit(1)
        step_predict(cfg, ckpt)
        step_submit(cfg)
        return

    # --- EVAL ONLY ---
    if args.eval_only:
        ckpt = Path(
            args.checkpoint) if args.checkpoint else get_best_checkpoint()
        if not ckpt:
            print("No checkpoint found. Train the model first.")
            sys.exit(1)
        step_evaluate(cfg, ckpt)
        return

    # --- FULL PIPELINE ---

    # Step 1: Download data
    if not args.skip_download:
        step_download(cfg, args)
    else:
        print("\nStep 1: Skipping download (--skip-download)")

    # Step 2: Train
    if not args.skip_train:
        checkpoint_path = step_train(cfg, args)
    else:
        checkpoint_path = Path(
            args.checkpoint) if args.checkpoint else get_best_checkpoint()
        if not checkpoint_path:
            print("No checkpoint found and --skip-train specified. Aborting.")
            sys.exit(1)
        print(f"\nStep 2: Using existing checkpoint: {checkpoint_path}")

    # Step 3: Evaluate
    metrics = step_evaluate(cfg, checkpoint_path)

    # Step 4: Predict
    output_paths = step_predict(cfg, checkpoint_path)

    # Step 5: Package submission
    if output_paths:
        step_submit(cfg)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed / 60:.1f} minutes.")
    if metrics:
        print(f"Final Val PSNR: {metrics.get('psnr_mean', 0):.4f} dB")
        print(f"Final Val SAM:  {metrics.get('sam_mean', 0):.6f}°")
    print("=" * 60)


if __name__ == "__main__":
    main()
