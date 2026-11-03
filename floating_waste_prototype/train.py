"""Fine-tune a small YOLOv8 baseline on the prepared TUD-GV pilot split."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def main() -> None:
    project_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "data" / "tud_gv_pilot" / "dataset.yaml",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--name", default="tud_gv_pilot")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset config not found: {args.data}. Run prepare_dataset.py first.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.batch < 1 or args.image_size < 32:
        parser.error("--batch must be positive and --image-size at least 32.")

    started = time.perf_counter()
    pretrained = project_dir / "weights" / "yolov8n.pt"
    model = YOLO(str(pretrained if pretrained.is_file() else Path("yolov8n.pt")))
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=0,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        cache=False,
        plots=True,
        project=str(project_dir / "runs"),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed without expected checkpoint: {best}")
    weights_dir = project_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    final_weights = weights_dir / f"floating_litter_yolov8n_{args.name}.pt"
    shutil.copy2(best, final_weights)
    manifest_path = args.data.parent / "manifest.json"
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch,
        "dataset": dataset_manifest,
        "run_directory": str(run_dir),
        "weights": str(final_weights),
        "training_wall_seconds": round(time.perf_counter() - started, 2),
    }
    (project_dir / f"training_record_{args.name}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
