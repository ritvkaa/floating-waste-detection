"""Run single-image floating-litter detection and save image plus JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from inference import load_model, predict_image


def main() -> None:
    project_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input water-body image")
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_dir / "weights" / "floating_litter_yolov8n.pt",
    )
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nms-iou", type=float)
    parser.add_argument("--output", type=Path, default=project_dir / "outputs")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input image does not exist: {args.image}")
    config_path = project_dir / "outputs" / "deployment_config.json"
    deployment = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    confidence = args.confidence if args.confidence is not None else deployment.get("confidence", 0.25)
    nms_iou = args.nms_iou if args.nms_iou is not None else deployment.get("nms_iou", 0.7)
    image_size = args.image_size if args.image_size is not None else deployment.get("image_size", 640)
    model = load_model(args.weights)
    with Image.open(args.image) as source:
        annotated, detections = predict_image(
            model, source, confidence, image_size, args.device, nms_iou
        )
    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.output / f"{args.image.stem}_detected.jpg"
    json_path = args.output / f"{args.image.stem}_detections.json"
    annotated.save(image_path, quality=92)
    payload = {
        "input": str(args.image.resolve()),
        "count": len(detections),
        "confidence_threshold": confidence,
        "nms_iou_threshold": nms_iou,
        "image_size": image_size,
        "detections": detections,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(detections), "image": str(image_path), "json": str(json_path)}))


if __name__ == "__main__":
    main()
