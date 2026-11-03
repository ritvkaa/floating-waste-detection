"""Evaluate one prepared test split and report detection and count metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO

from inference import predict_image


def box_iou(first: list[float], second: list[float]) -> float:
    x_left = max(first[0], second[0])
    y_top = max(first[1], second[1])
    x_right = min(first[2], second[2])
    y_bottom = min(first[3], second[3])
    intersection = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def read_ground_truth(label_path: Path, image_size: tuple[int, int]) -> list[list[float]]:
    width, height = image_size
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid label row in {label_path}: {line}")
        _, x_center, y_center, box_width, box_height = map(float, fields)
        boxes.append(
            [
                (x_center - box_width / 2) * width,
                (y_center - box_height / 2) * height,
                (x_center + box_width / 2) * width,
                (y_center + box_height / 2) * height,
            ]
        )
    return boxes


def match_boxes(
    predicted: list[dict[str, Any]], truth: list[list[float]], iou_threshold: float
) -> tuple[int, int, int]:
    matched: set[int] = set()
    true_positives = false_positives = 0
    for detection in sorted(predicted, key=lambda item: item["confidence"], reverse=True):
        candidates = [
            (box_iou(detection["xyxy"], box), index)
            for index, box in enumerate(truth)
            if index not in matched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            true_positives += 1
            matched.add(best_index)
        else:
            false_positives += 1
    return true_positives, false_positives, len(truth) - len(matched)


def aggregate_metrics(
    predictions: dict[str, list[dict[str, Any]]],
    truths: dict[str, list[list[float]]],
    confidence: float,
    iou_threshold: float,
) -> dict[str, float | int]:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    count_errors = []
    for image_name, detections in predictions.items():
        retained = [
            item for item in detections if item["confidence"] >= confidence
        ]
        tp, fp, fn = match_boxes(retained, truths[image_name], iou_threshold)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        count_errors.append(abs(len(retained) - len(truths[image_name])))
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_mae": float(np.mean(count_errors)) if count_errors else 0.0,
    }


def make_contact_sheet(entries: list[tuple[Path, int, int]], target: Path) -> None:
    if not entries:
        return
    tile_width, tile_height, label_height = 360, 260, 42
    columns = min(3, len(entries))
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image_path, predicted, actual) in enumerate(entries):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((tile_width - 12, tile_height - 12))
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        sheet.paste(image, (x + (tile_width - image.width) // 2, y + 6))
        draw.text(
            (x + 8, y + tile_height + 2),
            f"predicted: {predicted} | annotated: {actual}",
            fill="#182B3A",
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=90)


def main() -> None:
    project_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "data" / "tud_gv_pilot" / "dataset.yaml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_dir / "weights" / "floating_litter_yolov8n.pt",
    )
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-name", default="pilot")
    args = parser.parse_args()
    if not args.data.is_file() or not args.weights.is_file():
        parser.error("Dataset config and trained weights must both exist.")
    if (args.confidence is not None and not 0 <= args.confidence <= 1) or not 0 < args.iou <= 1:
        parser.error("Confidence must be in [0, 1] and IoU must be in (0, 1].")

    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    data_root = Path(config["path"])
    test_images = sorted((data_root / config["test"]).glob("*"))
    test_images = [path for path in test_images if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not test_images:
        parser.error(f"No test images found in {data_root / config['test']}")
    val_images = sorted((data_root / config["val"]).glob("*"))
    val_images = [path for path in val_images if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not val_images:
        parser.error(f"No validation images found in {data_root / config['val']}")
    model = YOLO(str(args.weights))
    ap_metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=0,
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=False,
    )
    val_truths: dict[str, list[list[float]]] = {}
    val_predictions_by_nms: dict[float, dict[str, list[dict[str, Any]]]] = {
        threshold: {}
        for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75)
    }
    for image_path in val_images:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            val_truths[image_path.name] = read_ground_truth(
                data_root / "labels" / "val" / f"{image_path.stem}.txt", image.size
            )
            for nms_iou, predictions in val_predictions_by_nms.items():
                _, detections = predict_image(
                    model,
                    image,
                    confidence=0.001,
                    image_size=args.image_size,
                    device=args.device,
                    nms_iou=nms_iou,
                )
                predictions[image_path.name] = detections
    confidence_candidates = tuple(
        [*np.arange(0.05, 0.951, 0.05).round(2), 0.975, 0.99, 1.0]
    )
    operating_points = []
    for nms_iou, predictions in val_predictions_by_nms.items():
        for confidence in confidence_candidates:
            scores = aggregate_metrics(predictions, val_truths, float(confidence), args.iou)
            operating_points.append(
                {
                    "confidence": float(confidence),
                    "nms_iou": nms_iou,
                    **scores,
                }
            )
    selected_point = max(
        operating_points,
        key=lambda item: (
            item["f1"],
            item["precision"],
            -item["count_mae"],
            item["confidence"],
            item["nms_iou"],
        ),
    )
    confidence = (
        args.confidence if args.confidence is not None else selected_point["confidence"]
    )
    nms_iou = selected_point["nms_iou"]
    predictions: dict[str, list[dict[str, Any]]] = {}
    output_root = project_dir / "outputs" / args.output_name
    output_dir = output_root / "test_annotated"
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_entries: list[tuple[Path, int, int]] = []
    total_inference_seconds = 0.0
    with Image.open(test_images[0]) as warmup_image:
        model.predict(
            warmup_image.convert("RGB"),
            conf=confidence,
            iou=nms_iou,
            imgsz=args.image_size,
            device=args.device,
            verbose=False,
        )
    for index, image_path in enumerate(test_images, start=1):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            ground_truth = read_ground_truth(
                data_root / "labels" / "test" / f"{image_path.stem}.txt", image.size
            )
            started = time.perf_counter()
            annotated, detections = predict_image(
                model,
                image,
                confidence=confidence,
                image_size=args.image_size,
                device=args.device,
                nms_iou=nms_iou,
            )
            total_inference_seconds += time.perf_counter() - started
            predictions[image_path.name] = detections
            saved_path = output_dir / image_path.name
            annotated.save(saved_path, quality=90)
            if len(contact_entries) < 6:
                contact_entries.append((saved_path, len(detections), len(ground_truth)))
        if index % 10 == 0 or index == len(test_images):
            print(f"Evaluated {index}/{len(test_images)} test images", flush=True)

    truths: dict[str, list[list[float]]] = {}
    for image_path in test_images:
        with Image.open(image_path) as source:
            truths[image_path.name] = read_ground_truth(
                data_root / "labels" / "test" / f"{image_path.stem}.txt",
                source.size,
            )
    test_metrics = aggregate_metrics(predictions, truths, confidence, args.iou)
    result = {
        "weights": str(args.weights.resolve()),
        "dataset": config.get("path"),
        "test_images": len(test_images),
        "confidence_threshold": confidence,
        "nms_iou_threshold": nms_iou,
        "iou_match_threshold": args.iou,
        "validation_selected_f1": selected_point["f1"],
        "validation_selected_precision": selected_point["precision"],
        "validation_selected_recall": selected_point["recall"],
        **test_metrics,
        "map50": float(ap_metrics.box.map50),
        "map50_95": float(ap_metrics.box.map),
        "mean_predict_annotate_seconds_per_image": total_inference_seconds / len(test_images),
        "metric_scope": (
            "confidence and NMS selected on validation by F1; test metrics are held out; "
            "fixed-threshold, one-to-one box matching at IoU 0.50; timing excludes image decoding and model startup"
        ),
    }
    output_json = output_root / "metrics.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_json.parent / "deployment_config.json").write_text(
        json.dumps(
            {"confidence": confidence, "nms_iou": nms_iou, "image_size": args.image_size},
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_json.parent / "pilot_predictions.json").write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )
    make_contact_sheet(contact_entries, output_json.parent / "examples.jpg")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
