"""Shared single-class prediction helpers for the CLI and web demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_model(weights: str | Path) -> YOLO:
    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. "
            "Run train.py first or pass --weights to an available YOLOv8 checkpoint."
        )
    return YOLO(str(weights_path))


def predict_image(
    model: YOLO,
    image: Image.Image,
    confidence: float = 0.25,
    image_size: int = 640,
    device: str = "cpu",
    nms_iou: float = 0.7,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    if image_size < 32:
        raise ValueError("Image size must be at least 32 pixels.")
    if not 0 < nms_iou <= 1:
        raise ValueError("NMS IoU threshold must be in (0, 1].")
    result = model.predict(
        image.convert("RGB"),
        conf=confidence,
        iou=nms_iou,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]
    plotted_bgr = result.plot()
    annotated = Image.fromarray(np.ascontiguousarray(plotted_bgr[:, :, ::-1]))
    detections: list[dict[str, Any]] = []
    for box in result.boxes:
        coordinates = box.xyxy[0].tolist()
        detections.append(
            {
                "class": "litter",
                "confidence": float(box.conf[0].item()),
                "xyxy": [float(value) for value in coordinates],
            }
        )
    return annotated, detections
