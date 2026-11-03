"""Streamlit image-upload interface for the floating-litter detector."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import load_model, predict_image


PROJECT_DIR = Path(__file__).parent
WEIGHTS = PROJECT_DIR / "weights" / "floating_litter_yolov8n_expanded_416.pt"
DEPLOYMENT_CONFIG = PROJECT_DIR / "outputs" / "expanded_416" / "deployment_config.json"
DEPLOYMENT_SETTINGS = (
    json.loads(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))
    if DEPLOYMENT_CONFIG.is_file()
    else {"confidence": 0.25, "nms_iou": 0.7}
)
NMS_IOU = float(DEPLOYMENT_SETTINGS["nms_iou"])
IMAGE_SIZE = int(DEPLOYMENT_SETTINGS["image_size"])

st.set_page_config(page_title="Floating Litter Detector", page_icon="🌊", layout="wide")
st.title("Floating Litter Detector")
st.caption("YOLOv8 single-class prototype — detections are visual estimates, not material identification.")


@st.cache_resource
def get_model(weights_path: str):
    return load_model(weights_path)


if not WEIGHTS.is_file():
    st.error("The fine-tuned model is not present yet. Follow the setup and training steps in README.md.")
    st.stop()

uploaded = st.file_uploader("Upload a water-body image", type=("jpg", "jpeg", "png", "webp"))
confidence = st.slider(
    "Detection confidence threshold",
    0.05,
    1.0,
    float(DEPLOYMENT_SETTINGS["confidence"]),
    0.005,
)
st.caption(f"Validation-selected NMS IoU: {NMS_IOU:.2f} · input size: {IMAGE_SIZE}px")

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        model = get_model(str(WEIGHTS))
        annotated, detections = predict_image(
            model,
            image,
            confidence,
            image_size=IMAGE_SIZE,
            nms_iou=NMS_IOU,
        )
        left, right = st.columns(2)
        left.image(image, caption="Input image", use_container_width=True)
        right.image(annotated, caption="Annotated detections", use_container_width=True)
        st.metric("Detected visible litter objects", len(detections))
        st.dataframe(
            [
                {"Class": item["class"], "Confidence": round(item["confidence"], 4)}
                for item in detections
            ],
            use_container_width=True,
        )
        output = io.BytesIO()
        annotated.save(output, format="JPEG", quality=92)
        st.download_button(
            "Download annotated image",
            data=output.getvalue(),
            file_name=f"{Path(uploaded.name).stem}_detected.jpg",
            mime="image/jpeg",
        )
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not process this image: {error}")
