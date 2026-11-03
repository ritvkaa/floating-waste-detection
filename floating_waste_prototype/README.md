# Floating litter detection prototype

A single-class YOLOv8n prototype for detecting visible floating litter in images. It produces bounding boxes, confidence scores, an object count, an annotated image, and a JSON record. The output class is only `litter`; it does not identify plastic types or material composition.

## Setup

Use Python 3.10 or later. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch and torchvision must be installed as a compatible pair for the selected platform. For a CPU-only Windows setup, install the current CPU wheels from the official PyTorch installation selector before installing the remaining requirements.

## Prepare the expanded experiment data

```powershell
python prepare_dataset.py --output data\tud_gv_expanded
```

The script obtains selected images and annotations from the official TUD-GV Zenodo record into `data\tud_gv_expanded`. Whole experiment groups are kept disjoint:

- Training: all images in `exp14`, `exp27`, and `exp29`
- Validation: all images in `exp16`
- Test: all images in `exp23`

This expanded experiment uses 321 training, 67 validation, and 70 test images, not the complete 1,501-image release. The earlier 54-image pilot was used for software checks; results below and in the paper refer to the expanded split. The dataset is CC BY 4.0. Cite the associated paper and dataset (DOI `10.1016/j.watres.2024.122405` and `10.5281/zenodo.13730228`) when redistributing or publishing results.

## Train and evaluate

```powershell
python train.py --data data\tud_gv_expanded\dataset.yaml --epochs 15 --image-size 416 --batch 4 --device cpu --name expanded_416
python evaluate.py --data data\tud_gv_expanded\dataset.yaml --weights weights\floating_litter_yolov8n_expanded_416.pt --image-size 416 --batch 4 --device cpu --output-name expanded_416
```

Training starts from Ultralytics `yolov8n.pt`, saves the fine-tuned weights and run record, and evaluation selects confidence and NMS thresholds on validation before testing. Evaluation reports fixed-threshold precision/recall/F1 and count MAE plus test mAP. This expanded run is evaluated on all images in the held-out test experiment group.

## Expanded-run results

On the held-out `exp23` group (70 images, 288 annotations), the run measured precision 0.769, recall 0.903, F1 0.831, mAP@0.50 0.883, mAP@0.50:0.95 0.595, and count MAE 0.829 objects/image. The confidence threshold (0.25) and NMS IoU (0.45) were selected on the `exp16` validation group. These are measurements from a single held-out experiment group—not a complete-dataset benchmark or an estimate of performance on other waterways.

## Run the application

```powershell
streamlit run app.py
```

Upload a JPG, PNG, or WebP water-body image. The app displays the input, annotation, confidence for each detection, count, and a downloadable annotated image.

For command-line inference:

```powershell
python predict.py --image path\to\water_image.jpg --device cpu
```

The CLI uses the validation-selected confidence/NMS thresholds and image size recorded by `evaluate.py`, then saves an annotated image and detection JSON under `outputs`. All detections are model predictions and should be reviewed before environmental decisions.
