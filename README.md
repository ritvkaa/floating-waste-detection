# YOLO-Based Floating Litter Detection

A single-class YOLOv8 prototype that detects visible floating litter in water-body images, draws boxes, counts detections, and exports annotated images. It does not identify litter material or product type.

## Deliverables

- Research paper: [`YOLO_Based_Detection_of_Floating_Waste.docx`](./YOLO_Based_Detection_of_Floating_Waste.docx)
- Prototype source, setup, and run instructions: [`floating_waste_prototype/README.md`](./floating_waste_prototype/README.md)
- Expanded-run weights: [`floating_waste_prototype/weights/floating_litter_yolov8n_expanded_416.pt`](./floating_waste_prototype/weights/floating_litter_yolov8n_expanded_416.pt)
- Expanded evaluation metrics, chart, and examples: [`floating_waste_prototype/outputs/expanded_416/`](./floating_waste_prototype/outputs/expanded_416/)

The expanded CPU run used 321 training, 67 validation, and 70 held-out test images from disjoint TUD-GV experiment groups. It measured precision 0.769, recall 0.903, F1 0.831, mAP@0.50 0.883, mAP@0.50:0.95 0.595, and count MAE 0.829 objects/image. The earlier 54/20/20-image pilot remains available as a separate software-feasibility result. Neither run is a full-dataset benchmark; see the paper for the evaluation protocol and limitations.

Raw dataset files and Ultralytics training caches are excluded from Git. Run the prototype instructions to download the dataset and reproduce the experiment.
