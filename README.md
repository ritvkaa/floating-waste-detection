# YOLO-Based Floating Litter Detection

A single-class YOLOv8 prototype that detects visible floating litter in water-body images, draws boxes, counts detections, and exports annotated images. It does not identify litter material or product type.

## Deliverables

- Research paper: [`YOLO_Based_Detection_of_Floating_Waste.docx`](./YOLO_Based_Detection_of_Floating_Waste.docx)
- Prototype source, setup, and run instructions: [`floating_waste_prototype/README.md`](./floating_waste_prototype/README.md)
- Trained pilot weights: [`floating_waste_prototype/weights/floating_litter_yolov8n.pt`](./floating_waste_prototype/weights/floating_litter_yolov8n.pt)
- Pilot evaluation and examples: [`floating_waste_prototype/outputs/`](./floating_waste_prototype/outputs/)

The initial CPU pilot used 54 training images, 20 validation images, and 20 held-out test images from disjoint TUD-GV experiment groups. Its measured mAP@0.50 was 0.636. This is a feasibility result, not a full-dataset benchmark; the paper reports the split and limitations.

Raw dataset files and Ultralytics training caches are excluded from Git. Run the prototype instructions to download the dataset and reproduce the experiment.
