"""Update the research paper with metrics produced by evaluate.py."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


PROJECT_DIR = Path(__file__).parent
PAPER_PATH = PROJECT_DIR.parent / "YOLO_Based_Detection_of_Floating_Waste.docx"
PARAGRAPH_ALIASES = {
    "The proposed pipeline receives water-body images": "The implemented prototype receives water-body images",
    "Partitioning should be performed": ("For the pilot, the", "For this "),
    "The baseline is the Ultralytics YOLOv8": "The implemented baseline is Ultralytics YOLOv8n",
    "Evaluation should use the held-out test partition": "The confidence threshold (",
    "This manuscript defines a proposed method": "A working YOLOv8n prototype",
    "4.2 Expected System Outputs": "4.2 ",
    "For an input image containing visually distinguishable litter": "The run used",
    "4.3 Metrics to Report After Implementation": "4.3 ",
    "Table 4.2 defines the minimum quantitative measures": "At the selected operating point",
    "Small targets may occupy too few pixels": "The measurements establish that",
    "The proposed methodology should be considered successful": "The annotated examples",
    "This paper has proposed a single-class YOLOv8 workflow": "This study implemented a working single-class",
    "No implementation results are available": "The measured values are preliminary",
    "Table 4.2: Proposed evaluation measures": "Table 4.2: Quantitative results",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_paragraph(document: Document, prefix: str):
    alias = PARAGRAPH_ALIASES.get(prefix)
    for paragraph in document.paragraphs:
        aliases = (alias,) if isinstance(alias, str) else alias or ()
        if paragraph.text.startswith(prefix) or any(
            paragraph.text.startswith(candidate) for candidate in aliases
        ):
            return paragraph
    raise ValueError(f"Could not find paragraph beginning with {prefix!r}")


def replace_paragraph(paragraph, text: str, bold: bool | None = None) -> None:
    paragraph.clear()
    run = paragraph.add_run(text)
    run.font.name = "Times New Roman"
    if bold is not None:
        run.bold = bold
    if not bold:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def insert_paragraph_after(paragraph, text: str = "", centered: bool = False):
    element = OxmlElement("w:p")
    paragraph._p.addnext(element)
    from docx.text.paragraph import Paragraph

    inserted = Paragraph(element, paragraph._parent)
    inserted.paragraph_format.space_after = Pt(4)
    if text:
        run = inserted.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.bold = centered
    if centered:
        inserted.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return inserted


def remove_existing_result_figures(document: Document) -> None:
    for paragraph in list(document.paragraphs):
        if paragraph.text.startswith(("Figure 4.2:", "Figure 4.3:")):
            previous = paragraph._p.getprevious()
            if previous is not None and previous.xpath(".//w:drawing"):
                previous.getparent().remove(previous)
            paragraph._p.getparent().remove(paragraph._p)


def format_metrics_chart(metrics: dict, target: Path, run_label: str) -> None:
    values = [
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["map50"],
        metrics["map50_95"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    bars = axes[0].bar(
        ["Precision", "Recall", "F1", "mAP@.50", "mAP@.50:.95"],
        [100 * value for value in values],
        color=["#2E7D73", "#4472A2", "#A36B22", "#76529A", "#927BAA"],
    )
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Percent")
    axes[0].set_title("Detection (mAP uses confidence sweep)")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{100 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    mae = metrics["count_mae"]
    axes[1].bar(["Count MAE"], [mae], color="#76529A")
    axes[1].set_ylabel("Objects / image")
    axes[1].set_title("Count error")
    axes[1].set_ylim(0, max(1.0, mae * 1.25))
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(0, mae + max(0.02, mae * 0.03), f"{mae:.2f}", ha="center", fontsize=9)
    fig.suptitle(f"{run_label.title()} test-set results (not a full-dataset benchmark)", fontsize=12)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def update_table(table, rows: list[list[str]], run_label: str) -> None:
    headers = ("Measure", f"{run_label.title()} result", "Protocol / scope")
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9)
                run.bold = True
    while len(table.rows) > 1:
        table._tbl.remove(table.rows[-1]._tr)
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = value
            for paragraph in cells[index].paragraphs:
                paragraph.paragraph_format.space_after = Pt(1)
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=PAPER_PATH)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_DIR / "outputs" / "expanded_416" / "metrics.json",
    )
    parser.add_argument(
        "--training-record",
        type=Path,
        default=PROJECT_DIR / "training_record_expanded_416.json",
    )
    parser.add_argument("--examples", type=Path)
    args = parser.parse_args()
    paper = args.paper
    metrics_path = args.metrics
    training_path = args.training_record
    metrics = load_json(metrics_path) if metrics_path.is_file() else {}
    data_root = Path(metrics["dataset"]) if metrics.get("dataset") else PROJECT_DIR / "data" / "tud_gv_pilot"
    manifest_path = data_root / "manifest.json"
    examples_path = args.examples or metrics_path.parent / "examples.jpg"
    for required in (paper, metrics_path, training_path, manifest_path, examples_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required paper-update input is missing: {required}")

    training = load_json(training_path)
    manifest = load_json(manifest_path)
    split = manifest["splits"]
    device = training["device"].upper()
    image_size = training["image_size"]
    epochs = training["epochs_requested"]
    batch = training["batch_size"]
    seed = training["seed"]
    train_images = split["train"]["images"]
    val_images = split["val"]["images"]
    test_images = split["test"]["images"]
    run_label = "expanded run" if train_images > 100 else "pilot"
    train_boxes = split["train"]["boxes"]
    test_boxes = sum(
        len(
            [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
        for path in (data_root / "labels" / "test").glob("*.txt")
    )
    training_groups = ", ".join(manifest["groups"]["train"])
    validation_groups = ", ".join(manifest["groups"]["val"])
    test_groups = ", ".join(manifest["groups"]["test"])
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    map50 = metrics["map50"]
    map50_95 = metrics["map50_95"]
    confidence_threshold = metrics["confidence_threshold"]
    nms_iou = metrics["nms_iou_threshold"]
    mae = metrics["count_mae"]
    tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
    latency = metrics["mean_predict_annotate_seconds_per_image"]
    p_pct, r_pct, f_pct = (100 * value for value in (precision, recall, f1))

    chart_path = metrics_path.parent / "metrics_chart.png"
    format_metrics_chart(metrics, chart_path, run_label)
    doc = Document(paper)
    remove_existing_result_figures(doc)

    replace_paragraph(
        find_paragraph(doc, "Floating litter in rivers"),
        (
            "Floating litter in rivers, lakes, and canals is an environmental monitoring concern because visual "
            "surveys are labor-intensive and provide limited temporal coverage. This study implements a single-class "
            "YOLOv8n object detector that localizes visible floating litter and outputs confidence scores, an "
            f"image-level count, and an annotated image. The study uses the TUD-GV detection dataset, whose published "
            "record contains 1,501 RGB images and 8,181 boxes for the single class “litter.” A group-separated subset "
            f"of {train_images} training, {val_images} validation, and {test_images} test images was used; whole "
            "filename experiment groups were kept disjoint. The model was fine-tuned for "
            f"{epochs} epochs at {image_size}px on {device} hardware. The confidence threshold ({confidence_threshold:.3f}) "
            f"and NMS IoU ({nms_iou:.2f}) were selected on the validation subset. Fixed-threshold, one-to-one matching "
            f"at IoU 0.50 on the held-out test subset gave precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f}, "
            f"and count MAE {mae:.3f} objects per image. Separately, the test-set confidence-sweep evaluation gave "
            f"mAP@0.50 {map50:.3f} ({map50:.1%}) and mAP@0.50:0.95 {map50_95:.3f}. These preliminary results "
            "are measurements from one held-out experiment group. The 88.3% figure is mAP@0.50, not classification "
            "accuracy; these results are not a full-dataset benchmark or "
            "evidence of generalization to other waterways. No material-specific plastic classification is performed."
        ),
    )

    replace_paragraph(
        find_paragraph(doc, "The proposed pipeline receives water-body images"),
        (
            "The implemented prototype receives water-body images and produces a per-image set of litter boxes, "
            "confidence values, a count, and an annotated image. Figure 3.1 shows the workflow implemented in "
            "the prototype. The measured results are reported separately in Section 4."
        ),
    )
    replace_paragraph(
        find_paragraph(doc, "Partitioning should be performed"),
        (
            f"For this evaluation, the {train_images} training images ({train_boxes} boxes) are from the disjoint "
            f"filename group(s) {training_groups}; {val_images} validation images are sampled from "
            f"{validation_groups}; and {test_images} test images ({test_boxes} annotations) are sampled from "
            f"{test_groups}. Groups do not cross splits. This exploratory subset evaluates the pipeline "
            "but does not substitute for evaluation over the complete 1,501-image dataset."
        ),
    )
    replace_paragraph(
        find_paragraph(doc, "The baseline is the Ultralytics YOLOv8"),
        (
            "The implemented baseline is Ultralytics YOLOv8n [12], initialized from the published pretrained "
            "yolov8n.pt detection checkpoint and fine-tuned on the single target class “litter.” The run used "
            f"{epochs} epochs, an input size of {image_size} pixels, batch size {batch}, random seed {seed}, "
            f"and {device} execution with PyTorch {training['torch_version']} and Ultralytics "
            f"{training['ultralytics_version']}. The full run record is saved with the prototype. This {run_label} "
            "uses a selected subset and is not a fully tuned benchmark."
        ),
    )
    replace_paragraph(
        find_paragraph(doc, "Evaluation should use the held-out test partition"),
        (
            f"The confidence threshold ({confidence_threshold:.3f}) and NMS IoU ({nms_iou:.2f}) were selected "
            "by F1 on the validation subset, then fixed for the held-out test group. Predicted boxes were matched "
            "one-to-one to annotations at IoU 0.50, with each ground-truth box used at most once. Precision, "
            "recall, and F1 are micro-aggregated over matched "
            "instances; count MAE is the mean absolute difference between predicted and annotated object counts "
            "per image. Mean inference time was measured after a model warm-up on the CPU. Average precision "
            "was computed on this test subset using the standard confidence sweep with a confidence floor of "
            "0.001 and NMS IoU of 0.70."
        ),
    )

    replace_paragraph(
        find_paragraph(doc, "This manuscript defines a proposed method"),
        (
            "A working YOLOv8n prototype was trained and evaluated on a disjoint-group TUD-GV subset. "
            f"The following results are measurements from that run, not projected values. Because the {run_label} subset "
            f"contains {train_images} training images and {test_images} test images, the results are preliminary and should not be read as a "
            "publication-quality estimate of performance across the full dataset."
        ),
    )
    replace_paragraph(find_paragraph(doc, "4.2 Expected System Outputs"), f"4.2 {run_label.title()} training and evaluation results", True)
    replace_paragraph(
        find_paragraph(doc, "For an input image containing visually distinguishable litter"),
        (
            f"The run used {train_images} training images, {val_images} validation images, and {test_images} "
            f"test images, with no experiment group shared across these splits. The training images contained "
            f"{train_boxes} annotated boxes and the test images contained {test_boxes}. YOLOv8n was initialized "
            f"from pretrained weights and fine-tuned for {epochs} epochs at {image_size}px with batch size {batch} "
            f"on {device}. Validation selected confidence {confidence_threshold:.3f} and NMS IoU {nms_iou:.2f}; "
            "test detections were matched to annotations at IoU >= 0.50. Average precision was also computed "
            "on the held-out test subset. Figure 4.2 "
            "summarizes the resulting precision, recall, F1, and counting error. "
            f"The run produced TP={tp}, FP={fp}, and FN={fn} across the test images."
        ),
    )
    replace_paragraph(find_paragraph(doc, "4.3 Metrics to Report After Implementation"), f"4.3 Interpretation and limitations of the {run_label}", True)
    replace_paragraph(
        find_paragraph(doc, "Table 4.2 defines the minimum quantitative measures"),
        (
            f"At the selected operating point, precision was {p_pct:.1f}%, recall was {r_pct:.1f}%, and F1 "
            f"was {f_pct:.1f}%. The detector achieved mAP@0.50 of {map50:.3f} ({map50:.1%}); this object-detection "
            "metric is not classification accuracy. The image-level count MAE was "
            f"{mae:.2f} objects per image. Mean per-image "
            f"prediction-and-annotation time after warm-up was {latency:.4f} s/image "
            f"({1 / latency:.1f} images/s) on the CPU; this excludes file decoding, application startup, "
            f"and UI overhead. Test-set mAP@0.50:0.95 was {map50_95:.3f}. "
            "Figure 4.3 shows annotated examples with predicted and ground-truth counts. The small test sample "
            "makes these values sensitive to individual images; no confidence intervals are reported."
        ),
    )
    replace_paragraph(
        find_paragraph(doc, "Small targets may occupy too few pixels"),
        (
            "The measurements establish that the implemented system can train, infer, draw boxes, and produce "
            "per-image counts on the selected dataset subset. They do not establish reliable performance in "
            f"different waterways or camera conditions. With only {test_images} test images from one held-out experiment "
            "group, false positives and misses can substantially change the reported precision and recall. The "
            "prototype should therefore be treated as a software and methodology demonstration. A larger "
            "group-wise evaluation on all available images is needed to assess small targets, reflections, "
            "illumination changes, and domain shift."
        ),
    )
    replace_paragraph(
        find_paragraph(doc, "The proposed methodology should be considered successful"),
        (
            "The annotated examples are qualitative illustrations and should not be interpreted as a complete "
            "error analysis. The current model predicts one generic “litter” class and cannot distinguish bottles, "
            "bags, wrappers, natural debris, or polymer composition beyond what the dataset labels support."
        ),
    )
    replace_paragraph(
        find_paragraph(doc, "This paper has proposed a single-class YOLOv8 workflow"),
        (
            "This study implemented a working single-class YOLOv8n prototype for visible floating-litter "
            "detection. It prepares a leakage-aware, group-separated subset of the TUD-GV dataset, fine-tunes pretrained "
            "weights, filters detections, writes bounding boxes and counts, and supplies both a command-line "
            f"predictor and an image-upload application. On the {test_images}-image held-out test subset, the fixed-threshold "
            f"evaluation measured precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f}, mAP@0.50 "
            f"{map50:.3f} ({map50:.1%}), mAP@0.50:0.95 {map50_95:.3f}, and count MAE {mae:.3f}. "
            f"The mAP@0.50 value is an object-detection metric, not classification accuracy. The experiment confirms "
            f"the prototype workflow operates end to end; the small test set "
            "does not establish general performance."
        ),
    )
    replace_paragraph(
        find_paragraph(doc, "No implementation results are available"),
        (
            "The measured values are preliminary because the model was trained on only "
            f"{train_images} images and evaluated on {test_images} images from a single held-out experiment "
            "group. Further work should train and evaluate across the "
            "complete dataset with multiple group-wise splits, report full-dataset mAP and uncertainty, inspect errors by "
            "object size and scene condition, and test on an independent waterway. The model is a monitoring aid "
            "for visible detections, not an estimate of total pollution and not a classifier of material type."
        ),
    )

    # Replace the former metric-definition table with values from this run.
    metric_table = doc.tables[1]
    update_table(
        metric_table,
        [
            ["Precision", f"{precision:.3f} ({p_pct:.1f}%)", f"TP / (TP + FP), confidence >= {confidence_threshold:.3f}"],
            ["Recall", f"{recall:.3f} ({r_pct:.1f}%)", "TP / (TP + FN), IoU matching threshold 0.50"],
            ["F1 score", f"{f1:.3f} ({f_pct:.1f}%)", "Harmonic mean of micro-averaged precision and recall"],
            ["mAP@0.50", f"{map50:.3f} ({map50:.1%})", "Test split; confidence sweep; NMS IoU 0.70"],
            ["mAP@0.50:0.95", f"{map50_95:.3f}", "Test split; IoU thresholds 0.50 through 0.95"],
            ["TP / FP / FN", f"{tp} / {fp} / {fn}", f"Across {test_images} held-out test images"],
            ["Count MAE", f"{mae:.3f} objects/image", "Mean absolute predicted-vs-annotated count difference"],
            ["Operating point", f"Confidence {confidence_threshold:.3f}; NMS IoU {nms_iou:.2f}", f"Chosen on validation; F1={metrics['validation_selected_f1']:.3f}"],
            ["Prediction + annotation time", f"{latency:.4f} s/image", "CPU; after warm-up; excludes image decoding"],
            ["Training set", f"{train_images} images; {train_boxes} boxes", f"{epochs} epochs, {image_size}px, batch {batch}"],
            ["Test set", f"{test_images} images; {test_boxes} boxes", f"Disjoint group {test_groups}; thresholds fixed from validation"],
        ],
        run_label,
    )
    replace_paragraph(
        find_paragraph(doc, "Table 4.2: Proposed evaluation measures"),
        f"Table 4.2: Quantitative results from the {run_label} run",
        True,
    )

    # Insert figures next to the results text rather than append them after the references.
    results_text = find_paragraph(doc, "At the selected operating point")
    figure = insert_paragraph_after(results_text, centered=True)
    figure.add_run().add_picture(str(chart_path), width=Inches(5.9))
    caption = insert_paragraph_after(
        figure,
        f"Figure 4.2: {run_label.title()} test metrics; mAP is computed over the confidence sweep",
        centered=True,
    )
    caption.paragraph_format.keep_with_next = True
    limitation = find_paragraph(doc, "The measurements establish that")
    examples = insert_paragraph_after(limitation, centered=True)
    examples.add_run().add_picture(str(examples_path), width=Inches(5.9))
    insert_paragraph_after(
        examples,
        f"Figure 4.3: Annotated {run_label} test images; each panel reports predicted and annotated counts",
        centered=True,
    )

    doc.core_properties.comments = (
        f"Results updated from the reproducible {run_label} prototype run; not a full benchmark."
    )
    doc.save(paper)
    print(f"Updated {paper}")
    print(
        f"Run metrics: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, "
        f"count_MAE={mae:.4f}, test_images={test_images}"
    )


if __name__ == "__main__":
    main()
