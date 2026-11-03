"""Download a reproducible, experiment-group-separated TUD-GV pilot subset."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import requests
import yaml
from remotezip import RemoteZip


RECORD_URL = "https://zenodo.org/records/13730228"
IMAGES_URL = f"{RECORD_URL}/files/images.zip?download=1"
LABELS_URL = f"{RECORD_URL}/files/labels_txt.zip?download=1"
VAL_GROUPS = ("exp16",)
TEST_GROUPS = ("exp23",)
TRAIN_GROUPS = ("exp14", "exp27", "exp29")


def load_labels() -> zipfile.ZipFile:
    import io

    response = requests.get(LABELS_URL, timeout=90)
    response.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(response.content))


def label_member(image_name: str) -> str:
    return f"labels_txt/{Path(image_name).stem}.txt"


def validate_label_text(text: str, member: str) -> int:
    count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"Unexpected one-class YOLO label at {member}:{line_number}")
        values = [float(value) for value in fields[1:]]
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(f"Out-of-range normalized box at {member}:{line_number}")
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"Non-positive box size at {member}:{line_number}")
        count += 1
    return count


def prepare(output: Path) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    label_names = set(labels.namelist())
    counts: dict[str, dict[str, int]] = {}
    with RemoteZip(IMAGES_URL) as archive:
        image_infos = [
            info
            for info in archive.infolist()
            if info.filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        groups_by_split = {
            "train": TRAIN_GROUPS,
            "val": VAL_GROUPS,
            "test": TEST_GROUPS,
        }
        selected: dict[str, list[Any]] = {split: [] for split in groups_by_split}
        for info in image_infos:
            filename = Path(info.filename).name
            group_match = re.match(r"(exp\d+)_", filename)
            if not group_match:
                continue
            group = group_match.group(1)
            for split, groups in groups_by_split.items():
                if group in groups:
                    selected[split].append(info)
                    break

        for split, infos in selected.items():
            infos = sorted(infos, key=lambda item: item.filename)
            images_dir = output / "images" / split
            labels_dir = output / "labels" / split
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            split_counts = Counter()
            for index, info in enumerate(infos, start=1):
                image_name = Path(info.filename).name
                annotation_name = label_member(image_name)
                if annotation_name not in label_names:
                    raise FileNotFoundError(f"No annotation found for {image_name}")
                label_text = labels.read(annotation_name).decode("utf-8")
                box_count = validate_label_text(label_text, annotation_name)
                image_path = images_dir / image_name
                label_path = labels_dir / f"{Path(image_name).stem}.txt"
                if not image_path.is_file():
                    image_path.write_bytes(archive.read(info.filename))
                label_path.write_text(label_text, encoding="utf-8")
                split_counts["images"] += 1
                split_counts["boxes"] += box_count
                if index % 100 == 0 or index == len(infos):
                    print(
                        f"{split}: prepared {index}/{len(infos)} images "
                        f"({split_counts['boxes']} boxes)",
                        flush=True,
                    )
            counts[split] = dict(split_counts)

    dataset_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "litter"},
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "source": "TUD-GV Dataset for Floating Litter Detection (object detection task)",
        "record_url": RECORD_URL,
        "doi": "10.5281/zenodo.13730228",
        "split_method": (
            "selected complete experiment groups; exp14, exp27, and exp29 for training; "
            "exp16 for validation; exp23 for test"
        ),
        "groups": groups_by_split,
        "splits": counts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "tud_gv_pilot",
        help="Directory for the selected images, labels, and dataset.yaml",
    )
    args = parser.parse_args()
    manifest = prepare(args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
