"""Convert LPLCv2 annotations + the camera-based splits into a YOLO detection dataset.

Reads data/yolo/splits.json (written by build_splits.py) and
data/raw/annotations_v2.json, and for every image in every split:
  - computes the plate bbox(es) from `xy` (the xmin/ymin/xmax/ymax logic
    already validated visually with visualize_annotations.py)
  - normalizes them to YOLO format `<class> <xc> <yc> <w> <h>`, all in
    [0, 1] relative to the image's *actual* width/height (resolutions vary
    by camera, so this can't be a fixed constant)
  - hardlinks (falls back to copy) the source image into
    data/yolo/images/<split>/
  - writes the matching label file into data/yolo/labels/<split>/

Also writes data/yolo/data.yaml, ready for `YOLO(...).train(data=...)`.

Images in splits.json already passed build_splits.py's filter (no
faulty=true, no leg=0 plates), so every annotation left here is used as-is -
no per-annotation filtering happens in this script.

Usage:
    .venv/Scripts/python.exe src/data_prep/convert_annotations.py
"""

import json
import os
import shutil
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_PATH = REPO_ROOT / "data" / "raw" / "annotations_v2.json"
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images"
SPLITS_PATH = REPO_ROOT / "data" / "yolo" / "splits.json"
OUT_DIR = REPO_ROOT / "data" / "yolo"

CLASS_ID = 0  # single class
CLASS_NAME = "placa"


def get_jpeg_size(path):
    """Read width/height straight from the JPEG SOF marker (no full decode)."""
    with open(path, "rb") as f:
        data = f.read(64 * 1024)
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            h = struct.unpack(">H", data[i + 5 : i + 7])[0]
            w = struct.unpack(">H", data[i + 7 : i + 9])[0]
            return w, h
        seglen = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + seglen
    return None


def xy_to_bbox(xy):
    xs = xy[0::2]
    ys = xy[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_yolo_line(xmin, ymin, xmax, ymax, img_w, img_h):
    xc = (xmin + xmax) / 2 / img_w
    yc = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink: instant, no extra disk space on the same volume
    except OSError:
        shutil.copy2(src, dst)


def convert_split(split_name, filenames, all_data):
    img_out = OUT_DIR / "images" / split_name
    lbl_out = OUT_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_ok, n_skipped, n_boxes = 0, 0, 0
    for i, name in enumerate(filenames, 1):
        src_path = IMAGES_DIR / name
        if not src_path.exists():
            print(f"[aviso] imagem ausente, pulando: {name}")
            n_skipped += 1
            continue

        dims = get_jpeg_size(src_path)
        if not dims:
            print(f"[aviso] não consegui ler dimensões, pulando: {name}")
            n_skipped += 1
            continue
        img_w, img_h = dims

        anns = all_data[name].get("anns", [])
        lines = [
            bbox_to_yolo_line(*xy_to_bbox(ann["xy"]), img_w, img_h) for ann in anns
        ]
        if not lines:
            # shouldn't happen after build_splits.py's filter, but skip defensively
            print(f"[aviso] sem placas válidas, pulando: {name}")
            n_skipped += 1
            continue

        link_or_copy(src_path, img_out / name)
        stem = Path(name).stem
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        n_ok += 1
        n_boxes += len(lines)
        if i % 5000 == 0:
            print(f"  [{split_name}] {i}/{len(filenames)} imagens processadas...")

    print(f"[{split_name}] OK: {n_ok} imagens, {n_boxes} placas, {n_skipped} puladas")
    return n_ok, n_boxes


def write_data_yaml():
    yaml_path = OUT_DIR / "data.yaml"
    content = (
        f"path: {OUT_DIR.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        f"names: ['{CLASS_NAME}']\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"data.yaml salvo em {yaml_path}")


def main():
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        all_data = json.load(f)
    with open(SPLITS_PATH, encoding="utf-8") as f:
        splits = json.load(f)

    summary = {}
    for split_name in ("train", "val", "test"):
        filenames = splits[split_name]
        summary[split_name] = convert_split(split_name, filenames, all_data)

    write_data_yaml()

    print("\nResumo:")
    for split_name, (n_ok, n_boxes) in summary.items():
        print(f"  {split_name}: {n_ok} imagens, {n_boxes} placas")


if __name__ == "__main__":
    main()
