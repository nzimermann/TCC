"""Visual sanity check for the license plate bounding box annotations.

Run this yourself and look at the saved image(s): if the drawn boxes don't
line up with the plates, the xy -> bbox interpretation used everywhere else
in src/data_prep (filters.py, build_splits.py, and the upcoming
convert_annotations.py) is wrong and needs fixing before Passo 4.

Interpretation being checked: `xy` is 4 (x, y) corner pairs
[x1,y1, x2,y2, x3,y3, x4,y4], in pixel coordinates of the original image,
origin at the top-left corner (OpenCV/image convention, y grows downward).
In this dataset they're always axis-aligned (verified earlier on a sample of
thousands of annotations: 0 rotated), so:

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

Usage:
    .venv/Scripts/python.exe src/data_prep/visualize_annotations.py
    .venv/Scripts/python.exe src/data_prep/visualize_annotations.py --n 5
    .venv/Scripts/python.exe src/data_prep/visualize_annotations.py --image 0a0a12e8-9c35-4dfe-a52f-9397173e4812.jpg
    .venv/Scripts/python.exe src/data_prep/visualize_annotations.py --show   # also pop up a window
"""

import argparse
import json
import random
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_PATH = REPO_ROOT / "data" / "raw" / "annotations_v2.json"
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images"
OUTPUT_DIR = REPO_ROOT / "outputs" / "reports" / "bbox_check"

LEG_LABELS = {0: "Illegible", 1: "Poor", 2: "Good", 3: "Perfect"}
LEG_COLORS = {  # BGR, since OpenCV draws in BGR not RGB
    0: (0, 0, 255),  # red
    1: (0, 128, 255),  # orange
    2: (0, 220, 220),  # yellow
    3: (0, 200, 0),  # green
}


def xy_to_bbox(xy):
    xs = xy[0::2]
    ys = xy[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def draw_image(name, ann, out_dir):
    path = IMAGES_DIR / name
    img = cv2.imread(str(path))
    if img is None:
        print(f"[aviso] não consegui abrir {path}")
        return None

    print(f"\n{name}  (cam={ann.get('cam')}, {img.shape[1]}x{img.shape[0]}px)")
    for i, box in enumerate(ann.get("anns", [])):
        xmin, ymin, xmax, ymax = xy_to_bbox(box["xy"])
        leg = box.get("leg")
        color = LEG_COLORS.get(leg, (255, 255, 255))
        label = f"{i}:{LEG_LABELS.get(leg, '?')} {box.get('ocr', '')}"
        print(
            f"  placa {i}: xy={box['xy']}  ->  bbox=({xmin},{ymin})-({xmax},{ymax})"
            f"  leg={leg} ocr={box.get('ocr')}"
        )

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            img,
            label,
            (xmin, max(0, ymin - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"check_{name}"
    cv2.imwrite(str(out_path), img)
    print(f"  salvo em {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default=None,
        help="nome exato do arquivo (ex: 0a0a12e8-9c35-4dfe-a52f-9397173e4812.jpg). Se omitido, sorteia.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="quantas imagens aleatórias verificar (ignorado com --image)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed do sorteio, para repetir a mesma amostra",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="também abre uma janela com a imagem (tecla fecha)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if args.image:
        names = [args.image]
    else:
        rng = random.Random(args.seed)
        names = rng.sample(list(data.keys()), args.n)

    for name in names:
        if name not in data:
            print(f"[erro] {name} não está em annotations_v2.json")
            continue
        out_path = draw_image(name, data[name], args.output_dir)
        if args.show and out_path is not None:
            img = cv2.imread(str(out_path))
            if img is None:
                print(f"[aviso] não consegui abrir {out_path}")
                continue
            cv2.imshow(name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
