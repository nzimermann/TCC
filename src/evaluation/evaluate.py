"""Evaluate a trained plate detector on the held-out test split.

Two layers of evaluation:
  1. The standard Ultralytics `model.val(split="test")` run — official
     precision/recall/mAP50/mAP50-95 numbers, the ones that go in the TCC.
  2. A custom IoU-matching pass broken down by subgroup (rain vs no rain,
     time of day, plate size), because Ultralytics' own val() doesn't slice
     metrics by dataset metadata. This answers the question from
     PLANO_PROJETO.md Passo 6: does the model do worse on small/distant
     plates or at night/in the rain?

Ground truth for the subgroup pass comes straight from
data/annotations_v2.json (same xy -> bbox logic used everywhere else in
src/data_prep), matched against the test split list in dataset_yolo/splits.json
- not from re-reading the YOLO .txt labels, so rain/time metadata is available.

Usage:
    .venv/Scripts/python.exe src/evaluation/evaluate.py --weights runs/detect/full_run/weights/best.pt
    .venv/Scripts/python.exe src/evaluation/evaluate.py --weights runs/detect/smoke_test/weights/best.pt --limit 50
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_PATH = REPO_ROOT / "data" / "annotations_v2.json"
IMAGES_DIR = REPO_ROOT / "data" / "images"
SPLITS_PATH = REPO_ROOT / "dataset_yolo" / "splits.json"
DATA_YAML = REPO_ROOT / "dataset_yolo" / "data.yaml"
RUNS_DIR = REPO_ROOT / "runs" / "detect"
REPORT_PATH = REPO_ROOT / "reports" / "eval_summary.md"

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25

# area(bbox) / area(image) cutoffs for the small/medium/large plate buckets,
# based on the full-dataset distribution measured in reports/eda_summary.md
# (median ~0.003, p10 ~0.0012, p90 ~0.0166).
SIZE_CUTOFFS = (0.0015, 0.008)

LEG_LABELS = {0: "Illegible", 1: "Poor", 2: "Good", 3: "Perfect"}


def xy_to_bbox(xy):
    xs = xy[0::2]
    ys = xy[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes, iou_threshold=IOU_THRESHOLD):
    """Greedy 1:1 matching by IoU.

    Returns (matched_gt_indices, n_fp): the indices into `gt_boxes` that
    found a matching prediction (i.e. true positives), and how many
    predicted boxes were left over (false positives). Unmatched GT indices
    are false negatives (`set(range(len(gt_boxes))) - matched_gt_indices`).
    """
    unmatched_preds = list(range(len(pred_boxes)))
    matched_gt = set()
    for i, gt in enumerate(gt_boxes):
        best_iou, best_j = 0.0, None
        for j in unmatched_preds:
            v = iou(gt, pred_boxes[j])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_threshold:
            matched_gt.add(i)
            unmatched_preds.remove(best_j)
    return matched_gt, len(unmatched_preds)


def size_bucket(area_ratio, cutoffs=SIZE_CUTOFFS):
    small_cut, large_cut = cutoffs
    if area_ratio < small_cut:
        return "pequena"
    if area_ratio > large_cut:
        return "grande"
    return "média"


def run_official_val(weights, batch):
    """Full Ultralytics val() over the entire test split. Skipped when
    --limit is set (see main()) since it always uses the whole split
    regardless of a subset - not meant for the quick debug path."""
    model = YOLO(weights)
    metrics = model.val(data=str(DATA_YAML), split="test", project=str(RUNS_DIR), name="eval_test", batch=batch)
    return model, {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def run_subgroup_eval(model, test_names, all_data, conf, batch, limit):
    if limit:
        test_names = test_names[:limit]

    # ground truth per image: list of pixel bboxes + metadata
    gt_by_name = {}
    for name in test_names:
        ann = all_data[name]
        anns = ann.get("anns", [])
        boxes = [xy_to_bbox(a["xy"]) for a in anns]
        legs = [a.get("leg") for a in anns]
        gt_by_name[name] = {"boxes": boxes, "legs": legs, "rain": bool(ann.get("rain")), "time": ann.get("time")}

    image_paths = [str(IMAGES_DIR / name) for name in test_names]
    results_stream = model.predict(source=image_paths, conf=conf, batch=batch, stream=True, verbose=False)

    preds_by_name = {}
    dims_by_name = {}
    for result in results_stream:
        name = Path(result.path).name
        preds_by_name[name] = [tuple(b) for b in result.boxes.xyxy.tolist()]
        h, w = result.orig_shape
        dims_by_name[name] = (w, h)

    subgroups = {
        "rain=sim": {"tp": 0, "fp": 0, "fn": 0, "n_imgs": 0},
        "rain=não": {"tp": 0, "fp": 0, "fn": 0, "n_imgs": 0},
        "time=morning": {"tp": 0, "fp": 0, "fn": 0, "n_imgs": 0},
        "time=afternoon": {"tp": 0, "fp": 0, "fn": 0, "n_imgs": 0},
        "time=evening": {"tp": 0, "fp": 0, "fn": 0, "n_imgs": 0},
        "time=night": {"tp": 0, "fp": 0, "fn": 0, "n_imgs": 0},
    }
    size_groups = {
        "placa pequena": {"tp": 0, "fn": 0},
        "placa média": {"tp": 0, "fn": 0},
        "placa grande": {"tp": 0, "fn": 0},
    }
    # leg=0 (Illegible) never appears here: Passo 2 already dropped every
    # image containing one, so the test set only has leg in {1, 2, 3}.
    leg_groups = {
        "leg=1 (Poor)": {"tp": 0, "fn": 0},
        "leg=2 (Good)": {"tp": 0, "fn": 0},
        "leg=3 (Perfect)": {"tp": 0, "fn": 0},
    }
    overall = {"tp": 0, "fp": 0, "fn": 0}

    for name in test_names:
        gt = gt_by_name[name]
        preds = preds_by_name.get(name, [])
        img_w, img_h = dims_by_name.get(name, (None, None))
        matched_gt, n_fp = match_boxes(gt["boxes"], preds)
        tp, fn = len(matched_gt), len(gt["boxes"]) - len(matched_gt)

        overall["tp"] += tp
        overall["fp"] += n_fp
        overall["fn"] += fn

        rain_key = "rain=sim" if gt["rain"] else "rain=não"
        subgroups[rain_key]["tp"] += tp
        subgroups[rain_key]["fp"] += n_fp
        subgroups[rain_key]["fn"] += fn
        subgroups[rain_key]["n_imgs"] += 1

        time_key = f"time={gt['time']}"
        if time_key in subgroups:
            subgroups[time_key]["tp"] += tp
            subgroups[time_key]["fp"] += n_fp
            subgroups[time_key]["fn"] += fn
            subgroups[time_key]["n_imgs"] += 1

        if img_w and img_h:
            for i, box in enumerate(gt["boxes"]):
                xmin, ymin, xmax, ymax = box
                area_ratio = ((xmax - xmin) * (ymax - ymin)) / (img_w * img_h)
                bucket = f"placa {size_bucket(area_ratio)}"
                if i in matched_gt:
                    size_groups[bucket]["tp"] += 1
                else:
                    size_groups[bucket]["fn"] += 1

        for i, leg in enumerate(gt["legs"]):
            leg_key = f"leg={leg} ({LEG_LABELS.get(leg, '?')})"
            if leg_key not in leg_groups:
                continue  # leg=0 shouldn't occur post-Passo 2, but don't crash if it does
            if i in matched_gt:
                leg_groups[leg_key]["tp"] += 1
            else:
                leg_groups[leg_key]["fn"] += 1

    return overall, subgroups, size_groups, leg_groups, len(test_names)


def precision_recall(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall


def recall_only(counts):
    tp, fn = counts["tp"], counts["fn"]
    return tp / (tp + fn) if (tp + fn) else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="ex: runs/detect/full_run/weights/best.pt")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="avaliar só as N primeiras imagens do test (debug)")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise RuntimeError(f"pesos não encontrados: {weights}")

    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        all_data = json.load(f)
    with open(SPLITS_PATH, encoding="utf-8") as f:
        splits = json.load(f)
    test_names = splits["test"]

    print(f"Pesos: {weights}")
    if args.limit:
        print(f"--limit {args.limit}: pulando o val() oficial (ele sempre roda no split inteiro) e indo direto pro subgrupo.")
        model = YOLO(str(weights))
        official = None
    else:
        print(f"Rodando val() oficial do Ultralytics no split de teste ({len(test_names)} imagens)...")
        model, official = run_official_val(str(weights), args.batch)

    n_eval = len(test_names) if not args.limit else min(args.limit, len(test_names))
    print(f"\nRodando avaliação por subgrupo (rain/time/tamanho/legibilidade) em {n_eval} imagens, IoU>={IOU_THRESHOLD}, conf>={args.conf}...")
    overall, subgroups, size_groups, leg_groups, n_used = run_subgroup_eval(
        model, test_names, all_data, args.conf, args.batch, args.limit
    )

    lines = ["# Avaliação no split de teste\n"]
    lines.append(f"- Pesos: `{weights}`")
    lines.append(f"- Imagens no split de teste: {len(test_names)} (avaliadas nesta rodada: {n_used})\n")

    lines.append("## Métricas oficiais (Ultralytics `model.val(split='test')`)")
    if official is None:
        lines.append("- Pulado (rodando com `--limit`, é só o debug rápido por subgrupo).\n")
    else:
        lines.append(f"- Precision: {official['precision']:.3f}")
        lines.append(f"- Recall: {official['recall']:.3f}")
        lines.append(f"- mAP50: {official['map50']:.3f}")
        lines.append(f"- mAP50-95: {official['map50_95']:.3f}\n")

    p, r = precision_recall(overall)
    lines.append(f"## Checagem cruzada (matching por IoU>={IOU_THRESHOLD}, conf>={args.conf})")
    lines.append(f"- TP={overall['tp']} FP={overall['fp']} FN={overall['fn']}")
    lines.append(f"- Precision: {p:.3f}  Recall: {r:.3f}\n")

    lines.append("## Por condição de chuva")
    lines.append("| Grupo | Imagens | TP | FP | FN | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in ("rain=não", "rain=sim"):
        c = subgroups[key]
        p, r = precision_recall(c)
        lines.append(f"| {key} | {c['n_imgs']} | {c['tp']} | {c['fp']} | {c['fn']} | {p:.3f} | {r:.3f} |")
    lines.append("")

    lines.append("## Por período do dia")
    lines.append("| Grupo | Imagens | TP | FP | FN | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in ("time=morning", "time=afternoon", "time=evening", "time=night"):
        c = subgroups[key]
        p, r = precision_recall(c)
        lines.append(f"| {key} | {c['n_imgs']} | {c['tp']} | {c['fp']} | {c['fn']} | {p:.3f} | {r:.3f} |")
    lines.append("")

    lines.append(f"## Por tamanho da placa (área bbox/imagem — cortes: {SIZE_CUTOFFS})")
    lines.append("- Sem coluna de precision/FP aqui: falso positivo não corresponde a nenhuma placa real, então não tem \"tamanho\" próprio.")
    lines.append("| Grupo | TP | FN | Recall |")
    lines.append("|---|---|---|---|")
    for key in ("placa pequena", "placa média", "placa grande"):
        c = size_groups[key]
        lines.append(f"| {key} | {c['tp']} | {c['fn']} | {recall_only(c):.3f} |")
    lines.append("")

    lines.append("## Por legibilidade da placa (`leg` do dataset original)")
    lines.append("- `leg=0` (Illegible) não aparece: o Passo 2 já descartou toda imagem com alguma placa ilegível.")
    lines.append("- Mesma ressalva do tamanho: sem precision/FP, falso positivo não tem legibilidade própria.")
    lines.append("| Grupo | TP | FN | Recall |")
    lines.append("|---|---|---|---|")
    for key in ("leg=1 (Poor)", "leg=2 (Good)", "leg=3 (Perfect)"):
        c = leg_groups[key]
        lines.append(f"| {key} | {c['tp']} | {c['fn']} | {recall_only(c):.3f} |")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRelatório salvo em {REPORT_PATH}")


if __name__ == "__main__":
    main()
