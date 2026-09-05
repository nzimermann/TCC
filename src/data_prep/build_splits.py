"""Build train/val/test splits over the filtered LPLCv2 annotations.

Split unit is the *camera* (`cam`), not the image: all images from the same
camera go to the same split, so no camera leaks between train and
val/test (same rationale as the dataset authors' own `cam_sep` scenario,
just applied to full-image detection instead of plate crops).

Images with `cam == null` have no camera to group by. Per the user's
decision, this is fine — they carry no leakage risk either way, so each is
treated as its own singleton group and can land in any split.

Balancing uses a greedy "largest group first, assign to the split furthest
below its target size" heuristic (a form of Longest-Processing-Time
scheduling), because camera sizes are very skewed (median 21 images/camera,
one camera alone has 3,603) — a naive random split over cameras would
produce wildly uneven split sizes.

Output: dataset_yolo/splits.json -> {"train": [...], "val": [...], "test": [...]}
(filenames only; Passo 4 combines this with annotations_v2.json to write the
actual YOLO images/labels folders).

Usage:
    .venv/Scripts/python.exe src/data_prep/build_splits.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from filters import load_filtered_annotations

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "yolo" / "splits.json"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "reports" / "split_summary.md"

SPLIT_NAMES = ("train", "val", "test")


def build_groups(data: dict) -> dict:
    """Map group_key -> list of image filenames. One group per real camera
    id; one singleton group per cam=null image."""
    groups = defaultdict(list)
    for name, ann in data.items():
        cam = ann.get("cam")
        key = f"cam_{cam}" if cam is not None else f"single_{name}"
        groups[key].append(name)
    return groups


def balanced_split(groups: dict, ratios=(0.7, 0.15, 0.15), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)  # break ties randomly instead of by insertion/camera-id order
    items.sort(key=lambda kv: len(kv[1]), reverse=True)  # largest groups first

    total = sum(len(imgs) for _, imgs in items)
    targets = [total * r for r in ratios]
    counts = [0, 0, 0]
    assignment = {name: [] for name in SPLIT_NAMES}
    cams_per_split = {name: set() for name in SPLIT_NAMES}

    for key, imgs in items:
        i = max(range(3), key=lambda idx: targets[idx] - counts[idx])
        split = SPLIT_NAMES[i]
        assignment[split].extend(imgs)
        counts[i] += len(imgs)
        cams_per_split[split].add(key)

    return assignment, cams_per_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    data = load_filtered_annotations()
    groups = build_groups(data)
    assignment, cams_per_split = balanced_split(groups, ratios=ratios, seed=args.seed)

    # sanity check: no group (camera) split across more than one partition
    seen = set()
    for split in SPLIT_NAMES:
        split_keys = {
            (
                f"cam_{data[name].get('cam')}"
                if data[name].get("cam") is not None
                else f"single_{name}"
            )
            for name in assignment[split]
        }
        overlap = seen & split_keys
        assert not overlap, f"leakage detected, groups in >1 split: {overlap}"
        seen |= split_keys

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(assignment, f, indent=2)

    lines = ["# Split treino/val/teste (por câmera)\n"]
    lines.append(f"- Seed: {args.seed}")
    lines.append(f"- Proporções alvo: {ratios}")
    lines.append(
        f"- Total de imagens (após filtro do Passo 2): {sum(len(v) for v in assignment.values())}"
    )
    lines.append("")
    for split in SPLIT_NAMES:
        imgs = assignment[split]
        n_anns = sum(len(data[name].get("anns", [])) for name in imgs)
        n_cams = len({k for k in cams_per_split[split] if not k.startswith("single_")})
        n_null = sum(1 for k in cams_per_split[split] if k.startswith("single_"))
        cam_sizes = sorted(
            (
                len(groups[k])
                for k in cams_per_split[split]
                if not k.startswith("single_")
            ),
            reverse=True,
        )
        largest_cam_share = (cam_sizes[0] / len(imgs)) if imgs and cam_sizes else 0.0
        lines.append(f"## {split}")
        lines.append(
            f"- Imagens: {len(imgs)} ({len(imgs) / sum(len(v) for v in assignment.values()):.1%})"
        )
        lines.append(f"- Placas: {n_anns}")
        lines.append(f"- Câmeras distintas: {n_cams}")
        lines.append(f"- Imagens com cam=null incluídas: {n_null}")
        lines.append(
            f"- Maior câmera do split: {cam_sizes[0] if cam_sizes else 0} imagens ({largest_cam_share:.1%} do split)"
        )
        lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Split salvo em {args.output}")
    print(f"Relatório salvo em {args.report}")


if __name__ == "__main__":
    main()
