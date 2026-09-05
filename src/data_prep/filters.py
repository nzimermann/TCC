"""Filtering rules applied before converting annotations to YOLO format.

Rules (decided in Passo 2, confirmed by the user):
  - drop the whole image if it has any illegible plate (leg == 0)
  - drop the whole image if the capturing camera was faulty (faulty == true)

These are intentionally whole-image drops, not per-annotation drops: an
illegible plate is still a real plate in the scene, but per the project's
scope, images containing one are excluded entirely rather than keeping the
image and discarding just that box.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_PATH = REPO_ROOT / "data" / "raw" / "annotations_v2.json"

ILLEGIBLE_LEG = 0


def keep_image(image_annotation: dict) -> bool:
    """Whether an image (its annotation dict, as found under a filename key
    in annotations_v2.json) should be kept for detection training."""
    if image_annotation.get("faulty"):
        return False
    anns = image_annotation.get("anns", [])
    return not any(ann.get("leg") == ILLEGIBLE_LEG for ann in anns)


def filter_annotations(data: dict) -> dict:
    """Return only the entries of `data` that pass `keep_image`."""
    return {name: ann for name, ann in data.items() if keep_image(ann)}


def load_filtered_annotations(path: Path = ANNOTATIONS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return filter_annotations(data)


if __name__ == "__main__":
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    filtered = filter_annotations(raw)
    n_anns_before = sum(len(v.get("anns", [])) for v in raw.values())
    n_anns_after = sum(len(v.get("anns", [])) for v in filtered.values())

    print(f"Imagens antes do filtro: {len(raw)}")
    print(f"Imagens depois do filtro: {len(filtered)} ({len(filtered) / len(raw):.1%})")
    print(f"Placas antes do filtro: {n_anns_before}")
    print(
        f"Placas depois do filtro: {n_anns_after} ({n_anns_after / n_anns_before:.1%})"
    )
