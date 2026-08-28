"""Passo 7 — promote a training run's best.pt to the project's "official" model.

Copies <run>/weights/best.pt to models/plate_detector.pt and sanity-checks
that the copy actually loads back with Ultralytics (same single class,
'placa', task='detect') before calling it done — a bad copy or a checkpoint
from the wrong run is a bad time to discover later, at webcam-demo time.

Usage:
    .venv/Scripts/python.exe src/training/export_model.py --weights runs/detect/full_run/weights/best.pt
    .venv/Scripts/python.exe src/training/export_model.py --weights runs/detect/smoke_test/weights/best.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "models" / "plate_detector.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="ex: runs/detect/full_run/weights/best.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    src = Path(args.weights)
    if not src.exists():
        raise RuntimeError(f"pesos não encontrados: {src}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, args.output)

    model = YOLO(str(args.output))
    size_mb = args.output.stat().st_size / (1024 * 1024)

    print(f"Copiado: {src} -> {args.output}")
    print(f"Tamanho: {size_mb:.1f} MB")
    print(f"Task: {model.task}")
    print(f"Classes: {model.names}")

    if model.task != "detect" or list(model.names.values()) != ["placa"]:
        raise RuntimeError(
            "o modelo copiado não parece ser o detector de placa esperado "
            f"(task={model.task}, names={model.names}) - confira se --weights aponta pro run certo."
        )

    print("\nOK — models/plate_detector.pt pronto.")


if __name__ == "__main__":
    main()
