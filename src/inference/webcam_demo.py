"""Passo 8 — real-time plate detection from a webcam (or video file/URL).

Opens `--source` (default 0 = default webcam) with OpenCV, runs the trained
detector frame by frame, and draws the boxes live. Press 'q' or Esc to quit.

Known limitation (see PLANO_PROJETO.md): the dataset is all fixed
CCTV-style cameras (elevated, fixed angle/distance). A handheld/laptop
webcam is a different domain, so don't be surprised if this performs worse
here than the test-set metrics suggest.

Usage:
    .venv/Scripts/python.exe src/inference/webcam_demo.py
    .venv/Scripts/python.exe src/inference/webcam_demo.py --source 1 --conf 0.4
    .venv/Scripts/python.exe src/inference/webcam_demo.py --source path/to/video.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "plate_detector.pt"

BOX_COLOR = (0, 200, 0)  # BGR
TEXT_COLOR = (255, 255, 255)


def parse_source(value):
    """--source 0/1/... -> webcam index; anything else -> file path or URL as-is."""
    try:
        return int(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--source",
        type=parse_source,
        default=0,
        help="índice da webcam (0, 1, ...) ou caminho de vídeo/URL",
    )
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="960 casa com o treino (configs/train.yaml) e deve detectar melhor, mas é mais lento no CPU",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise RuntimeError(
            f"pesos não encontrados: {args.weights} — rode o Passo 7 (export_model.py) primeiro."
        )

    print(f"Pesos: {args.weights}")
    print(
        "Aviso: o dataset de treino é todo de câmeras CFTV fixas (ângulo/altura/distância "
        "diferentes de uma webcam de notebook). Não estranhe se o desempenho aqui for pior "
        "que as métricas do split de teste (ver PLANO_PROJETO.md).\n"
    )

    model = YOLO(str(args.weights))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"não consegui abrir a fonte de vídeo: {args.source}")

    print("Pressione 'q' ou Esc para sair.")
    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Fonte de vídeo terminou ou falhou ao ler o frame.")
                break

            predictions = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            result = (
                predictions[0]
                if isinstance(predictions, (list, tuple))
                else predictions
            )
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), BOX_COLOR, 2)
                    label = f"placa {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (xmin, max(0, ymin - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        BOX_COLOR,
                        2,
                    )

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now
            cv2.putText(
                frame,
                f"{fps:.1f} FPS",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TEXT_COLOR,
                2,
            )

            cv2.imshow("Detector de placa - YOLO11n", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' ou Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
