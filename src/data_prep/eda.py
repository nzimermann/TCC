"""Exploratory data analysis over LPLCv2's annotations_v2.json.

Reads the full annotation file (no sampling) and:
  - reports distributions used to decide the filtering rules (Passo 2)
  - measures plate bbox size relative to image size (drives the imgsz choice)
  - simulates the effect of the proposed filters (leg == 0, faulty == True)
  - saves plots to reports/figures/ and a text summary to reports/eda_summary.md

Usage:
    .venv/Scripts/python.exe src/data_prep/eda.py
"""

import json
import struct
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_PATH = REPO_ROOT / "data" / "annotations_v2.json"
IMAGES_DIR = REPO_ROOT / "data" / "images"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
SUMMARY_PATH = REPO_ROOT / "reports" / "eda_summary.md"

LEG_LABELS = {0: "Illegible", 1: "Poor", 2: "Good", 3: "Perfect"}


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


def pct(sorted_list, p):
    if not sorted_list:
        return float("nan")
    idx = min(int(len(sorted_list) * p), len(sorted_list) - 1)
    return sorted_list[idx]


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    n_images = len(data)
    n_anns = sum(len(v.get("anns", [])) for v in data.values())

    leg_counts = Counter()
    faulty_count = 0
    rain_count = 0
    time_counts = Counter()
    day_counts = Counter()
    cam_counts = Counter()
    anns_per_image = Counter()

    for img in data.values():
        if img.get("faulty"):
            faulty_count += 1
        if img.get("rain"):
            rain_count += 1
        time_counts[img.get("time")] += 1
        day_counts[img.get("day")] += 1
        cam_counts[img.get("cam")] += 1
        anns = img.get("anns", [])
        anns_per_image[len(anns)] += 1
        for ann in anns:
            leg_counts[ann.get("leg")] += 1

    # --- bbox size relative to image (full dataset, header-only reads) ---
    width_ratios, height_ratios, area_ratios = [], [], []
    missing_images = 0
    for name, img in data.items():
        anns = [a for a in img.get("anns", []) if a.get("leg") != 0]
        if not anns:
            continue
        path = IMAGES_DIR / name
        if not path.exists():
            missing_images += 1
            continue
        dims = get_jpeg_size(path)
        if not dims:
            continue
        w_img, h_img = dims
        for ann in anns:
            xs = ann["xy"][0::2]
            ys = ann["xy"][1::2]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            width_ratios.append(w / w_img)
            height_ratios.append(h / h_img)
            area_ratios.append((w * h) / (w_img * h_img))

    width_ratios.sort()
    height_ratios.sort()
    area_ratios.sort()

    # --- simulate filters: drop whole image if it has any leg==0 ann; drop faulty images ---
    kept_images = 0
    kept_anns = 0
    dropped_illegible_img = 0
    dropped_faulty_img = 0
    for img in data.values():
        anns = img.get("anns", [])
        has_illegible = any(a.get("leg") == 0 for a in anns)
        if img.get("faulty"):
            dropped_faulty_img += 1
            continue
        if has_illegible:
            dropped_illegible_img += 1
            continue
        kept_images += 1
        kept_anns += len(anns)

    # --- plots ---
    plt.figure(figsize=(5, 4))
    labels = [LEG_LABELS[k] for k in sorted(leg_counts)]
    values = [leg_counts[k] for k in sorted(leg_counts)]
    plt.bar(labels, values, color="#4c72b0")
    plt.title("Placas por nível de legibilidade")
    plt.ylabel("quantidade")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "leg_distribution.png", dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4))
    order = ["morning", "afternoon", "evening", "night"]
    plt.bar(order, [time_counts.get(k, 0) for k in order], color="#dd8452")
    plt.title("Imagens por período do dia")
    plt.ylabel("quantidade")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "time_distribution.png", dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.hist(area_ratios, bins=60, color="#55a868")
    plt.xlim(0, pct(area_ratios, 0.98))
    plt.title("Área da placa / área da imagem (leg > 0)")
    plt.xlabel("razão de área")
    plt.ylabel("quantidade de placas")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bbox_area_ratio.png", dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4))
    max_anns = max(anns_per_image)
    xs = list(range(max_anns + 1))
    plt.bar(xs, [anns_per_image.get(x, 0) for x in xs], color="#c44e52")
    plt.title("Placas por imagem")
    plt.xlabel("nº de placas na imagem")
    plt.ylabel("nº de imagens")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "anns_per_image.png", dpi=120)
    plt.close()

    cam_images = sorted(v for k, v in cam_counts.items() if k is not None)
    n_cams_null = cam_counts.get(None, 0)

    # --- write summary ---
    lines = []
    lines.append("# EDA — LPLCv2 (annotations_v2.json)\n")
    lines.append(f"- Total de imagens: **{n_images}**")
    lines.append(f"- Total de placas anotadas: **{n_anns}**")
    lines.append(
        f"- Imagens ausentes em `data/images/` (referenciadas no JSON mas sem arquivo): **{missing_images}**\n"
    )

    lines.append("## Legibilidade (`leg`)")
    for k in sorted(leg_counts):
        lines.append(
            f"- {LEG_LABELS[k]} ({k}): {leg_counts[k]} ({leg_counts[k] / n_anns:.1%})"
        )
    lines.append("")

    lines.append("## Outros atributos")
    lines.append(
        f"- `faulty=true`: {faulty_count} imagens ({faulty_count / n_images:.1%})"
    )
    lines.append(f"- `rain=true`: {rain_count} imagens ({rain_count / n_images:.1%})")
    lines.append(f"- Períodos do dia: {dict(time_counts)}")
    lines.append(f"- Dias distintos (`day`): {sorted(day_counts)}")
    lines.append(f"- Câmeras distintas (`cam`, excluindo nulo): {len(cam_images)}")
    lines.append(f"- Imagens com `cam=null`: {n_cams_null}")
    if cam_images:
        lines.append(
            f"- Imagens por câmera — mediana: {pct(cam_images, 0.5)}, "
            f"p10: {pct(cam_images, 0.1)}, p90: {pct(cam_images, 0.9)}, max: {cam_images[-1]}"
        )
    lines.append("")

    lines.append("## Placas por imagem")
    lines.append(f"- Distribuição: {dict(sorted(anns_per_image.items()))}")
    lines.append("")

    lines.append(
        "## Tamanho da bbox em relação à imagem (apenas leg > 0, dataset completo)"
    )
    lines.append(f"- Amostras: {len(area_ratios)}")
    lines.append(
        f"- Largura relativa — mediana: {pct(width_ratios,0.5):.4f}, p10: {pct(width_ratios,0.1):.4f}, "
        f"p90: {pct(width_ratios,0.9):.4f}"
    )
    lines.append(
        f"- Altura relativa — mediana: {pct(height_ratios,0.5):.4f}, p10: {pct(height_ratios,0.1):.4f}, "
        f"p90: {pct(height_ratios,0.9):.4f}"
    )
    lines.append(
        f"- Área relativa — mediana: {pct(area_ratios,0.5):.5f}, p10: {pct(area_ratios,0.1):.5f}, "
        f"p90: {pct(area_ratios,0.9):.5f}"
    )
    lines.append("")

    lines.append("## Efeito dos filtros propostos (Passo 2)")
    lines.append(f"- Imagens descartadas por `faulty=true`: {dropped_faulty_img}")
    lines.append(
        f"- Imagens descartadas por conter ao menos 1 placa `leg=0`: {dropped_illegible_img}"
    )
    lines.append(
        f"- Imagens restantes para treino: **{kept_images}** ({kept_images / n_images:.1%} do total)"
    )
    lines.append(
        f"- Placas restantes para treino: **{kept_anns}** ({kept_anns / n_anns:.1%} do total)"
    )
    lines.append("")

    lines.append("## Figuras geradas")
    lines.append("- `reports/figures/leg_distribution.png`")
    lines.append("- `reports/figures/time_distribution.png`")
    lines.append("- `reports/figures/bbox_area_ratio.png`")
    lines.append("- `reports/figures/anns_per_image.png`")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nResumo salvo em {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
