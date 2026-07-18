"""
09b_summarize_unet_metrics.py

Resume las métricas Dice e IoU generadas por 09_unet_inference.py.

Entrada:
    data/processed/auto_masks/
        unet_metrics_classifiers_group.csv

Salidas:
    outputs/tables/09_unet_metrics_summary.csv
    outputs/figures/unet/09_unet_metrics_distribution.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/auto_masks/"
            "unet_metrics_classifiers_group.csv"
        ),
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path(
            "outputs/tables/09_unet_metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path(
            "outputs/figures/unet/"
            "09_unet_metrics_distribution.png"
        ),
    )

    return parser.parse_args()


def summarize_group(
    group_name: str,
    data: pd.DataFrame,
) -> dict[str, object]:
    dice = data["dice"].to_numpy(dtype=float)
    iou = data["iou"].to_numpy(dtype=float)

    return {
        "group": group_name,
        "n": int(len(data)),
        "dice_mean": float(np.mean(dice)),
        "dice_std": float(np.std(dice, ddof=0)),
        "dice_median": float(np.median(dice)),
        "dice_min": float(np.min(dice)),
        "dice_max": float(np.max(dice)),
        "iou_mean": float(np.mean(iou)),
        "iou_std": float(np.std(iou, ddof=0)),
        "iou_median": float(np.median(iou)),
        "iou_min": float(np.min(iou)),
        "iou_max": float(np.max(iou)),
        "n_dice_equal_zero": int(
            np.isclose(dice, 0.0).sum()
        ),
        "n_dice_le_0_10": int((dice <= 0.10).sum()),
        "n_dice_lt_0_50": int((dice < 0.50).sum()),
    }


def save_distribution_figure(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    bins = np.linspace(0.0, 1.0, 21)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
    )

    for class_name in ("benign", "malignant"):
        class_data = data[data["class"] == class_name]

        axes[0].hist(
            class_data["dice"],
            bins=bins,
            alpha=0.55,
            label=class_name,
        )
        axes[1].hist(
            class_data["iou"],
            bins=bins,
            alpha=0.55,
            label=class_name,
        )

    axes[0].set_title("Distribución de Dice")
    axes[0].set_xlabel("Dice")
    axes[0].set_ylabel("Cantidad de imágenes")
    axes[0].set_xlim(0, 1)
    axes[0].legend()

    axes[1].set_title("Distribución de IoU")
    axes[1].set_xlabel("IoU")
    axes[1].set_ylabel("Cantidad de imágenes")
    axes[1].set_xlim(0, 1)
    axes[1].legend()

    fig.suptitle(
        "U-Net sobre el grupo classifiers",
        fontsize=13,
    )
    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"No se encontró: {args.input}"
        )

    data = pd.read_csv(args.input)

    required = {"filename", "class", "dice", "iou"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    data["dice"] = pd.to_numeric(
        data["dice"],
        errors="coerce",
    )
    data["iou"] = pd.to_numeric(
        data["iou"],
        errors="coerce",
    )

    if data[["dice", "iou"]].isna().any().any():
        raise ValueError(
            "El archivo contiene Dice o IoU no numéricos."
        )

    if not data["class"].isin(
        ["benign", "malignant"]
    ).all():
        invalid = sorted(
            set(data["class"])
            - {"benign", "malignant"}
        )
        raise ValueError(
            f"Clases inesperadas: {invalid}"
        )

    if not (
        data["dice"].between(0, 1).all()
        and data["iou"].between(0, 1).all()
    ):
        raise ValueError(
            "Dice e IoU deben estar en el intervalo [0, 1]."
        )

    rows = [
        summarize_group("global", data),
        summarize_group(
            "benign",
            data[data["class"] == "benign"],
        ),
        summarize_group(
            "malignant",
            data[data["class"] == "malignant"],
        ),
    ]

    summary = pd.DataFrame(rows)

    args.output_table.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary.to_csv(
        args.output_table,
        index=False,
    )

    save_distribution_figure(
        data,
        args.output_figure,
    )

    print("\nResumen U-Net:")
    print(summary.to_string(index=False))

    print(f"\nGuardado: {args.output_table}")
    print(f"Guardado: {args.output_figure}")


if __name__ == "__main__":
    main()