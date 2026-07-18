"""
14_compare_models.py

Consolida los resultados finales en classifiers/test:
- SVM-RBF con máscara manual
- SVM-RBF con máscara automática
- EfficientNet-B0 con máscara manual
- EfficientNet-B0 con máscara automática

Además verifica que las cuatro evaluaciones usaron exactamente los mismos
filenames de test.
"""

from __future__ import annotations

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_VERSION = "14-clean-v2"

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "roc_auc",
]


def read_test_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "split" not in df.columns:
        raise ValueError(f"{path} no contiene la columna split.")

    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError(f"{path} no contiene resultados de test.")

    return test_df


def read_prediction_filenames(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "filename" not in df.columns:
        raise ValueError(f"{path} no contiene la columna filename.")

    return set(df["filename"].astype(str))


def validate_same_test_cohort(paths: list[Path]) -> None:
    cohorts = [read_prediction_filenames(path) for path in paths]
    reference = cohorts[0]

    for path, cohort in zip(paths[1:], cohorts[1:]):
        if cohort != reference:
            raise ValueError(
                "Las predicciones no usan exactamente el mismo conjunto de "
                f"test. Archivo conflictivo: {path}"
            )

    print(
        f"Cohorte de test validada: {len(reference)} imágenes "
        "idénticas en los cuatro modelos."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--svm-summary",
        type=Path,
        default=Path("outputs/tables/12_svm_summary.csv"),
    )
    parser.add_argument(
        "--cnn-manual-summary",
        type=Path,
        default=Path(
            "outputs/efficientnet/final_manual/summary.csv"
        ),
    )
    parser.add_argument(
        "--cnn-auto-summary",
        type=Path,
        default=Path(
            "outputs/efficientnet/final_auto/summary.csv"
        ),
    )

    parser.add_argument(
        "--svm-manual-predictions",
        type=Path,
        default=Path(
            "outputs/tables/12_svm_manual_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--svm-auto-predictions",
        type=Path,
        default=Path(
            "outputs/tables/12_svm_auto_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--cnn-manual-predictions",
        type=Path,
        default=Path(
            "outputs/efficientnet/final_manual/test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--cnn-auto-predictions",
        type=Path,
        default=Path(
            "outputs/efficientnet/final_auto/test_predictions.csv"
        ),
    )

    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("outputs/tables/14_model_comparison.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path(
            "outputs/figures/comparison/14_model_comparison.png"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Script version: {SCRIPT_VERSION}")

    validate_same_test_cohort(
        [
            args.svm_manual_predictions,
            args.svm_auto_predictions,
            args.cnn_manual_predictions,
            args.cnn_auto_predictions,
        ]
    )

    svm = read_test_summary(args.svm_summary)
    cnn_manual = read_test_summary(args.cnn_manual_summary)
    cnn_auto = read_test_summary(args.cnn_auto_summary)

    comparison = pd.concat(
        [svm, cnn_manual, cnn_auto],
        ignore_index=True,
    )

    required = {
        "model",
        "mask_source",
        "split",
        *METRIC_COLUMNS,
    }
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en los summaries: {sorted(missing)}"
        )

    comparison = comparison[
        [
            "model",
            "mask_source",
            "split",
            *METRIC_COLUMNS,
        ]
    ].copy()

    comparison = comparison.sort_values(
        ["model", "mask_source"]
    ).reset_index(drop=True)

    args.output_table.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    comparison.to_csv(args.output_table, index=False)

    print("\nComparación final:")
    print(comparison.to_string(index=False))

    labels = (
        comparison["model"]
        + "\n"
        + comparison["mask_source"]
    )

    plotted_metrics = [
        "balanced_accuracy",
        "f1",
        "roc_auc",
    ]

    x = list(range(len(comparison)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for index, metric in enumerate(plotted_metrics):
        positions = [
            value + (index - 1) * width
            for value in x
        ]
        ax.bar(
            positions,
            comparison[metric],
            width,
            label=metric,
        )

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor")
    ax.set_title(
        "Comparación final en classifiers/test"
    )
    ax.legend()
    fig.tight_layout()

    args.output_figure.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        args.output_figure,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nGuardado: {args.output_table}")
    print(f"Guardado: {args.output_figure}")


if __name__ == "__main__":
    main()
