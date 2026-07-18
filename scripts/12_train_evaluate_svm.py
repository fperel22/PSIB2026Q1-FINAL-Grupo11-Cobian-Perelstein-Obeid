"""
12_train_evaluate_svm.py

Entrena y evalúa dos SVM-RBF:
- características derivadas de máscara manual
- características derivadas de máscara automática

Entrada esperada:
    outputs/tables_aligned/11_manual_pca_components.csv
    outputs/tables_aligned/11_auto_pca_components.csv

Reglas metodológicas:
- Ambos casos deben contener exactamente los mismos filename/subset.
- GridSearchCV se ajusta solamente con classifiers/train.
- classifiers/val se informa como validación independiente.
- El modelo final se reajusta con train+val.
- classifiers/test se usa una sola vez al final.
- ROC-AUC se calcula con el puntaje continuo de decisión.
"""

from __future__ import annotations

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


META_COLUMNS = {"filename", "class", "subset"}
CLASS_MAP = {"benign": 0, "malignant": 1}
SCRIPT_VERSION = "12-clean-v2"


def load_pca_table(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {path}")

    df = pd.read_csv(path)
    required = META_COLUMNS
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} no contiene las columnas requeridas: {sorted(missing)}")

    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise ValueError(f"{path} no contiene columnas PC.")

    df = df.dropna(subset=pc_cols + ["filename", "class", "subset"]).copy()
    if not df["class"].isin(CLASS_MAP).all():
        raise ValueError(f"{path} contiene clases no reconocidas.")

    df["y"] = df["class"].map(CLASS_MAP).astype(int)
    return df, pc_cols


def validate_alignment(
    manual_df: pd.DataFrame,
    auto_df: pd.DataFrame,
) -> None:
    manual_keys = set(zip(manual_df["filename"], manual_df["subset"]))
    auto_keys = set(zip(auto_df["filename"], auto_df["subset"]))

    if manual_keys != auto_keys:
        only_manual = sorted(manual_keys - auto_keys)[:10]
        only_auto = sorted(auto_keys - manual_keys)[:10]
        raise ValueError(
            "Las tablas PCA manual y automática no contienen exactamente las "
            "mismas imágenes/subsets.\n"
            f"Solo manual, primeros casos: {only_manual}\n"
            f"Solo automática, primeros casos: {only_auto}\n"
            "Recalculá el PCA desde outputs/tables_aligned."
        )

    manual_labels = manual_df.set_index("filename")["class"].sort_index()
    auto_labels = auto_df.set_index("filename")["class"].sort_index()
    if not manual_labels.equals(auto_labels):
        raise ValueError("Las clases manual/automática no coinciden por filename.")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_score: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, decision_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source: str,
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_xticks([0, 1], ["Benigna", "Maligna"])
    ax.set_yticks([0, 1], ["Benigna", "Maligna"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(f"SVM-RBF ({source}) — test")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(
    y_true: np.ndarray,
    decision_score: np.ndarray,
    source: str,
    out_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, decision_score)
    auc = roc_auc_score(y_true, decision_score)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("1 − especificidad")
    ax.set_ylabel("Sensibilidad")
    ax.set_title(f"SVM-RBF ({source}) — ROC test")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_source(
    source: str,
    df: pd.DataFrame,
    pc_cols: list[str],
    output_tables: Path,
    output_models: Path,
    output_figures: Path,
    seed: int,
) -> list[dict[str, object]]:
    train = df[df["subset"] == "train"].copy()
    val = df[df["subset"] == "val"].copy()
    test = df[df["subset"] == "test"].copy()

    print(
        f"\n[{source}] train={len(train)} | val={len(val)} | "
        f"test={len(test)} | PCs={len(pc_cols)}"
    )

    for split_name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            raise ValueError(f"[{source}] subset {split_name} vacío.")
        if part["y"].nunique() != 2:
            raise ValueError(
                f"[{source}] subset {split_name} no contiene ambas clases."
            )

    base_model = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=seed,
    )

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(train[pc_cols], train["y"])

    print(f"[{source}] mejores hiperparámetros: {search.best_params_}")
    print(
        f"[{source}] balanced accuracy CV train: "
        f"{search.best_score_:.4f}"
    )

    rows: list[dict[str, object]] = []

    val_pred = search.best_estimator_.predict(val[pc_cols])
    val_score = search.best_estimator_.decision_function(val[pc_cols])
    val_metrics = compute_metrics(
        val["y"].to_numpy(),
        val_pred,
        val_score,
    )
    rows.append(
        {
            "model": "SVM-RBF",
            "mask_source": source,
            "split": "val",
            **val_metrics,
            "best_C": search.best_params_["C"],
            "best_gamma": str(search.best_params_["gamma"]),
        }
    )

    print(
        f"[{source}] VAL balanced_accuracy="
        f"{val_metrics['balanced_accuracy']:.4f} | "
        f"AUC={val_metrics['roc_auc']:.4f}"
    )

    final_model = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=seed,
        C=search.best_params_["C"],
        gamma=search.best_params_["gamma"],
    )

    train_val = pd.concat([train, val], ignore_index=True)
    final_model.fit(train_val[pc_cols], train_val["y"])

    test_pred = final_model.predict(test[pc_cols])
    test_score = final_model.decision_function(test[pc_cols])
    test_metrics = compute_metrics(
        test["y"].to_numpy(),
        test_pred,
        test_score,
    )

    rows.append(
        {
            "model": "SVM-RBF",
            "mask_source": source,
            "split": "test",
            **test_metrics,
            "best_C": search.best_params_["C"],
            "best_gamma": str(search.best_params_["gamma"]),
        }
    )

    print(
        f"[{source}] TEST balanced_accuracy="
        f"{test_metrics['balanced_accuracy']:.4f} | "
        f"AUC={test_metrics['roc_auc']:.4f}"
    )

    output_tables.mkdir(parents=True, exist_ok=True)
    output_models.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    predictions = test[["filename", "class", "subset"]].copy()
    predictions["y_true"] = test["y"].to_numpy()
    predictions["y_pred"] = test_pred
    predictions["decision_score_malignant"] = test_score
    predictions.to_csv(
        output_tables / f"12_svm_{source}_test_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": final_model,
            "pc_columns": pc_cols,
            "class_map": CLASS_MAP,
        },
        output_models / f"12_svm_{source}.joblib",
    )

    params = {
        "best_params": search.best_params_,
        "cv_best_balanced_accuracy": float(search.best_score_),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_pcs": int(len(pc_cols)),
        "seed": seed,
    }
    (
        output_models / f"12_svm_{source}_params.json"
    ).write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    save_confusion_matrix(
        test["y"].to_numpy(),
        test_pred,
        source,
        output_figures / f"12_svm_{source}_confusion_matrix.png",
    )
    save_roc_curve(
        test["y"].to_numpy(),
        test_score,
        source,
        output_figures / f"12_svm_{source}_roc.png",
    )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("outputs/tables_aligned"),
        help="Carpeta con 11_manual_pca_components.csv y 11_auto_pca_components.csv.",
    )
    parser.add_argument(
        "--output-tables",
        type=Path,
        default=Path("outputs/tables"),
    )
    parser.add_argument(
        "--output-models",
        type=Path,
        default=Path("outputs/models"),
    )
    parser.add_argument(
        "--output-figures",
        type=Path,
        default=Path("outputs/figures/svm"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Script version: {SCRIPT_VERSION}")

    manual_df, manual_pc_cols = load_pca_table(
        args.tables_dir / "11_manual_pca_components.csv"
    )
    auto_df, auto_pc_cols = load_pca_table(
        args.tables_dir / "11_auto_pca_components.csv"
    )

    validate_alignment(manual_df, auto_df)

    print(
        "Alineación validada: manual y automática contienen exactamente "
        "las mismas imágenes y subsets."
    )

    rows: list[dict[str, object]] = []
    rows.extend(
        run_source(
            "manual",
            manual_df,
            manual_pc_cols,
            args.output_tables,
            args.output_models,
            args.output_figures,
            args.seed,
        )
    )
    rows.extend(
        run_source(
            "auto",
            auto_df,
            auto_pc_cols,
            args.output_tables,
            args.output_models,
            args.output_figures,
            args.seed,
        )
    )

    summary = pd.DataFrame(rows)
    args.output_tables.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_tables / "12_svm_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nGuardado: {summary_path}")


if __name__ == "__main__":
    main()
