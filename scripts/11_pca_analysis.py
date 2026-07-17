"""
11_pca_analysis.py

Modulo 11 - PCA sobre las features del modulo 10.

Toma las tablas de features (una fila por lesion, 53 features + metadatos) y, para
cada fuente de mascara (manual y auto) POR SEPARADO:

    1. Descarta las columnas meta y se queda con las 53 features numericas.
    2. Descarta filas con NaN (en auto: mascaras vacias / nivel wavelet incompleto)
       y las loguea. En manual normalmente no hay ninguna.
    3. Estandariza (StandardScaler) y aplica PCA.
       *** CRITICO - sin fuga de datos: el scaler y el PCA se AJUSTAN SOLO con las
           filas subset=='train'. val y test se transforman con esos objetos ya
           fiteados. Nunca se fitea con el dataset completo. ***
    4. Elige el numero de componentes que alcanza ~95% de varianza explicada.
    5. Analiza los loadings -> ranking de las FEATURES ORIGINALES mas significativas.

Salidas (por fuente <src> in {manual, auto}):
    outputs/tables/
        11_<src>_pca_components.csv        matriz transformada (PC1..PCk) con
                                           filename/class/subset -> entrada del SVM
        11_<src>_pca_explained_variance.csv
        11_<src>_pca_loadings.csv          loadings (features x PCk retenidas)
        11_<src>_pca_feature_ranking.csv   ranking de features por loading
    outputs/figures/
        11_<src>_pca_explained_variance.png
        11_<src>_pca_top_features.png
        11_<src>_pca_loadings_heatmap.png
    outputs/models/
        11_<src>_scaler.joblib             StandardScaler fiteado (solo train)
        11_<src>_pca.joblib                PCA fiteado (k componentes, solo train)
        11_<src>_pca_meta.json             orden de features, k, varianza, n_train...

Reusa el naming del manifest (las columnas filename/class/subset ya vienen en los
CSV del modulo 10). No toca modulos previos.

Ejecucion (desde la raiz del repo):
    python scripts/11_pca_analysis.py                 # manual y auto
    python scripts/11_pca_analysis.py --sources auto  # solo auto
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

# Columnas meta del modulo 10 (todo lo demas son features numericas).
META_COLUMNS = ["filename", "class", "subset", "mask_source", "mask_area_px", "wavelet_levels"]
# Columnas que se conservan en la matriz transformada para alinear con el label.
KEEP_COLUMNS = ["filename", "class", "subset"]

DEFAULT_TABLES_DIR = REPO_ROOT / "outputs" / "tables"
DEFAULT_FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
DEFAULT_MODELS_DIR = REPO_ROOT / "outputs" / "models"
DEFAULT_SOURCES = ["manual", "auto"]
DEFAULT_VARIANCE_THRESHOLD = 0.95
DEFAULT_TOP_FEATURES = 10


# --------------------------------------------------------------------------- #
# Carga y limpieza
# --------------------------------------------------------------------------- #
def load_features(tables_dir: Path, source: str) -> Tuple[pd.DataFrame, List[str]]:
    path = tables_dir / f"features_{source}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {path}. Corre antes scripts/10_extract_features.py.")
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    return df, feature_cols


def drop_nan_rows(df: pd.DataFrame, feature_cols: List[str], source: str) -> pd.DataFrame:
    valid_mask = df[feature_cols].notna().all(axis=1)
    dropped = df[~valid_mask]
    if len(dropped) > 0:
        print(f"[{source}] descartando {len(dropped)} fila(s) con NaN (no entran al PCA):")
        for _, r in dropped.iterrows():
            nan_cols = [c for c in feature_cols if pd.isna(r[c])]
            resumen = "todas las features" if len(nan_cols) == len(feature_cols) else f"{len(nan_cols)} cols (ej: {nan_cols[:3]})"
            print(f"    - {r['class']}/{r['filename']} (subset={r['subset']}, "
                  f"mask_area_px={r['mask_area_px']}, wavelet_levels={r['wavelet_levels']}) -> NaN en {resumen}")
    else:
        print(f"[{source}] sin filas con NaN.")
    return df[valid_mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Ajuste PCA (SOLO train) y transformacion
# --------------------------------------------------------------------------- #
def fit_scaler_pca_on_train(
    df_valid: pd.DataFrame,
    feature_cols: List[str],
    variance_threshold: float,
    random_state: int,
) -> Tuple[StandardScaler, PCA, PCA, int, np.ndarray]:
    """
    Ajusta StandardScaler y PCA usando UNICAMENTE las filas subset=='train'.
    Devuelve (scaler, pca_k, pca_full, k, cum_evr_full).
    """
    train_df = df_valid[df_valid["subset"] == "train"]
    if len(train_df) == 0:
        raise ValueError("No hay filas subset=='train'; no se puede ajustar sin fuga de datos.")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # <-- fit SOLO con train

    # PCA completo para la curva de varianza (svd_solver='full' -> deterministico).
    pca_full = PCA(svd_solver="full", random_state=random_state)
    pca_full.fit(X_train_scaled)                    # <-- fit SOLO con train
    cum_evr = np.cumsum(pca_full.explained_variance_ratio_)

    # k = menor nro de componentes que llega al umbral de varianza.
    k = int(np.searchsorted(cum_evr, variance_threshold) + 1)
    k = min(k, pca_full.n_components_)

    # PCA definitivo a k componentes (mismos ejes que pca_full por svd 'full').
    pca_k = PCA(n_components=k, svd_solver="full", random_state=random_state)
    pca_k.fit(X_train_scaled)                       # <-- fit SOLO con train

    print(f"    ajuste (scaler + PCA) con {len(train_df)} muestras subset=='train' "
          f"(features={len(feature_cols)})")
    print(f"    componentes para >= {variance_threshold:.0%} varianza: k={k} "
          f"(varianza acumulada en k = {cum_evr[k - 1]:.4f})")
    return scaler, pca_k, pca_full, k, cum_evr


def transform_all(
    df_valid: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    pca_k: PCA,
) -> pd.DataFrame:
    """Transforma TODAS las filas validas con los objetos fiteados en train."""
    X = df_valid[feature_cols].to_numpy(dtype=np.float64)
    Z = pca_k.transform(scaler.transform(X))
    out = df_valid[KEEP_COLUMNS].reset_index(drop=True).copy()
    pc_cols = [f"PC{i + 1}" for i in range(Z.shape[1])]
    out[pc_cols] = Z
    return out


# --------------------------------------------------------------------------- #
# Loadings y ranking de features
# --------------------------------------------------------------------------- #
def compute_loadings_and_ranking(
    pca_k: PCA,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    k = pca_k.n_components_
    pc_cols = [f"PC{i + 1}" for i in range(k)]

    # loadings: filas = features, columnas = componentes.
    loadings = pd.DataFrame(pca_k.components_.T, index=feature_cols, columns=pc_cols)

    # Importancia de cada feature = suma sobre las k componentes retenidas de
    # |loading| ponderado por la varianza explicada de esa componente. Asi las
    # componentes que explican mas pesan mas en el ranking.
    evr = pca_k.explained_variance_ratio_
    importance = (loadings.abs().to_numpy() * evr[np.newaxis, :]).sum(axis=1)
    ranking = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
        "abs_loading_PC1": loadings["PC1"].abs().to_numpy(),
    })
    ranking["importance_pct"] = 100.0 * ranking["importance"] / ranking["importance"].sum()
    ranking = ranking.sort_values("importance", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return loadings, ranking


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def plot_explained_variance(
    pca_full: PCA, k: int, threshold: float, source: str, out_path: Path,
) -> None:
    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)
    comps = np.arange(1, len(evr) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(comps, evr, color="#7fa8d0", label="Varianza individual")
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Varianza explicada (individual)")
    ax1.set_ylim(0, max(evr) * 1.15)

    ax2 = ax1.twinx()
    ax2.plot(comps, cum, color="#c0392b", marker="o", markersize=3, label="Varianza acumulada")
    ax2.axhline(threshold, color="gray", linestyle="--", linewidth=1)
    ax2.axvline(k, color="green", linestyle="--", linewidth=1)
    ax2.set_ylabel("Varianza explicada (acumulada)")
    ax2.set_ylim(0, 1.02)
    ax2.annotate(f"k={k} -> {cum[k - 1]:.3f}", xy=(k, cum[k - 1]),
                 xytext=(k + max(1, len(evr) * 0.03), min(cum[k - 1], 0.9)),
                 arrowprops=dict(arrowstyle="->", color="green"), color="green", fontsize=9)

    ax1.set_title(f"Modulo 11 - PCA ({source}) - varianza explicada\n"
                  f"k={k} componentes para >= {threshold:.0%} (linea gris = umbral, verde = k)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_top_features(ranking: pd.DataFrame, top_n: int, source: str, out_path: Path) -> None:
    top = ranking.head(top_n).iloc[::-1]  # invertido para que el 1 quede arriba
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.5 * top_n + 1.5)))
    ax.barh(top["feature"], top["importance"], color="#5aa469")
    ax.set_xlabel("Importancia (sum_k |loading| * varianza explicada)")
    ax.set_title(f"Modulo 11 - PCA ({source}) - top {top_n} features por loading")
    for y, (val, pct) in enumerate(zip(top["importance"], top["importance_pct"])):
        ax.text(val, y, f" {pct:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loadings_heatmap(
    loadings: pd.DataFrame, ranking: pd.DataFrame, source: str, out_path: Path,
    top_features: int = 15, max_components: int = 8,
) -> None:
    feats = ranking.head(top_features)["feature"].tolist()
    pcs = list(loadings.columns[:max_components])
    mat = loadings.loc[feats, pcs].to_numpy()

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(pcs) + 3), 0.45 * len(feats) + 2))
    vmax = np.abs(mat).max()
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pcs)))
    ax.set_xticklabels(pcs)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_title(f"Modulo 11 - PCA ({source}) - loadings\n(top {top_features} features x primeras {len(pcs)} PC)")
    for i in range(len(feats)):
        for j in range(len(pcs)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if abs(mat[i, j]) < 0.6 * vmax else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="loading")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Corrida por fuente
# --------------------------------------------------------------------------- #
def run_source(
    source: str, tables_dir: Path, out_tables_dir: Path, figures_dir: Path,
    models_dir: Path, variance_threshold: float, top_features: int, random_state: int,
) -> None:
    print(f"\n########## Fuente de mascara: {source} ##########")
    df, feature_cols = load_features(tables_dir, source)
    print(f"[{source}] filas={len(df)} | features={len(feature_cols)}")

    df_valid = drop_nan_rows(df, feature_cols, source)
    counts = df_valid["subset"].value_counts().to_dict()
    print(f"[{source}] validas={len(df_valid)} | por subset={counts}")

    scaler, pca_k, pca_full, k, cum_evr = fit_scaler_pca_on_train(
        df_valid, feature_cols, variance_threshold, random_state)

    components = transform_all(df_valid, feature_cols, scaler, pca_k)
    loadings, ranking = compute_loadings_and_ranking(pca_k, feature_cols)

    # --- guardar tablas ---
    out_tables_dir.mkdir(parents=True, exist_ok=True)
    comp_path = out_tables_dir / f"11_{source}_pca_components.csv"
    var_path = out_tables_dir / f"11_{source}_pca_explained_variance.csv"
    load_path = out_tables_dir / f"11_{source}_pca_loadings.csv"
    rank_path = out_tables_dir / f"11_{source}_pca_feature_ranking.csv"

    components.to_csv(comp_path, index=False)
    var_df = pd.DataFrame({
        "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "cumulative_explained_variance": cum_evr,
        "retained": np.arange(1, len(cum_evr) + 1) <= k,
    })
    var_df.to_csv(var_path, index=False)
    loadings.to_csv(load_path, index_label="feature")
    ranking.to_csv(rank_path, index=False)

    # --- guardar figuras ---
    plot_explained_variance(pca_full, k, variance_threshold, source,
                            figures_dir / f"11_{source}_pca_explained_variance.png")
    plot_top_features(ranking, top_features, source,
                      figures_dir / f"11_{source}_pca_top_features.png")
    plot_loadings_heatmap(loadings, ranking, source,
                          figures_dir / f"11_{source}_pca_loadings_heatmap.png")

    # --- guardar modelos (scaler + pca) + metadata ---
    models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / f"11_{source}_scaler.joblib"
    pca_path = models_dir / f"11_{source}_pca.joblib"
    meta_path = models_dir / f"11_{source}_pca_meta.json"
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca_k, pca_path)
    meta = {
        "source": source,
        "feature_names": feature_cols,            # ORDEN exacto para reconstruir X
        "n_features": len(feature_cols),
        "variance_threshold": variance_threshold,
        "n_components": k,
        "cumulative_variance_at_k": float(cum_evr[k - 1]),
        "n_train_fit": int((df_valid["subset"] == "train").sum()),
        "n_valid_rows": int(len(df_valid)),
        "n_dropped_nan": int(len(df) - len(df_valid)),
        "subset_counts": counts,
        "explained_variance_ratio_retained": pca_k.explained_variance_ratio_.tolist(),
        "random_state": random_state,
        "scaler_joblib": scaler_path.name,
        "pca_joblib": pca_path.name,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # --- verificacion / reporte ---
    pc_cols = [c for c in components.columns if c.startswith("PC")]
    n_nan = int(components[pc_cols].isna().to_numpy().sum())
    print(f"\n    --- Verificacion [{source}] ---")
    print(f"    muestras usadas para FITEAR (solo train): {meta['n_train_fit']}")
    print(f"    varianza explicada por las k={k} componentes: {cum_evr[k - 1]:.4f}")
    print(f"    matriz transformada: {components.shape} (PC1..PC{k}) | NaN en PCs: {n_nan}")
    print(f"    class/subset preservados: columnas {KEEP_COLUMNS} presentes = "
          f"{all(c in components.columns for c in KEEP_COLUMNS)}")
    print(f"    filas por subset en la salida: {components['subset'].value_counts().to_dict()}")
    print(f"\n    Top {top_features} features por loading:")
    for _, r in ranking.head(top_features).iterrows():
        print(f"      {int(r['rank']):2d}. {r['feature']:28s} imp={r['importance']:.4f} ({r['importance_pct']:.1f}%)")

    print(f"\n    Guardado:")
    for p in [comp_path, var_path, load_path, rank_path, scaler_path, pca_path, meta_path]:
        print(f"      {p}")

    assert n_nan == 0, f"Hay NaN en la matriz transformada de {source}."


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR,
                   help="Carpeta con features_<source>.csv (default: outputs/tables)")
    p.add_argument("--output-tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, choices=["manual", "auto"])
    p.add_argument("--variance-threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD)
    p.add_argument("--top-features", type=int, default=DEFAULT_TOP_FEATURES)
    p.add_argument("--random-state", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Tablas de entrada : {args.tables_dir}")
    print(f"Umbral varianza   : {args.variance_threshold:.0%}")
    print(f"Fuentes           : {args.sources}")
    for source in args.sources:
        run_source(source, args.tables_dir, args.output_tables_dir, args.figures_dir,
                   args.models_dir, args.variance_threshold, args.top_features, args.random_state)
    print("\nListo.")


if __name__ == "__main__":
    main()
