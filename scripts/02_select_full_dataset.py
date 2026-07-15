"""
02_select_full_dataset.py

Modulo 02 (TP Final) - Seleccion del dataset completo y caracterizacion inicial.


Objetivo:

1. Cargar la metadata del dataset BUSI usando data_loading.py.
2. Seleccionar TODAS las imagenes benignas y malignas con mascara unica
   (se excluyen imagenes sin mascara o con mas de una mascara).
3. Copiar las imagenes seleccionadas y sus mascaras a una carpeta propia,
   con la misma estructura que uso el TPI:
       data/selected/busi_subset/{label}/images/
       data/selected/busi_subset/{label}/masks/
4. Caracterizar las imagenes seleccionadas.
5. Guardar tablas completas y tabla visual resumen por grupo.

Este modulo es previo al preprocesamiento (03) y al split (00_split_dataset.py
del pipeline de U-Net + clasificadores). No modifica intensidades, no
segmenta, no aplica filtros.
"""

from pathlib import Path
from shutil import copy2
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loading import build_busi_metadata, read_grayscale_image


# ============================================================
# Configuracion general
# ============================================================

RANDOM_STATE = 42  # se mantiene por consistencia, aunque ya no se usa para samplear

OUTPUT_DATA_DIR = Path("data/selected/busi_subset")
OUTPUT_TABLES_DIR = Path("outputs/tables")
OUTPUT_FIGURES_DIR = Path("outputs/figures/selected_subset")


# ============================================================
# Utilidades (identicas al TPI)
# ============================================================

def sanitize_filename(name: str) -> str:
    """
    Convierte un nombre como 'benign (68)' en un nombre seguro para archivos.
    """
    return (
        name
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convierte una mascara PNG en mascara booleana.
    """
    return mask > 0


def compute_skewness(values: np.ndarray) -> float:
    values = values.astype(np.float32).ravel()
    std = float(np.std(values))
    if std == 0:
        return 0.0
    mean = float(np.mean(values))
    z = (values - mean) / std
    return float(np.mean(z ** 3))


def compute_kurtosis(values: np.ndarray) -> float:
    values = values.astype(np.float32).ravel()
    std = float(np.std(values))
    if std == 0:
        return 0.0
    mean = float(np.mean(values))
    z = (values - mean) / std
    return float(np.mean(z ** 4) - 3.0)


def compute_image_statistics(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Identica a la version del TPI: estadisticas de intensidad global,
    de lesion, de fondo y contraste relativo lesion/fondo.
    """
    image_float = image.astype(np.float32)
    mask_bool = binarize_mask(mask)

    height, width = image.shape
    n_pixels = height * width

    image_mean = float(np.mean(image_float))
    image_std = float(np.std(image_float))
    image_min = float(np.min(image_float))
    image_max = float(np.max(image_float))
    image_median = float(np.median(image_float))
    image_p05 = float(np.percentile(image_float, 5))
    image_p95 = float(np.percentile(image_float, 95))
    dynamic_range = image_max - image_min
    image_skewness = compute_skewness(image_float)
    image_kurtosis = compute_kurtosis(image_float)

    lesion_area_px = int(mask_bool.sum())
    lesion_area_fraction = float(lesion_area_px / n_pixels)

    if lesion_area_px > 0:
        lesion_values = image_float[mask_bool]
        lesion_mean = float(np.mean(lesion_values))
        lesion_std = float(np.std(lesion_values))
        lesion_min = float(np.min(lesion_values))
        lesion_max = float(np.max(lesion_values))
    else:
        lesion_mean = np.nan
        lesion_std = np.nan
        lesion_min = np.nan
        lesion_max = np.nan

    background_mask = ~mask_bool
    if background_mask.sum() > 0:
        background_values = image_float[background_mask]
        background_mean = float(np.mean(background_values))
        background_std = float(np.std(background_values))
        background_min = float(np.min(background_values))
        background_max = float(np.max(background_values))
    else:
        background_mean = np.nan
        background_std = np.nan
        background_min = np.nan
        background_max = np.nan

    if np.isfinite(background_mean) and background_mean != 0:
        lesion_background_contrast = float(
            (background_mean - lesion_mean) / (background_mean + 1e-8)
        )
    else:
        lesion_background_contrast = np.nan

    return {
        "height": height,
        "width": width,
        "n_pixels": n_pixels,
        "image_mean": image_mean,
        "image_std": image_std,
        "image_min": image_min,
        "image_max": image_max,
        "image_median": image_median,
        "image_p05": image_p05,
        "image_p95": image_p95,
        "dynamic_range": dynamic_range,
        "image_skewness": image_skewness,
        "image_kurtosis_excess": image_kurtosis,
        "lesion_area_px": lesion_area_px,
        "lesion_area_fraction": lesion_area_fraction,
        "lesion_mean": lesion_mean,
        "lesion_std": lesion_std,
        "lesion_min": lesion_min,
        "lesion_max": lesion_max,
        "background_mean": background_mean,
        "background_std": background_std,
        "background_min": background_min,
        "background_max": background_max,
        "lesion_background_contrast": lesion_background_contrast,
    }


def copy_selected_image_and_mask(row: pd.Series) -> Dict[str, str]:
    """
    Copia una imagen seleccionada y su mascara manual a data/selected/busi_subset.
    Misma estructura que el TPI:

    data/selected/busi_subset/
    ├── benign/
    │   ├── images/
    │   └── masks/
    └── malignant/
        ├── images/
        └── masks/
    """
    label = row["label"]
    image_id = row["image_id"]
    safe_id = sanitize_filename(image_id)

    image_src = Path(row["image_path"])
    mask_src = Path(row["mask_path"])

    image_dst_dir = OUTPUT_DATA_DIR / label / "images"
    mask_dst_dir = OUTPUT_DATA_DIR / label / "masks"
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    mask_dst_dir.mkdir(parents=True, exist_ok=True)

    image_dst = image_dst_dir / f"{safe_id}.png"
    mask_dst = mask_dst_dir / f"{safe_id}_mask.png"

    copy2(image_src, image_dst)
    copy2(mask_src, mask_dst)

    return {
        "selected_image_path": str(image_dst),
        "selected_mask_path": str(mask_dst),
    }


# ============================================================
# Seleccion y caracterizacion (funcion modificada respecto al TPI)
# ============================================================

def select_subset(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona TODAS las imagenes benignas y malignas elegibles.

    Criterios (identicos al TPI, sin el muestreo de N fijo):
    - label en {benign, malignant}  (se excluye 'normal': no hay lesion)
    - debe tener mascara
    - debe tener exactamente una mascara asociada (se descartan casos
      multi-lesion para evitar ambiguedad, igual que en el TPI)
    """
    eligible = metadata[
        metadata["label"].isin(["benign", "malignant"])
        & (metadata["has_mask"] == True)
        & (metadata["n_masks"] == 1)
    ].copy()

    n_benign = int((eligible["label"] == "benign").sum())
    n_malignant = int((eligible["label"] == "malignant").sum())
    n_excluded_multi = int(
        (metadata["label"].isin(["benign", "malignant"]) & (metadata["n_masks"] > 1)).sum()
    )
    n_excluded_no_mask = int(
        (metadata["label"].isin(["benign", "malignant"]) & (metadata["has_mask"] == False)).sum()
    )

    print(f"Elegibles: {n_benign} benignas + {n_malignant} malignas = {n_benign + n_malignant}")
    print(f"Excluidas por mascara multiple: {n_excluded_multi}")
    print(f"Excluidas por falta de mascara: {n_excluded_no_mask}")

    # Se mezclan benignas y malignas y se baraja el orden (no se muestrea
    # cantidad, se usa el shuffle solo para que el CSV no quede ordenado
    # por clase, es cosmetico)
    subset = eligible.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return subset


def characterize_subset(subset: pd.DataFrame) -> pd.DataFrame:
    """
    Copia imagenes/mascaras seleccionadas y calcula estadisticas por imagen.
    Identica a la version del TPI.
    """
    records = []
    total = len(subset)

    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        if i % 50 == 0 or i == total:
            print(f"  procesando {i}/{total}...")

        image = read_grayscale_image(row["image_path"])
        mask = read_grayscale_image(row["mask_path"])

        copied_paths = copy_selected_image_and_mask(row)
        stats = compute_image_statistics(image, mask)

        record = {
            "image_id": row["image_id"],
            "label": row["label"],
            "original_image_path": row["image_path"],
            "original_mask_path": row["mask_path"],
            "selected_image_path": copied_paths["selected_image_path"],
            "selected_mask_path": copied_paths["selected_mask_path"],
            **stats,
        }
        records.append(record)

    return pd.DataFrame(records)


def build_full_class_summary(characterized_subset: pd.DataFrame) -> pd.DataFrame:
    return (
        characterized_subset
        .groupby("label")
        .agg(
            n_images=("image_id", "count"),
            mean_image_mean=("image_mean", "mean"),
            std_image_mean=("image_mean", "std"),
            mean_image_min=("image_min", "mean"),
            std_image_min=("image_min", "std"),
            mean_image_max=("image_max", "mean"),
            std_image_max=("image_max", "std"),
            mean_image_std=("image_std", "mean"),
            std_image_std=("image_std", "std"),
            mean_skewness=("image_skewness", "mean"),
            std_skewness=("image_skewness", "std"),
            mean_kurtosis_excess=("image_kurtosis_excess", "mean"),
            std_kurtosis_excess=("image_kurtosis_excess", "std"),
            mean_lesion_area_px=("lesion_area_px", "mean"),
            std_lesion_area_px=("lesion_area_px", "std"),
            mean_lesion_area_fraction=("lesion_area_fraction", "mean"),
            std_lesion_area_fraction=("lesion_area_fraction", "std"),
            mean_lesion_intensity=("lesion_mean", "mean"),
            std_lesion_intensity=("lesion_mean", "std"),
            mean_background_intensity=("background_mean", "mean"),
            std_background_intensity=("background_mean", "std"),
            mean_lesion_background_contrast=("lesion_background_contrast", "mean"),
            std_lesion_background_contrast=("lesion_background_contrast", "std"),
        )
        .reset_index()
    )


def build_display_summary(characterized_subset: pd.DataFrame) -> pd.DataFrame:
    display_summary = (
        characterized_subset
        .groupby("label")
        .agg(
            n=("image_id", "count"),
            I_media=("image_mean", "mean"),
            I_min=("image_min", "mean"),
            I_max=("image_max", "mean"),
            I_desvio=("image_std", "mean"),
            asimetria=("image_skewness", "mean"),
            curtosis=("image_kurtosis_excess", "mean"),
            area_lesion_px=("lesion_area_px", "mean"),
            desvio_area_px=("lesion_area_px", "std"),
            I_lesion=("lesion_mean", "mean"),
            I_fondo=("background_mean", "mean"),
            contraste=("lesion_background_contrast", "mean"),
        )
        .reset_index()
    )

    display_summary = display_summary.rename(
        columns={
            "label": "grupo",
            "I_media": "I media",
            "I_min": "I min",
            "I_max": "I max",
            "I_desvio": "I desvio",
            "area_lesion_px": "area lesion (px)",
            "desvio_area_px": "desvio area (px)",
            "I_lesion": "I lesion",
            "I_fondo": "I fondo",
            "contraste": "contraste lesion/fondo",
        }
    )
    return display_summary


def save_summary_tables(characterized_subset, full_class_summary, display_summary) -> None:
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    # OJO: este nombre de archivo tiene que coincidir con INPUT_METADATA_PATH
    # en 03_preprocessing_comparison.py (selected_subset_metadata.csv), para
    # no tener que tocar ese script.
    characterized_subset.to_csv(OUTPUT_TABLES_DIR / "selected_subset_metadata.csv", index=False)
    full_class_summary.to_csv(OUTPUT_TABLES_DIR / "full_dataset_class_summary_full.csv", index=False)
    display_summary.to_csv(OUTPUT_TABLES_DIR / "full_dataset_class_summary_display.csv", index=False)


def print_display_summary(display_summary: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("\n==============================")
    print("RESUMEN POR GRUPO (dataset completo)")
    print("==============================")
    print("Intensidades en niveles de gris [0,255]. Areas en pixeles.\n")
    print(display_summary.round(3).to_string(index=False))


def save_display_summary_image(display_summary: pd.DataFrame) -> None:
    output_path = OUTPUT_FIGURES_DIR / "class_summary_table_full.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_df = display_summary.copy()
    numeric_cols = display_df.select_dtypes(include=["float", "int"]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(2)

    block_1_cols = ["grupo", "n", "I media", "I min", "I max", "I desvio", "asimetria", "curtosis"]
    block_2_cols = ["grupo", "area lesion (px)", "desvio area (px)", "I lesion", "I fondo", "contraste lesion/fondo"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 4.8))
    for ax, cols, title in zip(
        axes, [block_1_cols, block_2_cols],
        ["Intensidad global por grupo", "Lesion, fondo y contraste por grupo"],
    ):
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=8)
        table = ax.table(cellText=display_df[cols].values, colLabels=cols, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)

    fig.suptitle(
        "Resumen del dataset completo (benignas + malignas). Intensidades en [0,255]; areas en pixeles.",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Tabla visual guardada en: {output_path}")


def main() -> None:
    """
    Ejecuta el modulo completo.

    Correr desde la raiz del repo:
        python src/02_select_full_dataset.py
    """
    print("\nCargando metadata BUSI...")
    metadata = build_busi_metadata("data/raw")

    print("\nSeleccionando TODAS las imagenes benignas y malignas elegibles")
    print("(excluyendo imagenes sin mascara o con mascara multiple)...")
    subset = select_subset(metadata)

    print(f"\nTotal seleccionado: {len(subset)} imagenes")
    print("\nCaracterizando y copiando imagenes seleccionadas (puede tardar unos minutos)...")
    characterized_subset = characterize_subset(subset)

    print("\nConstruyendo resumen por grupo...")
    full_class_summary = build_full_class_summary(characterized_subset)
    display_summary = build_display_summary(characterized_subset)
    print_display_summary(display_summary)

    print("\nGuardando tablas...")
    save_summary_tables(characterized_subset, full_class_summary, display_summary)

    print("\nGuardando tabla visual...")
    save_display_summary_image(display_summary)

    print("\nListo.")
    print(f"Imagenes seleccionadas guardadas en: {OUTPUT_DATA_DIR}")
    print("Tablas guardadas en outputs/tables/full_dataset_*.csv")


if __name__ == "__main__":
    main()
