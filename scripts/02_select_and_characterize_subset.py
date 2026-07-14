"""
select_and_characterize_subset.py

Modulo 01 - Seleccion y caracterizacion inicial del subset BUSI.

Objetivo:
1. Cargar la metadata del dataset BUSI usando data_loading.py.
2. Seleccionar al azar:
   - 25 imagenes benignas
   - 15 imagenes malignas
3. Excluir imagenes con mas de una mascara asociada.
4. Copiar las imagenes seleccionadas y sus mascaras a una carpeta propia.
5. Caracterizar las imagenes seleccionadas.
6. Guardar tablas completas.
7. Guardar una tabla visual legible con las estadisticas principales por grupo.

Este modulo es previo al preprocesamiento.
No modifica intensidades.
No segmenta.
No aplica filtros.
No genera histogramas.
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

N_BENIGN = 25
N_MALIGNANT = 15
RANDOM_STATE = 42

OUTPUT_DATA_DIR = Path("data/selected/busi_subset")
OUTPUT_TABLES_DIR = Path("outputs/tables")
OUTPUT_FIGURES_DIR = Path("outputs/figures/selected_subset")


# ============================================================
# Utilidades
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
    """
    Calcula el coeficiente de asimetria.

    skewness = E[((X - mu) / sigma)^3]
    """

    values = values.astype(np.float32).ravel()
    std = float(np.std(values))

    if std == 0:
        return 0.0

    mean = float(np.mean(values))
    z = (values - mean) / std

    return float(np.mean(z ** 3))


def compute_kurtosis(values: np.ndarray) -> float:
    """
    Calcula la curtosis en exceso.

    kurtosis_excess = E[((X - mu) / sigma)^4] - 3
    """

    values = values.astype(np.float32).ravel()
    std = float(np.std(values))

    if std == 0:
        return 0.0

    mean = float(np.mean(values))
    z = (values - mean) / std

    return float(np.mean(z ** 4) - 3.0)


def compute_image_statistics(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Calcula estadisticas de intensidad y de mascara para una imagen.

    Las intensidades estan en niveles de gris del PNG, es decir, entre 0 y 255.

    Estadisticas globales:
    - media, minimo, maximo, desvio
    - mediana
    - percentiles 5 y 95
    - rango dinamico
    - asimetria
    - curtosis en exceso

    Estadisticas usando mascara:
    - area de lesion en pixeles
    - fraccion de area
    - intensidad media dentro de lesion
    - intensidad media fuera de lesion
    - contraste relativo lesion/fondo
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

    Estructura de salida:

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
# Seleccion y caracterizacion
# ============================================================

def select_subset(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona 25 benignas y 15 malignas.

    Criterios:
    - label en {benign, malignant}
    - debe tener mascara
    - debe tener exactamente una mascara asociada
    """

    eligible = metadata[
        metadata["label"].isin(["benign", "malignant"])
        & (metadata["has_mask"] == True)
        & (metadata["n_masks"] == 1)
    ].copy()

    n_benign_available = int((eligible["label"] == "benign").sum())
    n_malignant_available = int((eligible["label"] == "malignant").sum())

    if n_benign_available < N_BENIGN:
        raise ValueError(
            f"No hay suficientes benignas elegibles: {n_benign_available} disponibles, {N_BENIGN} requeridas."
        )

    if n_malignant_available < N_MALIGNANT:
        raise ValueError(
            f"No hay suficientes malignas elegibles: {n_malignant_available} disponibles, {N_MALIGNANT} requeridas."
        )

    benign = eligible[eligible["label"] == "benign"].sample(
        n=N_BENIGN,
        random_state=RANDOM_STATE,
    )

    malignant = eligible[eligible["label"] == "malignant"].sample(
        n=N_MALIGNANT,
        random_state=RANDOM_STATE,
    )

    subset = pd.concat([benign, malignant], ignore_index=True)

    subset = subset.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return subset


def characterize_subset(subset: pd.DataFrame) -> pd.DataFrame:
    """
    Copia imagenes/mascaras seleccionadas y calcula estadisticas por imagen.
    """

    records = []

    for _, row in subset.iterrows():
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
    """
    Construye un resumen completo por grupo.

    Esta tabla se guarda como CSV para trazabilidad, pero no se usa como
    tabla visual principal porque tiene demasiadas columnas.
    """

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
    """
    Construye la tabla resumida que se imprime y se guarda como imagen.

    La tabla usa titulos cortos y claros.
    Todas las intensidades estan en niveles de gris [0, 255].
    Las areas estan en pixeles, porque BUSI no incluye escala fisica uniforme
    en mm/pixel en los PNG del dataset.
    """

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
            "n": "n",
            "I_media": "I media",
            "I_min": "I min",
            "I_max": "I max",
            "I_desvio": "I desvio",
            "asimetria": "asimetria",
            "curtosis": "curtosis",
            "area_lesion_px": "area lesion (px)",
            "desvio_area_px": "desvio area (px)",
            "I_lesion": "I lesion",
            "I_fondo": "I fondo",
            "contraste": "contraste lesion/fondo",
        }
    )

    return display_summary


def save_summary_tables(
    characterized_subset: pd.DataFrame,
    full_class_summary: pd.DataFrame,
    display_summary: pd.DataFrame,
) -> None:
    """
    Guarda:
    1. Tabla completa por imagen seleccionada.
    2. Resumen completo agrupado por clase.
    3. Resumen reducido agrupado por clase.
    """

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    characterized_subset.to_csv(
        OUTPUT_TABLES_DIR / "selected_subset_metadata.csv",
        index=False,
    )

    full_class_summary.to_csv(
        OUTPUT_TABLES_DIR / "selected_subset_class_summary_full.csv",
        index=False,
    )

    display_summary.to_csv(
        OUTPUT_TABLES_DIR / "selected_subset_class_summary_display.csv",
        index=False,
    )


def print_display_summary(display_summary: pd.DataFrame) -> None:
    """
    Imprime en consola la tabla resumida.
    """

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("\n==============================")
    print("RESUMEN POR GRUPO")
    print("==============================")
    print("Intensidades en niveles de gris [0,255]. Areas en pixeles.\n")

    print(display_summary.round(3).to_string(index=False))


def save_display_summary_image(display_summary: pd.DataFrame) -> None:
    """
    Guarda el resumen reducido como imagen PNG legible.

    La tabla se parte en dos bloques para evitar celdas largas e ilegibles.
    """

    output_path = OUTPUT_FIGURES_DIR / "class_summary_table.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_df = display_summary.copy()

    numeric_cols = display_df.select_dtypes(include=["float", "int"]).columns
    display_df[numeric_cols] = display_df[numeric_cols].round(2)

    block_1_cols = [
        "grupo",
        "n",
        "I media",
        "I min",
        "I max",
        "I desvio",
        "asimetria",
        "curtosis",
    ]

    block_2_cols = [
        "grupo",
        "area lesion (px)",
        "desvio area (px)",
        "I lesion",
        "I fondo",
        "contraste lesion/fondo",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(15, 4.8))

    for ax, cols, title in zip(
        axes,
        [block_1_cols, block_2_cols],
        [
            "Intensidad global por grupo",
            "Lesion, fondo y contraste por grupo",
        ],
    ):
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=8)

        table = ax.table(
            cellText=display_df[cols].values,
            colLabels=cols,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)

    fig.suptitle(
        "Resumen del subset seleccionado. Intensidades en [0,255]; areas en pixeles.",
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

        python src/select_and_characterize_subset.py
    """

    print("\nCargando metadata BUSI...")
    metadata = build_busi_metadata("data/raw")

    print("\nSeleccionando subset:")
    print(f"- {N_BENIGN} benignas")
    print(f"- {N_MALIGNANT} malignas")
    print("- excluyendo imagenes con mas de una mascara")

    subset = select_subset(metadata)

    print("\nCaracterizando y copiando imagenes seleccionadas...")
    characterized_subset = characterize_subset(subset)

    print("\nConstruyendo resumen por grupo...")
    full_class_summary = build_full_class_summary(characterized_subset)
    display_summary = build_display_summary(characterized_subset)

    print_display_summary(display_summary)

    print("\nGuardando tablas...")
    save_summary_tables(
        characterized_subset=characterized_subset,
        full_class_summary=full_class_summary,
        display_summary=display_summary,
    )

    print("\nGuardando tabla visual...")
    save_display_summary_image(display_summary)

    print("\nListo.")
    print("Imagenes seleccionadas guardadas en:")
    print(f"  {OUTPUT_DATA_DIR}")
    print("Tablas guardadas en:")
    print("  outputs/tables/selected_subset_metadata.csv")
    print("  outputs/tables/selected_subset_class_summary_full.csv")
    print("  outputs/tables/selected_subset_class_summary_display.csv")
    print("Tabla visual guardada en:")
    print("  outputs/figures/selected_subset/class_summary_table.png")


if __name__ == "__main__":
    main()
