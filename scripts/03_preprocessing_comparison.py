"""
preprocessing_comparison.py

Modulo 02 - Comparacion de preprocesamientos para BUSI.

Este modulo recibe las imagenes seleccionadas por:
    src/select_and_characterize_subset.py

Entrada obligatoria:
    outputs/tables/selected_subset_metadata.csv

Preprocesamientos evaluados:
1. robust_bilateral:
   - normalizacion robusta por percentiles 1-99
   - filtro mediano 3x3
   - filtro bilateral d=7, sigmaColor=25, sigmaSpace=25

2. robust_nlm:
   - normalizacion robusta por percentiles 1-99
   - Non-Local Means h=10, templateWindowSize=7, searchWindowSize=21

Salidas:
1. Imagenes preprocesadas:
   data/processed/preprocessed/robust_bilateral/
   data/processed/preprocessed/robust_nlm/

2. Tablas:
   outputs/tables/preprocessing_comparison_per_image.csv
   outputs/tables/preprocessing_comparison_summary.csv

3. Tabla visual resumen:
   outputs/figures/preprocessing_comparison/preprocessing_summary_table.png

4. Figuras comparativas:
   outputs/figures/preprocessing_comparison/examples/benign/
   outputs/figures/preprocessing_comparison/examples/malignant/

Cada figura comparativa muestra, para una imagen:
- original;
- preprocesamiento bilateral;
- preprocesamiento NLM;
- imagen espacial con contorno de mascara manual;
- mascara manual;
- histograma;
- espectro de Fourier.

El modulo NO segmenta.
Las mascaras manuales se usan solo para visualizacion y para calcular contraste lesion/fondo.
"""

from pathlib import Path
from shutil import copy2
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuracion general
# ============================================================

INPUT_METADATA_PATH = Path("outputs/tables/selected_subset_metadata.csv")

OUTPUT_DATA_DIR = Path("data/processed/preprocessed")
OUTPUT_TABLES_DIR = Path("outputs/tables")
OUTPUT_FIGURES_DIR = Path("outputs/figures/preprocessing_comparison")

N_EXAMPLES_PER_GROUP = 3

# Si se pone True, ademas de guardar las figuras, las abre en pantalla.
# Recomendacion: dejar en False y abrir las imagenes guardadas desde Explorer.
SHOW_FIGURES = False


# ============================================================
# Lectura y utilidades basicas
# ============================================================

def read_grayscale_image(path: str | Path) -> np.ndarray:
    """
    Lee una imagen en escala de grises.
    """

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

    return image


def sanitize_filename(name: str) -> str:
    """
    Convierte un image_id como 'benign (68)' en un nombre seguro.
    """

    return (
        name
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convierte una mascara PNG en booleana.
    """

    return mask > 0


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Rellena huecos internos de una mascara binaria.
    """

    mask_u8 = (mask.astype(np.uint8)) * 255

    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, flood_inv)

    return filled > 0


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Conserva el mayor componente conectado de una mascara binaria.
    """

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    if n_labels <= 1:
        return np.zeros_like(mask, dtype=bool)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    return labels == largest_label


def get_valid_ultrasound_region(image: np.ndarray) -> np.ndarray:
    """
    Estima la region valida de ecografia.

    Objetivo:
    - excluir fondo negro externo;
    - no excluir lesiones oscuras internas.

    Por eso se umbraliza muy bajo, se conserva el mayor componente y luego
    se rellenan huecos internos.
    """

    valid = image > 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    valid_closed = cv2.morphologyEx(
        (valid.astype(np.uint8)) * 255,
        cv2.MORPH_CLOSE,
        kernel,
    ) > 0

    valid_largest = keep_largest_component(valid_closed)

    if valid_largest.sum() == 0:
        valid_largest = valid_closed

    valid_filled = fill_holes(valid_largest)

    return valid_filled


# ============================================================
# Preprocesamientos
# ============================================================

def robust_normalization(
    image: np.ndarray,
    p_low: float = 1,
    p_high: float = 99,
) -> np.ndarray:
    """
    Normalizacion robusta por percentiles.

    En lugar de usar minimo y maximo absolutos, usa P1 y P99.
    Esto reduce el efecto de outliers muy negros o muy brillantes.

    Salida: uint8 en [0, 255].
    """

    image_float = image.astype(np.float32)

    low = np.percentile(image_float, p_low)
    high = np.percentile(image_float, p_high)

    if high <= low:
        return image.copy()

    normalized = (image_float - low) / (high - low)
    normalized = np.clip(normalized, 0, 1)

    return (255 * normalized).astype(np.uint8)


def preprocess_robust_bilateral(image: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento 1:
    normalizacion robusta + mediana 3x3 + bilateral.

    Hiperparametros:
    - mediana 3x3: remueve puntos aislados sin deformar demasiado bordes.
    - bilateral d=7, sigmaColor=25, sigmaSpace=25: suavizado local moderado
      con preservacion de bordes.
    """

    normalized = robust_normalization(image, p_low=1, p_high=99)

    median = cv2.medianBlur(normalized, ksize=3)

    bilateral = cv2.bilateralFilter(
        median,
        d=7,
        sigmaColor=25,
        sigmaSpace=25,
    )

    return bilateral


def preprocess_robust_nlm(image: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento 2:
    normalizacion robusta + Non-Local Means.

    Hiperparametros:
    - h=10: fuerza de filtrado moderada.
    - templateWindowSize=7: compara parches locales suficientemente informativos.
    - searchWindowSize=21: busca parches similares en una vecindad razonable.
    """

    normalized = robust_normalization(image, p_low=1, p_high=99)

    denoised = cv2.fastNlMeansDenoising(
        normalized,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    return denoised


# ============================================================
# Indices cuantitativos
# ============================================================

def compute_speckle_index(image: np.ndarray, valid_region: np.ndarray) -> float:
    """
    Estima un indice simple de speckle:

        SI = std(I) / mean(I)

    Se calcula dentro de la region valida de ecografia para evitar fondo negro.
    """

    values = image[valid_region].astype(np.float32)

    if values.size == 0:
        return np.nan

    mean = float(np.mean(values))
    std = float(np.std(values))

    if mean == 0:
        return np.nan

    return std / mean


def compute_high_frequency_energy_ratio(image: np.ndarray) -> float:
    """
    Calcula proporcion de energia en altas frecuencias.

    Procedimiento:
    - FFT 2D.
    - Energia total = suma(|F|^2).
    - Energia alta = energia fuera de un radio central.
    """

    image_float = image.astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(image_float))
    magnitude_squared = np.abs(spectrum) ** 2

    h, w = image.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    cutoff = 0.25 * min(h, w)
    high_freq_mask = radius > cutoff

    total_energy = float(np.sum(magnitude_squared))

    if total_energy == 0:
        return np.nan

    high_freq_energy = float(np.sum(magnitude_squared[high_freq_mask]))

    return high_freq_energy / total_energy


def compute_lesion_background_contrast(
    image: np.ndarray,
    mask: np.ndarray,
    valid_region: np.ndarray,
) -> float:
    """
    Calcula contraste relativo lesion/fondo:

        C = (mu_fondo - mu_lesion) / (mu_fondo + eps)

    Fondo = region valida de ecografia fuera de la mascara manual.
    """

    mask_bool = binarize_mask(mask)

    lesion_region = np.logical_and(mask_bool, valid_region)
    background_region = np.logical_and(~mask_bool, valid_region)

    if lesion_region.sum() == 0 or background_region.sum() == 0:
        return np.nan

    image_float = image.astype(np.float32)

    lesion_mean = float(np.mean(image_float[lesion_region]))
    background_mean = float(np.mean(image_float[background_region]))

    if background_mean == 0:
        return np.nan

    return (background_mean - lesion_mean) / (background_mean + 1e-8)


def compute_metrics_for_image(
    image: np.ndarray,
    mask: np.ndarray,
    valid_region: np.ndarray,
) -> Dict[str, float]:
    """
    Calcula los tres indices pedidos para una imagen.
    """

    return {
        "speckle_index": compute_speckle_index(image, valid_region),
        "high_frequency_energy": compute_high_frequency_energy_ratio(image),
        "lesion_background_contrast": compute_lesion_background_contrast(
            image=image,
            mask=mask,
            valid_region=valid_region,
        ),
    }


def compute_log_fourier_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Devuelve espectro logaritmico de Fourier normalizado para visualizacion.
    """

    image_float = image.astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(image_float))
    magnitude = np.log1p(np.abs(spectrum))

    magnitude -= magnitude.min()

    if magnitude.max() > 0:
        magnitude /= magnitude.max()

    return magnitude


# ============================================================
# Guardado de imagenes y figuras
# ============================================================

def overlay_contour(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Superpone el contorno de la mascara manual sobre la imagen.
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    contours, _ = cv2.findContours(
        (binarize_mask(mask).astype(np.uint8)) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(rgb, contours, -1, (255, 0, 0), thickness=2)

    return rgb


def save_preprocessed_image_and_mask(
    image: np.ndarray,
    mask_path: str | Path,
    image_id: str,
    label: str,
    method: str,
) -> Dict[str, str]:
    """
    Guarda imagen preprocesada y copia la mascara manual asociada.
    """

    safe_id = sanitize_filename(image_id)

    image_dir = OUTPUT_DATA_DIR / method / label / "images"
    mask_dir = OUTPUT_DATA_DIR / method / label / "masks"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_output_path = image_dir / f"{safe_id}.png"
    mask_output_path = mask_dir / f"{safe_id}_mask.png"

    cv2.imwrite(str(image_output_path), image)
    copy2(mask_path, mask_output_path)

    return {
        f"{method}_image_path": str(image_output_path),
        f"{method}_mask_path": str(mask_output_path),
    }


def save_comparison_figure_for_one_image(
    raw_image: np.ndarray,
    bilateral_image: np.ndarray,
    nlm_image: np.ndarray,
    mask: np.ndarray,
    label: str,
    image_id: str,
    output_path: Path,
) -> None:
    """
    Guarda una figura comparativa completa para una imagen.

    Filas:
    - Original
    - Bilateral
    - NLM

    Columnas:
    - Imagen espacial con contorno de mascara manual.
    - Mascara manual.
    - Histograma.
    - Fourier.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = [
        ("Original", raw_image),
        ("Bilateral", bilateral_image),
        ("NLM", nlm_image),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    for row_idx, (method_label, image) in enumerate(images):
        axes[row_idx, 0].imshow(overlay_contour(image, mask))
        axes[row_idx, 0].set_title(f"{method_label}: espacial + mascara")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(binarize_mask(mask), cmap="gray")
        axes[row_idx, 1].set_title("Mascara manual")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].hist(image.ravel(), bins=64)
        axes[row_idx, 2].set_title(f"{method_label}: histograma")
        axes[row_idx, 2].set_xlabel("Intensidad")
        axes[row_idx, 2].set_ylabel("Frecuencia")

        axes[row_idx, 3].imshow(compute_log_fourier_spectrum(image), cmap="gray")
        axes[row_idx, 3].set_title(f"{method_label}: Fourier")
        axes[row_idx, 3].axis("off")

    fig.suptitle(f"{label} - {image_id}", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_group_overview_figure(
    rows: pd.DataFrame,
    group_label: str,
    output_path: Path,
) -> None:
    """
    Guarda una figura resumen por grupo con 3 imagenes.

    Para cada imagen seleccionada se muestran tres columnas:
    - Original con mascara manual.
    - Bilateral con mascara manual.
    - NLM con mascara manual.

    Esto permite ver directamente antes/despues lado a lado.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.2 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (_, row) in enumerate(rows.iterrows()):
        image_id = row["image_id"]

        raw_image = read_grayscale_image(row["selected_image_path"])
        mask = read_grayscale_image(row["selected_mask_path"])
        bilateral_image = preprocess_robust_bilateral(raw_image)
        nlm_image = preprocess_robust_nlm(raw_image)

        panels = [
            ("Original", raw_image),
            ("Bilateral", bilateral_image),
            ("NLM", nlm_image),
        ]

        for col_idx, (title, image) in enumerate(panels):
            axes[row_idx, col_idx].imshow(overlay_contour(image, mask))
            axes[row_idx, col_idx].set_title(f"{image_id} - {title}")
            axes[row_idx, col_idx].axis("off")

    fig.suptitle(f"Comparacion espacial antes/despues - {group_label}", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_summary_table_image(summary: pd.DataFrame) -> None:
    """
    Guarda la tabla resumen como imagen PNG legible.
    """

    output_path = OUTPUT_FIGURES_DIR / "preprocessing_summary_table.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_df = summary.copy()

    method_names = {
        "raw": "Original",
        "robust_bilateral": "Bilateral",
        "robust_nlm": "NLM",
    }

    group_names = {
        "benign": "Benigno",
        "malignant": "Maligno",
    }

    display_df["Grupo"] = display_df["label"].map(group_names)
    display_df["Metodo"] = display_df["method"].map(method_names)

    table_noise = display_df[
        [
            "Grupo",
            "Metodo",
            "n_images",
            "mean_speckle",
            "std_speckle",
            "mean_high_freq_energy",
            "std_high_freq_energy",
        ]
    ].copy()

    table_noise = table_noise.rename(
        columns={
            "n_images": "n",
            "mean_speckle": "Speckle",
            "std_speckle": "Desvio speckle",
            "mean_high_freq_energy": "Energia alta freq.",
            "std_high_freq_energy": "Desvio energia",
        }
    )

    table_contrast = display_df[
        [
            "Grupo",
            "Metodo",
            "n_images",
            "mean_lesion_background_contrast",
            "std_lesion_background_contrast",
        ]
    ].copy()

    table_contrast = table_contrast.rename(
        columns={
            "n_images": "n",
            "mean_lesion_background_contrast": "Contraste lesion/fondo",
            "std_lesion_background_contrast": "Desvio contraste",
        }
    )

    for df in [table_noise, table_contrast]:
        numeric_cols = df.select_dtypes(include=["float", "int"]).columns
        df[numeric_cols] = df[numeric_cols].round(4)

    fig, axes = plt.subplots(2, 1, figsize=(13, 5.8))

    tables = [
        (table_noise, "Ruido y contenido de alta frecuencia"),
        (table_contrast, "Contraste relativo lesion/fondo"),
    ]

    for ax, (df, title) in zip(axes, tables):
        ax.axis("off")
        ax.set_title(title, fontsize=11, pad=8)

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.35)

    fig.suptitle(
        "Comparacion de preprocesamientos por grupo",
        fontsize=12,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Tabla visual guardada en: {output_path}")


# ============================================================
# Procesamiento completo
# ============================================================

def process_selected_subset(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica ambos preprocesamientos a todas las imagenes seleccionadas.
    """

    records = []

    for _, row in metadata.iterrows():
        image_id = row["image_id"]
        label = row["label"]

        raw_image = read_grayscale_image(row["selected_image_path"])
        mask = read_grayscale_image(row["selected_mask_path"])

        valid_region_raw = get_valid_ultrasound_region(raw_image)

        bilateral_image = preprocess_robust_bilateral(raw_image)
        nlm_image = preprocess_robust_nlm(raw_image)

        preprocessed_images = {
            "raw": raw_image,
            "robust_bilateral": bilateral_image,
            "robust_nlm": nlm_image,
        }

        for method, image in preprocessed_images.items():
            metrics = compute_metrics_for_image(
                image=image,
                mask=mask,
                valid_region=valid_region_raw,
            )

            output_paths = {}

            if method != "raw":
                output_paths = save_preprocessed_image_and_mask(
                    image=image,
                    mask_path=row["selected_mask_path"],
                    image_id=image_id,
                    label=label,
                    method=method,
                )

            record = {
                "image_id": image_id,
                "label": label,
                "method": method,
                **metrics,
                **output_paths,
            }

            records.append(record)

    return pd.DataFrame(records)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Resume indices promedio por grupo y metodo.
    """

    summary = (
        results
        .groupby(["label", "method"])
        .agg(
            n_images=("image_id", "count"),
            mean_speckle=("speckle_index", "mean"),
            std_speckle=("speckle_index", "std"),
            mean_high_freq_energy=("high_frequency_energy", "mean"),
            std_high_freq_energy=("high_frequency_energy", "std"),
            mean_lesion_background_contrast=("lesion_background_contrast", "mean"),
            std_lesion_background_contrast=("lesion_background_contrast", "std"),
        )
        .reset_index()
    )

    return summary


def print_summary(summary: pd.DataFrame) -> None:
    """
    Imprime resumen en consola.
    """

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n===============================================")
    print("RESUMEN DE PREPROCESAMIENTO POR GRUPO Y METODO")
    print("===============================================\n")

    print(summary.to_string(index=False))


def save_example_figures(metadata: pd.DataFrame) -> None:
    """
    Guarda las figuras exigidas para inspeccion visual.

    Para cada grupo guarda:
    1. Una figura general con 3 imagenes del grupo:
       Original vs Bilateral vs NLM lado a lado.
    2. Una figura detallada por cada una de esas 3 imagenes:
       espacial + mascara, mascara manual, histograma y Fourier.
    """

    examples = (
        metadata
        .groupby("label", group_keys=False)
        .head(N_EXAMPLES_PER_GROUP)
        .reset_index(drop=True)
    )

    for label, group_rows in examples.groupby("label"):
        group_overview_path = (
            OUTPUT_FIGURES_DIR
            / "examples"
            / label
            / f"{label}_overview_original_vs_preprocessed.png"
        )

        save_group_overview_figure(
            rows=group_rows,
            group_label=label,
            output_path=group_overview_path,
        )

        for _, row in group_rows.iterrows():
            image_id = row["image_id"]

            raw_image = read_grayscale_image(row["selected_image_path"])
            mask = read_grayscale_image(row["selected_mask_path"])

            bilateral_image = preprocess_robust_bilateral(raw_image)
            nlm_image = preprocess_robust_nlm(raw_image)

            safe_id = sanitize_filename(image_id)

            detailed_output_path = (
                OUTPUT_FIGURES_DIR
                / "examples"
                / label
                / f"{safe_id}_spatial_hist_fourier_mask.png"
            )

            save_comparison_figure_for_one_image(
                raw_image=raw_image,
                bilateral_image=bilateral_image,
                nlm_image=nlm_image,
                mask=mask,
                label=label,
                image_id=image_id,
                output_path=detailed_output_path,
            )


def save_results(results: pd.DataFrame, summary: pd.DataFrame) -> None:
    """
    Guarda tablas de resultados.
    """

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_TABLES_DIR / "preprocessing_comparison_per_image.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_TABLES_DIR / "preprocessing_comparison_summary.csv",
        index=False,
    )


def main() -> None:
    """
    Ejecuta el modulo completo.

    Correr desde la raiz del repo:

        python src/preprocessing_comparison.py
    """

    if not INPUT_METADATA_PATH.exists():
        raise FileNotFoundError(
            "No existe outputs/tables/selected_subset_metadata.csv. "
            "Primero ejecuta: python src/select_and_characterize_subset.py"
        )

    print("\nCargando subset seleccionado...")
    metadata = pd.read_csv(INPUT_METADATA_PATH)

    print("\nAplicando preprocesamientos a imagenes seleccionadas...")
    results = process_selected_subset(metadata)

    print("\nConstruyendo resumen por grupo y metodo...")
    summary = build_summary(results)

    print_summary(summary)

    print("\nGuardando tablas...")
    save_results(results, summary)

    print("\nGuardando tabla visual del resumen...")
    save_summary_table_image(summary)

    print("\nGuardando figuras comparativas exigidas...")
    save_example_figures(metadata)

    print("\nListo.")
    print("Imagenes preprocesadas guardadas en:")
    print("  data/processed/preprocessed/robust_bilateral/")
    print("  data/processed/preprocessed/robust_nlm/")
    print("Tablas guardadas en:")
    print("  outputs/tables/preprocessing_comparison_per_image.csv")
    print("  outputs/tables/preprocessing_comparison_summary.csv")
    print("Tabla visual guardada en:")
    print("  outputs/figures/preprocessing_comparison/preprocessing_summary_table.png")
    print("Figuras comparativas guardadas en:")
    print("  outputs/figures/preprocessing_comparison/examples/")


if __name__ == "__main__":
    main()
