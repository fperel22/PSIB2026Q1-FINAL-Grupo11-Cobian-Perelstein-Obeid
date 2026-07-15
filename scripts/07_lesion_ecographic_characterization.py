"""
06_lesion_ecographic_characterization.py

Modulo 06 - Caracterizacion ecografica de la lesion.

Entrada esperada:
    outputs/segmentation_gui/<grupo>/<imagen>/<metodo_timestamp>/
        image_preprocessed.png
        manual_mask.png
        predicted_mask.png
        metrics_and_parameters.json

Calcula, para mascara manual y mascara predicha:
- intensidad relativa lesion/fondo
- irregularidad del borde
- orientacion de la lesion, como relacion ancho/alto del bounding box
- intensidad media en banda perilesional
- diferencia de intensidad media entre banda perilesional y lesion

Banda perilesional:
    B_peri = dilate(M, r) - M

donde:
    D_eq = 2 * sqrt(A / pi)
    r = max(5, min(20, 0.10 * D_eq))

Salidas:
    outputs/tables/06_characterization_manual_per_image.csv
    outputs/tables/06_characterization_predicted_per_image.csv
    outputs/tables/06_characterization_manual_summary.csv
    outputs/tables/06_characterization_predicted_summary.csv
    outputs/figures/06_characterization_manual_summary.png
    outputs/figures/06_characterization_predicted_summary.png

Ejecucion:
    python src/06_lesion_ecographic_characterization.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEGMENTATION_ROOT = Path("outputs/segmentation_gui")
OUTPUT_TABLES_DIR = Path("outputs/tables")
OUTPUT_FIGURES_DIR = Path("outputs/figures")
EPS = 1e-8


def read_grayscale_image(path: str | Path) -> np.ndarray:
    """Lee imagen en escala de grises soportando rutas Unicode de Windows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return image


def discover_segmentation_cases(root: Path) -> List[Path]:
    cases = []
    for predicted_path in root.rglob("predicted_mask.png"):
        case_dir = predicted_path.parent
        required = [case_dir / "image_preprocessed.png", case_dir / "manual_mask.png", case_dir / "predicted_mask.png"]
        if all(p.exists() for p in required):
            cases.append(case_dir)
    return sorted(cases)


def infer_group(case_dir: Path) -> str:
    parts = [p.lower() for p in case_dir.parts]
    if "benign" in parts:
        return "benign"
    if "malignant" in parts:
        return "malignant"
    return "unknown"


def get_image_id(case_dir: Path) -> str:
    return case_dir.parent.name


def ensure_same_shape(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8)) * 255
    if mask_u8.max() == 0:
        return mask.astype(bool)
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, holes)
    return filled > 0


def get_valid_ultrasound_region(image: np.ndarray) -> np.ndarray:
    """Region valida de ecografia: elimina fondo negro externo y rellena huecos."""
    valid = image > 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx((valid.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, kernel) > 0
    largest = keep_largest_component(closed)
    if largest.sum() == 0:
        largest = closed
    return fill_binary_holes(largest)


def equivalent_diameter(mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    return float(2.0 * np.sqrt(area / np.pi))


def perilesional_radius(mask: np.ndarray) -> int:
    d_eq = equivalent_diameter(mask)
    radius = max(5.0, min(20.0, 0.10 * d_eq))
    return int(round(radius))


def get_perilesional_band(mask: np.ndarray, valid_region: np.ndarray) -> Tuple[np.ndarray, int]:
    mask_bool = mask.astype(bool)
    radius = perilesional_radius(mask_bool)
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0
    band = dilated & (~mask_bool) & valid_region
    return band, radius


def contour_perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours((mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def bounding_box_aspect_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.nan
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    if height <= 0:
        return np.nan
    return float(width / height)


def compute_case_characterization(image: np.ndarray, mask: np.ndarray, group: str, image_id: str, case_dir: Path, mask_type: str) -> Dict[str, object]:
    mask_bool = mask > 0
    valid_region = get_valid_ultrasound_region(image)
    background = valid_region & (~mask_bool)
    band, band_radius = get_perilesional_band(mask_bool, valid_region)

    if mask_bool.sum() == 0:
        lesion_mean = np.nan
    else:
        lesion_mean = float(np.mean(image[mask_bool]))

    if background.sum() == 0:
        background_mean = np.nan
    else:
        background_mean = float(np.mean(image[background]))

    if band.sum() == 0:
        perilesional_mean = np.nan
    else:
        perilesional_mean = float(np.mean(image[band]))

    relative_intensity_lesion_background = (background_mean - lesion_mean) / (background_mean + EPS) if pd.notna(background_mean) and pd.notna(lesion_mean) else np.nan
    perilesional_minus_lesion = perilesional_mean - lesion_mean if pd.notna(perilesional_mean) and pd.notna(lesion_mean) else np.nan

    area = float(mask_bool.sum())
    perimeter = contour_perimeter(mask_bool)
    circularity = (4.0 * np.pi * area) / (perimeter ** 2 + EPS) if area > 0 and perimeter > 0 else np.nan
    irregularity = 1.0 - circularity if pd.notna(circularity) else np.nan
    aspect_ratio_width_height = bounding_box_aspect_ratio(mask_bool)

    return {
        "group": group,
        "image_id": image_id,
        "case_dir": str(case_dir),
        "mask_type": mask_type,
        "area_px": area,
        "perimeter_px": perimeter,
        "lesion_mean_intensity": lesion_mean,
        "background_mean_intensity": background_mean,
        "relative_intensity_lesion_background": relative_intensity_lesion_background,
        "irregularity_1_minus_circularity": irregularity,
        "circularity": circularity,
        "orientation_width_height_ratio": aspect_ratio_width_height,
        "perilesional_radius_px": band_radius,
        "perilesional_mean_intensity": perilesional_mean,
        "perilesional_minus_lesion_mean_intensity": perilesional_minus_lesion,
    }


def summarize(per_image: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "relative_intensity_lesion_background",
        "irregularity_1_minus_circularity",
        "orientation_width_height_ratio",
        "perilesional_mean_intensity",
        "perilesional_minus_lesion_mean_intensity",
    ]
    if per_image.empty:
        return pd.DataFrame()

    summary = per_image.groupby("group", dropna=False).agg(
        n_images=("image_id", "nunique"),
        **{f"{f}_mean": (f, "mean") for f in feature_cols},
        **{f"{f}_std": (f, "std") for f in feature_cols},
    ).reset_index()
    return summary


def save_summary_figure(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    """
    Guarda una figura tipo tabla con nombres de columnas compactos.

    La tabla CSV conserva los nombres completos de las variables. Esta funcion
    solo acorta las etiquetas visibles de la figura para evitar que se corten.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return

    feature_labels = {
        "relative_intensity_lesion_background": "I rel.\nlesion/fondo",
        "irregularity_1_minus_circularity": "Irregularidad\n1 - circularidad",
        "orientation_width_height_ratio": "Orientacion\nancho/alto",
        "perilesional_mean_intensity": "I media\nperilesional",
        "perilesional_minus_lesion_mean_intensity": "Delta I\nperi - lesion",
    }

    compact = pd.DataFrame()
    compact["grupo"] = summary["group"]
    compact["n"] = summary["n_images"]

    for feature, label in feature_labels.items():
        compact[label] = (
            summary[f"{feature}_mean"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
            + " ± "
            + summary[f"{feature}_std"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
        )

    fig_width = 15
    fig_height = max(3.2, 0.65 * len(compact) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    table = ax.table(
        cellText=compact.values,
        colLabels=compact.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.45)

    # Ajuste extra para que las columnas largas no invadan columnas vecinas.
    try:
        table.auto_set_column_width(col=list(range(len(compact.columns))))
    except Exception:
        pass

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_for_mask_type(cases: Iterable[Path], mask_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask_filename = "manual_mask.png" if mask_type == "manual" else "predicted_mask.png"
    records = []
    for case_dir in cases:
        image_path = case_dir / "image_preprocessed.png"
        mask_path = case_dir / mask_filename
        if not image_path.exists() or not mask_path.exists():
            continue
        image = read_grayscale_image(image_path)
        mask = ensure_same_shape(read_grayscale_image(mask_path), image.shape)
        records.append(compute_case_characterization(image, mask, infer_group(case_dir), get_image_id(case_dir), case_dir, mask_type))
    per_image = pd.DataFrame(records)
    summary = summarize(per_image)
    return per_image, summary


def main() -> None:
    cases = discover_segmentation_cases(SEGMENTATION_ROOT)
    if not cases:
        raise FileNotFoundError(f"No se encontraron resultados de segmentacion en {SEGMENTATION_ROOT}.")

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for mask_type in ["manual", "predicted"]:
        per_image, summary = run_for_mask_type(cases, mask_type)
        per_image_path = OUTPUT_TABLES_DIR / f"06_characterization_{mask_type}_per_image.csv"
        summary_path = OUTPUT_TABLES_DIR / f"06_characterization_{mask_type}_summary.csv"
        figure_path = OUTPUT_FIGURES_DIR / f"06_characterization_{mask_type}_summary.png"
        per_image.to_csv(per_image_path, index=False)
        summary.to_csv(summary_path, index=False)
        save_summary_figure(summary, figure_path, f"Modulo 06 - Caracterizacion ecografica usando mascara {mask_type}")

        print(f"\nMascara: {mask_type}")
        print(f"Tabla por imagen: {per_image_path}")
        print(f"Tabla resumen: {summary_path}")
        print(f"Figura resumen: {figure_path}")

    print("\nListo.")


if __name__ == "__main__":
    main()
