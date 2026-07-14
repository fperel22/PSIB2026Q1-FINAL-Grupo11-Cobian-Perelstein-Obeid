"""
05_texture_glcm_features.py

Modulo 05 - Texturas GLCM dentro de la lesion.

Entrada esperada:
    outputs/segmentation_gui/<grupo>/<imagen>/<metodo_timestamp>/
        image_preprocessed.png
        manual_mask.png
        predicted_mask.png
        metrics_and_parameters.json

Calcula, para mascara manual y mascara predicha:
- homogeneidad / IDM
- contraste
- entropia
- correlacion

Configuracion GLCM:
- distancias: 1 y 4 pixeles
- angulos: 0 y 90 grados
- cuantizacion: 32 niveles de gris
- pares considerados: solo pares donde ambos pixeles pertenecen a la mascara

Salidas:
    outputs/tables/05_glcm_manual_per_image.csv
    outputs/tables/05_glcm_predicted_per_image.csv
    outputs/tables/05_glcm_manual_summary.csv
    outputs/tables/05_glcm_predicted_summary.csv
    outputs/figures/05_glcm_manual_summary.png
    outputs/figures/05_glcm_predicted_summary.png

Ejecucion:
    python src/05_texture_glcm_features.py
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

DISTANCES = [1, 4]
ANGLES_DEGREES = [0, 90]
N_GRAY_LEVELS = 32
EPS = 1e-12


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
    """Encuentra carpetas que contienen una segmentacion guardada."""
    cases = []
    for predicted_path in root.rglob("predicted_mask.png"):
        case_dir = predicted_path.parent
        required = [
            case_dir / "image_preprocessed.png",
            case_dir / "manual_mask.png",
            case_dir / "predicted_mask.png",
        ]
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


def quantize_image(image: np.ndarray, n_levels: int = N_GRAY_LEVELS) -> np.ndarray:
    q = np.floor(image.astype(np.float32) * n_levels / 256.0)
    q = np.clip(q, 0, n_levels - 1)
    return q.astype(np.uint8)


def angle_to_offset(distance: int, angle_degrees: int) -> Tuple[int, int]:
    if angle_degrees == 0:
        return 0, distance
    if angle_degrees == 90:
        return distance, 0
    raise ValueError("Este modulo solo usa angulos 0 y 90 grados.")


def masked_glcm(image_q: np.ndarray, mask: np.ndarray, distance: int, angle_degrees: int, n_levels: int = N_GRAY_LEVELS) -> np.ndarray:
    """GLCM normalizada usando solo pares donde ambos pixeles estan dentro de la mascara."""
    mask_bool = mask.astype(bool)
    h, w = image_q.shape
    dy, dx = angle_to_offset(distance, angle_degrees)

    image_a = image_q[0:h - dy, 0:w - dx]
    image_b = image_q[dy:h, dx:w]
    mask_a = mask_bool[0:h - dy, 0:w - dx]
    mask_b = mask_bool[dy:h, dx:w]
    valid = mask_a & mask_b

    glcm = np.zeros((n_levels, n_levels), dtype=np.float64)
    if valid.sum() == 0:
        return glcm

    a = image_a[valid].ravel()
    b = image_b[valid].ravel()
    np.add.at(glcm, (a, b), 1)
    np.add.at(glcm, (b, a), 1)  # matriz simetrica

    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def glcm_features(glcm: np.ndarray) -> Dict[str, float]:
    if glcm.sum() <= 0:
        return {"homogeneity_idm": np.nan, "contrast": np.nan, "entropy": np.nan, "correlation": np.nan}

    n = glcm.shape[0]
    i, j = np.indices((n, n))

    contrast = float(np.sum(((i - j) ** 2) * glcm))
    homogeneity = float(np.sum(glcm / (1.0 + (i - j) ** 2)))
    entropy = float(-np.sum(glcm * np.log2(glcm + EPS)))

    px = glcm.sum(axis=1)
    py = glcm.sum(axis=0)
    levels = np.arange(n, dtype=np.float64)
    mean_x = float(np.sum(levels * px))
    mean_y = float(np.sum(levels * py))
    std_x = float(np.sqrt(np.sum(((levels - mean_x) ** 2) * px)))
    std_y = float(np.sqrt(np.sum(((levels - mean_y) ** 2) * py)))

    if std_x <= EPS or std_y <= EPS:
        correlation = np.nan
    else:
        correlation = float(np.sum((i - mean_x) * (j - mean_y) * glcm) / (std_x * std_y))

    return {
        "homogeneity_idm": homogeneity,
        "contrast": contrast,
        "entropy": entropy,
        "correlation": correlation,
    }


def compute_case_features(image: np.ndarray, mask: np.ndarray, group: str, image_id: str, case_dir: Path, mask_type: str) -> List[Dict[str, object]]:
    mask_bool = mask > 0
    if mask_bool.sum() < 5:
        return []

    image_q = quantize_image(image, N_GRAY_LEVELS)
    records = []
    for distance in DISTANCES:
        for angle in ANGLES_DEGREES:
            glcm = masked_glcm(image_q, mask_bool, distance, angle, N_GRAY_LEVELS)
            features = glcm_features(glcm)
            records.append({
                "group": group,
                "image_id": image_id,
                "case_dir": str(case_dir),
                "mask_type": mask_type,
                "distance_px": distance,
                "angle_deg": angle,
                **features,
            })
    return records


def summarize(records_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["homogeneity_idm", "contrast", "entropy", "correlation"]
    if records_df.empty:
        return pd.DataFrame()

    by_da = records_df.groupby(["group", "distance_px", "angle_deg"], dropna=False).agg(
        n_images=("image_id", "nunique"),
        **{f"{f}_mean": (f, "mean") for f in feature_cols},
        **{f"{f}_std": (f, "std") for f in feature_cols},
    ).reset_index()

    per_image_avg = records_df.groupby(["group", "image_id", "case_dir"], dropna=False)[feature_cols].mean().reset_index()
    overall = per_image_avg.groupby("group", dropna=False).agg(
        n_images=("image_id", "nunique"),
        **{f"{f}_mean": (f, "mean") for f in feature_cols},
        **{f"{f}_std": (f, "std") for f in feature_cols},
    ).reset_index()
    overall["distance_px"] = "ALL"
    overall["angle_deg"] = "ALL"

    return pd.concat([by_da, overall], ignore_index=True)


def save_summary_figure(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return

    display_df = summary.copy()
    compact = pd.DataFrame()
    compact["grupo"] = display_df["group"]
    compact["d"] = display_df["distance_px"].astype(str)
    compact["angulo"] = display_df["angle_deg"].astype(str)
    compact["n"] = display_df["n_images"]

    for f in ["homogeneity_idm", "contrast", "entropy", "correlation"]:
        compact[f] = (
            display_df[f"{f}_mean"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
            + " ± "
            + display_df[f"{f}_std"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
        )

    fig_height = max(3.8, 0.42 * len(compact) + 1.4)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)
    table = ax.table(cellText=compact.values, colLabels=compact.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_for_mask_type(cases: Iterable[Path], mask_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask_filename = "manual_mask.png" if mask_type == "manual" else "predicted_mask.png"
    all_records = []

    for case_dir in cases:
        image_path = case_dir / "image_preprocessed.png"
        mask_path = case_dir / mask_filename
        if not image_path.exists() or not mask_path.exists():
            continue

        image = read_grayscale_image(image_path)
        mask = ensure_same_shape(read_grayscale_image(mask_path), image.shape)
        all_records.extend(compute_case_features(image, mask, infer_group(case_dir), get_image_id(case_dir), case_dir, mask_type))

    per_image = pd.DataFrame(all_records)
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
        per_image_path = OUTPUT_TABLES_DIR / f"05_glcm_{mask_type}_per_image.csv"
        summary_path = OUTPUT_TABLES_DIR / f"05_glcm_{mask_type}_summary.csv"
        figure_path = OUTPUT_FIGURES_DIR / f"05_glcm_{mask_type}_summary.png"

        per_image.to_csv(per_image_path, index=False)
        summary.to_csv(summary_path, index=False)
        save_summary_figure(
            summary,
            figure_path,
            f"Modulo 05 - GLCM usando mascara {mask_type}\nDistancias {DISTANCES}; angulos {ANGLES_DEGREES}; cuantizacion {N_GRAY_LEVELS} niveles",
        )

        print(f"\nMascara: {mask_type}")
        print(f"Tabla por imagen: {per_image_path}")
        print(f"Tabla resumen: {summary_path}")
        print(f"Figura resumen: {figure_path}")

    print("\nListo.")


if __name__ == "__main__":
    main()
