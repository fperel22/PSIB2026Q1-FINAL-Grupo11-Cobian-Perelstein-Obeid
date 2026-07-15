"""
07_single_case_feature_viewer_GUI.py

Modulo 07 - Interfaz para inspeccionar una imagen segmentada y sus atributos.

El usuario elige una carpeta de resultado dentro de:
    outputs/segmentation_gui/

La carpeta elegida debe contener:
    image_preprocessed.png
    manual_mask.png
    predicted_mask.png
    metrics_and_parameters.json

La interfaz muestra:
- imagen preprocesada
- mascara manual
- mascara predicha
- tipo de segmentacion usado
- atributos GLCM promedio sobre distancias 1 y 4, angulos 0 y 90
- intensidad relativa lesion/fondo
- irregularidad del borde
- orientacion ancho/alto
- diferencia de intensidad media banda perilesional - lesion

Ejecucion:
    python src/07_single_case_feature_viewer_GUI.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

SEGMENTATION_ROOT = Path("outputs/segmentation_gui")
DISTANCES = [1, 4]
ANGLES_DEGREES = [0, 90]
N_GRAY_LEVELS = 32
EPS = 1e-12


def read_grayscale_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return image


def ensure_same_shape(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


def image_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def mask_to_rgb(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask.astype(bool)] = np.array(color, dtype=np.uint8)
    return rgb


def overlay_contours(image: np.ndarray, manual_mask: np.ndarray, predicted_mask: np.ndarray) -> np.ndarray:
    rgb = image_to_rgb(image)
    manual_contours, _ = cv2.findContours((manual_mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours((predicted_mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, manual_contours, -1, (0, 255, 255), 2)  # cyan manual
    cv2.drawContours(rgb, pred_contours, -1, (255, 0, 0), 2)      # rojo predicha
    return rgb


def quantize_image(image: np.ndarray, n_levels: int = N_GRAY_LEVELS) -> np.ndarray:
    q = np.floor(image.astype(np.float32) * n_levels / 256.0)
    return np.clip(q, 0, n_levels - 1).astype(np.uint8)


def angle_to_offset(distance: int, angle_degrees: int) -> Tuple[int, int]:
    if angle_degrees == 0:
        return 0, distance
    if angle_degrees == 90:
        return distance, 0
    raise ValueError("Este modulo solo usa angulos 0 y 90 grados.")


def masked_glcm(image_q: np.ndarray, mask: np.ndarray, distance: int, angle_degrees: int, n_levels: int = N_GRAY_LEVELS) -> np.ndarray:
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
    np.add.at(glcm, (b, a), 1)
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
    correlation = np.nan if std_x <= EPS or std_y <= EPS else float(np.sum((i - mean_x) * (j - mean_y) * glcm) / (std_x * std_y))
    return {"homogeneity_idm": homogeneity, "contrast": contrast, "entropy": entropy, "correlation": correlation}


def average_glcm_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    mask_bool = mask > 0
    if mask_bool.sum() < 5:
        return {"homogeneity_idm": np.nan, "contrast": np.nan, "entropy": np.nan, "correlation": np.nan}
    image_q = quantize_image(image, N_GRAY_LEVELS)
    records = []
    for d in DISTANCES:
        for angle in ANGLES_DEGREES:
            records.append(glcm_features(masked_glcm(image_q, mask_bool, d, angle, N_GRAY_LEVELS)))
    df = pd.DataFrame(records)
    return {col: float(df[col].mean()) for col in df.columns}


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
    valid = image > 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx((valid.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, kernel) > 0
    largest = keep_largest_component(closed)
    if largest.sum() == 0:
        largest = closed
    return fill_binary_holes(largest)


def perilesional_radius(mask: np.ndarray) -> int:
    area = float(mask.astype(bool).sum())
    if area <= 0:
        return 5
    d_eq = 2.0 * np.sqrt(area / np.pi)
    return int(round(max(5.0, min(20.0, 0.10 * d_eq))))


def get_perilesional_band(mask: np.ndarray, valid_region: np.ndarray) -> Tuple[np.ndarray, int]:
    mask_bool = mask.astype(bool)
    r = perilesional_radius(mask_bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dilated = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1) > 0
    return dilated & (~mask_bool) & valid_region, r


def contour_perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours((mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(sum(cv2.arcLength(c, True) for c in contours)) if contours else 0.0


def bounding_box_aspect_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return np.nan
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    return float(w / h) if h > 0 else np.nan


def ecographic_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    mask_bool = mask > 0
    valid_region = get_valid_ultrasound_region(image)
    background = valid_region & (~mask_bool)
    band, radius = get_perilesional_band(mask_bool, valid_region)
    lesion_mean = float(np.mean(image[mask_bool])) if mask_bool.sum() > 0 else np.nan
    background_mean = float(np.mean(image[background])) if background.sum() > 0 else np.nan
    peri_mean = float(np.mean(image[band])) if band.sum() > 0 else np.nan
    rel = (background_mean - lesion_mean) / (background_mean + 1e-8) if pd.notna(background_mean) and pd.notna(lesion_mean) else np.nan
    diff = peri_mean - lesion_mean if pd.notna(peri_mean) and pd.notna(lesion_mean) else np.nan
    area = float(mask_bool.sum())
    perimeter = contour_perimeter(mask_bool)
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8) if area > 0 and perimeter > 0 else np.nan
    irregularity = 1.0 - circularity if pd.notna(circularity) else np.nan
    return {
        "relative_intensity_lesion_background": rel,
        "irregularity_1_minus_circularity": irregularity,
        "orientation_width_height_ratio": bounding_box_aspect_ratio(mask_bool),
        "perilesional_radius_px": float(radius),
        "perilesional_minus_lesion_mean_intensity": diff,
    }


def format_value(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.4f}"


def describe_segmentation(metadata: Dict[str, object]) -> str:
    params = metadata.get("method_parameters", {}) if isinstance(metadata, dict) else {}
    if not isinstance(params, dict):
        return "No disponible"
    parts = []
    for key in ["method", "mode", "tolerance", "lesion_polarity", "segmentation_image", "opening_kernel_size", "closing_kernel_size", "morphology_order"]:
        if key in params:
            parts.append(f"{key}: {params[key]}")
    return "\n".join(parts) if parts else "No disponible"


class ImageLabel(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(310, 250)
        self.setStyleSheet("border: 1px solid #888; background-color: #111; color: #ddd;")
        self._qimage: Optional[QImage] = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        image_rgb = np.ascontiguousarray(image_rgb.astype(np.uint8))
        h, w, c = image_rgb.shape
        qimage = QImage(image_rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        self._qimage = qimage

    def resizeEvent(self, event) -> None:  # noqa: N802
        if self._qimage is not None:
            pixmap = QPixmap.fromImage(self._qimage).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(pixmap)
        super().resizeEvent(event)


class FeatureViewerGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Modulo 07 - Visualizacion de atributos por imagen")
        self.resize(1450, 900)
        self._build_widgets()
        self._build_layout()
        self.load_button.clicked.connect(self.load_case_folder)

    def _build_widgets(self) -> None:
        self.load_button = QPushButton("Cargar carpeta de imagen segmentada")
        self.image_label = ImageLabel("Imagen preprocesada")
        self.manual_label = ImageLabel("Mascara manual")
        self.predicted_label = ImageLabel("Mascara predicha")
        self.overlay_label = ImageLabel("Overlay: manual cyan, predicha roja")
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")

    def _build_layout(self) -> None:
        top = QHBoxLayout()
        top.addWidget(self.load_button)
        top.addStretch()

        grid = QGridLayout()
        grid.addWidget(QLabel("Imagen preprocesada"), 0, 0)
        grid.addWidget(QLabel("Mascara manual"), 0, 1)
        grid.addWidget(QLabel("Mascara predicha"), 0, 2)
        grid.addWidget(QLabel("Overlay"), 0, 3)
        grid.addWidget(self.image_label, 1, 0)
        grid.addWidget(self.manual_label, 1, 1)
        grid.addWidget(self.predicted_label, 1, 2)
        grid.addWidget(self.overlay_label, 1, 3)

        main = QVBoxLayout()
        main.addLayout(top)
        main.addLayout(grid)
        main.addWidget(QLabel("Atributos y segmentacion"))
        main.addWidget(self.info_text, stretch=1)
        central = QWidget()
        central.setLayout(main)
        self.setCentralWidget(central)

    def load_case_folder(self) -> None:
        start_dir = SEGMENTATION_ROOT if SEGMENTATION_ROOT.exists() else Path.cwd()
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de resultado", str(start_dir))
        if not folder:
            return
        case_dir = Path(folder)
        try:
            self.load_case(case_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def load_case(self, case_dir: Path) -> None:
        image_path = case_dir / "image_preprocessed.png"
        manual_path = case_dir / "manual_mask.png"
        pred_path = case_dir / "predicted_mask.png"
        json_path = case_dir / "metrics_and_parameters.json"
        for path in [image_path, manual_path, pred_path]:
            if not path.exists():
                raise FileNotFoundError(f"No existe: {path}")

        image = read_grayscale_image(image_path)
        manual = ensure_same_shape(read_grayscale_image(manual_path), image.shape) > 0
        pred = ensure_same_shape(read_grayscale_image(pred_path), image.shape) > 0

        metadata = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        self.image_label.set_image(image_to_rgb(image))
        self.manual_label.set_image(mask_to_rgb(manual, (0, 255, 255)))
        self.predicted_label.set_image(mask_to_rgb(pred, (255, 0, 0)))
        self.overlay_label.set_image(overlay_contours(image, manual, pred))

        manual_glcm = average_glcm_features(image, manual)
        pred_glcm = average_glcm_features(image, pred)
        manual_echo = ecographic_features(image, manual)
        pred_echo = ecographic_features(image, pred)

        self.info_text.setPlainText(self.build_report(case_dir, metadata, manual_glcm, pred_glcm, manual_echo, pred_echo))

    def build_report(self, case_dir: Path, metadata: Dict[str, object], manual_glcm: Dict[str, float], pred_glcm: Dict[str, float], manual_echo: Dict[str, float], pred_echo: Dict[str, float]) -> str:
        lines = []
        lines.append(f"Carpeta: {case_dir}")
        lines.append("")
        lines.append("Segmentacion utilizada:")
        lines.append(describe_segmentation(metadata))
        lines.append("")
        lines.append(f"GLCM: promedio sobre distancias {DISTANCES} y angulos {ANGLES_DEGREES}; cuantizacion {N_GRAY_LEVELS} niveles")
        lines.append("")
        lines.append("ATRIBUTOS GLCM")
        lines.append("atributo                         manual        predicha")
        for key in ["homogeneity_idm", "contrast", "entropy", "correlation"]:
            lines.append(f"{key:<32} {format_value(manual_glcm[key]):>10}   {format_value(pred_glcm[key]):>10}")
        lines.append("")
        lines.append("ATRIBUTOS ECOGRAFICOS / MORFOLOGICOS")
        lines.append("atributo                         manual        predicha")
        for key in ["relative_intensity_lesion_background", "irregularity_1_minus_circularity", "orientation_width_height_ratio", "perilesional_radius_px", "perilesional_minus_lesion_mean_intensity"]:
            lines.append(f"{key:<40} {format_value(manual_echo[key]):>10}   {format_value(pred_echo[key]):>10}")
        return "\n".join(lines)


def main() -> None:
    app = QApplication([])
    window = FeatureViewerGUI()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
