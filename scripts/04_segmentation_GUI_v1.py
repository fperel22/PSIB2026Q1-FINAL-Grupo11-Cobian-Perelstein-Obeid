"""
04_segmentation_GUI.py

Modulo 04 - Segmentacion semi-automatica de lesiones BUSI con GUI local.

Entrada esperada:
    data/processed/preprocessed/robust_bilateral/
        benign/images/
        benign/masks/
        malignant/images/
        malignant/masks/

Metodos implementados:
1. Region growing con semilla manual.
2. Umbral local dentro de la ROI.
3. Chan-Vese dentro de la ROI.

Dependencias:
    pip install PySide6 opencv-python numpy scikit-image

Ejecucion:
    python src/04_segmentation_GUI.py
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from skimage.segmentation import chan_vese
    SKIMAGE_AVAILABLE = True
except ImportError:
    chan_vese = None
    SKIMAGE_AVAILABLE = False

RoiRect = Tuple[int, int, int, int]
Point = Tuple[int, int]

INPUT_ROOT = Path("data/processed/preprocessed/robust_bilateral")
OUTPUT_ROOT = Path("outputs/segmentation_gui")


class ClickableImageLabel(QLabel):
    """QLabel que traduce clicks/drag a coordenadas reales de imagen."""

    image_clicked = Signal(int, int)
    roi_selected = Signal(int, int, int, int)

    def __init__(self, text: str, minimum_size: Tuple[int, int] = (560, 360)) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(*minimum_size)
        self.setStyleSheet("border: 1px solid #888; background-color: #111; color: #ddd;")
        self._qimage: Optional[QImage] = None
        self._image_width = 0
        self._image_height = 0
        self._mode = "none"
        self._drag_start: Optional[Point] = None

    def set_interaction_mode(self, mode: str) -> None:
        if mode not in {"none", "roi", "seed"}:
            raise ValueError("Modo invalido. Use 'none', 'roi' o 'seed'.")
        self._mode = mode
        self._drag_start = None

    def set_image(self, image_rgb: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("set_image espera imagen RGB con shape (H, W, 3).")
        image_rgb = np.ascontiguousarray(image_rgb.astype(np.uint8))
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width
        self._qimage = QImage(
            image_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        self._image_width = width
        self._image_height = height
        self._update_pixmap()

    def clear_image(self, text: str) -> None:
        self._qimage = None
        self._image_width = 0
        self._image_height = 0
        self.setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._update_pixmap()
        super().resizeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._mode == "none" or self.pixmap() is None or self._qimage is None:
            return
        if event.button() != Qt.LeftButton:
            return
        mapped = self._event_to_image_coordinates(event)
        if mapped is None:
            return
        if self._mode == "roi":
            self._drag_start = mapped
            return
        if self._mode == "seed":
            x_img, y_img = mapped
            self.image_clicked.emit(x_img, y_img)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._mode != "roi" or self._drag_start is None:
            return
        if event.button() != Qt.LeftButton:
            return
        mapped = self._event_to_image_coordinates(event)
        if mapped is None:
            self._drag_start = None
            return
        x_start, y_start = self._drag_start
        x_end, y_end = mapped
        self._drag_start = None
        x0, x1 = sorted((x_start, x_end))
        y0, y1 = sorted((y_start, y_end))
        if (x1 - x0) < 10 or (y1 - y0) < 10:
            return
        self.roi_selected.emit(x0, y0, x1, y1)

    def _update_pixmap(self) -> None:
        if self._qimage is None:
            return
        pixmap = QPixmap.fromImage(self._qimage)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def _event_to_image_coordinates(self, event: QMouseEvent) -> Optional[Point]:
        pos = event.position()
        return self._label_to_image_coordinates(int(pos.x()), int(pos.y()))

    def _label_to_image_coordinates(self, x_label: int, y_label: int) -> Optional[Point]:
        pixmap = self.pixmap()
        if pixmap is None or self._image_width <= 0 or self._image_height <= 0:
            return None
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        offset_x = (self.width() - pixmap_width) // 2
        offset_y = (self.height() - pixmap_height) // 2
        x_in_pixmap = x_label - offset_x
        y_in_pixmap = y_label - offset_y
        if not (0 <= x_in_pixmap < pixmap_width and 0 <= y_in_pixmap < pixmap_height):
            return None
        x_img = int(x_in_pixmap * self._image_width / pixmap_width)
        y_img = int(y_in_pixmap * self._image_height / pixmap_height)
        x_img = int(np.clip(x_img, 0, self._image_width - 1))
        y_img = int(np.clip(y_img, 0, self._image_height - 1))
        return x_img, y_img


def read_grayscale_image(path: str | Path) -> np.ndarray:
    """
    Lee una imagen en escala de grises uint8 soportando rutas Unicode de Windows.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """
    Guarda una imagen soportando rutas Unicode de Windows.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    extension = path.suffix.lower()
    if extension == "":
        extension = ".png"

    ok, encoded = cv2.imencode(extension, image)

    if not ok:
        raise IOError(f"No se pudo codificar la imagen: {path}")

    encoded.tofile(str(path))


def read_manual_mask(path: str | Path, target_shape: Tuple[int, int]) -> np.ndarray:
    mask_gray = read_grayscale_image(path)
    if mask_gray.shape != target_shape:
        mask_gray = cv2.resize(mask_gray, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_gray > 127


def sanitize_filename(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("\\", "_").replace(":", "_")


def infer_label_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "benign" in parts:
        return "benign"
    if "malignant" in parts:
        return "malignant"
    return "unknown"


def find_manual_mask_path(image_path: Path) -> Optional[Path]:
    candidates = []
    if image_path.parent.name.lower() == "images":
        masks_dir = image_path.parent.parent / "masks"
        candidates.append(masks_dir / f"{image_path.stem}_mask.png")
    candidates.append(image_path.with_name(f"{image_path.stem}_mask.png"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_roi_mask(shape: Tuple[int, int], roi_rect: RoiRect) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = clip_roi(roi_rect, shape)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask


def clip_roi(roi_rect: RoiRect, shape: Tuple[int, int]) -> RoiRect:
    height, width = shape
    x0, y0, x1, y1 = roi_rect
    x0 = int(np.clip(x0, 0, width - 1))
    x1 = int(np.clip(x1, 0, width - 1))
    y0 = int(np.clip(y0, 0, height - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return x0, y0, x1, y1


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8)) * 255
    if mask_u8.max() == 0:
        return mask.astype(bool)
    height, width = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, holes)
    return filled > 0


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def keep_component_containing_point(mask: np.ndarray, point: Point) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask, dtype=bool)
    x, y = point
    selected_label = 0
    if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
        selected_label = int(labels[y, x])
    if selected_label == 0:
        selected_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == selected_label


def postprocess_mask(
    mask: np.ndarray,
    roi_rect: RoiRect,
    seed_xy: Optional[Point] = None,
    close_kernel_size: int = 7,
    open_kernel_size: int = 3,
) -> np.ndarray:
    roi_mask = make_roi_mask(mask.shape, roi_rect)

    clean = mask.astype(bool) & roi_mask

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel_size, close_kernel_size),
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_kernel_size, open_kernel_size),
    )

    clean_u8 = (clean.astype(np.uint8)) * 255
    clean_u8 = cv2.morphologyEx(
        clean_u8,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )

    clean = fill_binary_holes(clean_u8 > 0)

    clean_u8 = (clean.astype(np.uint8)) * 255
    clean_u8 = cv2.morphologyEx(
        clean_u8,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    clean = (clean_u8 > 0) & roi_mask

    if seed_xy is not None:
        clean = keep_component_containing_point(clean, seed_xy)
    else:
        clean = keep_largest_component(clean)

    return clean & roi_mask


def segment_region_growing(image: np.ndarray, roi_rect: RoiRect, seed_xy: Point, tolerance: int) -> np.ndarray:
    height, width = image.shape
    seed_x, seed_y = seed_xy
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError("La semilla esta fuera de la imagen.")
    roi_mask = make_roi_mask(image.shape, roi_rect)
    if not roi_mask[seed_y, seed_x]:
        raise ValueError("La semilla debe estar dentro de la ROI.")

    r = 2
    y0 = max(0, seed_y - r)
    y1 = min(height, seed_y + r + 1)
    x0 = max(0, seed_x - r)
    x1 = min(width, seed_x + r + 1)
    seed_value = float(np.median(image[y0:y1, x0:x1]))
    upper = min(255.0, seed_value + tolerance)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    mask = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)
    queue: deque[Point] = deque([(seed_x, seed_y)])
    visited[seed_y, seed_x] = True
    max_area = int(0.95 * roi_mask.sum())
    grown_area = 0

    while queue:
        x, y = queue.popleft()
        if float(image[y, x]) <= upper:
            mask[y, x] = True
            grown_area += 1
            if grown_area >= max_area:
                break
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and roi_mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))

    return postprocess_mask(mask, roi_rect=roi_rect, seed_xy=seed_xy)


def segment_local_threshold(image: np.ndarray, roi_rect: RoiRect, mode: str) -> np.ndarray:
    x0, y0, x1, y1 = clip_roi(roi_rect, image.shape)
    crop = image[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        return np.zeros_like(image, dtype=bool)

    if mode == "otsu":
        threshold, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "percentile_15":
        threshold = float(np.percentile(crop, 15))

    elif mode == "percentile_30":
        threshold = float(np.percentile(crop, 30))

    elif mode == "percentile_45":
        threshold = float(np.percentile(crop, 45))
    else:
        raise ValueError(f"Modo de umbral local no reconocido: {mode}")

    full_mask = np.zeros_like(image, dtype=bool)
    full_mask[y0:y1 + 1, x0:x1 + 1] = crop <= threshold
    return postprocess_mask(full_mask, roi_rect=roi_rect, seed_xy=None)


def make_chan_vese_initial_mask(
    crop_norm: np.ndarray,
    init_percentile: float,
) -> np.ndarray:
    """
    Crea una mascara inicial para Chan-Vese dentro del ROI.

    Se inicializa desde pixeles oscuros del ROI. Esto es mas adecuado que
    una elipse central fija porque muchas lesiones BUSI no estan centradas
    y porque la lesion suele ser hipoecoica.
    """

    crop_u8 = np.clip(255 * crop_norm, 0, 255).astype(np.uint8)

    threshold = float(np.percentile(crop_u8, init_percentile))
    init = crop_u8 <= threshold

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    init_u8 = (init.astype(np.uint8)) * 255
    init_u8 = cv2.morphologyEx(init_u8, cv2.MORPH_OPEN, kernel_open, iterations=1)
    init_u8 = cv2.morphologyEx(init_u8, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    init = init_u8 > 0

    if init.sum() < 10:
        height, width = crop_norm.shape
        cy = height // 2
        cx = width // 2
        ry = max(2, int(0.30 * height))
        rx = max(2, int(0.30 * width))
        y, x = np.ogrid[:height, :width]
        init = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0

    return init

def segment_chan_vese_roi(
    image: np.ndarray,
    roi_rect: RoiRect,
    iterations: int,
    mu: float,
    init_percentile: float,
    dark_limit_percentile: float,
) -> np.ndarray:
    """
    Chan-Vese dentro de la ROI, ajustado para lesiones hipoecoicas.

    Cambios respecto de la version anterior:
    - inicializacion por pixeles oscuros dentro del ROI;
    - menor regularizacion inicial para que el contorno se adapte mas;
    - restriccion hipoecoica suave para evitar expansion hacia tejido brillante;
    - closing 5x5 en vez de 7x7 para no inflar tanto la mascara.
    """

    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image no esta instalado. Instala con: pip install scikit-image")

    x0, y0, x1, y1 = clip_roi(roi_rect, image.shape)
    crop = image[y0:y1 + 1, x0:x1 + 1].astype(np.float32)

    if crop.size == 0:
        return np.zeros_like(image, dtype=bool)

    crop_norm = crop / 255.0

    init_mask = make_chan_vese_initial_mask(
        crop_norm=crop_norm,
        init_percentile=init_percentile,
    )

    cv_mask = chan_vese(
        crop_norm,
        mu=mu,
        lambda1=1.0,
        lambda2=1.4,
        tol=1e-3,
        max_num_iter=iterations,
        dt=0.5,
        init_level_set=init_mask,
        extended_output=False,
    ).astype(bool)

    # La lesion esperada suele ser hipoecoica. Si Chan-Vese devuelve la
    # region mas clara, se invierte la mascara dentro de la ROI.
    if cv_mask.any() and (~cv_mask).any():
        mean_inside = float(np.mean(crop_norm[cv_mask]))
        mean_outside = float(np.mean(crop_norm[~cv_mask]))
        if mean_inside > mean_outside:
            cv_mask = ~cv_mask

    # Restriccion hipoecoica suave. No es una segmentacion por umbral puro:
    # solo evita que el contorno activo se expanda hacia zonas claramente
    # mas brillantes que no corresponden a la lesion oscura.
    dark_limit = float(np.percentile(crop, dark_limit_percentile))
    cv_mask = np.logical_and(cv_mask, crop <= dark_limit)

    full_mask = np.zeros_like(image, dtype=bool)
    full_mask[y0:y1 + 1, x0:x1 + 1] = cv_mask

    return postprocess_mask(
        full_mask,
        roi_rect=roi_rect,
        seed_xy=None,
        close_kernel_size=5,
        open_kernel_size=3,
    )

def image_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def mask_to_rgb(mask: Optional[np.ndarray], color: Tuple[int, int, int], shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if mask is None:
        if shape is None:
            return np.zeros((256, 256, 3), dtype=np.uint8)
        return np.zeros((*shape, 3), dtype=np.uint8)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask.astype(bool)] = np.array(color, dtype=np.uint8)
    return rgb


def make_main_overlay(image: np.ndarray, predicted_mask: Optional[np.ndarray], roi_rect: Optional[RoiRect], seed_xy: Optional[Point]) -> np.ndarray:
    base = image_to_rgb(image)
    output = base.copy()
    if predicted_mask is not None and predicted_mask.any():
        overlay = base.copy()
        overlay[predicted_mask] = np.array([255, 0, 0], dtype=np.uint8)
        output = cv2.addWeighted(overlay, 0.35, base, 0.65, 0)
        contours, _ = cv2.findContours((predicted_mask.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    if roi_rect is not None:
        x0, y0, x1, y1 = roi_rect
        cv2.rectangle(output, (x0, y0), (x1, y1), (0, 180, 255), 2)
    if seed_xy is not None:
        cv2.drawMarker(output, seed_xy, color=(255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    return output


def make_comparison_overlay(image: np.ndarray, predicted_mask: Optional[np.ndarray], manual_mask: Optional[np.ndarray]) -> np.ndarray:
    base = image_to_rgb(image)
    if predicted_mask is None or manual_mask is None:
        return base
    pred = predicted_mask.astype(bool)
    manual = manual_mask.astype(bool)
    tp = pred & manual
    fp = pred & ~manual
    fn = ~pred & manual
    overlay = base.copy()
    overlay[tp] = np.array([0, 255, 0], dtype=np.uint8)
    overlay[fp] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[fn] = np.array([0, 120, 255], dtype=np.uint8)
    return cv2.addWeighted(overlay, 0.55, base, 0.45, 0)


def compute_segmentation_metrics(predicted_mask: np.ndarray, manual_mask: np.ndarray) -> Dict[str, float | int]:
    pred = predicted_mask.astype(bool)
    manual = manual_mask.astype(bool)
    tp = int(np.logical_and(pred, manual).sum())
    fp = int(np.logical_and(pred, ~manual).sum())
    fn = int(np.logical_and(~pred, manual).sum())
    tn = int(np.logical_and(~pred, ~manual).sum())

    def safe_div(num: float, den: float) -> float:
        return float("nan") if den == 0 else num / den

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Dice": safe_div(2 * tp, 2 * tp + fp + fn),
        "Jaccard": safe_div(tp, tp + fp + fn),
        "Sensibilidad": safe_div(tp, tp + fn),
        "Precision": safe_div(tp, tp + fp),
    }


def format_metric(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if np.isnan(value):
        return "N/A"
    return f"{value:.4f}"


class SegmentationGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BUSI - Segmentacion semi-automatica por ROI")
        self.resize(1450, 900)

        self.image_path: Optional[Path] = None
        self.manual_mask_path: Optional[Path] = None
        self.image: Optional[np.ndarray] = None
        self.manual_mask: Optional[np.ndarray] = None
        self.predicted_mask: Optional[np.ndarray] = None
        self.roi_rect: Optional[RoiRect] = None
        self.seed_xy: Optional[Point] = None
        self.current_metrics: Optional[Dict[str, float | int]] = None
        self.current_parameters: Optional[Dict[str, object]] = None

        self._build_widgets()
        self._build_layout()
        self._connect_signals()
        self._update_method_presets()
        self._update_buttons()

    def _build_widgets(self) -> None:
        self.main_image_label = ClickableImageLabel("Imagen preprocesada + ROI + mascara calculada", (620, 420))
        self.manual_mask_label = ClickableImageLabel("Mascara manual", (360, 260))
        self.predicted_mask_label = ClickableImageLabel("Mascara calculada", (360, 260))
        self.comparison_label = ClickableImageLabel("Overlay comparativo", (520, 320))
        self.manual_mask_label.set_interaction_mode("none")
        self.predicted_mask_label.set_interaction_mode("none")
        self.comparison_label.set_interaction_mode("none")

        self.load_image_button = QPushButton("Cargar imagen bilateral")
        self.load_manual_button = QPushButton("Cargar mascara manual")
        self.draw_roi_button = QPushButton("Dibujar ROI")
        self.seed_button = QPushButton("Elegir semilla")
        self.segment_button = QPushButton("Segmentar")
        self.clear_button = QPushButton("Borrar ROI/segmentacion")
        self.save_button = QPushButton("Guardar resultado")

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Region growing", "Umbral local", "Chan-Vese"])
        self.preset_combo = QComboBox()

        self.metrics_label = QLabel()
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("font-family: Consolas, monospace; background-color: #222; color: #eee; border: 1px solid #777; padding: 8px;")
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 4px;")
        self._reset_metrics_text()
        self.status_label.setText("Flujo: cargar imagen bilateral -> dibujar ROI -> elegir metodo/preset -> elegir semilla si es region growing -> segmentar -> guardar resultado.")

    def _build_layout(self) -> None:
        controls_1 = QHBoxLayout()
        controls_1.addWidget(self.load_image_button)
        controls_1.addWidget(self.load_manual_button)
        controls_1.addWidget(self.draw_roi_button)
        controls_1.addWidget(self.seed_button)
        controls_1.addWidget(self.clear_button)
        controls_1.addStretch()

        controls_2 = QHBoxLayout()
        controls_2.addWidget(QLabel("Metodo:"))
        controls_2.addWidget(self.method_combo)
        controls_2.addWidget(QLabel("Preset:"))
        controls_2.addWidget(self.preset_combo)
        controls_2.addWidget(self.segment_button)
        controls_2.addWidget(self.save_button)
        controls_2.addStretch()

        views = QGridLayout()
        views.addWidget(QLabel("Imagen preprocesada + ROI/semilla/mascara"), 0, 0)
        views.addWidget(QLabel("Mascara manual"), 0, 1)
        views.addWidget(QLabel("Mascara calculada"), 0, 2)
        views.addWidget(self.main_image_label, 1, 0, 2, 1)
        views.addWidget(self.manual_mask_label, 1, 1)
        views.addWidget(self.predicted_mask_label, 1, 2)
        views.addWidget(QLabel("Comparacion: verde=TP, rojo=FP, azul=FN"), 2, 1)
        views.addWidget(QLabel("Metricas"), 2, 2)
        views.addWidget(self.comparison_label, 3, 1)
        views.addWidget(self.metrics_label, 3, 2)
        views.setColumnStretch(0, 2)
        views.setColumnStretch(1, 1)
        views.setColumnStretch(2, 1)

        main = QVBoxLayout()
        main.addLayout(controls_1)
        main.addLayout(controls_2)
        main.addLayout(views, stretch=1)
        main.addWidget(self.status_label)
        central = QWidget()
        central.setLayout(main)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.load_image_button.clicked.connect(self.load_image)
        self.load_manual_button.clicked.connect(self.load_manual_mask)
        self.draw_roi_button.clicked.connect(self.start_roi_selection)
        self.seed_button.clicked.connect(self.start_seed_selection)
        self.segment_button.clicked.connect(self.segment_current_image)
        self.clear_button.clicked.connect(self.clear_segmentation_state)
        self.save_button.clicked.connect(self.save_current_result)
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.main_image_label.roi_selected.connect(self.on_roi_selected)
        self.main_image_label.image_clicked.connect(self.on_seed_selected)

    def _on_method_changed(self) -> None:
        self._update_method_presets()
        self._update_buttons()

    def _update_method_presets(self) -> None:
        method = self.method_combo.currentText()
        self.preset_combo.clear()
        if method == "Region growing":
            self.preset_combo.addItem("Tolerancia baja (T=25)", {"tolerance": 25})
            self.preset_combo.addItem("Tolerancia media (T=35)", {"tolerance": 35})
            self.preset_combo.addItem("Tolerancia alta (T=45)", {"tolerance": 45})
            self.preset_combo.addItem("Tolerancia muy alta (T=60)", {"tolerance": 60})
        elif method == "Umbral local":
            self.preset_combo.addItem("Otsu local", {"mode": "otsu"})
            self.preset_combo.addItem("Percentil oscuro 15", {"mode": "percentile_15"})
            self.preset_combo.addItem("Percentil oscuro 30", {"mode": "percentile_30"})
            self.preset_combo.addItem("Percentil oscuro 45", {"mode": "percentile_45"})
        elif method == "Chan-Vese":
            self.preset_combo.addItem(
                "Conservador (250 iter, mu=0.05)",
                {"iterations": 250, "mu": 0.05, "init_percentile": 20, "dark_limit_percentile": 55},
            )
            self.preset_combo.addItem(
                "Medio (300 iter, mu=0.08)",
                {"iterations": 300, "mu": 0.08, "init_percentile": 25, "dark_limit_percentile": 60},
            )
            self.preset_combo.addItem(
                "Expansivo (350 iter, mu=0.12)",
                {"iterations": 350, "mu": 0.12, "init_percentile": 30, "dark_limit_percentile": 65},
            )

    def _update_buttons(self) -> None:
        has_image = self.image is not None
        has_roi = self.roi_rect is not None
        has_prediction = self.predicted_mask is not None
        needs_seed = self.method_combo.currentText() == "Region growing"
        self.load_manual_button.setEnabled(has_image)
        self.draw_roi_button.setEnabled(has_image)
        self.seed_button.setEnabled(has_image and has_roi and needs_seed)
        self.segment_button.setEnabled(has_image and has_roi)
        self.clear_button.setEnabled(has_image and (has_roi or has_prediction))
        self.save_button.setEnabled(has_image and has_prediction)

    def load_image(self) -> None:
        start_dir = INPUT_ROOT if INPUT_ROOT.exists() else Path.cwd()
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen preprocesada bilateral", str(start_dir), "Imagenes (*.png *.jpg *.jpeg *.bmp);;Todos los archivos (*)")
        if not file_path:
            return
        path = Path(file_path)
        if "_mask" in path.stem.lower():
            QMessageBox.warning(self, "Archivo incorrecto", "Seleccionaste una mascara. Carga una imagen de la carpeta images.")
            return
        try:
            image = read_grayscale_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error al cargar imagen", str(exc))
            return

        self.image_path = path
        self.image = image
        self.manual_mask_path = None
        self.manual_mask = None
        self.predicted_mask = None
        self.roi_rect = None
        self.seed_xy = None
        self.current_metrics = None
        self.current_parameters = None

        mask_path = find_manual_mask_path(path)
        if mask_path is not None:
            try:
                self.manual_mask = read_manual_mask(mask_path, image.shape)
                self.manual_mask_path = mask_path
                mask_msg = f"Mascara manual cargada automaticamente: {mask_path.name}"
            except Exception as exc:
                self.manual_mask = None
                self.manual_mask_path = None
                mask_msg = f"No se pudo cargar mascara manual automaticamente: {exc}"
        else:
            mask_msg = "No se encontro mascara manual automaticamente. Puede cargarla manualmente."

        self.main_image_label.set_interaction_mode("none")
        self.display_all()
        self._reset_metrics_text()
        self._update_buttons()
        self.status_label.setText(f"Imagen cargada: {path.name}. {mask_msg}. Ahora dibuje una ROI alrededor de la lesion.")

    def load_manual_mask(self) -> None:
        if self.image is None:
            return
        start_dir = self.image_path.parent.parent / "masks" if self.image_path else Path.cwd()
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar mascara manual", str(start_dir), "Imagenes (*.png *.jpg *.jpeg *.bmp);;Todos los archivos (*)")
        if not file_path:
            return
        try:
            self.manual_mask = read_manual_mask(file_path, self.image.shape)
        except Exception as exc:
            QMessageBox.critical(self, "Error al cargar mascara manual", str(exc))
            return
        self.manual_mask_path = Path(file_path)
        self.display_all()
        self.update_metrics_if_possible()
        self.status_label.setText(f"Mascara manual cargada: {self.manual_mask_path.name}")

    def start_roi_selection(self) -> None:
        if self.image is None:
            return
        self.main_image_label.set_interaction_mode("roi")
        self.status_label.setText("Modo ROI activo: haga click izquierdo y arrastre un rectangulo alrededor de la lesion.")

    def on_roi_selected(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self.image is None:
            return
        self.roi_rect = clip_roi((x0, y0, x1, y1), self.image.shape)
        self.predicted_mask = None
        self.seed_xy = None
        self.current_metrics = None
        self.current_parameters = None
        self.main_image_label.set_interaction_mode("none")
        self.display_all()
        self._reset_metrics_text()
        self._update_buttons()
        if self.method_combo.currentText() == "Region growing":
            self.status_label.setText(f"ROI seleccionada: {self.roi_rect}. Ahora presione 'Elegir semilla' y haga click dentro de la lesion.")
        else:
            self.status_label.setText(f"ROI seleccionada: {self.roi_rect}. Ahora puede presionar 'Segmentar'.")

    def start_seed_selection(self) -> None:
        if self.image is None or self.roi_rect is None:
            return
        self.main_image_label.set_interaction_mode("seed")
        self.status_label.setText("Modo semilla activo: haga click dentro de la lesion, dentro de la ROI.")

    def on_seed_selected(self, x: int, y: int) -> None:
        if self.image is None or self.roi_rect is None:
            return
        roi_mask = make_roi_mask(self.image.shape, self.roi_rect)
        if not roi_mask[y, x]:
            QMessageBox.warning(self, "Semilla fuera de ROI", "La semilla debe estar dentro de la ROI.")
            return
        self.seed_xy = (x, y)
        self.main_image_label.set_interaction_mode("none")
        self.display_all()
        self._update_buttons()
        self.status_label.setText(f"Semilla seleccionada: (x={x}, y={y}). Ahora presione 'Segmentar'.")

    def segment_current_image(self) -> None:
        if self.image is None or self.roi_rect is None:
            return
        method = self.method_combo.currentText()
        preset_data = self.preset_combo.currentData()
        try:
            if method == "Region growing":
                if self.seed_xy is None:
                    QMessageBox.warning(self, "Falta semilla", "Region growing requiere una semilla manual dentro de la lesion.")
                    return
                tolerance = int(preset_data["tolerance"])
                predicted = segment_region_growing(self.image, self.roi_rect, self.seed_xy, tolerance)
                parameters = {"method": "region_growing", "tolerance": tolerance, "postprocessing": "ROI + closing 7x7 + fill holes + opening 3x3 + seed component"}
            elif method == "Umbral local":
                mode = str(preset_data["mode"])
                predicted = segment_local_threshold(self.image, self.roi_rect, mode)
                parameters = {"method": "local_threshold", "mode": mode, "postprocessing": "ROI + closing 7x7 + fill holes + opening 3x3 + largest component"}
            elif method == "Chan-Vese":
                iterations = int(preset_data["iterations"])
                mu = float(preset_data["mu"])
                init_percentile = float(preset_data["init_percentile"])
                dark_limit_percentile = float(preset_data["dark_limit_percentile"])
                predicted = segment_chan_vese_roi(
                    self.image,
                    self.roi_rect,
                    iterations,
                    mu,
                    init_percentile,
                    dark_limit_percentile,
                )
                parameters = {
                    "method": "chan_vese",
                    "iterations": iterations,
                    "mu": mu,
                    "init_percentile": init_percentile,
                    "dark_limit_percentile": dark_limit_percentile,
                    "lambda1": 1.0,
                    "lambda2": 1.4,
                    "postprocessing": "ROI + dark constraint + closing 5x5 + fill holes + opening 3x3 + largest component",
                }
            else:
                raise ValueError(f"Metodo no reconocido: {method}")
        except Exception as exc:
            QMessageBox.critical(self, "Error en segmentacion", str(exc))
            return

        self.predicted_mask = predicted
        self.current_parameters = parameters
        self.update_metrics_if_possible()
        self.display_all()
        self._update_buttons()
        self.status_label.setText(f"Segmentacion realizada con {method} | preset: {self.preset_combo.currentText()} | area calculada: {int(predicted.sum())} px.")

    def display_all(self) -> None:
        if self.image is None:
            self.main_image_label.clear_image("Imagen preprocesada + ROI + mascara")
            self.manual_mask_label.clear_image("Mascara manual")
            self.predicted_mask_label.clear_image("Mascara calculada")
            self.comparison_label.clear_image("Overlay comparativo")
            return
        self.main_image_label.set_image(make_main_overlay(self.image, self.predicted_mask, self.roi_rect, self.seed_xy))
        self.manual_mask_label.set_image(mask_to_rgb(self.manual_mask, (0, 255, 255), self.image.shape))
        self.predicted_mask_label.set_image(mask_to_rgb(self.predicted_mask, (255, 0, 0), self.image.shape))
        self.comparison_label.set_image(make_comparison_overlay(self.image, self.predicted_mask, self.manual_mask))

    def update_metrics_if_possible(self) -> None:
        if self.predicted_mask is None or self.manual_mask is None:
            self._reset_metrics_text()
            return
        metrics = compute_segmentation_metrics(self.predicted_mask, self.manual_mask)
        self.current_metrics = metrics
        self.metrics_label.setText(
            "Indices contra mascara manual\n"
            f"Dice         : {format_metric(metrics['Dice'])}\n"
            f"Jaccard      : {format_metric(metrics['Jaccard'])}\n"
            f"Sensibilidad : {format_metric(metrics['Sensibilidad'])}\n"
            f"Precision    : {format_metric(metrics['Precision'])}\n"
            "\nMatriz pixel a pixel\n"
            f"TP: {metrics['TP']} | FP: {metrics['FP']}\n"
            f"FN: {metrics['FN']} | TN: {metrics['TN']}"
        )

    def _reset_metrics_text(self) -> None:
        self.metrics_label.setText(
            "Indices contra mascara manual\n"
            "Dice         : -\n"
            "Jaccard      : -\n"
            "Sensibilidad : -\n"
            "Precision    : -\n"
            "\nMatriz pixel a pixel\n"
            "TP: - | FP: -\n"
            "FN: - | TN: -"
        )

    def clear_segmentation_state(self) -> None:
        self.roi_rect = None
        self.seed_xy = None
        self.predicted_mask = None
        self.current_metrics = None
        self.current_parameters = None
        self.main_image_label.set_interaction_mode("none")
        self.display_all()
        self._reset_metrics_text()
        self._update_buttons()
        self.status_label.setText("ROI, semilla y segmentacion borradas. Puede dibujar una nueva ROI.")

    def save_current_result(self) -> None:
        if self.image is None or self.predicted_mask is None or self.image_path is None:
            return
        label = infer_label_from_path(self.image_path)
        image_id = sanitize_filename(self.image_path.stem)
        method = "unknown"
        if self.current_parameters is not None:
            method = str(self.current_parameters.get("method", "unknown"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_ROOT / label / image_id / f"{method}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_output_path = output_dir / "image_preprocessed.png"
        manual_output_path = output_dir / "manual_mask.png"
        predicted_output_path = output_dir / "predicted_mask.png"
        overlay_output_path = output_dir / "comparison_overlay.png"
        metadata_output_path = output_dir / "metrics_and_parameters.json"

        write_image(image_output_path, self.image)
        write_image(predicted_output_path, (self.predicted_mask.astype(np.uint8)) * 255)
        if self.manual_mask is not None:
            write_image(manual_output_path, (self.manual_mask.astype(np.uint8)) * 255)
        overlay = make_comparison_overlay(self.image, self.predicted_mask, self.manual_mask)
        write_image(overlay_output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        metadata = {
            "image_path": str(self.image_path),
            "manual_mask_path": str(self.manual_mask_path) if self.manual_mask_path else None,
            "roi_rect": list(self.roi_rect) if self.roi_rect else None,
            "seed_xy": list(self.seed_xy) if self.seed_xy else None,
            "method_parameters": self.current_parameters,
            "metrics": self.current_metrics,
            "outputs": {
                "image_preprocessed": str(image_output_path),
                "manual_mask": str(manual_output_path) if self.manual_mask is not None else None,
                "predicted_mask": str(predicted_output_path),
                "comparison_overlay": str(overlay_output_path),
            },
        }
        with open(metadata_output_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)
        self.status_label.setText(f"Resultado guardado en: {output_dir}")
        QMessageBox.information(self, "Resultado guardado", f"Resultado guardado en:\n{output_dir}")


def main() -> None:
    app = QApplication([])
    window = SegmentationGUI()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
