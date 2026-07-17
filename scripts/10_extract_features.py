"""
10_extract_features.py

Modulo 10 - Extraccion unificada de caracteristicas por lesion.

Recibe una imagen preprocesada (bilateral robusto) + una mascara y devuelve un
vector de >= 20 caracteristicas en 4 familias:

    1. Intensidad     (reusa la logica de 06): media, desvio, skew, curtosis,
                       contraste lesion/fondo y Delta I perilesional.
    2. Morfologicas   (reusa 06 + solidez/extent nuevos): area relativa,
                       perimetro, circularidad, irregularidad (1-circ),
                       relacion ancho/alto, solidez y extent.
    3. GLCM/Haralick  (reusa 05): homogeneidad/IDM, contraste, entropia y
                       correlacion, sobre distancias {1,4} y angulos {0,90}.
    4. Wavelet (NUEVO, lo pidio la catedra): DWT 2D con PyWavelets sobre el
                       recorte (bounding box) de la lesion, PONDERANDO por la
                       mascara (ver nota abajo). Por subbanda (LL, LH, HL, HH)
                       y nivel: energia, entropia, media y curtosis.

Se corre SOLO sobre el grupo "classifiers" del manifest (las 315 imagenes que
la U-Net nunca vio), con DOS fuentes de mascara:

    - manual : la mascara de referencia ya alineada al preprocesado, en
               <preprocessed-dir>/<class>/masks/<filename>_mask.png
    - auto   : la mascara automatica de la U-Net, en
               <auto-masks-dir>/<class>/<filename>_automask.png

NOTA sobre "manual": se usa la mascara de referencia que ya quedo alineada
(mismo tamano) con la imagen preprocesada y con el naming del manifest
(<filename>_mask.png). Es la MISMA verdad de terreno del dataset BUSI, solo
que resampleada al preprocesado por el modulo 03 (y ya con los multi-mask
fusionados). Usarla directamente evita tener que reescalar y re-mapear los
nombres crudos "benign (N)_mask.png", y garantiza que la mascara manual y la
automatica esten a la MISMA resolucion -> las features quedan comparables
entre ambas fuentes (que es justo el objetivo del modulo 4 del plan). Se puede
apuntar a otra carpeta con --manual-masks-dir si se quisiera.

Salida:
    outputs/tables/features_manual.csv
    outputs/tables/features_auto.csv
Ambos CSV tienen las MISMAS filas (una por imagen, alineadas por el orden del
manifest) y las MISMAS columnas: filename, class, subset, mask_source,
mask_area_px, wavelet_levels + todas las features.

Guardrails:
    - Reusa el codigo de 05 y 06 (no reescribe esos calculos).
    - No toca el preprocesamiento ni la U-Net.
    - Mascaras vacias/degeneradas -> features en NaN, se registra mask_area_px
      y se loguea el caso (nunca numeros basura en silencio).

Ejecucion (desde la raiz del repo):
    # 1) chequeo rapido: imprime el vector de las primeras 4 imagenes
    python scripts/10_extract_features.py --limit 4

    # 2) corrida completa: escribe los dos CSV
    python scripts/10_extract_features.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Backend sin GUI: 05/06 importan matplotlib.pyplot a nivel de modulo.
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

# --------------------------------------------------------------------------- #
# Reuso de los modulos 05 y 06
# --------------------------------------------------------------------------- #
# Sus nombres empiezan con digito, asi que no se pueden importar con "import".
# Los cargamos por ruta con importlib. Importarlos NO ejecuta su main() (esta
# bajo if __name__ == "__main__"), solo define constantes y funciones.
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent


def _load_sibling_module(module_name: str, filename: str):
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo preparar el import de {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


glcm05 = _load_sibling_module("mod05_glcm", "05_texture_glcm_features.py")
char06 = _load_sibling_module("mod06_characterization", "06_lesion_ecographic_characterization.py")

# Se reutilizan tal cual (no se reescriben):
read_grayscale_image = glcm05.read_grayscale_image           # lectura Unicode-safe
ensure_same_shape = glcm05.ensure_same_shape
quantize_image = glcm05.quantize_image                       # cuantizacion GLCM
masked_glcm = glcm05.masked_glcm                             # GLCM enmascarada (05)
glcm_features = glcm05.glcm_features                         # 4 descriptores GLCM (05)
N_GRAY_LEVELS = glcm05.N_GRAY_LEVELS
DISTANCES = glcm05.DISTANCES                                 # [1, 4]
ANGLES_DEGREES = glcm05.ANGLES_DEGREES                       # [0, 90]

get_valid_ultrasound_region = char06.get_valid_ultrasound_region   # region valida (06)
get_perilesional_band = char06.get_perilesional_band               # banda perilesional (06)
contour_perimeter = char06.contour_perimeter                       # perimetro por contorno (06)
bounding_box_aspect_ratio = char06.bounding_box_aspect_ratio       # ancho/alto (06)

# --------------------------------------------------------------------------- #
# Configuracion del modulo
# --------------------------------------------------------------------------- #
EPS = 1e-8

# --- Wavelet ---------------------------------------------------------------- #
# Wavelet: db2 (Daubechies-2). Justificacion:
#   - Soporte compacto (4 taps) -> sigue siendo local, no promedia texturas
#     lejanas dentro de la lesion.
#   - Tiene 1 momento nulo mas que Haar -> es algo mas suave y captura mejor
#     los gradientes de textura/speckle tipicos del ultrasonido, en vez de
#     responder solo a saltos tipo escalon como Haar.
#   Es una eleccion estandar y de bajo costo para caracterizar textura US.
WAVELET = "db2"
# Nivel: objetivo 2. Se limita automaticamente segun el tamano del bounding box
# (pywt.dwt_max_level) para no descomponer de mas en lesiones chicas. Si ni
# siquiera entra 1 nivel (lesion diminuta o mascara vacia) -> features wavelet NaN.
MAX_WAVELET_LEVEL = 2
WAVELET_MODE = "symmetric"  # extension de borde; evita el wrap-around de 'periodization'

# Area minima (px) para considerar la mascara utilizable. Por debajo de esto la
# lesion es demasiado chica para dar GLCM/wavelet con sentido -> se marca NaN.
MIN_LESION_AREA_PX = 16

# Fuentes de mascara soportadas.
MASK_SOURCES = ("manual", "auto")

# Defaults de rutas (relativas a la raiz del repo, robustas a la CWD).
DEFAULT_MANIFEST = REPO_ROOT / "data" / "splits" / "manifest.csv"
DEFAULT_PREPROCESSED_DIR = REPO_ROOT / "data" / "processed" / "preprocessed" / "robust_bilateral"
DEFAULT_AUTO_MASKS_DIR = REPO_ROOT / "data" / "processed" / "auto_masks"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "tables"
DEFAULT_GROUP = "classifiers"


# --------------------------------------------------------------------------- #
# Resolucion de rutas por imagen
# --------------------------------------------------------------------------- #
def image_path(preprocessed_dir: Path, cls: str, filename: str) -> Path:
    return preprocessed_dir / cls / "images" / f"{filename}.png"


def manual_mask_path(manual_masks_dir: Path, cls: str, filename: str) -> Path:
    return manual_masks_dir / cls / "masks" / f"{filename}_mask.png"


def auto_mask_path(auto_masks_dir: Path, cls: str, filename: str) -> Path:
    return auto_masks_dir / cls / f"{filename}_automask.png"


# --------------------------------------------------------------------------- #
# Familia 1 - Intensidad (reusa la logica de 06)
# --------------------------------------------------------------------------- #
def intensity_features(image: np.ndarray, mask_bool: np.ndarray) -> Dict[str, float]:
    """
    media/desvio/skew/curtosis dentro de la lesion + contraste lesion/fondo y
    Delta I perilesional, estos ultimos con las MISMAS definiciones del modulo 06.
    """
    lesion_vals = image[mask_bool].astype(np.float64)
    lesion_mean = float(np.mean(lesion_vals))
    lesion_std = float(np.std(lesion_vals))
    # skew/curtosis: NaN si no hay dispersion (evita divisiones 0/0 ruidosas).
    if lesion_std <= EPS or lesion_vals.size < 3:
        lesion_skew = np.nan
        lesion_kurt = np.nan
    else:
        lesion_skew = float(scipy_skew(lesion_vals))
        lesion_kurt = float(scipy_kurtosis(lesion_vals, fisher=True))  # exceso

    # --- fondo y banda perilesional: definiciones de 06 ---
    valid_region = get_valid_ultrasound_region(image)
    background = valid_region & (~mask_bool)
    band, _ = get_perilesional_band(mask_bool, valid_region)

    background_mean = float(np.mean(image[background])) if background.sum() > 0 else np.nan
    perilesional_mean = float(np.mean(image[band])) if band.sum() > 0 else np.nan

    # contraste lesion/fondo == relative_intensity_lesion_background de 06
    if np.isfinite(background_mean):
        contrast_lesion_bg = (background_mean - lesion_mean) / (background_mean + EPS)
    else:
        contrast_lesion_bg = np.nan
    # Delta I perilesional == perilesional_minus_lesion de 06
    delta_perilesional = (perilesional_mean - lesion_mean) if np.isfinite(perilesional_mean) else np.nan

    return {
        "int_mean": lesion_mean,
        "int_std": lesion_std,
        "int_skew": lesion_skew,
        "int_kurtosis": lesion_kurt,
        "int_contrast_lesion_bg": float(contrast_lesion_bg) if np.isfinite(contrast_lesion_bg) else np.nan,
        "int_delta_perilesional": float(delta_perilesional) if np.isfinite(delta_perilesional) else np.nan,
    }


# --------------------------------------------------------------------------- #
# Familia 2 - Morfologicas (reusa 06 + solidez/extent nuevos)
# --------------------------------------------------------------------------- #
def morphological_features(image: np.ndarray, mask_bool: np.ndarray) -> Dict[str, float]:
    area = float(mask_bool.sum())

    # area relativa respecto de la region valida de ecografia (definicion de 06).
    valid_region = get_valid_ultrasound_region(image)
    valid_area = float(valid_region.sum())
    area_relative = area / (valid_area + EPS) if valid_area > 0 else np.nan

    # perimetro / circularidad / irregularidad / ancho-alto: funciones de 06.
    perimeter = contour_perimeter(mask_bool)
    circularity = (4.0 * np.pi * area) / (perimeter ** 2 + EPS) if area > 0 and perimeter > 0 else np.nan
    irregularity = (1.0 - circularity) if np.isfinite(circularity) else np.nan
    aspect_ratio = bounding_box_aspect_ratio(mask_bool)

    # solidez = area / area(convex hull)  y  extent = area / area(bbox)  (nuevos).
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        solidity = np.nan
        extent = np.nan
    else:
        x_min, y_min = int(xs.min()), int(ys.min())
        bbox_w = float(xs.max() - x_min + 1)
        bbox_h = float(ys.max() - y_min + 1)
        extent = area / (bbox_w * bbox_h + EPS)
        # solidez = area / area(convex hull). El hull se rasteriza y se cuentan
        # pixeles (misma unidad que 'area') para que la solidez quede acotada en
        # (0,1]; usar cv2.contourArea del poligono da un area levemente menor por
        # el medio pixel de borde y puede dar solidez > 1.
        points = np.column_stack([xs - x_min, ys - y_min]).astype(np.int32)
        if points.shape[0] >= 3:
            hull = cv2.convexHull(points)
            hull_canvas = np.zeros((int(bbox_h), int(bbox_w)), dtype=np.uint8)
            cv2.fillConvexPoly(hull_canvas, hull, 1)
            hull_area = float(hull_canvas.sum())
            solidity = area / (hull_area + EPS) if hull_area > 0 else np.nan
        else:
            solidity = np.nan

    return {
        "morph_area_relative": float(area_relative) if np.isfinite(area_relative) else np.nan,
        "morph_perimeter_px": float(perimeter),
        "morph_circularity": float(circularity) if np.isfinite(circularity) else np.nan,
        "morph_irregularity": float(irregularity) if np.isfinite(irregularity) else np.nan,
        "morph_aspect_ratio_wh": float(aspect_ratio) if np.isfinite(aspect_ratio) else np.nan,
        "morph_solidity": float(solidity) if np.isfinite(solidity) else np.nan,
        "morph_extent": float(extent) if np.isfinite(extent) else np.nan,
    }


# --------------------------------------------------------------------------- #
# Familia 3 - GLCM/Haralick (reusa 05)
# --------------------------------------------------------------------------- #
def glcm_texture_features(image: np.ndarray, mask_bool: np.ndarray) -> Dict[str, float]:
    """
    Para cada distancia {1,4}: se promedia la GLCM enmascarada de 05 sobre los
    angulos {0,90} (robustez a la orientacion) y se calculan los 4 descriptores
    de 05: homogeneidad/IDM, contraste, entropia y correlacion.
    """
    image_q = quantize_image(image, N_GRAY_LEVELS)
    feats: Dict[str, float] = {}
    for distance in DISTANCES:
        glcm_acc = np.zeros((N_GRAY_LEVELS, N_GRAY_LEVELS), dtype=np.float64)
        n_valid = 0
        for angle in ANGLES_DEGREES:
            glcm = masked_glcm(image_q, mask_bool, distance, angle, N_GRAY_LEVELS)
            if glcm.sum() > 0:
                glcm_acc += glcm
                n_valid += 1
        if n_valid > 0:
            glcm_acc /= n_valid  # promedio de distribuciones -> vuelve a sumar 1
            descriptors = glcm_features(glcm_acc)
        else:
            descriptors = {"homogeneity_idm": np.nan, "contrast": np.nan, "entropy": np.nan, "correlation": np.nan}
        feats[f"glcm_homogeneity_d{distance}"] = descriptors["homogeneity_idm"]
        feats[f"glcm_contrast_d{distance}"] = descriptors["contrast"]
        feats[f"glcm_entropy_d{distance}"] = descriptors["entropy"]
        feats[f"glcm_correlation_d{distance}"] = descriptors["correlation"]
    return feats


# --------------------------------------------------------------------------- #
# Familia 4 - Wavelet (NUEVO)
# --------------------------------------------------------------------------- #
WAVELET_SUBBANDS = ("LL", "LH", "HL", "HH")
WAVELET_STATS = ("energy", "entropy", "mean", "kurt")


def _wavelet_feature_keys(max_level: int) -> List[str]:
    keys = []
    for level in range(1, max_level + 1):
        for subband in WAVELET_SUBBANDS:
            for stat in WAVELET_STATS:
                keys.append(f"wav_L{level}_{subband}_{stat}")
    return keys


def _weighted_subband_stats(coeffs: np.ndarray, weight: np.ndarray) -> Dict[str, float]:
    """
    Estadisticos de una subbanda ponderados por la cobertura de la mascara.
    weight en [0,1] tiene la MISMA forma que coeffs (mascara reescalada a la
    resolucion de la subbanda). Asi los coeficientes que provienen del exterior
    de la lesion pesan ~0 y no contaminan energia/entropia/curtosis.
    """
    w = weight.astype(np.float64)
    c = coeffs.astype(np.float64)
    sw = float(w.sum())
    if sw <= EPS:
        return {"energy": np.nan, "entropy": np.nan, "mean": np.nan, "kurt": np.nan}

    mean = float((w * c).sum() / sw)
    energy = float((w * c * c).sum() / sw)        # energia media por pixel de lesion
    var = float((w * (c - mean) ** 2).sum() / sw)
    if var <= EPS:
        kurt = np.nan
    else:
        m4 = float((w * (c - mean) ** 4).sum() / sw)
        kurt = float(m4 / (var * var) - 3.0)      # curtosis en exceso (Fisher)

    # entropia de la distribucion de energia dentro de la lesion.
    p = w * (c * c)
    ptot = float(p.sum())
    if ptot <= EPS:
        entropy = np.nan
    else:
        p = p / ptot
        nz = p > 0
        entropy = float(-(p[nz] * np.log2(p[nz])).sum())

    return {"energy": energy, "entropy": entropy, "mean": mean, "kurt": kurt}


def wavelet_features(
    image: np.ndarray,
    mask_bool: np.ndarray,
    wavelet: str = WAVELET,
    max_level: int = MAX_WAVELET_LEVEL,
) -> Tuple[Dict[str, float], int]:
    """
    DWT 2D sobre el bounding box de la lesion, ponderando por la mascara.

    Clave (lo pidio la catedra): NO se rellena con ceros fuera de la mascara,
    porque un borde a cero crea un escalon artificial que contamina LH/HL/HH
    (las subbandas de alta frecuencia que justamente interesan). En su lugar,
    doble estrategia:
      1) el exterior de la lesion (dentro del bbox) se rellena con la MEDIA de
         intensidad de la lesion -> elimina el salto DC del borde;
      2) cada subbanda se PONDERA por la mascara reescalada a su resolucion, de
         modo que los coeficientes de borde/exterior aportan ~0.

    Devuelve (features, niveles_calculados). Las columnas siguen un esquema fijo
    (niveles 1..max_level); si una lesion es muy chica y solo llega a nivel 1,
    las columnas de nivel 2 quedan en NaN (NaN esperado, se loguea el conteo).
    """
    feats = {k: np.nan for k in _wavelet_feature_keys(max_level)}

    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return feats, 0

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = image[y0:y1, x0:x1].astype(np.float64)
    mcrop = mask_bool[y0:y1, x0:x1].astype(np.float64)  # 0/1

    lesion_vals = crop[mcrop > 0]
    if lesion_vals.size == 0:
        return feats, 0

    # (1) relleno con la media de la lesion (no ceros).
    filled = crop.copy()
    filled[mcrop <= 0] = float(lesion_vals.mean())

    # nivel factible segun tamano del bbox y largo del filtro.
    dec_len = pywt.Wavelet(wavelet).dec_len
    feasible = pywt.dwt_max_level(int(min(crop.shape)), dec_len)
    levels = int(max(0, min(max_level, feasible)))
    if levels < 1:
        return feats, 0

    approx = filled
    for level in range(1, levels + 1):
        cA, (cH, cV, cD) = pywt.dwt2(approx, wavelet, mode=WAVELET_MODE)
        # pywt: cA=aproximacion (LL), cH=detalle horizontal (LH),
        #       cV=detalle vertical (HL), cD=detalle diagonal (HH).
        subbands = {"LL": cA, "LH": cH, "HL": cV, "HH": cD}
        for name, coef in subbands.items():
            # (2) peso = cobertura fraccional de la mascara a la resolucion de la
            # subbanda (INTER_AREA da fracciones 0..1 por promedio de area).
            wsb = cv2.resize(mcrop, (coef.shape[1], coef.shape[0]), interpolation=cv2.INTER_AREA)
            stats = _weighted_subband_stats(coef, wsb)
            for stat, value in stats.items():
                feats[f"wav_L{level}_{name}_{stat}"] = value
        approx = cA  # la LL alimenta el siguiente nivel

    return feats, levels


# --------------------------------------------------------------------------- #
# Ensamble por caso
# --------------------------------------------------------------------------- #
FEATURE_ORDER = (
    # intensidad
    ["int_mean", "int_std", "int_skew", "int_kurtosis", "int_contrast_lesion_bg", "int_delta_perilesional"]
    # morfologicas
    + ["morph_area_relative", "morph_perimeter_px", "morph_circularity", "morph_irregularity",
       "morph_aspect_ratio_wh", "morph_solidity", "morph_extent"]
    # glcm
    + [f"glcm_{m}_d{d}" for d in DISTANCES for m in ("homogeneity", "contrast", "entropy", "correlation")]
    # wavelet
    + _wavelet_feature_keys(MAX_WAVELET_LEVEL)
)

META_COLUMNS = ["filename", "class", "subset", "mask_source", "mask_area_px", "wavelet_levels"]
ALL_COLUMNS = META_COLUMNS + FEATURE_ORDER


def _nan_feature_row() -> Dict[str, float]:
    return {k: np.nan for k in FEATURE_ORDER}


def extract_case_features(
    image: np.ndarray,
    mask: np.ndarray,
    wavelet: str = WAVELET,
    max_level: int = MAX_WAVELET_LEVEL,
) -> Tuple[Dict[str, float], int, int, str]:
    """
    Devuelve (features, mask_area_px, wavelet_levels, status).
    status in {"ok", "empty", "degenerate"}. Si la mascara esta vacia o es
    degenerada (area < MIN_LESION_AREA_PX), todas las features van a NaN.
    """
    mask = ensure_same_shape(mask, image.shape)
    mask_bool = mask > 127
    area_px = int(mask_bool.sum())

    if area_px == 0:
        return _nan_feature_row(), 0, 0, "empty"
    if area_px < MIN_LESION_AREA_PX:
        return _nan_feature_row(), area_px, 0, "degenerate"

    feats: Dict[str, float] = {}
    feats.update(intensity_features(image, mask_bool))
    feats.update(morphological_features(image, mask_bool))
    feats.update(glcm_texture_features(image, mask_bool))
    wav_feats, wav_levels = wavelet_features(image, mask_bool, wavelet=wavelet, max_level=max_level)
    feats.update(wav_feats)
    return feats, area_px, wav_levels, "ok"


# --------------------------------------------------------------------------- #
# Manifest y corrida por fuente
# --------------------------------------------------------------------------- #
def load_classifiers_manifest(manifest_path: Path, group: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    subset = manifest[manifest["group"] == group].copy()
    # Orden estable para que las filas queden alineadas entre manual y auto.
    subset = subset.sort_values(["class", "filename"]).reset_index(drop=True)
    return subset[["filename", "class", "subset"]]


def build_feature_table(
    manifest_rows: pd.DataFrame,
    mask_source: str,
    preprocessed_dir: Path,
    manual_masks_dir: Path,
    auto_masks_dir: Path,
    wavelet: str,
    max_level: int,
    limit: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    rows = manifest_rows if limit <= 0 else manifest_rows.head(limit)
    records: List[Dict[str, object]] = []
    n_empty = n_degenerate = n_partial_wavelet = 0

    for _, r in rows.iterrows():
        filename, cls, subset = r["filename"], r["class"], r["subset"]
        img_p = image_path(preprocessed_dir, cls, filename)
        mask_p = (manual_mask_path(manual_masks_dir, cls, filename) if mask_source == "manual"
                  else auto_mask_path(auto_masks_dir, cls, filename))

        record: Dict[str, object] = {
            "filename": filename, "class": cls, "subset": subset, "mask_source": mask_source,
        }

        if not img_p.exists() or not mask_p.exists():
            print(f"[AVISO] falta archivo ({'img' if not img_p.exists() else 'mask'}): "
                  f"{img_p if not img_p.exists() else mask_p} -> fila en NaN")
            record.update({"mask_area_px": 0, "wavelet_levels": 0})
            record.update(_nan_feature_row())
            records.append(record)
            continue

        image = read_grayscale_image(img_p)
        mask = read_grayscale_image(mask_p)
        feats, area_px, wav_levels, status = extract_case_features(image, mask, wavelet, max_level)

        record.update({"mask_area_px": area_px, "wavelet_levels": wav_levels})
        record.update(feats)
        records.append(record)

        if status == "empty":
            n_empty += 1
            print(f"[MASCARA VACIA] {mask_source}: {cls}/{filename} (area=0) -> features en NaN")
        elif status == "degenerate":
            n_degenerate += 1
            print(f"[MASCARA DEGENERADA] {mask_source}: {cls}/{filename} "
                  f"(area={area_px}px < {MIN_LESION_AREA_PX}) -> features en NaN")
        elif wav_levels < max_level:
            n_partial_wavelet += 1

    df = pd.DataFrame.from_records(records)
    df = df.reindex(columns=ALL_COLUMNS)  # orden de columnas fijo

    if verbose:
        print(f"\n[{mask_source}] filas={len(df)} | mascaras vacias={n_empty} | "
              f"degeneradas={n_degenerate} | wavelet<{max_level} niveles={n_partial_wavelet}")
    return df


# --------------------------------------------------------------------------- #
# Validacion / diagnostico
# --------------------------------------------------------------------------- #
def print_feature_vectors(df: pd.DataFrame, n: int = 4) -> None:
    print(f"\n===== Vector de features (primeras {min(n, len(df))} imagenes) =====")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", lambda x: f"{x:.4f}"):
        for _, r in df.head(n).iterrows():
            print(f"\n--- {r['mask_source']} | {r['class']}/{r['filename']} "
                  f"(subset={r['subset']}, area={r['mask_area_px']}px, wav_levels={r['wavelet_levels']}) ---")
            for col in FEATURE_ORDER:
                print(f"    {col:28s} = {r[col]:.6g}" if pd.notna(r[col]) else f"    {col:28s} = NaN")


def validate_tables(df_manual: pd.DataFrame, df_auto: pd.DataFrame) -> None:
    print("\n===== Validacion =====")
    print(f"manual: {df_manual.shape}  |  auto: {df_auto.shape}")
    assert list(df_manual.columns) == list(df_auto.columns), "Las columnas no coinciden entre manual y auto."
    assert len(df_manual) == len(df_auto), "Distinta cantidad de filas."
    aligned = (df_manual[["filename", "class", "subset"]].reset_index(drop=True)
               .equals(df_auto[["filename", "class", "subset"]].reset_index(drop=True)))
    print(f"filas alineadas (filename/class/subset identicos): {aligned}")
    assert aligned, "Las filas no estan alineadas entre manual y auto."

    for name, df in [("manual", df_manual), ("auto", df_auto)]:
        feat = df[FEATURE_ORDER]
        n_inf = int(np.isinf(feat.to_numpy(dtype=np.float64)).sum())
        n_nan_rows = int(feat.isna().any(axis=1).sum())
        empty_rows = int((df["mask_area_px"] == 0).sum())
        print(f"\n[{name}] inf en features: {n_inf} | filas con algun NaN: {n_nan_rows} "
              f"(mascaras vacias/degeneradas explican {empty_rows} vacias)")
        # columnas con NaN inesperado (filas con mascara valida y NaN igualmente).
        valid = df["mask_area_px"] >= MIN_LESION_AREA_PX
        # nivel 2 wavelet puede ser NaN legitimamente si wavelet_levels < 2.
        unexpected = {}
        for col in FEATURE_ORDER:
            bad = valid & df[col].isna()
            if col.startswith("wav_L2_"):
                bad = bad & (df["wavelet_levels"] >= 2)
            cnt = int(bad.sum())
            if cnt > 0:
                unexpected[col] = cnt
        if unexpected:
            print(f"    [!] NaN inesperado (mascara valida) por columna: {unexpected}")
        else:
            print("    sin NaN inesperado en mascaras validas.")
        assert n_inf == 0, f"Hay valores inf en {name}."


def print_class_separation(df: pd.DataFrame, source: str, cols: List[str]) -> None:
    """Chequeo de que las features (en particular wavelet) varian entre clases."""
    print(f"\n===== Separacion benigno vs maligno ({source}) =====")
    g = df.groupby("class")[cols].mean(numeric_only=True)
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", lambda x: f"{x:.4f}"):
        print(g.T)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR,
                   help="Raiz de imagenes preprocesadas: <dir>/<class>/images/<filename>.png")
    p.add_argument("--manual-masks-dir", type=Path, default=None,
                   help="Raiz de mascaras manuales alineadas (default: = preprocessed-dir, "
                        "usa <dir>/<class>/masks/<filename>_mask.png)")
    p.add_argument("--auto-masks-dir", type=Path, default=DEFAULT_AUTO_MASKS_DIR,
                   help="Raiz de mascaras automaticas: <dir>/<class>/<filename>_automask.png")
    p.add_argument("--group", default=DEFAULT_GROUP, help="Grupo del manifest a procesar (default: classifiers)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--wavelet", default=WAVELET)
    p.add_argument("--max-level", type=int, default=MAX_WAVELET_LEVEL)
    p.add_argument("--limit", type=int, default=0,
                   help="Procesa solo las primeras N imagenes (para pruebas). 0 = todas.")
    p.add_argument("--preview", type=int, default=4,
                   help="Imprime el vector de features de las primeras N imagenes.")
    p.add_argument("--no-write", action="store_true",
                   help="No escribe los CSV (solo procesa e imprime; util con --limit).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manual_masks_dir = args.manual_masks_dir if args.manual_masks_dir is not None else args.preprocessed_dir

    print(f"Manifest        : {args.manifest}")
    print(f"Preprocesadas   : {args.preprocessed_dir}")
    print(f"Mascaras manual : {manual_masks_dir}/<class>/masks/<filename>_mask.png")
    print(f"Mascaras auto   : {args.auto_masks_dir}/<class>/<filename>_automask.png")
    print(f"Wavelet         : {args.wavelet}  |  niveles objetivo: {args.max_level}")
    print(f"Grupo           : {args.group}")

    manifest_rows = load_classifiers_manifest(args.manifest, args.group)
    print(f"Imagenes en grupo '{args.group}': {len(manifest_rows)} "
          f"({manifest_rows['class'].value_counts().to_dict()})")

    tables: Dict[str, pd.DataFrame] = {}
    for source in MASK_SOURCES:
        print(f"\n########## Fuente de mascara: {source} ##########")
        df = build_feature_table(
            manifest_rows, source, args.preprocessed_dir, manual_masks_dir, args.auto_masks_dir,
            args.wavelet, args.max_level, limit=args.limit,
        )
        tables[source] = df
        if args.preview > 0:
            print_feature_vectors(df, n=args.preview)

    df_manual, df_auto = tables["manual"], tables["auto"]
    validate_tables(df_manual, df_auto)

    # Chequeo pedido: que las features wavelet varien entre clases.
    wavelet_check_cols = ["wav_L1_LH_energy", "wav_L1_HH_energy", "wav_L1_HL_energy",
                          "wav_L2_HH_energy", "wav_L1_HH_entropy"]
    wavelet_check_cols = [c for c in wavelet_check_cols if c in df_manual.columns]
    print_class_separation(df_manual, "manual", wavelet_check_cols)

    if args.no_write:
        print("\n--no-write: no se escriben CSV.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_manual = args.output_dir / "features_manual.csv"
    out_auto = args.output_dir / "features_auto.csv"
    df_manual.to_csv(out_manual, index=False)
    df_auto.to_csv(out_auto, index=False)
    print(f"\nGuardado: {out_manual}")
    print(f"Guardado: {out_auto}")
    print(f"Columnas: {len(ALL_COLUMNS)} ({len(FEATURE_ORDER)} features + {len(META_COLUMNS)} meta)")
    print("\nListo.")


if __name__ == "__main__":
    main()
