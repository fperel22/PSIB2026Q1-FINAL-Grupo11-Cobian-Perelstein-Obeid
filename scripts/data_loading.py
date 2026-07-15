"""
Bloque 1 - Carga y organizacion del dataset BUSI.

Este modulo hace solamente la primera etapa del proyecto:
1. Busca la carpeta del dataset BUSI.
2. Recorre las clases normal, benign y malignant.
3. Separa imagenes originales de mascaras manuales.
4. Asocia cada imagen con su mascara, si existe.
5. Construye una tabla de metadatos.
6. Visualiza ejemplos crudos en dominio espacial, histograma y dominio frecuencial.

No realiza preprocesamiento.
No realiza segmentacion automatica.
No modifica las imagenes.
"""

from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VALID_CLASSES = ["normal", "benign", "malignant"]


def find_busi_root(data_root: str | Path) -> Path:
    """
    Busca la carpeta raiz del dataset BUSI.

    La funcion acepta:
    - data/raw
    - data/raw/Dataset_BUSI_with_GT
    - una carpeta que contenga internamente Dataset_BUSI_with_GT

    Devuelve la carpeta que contiene:
    normal, benign y malignant.
    """

    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"No existe la ruta indicada: {data_root}")

    if all((data_root / cls).is_dir() for cls in VALID_CLASSES):
        return data_root

    candidate = data_root / "Dataset_BUSI_with_GT"
    if all((candidate / cls).is_dir() for cls in VALID_CLASSES):
        return candidate

    for folder in data_root.rglob("*"):
        if folder.is_dir() and all((folder / cls).is_dir() for cls in VALID_CLASSES):
            return folder

    raise FileNotFoundError(
        "No se encontro una estructura compatible con BUSI. "
        "La carpeta debe contener subcarpetas: normal, benign y malignant."
    )


def is_mask_file(path: Path) -> bool:
    """
    Indica si un archivo corresponde a una mascara manual.

    En BUSI, las mascaras suelen tener '_mask' en el nombre.
    Ejemplo:
    benign (1).png       -> imagen original
    benign (1)_mask.png  -> mascara manual
    """

    return "_mask" in path.stem.lower()


def find_masks_for_image(image_path: Path) -> List[Path]:
    """
    Busca las mascaras asociadas a una imagen.

    Para una imagen:
    benign (1).png

    busca archivos como:
    benign (1)_mask.png
    benign (1)_mask_1.png

    Devuelve una lista porque algunas imagenes pueden tener mas de una mascara.
    """

    image_stem = image_path.stem
    image_folder = image_path.parent

    mask_paths = sorted(image_folder.glob(f"{image_stem}_mask*.png"))

    return mask_paths


def read_grayscale_image(path: str | Path) -> np.ndarray:
    """
    Lee una imagen como escala de grises.

    Devuelve una matriz 2D de intensidades.
    """

    path = Path(path)

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")

    return image


def build_busi_metadata(data_root: str | Path) -> pd.DataFrame:
    """
    Construye una tabla con la informacion basica del dataset.

    Cada fila representa una imagen ecografica original.

    Columnas:
    - image_id: nombre de la imagen sin extension.
    - label: clase original del dataset.
    - image_path: ruta de la imagen.
    - mask_path: ruta de la mascara principal, si existe.
    - has_mask: True si se encontro mascara.
    - n_masks: cantidad de mascaras asociadas.
    - height: alto de la imagen.
    - width: ancho de la imagen.
    - n_pixels: numero total de pixeles.

    Esta tabla sera la entrada de los bloques posteriores:
    preprocesamiento, segmentacion y extraccion de caracteristicas.
    """

    busi_root = find_busi_root(data_root)

    records: List[Dict] = []

    for label in VALID_CLASSES:
        class_folder = busi_root / label

        image_paths = sorted(
            path for path in class_folder.glob("*.png")
            if not is_mask_file(path)
        )

        for image_path in image_paths:
            image = read_grayscale_image(image_path)
            height, width = image.shape

            mask_paths = find_masks_for_image(image_path)

            if len(mask_paths) > 0:
                main_mask_path: Optional[Path] = mask_paths[0]
            else:
                main_mask_path = None

            records.append(
                {
                    "image_id": image_path.stem,
                    "label": label,
                    "image_path": str(image_path),
                    "mask_path": str(main_mask_path) if main_mask_path is not None else "",
                    "has_mask": main_mask_path is not None,
                    "n_masks": len(mask_paths),
                    "height": height,
                    "width": width,
                    "n_pixels": height * width,
                }
            )

    metadata = pd.DataFrame(records)

    if metadata.empty:
        raise ValueError("No se encontraron imagenes originales en el dataset.")

    return metadata


def summarize_dataset(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Resume el dataset por clase.

    Permite verificar:
    - cuantas imagenes hay por clase;
    - cuantas tienen mascara;
    - que dimensiones tienen las imagenes.
    """

    summary = (
        metadata
        .groupby("label")
        .agg(
            n_images=("image_id", "count"),
            n_with_mask=("has_mask", "sum"),
            mean_height=("height", "mean"),
            mean_width=("width", "mean"),
            min_height=("height", "min"),
            max_height=("height", "max"),
            min_width=("width", "min"),
            max_width=("width", "max"),
        )
        .reset_index()
    )

    return summary


def compute_log_fourier_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Calcula el espectro de Fourier para visualizacion.

    La imagen se transforma al dominio frecuencial mediante FFT 2D.
    Luego se centra la baja frecuencia y se usa escala logaritmica para
    que el espectro sea visualmente interpretable.

    Esto no preprocesa la imagen.
    Solo sirve para exploracion inicial.
    """

    image_float = image.astype(np.float32)

    fourier = np.fft.fft2(image_float)
    fourier_shifted = np.fft.fftshift(fourier)

    magnitude = np.abs(fourier_shifted)
    log_magnitude = np.log1p(magnitude)

    return log_magnitude


def plot_raw_examples(
    metadata: pd.DataFrame,
    label: str,
    n_examples: int = 3,
    random_state: int = 0,
) -> None:
    """
    Muestra ejemplos crudos de una clase del dataset.

    Para cada imagen muestra:
    1. Imagen original en dominio espacial.
    2. Mascara manual, si existe.
    3. Histograma de intensidades.
    4. Espectro de Fourier en escala logaritmica.

    Esto sirve para contextualizar la base antes de preprocesar.
    """

    if label not in VALID_CLASSES:
        raise ValueError(f"Clase invalida: {label}")

    subset = metadata[metadata["label"] == label].copy()

    if subset.empty:
        raise ValueError(f"No hay imagenes para la clase: {label}")

    n_examples = min(n_examples, len(subset))
    subset = subset.sample(n=n_examples, random_state=random_state)

    for _, row in subset.iterrows():
        image = read_grayscale_image(row["image_path"])

        if row["has_mask"]:
            mask = read_grayscale_image(row["mask_path"])
        else:
            mask = np.zeros_like(image)

        spectrum = compute_log_fourier_spectrum(image)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title(f"Imagen original\n{row['label']} - {row['image_id']}")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mascara manual")
        axes[1].axis("off")

        axes[2].hist(image.ravel(), bins=64)
        axes[2].set_title("Histograma")
        axes[2].set_xlabel("Intensidad")
        axes[2].set_ylabel("Frecuencia")

        axes[3].imshow(spectrum, cmap="gray")
        axes[3].set_title("Fourier log-magnitud")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()


def save_metadata(metadata: pd.DataFrame, output_path: str | Path) -> None:
    """
    Guarda la tabla de metadatos en formato CSV.

    Este archivo documenta que imagenes fueron detectadas y como se
    asociaron con sus mascaras.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(output_path, index=False)


def main() -> None:
    """
    Ejecuta una prueba completa del bloque de carga.

    Debe correrse desde la raiz del repo:
    python src/data_loading.py
    """

    data_root = Path("data/raw")

    metadata = build_busi_metadata(data_root)
    summary = summarize_dataset(metadata)

    print("\nResumen del dataset BUSI:\n")
    print(summary)

    save_metadata(metadata, "outputs/tables/metadata_busi.csv")

    print("\nPrimeras filas de metadata:\n")
    print(metadata.head())

    print("\nMetadata guardada en outputs/tables/metadata_busi.csv")

    for label in VALID_CLASSES:
        print(f"\nVisualizando ejemplos crudos de la clase: {label}")
        plot_raw_examples(metadata, label=label, n_examples=2, random_state=42)


if __name__ == "__main__":
    main()