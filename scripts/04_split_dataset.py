"""
00_split_dataset.py

Genera el split reproducible del dataset BUSI completo, compartido por todo
el equipo. Es CRITICO que este archivo se corra una sola vez, se suba al
repo (data/splits/manifest.csv) y que nadie lo vuelva a generar con otra
semilla, para que el split de la U-Net y el de los clasificadores sea
consistente entre los tres.

Logica del split:
  1. 50% de las imagenes (estratificado por clase) -> grupo "unet"
     - Se subdivide a su vez en train/val para entrenar la U-Net.
  2. 50% restante -> grupo "classifiers"
     - Se subdivide en train/val/test para los clasificadores (Modulo 2 en
       adelante). La U-Net NUNCA ve estas imagenes durante su entrenamiento;
       sobre ellas se generan mascaras automaticas en 09_unet_inference.py.

Uso:
    python 00_split_dataset.py \
        --images-dir data/processed/bilateral \
        --output data/splits/manifest.csv
"""
import argparse
import glob
import os
import csv
from sklearn.model_selection import train_test_split


def collect_images(data_dir):
    """Lista todas las imagenes con su clase, sin depender de las mascaras
    (el split se hace sobre imagenes; el chequeo de mascara existente se
    hace despues, en cada script que las use).

    Espera la estructura que genera 03_preprocessing_comparison.py:
        {data_dir}/{benign,malignant}/images/{id}.png
    """
    items = []
    for cls in ["benign", "malignant"]:
        for path in sorted(glob.glob(os.path.join(data_dir, cls, "images", "*.png"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            items.append({"filename": stem, "class": cls})
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                         help="Raiz de imagenes preprocesadas, ej: data/processed/preprocessed/robust_bilateral "
                              "(debe contener benign/images/ y malignant/images/)")
    parser.add_argument("--output", required=True,
                         help="Ruta del CSV de salida (manifest)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unet-fraction", type=float, default=0.5,
                         help="Fraccion del dataset total reservada para entrenar/validar la U-Net")
    parser.add_argument("--unet-val-fraction", type=float, default=0.15,
                         help="Fraccion (dentro del grupo unet) reservada para validacion")
    parser.add_argument("--clf-val-fraction", type=float, default=0.15,
                         help="Fraccion (dentro del grupo classifiers) reservada para validacion")
    parser.add_argument("--clf-test-fraction", type=float, default=0.15,
                         help="Fraccion (dentro del grupo classifiers) reservada para test")
    args = parser.parse_args()

    items = collect_images(args.data_dir)
    labels = [it["class"] for it in items]
    print(f"Total de imagenes encontradas: {len(items)}")
    print(f"  benignas: {labels.count('benign')} | malignas: {labels.count('malignant')}")

    # Paso 1: split 50/50 estratificado por clase -> unet vs classifiers
    unet_items, clf_items = train_test_split(
        items, train_size=args.unet_fraction, random_state=args.seed, stratify=labels
    )

    # Paso 2: dentro de "unet", split train/val
    unet_labels = [it["class"] for it in unet_items]
    unet_train, unet_val = train_test_split(
        unet_items, test_size=args.unet_val_fraction, random_state=args.seed, stratify=unet_labels
    )
    for it in unet_train:
        it["group"], it["subset"] = "unet", "train"
    for it in unet_val:
        it["group"], it["subset"] = "unet", "val"

    # Paso 3: dentro de "classifiers", split train/val/test
    clf_labels = [it["class"] for it in clf_items]
    clf_trainval, clf_test = train_test_split(
        clf_items, test_size=args.clf_test_fraction, random_state=args.seed, stratify=clf_labels
    )
    clf_trainval_labels = [it["class"] for it in clf_trainval]
    val_relative = args.clf_val_fraction / (1 - args.clf_test_fraction)
    clf_train, clf_val = train_test_split(
        clf_trainval, test_size=val_relative, random_state=args.seed, stratify=clf_trainval_labels
    )
    for it in clf_train:
        it["group"], it["subset"] = "classifiers", "train"
    for it in clf_val:
        it["group"], it["subset"] = "classifiers", "val"
    for it in clf_test:
        it["group"], it["subset"] = "classifiers", "test"

    all_items = unet_train + unet_val + clf_train + clf_val + clf_test

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "group", "subset"])
        writer.writeheader()
        writer.writerows(all_items)

    print(f"\nManifest guardado en: {args.output}")
    print(f"  unet/train:        {len(unet_train)}")
    print(f"  unet/val:          {len(unet_val)}")
    print(f"  classifiers/train: {len(clf_train)}")
    print(f"  classifiers/val:   {len(clf_val)}")
    print(f"  classifiers/test:  {len(clf_test)}")
    print("\nIMPORTANTE: subir este CSV al repo. Todo el equipo debe leer los")
    print("splits desde aca, no regenerarlos con otra semilla.")


if __name__ == "__main__":
    main()
