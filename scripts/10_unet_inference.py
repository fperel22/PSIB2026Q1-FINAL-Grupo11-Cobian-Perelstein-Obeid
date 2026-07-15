"""
09_unet_inference.py

Genera mascaras automaticas con la U-Net ya entrenada, unicamente para las
imagenes marcadas como group="classifiers" en el manifest (train+val+test
de esa particion) -- es decir, las imagenes que la U-Net nunca vio durante
su entrenamiento. Estas mascaras son las que el resto del equipo usa en el
modulo de extraccion de features para el caso "mascara automatica".

Uso:
    python 10_unet_inference.py \
        --manifest data/splits/manifest.csv \
        --images-dir data/processed/bilateral \
        --checkpoint outputs/unet/unet_best.pt \
        --output-dir data/processed/auto_masks
"""
import argparse
import csv
import os

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


def load_manifest(manifest_path, group):
    items = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["group"] == group:
                items.append(row)
    return items


def build_model(encoder_name, checkpoint_path, device):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,  # se cargan los pesos entrenados, no ImageNet
        in_channels=1,
        classes=1,
        activation=None,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True,
                         help="Raiz de imagenes preprocesadas, ej: data/processed/preprocessed/robust_bilateral")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(args.encoder_name, args.checkpoint, device)

    tf = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

    items = load_manifest(args.manifest, group="classifiers")
    print(f"Generando mascaras automaticas para {len(items)} imagenes (grupo classifiers)")

    for cls in ["benign", "malignant"]:
        os.makedirs(os.path.join(args.output_dir, cls), exist_ok=True)

    n_ok, n_fail = 0, 0
    with torch.no_grad():
        for it in items:
            img_path = os.path.join(args.data_dir, it["class"], "images", f'{it["filename"]}.png')
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"[AVISO] no se pudo leer {img_path}, se omite")
                n_fail += 1
                continue

            orig_h, orig_w = image.shape[:2]
            augmented = tf(image=image)
            input_t = augmented["image"].float().unsqueeze(0).to(device)

            logits = model(input_t)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            mask_pred = (prob > args.threshold).astype(np.uint8) * 255

            # Volver al tamano original de la imagen para que sea comparable
            # con la mascara manual y con el resto del pipeline de features
            mask_pred = cv2.resize(mask_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            out_path = os.path.join(args.output_dir, it["class"], f'{it["filename"]}_automask.png')
            cv2.imwrite(out_path, mask_pred)
            n_ok += 1

    print(f"\nListo. Mascaras generadas: {n_ok} | fallidas: {n_fail}")
    print(f"Guardadas en: {args.output_dir}")
    print("\nEstas mascaras son el insumo para la extraccion de features 'caso automatico'.")


if __name__ == "__main__":
    main()
