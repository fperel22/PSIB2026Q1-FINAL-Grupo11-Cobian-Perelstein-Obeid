"""
09_unet_inference.py

Genera mascaras automaticas con la U-Net ya entrenada, unicamente para las
imagenes marcadas como group="classifiers" en el manifest (train+val+test
de esa particion) -- es decir, las imagenes que la U-Net nunca vio durante
su entrenamiento. Estas mascaras son las que el resto del equipo usa en el
modulo de extraccion de features para el caso "mascara automatica".

Ademas calcula Dice/IoU contra la mascara manual (disponible para estas
imagenes aunque no se uso para entrenar), y guarda las metricas en
unet_metrics_classifiers_group.csv.

Uso:
    python 09_unet_inference.py \
        --manifest data/splits/manifest.csv \
        --data-dir data/processed/preprocessed/robust_bilateral \
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


def fill_holes(mask_bool):
    """Rellena huecos internos de una mascara binaria."""
    mask_u8 = mask_bool.astype(np.uint8) * 255
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, flood_inv)
    return filled > 0


def compute_dice_iou(pred_bool, manual_bool, eps=1e-6):
    intersection = np.logical_and(pred_bool, manual_bool).sum()
    union = np.logical_or(pred_bool, manual_bool).sum()
    dice = (2 * intersection + eps) / (pred_bool.sum() + manual_bool.sum() + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


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
    records = []
    with torch.no_grad():
        for it in items:
            img_path = os.path.join(args.data_dir, it["class"], "images", f'{it["filename"]}.png')
            mask_manual_path = os.path.join(args.data_dir, it["class"], "masks", f'{it["filename"]}_mask.png')

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
            mask_bool = prob > args.threshold
            if mask_bool.sum() > 0:
                mask_bool = fill_holes(mask_bool)

            mask_pred = (mask_bool.astype(np.uint8)) * 255
            mask_pred = cv2.resize(mask_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            out_path = os.path.join(args.output_dir, it["class"], f'{it["filename"]}_automask.png')
            cv2.imwrite(out_path, mask_pred)
            n_ok += 1

            manual_raw = cv2.imread(mask_manual_path, cv2.IMREAD_GRAYSCALE)
            if manual_raw is not None:
                manual_bool = manual_raw > 127
                dice, iou = compute_dice_iou(mask_pred > 127, manual_bool)
                records.append({
                    "filename": it["filename"], "class": it["class"],
                    "dice": round(dice, 4), "iou": round(iou, 4),
                })

    metrics_path = os.path.join(args.output_dir, "unet_metrics_classifiers_group.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "dice", "iou"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nListo. Mascaras generadas: {n_ok} | fallidas: {n_fail}")
    print(f"Guardadas en: {args.output_dir}")
    print(f"Metricas por imagen guardadas en: {metrics_path}")

    if records:
        for cls in ["benign", "malignant"]:
            cls_dice = [r["dice"] for r in records if r["class"] == cls]
            cls_iou = [r["iou"] for r in records if r["class"] == cls]
            if cls_dice:
                print(f"\n{cls}: n={len(cls_dice)} | Dice medio={np.mean(cls_dice):.4f}"
                      f" (std={np.std(cls_dice):.4f}) | IoU medio={np.mean(cls_iou):.4f}"
                      f" (std={np.std(cls_iou):.4f})")

        all_dice = [r["dice"] for r in records]
        all_iou = [r["iou"] for r in records]
        print(f"\nGlobal: n={len(all_dice)} | Dice medio={np.mean(all_dice):.4f}"
              f" (std={np.std(all_dice):.4f}) | IoU medio={np.mean(all_iou):.4f}"
              f" (std={np.std(all_iou):.4f})")

    print("\nEstas mascaras son el insumo para la extraccion de features 'caso automatico'.")


if __name__ == "__main__":
    main()
