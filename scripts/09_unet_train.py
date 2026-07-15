"""
08_unet_train.py

Entrena una U-Net (encoder preentrenado en ImageNet) para segmentar lesiones
en ecografias mamarias, usando UNICAMENTE las imagenes marcadas como
group="unet" en el manifest generado por 04_split_dataset.py.

Uso local (con GPU):
    python 09_unet_train.py \
        --manifest data/splits/manifest.csv \
        --images-dir data/processed/bilateral \
        --masks-dir data/raw/masks \
        --output-dir outputs/unet \
        --epochs 40

Smoke test (subset chico, pocas epocas, para validar el pipeline y estimar
tiempos antes de lanzar el entrenamiento completo):
    python 08_unet_train.py --smoke-test --epochs 3 ... (mismos paths)

En Colab: montar Drive con los datos, clonar el repo para traer este
script, y correr con !python 08_unet_train.py --manifest ... (rutas dentro
de /content/drive/...).
"""
import argparse
import csv
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class BusiSegDataset(Dataset):
    def __init__(self, items, data_dir, img_size, augment=False):
        """data_dir: raiz de imagenes preprocesadas, ej:
        data/processed/preprocessed/robust_bilateral (con {clase}/images/
        y {clase}/masks/ adentro, como genera 03_preprocessing_comparison.py)
        """
        self.items = items
        self.data_dir = data_dir
        if augment:
            self.tf = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img_path = os.path.join(self.data_dir, it["class"], "images", f'{it["filename"]}.png')
        mask_path = os.path.join(self.data_dir, it["class"], "masks", f'{it["filename"]}_mask.png')

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"No se pudo leer {img_path} o {mask_path}")
        mask = (mask > 127).astype(np.float32)

        augmented = self.tf(image=image, mask=mask)
        return augmented["image"].float(), augmented["mask"].float().unsqueeze(0)


def load_manifest(manifest_path, group, subset=None):
    items = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["group"] != group:
                continue
            if subset is not None and row["subset"] != subset:
                continue
            items.append(row)
    return items


# --------------------------------------------------------------------------
# Modelo, loss, metricas
# --------------------------------------------------------------------------
def build_model(encoder_name, encoder_weights, device):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
        activation=None,
    )
    return model.to(device)


dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
bce_loss_fn = nn.BCEWithLogitsLoss()


def combined_loss(logits, target):
    return dice_loss_fn(logits, target) + bce_loss_fn(logits, target)


@torch.no_grad()
def dice_coefficient(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean().item()


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------
def train(model, train_loader, val_loader, num_epochs, lr, checkpoint_path,
          log_csv_path, device, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_dice = 0.0
    epochs_without_improvement = 0

    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "lr", "seconds"])

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            model.train()
            running_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = combined_loss(logits, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            train_loss = running_loss / len(train_loader.dataset)

            model.eval()
            val_loss, val_dice, n = 0.0, 0.0, 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    loss = combined_loss(logits, masks)
                    val_loss += loss.item() * images.size(0)
                    val_dice += dice_coefficient(logits, masks) * images.size(0)
                    n += images.size(0)
            val_loss /= n
            val_dice /= n
            scheduler.step(val_dice)

            improved = val_dice > best_dice
            if improved:
                best_dice = val_dice
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1

            dt = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                              f"{val_dice:.4f}", f"{current_lr:.2e}", f"{dt:.1f}"])
            f.flush()

            marker = " *" if improved else ""
            print(f"Epoca {epoch}/{num_epochs} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_dice={val_dice:.4f}{marker} | "
                  f"{dt:.1f}s/epoca")

            if epochs_without_improvement >= patience:
                print(f"Sin mejora en {patience} epocas consecutivas. Early stopping.")
                break

    return best_dice


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="CSV generado por 00_split_dataset.py")
    parser.add_argument("--data-dir", required=True,
                         help="Raiz de imagenes preprocesadas, ej: data/processed/preprocessed/robust_bilateral")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--smoke-test", action="store_true",
                         help="Corre sobre un subset chico, para validar el pipeline y estimar tiempos")
    parser.add_argument("--smoke-test-n", type=int, default=50)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("AVISO: no hay GPU disponible, el entrenamiento va a ser mucho mas lento.")

    os.makedirs(args.output_dir, exist_ok=True)

    train_items = load_manifest(args.manifest, group="unet", subset="train")
    val_items = load_manifest(args.manifest, group="unet", subset="val")
    print(f"Imagenes unet/train: {len(train_items)} | unet/val: {len(val_items)}")

    if args.smoke_test:
        train_items = random.sample(train_items, min(args.smoke_test_n, len(train_items)))
        val_items = random.sample(val_items, min(max(args.smoke_test_n // 4, 1), len(val_items)))
        print(f"[SMOKE TEST] usando {len(train_items)} train / {len(val_items)} val")

    train_ds = BusiSegDataset(train_items, args.data_dir, args.img_size, augment=True)
    val_ds = BusiSegDataset(val_items, args.data_dir, args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.encoder_name, args.encoder_weights, device)

    suffix = "smoke" if args.smoke_test else "best"
    checkpoint_path = os.path.join(args.output_dir, f"unet_{suffix}.pt")
    log_csv_path = os.path.join(args.output_dir, f"train_log_{suffix}.csv")

    best_dice = train(
        model, train_loader, val_loader,
        num_epochs=args.epochs, lr=args.lr,
        checkpoint_path=checkpoint_path, log_csv_path=log_csv_path,
        device=device, patience=args.patience,
    )

    print(f"\nMejor Dice en validacion: {best_dice:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log de entrenamiento: {log_csv_path}")
    if args.smoke_test:
        print("\nEsto fue un smoke test. Revisar el log y, si el Dice mejora y el")
        print("tiempo/epoca es razonable, correr sin --smoke-test para el entrenamiento completo.")


if __name__ == "__main__":
    main()
