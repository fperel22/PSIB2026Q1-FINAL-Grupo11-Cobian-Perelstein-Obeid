"""
13_train_efficientnet.py

Transfer learning con EfficientNet-B0 para clasificación binaria:
- ROI obtenida con máscara manual
- ROI obtenida con máscara automática

La lista de casos elegibles se toma de:
    outputs/tables_aligned/features_auto.csv

Esto obliga a que los dos modelos usen exactamente los mismos casos:
    train=216, val=48, test=47

Reglas:
- train y val se usan para entrenamiento/selección.
- test solo se evalúa cuando se agrega --evaluate-test.
- No se ocultan máscaras inválidas mediante fallback.
- Usa la interfaz AMP vigente de PyTorch.
"""

from __future__ import annotations

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


CLASS_MAP = {"benign": 0, "malignant": 1}
SCRIPT_VERSION = "13-clean-v2"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_eligible_filenames(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Primero generá outputs/tables_aligned."
        )

    df = pd.read_csv(path)
    if "filename" not in df.columns:
        raise ValueError(f"{path} no contiene la columna filename.")

    return set(df["filename"].astype(str))


def load_manifest_items(
    manifest_path: Path,
    subset: str,
    eligible_filenames: set[str],
) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    items = [
        row
        for row in rows
        if row["group"] == "classifiers"
        and row["subset"] == subset
        and row["filename"] in eligible_filenames
    ]

    return items


def stratified_sample(
    items: list[dict[str, str]],
    n: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)

    benign = [item for item in items if item["class"] == "benign"]
    malignant = [item for item in items if item["class"] == "malignant"]

    rng.shuffle(benign)
    rng.shuffle(malignant)

    total = max(len(items), 1)
    n_benign = max(1, round(n * len(benign) / total))
    n_malignant = max(1, n - n_benign)

    sampled = (
        benign[: min(n_benign, len(benign))]
        + malignant[: min(n_malignant, len(malignant))]
    )
    rng.shuffle(sampled)
    return sampled


def crop_masked_roi(
    image: np.ndarray,
    mask: np.ndarray,
    padding_fraction: float,
) -> np.ndarray:
    mask_bool = mask > 127

    if int(mask_bool.sum()) < 16:
        raise ValueError(
            "Máscara vacía o degenerada dentro del conjunto elegible."
        )

    ys, xs = np.where(mask_bool)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    side = max(x_max - x_min + 1, y_max - y_min + 1)
    padding = int(round(side * padding_fraction))

    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1] - 1, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0] - 1, y_max + padding)

    masked = image.copy()
    masked[~mask_bool] = 0

    roi = masked[y_min : y_max + 1, x_min : x_max + 1]
    if roi.size == 0:
        raise ValueError("ROI vacía después del recorte.")

    return roi


class BusiROIDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, str]],
        preprocessed_dir: Path,
        auto_masks_dir: Path,
        mask_source: str,
        transform,
        padding_fraction: float,
    ) -> None:
        self.items = items
        self.preprocessed_dir = preprocessed_dir
        self.auto_masks_dir = auto_masks_dir
        self.mask_source = mask_source
        self.transform = transform
        self.padding_fraction = padding_fraction

    def __len__(self) -> int:
        return len(self.items)

    def mask_path(self, item: dict[str, str]) -> Path:
        if self.mask_source == "manual":
            return (
                self.preprocessed_dir
                / item["class"]
                / "masks"
                / f"{item['filename']}_mask.png"
            )

        return (
            self.auto_masks_dir
            / item["class"]
            / f"{item['filename']}_automask.png"
        )

    def __getitem__(self, index: int):
        item = self.items[index]

        image_path = (
            self.preprocessed_dir
            / item["class"]
            / "images"
            / f"{item['filename']}.png"
        )
        mask_path = self.mask_path(item)

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(image_path)
        if mask is None:
            raise FileNotFoundError(mask_path)

        if mask.shape != image.shape:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        roi = crop_masked_roi(
            image,
            mask,
            self.padding_fraction,
        )

        rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        tensor = self.transform(Image.fromarray(rgb))
        label = CLASS_MAP[item["class"]]

        return tensor, label, item["filename"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability_malignant: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "sensitivity": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probability_malignant)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[float] = []
    filenames: list[str] = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for inputs, labels, names in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(inputs)
                loss = criterion(logits, labels)

            if training:
                if scaler is None:
                    raise RuntimeError("Falta GradScaler durante entrenamiento.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            probability = torch.softmax(logits, dim=1)[:, 1]
            prediction = (probability >= 0.5).long()

            total_loss += float(loss.item()) * inputs.size(0)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(prediction.detach().cpu().numpy().tolist())
            probabilities.extend(
                probability.detach().cpu().numpy().tolist()
            )
            filenames.extend(list(names))

    average_loss = total_loss / len(loader.dataset)

    return (
        average_loss,
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(probabilities),
        filenames,
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_xticks([0, 1], ["Benigna", "Maligna"])
    ax.set_yticks([0, 1], ["Benigna", "Maligna"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(
    y_true: np.ndarray,
    probability: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, probability)
    auc = roc_auc_score(y_true, probability)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("1 − especificidad")
    ax.set_ylabel("Sensibilidad")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_model(dropout: float) -> nn.Module:
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 2),
    )

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask-source",
        choices=["manual", "auto"],
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/manifest.csv"),
    )
    parser.add_argument(
        "--eligible-csv",
        type=Path,
        default=Path("outputs/tables_aligned/features_auto.csv"),
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path(
            "data/processed/preprocessed/robust_bilateral"
        ),
    )
    parser.add_argument(
        "--auto-masks-dir",
        type=Path,
        default=Path("data/processed/auto_masks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/efficientnet"),
    )
    parser.add_argument("--run-name", required=True)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-epochs", type=int, default=15)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Script version: {SCRIPT_VERSION}")
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    eligible_filenames = load_eligible_filenames(
        args.eligible_csv
    )

    train_items = load_manifest_items(
        args.manifest,
        "train",
        eligible_filenames,
    )
    val_items = load_manifest_items(
        args.manifest,
        "val",
        eligible_filenames,
    )
    test_items = load_manifest_items(
        args.manifest,
        "test",
        eligible_filenames,
    )

    expected_counts = (216, 48, 47)
    actual_counts = (
        len(train_items),
        len(val_items),
        len(test_items),
    )

    if actual_counts != expected_counts:
        raise ValueError(
            "El conjunto elegible no coincide con el conjunto alineado "
            f"esperado. Obtenido={actual_counts}, esperado={expected_counts}."
        )

    if args.smoke_test:
        train_items = stratified_sample(
            train_items,
            n=40,
            seed=args.seed,
        )
        val_items = stratified_sample(
            val_items,
            n=16,
            seed=args.seed,
        )
        args.head_epochs = 1
        args.fine_epochs = 1
        args.patience = 2

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset_kwargs = {
        "preprocessed_dir": args.preprocessed_dir,
        "auto_masks_dir": args.auto_masks_dir,
        "mask_source": args.mask_source,
        "padding_fraction": args.padding,
    }

    train_dataset = BusiROIDataset(
        train_items,
        transform=train_transform,
        **dataset_kwargs,
    )
    val_dataset = BusiROIDataset(
        val_items,
        transform=evaluation_transform,
        **dataset_kwargs,
    )
    test_dataset = BusiROIDataset(
        test_items,
        transform=evaluation_transform,
        **dataset_kwargs,
    )

    print(
        f"train={len(train_dataset)} | "
        f"val={len(val_dataset)} | "
        f"test={len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    train_labels = [
        CLASS_MAP[item["class"]]
        for item in train_items
    ]
    counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(
        len(train_labels) / (2 * np.maximum(counts, 1)),
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(args.dropout)
    model.to(device)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    checkpoint_path = run_dir / "best.pt"
    history: list[dict[str, object]] = []

    best_balanced_accuracy = -np.inf
    epochs_without_improvement = 0
    global_epoch = 0

    stages = [
        ("head", args.head_epochs, args.head_lr),
        ("fine", args.fine_epochs, args.fine_lr),
    ]

    stop_training = False

    for stage_name, epochs, learning_rate in stages:
        if stage_name == "fine":
            for block in list(model.features.children())[-2:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            lr=learning_rate,
            weight_decay=1e-4,
        )

        for _ in range(epochs):
            global_epoch += 1

            (
                train_loss,
                _,
                _,
                _,
                _,
            ) = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
            )

            (
                val_loss,
                y_true,
                y_pred,
                probability,
                _,
            ) = run_epoch(
                model,
                val_loader,
                criterion,
                device,
            )

            metrics = compute_metrics(
                y_true,
                y_pred,
                probability,
            )
            score = metrics["balanced_accuracy"]

            history.append(
                {
                    "epoch": global_epoch,
                    "stage": stage_name,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **metrics,
                    "learning_rate": learning_rate,
                }
            )

            print(
                f"epoch={global_epoch:02d} | "
                f"stage={stage_name:4s} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_bal_acc={score:.4f} | "
                f"val_auc={metrics['roc_auc']:.4f}"
            )

            if score > best_balanced_accuracy:
                best_balanced_accuracy = score
                epochs_without_improvement = 0

                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "args": vars(args),
                        "best_val_balanced_accuracy": score,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping: {args.patience} épocas "
                    "sin mejora."
                )
                stop_training = True
                break

        if stop_training:
            break

    pd.DataFrame(history).to_csv(
        run_dir / "train_log.csv",
        index=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])

    (
        _,
        val_true,
        val_pred,
        val_probability,
        val_names,
    ) = run_epoch(
        model,
        val_loader,
        criterion,
        device,
    )

    val_metrics = compute_metrics(
        val_true,
        val_pred,
        val_probability,
    )

    summary_rows: list[dict[str, object]] = [
        {
            "model": "EfficientNet-B0",
            "mask_source": args.mask_source,
            "run_name": args.run_name,
            "split": "val",
            **val_metrics,
            "dropout": args.dropout,
            "padding": args.padding,
            "head_lr": args.head_lr,
            "fine_lr": args.fine_lr,
            "seed": args.seed,
        }
    ]

    val_predictions = pd.DataFrame(
        {
            "filename": val_names,
            "y_true": val_true,
            "y_pred": val_pred,
            "p_malignant": val_probability,
        }
    )
    val_predictions.to_csv(
        run_dir / "val_predictions.csv",
        index=False,
    )

    if args.evaluate_test and not args.smoke_test:
        (
            _,
            test_true,
            test_pred,
            test_probability,
            test_names,
        ) = run_epoch(
            model,
            test_loader,
            criterion,
            device,
        )

        test_metrics = compute_metrics(
            test_true,
            test_pred,
            test_probability,
        )

        summary_rows.append(
            {
                "model": "EfficientNet-B0",
                "mask_source": args.mask_source,
                "run_name": args.run_name,
                "split": "test",
                **test_metrics,
                "dropout": args.dropout,
                "padding": args.padding,
                "head_lr": args.head_lr,
                "fine_lr": args.fine_lr,
                "seed": args.seed,
            }
        )

        test_predictions = pd.DataFrame(
            {
                "filename": test_names,
                "y_true": test_true,
                "y_pred": test_pred,
                "p_malignant": test_probability,
            }
        )
        test_predictions.to_csv(
            run_dir / "test_predictions.csv",
            index=False,
        )

        save_confusion_matrix(
            test_true,
            test_pred,
            f"EfficientNet-B0 ({args.mask_source}) — test",
            run_dir / "confusion_matrix_test.png",
        )
        save_roc_curve(
            test_true,
            test_probability,
            f"EfficientNet-B0 ({args.mask_source}) — ROC test",
            run_dir / "roc_test.png",
        )

        print(
            f"TEST balanced_accuracy="
            f"{test_metrics['balanced_accuracy']:.4f} | "
            f"AUC={test_metrics['roc_auc']:.4f}"
        )

    pd.DataFrame(summary_rows).to_csv(
        run_dir / "summary.csv",
        index=False,
    )

    (
        run_dir / "config.json"
    ).write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"Guardado en: {run_dir}")


if __name__ == "__main__":
    main()
