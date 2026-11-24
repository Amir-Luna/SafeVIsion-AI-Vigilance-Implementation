"""Script de entrenamiento desde cero de la CNN propia (SafeVisionCNN) en PyTorch.

Requisitos cumplidos:
- Sin pesos preentrenados ni transfer learning (inicialización aleatoria).
- Entrenamiento sobre datasets/cnn/train y validación sobre datasets/cnn/val.
- Normalización, data augmentation razonable, CrossEntropyLoss (con pesos de clase opcionales),
  AdamW, ReduceLROnPlateau y early stopping.
- Guardado del mejor modelo en models/cnn/best_cnn.pth junto con mapa de clases y configuración.
- Generación de curvas de entrenamiento, classification report y matriz de confusión.
"""

from __future__ import annotations
from PIL import Image

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.classifier import SafeVisionCNN, TORCH_AVAILABLE
from src.utils import (
    CNN_DEFAULT_CLASSES,
    SUPPORTED_IMAGE_EXTENSIONS,
    load_config,
    resolve_project_path,
    select_device,
    setup_logging,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def collect_samples(split_dir: Path, class_names: List[str]) -> List[Tuple[Path, int]]:
    """Recopila pares (ruta_imagen, idx_clase) respetando el orden fijo de las 8 clases."""
    if not split_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de conjunto CNN: {split_dir}")

    samples: List[Tuple[Path, int]] = []
    for idx, cls_name in enumerate(class_names):
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                samples.append((img_path, idx))

    if not samples:
        raise ValueError(
            f"No se encontraron imágenes en {split_dir}. "
            "Ejecute primero scripts/generar_dataset_cnn.py para construir el dataset."
        )
    return samples


def plot_training_history(history: List[Dict[str, Any]], output_path: Path) -> None:
    """Genera gráfica de pérdida y accuracy/F1 por época."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_title("Pérdida por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label="Train Accuracy")
    axes[1].plot(epochs, val_acc, label="Val Accuracy")
    axes[1].plot(epochs, val_f1, label="Val Macro F1")
    axes[1].set_title("Métricas por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Valor")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    """Guarda la matriz de confusión como imagen PNG legible."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Etiqueta real",
        xlabel="Predicción",
        title="Matriz de Confusión - SafeVisionCNN",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento desde cero de la CNN propia de SafeVision AI."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Ruta a config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=None, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate inicial")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo ('cuda' o 'cpu')")
    return parser.parse_args()

class FixedClassImageDataset(Dataset):
        """Dataset PyTorch que garantiza mapeo determinista a las 8 clases oficiales."""

        def __init__(self, samples: List[Tuple[Path, int]], transform: Any) -> None:
            self.samples = samples
            self.transform = transform

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            img_path, label = self.samples[idx]
            try:
                with Image.open(img_path) as im:
                    rgb = im.convert("RGB")
            except Exception as exc:
                raise RuntimeError(f"Imagen dañada en dataset CNN: {img_path}") from exc
            return self.transform(rgb), label


def main() -> None:
    logger = setup_logging()
    args = parse_args()
    cfg = load_config(args.config)

    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch y TorchVision son obligatorios para entrenar la CNN. "
            "Instálelos siguiendo las instrucciones del README.md."
        )

    import torch
    import torch.nn as nn
    from PIL import Image
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms


    seed = int(cfg.get("project", {}).get("seed", 42))
    set_seed(seed)

    paths_cfg = cfg["paths"]
    cnn_cfg = cfg["cnn"]

    cnn_dataset_dir = resolve_project_path(paths_cfg["cnn_dataset_dir"], PROJECT_ROOT)
    best_model_path = resolve_project_path(paths_cfg["cnn_best_weights"], PROJECT_ROOT)
    runs_dir = resolve_project_path(paths_cfg.get("runs_dir", "outputs/runs"), PROJECT_ROOT) / "cnn_training"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    img_size = int(cnn_cfg.get("img_size", 224))
    epochs = int(args.epochs if args.epochs is not None else cnn_cfg.get("epochs", 60))
    batch_size = int(args.batch_size if args.batch_size is not None else cnn_cfg.get("batch_size", 32))
    lr = float(args.lr if args.lr is not None else cnn_cfg.get("learning_rate", 1e-3))
    weight_decay = float(cnn_cfg.get("weight_decay", 1e-4))
    dropout = float(cnn_cfg.get("dropout", 0.4))
    patience = int(args.patience if args.patience is not None else cnn_cfg.get("patience", 12))
    num_workers = int(cnn_cfg.get("num_workers", 2))
    mean = list(cnn_cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
    std = list(cnn_cfg.get("normalize_std", [0.229, 0.224, 0.225]))

    # Respetar el orden explícito de clases (0..7)
    classes_dict = cnn_cfg.get("classes", {})
    if isinstance(classes_dict, dict) and len(classes_dict) == 8:
        class_names = [str(classes_dict[i]) for i in sorted(int(k) for k in classes_dict.keys())]
    else:
        class_names = list(CNN_DEFAULT_CLASSES)

    train_samples = collect_samples(cnn_dataset_dir / "train", class_names)
    val_samples = collect_samples(cnn_dataset_dir / "val", class_names)

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_ds = FixedClassImageDataset(train_samples, train_transform)
    val_ds = FixedClassImageDataset(val_samples, eval_transform)

    device_str = args.device or select_device(prefer_cuda=True)
    device = torch.device(device_str)
    pin_mem = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )

    model = SafeVisionCNN(num_classes=len(class_names), dropout=dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Entrenando SafeVisionCNN DESDE CERO | parámetros=%s | train=%d | val=%d | device=%s",
        f"{total_params:,}",
        len(train_ds),
        len(val_ds),
        device,
    )

    # Pesos de clase inversos a la frecuencia si hay desbalance
    class_weights_tensor = None
    if bool(cnn_cfg.get("use_class_weights", True)):
        counts = np.zeros(len(class_names), dtype=np.float32)
        for _, lbl in train_samples:
            counts[lbl] += 1.0
        weights = np.where(counts > 0, counts.sum() / (len(class_names) * np.maximum(counts, 1.0)), 0.0)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(2, patience // 3)
    )

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_val_preds: List[int] = []
    best_val_targets: List[int] = []
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds: List[int] = []
        train_targets: List[int] = []

        try:
            for images, labels in train_loader:
                images = images.to(device, non_blocking=pin_mem)
                labels = labels.to(device, non_blocking=pin_mem)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * images.size(0)
                preds = torch.argmax(logits, dim=1)
                train_preds.extend(preds.cpu().tolist())
                train_targets.extend(labels.cpu().tolist())
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error(
                    "Falta de memoria GPU (CUDA OOM) durante entrenamiento CNN. "
                    "Reduzca --batch-size o ejecute con --device cpu."
                )
            raise

        train_loss = running_loss / max(1, len(train_ds))
        train_acc = float(accuracy_score(train_targets, train_preds))

        # Evaluación en validación
        model.eval()
        val_running_loss = 0.0
        val_preds: List[int] = []
        val_targets: List[int] = []
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=pin_mem)
                labels = labels.to(device, non_blocking=pin_mem)
                logits = model(images)
                loss = criterion(logits, labels)
                val_running_loss += float(loss.item()) * images.size(0)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

        val_loss = val_running_loss / max(1, len(val_ds))
        val_acc = float(accuracy_score(val_targets, val_preds))
        val_macro_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_macro_f1)

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "val_macro_f1": round(val_macro_f1, 6),
            }
        )

        logger.info(
            "Época %03d/%03d | lr=%.2e | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f macro_f1=%.4f",
            epoch,
            epochs,
            current_lr,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_macro_f1,
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_val_preds = list(val_preds)
            best_val_targets = list(val_targets)
            epochs_without_improve = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "arch_config": model.get_arch_config(),
                "class_names": class_names,
                "class_to_idx": {name: i for i, name in enumerate(class_names)},
                "img_size": img_size,
                "normalize_mean": mean,
                "normalize_std": std,
                "best_val_macro_f1": best_val_f1,
                "best_val_acc": val_acc,
                "best_epoch": epoch,
            }
            torch.save(checkpoint, str(best_model_path))
            logger.info("  -> Nuevo mejor modelo guardado en %s (val_macro_f1=%.4f)", best_model_path, best_val_f1)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                logger.info("Early stopping activado tras %d épocas sin mejora.", patience)
                break

    # Guardar historial CSV y gráficas
    history_csv = runs_dir / "training_history.csv"
    with history_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plot_training_history(history, runs_dir / "training_curves.png")

    # Reporte de clasificación y matriz de confusión sobre validación del mejor modelo
    labels_range = list(range(len(class_names)))
    report_txt = classification_report(
        best_val_targets,
        best_val_preds,
        labels=labels_range,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        best_val_targets,
        best_val_preds,
        labels=labels_range,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(best_val_targets, best_val_preds, labels=labels_range)

    (runs_dir / "val_classification_report.txt").write_text(report_txt, encoding="utf-8")
    (runs_dir / "val_classification_report.json").write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_confusion_matrix(cm, class_names, runs_dir / "val_confusion_matrix.png")

    logger.info("Entrenamiento de SafeVisionCNN finalizado.")
    logger.info("Classification report (validación):\n%s", report_txt)


if __name__ == "__main__":
    main()
