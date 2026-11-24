"""Script de evaluación independiente y conjunta de YOLO y CNN para SafeVision AI.

Modos soportados (--mode yolo | cnn | joint | all):
- yolo: Evalúa models/yolo/best.pt sobre el conjunto de prueba de configs/data.yaml.
- cnn: Evalúa models/cnn/best_cnn.pth sobre datasets/cnn/test/ generando accuracy,
  precision/recall/F1 por clase, classification report y matriz de confusión.
- joint: Evalúa el pipeline completo (YOLO + CNN + DecisionEngine) sobre un directorio
  de imágenes y metadata.csv, guardando el resumen de estados CUMPLE / NO_CUMPLE / NO_VERIFICABLE.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.generar_dataset_cnn import load_metadata_csv
from scripts.train_cnn import collect_samples, plot_confusion_matrix
from src.classifier import PPEClassifier, TORCH_AVAILABLE
from src.decision_engine import DecisionEngine
from src.detector import YOLODetector
from src.utils import (
    CNN_DEFAULT_CLASSES,
    SUPPORTED_IMAGE_EXTENSIONS,
    load_config,
    resolve_project_path,
    safe_imread,
    select_device,
    setup_logging,
)


def evaluate_yolo(cfg: Dict[str, Any], out_dir: Path, device: str) -> Dict[str, Any]:
    """Evalúa YOLO sobre el conjunto de prueba definido en data.yaml."""
    logger = setup_logging()
    weights = resolve_project_path(cfg["paths"]["yolo_best_weights"], PROJECT_ROOT)
    data_yaml = resolve_project_path(cfg["paths"]["data_yaml"], PROJECT_ROOT)

    if not weights.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo YOLO entrenado en {weights}. Ejecute scripts/train_yolo.py."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics no está instalado.") from exc

    logger.info("Evaluando YOLO (%s) sobre el conjunto 'test'...", weights)
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=int(cfg["yolo"].get("imgsz", 640)),
        batch=int(cfg["yolo"].get("batch", 16)),
        device=device,
        plots=True,
        project=str(out_dir),
        name="yolo_test",
        exist_ok=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    result: Dict[str, Any] = {
        "weights": str(weights),
        "split": "test",
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "mAP50": round(float(metrics.box.map50), 6),
        "mAP50_95": round(float(metrics.box.map), 6),
    }

    (out_dir / "yolo_test_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Métricas YOLO (test): %s", result)
    return result


def evaluate_cnn(cfg: Dict[str, Any], out_dir: Path, device: str) -> Dict[str, Any]:
    """Evalúa la CNN propia sobre datasets/cnn/test/ generando reportes y matriz de confusión."""
    logger = setup_logging()
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch no está instalado.")

    import torch
    from PIL import Image
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    weights = resolve_project_path(cfg["paths"]["cnn_best_weights"], PROJECT_ROOT)
    test_dir = resolve_project_path(cfg["paths"]["cnn_dataset_dir"], PROJECT_ROOT) / "test"

    classifier = PPEClassifier(weights_path=weights, device=device)
    class_names = classifier.class_names or list(CNN_DEFAULT_CLASSES)
    samples = collect_samples(test_dir, class_names)

    eval_tf = transforms.Compose(
        [
            transforms.Resize((classifier.img_size, classifier.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=classifier.mean.reshape(-1).tolist(),
                std=classifier.std.reshape(-1).tolist(),
            ),
        ]
    )

    class EvalDataset(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, idx: int):
            p, lbl = samples[idx]
            with Image.open(p) as im:
                return eval_tf(im.convert("RGB")), lbl

    loader = DataLoader(EvalDataset(), batch_size=int(cfg["cnn"].get("batch_size", 32)), shuffle=False)

    y_true: List[int] = []
    y_pred: List[int] = []
    classifier.model.eval()
    with torch.inference_mode():
        for imgs, labels in loader:
            logits = classifier.model(imgs.to(classifier.device))
            preds = torch.argmax(logits, dim=1)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(labels.tolist())

    labels_range = list(range(len(class_names)))
    acc = float(accuracy_score(y_true, y_pred))
    report_txt = classification_report(
        y_true, y_pred, labels=labels_range, target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)

    cnn_out = out_dir / "cnn_test"
    cnn_out.mkdir(parents=True, exist_ok=True)
    (cnn_out / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    (cnn_out / "classification_report.json").write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savetxt(str(cnn_out / "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, class_names, cnn_out / "confusion_matrix.png")

    logger.info("Accuracy CNN (test): %.4f", acc)
    logger.info("Classification Report CNN (test):\n%s", report_txt)
    return {"accuracy": round(acc, 6), "num_samples": len(y_true), "report": report_dict}


def evaluate_joint_pipeline(cfg: Dict[str, Any], out_dir: Path, device: str) -> Dict[str, Any]:
    """Evalúa YOLO + CNN + DecisionEngine de forma conjunta sobre imágenes de prueba."""
    logger = setup_logging()
    yolo_test_images = resolve_project_path(cfg["paths"]["yolo_dataset_dir"], PROJECT_ROOT) / "images" / "test"
    metadata_path = resolve_project_path(cfg["paths"]["metadata_csv"], PROJECT_ROOT)

    if not yolo_test_images.exists():
        raise FileNotFoundError(f"No existe la carpeta de prueba conjunta: {yolo_test_images}")

    metadata = load_metadata_csv(metadata_path)
    detector = YOLODetector(
        weights_path=cfg["paths"]["yolo_best_weights"],
        conf_threshold=float(cfg["yolo"].get("conf_threshold", 0.35)),
        iou_threshold=float(cfg["yolo"].get("iou_threshold", 0.45)),
        imgsz=int(cfg["yolo"].get("imgsz", 640)),
        device=device,
    )
    classifier = PPEClassifier(weights_path=cfg["paths"]["cnn_best_weights"], device=device)
    engine = DecisionEngine(cfg)

    status_counts: Counter[str] = Counter()
    rows_out: List[Dict[str, str]] = []

    image_files = sorted(
        p for p in yolo_test_images.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No hay imágenes de prueba en {yolo_test_images}.")

    for img_path in image_files:
        meta = metadata.get(img_path.name)
        if meta is None:
            logger.warning("Imagen omitida en evaluación conjunta (sin actividad en metadata.csv): %s", img_path.name)
            continue

        frame = safe_imread(img_path)
        if frame is None:
            logger.warning("Imagen ilegible omitida: %s", img_path.name)
            continue

        detections = detector.detect(frame)
        eval_res = engine.evaluate(frame, detections, classifier, activity=meta["actividad"])
        status_counts[eval_res.overall_status] += 1

        rows_out.append(
            {
                "imagen": img_path.name,
                "actividad": meta["actividad"],
                "estado_global": eval_res.overall_status,
                "detalle": " | ".join(eval_res.summary_messages),
            }
        )

    joint_out = out_dir / "joint_test"
    joint_out.mkdir(parents=True, exist_ok=True)
    csv_path = joint_out / "resultados_evaluacion_conjunta.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["imagen", "actividad", "estado_global", "detalle"])
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "imagenes_evaluadas": len(rows_out),
        "distribucion_estados": dict(status_counts),
        "csv_resultados": str(csv_path),
    }
    (joint_out / "resumen_conjunto.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Evaluación conjunta completada: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación de modelos de SafeVision AI.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("yolo", "cnn", "joint", "all"),
        default="all",
        help="Modo de evaluación.",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Ruta a config.yaml")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo ('cuda' o 'cpu')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device or select_device(prefer_cuda=True)
    out_dir = resolve_project_path(cfg["paths"].get("evaluation_output_dir", "outputs/evaluation"), PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("yolo", "all"):
        evaluate_yolo(cfg, out_dir, device)
    if args.mode in ("cnn", "all"):
        evaluate_cnn(cfg, out_dir, device)
    if args.mode in ("joint", "all"):
        evaluate_joint_pipeline(cfg, out_dir, device)


if __name__ == "__main__":
    main()
