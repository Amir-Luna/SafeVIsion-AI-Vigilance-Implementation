"""Script de entrenamiento de YOLO con Transfer Learning para SafeVision AI.

Características:
- Carga pesos preentrenados oficiales de Ultralytics (checkpoint configurable).
- Entrena exclusivamente con el dataset propio indicado en configs/data.yaml.
- Detección automática de GPU CUDA o CPU.
- Early stopping (patience), validación, generación de gráficas y métricas.
- Copia el mejor modelo entrenado a models/yolo/best.pt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.utils import (
    SUPPORTED_IMAGE_EXTENSIONS,
    load_config,
    resolve_project_path,
    select_device,
    setup_logging,
)


def validate_yolo_dataset(data_yaml_path: Path) -> Dict[str, Any]:
    """Valida que data.yaml y las carpetas de entrenamiento y validación contengan imágenes."""
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"No se encontró data.yaml en: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    if not isinstance(data_cfg, dict):
        raise ValueError(f"data.yaml inválido: {data_yaml_path}")

    base_path = Path(data_cfg.get("path", ""))
    if not base_path.is_absolute():
        base_path = (data_yaml_path.parent / base_path).resolve()

    summary: Dict[str, Any] = {"base_path": str(base_path), "splits": {}}
    for split_key in ("train", "val", "test"):
        rel = data_cfg.get(split_key)
        if not rel:
            continue
        split_dir = (base_path / rel).resolve()
        if not split_dir.exists():
            if split_key in ("train", "val"):
                raise FileNotFoundError(
                    f"Dataset YOLO incompleto: no existe la carpeta '{split_key}' en {split_dir}"
                )
            summary["splits"][split_key] = 0
            continue

        images = [
            p
            for p in split_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        summary["splits"][split_key] = len(images)
        if split_key in ("train", "val") and len(images) == 0:
            raise ValueError(
                f"Dataset YOLO incompleto: la carpeta '{split_key}' ({split_dir}) no contiene imágenes. "
                "Coloque sus fotografías anotadas antes de entrenar."
            )

    return summary


def compute_f1(precision: float, recall: float) -> float:
    """Calcula F1-score evitando división por cero."""
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom > 0.0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de YOLO (Transfer Learning) para SafeVision AI."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Ruta a config.yaml")
    parser.add_argument("--data", type=str, default=None, help="Ruta a data.yaml (sobreescribe config)")
    parser.add_argument("--weights", type=str, default=None, help="Checkpoint preentrenado (ej: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=None, help="Tamaño de imagen")
    parser.add_argument("--batch", type=int, default=None, help="Tamaño de batch")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=None, help="Número de workers del DataLoader")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo ('cuda', '0' o 'cpu')")
    parser.add_argument("--run-name", type=str, default="yolo_safevision", help="Nombre de la corrida")
    return parser.parse_args()


def main() -> None:
    logger = setup_logging()
    args = parse_args()
    cfg = load_config(args.config)

    paths_cfg = cfg["paths"]
    yolo_cfg = cfg["yolo"]

    data_yaml = resolve_project_path(args.data or paths_cfg["data_yaml"], PROJECT_ROOT)
    pretrained_weights = args.weights or yolo_cfg["pretrained_checkpoint"]
    epochs = int(args.epochs if args.epochs is not None else yolo_cfg["epochs"])
    imgsz = int(args.imgsz if args.imgsz is not None else yolo_cfg["imgsz"])
    batch = int(args.batch if args.batch is not None else yolo_cfg["batch"])
    patience = int(args.patience if args.patience is not None else yolo_cfg["patience"])
    workers = int(args.workers if args.workers is not None else yolo_cfg["workers"])
    device = args.device or select_device(prefer_cuda=True)

    runs_dir = resolve_project_path(paths_cfg.get("runs_dir", "outputs/runs"), PROJECT_ROOT)
    best_dest = resolve_project_path(paths_cfg["yolo_best_weights"], PROJECT_ROOT)
    best_dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Validando dataset YOLO en %s...", data_yaml)
    ds_info = validate_yolo_dataset(data_yaml)
    logger.info("Imágenes encontradas por conjunto: %s", ds_info["splits"])

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics no está instalado. Instálelo con: pip install ultralytics"
        ) from exc

    logger.info(
        "Iniciando entrenamiento YOLO | pesos=%s | epochs=%d | imgsz=%d | batch=%d | device=%s",
        pretrained_weights,
        epochs,
        imgsz,
        batch,
        device,
    )

    model = YOLO(pretrained_weights)
    try:
        train_results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            workers=workers,
            device=device,
            project=str(runs_dir),
            name=args.run_name,
            exist_ok=True,
            pretrained=True,
            plots=True,
            seed=int(cfg.get("project", {}).get("seed", 42)),
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error(
                "Error de falta de memoria GPU (CUDA OOM). "
                "Reduzca --batch (ej. --batch 8 o --batch 4) o --imgsz, o ejecute con --device cpu."
            )
        raise

    save_dir = Path(getattr(train_results, "save_dir", runs_dir / args.run_name))
    trained_best = save_dir / "weights" / "best.pt"
    if not trained_best.exists():
        raise FileNotFoundError(
            f"El entrenamiento finalizó pero no se generó best.pt en {trained_best}."
        )

    shutil.copy2(trained_best, best_dest)
    logger.info("Mejor modelo YOLO guardado en: %s", best_dest)

    # Validación explícita del mejor modelo para registrar métricas consolidadas
    best_model = YOLO(str(best_dest))
    val_metrics = best_model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=True,
        project=str(runs_dir),
        name=f"{args.run_name}_val",
        exist_ok=True,
    )

    precision = float(val_metrics.box.mp)
    recall = float(val_metrics.box.mr)
    map50 = float(val_metrics.box.map50)
    map50_95 = float(val_metrics.box.map)
    f1_score = compute_f1(precision, recall)

    metrics_payload: Dict[str, Any] = {
        "pretrained_checkpoint": pretrained_weights,
        "best_weights_path": str(best_dest),
        "device": device,
        "epochs_configured": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "metrics_val": {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1_score, 6),
            "mAP50": round(map50, 6),
            "mAP50_95": round(map50_95, 6),
        },
        "dataset_splits": ds_info["splits"],
    }

    metrics_file = save_dir / "metrics_summary.json"
    metrics_file.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Métricas de validación YOLO:")
    logger.info("  Precision : %.4f", precision)
    logger.info("  Recall    : %.4f", recall)
    logger.info("  F1-score  : %.4f", f1_score)
    logger.info("  mAP@50    : %.4f", map50)
    logger.info("  mAP@50-95 : %.4f", map50_95)
    logger.info("Resumen guardado en: %s", metrics_file)


if __name__ == "__main__":
    main()
