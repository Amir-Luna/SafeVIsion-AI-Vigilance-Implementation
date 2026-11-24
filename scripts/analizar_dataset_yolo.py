"""Análisis de distribución de clases del dataset YOLO de SafeVision AI.

Responde a la pregunta del diagnóstico: ¿el modelo ignora MANO porque hay pocas etiquetas
de MANO (o son muy pequeñas) frente a PERSONA y CABEZA?

Para cada split (train / val / test) reporta por clase:
- instancias (cajas) e imágenes que contienen la clase,
- tamaño mediano de caja (% del área de la imagen) y % de cajas "pequeñas",
- advertencias de desbalance según `dataset_split.imbalance_ratio_warning` de config.yaml.

Opcionalmente (--balance) genera una lista de entrenamiento sobremuestreada que repite las
imágenes que contienen clases minoritarias (p. ej. MANO) y un data_balanced.yaml listo
para `python scripts/train_yolo.py --data configs/data_balanced.yaml`.

Uso:
    python scripts/analizar_dataset_yolo.py
    python scripts/analizar_dataset_yolo.py --balance --target-classes MANO GAFAS_SEGURIDAD
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.utils import SUPPORTED_IMAGE_EXTENSIONS, load_config, resolve_project_path, setup_logging


def images_to_labels_dir(images_dir: Path) -> Path:
    """Convención Ultralytics: .../images/<split> -> .../labels/<split>."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels" / images_dir.name


def read_label_file(label_path: Path) -> List[Tuple[int, float, float]]:
    """Devuelve [(class_id, w, h)] normalizados; ignora líneas inválidas."""
    rows: List[Tuple[int, float, float]] = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = line.strip().split()
        if len(tokens) < 5:
            continue
        try:
            cls_id = int(float(tokens[0]))
            w, h = float(tokens[3]), float(tokens[4])
        except ValueError:
            continue
        rows.append((cls_id, w, h))
    return rows


def analyze_split(
    images_dir: Path,
    names: Dict[int, str],
    small_area_ratio: float,
) -> Dict[str, Any]:
    """Estadísticas por clase de un split."""
    labels_dir = images_to_labels_dir(images_dir)
    images = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    instances: Counter = Counter()
    images_with: Counter = Counter()
    areas: Dict[int, List[float]] = defaultdict(list)
    image_classes: Dict[str, List[int]] = {}
    missing_labels = 0
    empty_labels = 0
    unknown_ids: Counter = Counter()

    for img in images:
        rel = img.relative_to(images_dir).with_suffix(".txt")
        label_path = labels_dir / rel
        if not label_path.exists():
            missing_labels += 1
            continue
        rows = read_label_file(label_path)
        if not rows:
            empty_labels += 1
        present = set()
        for cls_id, w, h in rows:
            if cls_id not in names:
                unknown_ids[cls_id] += 1
                continue
            instances[cls_id] += 1
            areas[cls_id].append(max(0.0, w) * max(0.0, h))
            present.add(cls_id)
        for cls_id in present:
            images_with[cls_id] += 1
        image_classes[str(img)] = sorted(present)

    per_class: Dict[str, Dict[str, Any]] = {}
    for cls_id, cls_name in sorted(names.items()):
        a = sorted(areas.get(cls_id, []))
        median = a[len(a) // 2] if a else 0.0
        small = sum(1 for v in a if v < small_area_ratio)
        per_class[cls_name] = {
            "instancias": int(instances.get(cls_id, 0)),
            "imagenes": int(images_with.get(cls_id, 0)),
            "area_mediana_pct": round(median * 100.0, 3),
            "cajas_pequenas_pct": round(100.0 * small / len(a), 1) if a else 0.0,
        }

    return {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "total_imagenes": len(images),
        "sin_archivo_label": missing_labels,
        "labels_vacios": empty_labels,
        "ids_desconocidos": dict(unknown_ids),
        "por_clase": per_class,
        "_image_classes": image_classes,
    }


def print_table(split: str, stats: Dict[str, Any]) -> None:
    print(f"\n=== {split.upper()} — {stats['total_imagenes']} imágenes ({stats['images_dir']}) ===")
    if stats["sin_archivo_label"] or stats["labels_vacios"]:
        print(
            f"  Imágenes sin .txt: {stats['sin_archivo_label']} | .txt vacíos (fondo): {stats['labels_vacios']}"
        )
    if stats["ids_desconocidos"]:
        print(f"  ¡IDs de clase fuera de data.yaml!: {stats['ids_desconocidos']}")
    print(f"  {'CLASE':<20}{'INSTANCIAS':>11}{'IMÁGENES':>10}{'ÁREA MED.%':>12}{'PEQUEÑAS%':>11}")
    for cls_name, s in stats["por_clase"].items():
        print(
            f"  {cls_name:<20}{s['instancias']:>11}{s['imagenes']:>10}"
            f"{s['area_mediana_pct']:>12.2f}{s['cajas_pequenas_pct']:>11.1f}"
        )


def build_warnings(
    stats: Dict[str, Any],
    imbalance_ratio: float,
    min_samples: int,
    focus_classes: List[str],
) -> List[str]:
    warnings: List[str] = []
    per_class = stats["por_clase"]
    counts = {k: v["instancias"] for k, v in per_class.items()}
    max_cls = max(counts, key=lambda k: counts[k]) if counts else None
    max_count = counts.get(max_cls, 0) if max_cls else 0
    for cls_name, n in counts.items():
        if n == 0:
            warnings.append(f"{cls_name}: 0 instancias — el modelo no puede aprender esta clase.")
            continue
        if n < min_samples:
            warnings.append(f"{cls_name}: solo {n} instancias (< {min_samples}).")
        if max_count and max_count / n >= imbalance_ratio:
            warnings.append(
                f"{cls_name}: desbalance {max_count / n:.1f}x respecto a {max_cls} ({n} vs {max_count})."
            )
    for cls_name in focus_classes:
        s = per_class.get(cls_name)
        if s and s["cajas_pequenas_pct"] >= 30.0:
            warnings.append(
                f"{cls_name}: {s['cajas_pequenas_pct']:.0f}% de cajas pequeñas — considere imgsz 768/960 "
                "o un min_area_ratio menor en inferencia."
            )
    return warnings


def build_balanced_list(
    image_classes: Dict[str, List[int]],
    names: Dict[int, str],
    target_classes: List[str],
    max_factor: int,
) -> Tuple[List[str], Dict[str, int]]:
    """Repite imágenes que contienen clases objetivo hasta acercarlas a la clase mayoritaria."""
    name_to_id = {v.upper(): k for k, v in names.items()}
    counts: Counter = Counter()
    for cls_list in image_classes.values():
        for c in cls_list:
            counts[c] += 1
    if not counts:
        return list(image_classes.keys()), {}
    max_imgs = max(counts.values())
    factors: Dict[int, int] = {}
    for cls_name in target_classes:
        cid = name_to_id.get(cls_name.upper())
        if cid is None or counts.get(cid, 0) == 0:
            continue
        factors[cid] = max(1, min(max_factor, int(math.ceil(max_imgs / counts[cid]))))

    lines: List[str] = []
    for img_path, cls_list in image_classes.items():
        repeat = max([factors.get(c, 1) for c in cls_list] or [1])
        lines.extend([img_path] * repeat)
    return lines, {names[c]: f for c, f in factors.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distribución de clases del dataset YOLO de SafeVision AI.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default=None, help="Ruta a data.yaml (por defecto la de config.yaml)")
    p.add_argument(
        "--small-area",
        type=float,
        default=0.002,
        help="Área relativa bajo la cual una caja se considera pequeña (por defecto 0.002).",
    )
    p.add_argument("--balance", action="store_true", help="Generar lista de train sobremuestreada.")
    p.add_argument(
        "--target-classes",
        nargs="+",
        default=["MANO"],
        help="Clases a sobremuestrear con --balance (por defecto MANO).",
    )
    p.add_argument("--max-factor", type=int, default=4, help="Repetición máxima por imagen (por defecto 4).")
    return p.parse_args()


def main() -> None:
    logger = setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    data_yaml = resolve_project_path(args.data or cfg["paths"]["data_yaml"], PROJECT_ROOT)
    if not data_yaml.exists():
        logger.error("No se encontró data.yaml en %s", data_yaml)
        sys.exit(1)

    data_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw_names = data_cfg.get("names", {})
    names: Dict[int, str] = (
        {i: str(n) for i, n in enumerate(raw_names)}
        if isinstance(raw_names, list)
        else {int(k): str(v) for k, v in raw_names.items()}
    )
    base = Path(data_cfg.get("path", ""))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    # Coherencia de nombres entre data.yaml y config.yaml
    cfg_names = {str(v) for v in cfg.get("yolo", {}).get("classes", {}).values()}
    if cfg_names and cfg_names != set(names.values()):
        logger.warning(
            "Los nombres de clase de data.yaml y config.yaml no coinciden: solo_data=%s solo_config=%s",
            sorted(set(names.values()) - cfg_names),
            sorted(cfg_names - set(names.values())),
        )

    split_cfg = cfg.get("dataset_split", {})
    imbalance_ratio = float(split_cfg.get("imbalance_ratio_warning", 3.0))
    min_samples = int(split_cfg.get("min_samples_warning", 20))

    report: Dict[str, Any] = {"data_yaml": str(data_yaml), "names": names, "splits": {}}
    train_image_classes: Optional[Dict[str, List[int]]] = None

    for split in ("train", "val", "test"):
        rel = data_cfg.get(split)
        if not rel or not isinstance(rel, str):
            continue
        images_dir = (base / rel).resolve()
        if not images_dir.is_dir():
            logger.warning("Split '%s' no encontrado en %s", split, images_dir)
            continue
        stats = analyze_split(images_dir, names, args.small_area)
        print_table(split, stats)
        warns = build_warnings(stats, imbalance_ratio, min_samples, args.target_classes)
        for w in warns:
            print(f"  ⚠ {w}")
        if split == "train":
            train_image_classes = stats["_image_classes"]
        stats.pop("_image_classes", None)
        stats["advertencias"] = warns
        report["splits"][split] = stats

    out_dir = resolve_project_path(cfg["paths"].get("evaluation_output_dir", "outputs/evaluation"), PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.balance:
        if not train_image_classes:
            logger.error("No hay split de entrenamiento para balancear.")
        else:
            lines, factors = build_balanced_list(
                train_image_classes, names, args.target_classes, args.max_factor
            )
            list_path = base / "train_balanced.txt"
            list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            balanced_yaml = data_yaml.parent / "data_balanced.yaml"
            balanced_cfg = dict(data_cfg)
            balanced_cfg["path"] = str(base)
            balanced_cfg["train"] = str(list_path)
            balanced_yaml.write_text(
                yaml.safe_dump(balanced_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            report["balance"] = {
                "factores": factors,
                "imagenes_train_original": len(train_image_classes),
                "imagenes_train_balanceado": len(lines),
                "lista": str(list_path),
                "data_yaml": str(balanced_yaml),
            }
            print(
                f"\nLista balanceada: {list_path} ({len(train_image_classes)} -> {len(lines)} entradas, "
                f"factores {factors})"
            )
            print(f"Entrene con: python scripts/train_yolo.py --data {balanced_yaml}")

    out_file = out_dir / "dataset_yolo_stats.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte guardado en: {out_file}")


if __name__ == "__main__":
    main()
