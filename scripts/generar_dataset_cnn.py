"""Script generador del dataset de la CNN propia para SafeVision AI.

Flujo modular en 3 modos (--mode extract | review | build | all):
1. extract: Lee fotografías originales, anotaciones YOLO y metadata.csv; valida
   anotaciones; extrae recortes de CABEZA y MANO con márgenes configurables y
   letterbox sin deformación; guarda en datasets/cnn/revision_pendiente/ y crea
   etiquetas_revision.csv (sin auto-etiquetar correcto/incorrecto).
2. review: Interfaz interactiva con OpenCV para revisión manual mediante teclado:
   - Tecla 1: uso correcto
   - Tecla 2: uso incorrecto
   - Tecla 3: no verificable / descartar
   - Tecla N: siguiente imagen sin clasificar
   - Tecla Q: guardar progreso y salir
   Permite reanudar sesiones sin perder el progreso.
3. build: Genera la estructura definitiva datasets/cnn/{train,val,test}/<8_clases>/
   con división 70/15/15 estrictamente por grupo_sesion / fotografía original
   (evitando fuga de datos) y emite un informe detallado con advertencias de
   clases desbalanceadas o con muestras insuficientes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.utils import (
    CNN_DEFAULT_CLASSES,
    SUPPORTED_IMAGE_EXTENSIONS,
    VALID_ACTIVITIES,
    VALID_REGIONS,
    extract_region_crop,
    get_valid_cnn_classes_for,
    load_config,
    load_data_yaml_classes,
    parse_yolo_label_file,
    resolve_project_path,
    safe_imread,
    safe_imwrite,
    setup_logging,
    yolo_to_pixel_bbox,
)

REVIEW_CSV_COLUMNS: List[str] = [
    "crop_filename",
    "original_image",
    "actividad",
    "grupo_sesion",
    "fuente",
    "division_preasignada",
    "region",
    "crop_index",
    "x1",
    "y1",
    "x2",
    "y2",
    "exp_x1",
    "exp_y1",
    "exp_x2",
    "exp_y2",
    "estado_revision",  # PENDIENTE | CORRECTO | INCORRECTO | DESCARTADO
    "clase_cnn",        # Una de las 8 clases o vacío/DESCARTADO
]


def load_metadata_csv(metadata_path: Path) -> Dict[str, Dict[str, str]]:
    """Carga metadata.csv indexado por nombre de archivo de fotografía original."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No se encontró metadata.csv en: {metadata_path}. "
            "Cree el archivo con columnas: archivo,actividad,grupo_sesion,fuente,division"
        )

    records: Dict[str, Dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
        filtered_lines = [line for line in f if line.strip() and not line.lstrip().startswith("#")]

    if not filtered_lines:
        raise ValueError(f"metadata.csv está vacío o solo contiene comentarios: {metadata_path}")

    reader = csv.DictReader(filtered_lines)
    required_cols = {"archivo", "actividad", "grupo_sesion", "fuente"}
    if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
        raise ValueError(
            f"metadata.csv debe incluir al menos las columnas {sorted(required_cols)}. "
            f"Encontradas: {reader.fieldnames}"
        )

    for row_num, row in enumerate(reader, start=2):
        filename = (row.get("archivo") or "").strip()
        activity = (row.get("actividad") or "").strip().lower()
        group_id = (row.get("grupo_sesion") or "").strip()
        source = (row.get("fuente") or "").strip().lower()
        split = (row.get("division") or "").strip().lower()

        if not filename:
            continue
        if activity not in VALID_ACTIVITIES:
            raise ValueError(
                f"metadata.csv línea {row_num}: actividad inválida '{activity}' para '{filename}'. "
                f"Use {VALID_ACTIVITIES}."
            )
        if split and split not in ("train", "val", "test"):
            raise ValueError(
                f"metadata.csv línea {row_num}: división inválida '{split}' para '{filename}'. "
                "Use 'train', 'val', 'test' o déjela vacía."
            )

        records[filename] = {
            "archivo": filename,
            "actividad": activity,
            "grupo_sesion": group_id or Path(filename).stem,
            "fuente": source or "desconocida",
            "division": split,
        }

    return records


def map_manual_decision_to_class(region: str, activity: str, decision_key: str) -> Tuple[str, str]:
    """Asigna estado de revisión y categoría CNN a partir de tecla ('1', '2', '3')."""
    correct_cls, incorrect_cls = get_valid_cnn_classes_for(region, activity)
    if decision_key == "1":
        return "CORRECTO", correct_cls
    if decision_key == "2":
        return "INCORRECTO", incorrect_cls
    if decision_key == "3":
        return "DESCARTADO", "DESCARTADO"
    raise ValueError(f"Tecla de decisión no válida: {decision_key}")


def assign_group_splits(
    rows: Sequence[Dict[str, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """Asigna cada grupo_sesion a train/val/test evitando filtración de datos.

    Reglas:
    - Todos los recortes de una misma fotografía original o grupo_sesion van al mismo conjunto.
    - Respeta 'division_preasignada' si fue definida en metadata.csv.
    - Estratifica aproximadamente por actividad para mantener equilibrio entre corte y soldadura.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Las proporciones train + val + test deben sumar 1.0.")

    group_to_split: Dict[str, str] = {}
    groups_by_activity: Dict[str, List[str]] = defaultdict(list)
    seen_groups = set()

    for row in rows:
        group_id = row["grupo_sesion"].strip() or Path(row["original_image"]).stem
        pre_split = (row.get("division_preasignada") or "").strip().lower()
        activity = row["actividad"].strip().lower()

        if pre_split in ("train", "val", "test"):
            if group_id in group_to_split and group_to_split[group_id] != pre_split:
                raise ValueError(
                    f"Conflicto de división preasignada para el grupo '{group_id}': "
                    f"{group_to_split[group_id]} vs {pre_split}."
                )
            group_to_split[group_id] = pre_split
            seen_groups.add(group_id)
        elif group_id not in seen_groups:
            seen_groups.add(group_id)
            groups_by_activity[activity].append(group_id)

    rng = random.Random(seed)
    for activity, grp_list in sorted(groups_by_activity.items()):
        shuffled = list(grp_list)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n == 0:
            continue
        if n == 1:
            group_to_split[shuffled[0]] = "train"
            continue
        if n == 2:
            group_to_split[shuffled[0]] = "train"
            group_to_split[shuffled[1]] = "val"
            continue

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        for idx, gid in enumerate(shuffled):
            if idx < n_train:
                group_to_split[gid] = "train"
            elif idx < n_train + n_val:
                group_to_split[gid] = "val"
            else:
                group_to_split[gid] = "test"

    return group_to_split


def load_review_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Carga el registro CSV de revisión si existe."""
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_review_csv(csv_path: Path, rows: Sequence[Dict[str, str]]) -> None:
    """Guarda el progreso de revisión en CSV de forma atómica."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in REVIEW_CSV_COLUMNS})
    tmp_path.replace(csv_path)


def extract_crops_pipeline(cfg: Dict[str, Any], target_size_override: Optional[int] = None) -> Dict[str, Any]:
    """Ejecuta la lectura de fotos + YOLO labels y extrae recortes de CABEZA y MANO."""
    logger = setup_logging()
    paths_cfg = cfg["paths"]
    crops_cfg = cfg["crops"]

    images_dir = resolve_project_path(paths_cfg["original_images_dir"], PROJECT_ROOT)
    labels_dir = resolve_project_path(paths_cfg["original_labels_dir"], PROJECT_ROOT)
    metadata_path = resolve_project_path(paths_cfg["metadata_csv"], PROJECT_ROOT)
    data_yaml_path = resolve_project_path(paths_cfg["data_yaml"], PROJECT_ROOT)
    pending_dir = resolve_project_path(paths_cfg["cnn_pending_review_dir"], PROJECT_ROOT)
    pending_dir.mkdir(parents=True, exist_ok=True)

    review_csv_path = pending_dir / "etiquetas_revision.csv"
    existing_rows = {r["crop_filename"]: r for r in load_review_csv(review_csv_path)}

    class_map = load_data_yaml_classes(data_yaml_path)
    metadata = load_metadata_csv(metadata_path)

    head_margin = float(crops_cfg.get("head_margin_ratio", 0.20))
    hand_margin = float(crops_cfg.get("hand_margin_ratio", 0.25))
    target_size = int(target_size_override or crops_cfg.get("target_size", 224))
    pad_color = tuple(int(c) for c in crops_cfg.get("pad_color", [114, 114, 114]))

    if not images_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de fotografías originales: {images_dir}")

    processed_images = 0
    generated_crops = 0
    warnings_list: List[str] = []
    new_rows: List[Dict[str, str]] = []

    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No se encontraron fotografías en {images_dir}.")

    for img_path in image_files:
        meta = metadata.get(img_path.name)
        if meta is None:
            warnings_list.append(
                f"Imagen omitida (sin registro en metadata.csv): {img_path.name}"
            )
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        annotations, label_errors = parse_yolo_label_file(label_path, class_map)
        warnings_list.extend(label_errors)
        if not annotations:
            warnings_list.append(f"Sin anotaciones válidas para {img_path.name}")
            continue

        image = safe_imread(img_path)
        if image is None or image.size == 0:
            warnings_list.append(f"Imagen dañada o ilegible: {img_path.name}")
            continue

        img_h, img_w = image.shape[:2]
        processed_images += 1

        region_counters: Dict[str, int] = defaultdict(int)
        for ann in annotations:
            region_name = ann.class_name.upper()
            if region_name not in VALID_REGIONS:
                # Solo se recortan regiones corporales CABEZA y MANO (no cajas de EPP solas)
                continue

            region_counters[region_name] += 1
            crop_idx = region_counters[region_name]
            margin = head_margin if region_name == "CABEZA" else hand_margin

            bbox = yolo_to_pixel_bbox(
                ann.x_center, ann.y_center, ann.width, ann.height, img_w, img_h
            )
            crop_img, exp_bbox = extract_region_crop(
                image,
                bbox=bbox,
                margin_ratio=margin,
                target_size=target_size,
                pad_color=pad_color,
            )

            crop_filename = f"{img_path.stem}_{region_name.lower()}_crop_{crop_idx:02d}.jpg"
            crop_path = pending_dir / crop_filename
            if not safe_imwrite(crop_path, crop_img):
                warnings_list.append(f"No se pudo guardar el recorte: {crop_filename}")
                continue

            generated_crops += 1
            prev = existing_rows.get(crop_filename)
            estado = prev["estado_revision"] if prev else "PENDIENTE"
            clase = prev["clase_cnn"] if prev else ""

            new_rows.append(
                {
                    "crop_filename": crop_filename,
                    "original_image": img_path.name,
                    "actividad": meta["actividad"],
                    "grupo_sesion": meta["grupo_sesion"],
                    "fuente": meta["fuente"],
                    "division_preasignada": meta["division"],
                    "region": region_name,
                    "crop_index": str(crop_idx),
                    "x1": str(bbox[0]),
                    "y1": str(bbox[1]),
                    "x2": str(bbox[2]),
                    "y2": str(bbox[3]),
                    "exp_x1": str(exp_bbox[0]),
                    "exp_y1": str(exp_bbox[1]),
                    "exp_x2": str(exp_bbox[2]),
                    "exp_y2": str(exp_bbox[3]),
                    "estado_revision": estado,
                    "clase_cnn": clase,
                }
            )

    save_review_csv(review_csv_path, new_rows)
    logger.info(
        "Extracción finalizada: %d fotografías procesadas | %d recortes generados | %d advertencias.",
        processed_images,
        generated_crops,
        len(warnings_list),
    )
    if warnings_list:
        for w in warnings_list[:20]:
            logger.warning("  - %s", w)
        if len(warnings_list) > 20:
            logger.warning("  ... (%d advertencias adicionales)", len(warnings_list) - 20)

    return {
        "processed_images": processed_images,
        "generated_crops": generated_crops,
        "warnings": warnings_list,
        "review_csv": str(review_csv_path),
    }


def interactive_review_pipeline(cfg: Dict[str, Any]) -> None:
    """Herramienta interactiva con OpenCV para etiquetar manualmente los recortes pendientes."""
    logger = setup_logging()
    pending_dir = resolve_project_path(cfg["paths"]["cnn_pending_review_dir"], PROJECT_ROOT)
    review_csv_path = pending_dir / "etiquetas_revision.csv"

    rows = load_review_csv(review_csv_path)
    if not rows:
        raise FileNotFoundError(
            f"No hay recortes registrados en {review_csv_path}. Ejecute primero --mode extract."
        )

    pending_indices = [i for i, r in enumerate(rows) if r.get("estado_revision", "PENDIENTE") == "PENDIENTE"]
    if not pending_indices:
        logger.info("Todos los recortes ya fueron revisados (%d registros).", len(rows))
        return

    window_name = "SafeVision AI - Revision Manual de Recortes CNN"
    logger.info(
        "Iniciando revisión manual: %d pendientes de %d totales.",
        len(pending_indices),
        len(rows),
    )
    logger.info("Controles: [1] Uso correcto | [2] Uso incorrecto | [3] Descartar | [N] Siguiente | [Q] Guardar y salir")

    pos = 0
    while pos < len(pending_indices):
        idx = pending_indices[pos]
        row = rows[idx]
        crop_path = pending_dir / row["crop_filename"]
        img = safe_imread(crop_path)
        if img is None:
            logger.warning("No se pudo abrir %s; se marcará como DESCARTADO.", crop_path)
            row["estado_revision"] = "DESCARTADO"
            row["clase_cnn"] = "DESCARTADO"
            save_review_csv(review_csv_path, rows)
            pos += 1
            continue

        correct_cls, incorrect_cls = get_valid_cnn_classes_for(row["region"], row["actividad"])
        display = cv2.resize(img, (448, 448), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.copyMakeBorder(display, 95, 70, 20, 20, cv2.BORDER_CONSTANT, value=(30, 30, 30))

        cv2.putText(
            canvas,
            f"Progreso: {pos + 1}/{len(pending_indices)} | {row['crop_filename']}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Actividad: {row['actividad'].upper()} | Region: {row['region']} | Grupo: {row['grupo_sesion']}",
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"[1] {correct_cls}  |  [2] {incorrect_cls}",
            (15, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (120, 255, 120),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "[1] Correcto  [2] Incorrecto  [3] Descartar  [N] Saltar  [Q] Guardar y salir",
            (15, canvas.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            save_review_csv(review_csv_path, rows)
            logger.info("Progreso guardado correctamente en %s.", review_csv_path)
            break
        if key in (ord("n"), ord("N")):
            pos += 1
            continue
        if key in (ord("1"), ord("2"), ord("3")):
            decision_char = chr(key)
            estado, clase = map_manual_decision_to_class(
                row["region"], row["actividad"], decision_char
            )
            row["estado_revision"] = estado
            row["clase_cnn"] = clase
            save_review_csv(review_csv_path, rows)
            pos += 1
        else:
            logger.info("Tecla no reconocida. Use 1, 2, 3, N o Q.")

    cv2.destroyAllWindows()
    save_review_csv(review_csv_path, rows)


def build_final_cnn_dataset(cfg: Dict[str, Any], clean_existing: bool = True) -> Dict[str, Any]:
    """Construye datasets/cnn/{train,val,test}/<8_clases>/ agrupado por sesión/foto y genera informe."""
    logger = setup_logging()
    paths_cfg = cfg["paths"]
    split_cfg = cfg["dataset_split"]

    cnn_dir = resolve_project_path(paths_cfg["cnn_dataset_dir"], PROJECT_ROOT)
    pending_dir = resolve_project_path(paths_cfg["cnn_pending_review_dir"], PROJECT_ROOT)
    review_csv_path = pending_dir / "etiquetas_revision.csv"

    rows = load_review_csv(review_csv_path)
    if not rows:
        raise FileNotFoundError(f"No se encontró {review_csv_path}. Ejecute --mode extract y --mode review.")

    valid_rows = [
        r for r in rows if r.get("estado_revision") in ("CORRECTO", "INCORRECTO") and r.get("clase_cnn") in CNN_DEFAULT_CLASSES
    ]
    discarded_rows = [r for r in rows if r.get("estado_revision") == "DESCARTADO"]
    pending_rows = [r for r in rows if r.get("estado_revision") == "PENDIENTE"]

    if not valid_rows:
        raise ValueError(
            "No existen recortes clasificados como CORRECTO o INCORRECTO. "
            "Complete primero la revisión manual con --mode review."
        )

    if clean_existing:
        for split in ("train", "val", "test"):
            split_path = cnn_dir / split
            if split_path.exists():
                shutil.rmtree(split_path)

    for split in ("train", "val", "test"):
        for cls_name in CNN_DEFAULT_CLASSES:
            (cnn_dir / split / cls_name).mkdir(parents=True, exist_ok=True)

    group_splits = assign_group_splits(
        valid_rows,
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )

    counts_per_class: Counter[str] = Counter()
    counts_per_split: Counter[str] = Counter()
    counts_matrix: Dict[str, Counter[str]] = {s: Counter() for s in ("train", "val", "test")}
    missing_files: List[str] = []

    for row in valid_rows:
        src_file = pending_dir / row["crop_filename"]
        if not src_file.exists():
            missing_files.append(row["crop_filename"])
            continue
        group_id = row["grupo_sesion"].strip() or Path(row["original_image"]).stem
        split = group_splits[group_id]
        cls_name = row["clase_cnn"]
        dst_file = cnn_dir / split / cls_name / row["crop_filename"]
        shutil.copy2(src_file, dst_file)

        counts_per_class[cls_name] += 1
        counts_per_split[split] += 1
        counts_matrix[split][cls_name] += 1

    min_samples_warn = int(split_cfg.get("min_samples_warning", 20))
    imbalance_ratio_warn = float(split_cfg.get("imbalance_ratio_warning", 3.0))

    insufficient_classes = [
        cls for cls in CNN_DEFAULT_CLASSES if counts_per_class.get(cls, 0) < min_samples_warn
    ]

    nonzero_counts = [counts_per_class[c] for c in CNN_DEFAULT_CLASSES if counts_per_class[c] > 0]
    imbalance_warnings: List[str] = []
    if nonzero_counts:
        max_c = max(nonzero_counts)
        min_c = min(nonzero_counts)
        ratio = max_c / float(min_c) if min_c > 0 else float("inf")
        if ratio >= imbalance_ratio_warn:
            imbalance_warnings.append(
                f"Desbalance de clases detectado (ratio máx/mín = {ratio:.2f} >= {imbalance_ratio_warn:.2f}). "
                "Se recomienda recolectar más muestras o activar use_class_weights: true en config.yaml."
            )

    original_images_unique = sorted({r["original_image"] for r in rows})
    report: Dict[str, Any] = {
        "fotografias_procesadas": len(original_images_unique),
        "recortes_totales_registrados": len(rows),
        "recortes_validos_exportados": sum(counts_per_split.values()),
        "recortes_descartados": len(discarded_rows),
        "recortes_pendientes_sin_revisar": len(pending_rows),
        "archivos_faltantes": missing_files,
        "cantidad_por_clase": {c: counts_per_class.get(c, 0) for c in CNN_DEFAULT_CLASSES},
        "cantidad_por_conjunto": {s: counts_per_split.get(s, 0) for s in ("train", "val", "test")},
        "detalle_por_conjunto_y_clase": {
            s: {c: counts_matrix[s].get(c, 0) for c in CNN_DEFAULT_CLASSES}
            for s in ("train", "val", "test")
        },
        "categorias_muestras_insuficientes": insufficient_classes,
        "advertencias_desbalance": imbalance_warnings,
    }

    report_path = cnn_dir / "informe_dataset_cnn.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("=== INFORME DEL DATASET CNN ===")
    logger.info("Fotografías originales procesadas : %d", report["fotografias_procesadas"])
    logger.info("Recortes exportados               : %d", report["recortes_validos_exportados"])
    logger.info("Recortes descartados              : %d", report["recortes_descartados"])
    logger.info("Recortes pendientes               : %d", report["recortes_pendientes_sin_revisar"])
    logger.info("Distribución por conjunto         : %s", report["cantidad_por_conjunto"])
    logger.info("Distribución por clase            : %s", report["cantidad_por_clase"])
    if insufficient_classes:
        logger.warning(
            "Categorías con menos de %d muestras: %s", min_samples_warn, insufficient_classes
        )
    for warn in imbalance_warnings:
        logger.warning(warn)
    logger.info("Informe completo guardado en: %s", report_path)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generador y revisor manual del dataset de la CNN propia de SafeVision AI."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("extract", "review", "build", "all"),
        required=True,
        help="Etapa a ejecutar: extract (recortar), review (clasificar manual), build (dividir 70/15/15), all.",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Ruta a config.yaml")
    parser.add_argument("--size", type=int, default=None, help="Resolución cuadrada del recorte (defecto: 224)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.mode in ("extract", "all"):
        extract_crops_pipeline(cfg, target_size_override=args.size)
    if args.mode in ("review", "all"):
        interactive_review_pipeline(cfg)
    if args.mode in ("build", "all"):
        build_final_cnn_dataset(cfg)


if __name__ == "__main__":
    main()
