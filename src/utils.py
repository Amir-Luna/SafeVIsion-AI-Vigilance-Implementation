"""Utilidades comunes de SafeVision AI.

Incluye carga de configuración centralizada, detección de dispositivo (CUDA/CPU),
lectura y validación de anotaciones YOLO, extracción de recortes con letterbox
sin deformación y evaluación de calidad visual de regiones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

# Constantes de actividades y regiones válidas
VALID_ACTIVITIES: Tuple[str, ...] = ("corte", "soldadura")
VALID_REGIONS: Tuple[str, ...] = ("CABEZA", "MANO")
SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Mapa determinista: (region, actividad) -> (clase_correcta, clase_incorrecta)
ACTIVITY_REGION_CLASS_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("CABEZA", "corte"): ("GAFAS_COLOCADAS", "GAFAS_NO_COLOCADAS"),
    ("MANO", "corte"): ("GUANTES_CORTE_COLOCADOS", "GUANTES_CORTE_NO_COLOCADOS"),
    ("CABEZA", "soldadura"): ("MASCARA_COLOCADA", "MASCARA_NO_COLOCADA"),
    ("MANO", "soldadura"): ("GUANTES_SOLDADURA_COLOCADOS", "GUANTES_SOLDADURA_NO_COLOCADOS"),
}

CNN_DEFAULT_CLASSES: List[str] = [
    "GAFAS_COLOCADAS",
    "GAFAS_NO_COLOCADAS",
    "GUANTES_CORTE_COLOCADOS",
    "GUANTES_CORTE_NO_COLOCADOS",
    "MASCARA_COLOCADA",
    "MASCARA_NO_COLOCADA",
    "GUANTES_SOLDADURA_COLOCADOS",
    "GUANTES_SOLDADURA_NO_COLOCADOS",
]

YOLO_DEFAULT_CLASSES: Dict[int, str] = {
    0: "PERSONA",
    1: "CABEZA",
    2: "MANO",
    3: "GAFAS_SEGURIDAD",
    4: "GUANTES_CORTE",
    5: "MASCARA_SOLDADURA",
    6: "GUANTES_SOLDADURA",
}


@dataclass(frozen=True)
class YoloAnnotation:
    """Representa una anotación YOLO validada."""

    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class QualityReport:
    """Resultado de la evaluación de calidad visual de una región recortada."""

    is_acceptable: bool
    blur_score: float
    brightness: float
    width: int
    height: int
    reasons: Tuple[str, ...]


def get_project_root() -> Path:
    """Devuelve la ruta raíz del proyecto SafeVision_AI de forma portable."""
    return Path(__file__).resolve().parent.parent


def resolve_project_path(relative_or_abs: str | Path, project_root: Optional[Path] = None) -> Path:
    """Resuelve una ruta relativa respecto a la raíz del proyecto."""
    path = Path(relative_or_abs)
    if path.is_absolute():
        return path
    root = project_root if project_root is not None else get_project_root()
    return (root / path).resolve()


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura el logger estándar de SafeVision AI."""
    logger = logging.getLogger("safevision")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Carga el archivo YAML de configuración centralizada."""
    root = get_project_root()
    cfg_file = (
        resolve_project_path(config_path, root)
        if config_path is not None
        else root / "configs" / "config.yaml"
    )
    if not cfg_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"El archivo de configuración está vacío o mal formado: {cfg_file}")

    return config


def load_data_yaml_classes(data_yaml_path: str | Path) -> Dict[int, str]:
    """Lee el diccionario de clases {id: nombre} desde configs/data.yaml."""
    path = resolve_project_path(data_yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró data.yaml en: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "names" not in data:
        raise ValueError(f"data.yaml no contiene la clave 'names': {path}")

    names = data["names"]
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    raise ValueError(f"Formato inválido para 'names' en {path}")


def select_device(prefer_cuda: bool = True) -> str:
    """Detecta automáticamente CUDA si está disponible; de lo contrario usa CPU."""
    if not prefer_cuda:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_valid_cnn_classes_for(region: str, activity: str) -> Tuple[str, str]:
    """Devuelve (clase_correcta, clase_incorrecta) para una región y actividad."""
    region_norm = region.strip().upper()
    activity_norm = activity.strip().lower()
    key = (region_norm, activity_norm)
    if key not in ACTIVITY_REGION_CLASS_MAP:
        raise ValueError(
            f"Combinación inválida región='{region}' y actividad='{activity}'. "
            f"Regiones válidas: {VALID_REGIONS}; actividades válidas: {VALID_ACTIVITIES}."
        )
    return ACTIVITY_REGION_CLASS_MAP[key]


def parse_yolo_label_file(
    label_path: str | Path,
    class_map: Dict[int, str],
) -> Tuple[List[YoloAnnotation], List[str]]:
    """Lee y valida un archivo de anotaciones YOLO (.txt).

    Retorna:
        (lista_anotaciones_validas, lista_advertencias_o_errores)
    """
    path = Path(label_path)
    annotations: List[YoloAnnotation] = []
    errors: List[str] = []

    if not path.exists():
        return annotations, [f"Archivo de etiqueta no encontrado: {path}"]

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return annotations, [f"Error al leer etiqueta {path}: {exc}"]

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"{path.name}:L{line_num} formato inválido (se esperaban 5 valores, se encontraron {len(parts)})."
            )
            continue
        try:
            class_id = int(parts[0])
            xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            errors.append(f"{path.name}:L{line_num} valores no numéricos: '{line}'.")
            continue

        if class_id not in class_map:
            errors.append(f"{path.name}:L{line_num} class_id desconocido: {class_id}.")
            continue

        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append(
                f"{path.name}:L{line_num} coordenadas fuera de rango normalizado [0, 1]: "
                f"({xc}, {yc}, {w}, {h})."
            )
            continue

        annotations.append(
            YoloAnnotation(
                class_id=class_id,
                class_name=class_map[class_id],
                x_center=xc,
                y_center=yc,
                width=w,
                height=h,
            )
        )

    return annotations, errors


def yolo_to_pixel_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> Tuple[int, int, int, int]:
    """Convierte coordenadas YOLO normalizadas a caja en píxeles (x1, y1, x2, y2) acotada."""
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Dimensiones de imagen inválidas: {img_width}x{img_height}")

    box_w = width * img_width
    box_h = height * img_height
    cx = x_center * img_width
    cy = y_center * img_height

    x1 = int(round(cx - box_w / 2.0))
    y1 = int(round(cy - box_h / 2.0))
    x2 = int(round(cx + box_w / 2.0))
    y2 = int(round(cy + box_h / 2.0))

    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))
    return x1, y1, x2, y2


def expand_bbox_with_margin(
    bbox: Tuple[int, int, int, int],
    img_width: int,
    img_height: int,
    margin_ratio: float = 0.20,
) -> Tuple[int, int, int, int]:
    """Expande un bounding box (x1, y1, x2, y2) con un margen relativo sin salir de la imagen."""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad_x = int(round(w * max(0.0, margin_ratio)))
    pad_y = int(round(h * max(0.0, margin_ratio)))

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(img_width, x2 + pad_x)
    ny2 = min(img_height, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def letterbox_resize(
    image: np.ndarray,
    target_size: int = 224,
    pad_color: Sequence[int] = (114, 114, 114),
) -> np.ndarray:
    """Redimensiona una imagen a (target_size, target_size) manteniendo la proporción original."""
    if image is None or image.size == 0:
        raise ValueError("No se puede redimensionar una imagen vacía.")

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Dimensiones inválidas para letterbox: {w}x{h}")

    scale = min(target_size / float(w), target_size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    color = tuple(int(c) for c in pad_color[:3])
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded


def extract_region_crop(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin_ratio: float = 0.20,
    target_size: Optional[int] = 224,
    pad_color: Sequence[int] = (114, 114, 114),
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extrae un recorte con margen acotado y opcionalmente aplica letterbox sin deformación.

    Retorna:
        (imagen_recortada, caja_expandida_x1_y1_x2_y2)
    """
    if image is None or image.size == 0:
        raise ValueError("La imagen de entrada está vacía.")

    img_h, img_w = image.shape[:2]
    expanded = expand_bbox_with_margin(bbox, img_w, img_h, margin_ratio=margin_ratio)
    x1, y1, x2, y2 = expanded
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"El recorte resultó vacío para la caja {expanded}.")

    if target_size is not None:
        crop = letterbox_resize(crop, target_size=target_size, pad_color=pad_color)

    return crop, expanded


def evaluate_crop_quality(
    raw_crop: np.ndarray,
    min_width: int = 32,
    min_height: int = 32,
    min_blur_variance: float = 35.0,
    min_brightness: float = 25.0,
    max_brightness: float = 240.0,
) -> QualityReport:
    """Evalúa resolución, desenfoque (varianza del Laplaciano) y brillo de un recorte."""
    if raw_crop is None or raw_crop.size == 0:
        return QualityReport(
            is_acceptable=False,
            blur_score=0.0,
            brightness=0.0,
            width=0,
            height=0,
            reasons=("Recorte vacío o inválido",),
        )

    h, w = raw_crop.shape[:2]
    gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY) if raw_crop.ndim == 3 else raw_crop
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    reasons: List[str] = []
    if w < min_width or h < min_height:
        reasons.append(f"Resolución insuficiente ({w}x{h}px < {min_width}x{min_height}px)")
    if blur_score < min_blur_variance:
        reasons.append(f"Imagen borrosa (nitidez={blur_score:.1f} < {min_blur_variance:.1f})")
    if brightness < min_brightness:
        reasons.append(f"Iluminación insuficiente (brillo={brightness:.1f})")
    elif brightness > max_brightness:
        reasons.append(f"Imagen sobreexpuesta (brillo={brightness:.1f})")

    return QualityReport(
        is_acceptable=len(reasons) == 0,
        blur_score=blur_score,
        brightness=brightness,
        width=w,
        height=h,
        reasons=tuple(reasons),
    )


def safe_imread(image_path: str | Path) -> Optional[np.ndarray]:
    """Lee una imagen soportando rutas con caracteres especiales/acentos en Windows."""
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        if raw_bytes.size == 0:
            return None
        return cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    except OSError:
        return None


def safe_imwrite(image_path: str | Path, image: np.ndarray) -> bool:
    """Guarda una imagen soportando rutas con caracteres especiales/acentos en Windows."""
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False
    try:
        encoded.tofile(str(path))
        return True
    except OSError:
        return False
