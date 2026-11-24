"""Módulo de detección de objetos basado en Ultralytics YOLO y estabilización espacial.

Incluye:
- YOLODetector con umbral diferenciado para la clase MANO (conf=0.20 en inferencia para
  captar manos desnudas, aplicando conf_threshold normal al resto de clases) y filtrado
  estricto de falsos positivos (área mínima relativa, vinculación espacial a PERSONA y
  límite top-k por clase).
- DetectionTracker: tracker liviano por IoU entre fotogramas consecutivos con suavizado
  exponencial (EMA) de coordenadas y confianza, más histéresis de visibilidad
  (min_hits / max_missed) para eliminar parpadeos y destellos fantasma.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils import YOLO_DEFAULT_CLASSES, resolve_project_path, select_device, setup_logging


DEFAULT_TOP_K_PER_CLASS: Dict[str, int] = {
    "PERSONA": 1,
    "CABEZA": 1,
    "MANO": 2,
    "GAFAS_SEGURIDAD": 1,
    "GUANTES_CORTE": 2,
    "MASCARA_SOLDADURA": 1,
    "GUANTES_SOLDADURA": 2,
}


@dataclass(frozen=True)
class Detection:
    """Representa una detección individual de YOLO en coordenadas de píxeles."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def compute_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """Calcula la Intersección sobre Unión (IoU) entre dos cajas (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def is_inside_expanded_box(
    det_bbox: Tuple[int, int, int, int],
    ref_bbox: Tuple[int, int, int, int],
    margin_ratio: float = 0.15,
) -> bool:
    """Verifica si el centro de `det_bbox` cae dentro del área expandida de `ref_bbox`."""
    rx1, ry1, rx2, ry2 = ref_bbox
    rw = max(1, rx2 - rx1)
    rh = max(1, ry2 - ry1)
    pad_x = rw * max(0.0, margin_ratio)
    pad_y = rh * max(0.0, margin_ratio)

    cx = (det_bbox[0] + det_bbox[2]) / 2.0
    cy = (det_bbox[1] + det_bbox[3]) / 2.0
    return (rx1 - pad_x) <= cx <= (rx2 + pad_x) and (ry1 - pad_y) <= cy <= (ry2 + pad_y)


def filter_false_positives(
    detections: List[Detection],
    frame_shape: Tuple[int, int],
    conf_threshold: float = 0.35,
    min_area_ratio: float = 0.008,
    top_k_per_class: Optional[Dict[str, int]] = None,
    link_to_person: bool = True,
    person_margin_ratio: float = 0.15,
    class_conf_thresholds: Optional[Dict[str, float]] = None,
) -> List[Detection]:
    """Aplica filtrado de falsos positivos con soporte de umbrales por clase (ej. MANO=0.20).

    1. Aplica `class_conf_thresholds[clase]` si existe; de lo contrario `conf_threshold`.
    2. Descarta cajas imposibles cuya área sea menor a `min_area_ratio` del fotograma.
    3. Si `link_to_person=True` y existe al menos una detección `PERSONA`, retiene solo
       regiones y EPP espacialmente vinculados a las personas detectadas.
    4. Conserva únicamente las `top_k` detecciones más confiables por clase.
    """
    if not detections:
        return []

    img_h, img_w = int(frame_shape[0]), int(frame_shape[1])
    frame_area = max(1, img_h * img_w)
    min_area_px = frame_area * max(0.0, float(min_area_ratio))

    per_class_conf = (
        {k.strip().upper(): float(v) for k, v in class_conf_thresholds.items()}
        if class_conf_thresholds is not None
        else {}
    )

    # 1 y 2. Filtro de confianza diferenciada por clase y área mínima
    filtered: List[Detection] = []
    for d in detections:
        cls_key = d.class_name.upper()
        min_conf = per_class_conf.get(cls_key, float(conf_threshold))
        if d.confidence >= min_conf and d.area >= min_area_px:
            filtered.append(d)

    if not filtered:
        return []

    # Limitar primero PERSONA si hay top_k definido para vincular con el operario principal
    top_k_map = (
        {k.strip().upper(): int(v) for k, v in top_k_per_class.items()}
        if top_k_per_class is not None
        else {}
    )

    persons = sorted(
        [d for d in filtered if d.class_name.upper() == "PERSONA"],
        key=lambda d: d.confidence,
        reverse=True,
    )
    if "PERSONA" in top_k_map and top_k_map["PERSONA"] > 0:
        persons = persons[: top_k_map["PERSONA"]]

    # 3. Vinculación espacial con PERSONA (si hay persona detectada en el encuadre)
    if link_to_person and persons:
        linked: List[Detection] = list(persons)
        for d in filtered:
            if d.class_name.upper() == "PERSONA":
                continue
            if any(
                is_inside_expanded_box(d.bbox, p.bbox, margin_ratio=person_margin_ratio)
                for p in persons
            ):
                linked.append(d)
        filtered = linked

    # 4. Top-k por clase según confianza descendente
    if not top_k_map:
        return filtered

    grouped: Dict[str, List[Detection]] = {}
    for d in filtered:
        cls_key = d.class_name.upper()
        grouped.setdefault(cls_key, []).append(d)

    result: List[Detection] = []
    for cls_key, items in grouped.items():
        items_sorted = sorted(items, key=lambda d: d.confidence, reverse=True)
        k = top_k_map.get(cls_key)
        if k is not None and k > 0:
            items_sorted = items_sorted[:k]
        result.extend(items_sorted)

    return result


@dataclass
class _TrackState:
    """Estado interno de una caja rastreada con suavizado EMA e histéresis."""

    track_id: int
    class_id: int
    class_name: str
    smoothed_bbox: Tuple[float, float, float, float]
    smoothed_conf: float
    hits: int = 1
    missed: int = 0
    confirmed: bool = False


class DetectionTracker:
    """Tracker liviano por IoU con suavizado exponencial (EMA) e histéresis de visibilidad.

    - Requiere `min_hits` frames consecutivos para mostrar una caja nueva (elimina destellos).
    - Mantiene visible una caja confirmada hasta `max_missed` frames si se pierde brevemente.
    - Suaviza coordenadas [x1, y1, x2, y2] y confianza mediante EMA (`ema_alpha`).
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        ema_alpha: float = 0.35,
        min_hits: int = 3,
        max_missed: int = 5,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.ema_alpha = max(0.05, min(1.0, float(ema_alpha)))
        self.min_hits = max(1, int(min_hits))
        self.max_missed = max(0, int(max_missed))
        self._next_id = 1
        self._tracks: List[_TrackState] = []

    def reset(self) -> None:
        """Limpia todos los tracks activos."""
        self._tracks.clear()
        self._next_id = 1

    def update(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],
    ) -> List[Detection]:
        """Asocia detecciones nuevas con tracks activos y devuelve las cajas estabilizadas."""
        img_h, img_w = int(frame_shape[0]), int(frame_shape[1])

        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()

        # Emparejamiento greedy por mayor IoU dentro de la misma clase
        candidate_pairs: List[Tuple[float, int, int]] = []
        for t_idx, track in enumerate(self._tracks):
            for d_idx, det in enumerate(detections):
                if track.class_name.upper() != det.class_name.upper():
                    continue
                iou = compute_iou(track.smoothed_bbox, det.bbox)
                if iou >= self.iou_threshold:
                    candidate_pairs.append((iou, t_idx, d_idx))

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)
        for _, t_idx, d_idx in candidate_pairs:
            if t_idx in matched_track_indices or d_idx in matched_det_indices:
                continue
            matched_track_indices.add(t_idx)
            matched_det_indices.add(d_idx)

            track = self._tracks[t_idx]
            det = detections[d_idx]
            a = self.ema_alpha
            ox1, oy1, ox2, oy2 = track.smoothed_bbox
            nx1, ny1, nx2, ny2 = det.bbox
            track.smoothed_bbox = (
                a * nx1 + (1.0 - a) * ox1,
                a * ny1 + (1.0 - a) * oy1,
                a * nx2 + (1.0 - a) * ox2,
                a * ny2 + (1.0 - a) * oy2,
            )
            track.smoothed_conf = a * det.confidence + (1.0 - a) * track.smoothed_conf
            track.hits += 1
            track.missed = 0
            if track.hits >= self.min_hits:
                track.confirmed = True

        # Tracks no emparejados: incrementar missed o eliminar si exceden max_missed
        surviving_tracks: List[_TrackState] = []
        for t_idx, track in enumerate(self._tracks):
            if t_idx in matched_track_indices:
                surviving_tracks.append(track)
                continue

            track.missed += 1
            if track.confirmed:
                if track.missed <= self.max_missed:
                    surviving_tracks.append(track)
            else:
                # Track aún no confirmado que se pierde inmediatamente -> descartar (destello)
                continue

        # Crear tracks nuevos para detecciones sin emparejar
        for d_idx, det in enumerate(detections):
            if d_idx in matched_det_indices:
                continue
            is_conf = self.min_hits <= 1
            surviving_tracks.append(
                _TrackState(
                    track_id=self._next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    smoothed_bbox=(
                        float(det.bbox[0]),
                        float(det.bbox[1]),
                        float(det.bbox[2]),
                        float(det.bbox[3]),
                    ),
                    smoothed_conf=float(det.confidence),
                    hits=1,
                    missed=0,
                    confirmed=is_conf,
                )
            )
            self._next_id += 1

        self._tracks = surviving_tracks

        # Emitir únicamente tracks confirmados, acotados a los límites del fotograma
        stabilized: List[Detection] = []
        for track in self._tracks:
            if not track.confirmed:
                continue
            sx1, sy1, sx2, sy2 = track.smoothed_bbox
            x1 = max(0, min(int(round(sx1)), img_w - 1))
            y1 = max(0, min(int(round(sy1)), img_h - 1))
            x2 = max(x1 + 1, min(int(round(sx2)), img_w))
            y2 = max(y1 + 1, min(int(round(sy2)), img_h))
            stabilized.append(
                Detection(
                    class_id=track.class_id,
                    class_name=track.class_name,
                    confidence=float(track.smoothed_conf),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return stabilized


class YOLODetector:
    """Envoltorio modular sobre Ultralytics YOLO con umbral diferenciado para MANO."""

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
        expected_classes: Optional[Dict[int, str]] = None,
        agnostic_nms: bool = True,
        min_area_ratio: float = 0.008,
        top_k_per_class: Optional[Dict[str, int]] = None,
        link_to_person: bool = True,
        person_margin_ratio: float = 0.15,
        hand_conf_threshold: float = 0.20,
    ) -> None:
        self.logger = setup_logging()
        self.weights_path = resolve_project_path(weights_path)
        self.conf_threshold = float(conf_threshold)
        self.hand_conf_threshold = float(hand_conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.device = device if device is not None else select_device(prefer_cuda=True)
        self.expected_classes = expected_classes or dict(YOLO_DEFAULT_CLASSES)
        self.agnostic_nms = bool(agnostic_nms)
        self.min_area_ratio = float(min_area_ratio)
        self.top_k_per_class = (
            dict(top_k_per_class) if top_k_per_class is not None else dict(DEFAULT_TOP_K_PER_CLASS)
        )
        self.link_to_person = bool(link_to_person)
        self.person_margin_ratio = float(person_margin_ratio)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de pesos de YOLO en: {self.weights_path}. "
                "Entrene el modelo primero ejecutando scripts/train_yolo.py."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "La librería 'ultralytics' no está instalada. "
                "Instálela con: pip install ultralytics"
            ) from exc

        try:
            self.model = YOLO(str(self.weights_path))
        except Exception as exc:
            raise RuntimeError(
                f"No se pudieron cargar los pesos de YOLO desde {self.weights_path}: {exc}"
            ) from exc

        self.class_names: Dict[int, str] = self._extract_class_names()
        self._validate_classes()
        self.logger.info(
            "Modelo YOLO cargado correctamente desde %s (dispositivo=%s, conf=%.2f, conf_mano=%.2f, iou=%.2f, agnostic_nms=%s).",
            self.weights_path,
            self.device,
            self.conf_threshold,
            self.hand_conf_threshold,
            self.iou_threshold,
            self.agnostic_nms,
        )

    def _extract_class_names(self) -> Dict[int, str]:
        """Obtiene el mapa de clases del modelo cargado."""
        raw_names = getattr(self.model, "names", None)
        if isinstance(raw_names, dict):
            return {int(k): str(v) for k, v in raw_names.items()}
        if isinstance(raw_names, list):
            return {i: str(v) for i, v in enumerate(raw_names)}
        return dict(self.expected_classes)

    def _validate_classes(self) -> None:
        """Verifica que el modelo contenga las clases esperadas y advierte si difieren."""
        missing = [
            name for name in self.expected_classes.values() if name not in self.class_names.values()
        ]
        if missing:
            self.logger.warning(
                "El modelo YOLO cargado no contiene todas las clases esperadas de SafeVision AI. "
                "Faltantes: %s. Clases presentes: %s",
                missing,
                list(self.class_names.values()),
            )

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Ejecuta inferencia YOLO con conf=min(conf, hand_conf) y filtra por clase."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("El fotograma proporcionado a YOLODetector.detect() está vacío.")

        img_h, img_w = frame_bgr.shape[:2]
        # Usar el umbral mínimo (0.20) en la inferencia bruta para captar manos desnudas
        predict_conf = min(self.conf_threshold, self.hand_conf_threshold)

        try:
            results = self.model.predict(
                source=frame_bgr,
                conf=predict_conf,
                iou=self.iou_threshold,
                agnostic_nms=self.agnostic_nms,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device != "cpu":
                self.logger.warning(
                    "Memoria GPU agotada en YOLO. Cambiando automáticamente a CPU."
                )
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.device = "cpu"
                results = self.model.predict(
                    source=frame_bgr,
                    conf=predict_conf,
                    iou=self.iou_threshold,
                    agnostic_nms=self.agnostic_nms,
                    imgsz=self.imgsz,
                    device="cpu",
                    verbose=False,
                )
            else:
                raise

        raw_detections: List[Detection] = []
        if not results:
            return raw_detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return raw_detections

        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)

        for coords, conf, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
            x1 = max(0, min(int(round(coords[0])), img_w - 1))
            y1 = max(0, min(int(round(coords[1])), img_h - 1))
            x2 = max(x1 + 1, min(int(round(coords[2])), img_w))
            y2 = max(y1 + 1, min(int(round(coords[3])), img_h))
            class_name = self.class_names.get(int(cls_id), f"DESCONOCIDA_{cls_id}")
            raw_detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=(x1, y1, x2, y2),
                )
            )

        return filter_false_positives(
            detections=raw_detections,
            frame_shape=(img_h, img_w),
            conf_threshold=self.conf_threshold,
            min_area_ratio=self.min_area_ratio,
            top_k_per_class=self.top_k_per_class,
            link_to_person=self.link_to_person,
            person_margin_ratio=self.person_margin_ratio,
            class_conf_thresholds={"MANO": self.hand_conf_threshold},
        )

    @staticmethod
    def filter_by_class(detections: List[Detection], class_name: str) -> List[Detection]:
        """Filtra detecciones por nombre de clase (insensible a mayúsculas)."""
        target = class_name.strip().upper()
        return [d for d in detections if d.class_name.upper() == target]
