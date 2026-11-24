"""Programa principal de inferencia en tiempo real de SafeVision AI.

Uso desde consola (CMD de Windows / terminal):
    python main.py --activity corte --source 0
    python main.py --activity soldadura --source 0
    python main.py --activity corte --source ruta/a/video.mp4
    python main.py --activity soldadura --source ruta/a/imagen.jpg

Controles en ventana OpenCV:
    Tecla Q: cerrar el programa correctamente.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.camera import VideoSource
from src.classifier import PPEClassifier
from src.decision_engine import (
    DecisionEngine,
    EvaluationResult,
    STATUS_CUMPLE,
    STATUS_ESPERANDO_OPERARIO,
    STATUS_NO_CUMPLE,
    STATUS_NO_VERIFICABLE,
)
from src.detector import Detection, DetectionTracker, YOLODetector
from src.utils import VALID_ACTIVITIES, load_config, select_device, setup_logging

# Colores BGR para visualización
STATUS_COLORS: Dict[str, Tuple[int, int, int]] = {
    STATUS_CUMPLE: (40, 180, 40),                # Verde
    STATUS_NO_CUMPLE: (35, 35, 220),             # Rojo
    STATUS_NO_VERIFICABLE: (0, 165, 255),        # Naranja/Ámbar
    STATUS_ESPERANDO_OPERARIO: (140, 115, 85),   # Azul grisáceo neutro
}

YOLO_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "PERSONA": (180, 180, 180),
    "CABEZA": (255, 200, 0),
    "MANO": (255, 140, 0),
    "GAFAS_SEGURIDAD": (200, 255, 100),
    "GUANTES_CORTE": (100, 255, 200),
    "MASCARA_SOLDADURA": (255, 100, 200),
    "GUANTES_SOLDADURA": (200, 120, 255),
}


def draw_overlay(
    frame: np.ndarray,
    detections: List[Detection],
    evaluation: EvaluationResult,
    fps: float,
) -> np.ndarray:
    """Dibuja cajas YOLO estabilizadas, resultados suavizados de la CNN, estado global y FPS."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Dibujar todas las detecciones YOLO estabilizadas
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = YOLO_CLASS_COLORS.get(det.class_name.upper(), (200, 200, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    # 2. Resaltar veredictos sobre cada región corporal evaluada
    for verdict in evaluation.region_verdicts:
        if verdict.bbox is None:
            continue
        x1, y1, x2, y2 = verdict.bbox
        v_color = STATUS_COLORS.get(verdict.status, (0, 165, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), v_color, 3)

        if verdict.predicted_class and verdict.confidence is not None:
            cnn_txt = (
                f"{verdict.region_label}: {verdict.predicted_class} "
                f"({verdict.confidence * 100:.1f}%)"
            )
        else:
            cnn_txt = f"{verdict.region_label}: {verdict.status}"

        (tw, th), _ = cv2.getTextSize(cnn_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        by = min(h - 6, y2 + th + 8)
        cv2.rectangle(canvas, (x1, by - th - 6), (min(w, x1 + tw + 8), by), v_color, -1)
        cv2.putText(
            canvas,
            cnn_txt,
            (x1 + 4, by - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Banner superior de estado global, actividad y FPS
    banner_color = STATUS_COLORS.get(evaluation.overall_status, (0, 165, 255))
    num_lines = max(1, len(evaluation.summary_messages))
    panel_height = 42 + 24 * num_lines
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    cv2.rectangle(canvas, (0, 0), (w, 36), banner_color, -1)
    header_text = (
        f"SafeVision AI | Actividad: {evaluation.activity.upper()} | "
        f"Estado: {evaluation.overall_status} | FPS: {fps:.1f} | [Q] Salir"
    )
    cv2.putText(
        canvas,
        header_text,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 4. Detalle por región (o mensaje "Esperando operario") en el panel superior
    for i, msg in enumerate(evaluation.summary_messages):
        y_pos = 58 + i * 22
        if i < len(evaluation.region_verdicts):
            v_color = STATUS_COLORS.get(evaluation.region_verdicts[i].status, (255, 255, 255))
        else:
            v_color = (220, 220, 220)
        cv2.putText(
            canvas,
            msg,
            (12, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            v_color,
            1,
            cv2.LINE_AA,
        )

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SafeVision AI - Verificación de uso correcto de EPP en tiempo real."
    )
    parser.add_argument(
        "--activity",
        type=str,
        required=True,
        choices=VALID_ACTIVITIES,
        help="Actividad a verificar: 'corte' o 'soldadura'.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Índice de webcam (ej. 0), ruta a video (.mp4) o ruta a imagen (.jpg/.png).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Ruta al archivo de configuración centralizada.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Ruta alternativa a los pesos de YOLO (por defecto usa config.yaml).",
    )
    parser.add_argument(
        "--cnn-weights",
        type=str,
        default=None,
        help="Ruta alternativa a los pesos de la CNN (por defecto usa config.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo de inferencia ('cuda' o 'cpu'). Por defecto detección automática.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Ejecuta sin abrir ventana OpenCV (útil para pruebas en servidores sin pantalla).",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> int:
    logger = setup_logging()

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        logger.error("Error cargando configuración: %s", exc)
        return 1

    device = args.device or select_device(prefer_cuda=True)
    yolo_cfg = cfg.get("yolo", {})
    stab_cfg = cfg.get("stabilization", {})
    cam_cfg = cfg.get("camera", {})

    yolo_weights = args.yolo_weights or cfg["paths"]["yolo_best_weights"]
    cnn_weights = args.cnn_weights or cfg["paths"]["cnn_best_weights"]
    window_name = str(cam_cfg.get("window_name", "SafeVision AI - Verificacion de EPP"))

    # Cargar ambos modelos UNA SOLA VEZ al iniciar el programa
    try:
        detector = YOLODetector(
            weights_path=yolo_weights,
            conf_threshold=float(yolo_cfg.get("conf_threshold", 0.35)),
            hand_conf_threshold=float(yolo_cfg.get("hand_conf_threshold", 0.20)),
            iou_threshold=float(yolo_cfg.get("iou_threshold", 0.45)),
            imgsz=int(yolo_cfg.get("imgsz", 640)),
            device=device,
            agnostic_nms=bool(yolo_cfg.get("agnostic_nms", True)),
            min_area_ratio=float(yolo_cfg.get("min_area_ratio", 0.008)),
            top_k_per_class=yolo_cfg.get("top_k_per_class"),
            link_to_person=bool(yolo_cfg.get("link_to_person", True)),
            person_margin_ratio=float(yolo_cfg.get("person_margin_ratio", 0.15)),
        )
        classifier = PPEClassifier(
            weights_path=cnn_weights,
            device=device,
            pad_color=cfg["crops"].get("pad_color", [114, 114, 114]),
            smooth_window_size=int(stab_cfg.get("window_size", 10)),
            smooth_ema_alpha=float(stab_cfg.get("prob_ema_alpha", 0.25)),
        )
        engine = DecisionEngine(cfg)
    except Exception as exc:
        logger.error("Error al inicializar modelos: %s", exc)
        return 1

    stabilization_enabled = bool(stab_cfg.get("enabled", True))
    tracker = DetectionTracker(
        iou_threshold=float(stab_cfg.get("tracker_iou_threshold", 0.30)),
        ema_alpha=float(stab_cfg.get("ema_alpha", 0.35)),
        min_hits=int(stab_cfg.get("min_hits", 3)),
        max_missed=int(stab_cfg.get("max_missed", 5)),
    )

    try:
        video_source = VideoSource(
            source=args.source,
            width=int(cam_cfg.get("default_width", 1280)),
            height=int(cam_cfg.get("default_height", 720)),
            fps=int(cam_cfg.get("default_fps", 30)),
        )
    except Exception as exc:
        logger.error("Error al abrir la fuente de entrada '%s': %s", args.source, exc)
        return 1

    logger.info(
        "SafeVision AI iniciado | Actividad=%s | Fuente=%s | Dispositivo=%s | Estabilizacion=%s",
        args.activity.upper(),
        args.source,
        device,
        "ON" if (stabilization_enabled and not video_source.is_image) else "OFF",
    )

    last_logged_status: Optional[str] = None

    with video_source:
        while True:
            ok, frame = video_source.read()
            if not ok or frame is None:
                if not video_source.is_image:
                    logger.info("Fin de la secuencia de video o señal interrumpida.")
                break

            raw_detections = detector.detect(frame)
            if stabilization_enabled and not video_source.is_image:
                detections = tracker.update(raw_detections, frame.shape[:2])
            else:
                detections = raw_detections

            evaluation = engine.evaluate(
                frame_bgr=frame,
                detections=detections,
                classifier=classifier,
                activity=args.activity,
            )
            fps = video_source.tick_fps()

            # Imprimir alertas en consola solo cuando cambie el estado global estabilizado
            if video_source.is_image or evaluation.overall_status != last_logged_status:
                logger.info(
                    "Estado global [%s]: %s (FPS: %.1f)",
                    args.activity.upper(),
                    evaluation.overall_status,
                    fps,
                )
                for msg in evaluation.summary_messages:
                    logger.info("  %s", msg)
                for alert in evaluation.alerts:
                    logger.warning("  %s", alert)
                last_logged_status = evaluation.overall_status

            if not args.no_display:
                vis_frame = draw_overlay(frame, detections, evaluation, fps)
                cv2.imshow(window_name, vis_frame)
                wait_ms = 0 if video_source.is_image else 1
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    logger.info("Cierre solicitado por el usuario (tecla Q).")
                    break

            if video_source.is_image:
                break

    if not args.no_display:
        cv2.destroyAllWindows()
    logger.info("SafeVision AI finalizado correctamente.")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
