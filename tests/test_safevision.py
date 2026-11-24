"""Pruebas unitarias para verificar los componentes críticos de SafeVision AI.

Cubre:
- Lectura y validación de anotaciones YOLO.
- Conversión de coordenadas normalizadas a píxeles y expansión con márgenes.
- Extracción de recortes con letterbox sin deformación.
- Asignación de categorías por región y actividad.
- División de dataset agrupada por sesión/fotografía (sin fuga de datos).
- Control de calidad visual (resolución y desenfoque).
- Motor de decisión (CUMPLE, NO_CUMPLE, NO_VERIFICABLE, Esperando operario).
- Umbral diferenciado para MANO (0.20) vs resto de clases (0.35).
- Fusión de evidencia YOLO + CNN (guantes y cabeza estricta >= 0.85 sin YOLO).
- Regla Fail-Safe cuando hay PERSONA/CABEZA sin MANO ni guantes válidos.
- Estabilización espacial (DetectionTracker con IoU, EMA e histéresis min_hits / max_missed).
- Estabilización temporal de probabilidades (ProbabilitySmoother) e histéresis de estado.
- Verificación de errores por pesos inexistentes e incompatibles.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts.generar_dataset_cnn import assign_group_splits, map_manual_decision_to_class
from src.classifier import (
    ClassificationResult,
    PPEClassifier,
    ProbabilitySmoother,
    SafeVisionCNN,
    TORCH_AVAILABLE,
)
from src.decision_engine import (
    DecisionEngine,
    STATUS_CUMPLE,
    STATUS_ESPERANDO_OPERARIO,
    STATUS_NO_CUMPLE,
    STATUS_NO_VERIFICABLE,
)
from src.detector import (
    Detection,
    DetectionTracker,
    YOLODetector,
    filter_false_positives,
)
from src.utils import (
    YOLO_DEFAULT_CLASSES,
    evaluate_crop_quality,
    expand_bbox_with_margin,
    extract_region_crop,
    get_valid_cnn_classes_for,
    letterbox_resize,
    load_config,
    parse_yolo_label_file,
    yolo_to_pixel_bbox,
)


def _make_sharp_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Genera una imagen sintética nítida (patrón de tablero) para superar el filtro de desenfoque."""
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            if ((x // 8) + (y // 8)) % 2 == 0:
                img[y : y + 8, x : x + 8] = (200, 180, 160)
            else:
                img[y : y + 8, x : x + 8] = (60, 70, 80)
    return img


class MockClassifier:
    """Clasificador simulado para probar el motor de decisión de forma determinista."""

    def __init__(self, responses: Dict[str, List[ClassificationResult]]) -> None:
        self.responses = {k: list(v) for k, v in responses.items()}

    def predict_region(
        self,
        crop_bgr: np.ndarray,
        region: str,
        activity: str,
    ) -> ClassificationResult:
        queue = self.responses[region.upper()]
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]


class TestYoloUtilsAndCrops(unittest.TestCase):
    """Pruebas de anotaciones YOLO, coordenadas y extracción de recortes."""

    def test_parse_yolo_label_valid_and_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lbl = Path(tmp) / "img_001.txt"
            lbl.write_text(
                "0 0.5 0.5 0.4 0.8\n"
                "1 0.5 0.2 0.2 0.2\n"
                "99 0.5 0.5 0.1 0.1\n"       # clase desconocida
                "2 1.2 0.5 0.1 0.1\n"        # fuera de rango
                "2 0.5 0.5\n",               # columnas incompletas
                encoding="utf-8",
            )
            anns, errors = parse_yolo_label_file(lbl, YOLO_DEFAULT_CLASSES)
            self.assertEqual(len(anns), 2)
            self.assertEqual(anns[0].class_name, "PERSONA")
            self.assertEqual(anns[1].class_name, "CABEZA")
            self.assertEqual(len(errors), 3)

    def test_yolo_to_pixel_bbox_and_margin_clamping(self) -> None:
        bbox = yolo_to_pixel_bbox(0.5, 0.5, 0.2, 0.4, img_width=1000, img_height=500)
        self.assertEqual(bbox, (400, 150, 600, 350))

        # Caja pegada al borde superior izquierdo no debe generar coordenadas negativas
        edge_bbox = yolo_to_pixel_bbox(0.05, 0.05, 0.1, 0.1, img_width=200, img_height=200)
        expanded = expand_bbox_with_margin(edge_bbox, img_width=200, img_height=200, margin_ratio=0.5)
        self.assertGreaterEqual(expanded[0], 0)
        self.assertGreaterEqual(expanded[1], 0)
        self.assertLessEqual(expanded[2], 200)
        self.assertLessEqual(expanded[3], 200)

    def test_extract_crop_letterbox_preserves_dimensions(self) -> None:
        frame = _make_sharp_frame(640, 480)
        crop, exp_bbox = extract_region_crop(
            frame, bbox=(100, 50, 220, 290), margin_ratio=0.20, target_size=224
        )
        self.assertEqual(crop.shape, (224, 224, 3))
        self.assertLess(exp_bbox[0], 100)
        self.assertGreater(exp_bbox[2], 220)

    def test_letterbox_resize_rectangular_input(self) -> None:
        rect = np.zeros((100, 300, 3), dtype=np.uint8)
        out = letterbox_resize(rect, target_size=224, pad_color=(114, 114, 114))
        self.assertEqual(out.shape, (224, 224, 3))
        # Las franjas superior e inferior deben corresponder al padding
        self.assertEqual(tuple(out[0, 0]), (114, 114, 114))


class TestCategoryMappingAndSplit(unittest.TestCase):
    """Pruebas de asignación de categorías y división sin fuga de datos."""

    def test_activity_region_category_mapping(self) -> None:
        self.assertEqual(
            get_valid_cnn_classes_for("CABEZA", "corte"),
            ("GAFAS_COLOCADAS", "GAFAS_NO_COLOCADAS"),
        )
        self.assertEqual(
            get_valid_cnn_classes_for("MANO", "corte"),
            ("GUANTES_CORTE_COLOCADOS", "GUANTES_CORTE_NO_COLOCADOS"),
        )
        self.assertEqual(
            get_valid_cnn_classes_for("CABEZA", "soldadura"),
            ("MASCARA_COLOCADA", "MASCARA_NO_COLOCADA"),
        )
        self.assertEqual(
            get_valid_cnn_classes_for("MANO", "soldadura"),
            ("GUANTES_SOLDADURA_COLOCADOS", "GUANTES_SOLDADURA_NO_COLOCADOS"),
        )

        self.assertEqual(map_manual_decision_to_class("CABEZA", "corte", "1"), ("CORRECTO", "GAFAS_COLOCADAS"))
        self.assertEqual(map_manual_decision_to_class("MANO", "soldadura", "2"), ("INCORRECTO", "GUANTES_SOLDADURA_NO_COLOCADOS"))
        self.assertEqual(map_manual_decision_to_class("MANO", "corte", "3"), ("DESCARTADO", "DESCARTADO"))

    def test_group_split_prevents_data_leakage(self) -> None:
        rows = []
        for g in range(10):
            for crop_idx in range(3):
                rows.append(
                    {
                        "crop_filename": f"img_{g}_mano_{crop_idx}.jpg",
                        "original_image": f"img_{g}.jpg",
                        "actividad": "corte" if g % 2 == 0 else "soldadura",
                        "grupo_sesion": f"sesion_{g}",
                        "division_preasignada": "",
                    }
                )
        mapping = assign_group_splits(rows, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
        # Cada sesión debe aparecer en un único split
        self.assertEqual(len(mapping), 10)
        self.assertTrue(set(mapping.values()).issubset({"train", "val", "test"}))
        self.assertIn("train", mapping.values())
        self.assertIn("val", mapping.values())
        self.assertIn("test", mapping.values())


class TestDecisionEngine(unittest.TestCase):
    """Pruebas de la lógica de cumplimiento para corte, soldadura y regiones no verificables."""

    def setUp(self) -> None:
        self.cfg = load_config()
        self.engine = DecisionEngine(self.cfg)
        self.frame = _make_sharp_frame(640, 480)
        self.base_detections = [
            Detection(1, "CABEZA", 0.92, (260, 40, 380, 160)),
            Detection(2, "MANO", 0.89, (120, 260, 220, 360)),
            Detection(2, "MANO", 0.91, (420, 260, 520, 360)),
        ]

    def test_corte_cumple_completo(self) -> None:
        mock = MockClassifier(
            {
                "CABEZA": [
                    ClassificationResult("GAFAS_COLOCADAS", 0.95, True, {}, "GAFAS_COLOCADAS", 0.95)
                ],
                "MANO": [
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.92, True, {}, "GUANTES_CORTE_COLOCADOS", 0.92),
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.88, True, {}, "GUANTES_CORTE_COLOCADOS", 0.88),
                ],
            }
        )
        res = self.engine.evaluate(self.frame, self.base_detections, mock, activity="corte")
        self.assertEqual(res.overall_status, STATUS_CUMPLE)
        self.assertEqual(len(res.region_verdicts), 3)

    def test_soldadura_no_cumple_aunque_falte_una_mano(self) -> None:
        # Cabeza con máscara levantada (NO_CUMPLE) + solo 1 mano visible (la 2da es NO_VERIFICABLE)
        dets = [
            Detection(1, "CABEZA", 0.90, (260, 40, 380, 160)),
            Detection(2, "MANO", 0.88, (120, 260, 220, 360)),
        ]
        mock = MockClassifier(
            {
                "CABEZA": [
                    ClassificationResult("MASCARA_NO_COLOCADA", 0.91, False, {}, "MASCARA_NO_COLOCADA", 0.91)
                ],
                "MANO": [
                    ClassificationResult("GUANTES_SOLDADURA_COLOCADOS", 0.90, True, {}, "GUANTES_SOLDADURA_COLOCADOS", 0.90)
                ],
            }
        )
        res = self.engine.evaluate(self.frame, dets, mock, activity="soldadura")
        self.assertEqual(res.overall_status, STATUS_NO_CUMPLE)

    def test_no_verificable_si_solo_una_mano_visible(self) -> None:
        dets = [
            Detection(1, "CABEZA", 0.95, (260, 40, 380, 160)),
            Detection(2, "MANO", 0.93, (120, 260, 220, 360)),
        ]
        mock = MockClassifier(
            {
                "CABEZA": [
                    ClassificationResult("GAFAS_COLOCADAS", 0.95, True, {}, "GAFAS_COLOCADAS", 0.95)
                ],
                "MANO": [
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.93, True, {}, "GUANTES_CORTE_COLOCADOS", 0.93)
                ],
            }
        )
        res = self.engine.evaluate(self.frame, dets, mock, activity="corte")
        self.assertEqual(res.overall_status, STATUS_NO_VERIFICABLE)

    def test_no_verificable_por_baja_confianza_cnn(self) -> None:
        # Con apoyo YOLO de GAFAS_SEGURIDAD sobre CABEZA, baja confianza CNN (< 0.65) -> NO_VERIFICABLE
        dets = list(self.base_detections) + [
            Detection(3, "GAFAS_SEGURIDAD", 0.70, (280, 75, 360, 115)),
        ]
        mock = MockClassifier(
            {
                "CABEZA": [
                    ClassificationResult("GAFAS_COLOCADAS", 0.52, True, {}, "GAFAS_COLOCADAS", 0.52)
                ],
                "MANO": [
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.90, True, {}, "GUANTES_CORTE_COLOCADOS", 0.90),
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.90, True, {}, "GUANTES_CORTE_COLOCADOS", 0.90),
                ],
            }
        )
        res = self.engine.evaluate(self.frame, dets, mock, activity="corte")
        self.assertEqual(res.overall_status, STATUS_NO_VERIFICABLE)

    def test_no_verificable_por_imagen_borrosa(self) -> None:
        blurry_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        q = evaluate_crop_quality(blurry_frame[40:160, 260:380], min_blur_variance=35.0)
        self.assertFalse(q.is_acceptable)

        mock = MockClassifier(
            {
                "CABEZA": [
                    ClassificationResult("GAFAS_COLOCADAS", 0.99, True, {}, "GAFAS_COLOCADAS", 0.99)
                ],
                "MANO": [
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.99, True, {}, "GUANTES_CORTE_COLOCADOS", 0.99),
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.99, True, {}, "GUANTES_CORTE_COLOCADOS", 0.99),
                ],
            }
        )
        res = self.engine.evaluate(blurry_frame, self.base_detections, mock, activity="corte")
        self.assertEqual(res.overall_status, STATUS_NO_VERIFICABLE)


class TestEvidenceFusionAndFailSafe(unittest.TestCase):
    """Pruebas de umbral diferenciado para MANO, fusión YOLO+CNN y regla Fail-Safe."""

    def setUp(self) -> None:
        self.cfg = load_config()
        self.frame = _make_sharp_frame(640, 480)

    def test_differentiated_hand_threshold_keeps_bare_hand_at_020(self) -> None:
        dets = [
            Detection(0, "PERSONA", 0.85, (150, 20, 490, 460)),
            Detection(2, "MANO", 0.22, (160, 250, 240, 330)),       # >= 0.20 -> debe mantenerse
            Detection(2, "MANO", 0.15, (400, 250, 480, 330)),       # < 0.20 -> descartada
            Detection(3, "GAFAS_SEGURIDAD", 0.28, (270, 60, 360, 120)),  # < 0.35 -> descartada
            Detection(1, "CABEZA", 0.40, (260, 40, 380, 160)),      # >= 0.35 -> debe mantenerse
        ]
        filtered = filter_false_positives(
            detections=dets,
            frame_shape=(480, 640),
            conf_threshold=0.35,
            min_area_ratio=0.008,
            top_k_per_class={"PERSONA": 1, "CABEZA": 1, "MANO": 2, "GAFAS_SEGURIDAD": 1},
            link_to_person=True,
            class_conf_thresholds={"MANO": 0.20},
        )
        by_cls = [d.class_name for d in filtered]
        self.assertEqual(by_cls.count("MANO"), 1)
        self.assertEqual(by_cls.count("CABEZA"), 1)
        self.assertEqual(by_cls.count("GAFAS_SEGURIDAD"), 0)

    def test_yolo_gloves_override_uncertain_cnn(self) -> None:
        # YOLO detecta GUANTES_CORTE >= 0.35 dentro de PERSONA aunque la CNN dude sobre MANO
        engine = DecisionEngine(self.cfg)
        dets = [
            Detection(0, "PERSONA", 0.90, (100, 20, 540, 460)),
            Detection(1, "CABEZA", 0.92, (260, 40, 380, 160)),
            Detection(3, "GAFAS_SEGURIDAD", 0.80, (280, 75, 360, 115)),
            Detection(2, "MANO", 0.30, (120, 260, 220, 360)),
            Detection(4, "GUANTES_CORTE", 0.42, (125, 265, 218, 355)),
            Detection(4, "GUANTES_CORTE", 0.48, (420, 260, 520, 360)),  # sin caja MANO explícita
        ]
        mock = MockClassifier(
            {
                "CABEZA": [
                    ClassificationResult("GAFAS_COLOCADAS", 0.78, True, {}, "GAFAS_COLOCADAS", 0.78)
                ],
                "MANO": [
                    # Aunque la CNN prediga mal/baja confianza, los guantes YOLO validan ambos lados
                    ClassificationResult("GUANTES_CORTE_NO_COLOCADOS", 0.55, False, {}, "GUANTES_CORTE_NO_COLOCADOS", 0.55)
                ],
            }
        )
        res = engine.evaluate(self.frame, dets, mock, activity="corte")
        self.assertEqual(res.overall_status, STATUS_CUMPLE)
        hand_verdicts = [v for v in res.region_verdicts if v.region_type == "MANO"]
        self.assertEqual(len(hand_verdicts), 2)
        self.assertTrue(all(v.status == STATUS_CUMPLE for v in hand_verdicts))
        self.assertTrue(all(v.predicted_class == "GUANTES_CORTE_COLOCADOS" for v in hand_verdicts))

    def test_head_without_yolo_epp_requires_strict_cnn_conf(self) -> None:
        # Sin caja YOLO GAFAS_SEGURIDAD, CNN con 0.78 (< 0.85) debe clasificarse como GAFAS_NO_COLOCADAS
        engine = DecisionEngine(self.cfg)
        dets = [
            Detection(1, "CABEZA", 0.92, (260, 40, 380, 160)),
            Detection(2, "MANO", 0.89, (120, 260, 220, 360)),
            Detection(2, "MANO", 0.91, (420, 260, 520, 360)),
        ]
        mock_medium = MockClassifier(
            {
                "CABEZA": [
                    ClassificationResult("GAFAS_COLOCADAS", 0.78, True, {}, "GAFAS_COLOCADAS", 0.78)
                ],
                "MANO": [
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.92, True, {}, "GUANTES_CORTE_COLOCADOS", 0.92),
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.92, True, {}, "GUANTES_CORTE_COLOCADOS", 0.92),
                ],
            }
        )
        res_low = engine.evaluate(self.frame, dets, mock_medium, activity="corte")
        self.assertEqual(res_low.overall_status, STATUS_NO_CUMPLE)
        head_v = [v for v in res_low.region_verdicts if v.region_label == "CABEZA"][0]
        self.assertEqual(head_v.predicted_class, "GAFAS_NO_COLOCADAS")
        self.assertEqual(head_v.status, STATUS_NO_CUMPLE)

        # En soldadura, sin MASCARA_SOLDADURA de YOLO, CNN >= 0.85 (0.90) sí se acepta
        engine2 = DecisionEngine(self.cfg)
        mock_high = MockClassifier(
            {
                "CABEZA": [
                    ClassificationResult("MASCARA_COLOCADA", 0.90, True, {}, "MASCARA_COLOCADA", 0.90)
                ],
                "MANO": [
                    ClassificationResult("GUANTES_SOLDADURA_COLOCADOS", 0.92, True, {}, "GUANTES_SOLDADURA_COLOCADOS", 0.92),
                    ClassificationResult("GUANTES_SOLDADURA_COLOCADOS", 0.92, True, {}, "GUANTES_SOLDADURA_COLOCADOS", 0.92),
                ],
            }
        )
        res_high = engine2.evaluate(self.frame, dets, mock_high, activity="soldadura")
        self.assertEqual(res_high.overall_status, STATUS_CUMPLE)

    def test_fail_safe_person_or_head_without_hands_or_gloves_returns_no_cumple(self) -> None:
        # Operario visible (PERSONA + CABEZA con gafas) pero sin MANO ni guantes -> NO_CUMPLE
        engine = DecisionEngine(self.cfg)
        dets = [
            Detection(0, "PERSONA", 0.91, (100, 20, 540, 460)),
            Detection(1, "CABEZA", 0.93, (260, 40, 380, 160)),
            Detection(3, "GAFAS_SEGURIDAD", 0.88, (280, 75, 360, 115)),
        ]
        mock = MockClassifier(
            {
                "CABEZA": [
                    ClassificationResult("GAFAS_COLOCADAS", 0.95, True, {}, "GAFAS_COLOCADAS", 0.95)
                ],
            }
        )
        res = engine.evaluate(self.frame, dets, mock, activity="corte")
        self.assertEqual(res.overall_status, STATUS_NO_CUMPLE)
        self.assertTrue(
            any("Guantes no detectados / verifique manos" in msg for msg in res.summary_messages)
        )


class TestRealtimeStabilization(unittest.TestCase):
    """Pruebas de filtrado de falsos positivos, tracker IoU+EMA, suavizado e histéresis."""

    def test_filter_false_positives_area_person_link_and_topk(self) -> None:
        # Frame 640x480 = 307,200 px -> 0.8% = 2,457.6 px
        dets = [
            Detection(0, "PERSONA", 0.90, (150, 20, 490, 460)),
            Detection(1, "CABEZA", 0.88, (260, 40, 380, 160)),       # válida dentro de PERSONA
            Detection(1, "CABEZA", 0.72, (10, 10, 70, 70)),          # fantasma en el fondo fuera de PERSONA
            Detection(2, "MANO", 0.85, (160, 250, 240, 330)),        # válida
            Detection(2, "MANO", 0.82, (400, 250, 480, 330)),        # válida
            Detection(2, "MANO", 0.65, (280, 250, 360, 330)),        # excede top_k=2
            Detection(2, "MANO", 0.95, (200, 200, 220, 220)),        # área 400 px < 0.8%
        ]
        filtered = filter_false_positives(
            detections=dets,
            frame_shape=(480, 640),
            conf_threshold=0.35,
            min_area_ratio=0.008,
            top_k_per_class={"PERSONA": 1, "CABEZA": 1, "MANO": 2},
            link_to_person=True,
            person_margin_ratio=0.15,
        )
        names = [d.class_name for d in filtered]
        self.assertEqual(names.count("PERSONA"), 1)
        self.assertEqual(names.count("CABEZA"), 1)
        self.assertEqual(names.count("MANO"), 2)
        head = [d for d in filtered if d.class_name == "CABEZA"][0]
        self.assertEqual(head.bbox, (260, 40, 380, 160))

    def test_detection_tracker_hysteresis_and_ema(self) -> None:
        tracker = DetectionTracker(iou_threshold=0.30, ema_alpha=0.35, min_hits=3, max_missed=4)
        frame_shape = (480, 640)

        # Destello de 1 solo frame en el fondo -> nunca debe mostrarse
        flash = [Detection(1, "CABEZA", 0.80, (10, 10, 110, 110))]
        self.assertEqual(tracker.update(flash, frame_shape), [])
        self.assertEqual(tracker.update([], frame_shape), [])

        # Objeto real durante 3 frames consecutivos -> se confirma al 3er frame y suaviza coordenadas
        d1 = [Detection(1, "CABEZA", 0.90, (200, 100, 300, 200))]
        d2 = [Detection(1, "CABEZA", 0.90, (210, 100, 310, 200))]
        d3 = [Detection(1, "CABEZA", 0.90, (220, 100, 320, 200))]
        self.assertEqual(tracker.update(d1, frame_shape), [])
        self.assertEqual(tracker.update(d2, frame_shape), [])
        out3 = tracker.update(d3, frame_shape)
        self.assertEqual(len(out3), 1)
        self.assertGreater(out3[0].bbox[0], 200)
        self.assertLess(out3[0].bbox[0], 220)

        # Gracia de hasta max_missed=4 frames sin detección
        for _ in range(4):
            grace = tracker.update([], frame_shape)
            self.assertEqual(len(grace), 1)
        self.assertEqual(tracker.update([], frame_shape), [])

    def test_probability_smoother_dampens_spikes(self) -> None:
        smoother = ProbabilitySmoother(window_size=10, ema_alpha=0.25)
        for _ in range(5):
            p = smoother.update("CABEZA", {"GAFAS_COLOCADAS": 0.95, "GAFAS_NO_COLOCADAS": 0.05})
        self.assertGreater(p["GAFAS_COLOCADAS"], 0.90)

        after_spike = smoother.update("CABEZA", {"GAFAS_COLOCADAS": 0.10, "GAFAS_NO_COLOCADAS": 0.90})
        self.assertGreater(after_spike["GAFAS_COLOCADAS"], 0.80)

    def test_decision_engine_hysteresis_and_waiting_state(self) -> None:
        cfg = load_config()
        cfg["stabilization"]["min_consistent_frames"] = 4
        cfg["stabilization"]["window_size"] = 1
        cfg["stabilization"]["prob_ema_alpha"] = 1.0
        engine = DecisionEngine(cfg)
        frame = _make_sharp_frame(640, 480)
        dets = [
            Detection(1, "CABEZA", 0.92, (260, 40, 380, 160)),
            Detection(2, "MANO", 0.90, (120, 260, 220, 360)),
            Detection(2, "MANO", 0.90, (420, 260, 520, 360)),
        ]

        # Sin detecciones -> "Esperando operario"
        empty_res = engine.evaluate(frame, [], MockClassifier({}), activity="corte")
        self.assertEqual(empty_res.overall_status, STATUS_ESPERANDO_OPERARIO)
        self.assertEqual(len(empty_res.region_verdicts), 0)

        # Frame inicial con todo correcto -> CUMPLE
        mock_ok = MockClassifier(
            {
                "CABEZA": [ClassificationResult("GAFAS_COLOCADAS", 0.95, True, {}, "GAFAS_COLOCADAS", 0.95)],
                "MANO": [
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.94, True, {}, "GUANTES_CORTE_COLOCADOS", 0.94),
                ],
            }
        )
        res1 = engine.evaluate(frame, dets, mock_ok, activity="corte")
        self.assertEqual(res1.overall_status, STATUS_CUMPLE)

        # 3 frames consecutivos de fallo no alcanzan min_consistent_frames=4 -> mantiene CUMPLE
        mock_bad = MockClassifier(
            {
                "CABEZA": [ClassificationResult("GAFAS_NO_COLOCADAS", 0.95, False, {}, "GAFAS_NO_COLOCADAS", 0.95)],
                "MANO": [
                    ClassificationResult("GUANTES_CORTE_COLOCADOS", 0.94, True, {}, "GUANTES_CORTE_COLOCADOS", 0.94),
                ],
            }
        )
        for _ in range(3):
            res_mid = engine.evaluate(frame, dets, mock_bad, activity="corte")
            self.assertEqual(res_mid.overall_status, STATUS_CUMPLE)

        # Al 4to frame consecutivo consistente cambia limpiamente a NO_CUMPLE
        res_switch = engine.evaluate(frame, dets, mock_bad, activity="corte")
        self.assertEqual(res_switch.overall_status, STATUS_NO_CUMPLE)


class TestModelLoading(unittest.TestCase):
    """Pruebas de carga de pesos y validación de errores."""

    def test_yolo_missing_weights_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            YOLODetector(weights_path="models/yolo/inexistente.pt", device="cpu")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch no está instalado en este entorno")
    def test_cnn_forward_save_and_reload(self) -> None:
        import torch

        model = SafeVisionCNN(num_classes=8, dropout=0.4)
        dummy = torch.randn(2, 3, 224, 224)
        logits = model(dummy)
        self.assertEqual(tuple(logits.shape), (2, 8))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "test_cnn.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arch_config": model.get_arch_config(),
                    "class_names": [
                        "GAFAS_COLOCADAS",
                        "GAFAS_NO_COLOCADAS",
                        "GUANTES_CORTE_COLOCADOS",
                        "GUANTES_CORTE_NO_COLOCADOS",
                        "MASCARA_COLOCADA",
                        "MASCARA_NO_COLOCADA",
                        "GUANTES_SOLDADURA_COLOCADOS",
                        "GUANTES_SOLDADURA_NO_COLOCADOS",
                    ],
                    "img_size": 224,
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                },
                str(ckpt_path),
            )
            clf = PPEClassifier(weights_path=ckpt_path, device="cpu")
            res = clf.predict_region(_make_sharp_frame(120, 120), region="CABEZA", activity="corte")
            self.assertIn(res.predicted_class, ("GAFAS_COLOCADAS", "GAFAS_NO_COLOCADAS"))
            self.assertGreaterEqual(res.confidence, 0.0)
            self.assertLessEqual(res.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
