"""Pruebas de las correcciones del diagnóstico de detección (v1.4.0).

Cubre:
- Problema 1/2: umbrales por clase, área mínima menor, margen de vínculo por clase y
  registro de motivos de descarte.
- Problema 3: tracker que no pierde manos en movimiento ni por un frame perdido.
- Problema 4: overlay de depuración y resumen por clase.
- Plan 4: script de análisis/balanceo del dataset YOLO.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analizar_dataset_yolo import analyze_split, build_balanced_list, build_warnings
from src.camera import draw_debug_overlay, summarize_debug_counts
from src.detector import (
    REJECT_AREA,
    REJECT_CONF,
    REJECT_LINK,
    Detection,
    DetectionTracker,
    filter_false_positives,
)

FRAME_720P = (720, 1280)


class TestFiltroPorClase(unittest.TestCase):
    def setUp(self) -> None:
        self.person = Detection(0, "PERSONA", 0.90, (400, 100, 800, 700))  # 400 px de ancho

    def test_mano_con_brazo_extendido_se_conserva_con_margen_por_clase(self) -> None:
        # Centro de la mano en x=920: fuera del margen 0.15 (860) pero dentro de 0.40 (960)
        hand = Detection(2, "MANO", 0.55, (880, 300, 960, 380))
        rejected: list = []
        kept_old = filter_false_positives(
            [self.person, hand], FRAME_720P, min_area_ratio=0.002, person_margin_ratio=0.15,
            rejections=rejected,
        )
        self.assertNotIn("MANO", [d.class_name for d in kept_old])
        self.assertEqual(rejected[0].reason, REJECT_LINK)

        kept_new = filter_false_positives(
            [self.person, hand], FRAME_720P, min_area_ratio=0.002, person_margin_ratio=0.15,
            class_link_margins={"MANO": 0.40},
        )
        self.assertIn("MANO", [d.class_name for d in kept_new])

    def test_gafas_pequenas_pasan_con_area_0002_y_se_registra_motivo_con_0008(self) -> None:
        glasses = Detection(3, "GAFAS_SEGURIDAD", 0.60, (560, 160, 640, 200))  # 80x40 = 3200 px
        rejected: list = []
        kept = filter_false_positives(
            [self.person, glasses], FRAME_720P, min_area_ratio=0.008, rejections=rejected
        )
        self.assertNotIn("GAFAS_SEGURIDAD", [d.class_name for d in kept])
        self.assertEqual(rejected[0].reason, REJECT_AREA)

        kept = filter_false_positives([self.person, glasses], FRAME_720P, min_area_ratio=0.002)
        self.assertIn("GAFAS_SEGURIDAD", [d.class_name for d in kept])

    def test_umbral_por_clase_mano_010_y_gafas_025(self) -> None:
        dets = [
            self.person,
            Detection(2, "MANO", 0.12, (500, 400, 580, 480)),
            Detection(3, "GAFAS_SEGURIDAD", 0.27, (560, 160, 660, 210)),
            Detection(1, "CABEZA", 0.30, (540, 120, 680, 260)),  # usa el umbral general 0.35
        ]
        rejected: list = []
        kept = filter_false_positives(
            dets, FRAME_720P, conf_threshold=0.35, min_area_ratio=0.002,
            class_conf_thresholds={"MANO": 0.10, "GAFAS_SEGURIDAD": 0.25}, rejections=rejected,
        )
        names = {d.class_name for d in kept}
        self.assertEqual(names, {"PERSONA", "MANO", "GAFAS_SEGURIDAD"})
        self.assertEqual([(r.detection.class_name, r.reason) for r in rejected], [("CABEZA", REJECT_CONF)])


class TestTrackerMovimiento(unittest.TestCase):
    @staticmethod
    def _hand(x: int) -> list:
        return [Detection(2, "MANO", 0.6, (x, 300, x + 80, 380))]

    def test_mano_rapida_se_confirma_emparejando_contra_caja_cruda(self) -> None:
        # 30 px/frame: la IoU contra la caja suavizada (EMA) cae a ~0.23 en el 3er frame,
        # pero contra la última caja cruda es ~0.45.
        tracker = DetectionTracker(iou_threshold=0.30, ema_alpha=0.35, min_hits=3, max_missed=3)
        tracker.update(self._hand(100), FRAME_720P)
        tracker.update(self._hand(130), FRAME_720P)
        out = tracker.update(self._hand(160), FRAME_720P)
        self.assertEqual(len(out), 1)

    def test_un_frame_perdido_no_reinicia_el_contador(self) -> None:
        tracker = DetectionTracker(min_hits=2, max_missed_unconfirmed=1)
        self.assertEqual(tracker.update(self._hand(200), FRAME_720P), [])
        self.assertEqual(tracker.update([], FRAME_720P), [])  # parpadeo por desenfoque
        self.assertEqual(len(tracker.update(self._hand(205), FRAME_720P)), 1)

    def test_override_por_clase(self) -> None:
        tracker = DetectionTracker(min_hits=3, class_overrides={"MANO": {"min_hits": 1}})
        frame = self._hand(200) + [Detection(1, "CABEZA", 0.9, (500, 100, 600, 200))]
        out = tracker.update(frame, FRAME_720P)
        self.assertEqual([d.class_name for d in out], ["MANO"])

    def test_respaldo_por_distancia_de_centros(self) -> None:
        # Salto de 70 px con caja de 80 px: IoU ~0.06, pero centro a ~0.6 diagonales
        tracker = DetectionTracker(min_hits=2, iou_threshold=0.3, center_match_ratio=0.7)
        tracker.update(self._hand(100), FRAME_720P)
        self.assertEqual(len(tracker.update(self._hand(170), FRAME_720P)), 1)
        tracker_sin = DetectionTracker(min_hits=2, iou_threshold=0.3, center_match_ratio=0.0)
        tracker_sin.update(self._hand(100), FRAME_720P)
        self.assertEqual(tracker_sin.update(self._hand(170), FRAME_720P), [])


class TestOverlayDepuracion(unittest.TestCase):
    def test_overlay_y_resumen(self) -> None:
        person = Detection(0, "PERSONA", 0.9, (400, 100, 800, 700))
        hand_low = Detection(2, "MANO", 0.05, (500, 400, 580, 480))
        hand_ok = Detection(2, "MANO", 0.5, (600, 400, 680, 480))
        rejected: list = []
        filtered = filter_false_positives(
            [person, hand_low, hand_ok], FRAME_720P, min_area_ratio=0.002,
            class_conf_thresholds={"MANO": 0.10}, rejections=rejected,
        )
        summary = summarize_debug_counts([person, hand_low, hand_ok], filtered, [person])
        self.assertEqual(summary["MANO"]["raw"], 2)
        self.assertAlmostEqual(summary["MANO"]["max_conf"], 0.5)
        self.assertEqual(summary["MANO"]["filt"], 1)
        self.assertEqual(summary["MANO"]["stab"], 0)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = draw_debug_overlay(frame.copy(), [person, hand_low, hand_ok], filtered, [person], rejected)
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)


class TestAnalisisDataset(unittest.TestCase):
    def test_conteo_y_balanceo(self) -> None:
        names = {0: "CABEZA", 4: "MANO", 6: "PERSONA"}
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp) / "images" / "train"
            lbl_dir = Path(tmp) / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            for i in range(8):
                (img_dir / f"img{i}.jpg").write_bytes(b"x")
                lines = ["6 0.5 0.5 0.6 0.9", "0 0.5 0.2 0.1 0.1"]
                if i == 0:
                    lines.append("4 0.3 0.6 0.03 0.03")  # mano pequeña, una sola imagen
                (lbl_dir / f"img{i}.txt").write_text("\n".join(lines))

            stats = analyze_split(img_dir, names, small_area_ratio=0.002)
            self.assertEqual(stats["por_clase"]["PERSONA"]["instancias"], 8)
            self.assertEqual(stats["por_clase"]["MANO"]["instancias"], 1)
            self.assertEqual(stats["por_clase"]["MANO"]["cajas_pequenas_pct"], 100.0)

            warns = build_warnings(stats, imbalance_ratio=3.0, min_samples=5, focus_classes=["MANO"])
            self.assertTrue(any("MANO: desbalance" in w for w in warns))

            lines, factors = build_balanced_list(stats["_image_classes"], names, ["MANO"], max_factor=4)
            self.assertEqual(factors, {"MANO": 4})
            self.assertEqual(len(lines), 8 + 3)


if __name__ == "__main__":
    unittest.main()
