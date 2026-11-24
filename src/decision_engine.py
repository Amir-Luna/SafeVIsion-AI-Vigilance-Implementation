"""Motor de decisión de cumplimiento de EPP para SafeVision AI.

Combina las detecciones de regiones corporales de YOLO (PERSONA, CABEZA, MANO),
detecciones auxiliares de EPP (GAFAS_SEGURIDAD, MASCARA_SOLDADURA, GUANTES_CORTE,
GUANTES_SOLDADURA), predicciones de la CNN propia, suavizado temporal de
probabilidades, controles de calidad de imagen e histéresis de estado para emitir:
- CUMPLE
- NO_CUMPLE
- NO_VERIFICABLE
- Esperando operario (cuando no se detecta ninguna persona ni región válida)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from src.classifier import ClassificationResult, ProbabilitySmoother
from src.detector import Detection, compute_iou, is_inside_expanded_box
from src.utils import (
    QualityReport,
    VALID_ACTIVITIES,
    evaluate_crop_quality,
    extract_region_crop,
    get_valid_cnn_classes_for,
)

STATUS_CUMPLE = "CUMPLE"
STATUS_NO_CUMPLE = "NO_CUMPLE"
STATUS_NO_VERIFICABLE = "NO_VERIFICABLE"
STATUS_ESPERANDO_OPERARIO = "Esperando operario"

# Mapeo de clases auxiliares YOLO esperadas según actividad
EXPECTED_HEAD_YOLO_CLASS: Dict[str, str] = {
    "corte": "GAFAS_SEGURIDAD",
    "soldadura": "MASCARA_SOLDADURA",
}
EXPECTED_GLOVE_YOLO_CLASS: Dict[str, str] = {
    "corte": "GUANTES_CORTE",
    "soldadura": "GUANTES_SOLDADURA",
}


class RegionClassifierProtocol(Protocol):
    """Interfaz mínima requerida para clasificar recortes (facilita pruebas unitarias)."""

    def predict_region(
        self,
        crop_bgr: np.ndarray,
        region: str,
        activity: str,
    ) -> ClassificationResult: ...


@dataclass(frozen=True)
class RegionVerdict:
    """Resultado de verificación para una región corporal individual."""

    region_label: str  # Ej: "CABEZA", "MANO_1", "MANO_2"
    region_type: str   # "CABEZA" o "MANO"
    status: str        # CUMPLE | NO_CUMPLE | NO_VERIFICABLE
    bbox: Optional[Tuple[int, int, int, int]]
    expanded_bbox: Optional[Tuple[int, int, int, int]]
    predicted_class: Optional[str]
    confidence: Optional[float]
    reason: str
    quality: Optional[QualityReport] = None


@dataclass(frozen=True)
class EvaluationResult:
    """Resultado integral de cumplimiento de EPP para un fotograma."""

    activity: str
    overall_status: str  # CUMPLE | NO_CUMPLE | NO_VERIFICABLE | Esperando operario
    region_verdicts: Tuple[RegionVerdict, ...]
    auxiliary_epp_detections: Tuple[Detection, ...]
    summary_messages: Tuple[str, ...]
    alerts: Tuple[str, ...] = field(default_factory=tuple)


class DecisionEngine:
    """Evalúa cumplimiento de EPP fusionando evidencia YOLO + CNN con regla Fail-Safe."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        yolo_cfg = config.get("yolo", {})
        cnn_cfg = config.get("cnn", {})
        crops_cfg = config.get("crops", {})
        quality_cfg = config.get("quality", {})
        stab_cfg = config.get("stabilization", {})
        fusion_cfg = config.get("fusion", {})

        self.cnn_conf_threshold = float(cnn_cfg.get("conf_threshold", 0.65))
        self.head_margin_ratio = float(crops_cfg.get("head_margin_ratio", 0.20))
        self.hand_margin_ratio = float(crops_cfg.get("hand_margin_ratio", 0.25))
        self.target_size = int(crops_cfg.get("target_size", cnn_cfg.get("img_size", 224)))
        self.pad_color = tuple(int(c) for c in crops_cfg.get("pad_color", [114, 114, 114]))

        self.min_region_width = int(quality_cfg.get("min_region_width", 32))
        self.min_region_height = int(quality_cfg.get("min_region_height", 32))
        self.min_blur_variance = float(quality_cfg.get("min_blur_variance", 35.0))
        self.min_brightness = float(quality_cfg.get("min_brightness", 25.0))
        self.max_brightness = float(quality_cfg.get("max_brightness", 240.0))

        # Parámetros de fusión YOLO + CNN y regla Fail-Safe
        self.person_margin_ratio = float(yolo_cfg.get("person_margin_ratio", 0.15))
        self.glove_yolo_conf_threshold = float(fusion_cfg.get("glove_yolo_conf_threshold", 0.35))
        self.head_epp_yolo_conf_threshold = float(fusion_cfg.get("head_epp_yolo_conf_threshold", 0.35))
        self.head_epp_margin_ratio = float(fusion_cfg.get("head_epp_margin_ratio", 0.25))
        self.cnn_head_strict_conf_threshold = float(
            fusion_cfg.get("cnn_head_strict_conf_threshold", 0.85)
        )
        self.fail_safe_missing_hands = bool(fusion_cfg.get("fail_safe_missing_hands", True))

        # Parámetros de estabilización temporal centralizados en config.yaml
        self.stabilization_enabled = bool(stab_cfg.get("enabled", True))
        self.window_size = max(1, int(stab_cfg.get("window_size", 10)))
        self.prob_ema_alpha = float(stab_cfg.get("prob_ema_alpha", 0.25))
        self.min_consistent_frames = max(1, int(stab_cfg.get("min_consistent_frames", 5)))

        self._prob_smoother = ProbabilitySmoother(
            window_size=self.window_size,
            ema_alpha=self.prob_ema_alpha,
        )
        self._committed_status: Optional[str] = None
        self._candidate_status: Optional[str] = None
        self._candidate_count: int = 0

    def reset(self) -> None:
        """Reinicia todos los buffers de suavizado temporal e histéresis de estado."""
        self._prob_smoother.reset()
        self._committed_status = None
        self._candidate_status = None
        self._candidate_count = 0

    def _stabilize_status(self, raw_status: str) -> str:
        """Aplica histéresis temporal: exige N frames consecutivos consistentes para cambiar estado."""
        if not self.stabilization_enabled or self.min_consistent_frames <= 1:
            self._committed_status = raw_status
            self._candidate_status = None
            self._candidate_count = 0
            return raw_status

        if self._committed_status is None:
            # Primer fotograma activo: fijar estado inicial inmediatamente (compatible con imagen única)
            self._committed_status = raw_status
            self._candidate_status = None
            self._candidate_count = 0
            return raw_status

        if raw_status == self._committed_status:
            self._candidate_status = None
            self._candidate_count = 0
            return self._committed_status

        if raw_status == self._candidate_status:
            self._candidate_count += 1
        else:
            self._candidate_status = raw_status
            self._candidate_count = 1

        if self._candidate_count >= self.min_consistent_frames:
            self._committed_status = raw_status
            self._candidate_status = None
            self._candidate_count = 0

        return self._committed_status

    def _has_yolo_head_epp(
        self,
        head_det: Detection,
        epp_detections: List[Detection],
        activity: str,
    ) -> bool:
        """Verifica si YOLO detectó GAFAS_SEGURIDAD o MASCARA_SOLDADURA sobre/cerca de CABEZA."""
        expected_cls = EXPECTED_HEAD_YOLO_CLASS[activity]
        for det in epp_detections:
            if det.class_name.upper() != expected_cls:
                continue
            if det.confidence < self.head_epp_yolo_conf_threshold:
                continue
            if (
                compute_iou(det.bbox, head_det.bbox) > 0.0
                or is_inside_expanded_box(
                    det.bbox, head_det.bbox, margin_ratio=self.head_epp_margin_ratio
                )
            ):
                return True
        return False

    def _get_valid_yolo_gloves(
        self,
        epp_detections: List[Detection],
        persons: List[Detection],
        activity: str,
    ) -> List[Detection]:
        """Devuelve detecciones YOLO de guantes acordes a la actividad con conf >= umbral y dentro de PERSONA."""
        expected_cls = EXPECTED_GLOVE_YOLO_CLASS[activity]
        valid: List[Detection] = []
        for det in epp_detections:
            if det.class_name.upper() != expected_cls:
                continue
            if det.confidence < self.glove_yolo_conf_threshold:
                continue
            if persons and not any(
                is_inside_expanded_box(det.bbox, p.bbox, margin_ratio=self.person_margin_ratio)
                for p in persons
            ):
                continue
            valid.append(det)
        valid.sort(key=lambda d: d.confidence, reverse=True)
        return valid[:2]

    def evaluate(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
        classifier: RegionClassifierProtocol,
        activity: str,
    ) -> EvaluationResult:
        """Evalúa el cumplimiento de EPP fusionando YOLO + CNN y aplicando Fail-Safe de manos."""
        activity_norm = activity.strip().lower()
        if activity_norm not in VALID_ACTIVITIES:
            raise ValueError(
                f"Actividad desconocida '{activity}'. Actividades válidas: {VALID_ACTIVITIES}."
            )

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("El fotograma proporcionado al motor de decisión está vacío.")

        persons = [d for d in detections if d.class_name.upper() == "PERSONA"]
        heads = [d for d in detections if d.class_name.upper() == "CABEZA"]
        hands = [d for d in detections if d.class_name.upper() == "MANO"]
        epp_aux = [
            d
            for d in detections
            if d.class_name.upper()
            in {"GAFAS_SEGURIDAD", "GUANTES_CORTE", "MASCARA_SOLDADURA", "GUANTES_SOLDADURA"}
        ]

        valid_gloves = self._get_valid_yolo_gloves(epp_aux, persons, activity_norm)

        # Si no hay ninguna persona ni región corporal ni guante válido -> "Esperando operario"
        if not persons and not heads and not hands and not valid_gloves:
            self.reset()
            return EvaluationResult(
                activity=activity_norm,
                overall_status=STATUS_ESPERANDO_OPERARIO,
                region_verdicts=(),
                auxiliary_epp_detections=tuple(epp_aux),
                summary_messages=("Esperando operario en el encuadre...",),
                alerts=(),
            )

        selected_head = max(heads, key=lambda d: d.confidence) if heads else None
        selected_hands = sorted(hands, key=lambda d: d.confidence, reverse=True)[:2]
        selected_hands = sorted(selected_hands, key=lambda d: d.center[0])

        verdicts: List[RegionVerdict] = []

        # 1. Evaluación de CABEZA (fusión YOLO GAFAS/MASCARA + CNN)
        if selected_head is None:
            self._prob_smoother.clear_key("CABEZA")
            verdicts.append(
                RegionVerdict(
                    region_label="CABEZA",
                    region_type="CABEZA",
                    status=STATUS_NO_VERIFICABLE,
                    bbox=None,
                    expanded_bbox=None,
                    predicted_class=None,
                    confidence=None,
                    reason="Región CABEZA no visible o no detectada en el encuadre.",
                )
            )
        else:
            has_yolo_head_epp = self._has_yolo_head_epp(selected_head, epp_aux, activity_norm)
            verdicts.append(
                self._evaluate_head_region(
                    frame_bgr=frame_bgr,
                    detection=selected_head,
                    classifier=classifier,
                    activity=activity_norm,
                    has_yolo_head_epp=has_yolo_head_epp,
                )
            )

        # 2. Evaluación de MANOS (fusión YOLO GUANTES_* + CNN + Regla Fail-Safe)
        correct_glove_cls, _ = get_valid_cnn_classes_for("MANO", activity_norm)

        if (
            self.fail_safe_missing_hands
            and (persons or heads)
            and not selected_hands
            and not valid_gloves
        ):
            # Regla Fail-Safe: hay operario visible pero ni MANO ni guantes válidos -> NO_CUMPLE
            self._prob_smoother.clear_key("MANO_1")
            self._prob_smoother.clear_key("MANO_2")
            for idx in range(2):
                verdicts.append(
                    RegionVerdict(
                        region_label=f"MANO_{idx + 1}",
                        region_type="MANO",
                        status=STATUS_NO_CUMPLE,
                        bbox=None,
                        expanded_bbox=None,
                        predicted_class=None,
                        confidence=None,
                        reason="Guantes no detectados / verifique manos",
                    )
                )
        else:
            # Construir hasta 2 entradas de mano combinando MANO detectadas y guantes YOLO válidos
            # Emparejar cada MANO con su guante YOLO más cercano/solapado si existe
            remaining_gloves = list(valid_gloves)
            hand_slots: List[Tuple[Optional[Detection], Optional[Detection]]] = []

            for hand_det in selected_hands:
                matched_glove: Optional[Detection] = None
                best_score = -1.0
                best_idx = -1
                for g_idx, glove_det in enumerate(remaining_gloves):
                    iou = compute_iou(hand_det.bbox, glove_det.bbox)
                    near = is_inside_expanded_box(glove_det.bbox, hand_det.bbox, margin_ratio=0.50)
                    if iou > 0.0 or near:
                        score = iou + (0.1 if near else 0.0)
                        if score > best_score:
                            best_score = score
                            best_idx = g_idx
                if best_idx >= 0:
                    matched_glove = remaining_gloves.pop(best_idx)
                hand_slots.append((hand_det, matched_glove))

            # Si quedan guantes YOLO válidos dentro de la persona sin caja MANO asociada,
            # usarlos para completar hasta 2 manos (ej. cuando el guante oculta la clase MANO)
            for glove_det in sorted(remaining_gloves, key=lambda d: d.center[0]):
                if len(hand_slots) < 2:
                    hand_slots.append((None, glove_det))

            # Si hay 1+ guante válido dentro de la persona y alguna MANO no se emparejó por distancia,
            # pero el total de guantes válidos cubre las manos detectadas, ya fue resuelto arriba.
            for idx in range(2):
                hand_label = f"MANO_{idx + 1}"
                if idx < len(hand_slots):
                    hand_det, glove_det = hand_slots[idx]
                    if glove_det is not None:
                        # Fusión YOLO: guante válido detectado >= umbral dentro de la persona
                        # valida la mano como COLOCADOS aunque la CNN dude.
                        self._prob_smoother.update(
                            hand_label,
                            {
                                correct_glove_cls: float(glove_det.confidence),
                                get_valid_cnn_classes_for("MANO", activity_norm)[1]: max(
                                    0.0, 1.0 - float(glove_det.confidence)
                                ),
                            },
                        )
                        verdicts.append(
                            RegionVerdict(
                                region_label=hand_label,
                                region_type="MANO",
                                status=STATUS_CUMPLE,
                                bbox=hand_det.bbox if hand_det is not None else glove_det.bbox,
                                expanded_bbox=None,
                                predicted_class=correct_glove_cls,
                                confidence=float(glove_det.confidence),
                                reason=(
                                    f"Validado por detección YOLO ({glove_det.class_name} "
                                    f"{glove_det.confidence:.2f})."
                                ),
                            )
                        )
                    elif hand_det is not None:
                        verdicts.append(
                            self._evaluate_hand_region_with_cnn(
                                frame_bgr=frame_bgr,
                                detection=hand_det,
                                region_label=hand_label,
                                classifier=classifier,
                                activity=activity_norm,
                            )
                        )
                else:
                    self._prob_smoother.clear_key(hand_label)
                    verdicts.append(
                        RegionVerdict(
                            region_label=hand_label,
                            region_type="MANO",
                            status=STATUS_NO_VERIFICABLE,
                            bbox=None,
                            expanded_bbox=None,
                            predicted_class=None,
                            confidence=None,
                            reason=f"{hand_label} fuera del encuadre o no visible (se requieren ambas manos).",
                        )
                    )

        # 3. Determinar estado global bruto y aplicar histéresis temporal
        statuses = [v.status for v in verdicts]
        if STATUS_NO_CUMPLE in statuses:
            raw_status = STATUS_NO_CUMPLE
        elif all(s == STATUS_CUMPLE for s in statuses):
            raw_status = STATUS_CUMPLE
        else:
            raw_status = STATUS_NO_VERIFICABLE

        overall_status = self._stabilize_status(raw_status)

        summary_messages: List[str] = []
        alerts: List[str] = []
        for v in verdicts:
            conf_txt = f" ({v.confidence * 100:.1f}%)" if v.confidence is not None else ""
            cls_txt = f" [{v.predicted_class}{conf_txt}]" if v.predicted_class else ""
            msg = f"{v.region_label}: {v.status}{cls_txt} - {v.reason}"
            summary_messages.append(msg)
            if v.status == STATUS_NO_CUMPLE:
                alerts.append(f"ALERTA INCUMPLIMIENTO [{activity_norm.upper()}]: {msg}")
            elif v.status == STATUS_NO_VERIFICABLE:
                alerts.append(f"AVISO NO VERIFICABLE [{activity_norm.upper()}]: {msg}")

        return EvaluationResult(
            activity=activity_norm,
            overall_status=overall_status,
            region_verdicts=tuple(verdicts),
            auxiliary_epp_detections=tuple(epp_aux),
            summary_messages=tuple(summary_messages),
            alerts=tuple(alerts),
        )

    def _run_cnn_on_region(
        self,
        frame_bgr: np.ndarray,
        detection: Detection,
        region_label: str,
        region_type: str,
        margin_ratio: float,
        classifier: RegionClassifierProtocol,
        activity: str,
    ) -> Tuple[Optional[QualityReport], Optional[Tuple[int, int, int, int]], Optional[str], Optional[float], bool]:
        """Recorta, valida calidad y ejecuta clasificación CNN con suavizado temporal."""
        raw_crop, expanded_bbox = extract_region_crop(
            frame_bgr,
            bbox=detection.bbox,
            margin_ratio=margin_ratio,
            target_size=None,
        )

        quality = evaluate_crop_quality(
            raw_crop,
            min_width=self.min_region_width,
            min_height=self.min_region_height,
            min_blur_variance=self.min_blur_variance,
            min_brightness=self.min_brightness,
            max_brightness=self.max_brightness,
        )

        if not quality.is_acceptable:
            self._prob_smoother.clear_key(region_label)
            return quality, expanded_bbox, None, None, False

        prediction = classifier.predict_region(
            crop_bgr=raw_crop,
            region=region_type,
            activity=activity,
        )
        correct_cls, incorrect_cls = get_valid_cnn_classes_for(region_type, activity)

        if (
            prediction.valid_pair_probs
            and correct_cls in prediction.valid_pair_probs
            and incorrect_cls in prediction.valid_pair_probs
        ):
            raw_pair = {
                correct_cls: float(prediction.valid_pair_probs[correct_cls]),
                incorrect_cls: float(prediction.valid_pair_probs[incorrect_cls]),
            }
        else:
            p_pred = max(0.0, min(1.0, float(prediction.confidence)))
            if prediction.is_correct_usage and prediction.predicted_class == correct_cls:
                raw_pair = {correct_cls: p_pred, incorrect_cls: 1.0 - p_pred}
            else:
                raw_pair = {correct_cls: 1.0 - p_pred, incorrect_cls: p_pred}

        if self.stabilization_enabled:
            smoothed_pair = self._prob_smoother.update(region_label, raw_pair)
        else:
            smoothed_pair = raw_pair

        prob_ok = smoothed_pair[correct_cls]
        prob_bad = smoothed_pair[incorrect_cls]

        if prob_ok >= prob_bad:
            return quality, expanded_bbox, correct_cls, prob_ok, True
        return quality, expanded_bbox, incorrect_cls, prob_bad, False

    def _evaluate_head_region(
        self,
        frame_bgr: np.ndarray,
        detection: Detection,
        classifier: RegionClassifierProtocol,
        activity: str,
        has_yolo_head_epp: bool,
    ) -> RegionVerdict:
        """Evalúa CABEZA aplicando umbral estricto de CNN (>=0.85) si YOLO no detecta gafas/máscara."""
        quality, expanded_bbox, predicted_class, confidence, is_correct = self._run_cnn_on_region(
            frame_bgr=frame_bgr,
            detection=detection,
            region_label="CABEZA",
            region_type="CABEZA",
            margin_ratio=self.head_margin_ratio,
            classifier=classifier,
            activity=activity,
        )

        if quality is None or not quality.is_acceptable or predicted_class is None or confidence is None:
            return RegionVerdict(
                region_label="CABEZA",
                region_type="CABEZA",
                status=STATUS_NO_VERIFICABLE,
                bbox=detection.bbox,
                expanded_bbox=expanded_bbox,
                predicted_class=None,
                confidence=None,
                reason="; ".join(quality.reasons) if quality else "Calidad insuficiente.",
                quality=quality,
            )

        correct_cls, incorrect_cls = get_valid_cnn_classes_for("CABEZA", activity)

        # Regla de fusión para CABEZA: sin detección YOLO de gafas/máscara, exigir CNN >= 0.85
        if not has_yolo_head_epp:
            if is_correct and predicted_class == correct_cls:
                if confidence >= self.cnn_head_strict_conf_threshold:
                    return RegionVerdict(
                        region_label="CABEZA",
                        region_type="CABEZA",
                        status=STATUS_CUMPLE,
                        bbox=detection.bbox,
                        expanded_bbox=expanded_bbox,
                        predicted_class=correct_cls,
                        confidence=confidence,
                        reason="EPP obligatorio confirmado con alta confianza CNN.",
                        quality=quality,
                    )
                return RegionVerdict(
                    region_label="CABEZA",
                    region_type="CABEZA",
                    status=STATUS_NO_CUMPLE,
                    bbox=detection.bbox,
                    expanded_bbox=expanded_bbox,
                    predicted_class=incorrect_cls,
                    confidence=confidence,
                    reason=(
                        f"Sin detección YOLO de {EXPECTED_HEAD_YOLO_CLASS[activity]} y confianza CNN "
                        f"insuficiente ({confidence:.2f} < {self.cnn_head_strict_conf_threshold:.2f})."
                    ),
                    quality=quality,
                )

            # CNN ya predijo NO_COLOCADAS sin apoyo YOLO -> NO_CUMPLE
            return RegionVerdict(
                region_label="CABEZA",
                region_type="CABEZA",
                status=STATUS_NO_CUMPLE,
                bbox=detection.bbox,
                expanded_bbox=expanded_bbox,
                predicted_class=incorrect_cls,
                confidence=confidence,
                reason="EPP obligatorio ausente, mal colocado o sostenido fuera de posición.",
                quality=quality,
            )

        # Con apoyo YOLO de GAFAS_SEGURIDAD / MASCARA_SOLDADURA -> usar umbral estándar de la CNN
        if confidence < self.cnn_conf_threshold:
            return RegionVerdict(
                region_label="CABEZA",
                region_type="CABEZA",
                status=STATUS_NO_VERIFICABLE,
                bbox=detection.bbox,
                expanded_bbox=expanded_bbox,
                predicted_class=predicted_class,
                confidence=confidence,
                reason=(
                    f"Confianza de la CNN por debajo del umbral "
                    f"({confidence:.2f} < {self.cnn_conf_threshold:.2f})."
                ),
                quality=quality,
            )

        if is_correct and predicted_class == correct_cls:
            return RegionVerdict(
                region_label="CABEZA",
                region_type="CABEZA",
                status=STATUS_CUMPLE,
                bbox=detection.bbox,
                expanded_bbox=expanded_bbox,
                predicted_class=correct_cls,
                confidence=confidence,
                reason="EPP obligatorio correctamente colocado.",
                quality=quality,
            )

        return RegionVerdict(
            region_label="CABEZA",
            region_type="CABEZA",
            status=STATUS_NO_CUMPLE,
            bbox=detection.bbox,
            expanded_bbox=expanded_bbox,
            predicted_class=incorrect_cls,
            confidence=confidence,
            reason="EPP obligatorio ausente, mal colocado o sostenido fuera de posición.",
            quality=quality,
        )

    def _evaluate_hand_region_with_cnn(
        self,
        frame_bgr: np.ndarray,
        detection: Detection,
        region_label: str,
        classifier: RegionClassifierProtocol,
        activity: str,
    ) -> RegionVerdict:
        """Evalúa una región MANO mediante la CNN cuando no fue validada por guante YOLO."""
        quality, expanded_bbox, predicted_class, confidence, is_correct = self._run_cnn_on_region(
            frame_bgr=frame_bgr,
            detection=detection,
            region_label=region_label,
            region_type="MANO",
            margin_ratio=self.hand_margin_ratio,
            classifier=classifier,
            activity=activity,
        )

        if quality is None or not quality.is_acceptable or predicted_class is None or confidence is None:
            return RegionVerdict(
                region_label=region_label,
                region_type="MANO",
                status=STATUS_NO_VERIFICABLE,
                bbox=detection.bbox,
                expanded_bbox=expanded_bbox,
                predicted_class=None,
                confidence=None,
                reason="; ".join(quality.reasons) if quality else "Calidad insuficiente.",
                quality=quality,
            )

        correct_cls, _ = get_valid_cnn_classes_for("MANO", activity)
        if confidence < self.cnn_conf_threshold:
            return RegionVerdict(
                region_label=region_label,
                region_type="MANO",
                status=STATUS_NO_VERIFICABLE,
                bbox=detection.bbox,
                expanded_bbox=expanded_bbox,
                predicted_class=predicted_class,
                confidence=confidence,
                reason=(
                    f"Confianza de la CNN por debajo del umbral "
                    f"({confidence:.2f} < {self.cnn_conf_threshold:.2f})."
                ),
                quality=quality,
            )

        if is_correct and predicted_class == correct_cls:
            return RegionVerdict(
                region_label=region_label,
                region_type="MANO",
                status=STATUS_CUMPLE,
                bbox=detection.bbox,
                expanded_bbox=expanded_bbox,
                predicted_class=predicted_class,
                confidence=confidence,
                reason="EPP obligatorio correctamente colocado.",
                quality=quality,
            )

        return RegionVerdict(
            region_label=region_label,
            region_type="MANO",
            status=STATUS_NO_CUMPLE,
            bbox=detection.bbox,
            expanded_bbox=expanded_bbox,
            predicted_class=predicted_class,
            confidence=confidence,
            reason="EPP obligatorio ausente, mal colocado o sostenido fuera de posición.",
            quality=quality,
        )
