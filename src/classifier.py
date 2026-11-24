"""Módulo de CNN propia en PyTorch (entrenada desde cero) para SafeVision AI.

Incluye:
- Arquitectura SafeVisionCNN diseñada desde cero (sin pesos preentrenados ni transfer learning).
- Inicialización aleatoria controlada (Kaiming / Xavier).
- ProbabilitySmoother: suavizado temporal (ventana deslizante + EMA) sobre vectores de
  probabilidades softmax para estabilizar el porcentaje de confianza en tiempo real.
- Envoltorio PPEClassifier para inferencia en tiempo real por región corporal y actividad.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.utils import (
    CNN_DEFAULT_CLASSES,
    get_valid_cnn_classes_for,
    letterbox_resize,
    resolve_project_path,
    select_device,
    setup_logging,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - permite importar tipos sin torch instalado
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class ClassificationResult:
    """Resultado de clasificación de la CNN para una región corporal."""

    predicted_class: str
    confidence: float
    is_correct_usage: bool
    valid_pair_probs: Dict[str, float]
    global_top_class: str
    global_top_confidence: float


class ProbabilitySmoother:
    """Suavizador temporal de vectores de probabilidad softmax por clave de región.

    Combina una ventana deslizante (`deque` de los últimos `window_size` frames)
    con un suavizado exponencial (EMA con `ema_alpha`) para evitar saltos bruscos
    en el porcentaje mostrado y en las decisiones derivadas.
    """

    def __init__(self, window_size: int = 10, ema_alpha: float = 0.25) -> None:
        self.window_size = max(1, int(window_size))
        self.ema_alpha = max(0.05, min(1.0, float(ema_alpha)))
        self._windows: Dict[str, Deque[Dict[str, float]]] = {}
        self._ema_state: Dict[str, Dict[str, float]] = {}

    def update(self, key: str, probs: Dict[str, float]) -> Dict[str, float]:
        """Actualiza el historial de `key` con nuevas probabilidades y devuelve el vector suavizado."""
        if not probs:
            return {}

        if key not in self._windows:
            self._windows[key] = deque(maxlen=self.window_size)

        # Si el conjunto de clases cambió (ej. cambio de actividad), reiniciar buffer de la clave
        prev_ema = self._ema_state.get(key)
        if prev_ema is not None and set(prev_ema.keys()) != set(probs.keys()):
            self._windows[key].clear()
            prev_ema = None

        self._windows[key].append({k: float(v) for k, v in probs.items()})

        # Promedio en ventana deslizante
        keys = list(probs.keys())
        win_len = len(self._windows[key])
        avg_probs = {
            k: sum(sample[k] for sample in self._windows[key]) / float(win_len)
            for k in keys
        }

        # EMA sobre el promedio de ventana para transiciones suaves sin picos
        if prev_ema is None:
            smoothed = avg_probs
        else:
            a = self.ema_alpha
            smoothed = {
                k: a * avg_probs[k] + (1.0 - a) * prev_ema[k]
                for k in keys
            }

        # Renormalizar para que las probabilidades sumen 1.0
        total = sum(smoothed.values())
        if total > 0.0:
            smoothed = {k: v / total for k, v in smoothed.items()}

        self._ema_state[key] = smoothed
        return dict(smoothed)

    def clear_key(self, key: str) -> None:
        """Limpia el estado acumulado de una región específica."""
        self._windows.pop(key, None)
        self._ema_state.pop(key, None)

    def reset(self) -> None:
        """Limpia el estado de todas las regiones."""
        self._windows.clear()
        self._ema_state.clear()


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch no está instalado. Instálelo siguiendo las instrucciones del README.md."
        )


if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """Bloque convolucional reutilizable: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

        def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
            super().__init__()
            layers: List[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.block = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)


    class SafeVisionCNN(nn.Module):
        """Red Neuronal Convolucional propia diseñada desde cero para 224x224 RGB.

        Estructura:
        - 5 bloques convolucionales con BatchNorm, ReLU y MaxPool (224 -> 112 -> 56 -> 28 -> 14 -> 7).
        - AdaptiveAvgPool2d(2, 2) para robustez espacial manteniendo eficiencia.
        - Clasificador totalmente conectado con Dropout y 8 salidas.
        """

        def __init__(
            self,
            num_classes: int = 8,
            dropout: float = 0.4,
            base_channels: int = 32,
        ) -> None:
            super().__init__()
            self.num_classes = int(num_classes)
            self.dropout_rate = float(dropout)
            self.base_channels = int(base_channels)

            c1 = self.base_channels          # 32
            c2 = self.base_channels * 2      # 64
            c3 = self.base_channels * 4      # 128
            c4 = self.base_channels * 8      # 256
            c5 = self.base_channels * 8      # 256

            self.features = nn.Sequential(
                ConvBlock(3, c1, pool=True),
                ConvBlock(c1, c2, pool=True),
                ConvBlock(c2, c3, pool=True),
                ConvBlock(c3, c4, pool=True),
                ConvBlock(c4, c5, pool=True),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(c5 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate * 0.75),
                nn.Linear(256, self.num_classes),
            )

            self._initialize_weights()

        def _initialize_weights(self) -> None:
            """Inicialización aleatoria apropiada (sin pesos preentrenados)."""
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            return self.classifier(x)

        def get_arch_config(self) -> Dict[str, Any]:
            """Devuelve la configuración arquitectónica para validar compatibilidad de pesos."""
            return {
                "architecture": "SafeVisionCNN",
                "num_classes": self.num_classes,
                "dropout": self.dropout_rate,
                "base_channels": self.base_channels,
            }

else:

    class SafeVisionCNN:  # type: ignore[no-redef]
        """Marcador cuando PyTorch no está disponible en el entorno actual."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


class PPEClassifier:
    """Clasificador de uso correcto/incorrecto del EPP usando SafeVisionCNN."""

    def __init__(
        self,
        weights_path: str | Path,
        device: Optional[str] = None,
        pad_color: Sequence[int] = (114, 114, 114),
        smooth_window_size: int = 10,
        smooth_ema_alpha: float = 0.25,
    ) -> None:
        _require_torch()
        self.logger = setup_logging()
        self.weights_path = resolve_project_path(weights_path)
        self.device = torch.device(device if device is not None else select_device(prefer_cuda=True))
        self.pad_color = tuple(int(c) for c in pad_color[:3])
        self.prob_smoother = ProbabilitySmoother(
            window_size=smooth_window_size,
            ema_alpha=smooth_ema_alpha,
        )

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de pesos de la CNN en: {self.weights_path}. "
                "Entrene el modelo primero ejecutando scripts/train_cnn.py."
            )

        try:
            checkpoint = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                f"Error al cargar el checkpoint de la CNN ({self.weights_path}): {exc}"
            ) from exc

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                f"El archivo {self.weights_path} no tiene el formato de checkpoint válido de SafeVision AI "
                "(falta 'model_state_dict')."
            )

        arch_cfg = checkpoint.get("arch_config", {})
        if arch_cfg.get("architecture") not in (None, "SafeVisionCNN"):
            raise ValueError(
                f"Pesos incompatibles: arquitectura esperada 'SafeVisionCNN', encontrada '{arch_cfg.get('architecture')}'."
            )

        self.class_names: List[str] = list(checkpoint.get("class_names", CNN_DEFAULT_CLASSES))
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }
        self.img_size: int = int(checkpoint.get("img_size", 224))
        self.mean = np.array(
            checkpoint.get("normalize_mean", [0.485, 0.456, 0.406]), dtype=np.float32
        ).reshape(3, 1, 1)
        self.std = np.array(
            checkpoint.get("normalize_std", [0.229, 0.224, 0.225]), dtype=np.float32
        ).reshape(3, 1, 1)

        self.model = SafeVisionCNN(
            num_classes=int(arch_cfg.get("num_classes", len(self.class_names))),
            dropout=float(arch_cfg.get("dropout", 0.4)),
            base_channels=int(arch_cfg.get("base_channels", 32)),
        )

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Los pesos de la CNN en {self.weights_path} son incompatibles con SafeVisionCNN: {exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        self.logger.info(
            "CNN propia cargada correctamente desde %s (dispositivo=%s, img_size=%d).",
            self.weights_path,
            self.device,
            self.img_size,
        )

    def reset_smoothing(self) -> None:
        """Limpia los buffers de suavizado temporal de probabilidades."""
        self.prob_smoother.reset()

    def preprocess(self, crop_bgr: np.ndarray) -> "torch.Tensor":
        """Preprocesa un recorte BGR aplicando letterbox sin deformación y normalización RGB."""
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("El recorte enviado a la CNN está vacío.")

        resized = letterbox_resize(crop_bgr, target_size=self.img_size, pad_color=self.pad_color)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        normalized = (chw - self.mean) / self.std
        tensor = torch.from_numpy(normalized).unsqueeze(0)
        return tensor

    def predict_region(
        self,
        crop_bgr: np.ndarray,
        region: str,
        activity: str,
        smooth_key: Optional[str] = None,
    ) -> ClassificationResult:
        """Clasifica un recorte de región corporal interpretando las clases válidas para la actividad.

        Si se indica `smooth_key` (por ejemplo 'CABEZA', 'MANO_1', 'MANO_2'), aplica suavizado
        temporal sobre el vector softmax del par válido para estabilizar la confianza.
        """
        correct_class, incorrect_class = get_valid_cnn_classes_for(region, activity)
        if correct_class not in self.class_to_idx or incorrect_class not in self.class_to_idx:
            raise ValueError(
                f"Las clases '{correct_class}' / '{incorrect_class}' no existen en el modelo cargado: "
                f"{self.class_names}"
            )

        tensor = self.preprocess(crop_bgr)

        try:
            with torch.inference_mode():
                logits = self.model(tensor.to(self.device))[0]
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device.type != "cpu":
                self.logger.warning(
                    "Memoria GPU agotada en la CNN. Cambiando automáticamente a CPU."
                )
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.model.to(self.device)
                with torch.inference_mode():
                    logits = self.model(tensor.to(self.device))[0]
            else:
                raise

        global_probs = F.softmax(logits, dim=0)
        global_idx = int(torch.argmax(global_probs).item())
        global_top_class = self.class_names[global_idx]
        global_top_conf = float(global_probs[global_idx].item())

        # Softmax acotado al par válido (región + actividad) para decisión precisa
        idx_ok = self.class_to_idx[correct_class]
        idx_bad = self.class_to_idx[incorrect_class]
        pair_logits = torch.stack([logits[idx_ok], logits[idx_bad]])
        pair_probs = F.softmax(pair_logits, dim=0)

        raw_pair = {
            correct_class: float(pair_probs[0].item()),
            incorrect_class: float(pair_probs[1].item()),
        }
        if smooth_key is not None:
            smoothed_pair = self.prob_smoother.update(smooth_key, raw_pair)
            prob_ok = smoothed_pair[correct_class]
            prob_bad = smoothed_pair[incorrect_class]
        else:
            prob_ok = raw_pair[correct_class]
            prob_bad = raw_pair[incorrect_class]

        if prob_ok >= prob_bad:
            predicted_class = correct_class
            confidence = prob_ok
            is_correct = True
        else:
            predicted_class = incorrect_class
            confidence = prob_bad
            is_correct = False

        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            is_correct_usage=is_correct,
            valid_pair_probs={
                correct_class: prob_ok,
                incorrect_class: prob_bad,
            },
            global_top_class=global_top_class,
            global_top_confidence=global_top_conf,
        )
