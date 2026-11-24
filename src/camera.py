"""Módulo de captura de video, webcam e imágenes estáticas para SafeVision AI.

Diseñado con compatibilidad nativa para Windows (soporte CAP_DSHOW para webcams)
y soporte transparente para imágenes y archivos de video.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils import SUPPORTED_IMAGE_EXTENSIONS, safe_imread


class VideoSource:
    """Abstracción unificada para leer fotogramas desde webcam, video o imagen."""

    def __init__(
        self,
        source: str | int,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> None:
        self.raw_source = source
        self.is_image = False
        self.is_webcam = False
        self._static_image: Optional[np.ndarray] = None
        self._image_served = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_time = time.perf_counter()
        self._fps_ema = 0.0

        self._open(source, width=width, height=height, fps=fps)

    def _open(
        self,
        source: str | int,
        width: Optional[int],
        height: Optional[int],
        fps: Optional[int],
    ) -> None:
        # Detectar si es índice numérico de webcam
        cam_index: Optional[int] = None
        if isinstance(source, int):
            cam_index = source
        elif isinstance(source, str) and source.strip().isdigit():
            cam_index = int(source.strip())

        if cam_index is not None:
            self.is_webcam = True
            if platform.system().lower() == "windows":
                cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(cam_index)
            else:
                cap = cv2.VideoCapture(cam_index)

            if not cap.isOpened():
                cap.release()
                raise RuntimeError(
                    f"No se pudo abrir la webcam con índice {cam_index}. "
                    "Verifique que esté conectada y no esté siendo usada por otra aplicación."
                )

            if width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            if height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            if fps is not None:
                cap.set(cv2.CAP_PROP_FPS, float(fps))

            self._cap = cap
            return

        # Archivo local (imagen o video)
        path = Path(str(source))
        if not path.exists():
            raise FileNotFoundError(f"La fuente de entrada no existe: {path}")

        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            img = safe_imread(path)
            if img is None or img.size == 0:
                raise ValueError(f"No se pudo leer la imagen (archivo dañado o formato no soportado): {path}")
            self.is_image = True
            self._static_image = img
            return

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"No se pudo abrir el archivo de video: {path}")
        self._cap = cap

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lee el siguiente fotograma disponible."""
        if self.is_image:
            if self._image_served or self._static_image is None:
                return False, None
            self._image_served = True
            return True, self._static_image.copy()

        if self._cap is None:
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def tick_fps(self) -> float:
        """Actualiza y devuelve el conteo suavizado de FPS."""
        now = time.perf_counter()
        dt = max(1e-6, now - self._last_time)
        self._last_time = now
        inst_fps = 1.0 / dt
        if self._fps_ema <= 0.0:
            self._fps_ema = inst_fps
        else:
            self._fps_ema = 0.85 * self._fps_ema + 0.15 * inst_fps
        return self._fps_ema

    def release(self) -> None:
        """Libera la cámara o archivo de video."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
