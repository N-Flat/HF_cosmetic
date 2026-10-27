import threading
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


class Landmarker:
    """MediaPipe FaceLandmarker の薄いラッパー。起動時に1回だけ作る"""

    def __init__(self, path: Path):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._lock = threading.Lock()

    def detect(self, rgb: np.ndarray):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._lock:  # not thread-safe
            return self._lm.detect(img)

    def close(self) -> None:
        self._lm.close()
