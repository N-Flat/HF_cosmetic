"""顔認証: OpenCV の YuNet（顔検出）+ SFace（顔認識）

1. YuNet で顔と 5 点（両目・鼻・口の両端）を検出する
   大きく写った顔は検出しにくいので、長辺 FACE_DETECT_SIDE に縮小して検出し、座標を元に戻す
2. SFace の alignCrop で 112x112 に位置合わせし、128 次元の特徴量にする
3. 特徴量は長さ 1 に正規化して返す（内積 = コサイン類似度）

顔画像そのものは保存しない。保存するのは特徴量だけ（app/core/face_store.py）
"""
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app import config as cfg
from app.errors import ApiError


@dataclass
class FaceEmbedding:
    vector: np.ndarray  # (128,) 長さ 1
    bbox: tuple[float, float, float, float]  # [x, y, w, h]（画像に対する 0〜1）
    score: float  # 検出の信頼度
    warnings: list[str] = field(default_factory=list)


class FaceAuth:
    """YuNet と SFace をまとめたもの。起動時に1回だけ作る"""

    def __init__(self, detector_path: Path, recognizer_path: Path):
        for p in (detector_path, recognizer_path):
            if not p.is_file():
                raise FileNotFoundError(p)
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), cfg.FACE_DETECT_SCORE, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
        self._lock = threading.Lock()  # setInputSize で状態が変わるので1本ずつ

    def _detect(self, bgr: np.ndarray) -> np.ndarray:
        """顔ごとに [x, y, w, h, 5点(10), score] の行（元画像の px）。0人なら空"""
        h, w = bgr.shape[:2]
        s = min(1.0, cfg.FACE_DETECT_SIDE / max(h, w))
        small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
        with self._lock:
            self._detector.setInputSize((small.shape[1], small.shape[0]))
            _n, faces = self._detector.detect(small)
        if faces is None:
            return np.empty((0, 15), np.float32)
        faces = faces.copy()
        faces[:, :14] /= s
        return faces

    def embed(self, rgb: np.ndarray, allow_multiple: bool = True) -> FaceEmbedding:
        """一番大きい顔の特徴量。allow_multiple=False なら2人以上でエラー（登録用）"""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self._detect(bgr)
        if len(faces) == 0:
            raise ApiError("NO_FACE", 422)
        warnings = []
        if len(faces) > 1:
            if not allow_multiple:
                raise ApiError("MULTIPLE_FACES", 422)
            warnings.append("MULTIPLE_FACES")
        row = faces[np.argmax(faces[:, 2] * faces[:, 3])]
        if row[2] < cfg.FACE_MIN_PX:
            raise ApiError("LOW_RESOLUTION", 422)

        with self._lock:
            aligned = self._recognizer.alignCrop(bgr, row)
            vec = self._recognizer.feature(aligned).ravel().astype(np.float32)
        vec /= np.linalg.norm(vec)

        h, w = rgb.shape[:2]
        x, y, bw, bh = (float(v) for v in row[:4])
        x0, y0 = max(x, 0.0), max(y, 0.0)
        x1, y1 = min(x + bw, w), min(y + bh, h)
        bbox = (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)
        return FaceEmbedding(vec, tuple(round(v, 4) for v in bbox), round(float(row[-1]), 4), warnings)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """どちらも長さ 1 の前提"""
    return float(np.dot(a, b))


def best_score(query: np.ndarray, enrolled: np.ndarray) -> float:
    """登録済みの複数枚（N, 128）のうち、一番近いもののコサイン類似度"""
    return float((enrolled @ query).max())
