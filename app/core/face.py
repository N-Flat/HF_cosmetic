"""MediaPipe の検出結果から1人を選び、契約の Face に変換する（資料16枚目）"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from app.config import DECIMALS, MAX_POSE_DEG, MIN_BRIGHTNESS, MIN_FACE_PX
from app.errors import ApiError
from app.schemas import Face


def _bbox(landmarks) -> tuple[float, float, float, float]:
    """landmarks の最小・最大から [x, y, w, h]（0〜1 に収める）"""
    xs = [min(max(lm.x, 0.0), 1.0) for lm in landmarks]
    ys = [min(max(lm.y, 0.0), 1.0) for lm in landmarks]
    x0, y0 = min(xs), min(ys)
    return x0, y0, max(xs) - x0, max(ys) - y0


def _pose_deg(matrix: np.ndarray) -> tuple[float, float]:
    """transform の回転部から (yaw, pitch) を度で求める"""
    pitch, yaw, _roll = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    return yaw, pitch


def _brightness(rgb: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    """顔の bbox 内の平均輝度（Lab の L, 0〜255）"""
    h, w = rgb.shape[:2]
    x, y, bw, bh = bbox
    crop = rgb[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)]
    if crop.size == 0:
        return 255.0
    return float(cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[:, :, 0].mean())


def build_face(result, rgb: np.ndarray) -> tuple[Face, list[str]]:
    warnings: list[str] = []

    # 1. 0人
    if not result.face_landmarks:
        raise ApiError("NO_FACE", 422, "face not detected")

    # 2. 1人選ぶ（後ろの見物人を拾わないため、bbox 面積が最大の顔）
    boxes = [_bbox(lms) for lms in result.face_landmarks]
    i = max(range(len(boxes)), key=lambda k: boxes[k][2] * boxes[k][3])
    if len(boxes) > 1:
        warnings.append("MULTIPLE_FACES")
    bbox = boxes[i]

    # 3. 解像度
    if bbox[2] * rgb.shape[1] < MIN_FACE_PX:
        raise ApiError("LOW_RESOLUTION", 422)

    # 4〜6. landmarks / transform / blendshapes
    landmarks = [(round(lm.x, DECIMALS), round(lm.y, DECIMALS), round(lm.z, DECIMALS))
                 for lm in result.face_landmarks[i]]
    matrix = np.asarray(result.facial_transformation_matrixes[i], dtype=float)
    transform = [round(float(v), DECIMALS) for v in matrix.flatten()]  # 行優先16個
    blendshapes = {c.category_name: round(c.score, DECIMALS) for c in result.face_blendshapes[i]}

    # 7. 姿勢（エラーにせず warnings）
    yaw, pitch = _pose_deg(matrix)
    if abs(yaw) > MAX_POSE_DEG or abs(pitch) > MAX_POSE_DEG:
        warnings.append("FACE_NOT_FRONTAL")

    # 8. 明るさ（エラーにせず warnings）
    if _brightness(rgb, bbox) < MIN_BRIGHTNESS:
        warnings.append("LOW_LIGHT")

    face = Face(
        bbox=tuple(round(v, DECIMALS) for v in bbox),
        landmarks=landmarks,
        transform=transform,
        blendshapes=blendshapes,
    )
    return face, warnings
