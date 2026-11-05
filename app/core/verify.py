"""同一人物判定（簡易版）: 2枚の顔のランドマーク形状を比べる

顔認識モデル（顔の見た目の特徴量）ではなく、478点の位置関係だけを使う。
位置・大きさ・回転を取り除いて（Procrustes）残った形の差を一致度にする。
表情・強い横向き・レンズの歪みでも差が出るので、精度は限定的（性能確認用）
"""
import numpy as np

from app import config as cfg

# 表情で大きく動く点（唇・まぶた・虹彩）を除いた、骨格寄りの点
_MOVING = set(range(468, 478)) | {
    0, 11, 12, 13, 14, 15, 16, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
    185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
    7, 33, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
    249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
}
RIGID = np.array([i for i in range(468) if i not in _MOVING])


def _normalize(points: np.ndarray) -> np.ndarray:
    p = points - points.mean(axis=0)
    return p / np.linalg.norm(p)


def shape_points(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """正規化座標を、縦横の比が正しい座標に直す（z は x と同じ尺度）"""
    return landmarks[RIGID] * np.array([width, height, width], float)


def shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    """位置・大きさ・回転をそろえたあとの 1 点あたりの差（RMS、顔全体の大きさに対する割合）"""
    a, b = _normalize(a), _normalize(b)
    u, _s, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))  # 鏡映にならないようにする
    r = u @ np.diag([1, 1, d]) @ vt
    diff = a - b @ r
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()) * np.sqrt(len(a)))


def similarity(rms: float) -> float:
    return round(max(0.0, 1.0 - rms / cfg.VERIFY_RMS_ZERO) * 100, 1)
