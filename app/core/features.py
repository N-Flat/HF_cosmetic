"""特徴検出 v0: 学習なしの画像処理でほくろ・そばかすを探す（資料18枚目）

ここでは位置・大きさ・種類までを求める。anchor・uv は geometry.find_anchor で付ける
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from app import config as cfg
from app.core.regions import EXCLUDE_PARTS, EYES, FACE_OVAL, LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER


@dataclass
class Candidate:
    xy: tuple[float, float]  # 元画像の正規化座標
    type: str  # "mole" | "freckle"
    size: float  # 等価直径 ÷ 元画像幅
    confidence: float
    diff: float  # d の最大値（並べ替え用）


def _normalize(rgb: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1. 目の位置で回転を補正し、顔幅を NORM_FACE_WIDTH にそろえた切り抜きを作る

    pts: 元画像のピクセル座標 (478, 2)
    戻り値: (切り抜き画像, 切り抜き上の点, 元画像→切り抜きの 2x3 行列)
    """
    left, right = pts[LEFT_IRIS_CENTER], pts[RIGHT_IRIS_CENTER]
    angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    face_w = np.ptp(pts[:, 0])
    scale = cfg.NORM_FACE_WIDTH / face_w
    center = pts.min(axis=0) + np.ptp(pts, axis=0) / 2

    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, scale)
    rotated = pts @ m[:, :2].T + m[:, 2]
    margin = cfg.NORM_FACE_WIDTH * cfg.NORM_MARGIN
    origin = rotated.min(axis=0) - margin
    m[:, 2] -= origin
    size = np.ceil(np.ptp(rotated, axis=0) + 2 * margin).astype(int)

    crop = cv2.warpAffine(rgb, m, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return crop, rotated - origin, m


def _skin_mask(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    """2. 顔の輪郭から目・眉・唇・鼻の穴を余白付きで除外し、少し収縮"""
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [pts[FACE_OVAL].astype(np.int32)], 255)

    for groups, margin in ((EXCLUDE_PARTS, cfg.PART_MARGIN_PX), (EYES, cfg.EYE_MARGIN_PX)):
        parts = np.zeros(shape, np.uint8)
        for idx in groups:
            cv2.fillConvexPoly(parts, cv2.convexHull(pts[idx].astype(np.int32)), 255)
        k = margin * 2 + 1
        parts = cv2.dilate(parts, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mask[parts > 0] = 0

    k = cfg.SKIN_ERODE_PX * 2 + 1
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def detect_candidates(rgb: np.ndarray, landmarks: np.ndarray) -> list[Candidate]:
    """landmarks: (478, 3) の正規化座標"""
    h, w = rgb.shape[:2]
    pts = landmarks[:, :2] * (w, h)
    crop, crop_pts, m = _normalize(rgb, pts)
    skin = _skin_mask(crop.shape[:2], crop_pts)

    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    lum, red = lab[:, :, 0], lab[:, :, 1].astype(np.int16)

    # 3. 暗い点: d = medianBlur(L, 21) - L
    diff = cv2.medianBlur(lum, cfg.MEDIAN_KSIZE).astype(np.int16) - lum.astype(np.int16)
    dark = ((diff >= cfg.DARK_DIFF) & (skin > 0)).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    m_inv = cv2.invertAffineTransform(m)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    found: list[Candidate] = []

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        # 4. 形で絞る: 面積と円形度
        if not cfg.MIN_AREA <= area <= cfg.MAX_AREA:
            continue
        x, y, bw, bh = stats[i, :4]
        x0, y0 = max(x - 4, 0), max(y - 4, 0)
        blob = (labels[y0:y + bh + 4, x0:x + bw + 4] == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = cv2.arcLength(contours[0], True)
        circularity = min(1.0, 4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 1.0
        if circularity < cfg.MIN_CIRCULARITY:
            continue

        # 5. 赤みを除外: 周囲の肌より a* が RED_DIFF 以上高い
        ring = (cv2.dilate(blob, ring_kernel) > 0) & (blob == 0)
        ring &= skin[y0:y0 + blob.shape[0], x0:x0 + blob.shape[1]] > 0
        red_patch = red[y0:y0 + blob.shape[0], x0:x0 + blob.shape[1]]
        if ring.any() and red_patch[blob > 0].mean() - red_patch[ring].mean() >= cfg.RED_DIFF:
            continue

        # 6. 分類
        d = float(diff[y0:y0 + blob.shape[0], x0:x0 + blob.shape[1]][blob > 0].max())
        diameter = math.sqrt(4 * area / math.pi)
        kind = "mole" if diameter >= cfg.MOLE_MIN_DIAMETER and d >= cfg.MOLE_MIN_DIFF else "freckle"

        # 7. 座標を元画像の正規化座標に戻す
        cx, cy = (labels[y:y + bh, x:x + bw] == i).nonzero()[::-1]
        center = np.array([cx.mean() + x, cy.mean() + y])
        ox, oy = m_inv[:, :2] @ center + m_inv[:, 2]
        scale = cfg.NORM_FACE_WIDTH / np.ptp(pts[:, 0])
        found.append(Candidate(
            xy=(float(ox / w), float(oy / h)),
            type=kind,
            size=float(diameter / scale / w),
            confidence=float(min(1.0, d / 50 * circularity)),
            diff=d,
        ))

    found.sort(key=lambda c: c.diff, reverse=True)
    return found[:cfg.MAX_FEATURES]
