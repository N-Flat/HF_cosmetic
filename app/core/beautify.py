"""美肌加工 v0: 顔の輪郭・パーツの位置・ほくろ等の特徴はそのままに、軽く自然にきれいに見せる

方針
- 画像を変形（ワープ）しない。画素の色・明るさだけを変えるので、輪郭やパーツの位置は動かない
- ほくろ・そばかす（features.detect_candidates で検出）の周りは加工しない
- 加工は MediaPipe のランドマークから作ったマスク（肌・目・眉・唇）の中だけ。境目はぼかす
- 大きさに関わる値は、顔幅 600px のときの値で書き、実際の顔幅に合わせて拡大縮小する

方式（STYLES）
- smooth : 肌なめらか（周波数分離。中くらいのムラだけ消し、細かい肌の質感は残す）
- tone   : 色ムラ補正（赤み・黄み・くすみの斑を周りの肌の色に寄せる）
- bright : 明るさ・透明感（肌の明るさを持ち上げ、黄みを少し抑える）
- glow   : ソフトグロウ（光がにじむような柔らかさ。目元・口元はにじませない）
- clarity: 目元・口元くっきり（目・眉・唇だけ少しシャープに、唇の血色を少し上げる）
- natural: おまかせ（上の組み合わせを控えめに）
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from app import config as cfg
from app.core.features import detect_candidates
from app.core.regions import EXCLUDE_PARTS, EYES, FACE_OVAL

LIPS, BROWS, NOSTRILS = EXCLUDE_PARTS[2], EXCLUDE_PARTS[:2], EXCLUDE_PARTS[3]

STYLES: dict[str, tuple[str, str]] = {
    "natural": ("おまかせ", "なめらか・色ムラ補正・明るさ・くっきりを控えめに組み合わせる"),
    "smooth": ("肌なめらか", "肌のムラや小さな凹凸をならす。肌の細かい質感は残す"),
    "tone": ("色ムラ補正", "赤み・くすみの斑を周りの肌の色に寄せて均一にする"),
    "bright": ("明るさ・透明感", "肌の明るさを持ち上げ、黄みを少し抑える"),
    "glow": ("ソフトグロウ", "光がにじむような柔らかい印象にする"),
    "clarity": ("目元・口元くっきり", "目・眉・唇を少しシャープにし、唇の血色を少し上げる"),
}


def _odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def _feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(mask, (0, 0), max(sigma, 0.5))


def _hulls(shape: tuple[int, int], pts: np.ndarray, groups: list[list[int]], grow: float) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for idx in groups:
        cv2.fillConvexPoly(m, cv2.convexHull(pts[idx].astype(np.int32)), 255)
    if grow >= 1:
        k = _odd(grow * 2 + 1)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m.astype(np.float32) / 255


@dataclass
class Context:
    """1枚の写真について、方式によらず共通のもの（マスクは最初に1回だけ作る）"""
    roi: tuple[int, int, int, int]  # 顔の周りの切り抜き範囲 (x0, y0, x1, y1)
    s: float  # 顔幅 / 600
    face: np.ndarray  # 顔の輪郭の内側（境目ぼかし）
    skin: np.ndarray  # 肌（目・眉・唇・鼻の穴・ほくろを除く）
    parts: np.ndarray  # 目・眉・唇
    lips: np.ndarray
    protect: np.ndarray  # ほくろ・そばかす（加工しない）
    features: int = 0  # 守ったほくろ・そばかすの数
    cache: dict = field(default_factory=dict)


def build_context(rgb: np.ndarray, landmarks: np.ndarray) -> Context:
    """landmarks: (478, 3) の正規化座標"""
    h, w = rgb.shape[:2]
    pts_full = landmarks[:, :2] * (w, h)
    face_w = float(np.ptp(pts_full[:, 0]))
    s = face_w / cfg.NORM_FACE_WIDTH

    pad = face_w * 0.3
    lo = np.maximum(pts_full.min(axis=0) - pad, 0).astype(int)
    hi = np.minimum(pts_full.max(axis=0) + pad, (w, h)).astype(int)
    x0, y0, x1, y1 = int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])
    shape = (y1 - y0, x1 - x0)
    pts = pts_full - (x0, y0)

    oval = np.zeros(shape, np.uint8)
    cv2.fillPoly(oval, [pts[FACE_OVAL].astype(np.int32)], 255)
    # 輪郭のすぐ外（髪・背景）に色がにじまないよう、少し内側に縮めてからぼかす
    k = _odd(10 * s)
    oval = cv2.erode(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32) / 255
    face = _feather(oval, 8 * s)

    eyes = _hulls(shape, pts, EYES, 16 * s)
    others = _hulls(shape, pts, [*BROWS, LIPS, NOSTRILS], 6 * s)
    skin = oval * (1 - np.maximum(eyes, others))

    # ほくろ・そばかす: 検出した位置を円で守る（大きさの 1.6 倍＋余白）
    protect = np.zeros(shape, np.float32)
    candidates = detect_candidates(rgb, landmarks)
    for c in candidates:
        cx, cy = c.xy[0] * w - x0, c.xy[1] * h - y0
        r = c.size * w * 0.8 + 3 * s + 2
        cv2.circle(protect, (int(round(cx)), int(round(cy))), int(round(r)), 1.0, -1, cv2.LINE_AA)
    protect = np.clip(_feather(protect, 1.5 * s) * 1.5, 0, 1)
    skin = _feather(skin, 6 * s) * (1 - protect)

    parts = _feather(np.maximum(_hulls(shape, pts, EYES, 3 * s), _hulls(shape, pts, [*BROWS, LIPS], 2 * s)), 3 * s)
    lips = _feather(_hulls(shape, pts, [LIPS], 0), 2 * s)
    return Context((x0, y0, x1, y1), s, face, skin, parts, lips, protect, len(candidates))


def _mix(a: np.ndarray, b: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = alpha[..., None] if alpha.ndim == 2 else alpha
    return a * (1 - alpha) + b * alpha


_LAB_SCALE = np.array([255 / 100, 1, 1], np.float32)
_LAB_SHIFT = np.array([0, 128, 128], np.float32)


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    """小数のまま Lab にする（uint8 の往復で誤差がたまらないように）。値の範囲は uint8 の Lab と同じ 0〜255"""
    lab = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    return lab * _LAB_SCALE + _LAB_SHIFT


def _from_lab(lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(((lab - _LAB_SHIFT) / _LAB_SCALE).astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(rgb, 0, 1) * 255


def _masked_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """mask の内側だけで平均する（外側の髪・背景の色を混ぜない）。
    大きくぼかすだけなので 1/4 に縮小して計算し、元の大きさに戻す（速さのため）"""
    h, w = img.shape[:2]
    f = 4 if sigma >= 8 else 1
    size = (max(1, w // f), max(1, h // f))
    small_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    m = small_mask[..., None] if img.ndim == 3 else small_mask
    num = cv2.GaussianBlur(small_img * m, (0, 0), sigma / f)
    den = cv2.GaussianBlur(small_mask, (0, 0), sigma / f)
    den = den[..., None] if img.ndim == 3 else den
    return cv2.resize(num / np.maximum(den, 1e-3), (w, h), interpolation=cv2.INTER_LINEAR)


# --- 方式ごとの処理。crop は ROI の float32 RGB（0〜255）。strength は 0〜1 ---

def _guided(img: np.ndarray, r: int, eps: float) -> np.ndarray:
    """自分自身をガイドにしたガイデッドフィルタ（チャンネルごと）。
    明暗差が小さいムラはならし、輪郭・鼻筋などの差が大きいエッジは残す"""
    k = (2 * r + 1, 2 * r + 1)
    mean = cv2.boxFilter(img, -1, k)
    var = cv2.boxFilter(img * img, -1, k) - mean * mean
    a = var / (var + eps)
    b = mean * (1 - a)
    return cv2.boxFilter(a, -1, k) * img + cv2.boxFilter(b, -1, k)


def smooth(crop: np.ndarray, ctx: Context, strength: float) -> np.ndarray:
    """周波数分離: 低〜中周波（ムラ・小さな凹凸）はガイデッドフィルタでならし、
    高周波（毛穴・産毛などの細かい質感）は元の画像から戻す。のっぺりした顔にしない"""
    base = _guided(crop, max(3, int(round(12 * ctx.s))), 12.0 ** 2)
    fine = crop - cv2.GaussianBlur(crop, (0, 0), max(0.8, 1.2 * ctx.s))
    base_fine = base - cv2.GaussianBlur(base, (0, 0), max(0.8, 1.2 * ctx.s))
    out = base - base_fine + fine * 0.7
    return _mix(crop, out, ctx.skin * strength)


def tone(crop: np.ndarray, ctx: Context, strength: float) -> np.ndarray:
    """a*（赤み）・b*（黄み）・L（くすみ）を、周りの肌の平均に寄せる。肌全体の色味は変えない"""
    lab = _to_lab(crop)
    region = np.clip(ctx.skin * 2, 0, 1)
    local = _masked_blur(lab, region, 25 * ctx.s)
    k = np.array([0.35, 0.8, 0.6], np.float32) * strength  # L は控えめ（陰影を残す）
    out = lab - (lab - local) * k
    return _mix(crop, _from_lab(out), ctx.skin)


def bright(crop: np.ndarray, ctx: Context, strength: float) -> np.ndarray:
    """肌の明るさをガンマで持ち上げ（暗い所ほど上がる）、黄みを少し抑える。順番が入れ替わらないので陰影は残る"""
    lab = _to_lab(crop)
    gamma = 1 - 0.22 * strength
    lab[..., 0] = 255 * (lab[..., 0] / 255) ** gamma
    lab[..., 2] = 128 + (lab[..., 2] - 128) * (1 - 0.12 * strength)
    # 目・唇も含めた顔全体に広くかける（肌だけだと目の周りが浮く）。唇の色は変えない
    mask = ctx.face * (1 - ctx.lips)
    return _mix(crop, _from_lab(lab), mask)


def glow(img: np.ndarray, ctx: Context, strength: float, full: bool = True) -> np.ndarray:
    """ぼかした画像をスクリーン合成して光をにじませる。目・眉・唇はにじませない"""
    blur = cv2.GaussianBlur(img, (0, 0), 12 * ctx.s)
    screen = 255 - (255 - img) * (255 - blur) / 255
    soft = 0.6 * screen + 0.4 * blur  # 少し柔らかく
    k = 0.32 * strength
    if full:  # 画像全体（ROI の外も）
        alpha = np.full(img.shape[:2], k, np.float32)
        x0, y0, x1, y1 = ctx.roi
        alpha[y0:y1, x0:x1] *= 1 - 0.8 * ctx.parts
        return _mix(img, soft, alpha)
    return _mix(img, soft, k * (1 - 0.8 * ctx.parts))


def clarity(crop: np.ndarray, ctx: Context, strength: float) -> np.ndarray:
    """目・眉・唇にアンシャープマスク。唇は彩度を少し上げる"""
    blur = cv2.GaussianBlur(crop, (0, 0), max(1.0, 2.0 * ctx.s))
    sharp = crop + (crop - blur) * (1.2 * strength)
    out = _mix(crop, sharp, ctx.parts)
    lab = _to_lab(out)
    lab[..., 1:] = 128 + (lab[..., 1:] - 128) * (1 + 0.18 * strength)
    return _mix(out, _from_lab(lab), ctx.lips)


def natural(crop: np.ndarray, ctx: Context, strength: float) -> np.ndarray:
    out = tone(crop, ctx, 0.5 * strength)
    out = smooth(out, ctx, 0.7 * strength)
    out = bright(out, ctx, 0.4 * strength)
    return clarity(out, ctx, 0.5 * strength)


_LOCAL = {"natural": natural, "smooth": smooth, "tone": tone, "bright": bright, "clarity": clarity}


def apply(rgb: np.ndarray, ctx: Context, style: str, strength: float) -> np.ndarray:
    """rgb: 元画像（uint8 RGB）。同じ大きさの加工後の画像を返す"""
    if strength <= 0:
        return rgb.copy()
    img = rgb.astype(np.float32)
    if style == "glow":
        out = glow(img, ctx, strength)
    else:
        x0, y0, x1, y1 = ctx.roi
        out = img.copy()
        out[y0:y1, x0:x1] = _LOCAL[style](img[y0:y1, x0:x1], ctx, strength)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
