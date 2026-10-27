"""時代合成 v0: テンプレート合成方式（資料19枚目）

将来、画像生成方式に差し替えても API の形は変えない
"""
import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app import config as cfg
from app.core.regions import FACE_OVAL, LEFT_IRIS_CENTER, MOUTH_CENTER_PAIR, RIGHT_IRIS_CENTER
from app.schemas import Template

log = logging.getLogger("cosme-ai")


@dataclass
class TemplateAsset:
    info: Template  # /v1/templates で返す内容
    background: np.ndarray  # RGB
    overlay: np.ndarray | None  # RGBA（髪・帽子・襟。なければ None）
    slot: np.ndarray  # (3, 2) 左目・右目・口の位置（background 上の px）
    skin_mean: np.ndarray  # テンプレートの元の顔の肌の Lab 平均
    skin_std: np.ndarray  # 同じく標準偏差


def _read_rgb(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    # cv2.imread は日本語パスを読めないので、バイト列から読む
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    code = cv2.COLOR_BGRA2RGBA if img.ndim == 3 and img.shape[2] == 4 else cv2.COLOR_BGR2RGB
    return cv2.cvtColor(img, code)


def _skin_region(shape: tuple[int, int], slot: np.ndarray) -> np.ndarray:
    """slot（両目と口）から、テンプレートの元の顔の頬あたりを楕円で取る"""
    eye_dist = np.linalg.norm(slot[1] - slot[0])
    center = slot.mean(axis=0)
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, (int(center[0]), int(center[1])),
                (int(eye_dist * 0.7), int(eye_dist * 0.8)), 0, 0, 360, 255, -1)
    return mask


def _lab_stats(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab[mask > 0]
    return pixels.mean(axis=0), pixels.std(axis=0) + 1e-6


def load_template(folder: Path) -> TemplateAsset:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    background = _read_rgb(folder / "background.jpg")
    overlay_path = folder / "overlay.png"
    overlay = _read_rgb(overlay_path, cv2.IMREAD_UNCHANGED) if overlay_path.is_file() else None
    if overlay is not None and (overlay.ndim != 3 or overlay.shape[2] != 4):
        raise ValueError("overlay.png must have an alpha channel")
    if overlay is not None and overlay.shape[:2] != background.shape[:2]:
        raise ValueError("overlay.png must be the same size as background.jpg")

    s = meta["slot"]
    slot = np.array([s["left_eye"], s["right_eye"], s["mouth"]], dtype=np.float32)
    skin_mean, skin_std = _lab_stats(background, _skin_region(background.shape[:2], slot))
    info = Template(
        id=meta["id"],
        era=meta["era"],
        label=meta["label"],
        thumbnail=f"/v1/templates/{meta['id']}/thumb.jpg",
    )
    return TemplateAsset(info, background, overlay, slot, skin_mean, skin_std)


def load_templates(template_dir: Path) -> dict[str, TemplateAsset]:
    """templates/<id>/ を起動時に読む。壊れたテンプレートは飛ばしてログに残す"""
    templates: dict[str, TemplateAsset] = {}
    if not template_dir.is_dir():
        return templates
    for meta_path in sorted(template_dir.glob("*/meta.json")):
        try:
            t = load_template(meta_path.parent)
        except Exception:
            log.exception("skip broken template: %s", meta_path.parent)
            continue
        templates[t.info.id] = t
    return templates


def _match_color(face: np.ndarray, mask: np.ndarray, t: TemplateAsset) -> np.ndarray:
    """顔領域の Lab 平均・標準偏差を、テンプレートの肌に SKIN_COLOR_BLEND だけ寄せる"""
    lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab[mask > 0.5]
    if len(pixels) == 0:
        return face
    mean, std = pixels.mean(axis=0), pixels.std(axis=0) + 1e-6
    ratio = np.clip(t.skin_std / std, *cfg.SKIN_STD_RATIO)
    matched = (lab - mean) * ratio + t.skin_mean
    lab = lab + cfg.SKIN_COLOR_BLEND * (matched - lab)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def composite(rgb: np.ndarray, landmarks: np.ndarray, t: TemplateAsset) -> np.ndarray:
    """利用者の顔をテンプレートに合成した RGB 画像を返す

    landmarks: (478, 3) の正規化座標
    """
    h, w = rgb.shape[:2]
    pts = (landmarks[:, :2] * (w, h)).astype(np.float32)
    bh, bw = t.background.shape[:2]

    # 位置合わせ: 左目・右目・口の中心を slot に合わせる相似変換
    src = np.array([pts[LEFT_IRIS_CENTER], pts[RIGHT_IRIS_CENTER],
                    pts[list(MOUTH_CENTER_PAIR)].mean(axis=0)], dtype=np.float32)
    m, _ = cv2.estimateAffinePartial2D(src, t.slot)
    if m is None:
        raise ValueError("failed to estimate face alignment")
    face = cv2.warpAffine(rgb, m, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # マスク: FACE_OVAL を同じ変換で移し、収縮＋ぼかしで境界をなじませる
    oval = pts[FACE_OVAL] @ m[:, :2].T + m[:, 2]
    mask = np.zeros((bh, bw), np.uint8)
    cv2.fillPoly(mask, [oval.astype(np.int32)], 255)
    k = cfg.MASK_ERODE_PX * 2 + 1
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    alpha = cv2.GaussianBlur(mask, (cfg.MASK_BLUR_PX, cfg.MASK_BLUR_PX), 0).astype(np.float32) / 255

    # 色合わせ → マスクでアルファ合成
    face = _match_color(face, alpha, t)
    a = alpha[..., None]
    out = t.background.astype(np.float32) * (1 - a) + face.astype(np.float32) * a

    # 前景: overlay.png があれば最後に重ねる（髪・帽子で境目を隠す）
    if t.overlay is not None:
        oa = t.overlay[:, :, 3:4].astype(np.float32) / 255
        out = out * (1 - oa) + t.overlay[:, :, :3].astype(np.float32) * oa

    return np.clip(out, 0, 255).astype(np.uint8)


def to_data_url(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY])
    if not ok:
        raise ValueError("failed to encode JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
