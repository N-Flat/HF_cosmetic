"""仮のテンプレート素材を templates/ に作る（デザイン担当の素材が届くまでの代わり）

使い方:
    python tools/make_dummy_templates.py

作るもの（資料19枚目の形式）:
    templates/<id>/meta.json, background.jpg (1080x1350), overlay.png (任意), thumb.jpg (400x500)
本物の素材が届いたら、同じフォルダ構成で上書きすればよい（このスクリプトは不要になる）
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "templates"
W, H = 1080, 1350
SLOT = {"left_eye": [430, 540], "right_eye": [650, 540], "mouth": [540, 760]}

# id, era, label, 背景の上下の色, 服の色, 肌の色（RGB）, 前景
DUMMIES = [
    ("past_city", "past", "100年前・都市", (150, 125, 95), (90, 70, 50), (55, 45, 35), (200, 170, 140), "hat"),
    ("present_city", "present", "現在・都市", (190, 205, 220), (120, 135, 150), (40, 60, 90), (225, 185, 160), None),
    ("future_city", "future", "100年後・都市", (20, 30, 70), (60, 20, 90), (200, 210, 230), (215, 200, 220), "visor"),
]


def gradient(top, bottom) -> np.ndarray:
    t = np.linspace(0, 1, H)[:, None, None]
    return (np.array(top) * (1 - t) + np.array(bottom) * t).repeat(W, axis=1).astype(np.uint8)


def draw_person(img: np.ndarray, clothes, skin) -> None:
    """テンプレートの「元の人物」。顔の部分は合成で上書きされる

    肌は単色だと色合わせで陰影が消えるので、中心が明るいグラデーションとノイズを付ける
    """
    cv2.ellipse(img, (540, 1350), (430, 330), 0, 180, 360, clothes, -1)  # 肩・服
    person = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(person, (440, 820), (640, 1060), 255, -1)  # 首
    cv2.ellipse(person, (540, 650), (245, 310), 0, 0, 360, 255, -1)  # 顔

    yy, xx = np.mgrid[0:H, 0:W]
    shade = 1.05 - 0.25 * np.clip(np.hypot((xx - 540) / 245, (yy - 650) / 310), 0, 1.4)
    noise = np.random.default_rng(0).normal(0, 4, (H, W, 1))
    skin_img = np.clip(np.array(skin) * shade[..., None] + noise, 0, 255).astype(np.uint8)
    img[person > 0] = skin_img[person > 0]


def overlay_layer(kind: str | None) -> np.ndarray | None:
    if kind is None:
        return None
    layer = np.zeros((H, W, 4), np.uint8)
    if kind == "hat":  # 山高帽
        cv2.ellipse(layer, (540, 410), (330, 50), 0, 0, 360, (35, 28, 22, 255), -1)
        cv2.ellipse(layer, (540, 400), (225, 210), 0, 180, 360, (35, 28, 22, 255), -1)
        cv2.rectangle(layer, (318, 362), (762, 400), (90, 30, 30, 255), -1)
    elif kind == "visor":  # 光るバイザー（半透明）
        cv2.ellipse(layer, (540, 400), (265, 60), 0, 180, 360, (0, 230, 255, 255), 14)
        cv2.rectangle(layer, (290, 380), (790, 420), (0, 200, 255, 120), -1)
    return layer


def save_jpg(path: Path, rgb: np.ndarray, quality: int = 92) -> None:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    buf.tofile(str(path))


def main() -> None:
    for tid, era, label, bg_top, bg_bottom, clothes, skin, over in DUMMIES:
        folder = ROOT / tid
        folder.mkdir(parents=True, exist_ok=True)

        bg = gradient(bg_top, bg_bottom)
        draw_person(bg, clothes, skin)
        save_jpg(folder / "background.jpg", bg)

        layer = overlay_layer(over)
        overlay_path = folder / "overlay.png"
        if layer is not None:
            cv2.imencode(".png", cv2.cvtColor(layer, cv2.COLOR_RGBA2BGRA))[1].tofile(str(overlay_path))
        elif overlay_path.exists():
            overlay_path.unlink()

        thumb = bg.copy()
        if layer is not None:
            a = layer[:, :, 3:4] / 255
            thumb = (thumb * (1 - a) + layer[:, :, :3] * a).astype(np.uint8)
        save_jpg(folder / "thumb.jpg", cv2.resize(thumb, (400, 500), interpolation=cv2.INTER_AREA))

        meta = {"id": tid, "era": era, "label": label, "slot": SLOT,
                "_note": "仮の素材（tools/make_dummy_templates.py で生成）。デザイン担当の素材が届いたら差し替える"}
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
        print(f"作成: {folder}")


if __name__ == "__main__":
    main()
