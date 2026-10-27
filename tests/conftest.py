"""テスト共通の準備"""
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.image_io import decode
from app.main import app

# 顔が写ったテスト画像（本人の同意を取った写真か、フリー素材）
FACE_IMAGES = sorted(p for p in (Path(__file__).parent / "images").glob("*")
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

needs_face = pytest.mark.skipif(not FACE_IMAGES, reason="tests/images/ に顔写真がない")


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # with で lifespan（モデル読み込み）が動く
        yield c


def jpeg_bytes(w: int = 640, h: int = 480) -> bytes:
    """顔のない無地の画像"""
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 180, 160)).save(buf, "JPEG")
    return buf.getvalue()


def to_jpeg(rgb: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def post_image(client, data: bytes, path: str = "/v1/analyze", **form):
    return client.post(path, files={"image": ("a.jpg", data, "image/jpeg")}, data=form or None)


@pytest.fixture(scope="session")
def face_bytes() -> bytes:
    if not FACE_IMAGES:
        pytest.skip("tests/images/ に顔写真がない")
    return FACE_IMAGES[0].read_bytes()


def _darken(img: np.ndarray, p: tuple[int, int], r: int, factor: float, tint) -> np.ndarray:
    """p を中心に半径 r の範囲を factor 倍に暗くする（ふちは少しぼかす）"""
    mask = np.zeros(img.shape[:2], np.float32)
    cv2.circle(mask, p, r, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)[..., None]
    target = np.clip(img * factor * np.array(tint, np.float32), 0, 255)
    return (img * (1 - mask) + target * mask).astype(np.uint8)


# 人工の特徴: (名前, 期待する type（None = 検出されないこと）, 位置の基準になる3点, 半径, 暗さ, 色味)
SPOTS = [
    ("mole_left_cheek", "mole", (50, 187, 205), 5, 0.45, (1.0, 0.85, 0.75)),
    ("mole_right_cheek", "mole", (280, 411, 425), 5, 0.45, (1.0, 0.85, 0.75)),
    ("freckle_forehead", "freckle", (108, 151, 337), 2, 0.92, (1.0, 0.85, 0.75)),
    ("acne_chin", None, (175, 199, 200), 4, 0.95, (1.05, 0.7, 0.7)),
]


@pytest.fixture(scope="session")
def spotted_face(client, face_bytes):
    """顔写真に人工のほくろ・そばかす・ニキビを描いた画像と、その位置（px）"""
    rgb = decode(face_bytes)
    h, w = rgb.shape[:2]
    landmarks = np.array(post_image(client, face_bytes).json()["face"]["landmarks"])
    img, spots = rgb.copy(), []
    for name, kind, idx, r, factor, tint in SPOTS:
        p = landmarks[list(idx), :2].mean(axis=0) * (w, h)
        p = (int(p[0]), int(p[1]))
        img = _darken(img, p, r, factor, tint)
        spots.append((name, kind, p))
    return {"bytes": to_jpeg(img), "rgb": img, "width": w, "height": h, "spots": spots}
