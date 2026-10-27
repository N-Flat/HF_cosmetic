"""契約テスト（資料21枚目）: Next.js とつなぐ前に、形・座標・エラー・速度のズレを潰す"""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.core.geometry import restore_xy
from app.core.image_io import decode
from app.schemas import AnalyzeResponse, CompositeResponse, TemplatesResponse
from tests.conftest import FACE_IMAGES, jpeg_bytes, needs_face, post_image, to_jpeg


def assert_error(res, status: int, code: str) -> None:
    assert res.status_code == status
    err = res.json()["error"]
    assert err["code"] == code
    assert set(err) == {"code", "message", "request_id"}


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["models"]["landmarker"] == "face_landmarker float16/1"


def test_templates(client):
    res = client.get("/v1/templates")
    assert res.status_code == 200
    data = TemplatesResponse.model_validate(res.json())
    assert {t.era for t in data.templates} == {"past", "present", "future"}
    for t in data.templates:
        thumb = client.get(t.thumbnail)
        assert thumb.status_code == 200
        assert thumb.headers["content-type"] == "image/jpeg"


# --- 形・座標の範囲・anchor の検算 ---

@needs_face
@pytest.mark.parametrize("path", FACE_IMAGES, ids=lambda p: p.name)
def test_analyze_shape(client, path):
    res = post_image(client, path.read_bytes())
    assert res.status_code == 200, res.json()
    data = AnalyzeResponse.model_validate(res.json())
    assert len(data.face.landmarks) == 478
    assert len(data.face.transform) == 16
    assert len(data.face.blendshapes) == 52
    x, y, w, h = data.face.bbox
    assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1


def test_features_contract(client, spotted_face):
    data = AnalyzeResponse.model_validate(post_image(client, spotted_face["bytes"]).json())
    assert data.features, "人工のほくろがあるので 1 件以上"
    landmarks = np.array(data.face.landmarks)
    for f in data.features:
        assert 0 <= f.xy[0] <= 1 and 0 <= f.xy[1] <= 1
        assert 0 <= f.uv[0] <= 1 and 0 <= f.uv[1] <= 1
        assert abs(sum(f.anchor.bary) - 1) < 1e-3
        # anchor の検算: bary と landmarks から復元した位置と xy の差が 0.003 以内
        assert np.abs(restore_xy(f.anchor, landmarks) - np.array(f.xy)).max() < 0.003
    assert [f.id for f in data.features] == [f"f{i + 1}" for i in range(len(data.features))]


# --- 反転しない・EXIF ---

def test_not_mirrored(client, spotted_face):
    """左右反転した画像を送ると、同じ特徴の x が 1 - x に近い（差 0.02 以内）"""
    original = post_image(client, spotted_face["bytes"]).json()["features"]
    flipped = post_image(client, to_jpeg(spotted_face["rgb"][:, ::-1])).json()["features"]
    assert original
    for f in original:
        mirrored = (1 - f["xy"][0], f["xy"][1])
        assert any(abs(g["xy"][0] - mirrored[0]) < 0.02 and abs(g["xy"][1] - mirrored[1]) < 0.02
                   for g in flipped), f"{f['id']} {f['xy']} が反転画像で見つからない"


def test_exif_rotation(client, face_bytes):
    """回転情報付きの写真で、landmarks が正しい向きの顔を指す"""
    upright = post_image(client, face_bytes).json()["face"]["landmarks"]

    # 画素を 90° 回して保存し、EXIF に「表示時に戻す」向き（Orientation=6）を書く
    im = Image.open(io.BytesIO(face_bytes)).convert("RGB").rotate(90, expand=True)
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif, quality=95)

    rotated = post_image(client, buf.getvalue()).json()["face"]["landmarks"]
    for i in (1, 13, 468, 473):  # 鼻・口・両目
        assert abs(rotated[i][0] - upright[i][0]) < 0.01 and abs(rotated[i][1] - upright[i][1]) < 0.01


def test_decode_resizes_long_side():
    buf = io.BytesIO()
    Image.new("RGB", (3000, 2000)).save(buf, "JPEG")
    assert decode(buf.getvalue()).shape[:2] == (1280, 1920)


# --- 合成 ---

@pytest.mark.parametrize("template_id", ["past_city", "present_city", "future_city"])
def test_composite(client, face_bytes, template_id):
    res = post_image(client, face_bytes, "/v1/composite", template_id=template_id)
    assert res.status_code == 200, res.json()
    data = CompositeResponse.model_validate(res.json())
    assert data.image.startswith("data:image/jpeg;base64,")

    # width/height が background と一致し、画像の中身とも一致する
    background = client.app.state.templates[template_id].background
    assert (data.width, data.height) == (background.shape[1], background.shape[0])
    jpg = Image.open(io.BytesIO(base64.b64decode(data.image.split(",", 1)[1])))
    assert jpg.size == (data.width, data.height)


# --- 速度 ---

def test_speed(client, face_bytes):
    """analyze の timings_ms.total が 1000 未満（本番の数値は展示PCで計測する）"""
    post_image(client, face_bytes)  # 1回目は温まっていないので捨てる
    assert post_image(client, face_bytes).json()["timings_ms"]["total"] < 1000


# --- エラー ---

def test_error_no_face(client):
    assert_error(post_image(client, jpeg_bytes()), 422, "NO_FACE")
    assert_error(post_image(client, jpeg_bytes(), "/v1/composite", template_id="past_city"), 422, "NO_FACE")


def test_error_invalid_image(client):
    assert_error(post_image(client, b"not an image"), 400, "INVALID_IMAGE")


def test_error_missing_image(client):
    assert_error(client.post("/v1/analyze"), 400, "INVALID_REQUEST")


def test_error_too_large(client):
    assert_error(post_image(client, b"\xff" * (10 * 1024 * 1024 + 1)), 413, "IMAGE_TOO_LARGE")


def test_error_template_not_found(client):
    assert_error(post_image(client, jpeg_bytes(), "/v1/composite", template_id="nope"), 404, "TEMPLATE_NOT_FOUND")
