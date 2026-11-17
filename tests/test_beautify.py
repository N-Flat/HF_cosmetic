"""美肌加工（/v1/beautify）のテスト

コンセプト「輪郭・細かな特徴・パーツの位置はそのまま」を数値で確かめる
- 加工後の画像でもランドマークがほとんど動かない
- 人工のほくろが加工後も同じ位置に検出される
- 顔の周り（ROI）の外は変わらない
"""
import base64

import numpy as np
import pytest

from app.core.beautify import STYLES, apply, build_context
from app.core.image_io import decode
from tests.conftest import jpeg_bytes, post_image
from tests.test_contract import assert_error


def beautify(client, data: bytes, **form):
    return post_image(client, data, "/v1/beautify", **form)


def decode_data_url(url: str) -> np.ndarray:
    assert url.startswith("data:image/jpeg;base64,")
    return decode(base64.b64decode(url.split(",", 1)[1]))


def landmarks(client, data: bytes) -> np.ndarray:
    res = post_image(client, data)
    assert res.status_code == 200, res.json()
    return np.array(res.json()["face"]["landmarks"])


def to_bytes(rgb: np.ndarray) -> bytes:
    from tests.conftest import to_jpeg
    return to_jpeg(rgb)


def test_styles(client):
    data = client.get("/v1/beautify/styles").json()
    assert [s["id"] for s in data["styles"]] == list(STYLES)
    assert all(s["label"] and s["description"] for s in data["styles"])
    assert 0 <= data["default_strength"] <= 1


def test_beautify_all_styles(client, face_bytes):
    res = beautify(client, face_bytes)
    assert res.status_code == 200, res.json()
    data = res.json()
    assert [r["style"] for r in data["results"]] == list(STYLES)
    for r in data["results"]:
        assert decode_data_url(r["image"]).shape[:2] == (data["height"], data["width"])
    assert set(data["timings_ms"]) >= {"landmark", "mask", "beautify", "total"}


def test_beautify_selected_styles(client, face_bytes):
    data = beautify(client, face_bytes, styles="smooth, glow,smooth", strength="0.3").json()
    assert [r["style"] for r in data["results"]] == ["smooth", "glow"]
    assert data["strength"] == 0.3


@pytest.mark.parametrize("style", list(STYLES))
def test_geometry_preserved(client, face_bytes, style):
    """最大の強さでも、ランドマークのずれは画像の 0.5% 未満（顔を変形していない）"""
    before = landmarks(client, face_bytes)
    out = decode_data_url(beautify(client, face_bytes, styles=style, strength="1").json()["results"][0]["image"])
    after = landmarks(client, to_bytes(out))
    assert np.abs(after[:, :2] - before[:, :2]).max() < 0.005


@pytest.mark.parametrize("style", ["natural", "smooth", "tone"])
def test_moles_preserved(client, spotted_face, style):
    """人工のほくろが、加工後も同じ位置にほくろとして見つかる"""
    data = beautify(client, spotted_face["bytes"], styles=style, strength="1").json()
    assert data["protected_features"] >= 2
    out = decode_data_url(data["results"][0]["image"])
    found = post_image(client, to_bytes(out)).json()["features"]
    w, h = spotted_face["width"], spotted_face["height"]
    for name, kind, (px, py) in spotted_face["spots"]:
        if kind != "mole":
            continue
        assert any(f["type"] == "mole" and abs(f["xy"][0] * w - px) < 6 and abs(f["xy"][1] * h - py) < 6
                   for f in found), f"{name} が加工後に消えた"


def test_core_changes_only_face(client, face_bytes):
    """顔の周り（ROI）の外は1画素も変えない（glow 以外）。強さ 0 なら元のまま"""
    rgb = decode(face_bytes)
    ctx = build_context(rgb, landmarks(client, face_bytes))
    x0, y0, x1, y1 = ctx.roi
    outside = np.ones(rgb.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    for style in STYLES:
        out = apply(rgb, ctx, style, 1.0)
        assert out.shape == rgb.shape and out.dtype == np.uint8
        if style != "glow":
            assert np.array_equal(out[outside], rgb[outside]), style
        assert np.abs(out.astype(int) - rgb).mean() > 0.05, f"{style} で何も変わっていない"
        assert np.abs(apply(rgb, ctx, style, 0.0).astype(int) - rgb).max() <= 2, style


def test_beautify_errors(client, face_bytes):
    assert_error(beautify(client, face_bytes, styles="anime"), 400, "INVALID_REQUEST")
    assert_error(beautify(client, jpeg_bytes()), 422, "NO_FACE")
    assert_error(beautify(client, b"not an image"), 400, "INVALID_IMAGE")


def test_demo_page(client):
    res = client.get("/demo/beautify")
    assert res.status_code == 200 and "text/html" in res.headers["content-type"]
