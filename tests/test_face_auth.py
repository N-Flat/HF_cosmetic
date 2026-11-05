"""顔認証 API（/v1/face/*）のテスト。登録データは一時ディレクトリに保存する"""
import cv2
import numpy as np
import pytest

from app.core.face_store import FaceStore
from app.core.image_io import decode
from tests.conftest import jpeg_bytes, to_jpeg
from tests.test_contract import assert_error


@pytest.fixture
def store(client, tmp_path):
    """本番の data/face_db.json を汚さないよう、テスト中だけ差し替える"""
    if client.app.state.face_auth is None:
        pytest.skip("顔認証モデルがない（python tools/download_face_models.py）")
    original = client.app.state.face_store
    client.app.state.face_store = FaceStore(tmp_path / "face_db.json")
    yield client.app.state.face_store
    client.app.state.face_store = original


def _variants(face_bytes: bytes) -> list[bytes]:
    """同じ人の写真3枚の代わり: 元・少し暗い・少し回転"""
    rgb = decode(face_bytes)
    h, w = rgb.shape[:2]
    dark = cv2.convertScaleAbs(rgb, alpha=0.8, beta=0)
    rot = cv2.warpAffine(rgb, cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0), (w, h),
                         borderMode=cv2.BORDER_REPLICATE)
    return [to_jpeg(rgb), to_jpeg(dark), to_jpeg(rot)]


def enroll(client, user_id: str, images: list[bytes], consent: str = "true"):
    files = [("images", (f"{i}.jpg", b, "image/jpeg")) for i, b in enumerate(images)]
    return client.post("/v1/face/enroll", data={"user_id": user_id, "consent": consent}, files=files)


def identify(client, image: bytes):
    return client.post("/v1/face/identify", files={"image": ("a.jpg", image, "image/jpeg")})


def verify(client, user_id: str, image: bytes):
    return client.post("/v1/face/verify", data={"user_id": user_id},
                       files={"image": ("a.jpg", image, "image/jpeg")})


def test_health_lists_face_models(client, store):
    models = client.get("/health").json()["models"]
    assert models["face_detector"] == "yunet 2023mar"
    assert models["face_recognizer"] == "sface 2021dec"


def test_enroll_identify_verify_delete(client, store, face_bytes):
    res = enroll(client, "alice", _variants(face_bytes))
    assert res.status_code == 200, res.json()
    data = res.json()
    assert data["user_id"] == "alice" and data["images"] == 3 and data["replaced"] is False
    assert len(data["faces"]) == 3

    # 再登録は置き換え
    assert enroll(client, "alice", _variants(face_bytes)).json()["replaced"] is True

    # 左右反転（別の写真の代わり）でも同じ人として見つかる
    probe = to_jpeg(decode(face_bytes)[:, ::-1])
    res = identify(client, probe)
    assert res.status_code == 200, res.json()
    data = res.json()
    assert data["matched"] is True and data["user_id"] == "alice"
    assert data["score"] >= data["threshold"]
    assert data["candidates"][0]["user_id"] == "alice"

    res = verify(client, "alice", probe)
    assert res.status_code == 200, res.json()
    assert res.json()["matched"] is True

    assert client.delete("/v1/face/alice").status_code == 204
    assert_error(verify(client, "alice", probe), 404, "USER_NOT_FOUND")
    assert identify(client, probe).json()["matched"] is False


def test_identify_rejects_other_person(client, store, face_bytes):
    """登録されているのが別人の特徴量なら matched=False"""
    rng = np.random.default_rng(0)
    other = rng.normal(size=(3, 128)).astype(np.float32)
    store.put("bob", other / np.linalg.norm(other, axis=1, keepdims=True))
    data = identify(client, face_bytes).json()
    assert data["matched"] is False and data["user_id"] is None
    assert data["candidates"][0]["user_id"] == "bob"
    assert verify(client, "bob", face_bytes).json()["matched"] is False


def test_store_persists(tmp_path):
    path = tmp_path / "db.json"
    s = FaceStore(path)
    v = np.eye(3, 128, dtype=np.float32)
    s.put("u1", v)
    loaded = FaceStore(path).get("u1")
    assert loaded is not None and np.allclose(loaded, v)


def test_enroll_errors(client, store, face_bytes):
    imgs = _variants(face_bytes)
    assert_error(enroll(client, "alice", imgs, consent="false"), 400, "CONSENT_REQUIRED")
    assert_error(enroll(client, "alice", imgs[:2]), 400, "INVALID_REQUEST")
    assert_error(enroll(client, "bad id!", imgs), 400, "INVALID_REQUEST")
    res = enroll(client, "alice", [imgs[0], jpeg_bytes(), imgs[2]])
    assert_error(res, 422, "NO_FACE")
    assert res.json()["error"]["message"].startswith("2枚目")
    assert store.get("alice") is None


def test_enroll_rejects_multiple_faces(client, store, face_bytes):
    rgb = decode(face_bytes)
    two = to_jpeg(np.hstack([rgb, rgb]))
    assert_error(enroll(client, "alice", [two, two, two]), 422, "MULTIPLE_FACES")


def test_identify_no_face(client, store):
    assert_error(identify(client, jpeg_bytes()), 422, "NO_FACE")


def test_verify_unknown_user(client, store, face_bytes):
    assert_error(verify(client, "nobody", face_bytes), 404, "USER_NOT_FOUND")
    assert_error(client.delete("/v1/face/nobody"), 404, "USER_NOT_FOUND")


def test_demo_page(client):
    res = client.get("/demo/face")
    assert res.status_code == 200 and "text/html" in res.headers["content-type"]
