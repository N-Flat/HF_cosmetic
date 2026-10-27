"""build_face の単体テスト。MediaPipe の結果を偽物で作り、顔写真なしで変換ルールを確認する"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from app.core.face import build_face
from app.errors import ApiError


def fake_face(x0, y0, size, yaw_deg=0.0):
    """x0, y0 から size 四方に 478 点を並べた偽の顔"""
    xs = np.linspace(x0, x0 + size, 478)
    ys = np.linspace(y0, y0 + size, 478)
    landmarks = [SimpleNamespace(x=float(x), y=float(y), z=-0.01) for x, y in zip(xs, ys)]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
    blendshapes = [SimpleNamespace(category_name=f"bs{k}", score=0.123456) for k in range(52)]
    return landmarks, matrix, blendshapes


def fake_result(*faces):
    return SimpleNamespace(
        face_landmarks=[f[0] for f in faces],
        facial_transformation_matrixes=[f[1] for f in faces],
        face_blendshapes=[f[2] for f in faces],
    )


BRIGHT = np.full((1000, 1000, 3), 200, dtype=np.uint8)


def test_converts_to_contract_shape():
    face, warnings = build_face(fake_result(fake_face(0.2, 0.3, 0.4)), BRIGHT)
    assert len(face.landmarks) == 478
    assert len(face.transform) == 16
    assert face.bbox == pytest.approx((0.2, 0.3, 0.4, 0.4))
    assert face.blendshapes["bs0"] == 0.1235  # 小数4桁
    assert warnings == []


def test_no_face():
    with pytest.raises(ApiError) as e:
        build_face(fake_result(), BRIGHT)
    assert (e.value.code, e.value.status) == ("NO_FACE", 422)


def test_picks_largest_face():
    small, large = fake_face(0.0, 0.0, 0.25), fake_face(0.5, 0.5, 0.4)
    face, warnings = build_face(fake_result(small, large), BRIGHT)
    assert face.bbox[0] == pytest.approx(0.5)
    assert "MULTIPLE_FACES" in warnings


def test_low_resolution():
    with pytest.raises(ApiError) as e:
        build_face(fake_result(fake_face(0.1, 0.1, 0.1)), BRIGHT)  # 幅 100px
    assert e.value.code == "LOW_RESOLUTION"


def test_not_frontal():
    _, warnings = build_face(fake_result(fake_face(0.2, 0.2, 0.4, yaw_deg=30)), BRIGHT)
    assert "FACE_NOT_FRONTAL" in warnings


def test_low_light():
    dark = np.full((1000, 1000, 3), 30, dtype=np.uint8)
    _, warnings = build_face(fake_result(fake_face(0.2, 0.2, 0.4)), dark)
    assert "LOW_LIGHT" in warnings
