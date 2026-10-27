"""geometry の単体テスト（メッシュの読み込み・重心座標・anchor）"""
import numpy as np
import pytest

from app import config
from app.core.geometry import barycentric, find_anchor, load_canonical_mesh, restore_xy


@pytest.fixture(scope="module")
def mesh():
    return load_canonical_mesh(config.OBJ_PATH)


def canonical_vertices() -> np.ndarray:
    lines = config.OBJ_PATH.read_text(encoding="utf-8").splitlines()
    return np.array([[float(v) for v in line.split()[1:4]] for line in lines if line.startswith("v ")])


def test_mesh_counts(mesh):
    assert mesh.tris.shape == (898, 3)
    assert mesh.uv.shape == (468, 2)
    assert mesh.tris.min() == 0 and mesh.tris.max() == 467


def test_uv_matches_vertex_layout(mesh):
    """UV の並びが頂点の位置と対応しているか（vt[頂点番号] で引くと崩れる）

    UV は顔を平面に広げたものなので、頂点の x と u、y と v は強く相関するはず
    """
    verts = canonical_vertices()
    assert np.corrcoef(verts[:, 0], mesh.uv[:, 0])[0, 1] > 0.9
    assert np.corrcoef(verts[:, 1], mesh.uv[:, 1])[0, 1] > 0.9


def test_barycentric():
    a, b, c = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert barycentric(np.array([0.25, 0.25]), a, b, c) == pytest.approx((0.5, 0.25, 0.25))
    assert min(barycentric(np.array([1.0, 1.0]), a, b, c)) < 0  # 外側
    assert barycentric(np.array([0.5, 0.0]), a, b, b) is None  # 退化


def test_find_anchor_restores_xy(mesh):
    # canonical の頂点を平面に置いた「顔」で、三角形の重心を探す
    verts = canonical_vertices()
    landmarks = np.zeros((478, 3))
    landmarks[:468, :2] = (verts[:, :2] - verts[:, :2].min(0)) / np.ptp(verts[:, :2], axis=0)
    landmarks[:468, 1] = 1 - landmarks[:468, 1]  # obj は y 上向き、画像は y 下向き
    landmarks[:468, 2] = -verts[:, 2]

    tri = mesh.tris[100]
    xy = landmarks[tri, :2].mean(axis=0)
    anchor, uv = find_anchor(xy, landmarks, mesh)
    assert sum(anchor.bary) == pytest.approx(1)
    assert np.abs(restore_xy(anchor, landmarks) - xy).max() < 0.003
    assert 0 <= uv[0] <= 1 and 0 <= uv[1] <= 1


def test_find_anchor_outside_face(mesh):
    landmarks = np.zeros((478, 3))
    landmarks[:468, :2] = np.random.default_rng(0).uniform(0.4, 0.6, (468, 2))
    assert find_anchor((0.01, 0.01), landmarks, mesh) is None
