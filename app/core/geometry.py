"""顔メッシュ・anchor・uv の計算（資料17枚目）"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.schemas import Anchor


@dataclass(frozen=True)
class Mesh:
    tris: np.ndarray  # (898, 3) 三角形の頂点番号（0始まり＝MediaPipe の点番号）
    uv: np.ndarray  # (468, 2) 点番号ごとの UV（obj の vt そのまま、v は上向き）


def load_canonical_mesh(path: Path) -> Mesh:
    """canonical_face_model.obj から三角形と UV を読む

    注意: この obj は「頂点の番号」と「vt の番号」が一致しない（f 行は v/vt で別の番号）。
    そのため vt[頂点番号] ではなく、f 行の対応表で点番号ごとの UV に並べ直す。
    """
    vts: list[tuple[float, float]] = []
    tris: list[tuple[int, int, int]] = []
    vt_of_vertex: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "vt":
            vts.append((float(parts[1]), float(parts[2])))
        elif parts[0] == "f":
            tri = []
            for corner in parts[1:4]:
                v, vt = (int(n) - 1 for n in corner.split("/")[:2])  # obj は1始まり
                vt_of_vertex[v] = vt
                tri.append(v)
            tris.append(tuple(tri))

    uv = np.array([vts[vt_of_vertex[v]] for v in range(len(vt_of_vertex))])
    return Mesh(tris=np.array(tris), uv=uv)


def barycentric(p, a, b, c):
    """三角形 abc の中での p の重心座標。退化した三角形なら None"""
    v0, v1, v2 = b - a, c - a, p - a
    d = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(d) < 1e-12:
        return None
    l1 = (v2[0] * v1[1] - v1[0] * v2[1]) / d
    l2 = (v0[0] * v2[1] - v2[0] * v0[1]) / d
    return (1 - l1 - l2, l1, l2)


def find_anchor(xy, landmarks: np.ndarray, mesh: Mesh) -> tuple[Anchor, tuple[float, float]] | None:
    """xy を含む三角形を landmarks の xy 上で探し、(anchor, uv) を返す。見つからなければ None

    landmarks: (478, 3) の配列（MediaPipe の x, y, z）
    barycentric と同じ計算を、898 個の三角形まとめて numpy で行う
    """
    p = np.asarray(xy, dtype=float)
    a, b, c = (landmarks[mesh.tris[:, k], :2] for k in range(3))  # 各 (898, 2)
    v0, v1, v2 = b - a, c - a, p - a
    d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    valid = np.abs(d) >= 1e-12
    d = np.where(valid, d, 1.0)
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / d
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / d
    bary = np.stack([1 - l1 - l2, l1, l2], axis=1)

    inside = valid & (bary.min(axis=1) >= 0)
    if not inside.any():
        return None
    # 複数見つかったら3頂点の z 平均が最小（カメラに近い）もの
    z = landmarks[mesh.tris, 2].mean(axis=1)
    i = int(np.argmin(np.where(inside, z, np.inf)))

    tri, w = mesh.tris[i], bary[i]
    uv = sum(w[k] * mesh.uv[tri[k]] for k in range(3))
    anchor = Anchor(tri=tuple(int(t) for t in tri), bary=tuple(float(x) for x in w))
    return anchor, (float(uv[0]), float(uv[1]))


def restore_xy(anchor: Anchor, landmarks: np.ndarray) -> np.ndarray:
    """anchor と landmarks から位置を復元する（検算用。ブラウザ側と同じ計算）"""
    return sum(anchor.bary[k] * landmarks[anchor.tri[k], :2] for k in range(3))
