"""顔の部位ごとの点番号（MediaPipe の FaceLandmarksConnections から作る）"""
from mediapipe.tasks.python.vision import FaceLandmarksConnections as _C

# 虹彩の中心（資料19枚目: 左目 468 / 右目 473。画像で見て左が 468）
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
MOUTH_CENTER_PAIR = (13, 14)  # 上唇の内側・下唇の内側。この中点を口の中心とする


def _points(connections) -> list[int]:
    return sorted({p for c in connections for p in (c.start, c.end)})


def _loop(connections) -> list[int]:
    """線のつながりをたどって、多角形の順番に並べる"""
    nxt = {c.start: c.end for c in connections}
    start = connections[0].start
    order = [start]
    while nxt[order[-1]] != start:
        order.append(nxt[order[-1]])
    return order


FACE_OVAL = _loop(_C.FACE_LANDMARKS_FACE_OVAL)

# 肌マスクから除外する部位（それぞれ凸包で塗る）
# 目はまつげ・目の下の影を拾いやすいので、ほかより広く除外する
EYES = [
    _points(_C.FACE_LANDMARKS_LEFT_EYE),
    _points(_C.FACE_LANDMARKS_RIGHT_EYE),
]
EXCLUDE_PARTS = [
    _points(_C.FACE_LANDMARKS_LEFT_EYEBROW),
    _points(_C.FACE_LANDMARKS_RIGHT_EYEBROW),
    _points(_C.FACE_LANDMARKS_LIPS),
    # 鼻の穴（小鼻と鼻の下の点）
    [48, 64, 98, 97, 2, 326, 327, 294, 278, 219, 439, 240, 460, 75, 305],
]
