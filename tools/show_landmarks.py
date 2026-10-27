"""確認用: /v1/analyze の結果を写真に描いて output/ に保存する（API とは別の手元ツール）

使い方:
    python tools/show_landmarks.py tests/images/写真.jpg          # 点・bbox・特徴
    python tools/show_landmarks.py tests/images/写真.jpg --mask   # 肌マスクの範囲も重ねる

描くもの:
    緑の点 = landmarks / 青い四角 = bbox / 赤 = 虹彩・鼻・口
    特徴: 赤い丸 = mole、オレンジの丸 = freckle（丸の大きさ = size）
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # app を import するため

from app import config  # noqa: E402
from app.core.features import _normalize, _skin_mask  # noqa: E402
from app.core.image_io import decode  # noqa: E402
from app.main import app  # noqa: E402

LABELED_POINTS = {1: "nose", 468: "L iris", 473: "R iris", 13: "mouth"}
COLORS = {"mole": (0, 0, 255), "freckle": (0, 165, 255)}  # BGR


def draw_mask(canvas: np.ndarray, rgb: np.ndarray, landmarks: np.ndarray) -> None:
    """肌マスク（ほくろを探す範囲）を水色で重ねる"""
    h, w = rgb.shape[:2]
    crop, crop_pts, m = _normalize(rgb, landmarks[:, :2] * (w, h))
    skin = _skin_mask(crop.shape[:2], crop_pts)
    back = cv2.warpAffine(skin, cv2.invertAffineTransform(m), (w, h))
    tint = canvas.copy()
    tint[back > 0] = (255, 255, 0)
    cv2.addWeighted(tint, 0.25, canvas, 0.75, 0, dst=canvas)


def main(image_path: Path, show_mask: bool) -> None:
    data = image_path.read_bytes()
    with TestClient(app) as client:
        res = client.post("/v1/analyze", files={"image": (image_path.name, data, "image/jpeg")})
    body = res.json()
    if res.status_code != 200:
        print(f"検出できませんでした: {body['error']['code']}")
        return

    rgb = decode(data)  # API と同じ前処理（EXIF 回転・縮小）をした画像に描く
    h, w = rgb.shape[:2]
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # OpenCV は BGR
    radius = max(1, w // 500)
    face = body["face"]
    landmarks = np.array(face["landmarks"])

    if show_mask:
        draw_mask(canvas, rgb, landmarks)

    for x, y, _z in landmarks:
        cv2.circle(canvas, (int(x * w), int(y * h)), radius, (0, 255, 0), -1)
    for idx, name in LABELED_POINTS.items():
        p = (int(landmarks[idx, 0] * w), int(landmarks[idx, 1] * h))
        cv2.circle(canvas, p, radius * 3, (0, 0, 255), -1)
        cv2.putText(canvas, name, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 + w / 2500, (0, 0, 255), 1 + w // 1000)

    for f in body["features"]:
        p = (int(f["xy"][0] * w), int(f["xy"][1] * h))
        r = max(4, int(f["size"] * w))
        cv2.circle(canvas, p, r + 3, COLORS.get(f["type"], (255, 0, 255)), 2)

    bx, by, bw, bh = face["bbox"]
    cv2.rectangle(canvas, (int(bx * w), int(by * h)), (int((bx + bw) * w), int((by + bh) * h)),
                  (255, 128, 0), 2)

    out_dir = config.BASE_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_landmarks.jpg"
    cv2.imwrite(str(out_path), canvas)

    kinds = [f["type"] for f in body["features"]]
    top = sorted(face["blendshapes"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"保存しました: {out_path}")
    print(f"画像サイズ: {w}x{h}  bbox: {face['bbox']}")
    print(f"warnings: {body['warnings']}")
    print(f"features: {len(kinds)} 件（mole {kinds.count('mole')} / freckle {kinds.count('freckle')}）")
    print(f"timings_ms: {body['timings_ms']}")
    print("表情 上位5:", ", ".join(f"{k}={v}" for k, v in top))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        sys.exit(1)
    main(Path(args[0]), "--mask" in sys.argv)
