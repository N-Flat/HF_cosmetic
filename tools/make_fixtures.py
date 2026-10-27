"""Next.js 用のサンプル JSON を fixtures/ に作る（実際の API の出力そのもの）

使い方:
    python tools/make_fixtures.py

analyze・composite には tests/images/ の1枚目の写真を使う。
analyze_with_features.json は、その写真に人工のほくろ等を描いた画像の結果（features の例を見せるため）
"""
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
from tests.conftest import FACE_IMAGES, SPOTS, _darken, jpeg_bytes, post_image, to_jpeg  # noqa: E402
from app.core.image_io import decode  # noqa: E402

import numpy as np  # noqa: E402

OUT = ROOT / "fixtures"


def save(name: str, res) -> None:
    body = {"_status": res.status_code, **res.json()}
    (OUT / name).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"作成: fixtures/{name}  ({res.status_code})")


def main() -> None:
    if not FACE_IMAGES:
        sys.exit("tests/images/ に顔写真を置いてください")
    OUT.mkdir(exist_ok=True)
    face = FACE_IMAGES[0].read_bytes()

    with TestClient(app) as c:
        save("health.json", c.get("/health"))
        save("templates.json", c.get("/v1/templates"))
        save("analyze.json", post_image(c, face))

        rgb = decode(face)
        h, w = rgb.shape[:2]
        landmarks = np.array(post_image(c, face).json()["face"]["landmarks"])
        for _name, _kind, idx, r, factor, tint in SPOTS:
            p = landmarks[list(idx), :2].mean(axis=0) * (w, h)
            rgb = _darken(rgb, (int(p[0]), int(p[1])), r, factor, tint)
        save("analyze_with_features.json", post_image(c, to_jpeg(rgb)))

        save("composite.json", post_image(c, face, "/v1/composite", template_id="past_city"))

        save("error_no_face.json", post_image(c, jpeg_bytes()))
        save("error_invalid_image.json", post_image(c, b"not an image"))
        save("error_template_not_found.json",
             post_image(c, face, "/v1/composite", template_id="nope"))


if __name__ == "__main__":
    main()
