"""顔認証モデル（OpenCV Zoo の YuNet・SFace）を models/ にダウンロードする

使い方:
    python tools/download_face_models.py

SFace は約 37MB あるのでリポジトリには入れていない。
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import FACE_DETECTOR_PATH, FACE_RECOGNIZER_PATH  # noqa: E402

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = [
    (FACE_DETECTOR_PATH, f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    (FACE_RECOGNIZER_PATH, f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
     "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path, url, digest in MODELS:
        if path.is_file() and sha256(path) == digest:
            print(f"取得済み: {path.name}")
            continue
        print(f"ダウンロード中: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if sha256(tmp) != digest:
            tmp.unlink()
            sys.exit(f"ハッシュが一致しません: {path.name}（URL が変わった可能性があります）")
        tmp.replace(path)
        print(f"作成: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
