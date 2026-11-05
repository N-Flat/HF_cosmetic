"""登録した顔の特徴量の保存先（JSON ファイル1つ）

保存するのは user_id・特徴量（128 次元 × 枚数）・登録日時だけ。顔画像は保存しない。
展示用の小さな規模（数百人程度）を想定して、起動時に全部メモリに読む
"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class FaceStore:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._users: dict[str, dict] = {}
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            for uid, u in raw.get("users", {}).items():
                self._users[uid] = {**u, "embeddings": np.array(u["embeddings"], np.float32)}

    def _save(self) -> None:
        """一時ファイルに書いてから置き換える（途中で落ちても壊れないように）"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"users": {uid: {**u, "embeddings": np.round(u["embeddings"], 6).tolist()}
                          for uid, u in self._users.items()}}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)

    def put(self, user_id: str, embeddings: np.ndarray) -> bool:
        """登録（同じ user_id があれば置き換え）。置き換えたら True"""
        with self._lock:
            replaced = user_id in self._users
            self._users[user_id] = {
                "embeddings": embeddings.astype(np.float32),
                "enrolled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._save()
            return replaced

    def get(self, user_id: str) -> np.ndarray | None:
        with self._lock:
            u = self._users.get(user_id)
            return None if u is None else u["embeddings"]

    def items(self) -> list[tuple[str, np.ndarray]]:
        with self._lock:
            return [(uid, u["embeddings"]) for uid, u in self._users.items()]

    def delete(self, user_id: str) -> bool:
        with self._lock:
            if self._users.pop(user_id, None) is None:
                return False
            self._save()
            return True

    def __len__(self) -> int:
        return len(self._users)
