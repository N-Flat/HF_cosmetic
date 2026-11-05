# COSME2126 Python AI API

Next.js（展示アプリ）から呼ぶ顔解析・合成 API（FastAPI）。

## セットアップ

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tools/download_face_models.py   # 顔認証モデル（YuNet・SFace）を models/ に取得
```

## 起動

```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

http://localhost:8000/docs で動作確認できます。

## エンドポイント

| メソッド | パス | 内容 |
|---|---|---|
| GET | /health | モデルの読み込み状態 |
| GET | /v1/templates | 合成テンプレート一覧 |
| GET | /v1/templates/{id}/thumbnail | サムネイル画像 |
| POST | /v1/analyze | 顔のランドマーク・ほくろ等の特徴 |
| POST | /v1/composite | テンプレートへの顔合成 |
| POST | /v1/templates | 背景テンプレートの追加（multipart: image, label, era, 任意で face_x/face_y/face_w/face_h = 顔を入れる範囲 0〜1。省略時は背景の人物の顔を自動検出） |
| DELETE | /v1/templates/{id} | 画面から追加したテンプレートの削除 |
| POST | /v1/verify | 同一人物判定・簡易版（multipart: image_a, image_b, 任意で threshold）。ランドマーク形状の比較で、顔認識モデルではない。しきい値は app/config.py の VERIFY_THRESHOLD |
| POST | /v1/face/enroll | 顔認証の登録（multipart: user_id, consent=true, images を3枚）。同じ user_id は置き換え |
| POST | /v1/face/identify | 展示ブース用：1対N で探す（multipart: image）。matched・user_id・候補上位3人 |
| POST | /v1/face/verify | スマホ用：user_id を指定して 1対1 で確認（multipart: user_id, image） |
| DELETE | /v1/face/{user_id} | 顔認証の登録を削除 |
| GET | /demo/face | 顔認証のテストアプリ（ブラウザのカメラで撮って上の4つを呼ぶ） |

レスポンスの形の例は `fixtures/` にあります。

## 顔認証（YuNet + SFace）

- 顔検出は OpenCV の YuNet、特徴量は SFace（128次元）。score はコサイン類似度で、`app/config.py` の `FACE_MATCH_THRESHOLD`（0.363、OpenCV 推奨値）以上で本人とみなす。展示環境の照明・カメラで要調整
- 保存するのは特徴量だけ（`data/face_db.json`）。顔画像は保存しない。`data/` はリポジトリに入れない
- 登録時は3枚とも顔が1人だけ写っていること、3枚どうしが同じ人であること（類似度がしきい値以上）を確認する
- テストアプリ: サーバーを起動して http://localhost:8000/demo/face を開く

## テスト

```
pytest
```

顔写真を使うテストは `tests/images/` に写真（本人の同意を取ったもの、またはフリー素材）を置いたときだけ実行されます。
顔写真はリポジトリには入れていません。
