# COSME2126 Python AI API

Next.js（展示アプリ）から呼ぶ顔解析・合成 API（FastAPI）。

## セットアップ

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
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

レスポンスの形の例は `fixtures/` にあります。

## テスト

```
pytest
```

顔写真を使うテストは `tests/images/` に写真（本人の同意を取ったもの、またはフリー素材）を置いたときだけ実行されます。
顔写真はリポジトリには入れていません。
