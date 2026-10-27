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

レスポンスの形の例は `fixtures/` にあります。

## テスト

```
pytest
```

顔写真を使うテストは `tests/images/` に写真（本人の同意を取ったもの、またはフリー素材）を置いたときだけ実行されます。
顔写真はリポジトリには入れていません。
