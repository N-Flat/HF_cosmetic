"""テストアプリ

GET /demo/face      顔認証（/v1/face/*）
GET /demo/beautify  美肌加工（/v1/beautify）
"""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

_STATIC = Path(__file__).resolve().parent.parent / "static"


@router.get("/demo/face", include_in_schema=False)
def face_demo() -> FileResponse:
    return FileResponse(_STATIC / "face_demo.html", media_type="text/html")


@router.get("/demo/beautify", include_in_schema=False)
def beautify_demo() -> FileResponse:
    return FileResponse(_STATIC / "beautify_demo.html", media_type="text/html")
