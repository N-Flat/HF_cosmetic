"""顔認証のテストアプリ（GET /demo/face）。ブラウザのカメラで撮って /v1/face/* を呼ぶ"""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

_PAGE = Path(__file__).resolve().parent.parent / "static" / "face_demo.html"


@router.get("/demo/face", include_in_schema=False)
def face_demo() -> FileResponse:
    return FileResponse(_PAGE, media_type="text/html")
