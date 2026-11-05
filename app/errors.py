"""どんな失敗でも {"error": {...}} の形で返す"""
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("cosme-ai")


# フロントは message をそのまま利用者に見せるので日本語にする（スタックトレース等は入れない）
MESSAGES = {
    "NO_FACE": "顔を検出できませんでした。正面の写真でお試しください。",
    "LOW_RESOLUTION": "顔が小さすぎます。もう少し近づくか、高解像度の写真でお試しください。",
    "INVALID_IMAGE": "画像を読み込めませんでした。JPEG・PNG・WebP の写真を送ってください。",
    "IMAGE_TOO_LARGE": "画像が大きすぎます（10MB まで）。",
    "TEMPLATE_NOT_FOUND": "指定されたテンプレートが見つかりません。",
    "MODEL_NOT_READY": "モデルを読み込めていません。API サーバーのログを確認してください。",
    "INVALID_REQUEST": "リクエストの形式が正しくありません。",
    "INTERNAL": "サーバー内部でエラーが発生しました。",
}


class ApiError(Exception):
    def __init__(self, code: str, status: int, msg: str = ""):
        super().__init__(code)
        self.code, self.status = code, status
        self.msg = msg or MESSAGES.get(code, code.lower())


def body(code: str, msg: str, rid: str) -> dict:
    return {"error": {"code": code, "message": msg, "request_id": rid}}


def rid(req: Request) -> str:
    return getattr(req.state, "request_id", "")


def register_error_handlers(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        total_ms = int((time.perf_counter() - start) * 1000)
        # 1リクエスト1行。顔画像そのものは保存しない
        extra = getattr(request.state, "log_extra", {})
        log.info("request_id=%s endpoint=%s status=%s total_ms=%s features=%s warnings=%s",
                 request.state.request_id, request.url.path, response.status_code, total_ms,
                 extra.get("features", "-"), extra.get("warnings") or "-")
        return response

    @app.exception_handler(ApiError)
    def _api(req: Request, e: ApiError):
        return JSONResponse(body(e.code, e.msg, rid(req)), e.status)

    @app.exception_handler(RequestValidationError)
    def _val(req: Request, e: RequestValidationError):
        return JSONResponse(body("INVALID_REQUEST", f"{MESSAGES['INVALID_REQUEST']} {e}", rid(req)), 400)

    # 存在しないパスなど、FastAPI 標準の 404/405 もエラー形式に揃える
    @app.exception_handler(StarletteHTTPException)
    def _http(req: Request, e: StarletteHTTPException):
        return JSONResponse(body("INVALID_REQUEST", str(e.detail), rid(req)), e.status_code)

    @app.exception_handler(Exception)
    def _any(req: Request, e: Exception):
        log.exception("unhandled")
        return JSONResponse(body("INTERNAL", MESSAGES["INTERNAL"], rid(req)), 500)
