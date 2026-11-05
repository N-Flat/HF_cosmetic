from fastapi import APIRouter, Request

from app.config import FACE_DETECTOR_NAME, FACE_RECOGNIZER_NAME, FEATURES_VERSION, MODEL_NAME
from app.errors import ApiError
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    if getattr(state, "landmarker", None) is None or getattr(state, "mesh", None) is None:
        raise ApiError("MODEL_NOT_READY", 503)
    models = {"landmarker": MODEL_NAME, "features": FEATURES_VERSION}
    # 顔認証モデルは任意（無くても他の機能は動く）。読み込めたときだけ載せる
    if getattr(state, "face_auth", None) is not None:
        models |= {"face_detector": FACE_DETECTOR_NAME, "face_recognizer": FACE_RECOGNIZER_NAME}
    return HealthResponse(status="ok", models=models)
