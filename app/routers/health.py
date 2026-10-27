from fastapi import APIRouter, Request

from app.config import FEATURES_VERSION, MODEL_NAME
from app.errors import ApiError
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    if getattr(state, "landmarker", None) is None or getattr(state, "mesh", None) is None:
        raise ApiError("MODEL_NOT_READY", 503)
    return HealthResponse(
        status="ok",
        models={"landmarker": MODEL_NAME, "features": FEATURES_VERSION},
    )
