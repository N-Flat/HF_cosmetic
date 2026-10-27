from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.config import TEMPLATE_DIR
from app.errors import ApiError
from app.schemas import TemplatesResponse

router = APIRouter(prefix="/v1")


@router.get("/templates", response_model=TemplatesResponse)
def list_templates(request: Request) -> TemplatesResponse:
    return TemplatesResponse(templates=[t.info for t in request.app.state.templates.values()])


@router.get("/templates/{template_id}/thumb.jpg")
def thumbnail(template_id: str, request: Request) -> FileResponse:
    if template_id not in request.app.state.templates:
        raise ApiError("TEMPLATE_NOT_FOUND", 404)
    path = TEMPLATE_DIR / template_id / "thumb.jpg"
    if not path.is_file():
        raise ApiError("TEMPLATE_NOT_FOUND", 404, "thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")
