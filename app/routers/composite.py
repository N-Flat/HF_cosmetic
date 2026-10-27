"""POST /v1/composite: 利用者の顔を時代テンプレートに合成する"""
import time

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile

from app.core.composite import composite as compose
from app.core.composite import to_data_url
from app.core.face import build_face
from app.core.image_io import decode
from app.errors import ApiError
from app.schemas import CompositeResponse

router = APIRouter(prefix="/v1")


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


@router.post("/composite", response_model=CompositeResponse)
def composite(
    request: Request,
    image: UploadFile = File(...),
    template_id: str = Form(...),
) -> CompositeResponse:
    landmarker = request.app.state.landmarker
    if landmarker is None:
        raise ApiError("MODEL_NOT_READY", 503)
    template = request.app.state.templates.get(template_id)
    if template is None:
        raise ApiError("TEMPLATE_NOT_FOUND", 404)

    start = time.perf_counter()
    rgb = decode(image.file.read())
    decode_ms = _ms(start)

    t = time.perf_counter()
    face, warnings = build_face(landmarker.detect(rgb), rgb)
    landmark_ms = _ms(t)

    t = time.perf_counter()
    out = compose(rgb, np.array(face.landmarks), template)
    composite_ms = _ms(t)

    t = time.perf_counter()
    data_url = to_data_url(out)
    encode_ms = _ms(t)

    request.state.log_extra = {"warnings": ",".join(warnings)}
    return CompositeResponse(
        request_id=request.state.request_id,
        template_id=template_id,
        image=data_url,
        width=out.shape[1],
        height=out.shape[0],
        timings_ms={"decode": decode_ms, "landmark": landmark_ms, "composite": composite_ms,
                    "encode": encode_ms, "total": _ms(start)},
    )
