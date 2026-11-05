"""POST /v1/verify: 2枚の写真が同一人物かを判定する（簡易版。app/core/verify.py を参照）"""
import time

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile

from app.config import VERIFY_THRESHOLD
from app.core.face import build_face
from app.core.image_io import decode
from app.core.verify import shape_distance, shape_points, similarity
from app.errors import ApiError
from app.schemas import VerifyFace, VerifyResponse

router = APIRouter(prefix="/v1")


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


@router.post("/verify", response_model=VerifyResponse)
def verify(
    request: Request,
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    threshold: float | None = Form(None),  # 省略時は config.VERIFY_THRESHOLD
) -> VerifyResponse:
    landmarker = request.app.state.landmarker
    if landmarker is None:
        raise ApiError("MODEL_NOT_READY", 503)
    threshold = VERIFY_THRESHOLD if threshold is None else min(max(threshold, 0.0), 100.0)

    start = time.perf_counter()
    shapes, faces = [], []
    for n, upload in enumerate((image_a, image_b), 1):
        try:
            rgb = decode(upload.file.read())
            face, warnings = build_face(landmarker.detect(rgb), rgb)
        except ApiError as e:  # どちらの写真か分かるようにする
            raise ApiError(e.code, e.status, f"{n}枚目: {e.msg}")
        h, w = rgb.shape[:2]
        shapes.append(shape_points(np.array(face.landmarks), w, h))
        faces.append(VerifyFace(bbox=face.bbox, warnings=warnings))
    landmark_ms = _ms(start)

    rms = shape_distance(*shapes)
    score = similarity(rms)
    return VerifyResponse(
        request_id=request.state.request_id,
        similarity=score,
        threshold=threshold,
        same_person=score >= threshold,
        method="landmark-shape-procrustes",
        rms=round(rms, 5),
        faces=faces,
        timings_ms={"landmark": landmark_ms, "total": _ms(start)},
    )
