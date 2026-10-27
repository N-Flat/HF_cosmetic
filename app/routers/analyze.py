"""POST /v1/analyze: 顔解析（face）・特徴検出（features）・anchor/uv"""
import time

import numpy as np
from fastapi import APIRouter, File, Request, UploadFile

from app.config import DECIMALS
from app.core.face import build_face
from app.core.features import detect_candidates
from app.core.geometry import find_anchor
from app.core.image_io import decode
from app.errors import ApiError
from app.schemas import AnalyzeResponse, Anchor, Feature, ImageSize

router = APIRouter(prefix="/v1")


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


def _round(values) -> tuple[float, ...]:
    return tuple(round(float(v), DECIMALS) for v in values)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: Request, image: UploadFile = File(...)) -> AnalyzeResponse:
    landmarker = request.app.state.landmarker
    mesh = request.app.state.mesh
    if landmarker is None or mesh is None:
        raise ApiError("MODEL_NOT_READY", 503)

    start = time.perf_counter()
    rgb = decode(image.file.read())
    decode_ms = _ms(start)
    height, width = rgb.shape[:2]

    t = time.perf_counter()
    face, warnings = build_face(landmarker.detect(rgb), rgb)
    landmark_ms = _ms(t)
    landmarks = np.array(face.landmarks)

    t = time.perf_counter()
    candidates = detect_candidates(rgb, landmarks)
    features_ms = _ms(t)

    t = time.perf_counter()
    features: list[Feature] = []
    for c in candidates:
        found = find_anchor(c.xy, landmarks, mesh)
        if found is None:  # 三角形が見つからなければその特徴は捨てる
            continue
        anchor, uv = found
        features.append(Feature(
            id=f"f{len(features) + 1}",
            type=c.type,
            xy=_round(c.xy),
            uv=_round(uv),
            anchor=Anchor(tri=anchor.tri, bary=_round(anchor.bary)),
            size=round(c.size, DECIMALS),
            confidence=round(c.confidence, DECIMALS),
        ))
    anchor_ms = _ms(t)

    request.state.log_extra = {"features": len(features), "warnings": ",".join(warnings)}
    return AnalyzeResponse(
        request_id=request.state.request_id,
        image=ImageSize(width=width, height=height),
        face=face,
        features=features,
        warnings=warnings,
        timings_ms={"decode": decode_ms, "landmark": landmark_ms, "features": features_ms,
                    "anchor": anchor_ms, "total": _ms(start)},
    )
