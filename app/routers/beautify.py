"""美肌加工 API（app/core/beautify.py を参照）

GET  /v1/beautify/styles  加工方式の一覧
POST /v1/beautify         写真を加工する（複数の方式をまとめて返せる）
"""
import time

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile

from app.config import BEAUTIFY_DEFAULT_STRENGTH
from app.core.beautify import STYLES, apply, build_context
from app.core.composite import to_data_url
from app.core.face import build_face
from app.core.image_io import decode
from app.errors import ApiError
from app.schemas import BeautifyResponse, BeautifyResult, BeautifyStyle, BeautifyStylesResponse

router = APIRouter(prefix="/v1/beautify")


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


@router.get("/styles", response_model=BeautifyStylesResponse)
def list_styles() -> BeautifyStylesResponse:
    return BeautifyStylesResponse(
        styles=[BeautifyStyle(id=k, label=label, description=desc) for k, (label, desc) in STYLES.items()],
        default_strength=BEAUTIFY_DEFAULT_STRENGTH,
    )


def _parse_styles(styles: str | None) -> list[str]:
    if not styles or not styles.strip():
        return list(STYLES)
    names = list(dict.fromkeys(s.strip() for s in styles.split(",") if s.strip()))
    unknown = [n for n in names if n not in STYLES]
    if unknown or not names:
        raise ApiError("INVALID_REQUEST", 400,
                       f"styles に知らない方式があります: {', '.join(unknown)}（使えるもの: {', '.join(STYLES)}）")
    return names


@router.post("", response_model=BeautifyResponse)
def beautify(
    request: Request,
    image: UploadFile = File(...),
    styles: str | None = Form(None),  # カンマ区切り（例: "natural,smooth"）。省略時は全部
    strength: float | None = Form(None),  # 0〜1。省略時は BEAUTIFY_DEFAULT_STRENGTH
) -> BeautifyResponse:
    landmarker = request.app.state.landmarker
    if landmarker is None:
        raise ApiError("MODEL_NOT_READY", 503)
    names = _parse_styles(styles)
    strength = BEAUTIFY_DEFAULT_STRENGTH if strength is None else min(max(strength, 0.0), 1.0)

    start = time.perf_counter()
    rgb = decode(image.file.read())
    decode_ms = _ms(start)

    t = time.perf_counter()
    face, warnings = build_face(landmarker.detect(rgb), rgb)
    landmark_ms = _ms(t)

    t = time.perf_counter()
    ctx = build_context(rgb, np.array(face.landmarks))
    mask_ms = _ms(t)

    t = time.perf_counter()
    outputs = [(name, apply(rgb, ctx, name, strength)) for name in names]
    beautify_ms = _ms(t)

    t = time.perf_counter()
    results = [BeautifyResult(style=name, image=to_data_url(out)) for name, out in outputs]
    encode_ms = _ms(t)

    request.state.log_extra = {"features": ctx.features, "warnings": ",".join(warnings)}
    return BeautifyResponse(
        request_id=request.state.request_id,
        strength=strength,
        width=rgb.shape[1],
        height=rgb.shape[0],
        results=results,
        protected_features=ctx.features,
        warnings=warnings,
        timings_ms={"decode": decode_ms, "landmark": landmark_ms, "mask": mask_ms,
                    "beautify": beautify_ms, "encode": encode_ms, "total": _ms(start)},
    )
