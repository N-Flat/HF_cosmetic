"""顔認証 API（YuNet + SFace。app/core/face_auth.py を参照）

POST   /v1/face/enroll     登録（同意済みの user_id と画像3枚）
POST   /v1/face/identify   展示ブース用: 1対N で探す
POST   /v1/face/verify     スマホ用: user_id を指定して 1対1 で確認
DELETE /v1/face/{user_id}  削除
"""
import re
import time
from itertools import combinations

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile

from app import config as cfg
from app.core.face_auth import FaceAuth, FaceEmbedding, best_score, cosine
from app.core.face_store import FaceStore
from app.core.image_io import decode
from app.errors import ApiError
from app.schemas import (DetectedFace, FaceCandidate, FaceEnrollResponse, FaceIdentifyResponse,
                         FaceVerifyResponse)

router = APIRouter(prefix="/v1/face")


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


def _deps(request: Request) -> tuple[FaceAuth, FaceStore]:
    auth = getattr(request.app.state, "face_auth", None)
    if auth is None:
        raise ApiError("MODEL_NOT_READY", 503, "顔認証モデルを読み込めていません。"
                       "python tools/download_face_models.py を実行してから API を再起動してください。")
    return auth, request.app.state.face_store


def _check_user_id(user_id: str) -> str:
    user_id = user_id.strip()
    if not re.fullmatch(cfg.USER_ID_PATTERN, user_id):
        raise ApiError("INVALID_REQUEST", 400, "user_id は半角英数字・_・- の 1〜64 文字にしてください。")
    return user_id


def _embed(auth: FaceAuth, upload: UploadFile, allow_multiple: bool = True, label: str = "") -> FaceEmbedding:
    try:
        return auth.embed(decode(upload.file.read()), allow_multiple)
    except ApiError as e:
        if label:  # どの写真か分かるようにする
            raise ApiError(e.code, e.status, f"{label}: {e.msg}")
        raise


def _detected(e: FaceEmbedding) -> DetectedFace:
    return DetectedFace(bbox=e.bbox, score=e.score, warnings=e.warnings)


@router.post("/enroll", response_model=FaceEnrollResponse)
def enroll(
    request: Request,
    user_id: str = Form(...),
    consent: bool = Form(...),  # 本人の同意を取ったら true
    images: list[UploadFile] = File(...),  # 同じ名前 images で3枚
) -> FaceEnrollResponse:
    auth, store = _deps(request)
    user_id = _check_user_id(user_id)
    if not consent:
        raise ApiError("CONSENT_REQUIRED", 400)
    if len(images) != cfg.FACE_ENROLL_IMAGES:
        raise ApiError("INVALID_REQUEST", 400, f"画像は {cfg.FACE_ENROLL_IMAGES} 枚送ってください（{len(images)} 枚でした）。")

    start = time.perf_counter()
    embs = [_embed(auth, img, allow_multiple=False, label=f"{n}枚目") for n, img in enumerate(images, 1)]

    # 3枚が同じ人か（別人や他人の写真が混ざっていないか）
    pair = min(cosine(a.vector, b.vector) for a, b in combinations(embs, 2))
    if pair < cfg.FACE_MATCH_THRESHOLD:
        raise ApiError("ENROLL_INCONSISTENT", 422)

    replaced = store.put(user_id, np.stack([e.vector for e in embs]))
    return FaceEnrollResponse(
        request_id=request.state.request_id,
        user_id=user_id,
        images=len(embs),
        replaced=replaced,
        min_pair_score=round(pair, 4),
        faces=[_detected(e) for e in embs],
        timings_ms={"total": _ms(start)},
    )


@router.post("/identify", response_model=FaceIdentifyResponse)
def identify(request: Request, image: UploadFile = File(...)) -> FaceIdentifyResponse:
    auth, store = _deps(request)
    start = time.perf_counter()
    e = _embed(auth, image)
    embed_ms = _ms(start)

    t = time.perf_counter()
    ranked = sorted(((uid, best_score(e.vector, embs)) for uid, embs in store.items()),
                    key=lambda x: x[1], reverse=True)
    top = ranked[: cfg.FACE_IDENTIFY_TOP_K]
    best_uid, best = top[0] if top else (None, 0.0)
    matched = best_uid is not None and best >= cfg.FACE_MATCH_THRESHOLD
    return FaceIdentifyResponse(
        request_id=request.state.request_id,
        matched=matched,
        user_id=best_uid if matched else None,
        score=round(best, 4),
        threshold=cfg.FACE_MATCH_THRESHOLD,
        candidates=[FaceCandidate(user_id=uid, score=round(s, 4)) for uid, s in top],
        face=_detected(e),
        timings_ms={"embed": embed_ms, "search": _ms(t), "total": _ms(start)},
    )


@router.post("/verify", response_model=FaceVerifyResponse)
def verify(request: Request, user_id: str = Form(...), image: UploadFile = File(...)) -> FaceVerifyResponse:
    auth, store = _deps(request)
    user_id = _check_user_id(user_id)
    enrolled = store.get(user_id)
    if enrolled is None:
        raise ApiError("USER_NOT_FOUND", 404)

    start = time.perf_counter()
    e = _embed(auth, image)
    score = best_score(e.vector, enrolled)
    return FaceVerifyResponse(
        request_id=request.state.request_id,
        user_id=user_id,
        matched=score >= cfg.FACE_MATCH_THRESHOLD,
        score=round(score, 4),
        threshold=cfg.FACE_MATCH_THRESHOLD,
        face=_detected(e),
        timings_ms={"total": _ms(start)},
    )


@router.delete("/{user_id}", status_code=204)
def delete(user_id: str, request: Request) -> None:
    # 削除はモデルが無くてもできるようにする（特徴量を消すだけ）
    if not request.app.state.face_store.delete(_check_user_id(user_id)):
        raise ApiError("USER_NOT_FOUND", 404)
