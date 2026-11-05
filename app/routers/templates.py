import shutil
import uuid
from typing import Literal

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse

from app.config import CUSTOM_PREFIX, TEMPLATE_DIR
from app.core.composite import save_custom_template, slot_from_face
from app.core.image_io import decode
from app.core.regions import FACE_OVAL, LEFT_IRIS_CENTER, MOUTH_CENTER_PAIR, RIGHT_IRIS_CENTER
from app.errors import ApiError
from app.schemas import Template, TemplatesResponse

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
        raise ApiError("TEMPLATE_NOT_FOUND", 404, "サムネイル画像が見つかりません。")
    return FileResponse(path, media_type="image/jpeg")


def _detect_face(request: Request, rgb: np.ndarray) -> tuple[dict, np.ndarray]:
    """背景に人物が写っていれば、その顔から slot と楕円を作る"""
    landmarker = request.app.state.landmarker
    if landmarker is None:
        raise ApiError("MODEL_NOT_READY", 503)
    result = landmarker.detect(rgb)
    if not result.face_landmarks:
        raise ApiError("NO_FACE", 422, "背景画像に顔が見つかりません。顔を入れる範囲をドラッグで指定してください。")
    h, w = rgb.shape[:2]
    faces = [np.array([(p.x * w, p.y * h) for p in lms]) for lms in result.face_landmarks]
    pts = max(faces, key=lambda f: np.ptp(f[:, 0]) * np.ptp(f[:, 1]))
    slot = {"left_eye": pts[LEFT_IRIS_CENTER].tolist(), "right_eye": pts[RIGHT_IRIS_CENTER].tolist(),
            "mouth": pts[list(MOUTH_CENTER_PAIR)].mean(axis=0).tolist()}
    oval = pts[FACE_OVAL]
    lo, hi = oval.min(axis=0), oval.max(axis=0)
    return slot, np.array([*(lo + hi) / 2, *(hi - lo) / 2], np.float32)


@router.post("/templates", response_model=Template)
def add_template(
    request: Request,
    image: UploadFile = File(...),
    label: str = Form(...),
    era: Literal["past", "present", "future"] = Form("present"),
    face_x: float | None = Form(None),  # 顔を入れる範囲（画像に対する 0〜1）。省略時は顔を自動検出
    face_y: float | None = Form(None),
    face_w: float | None = Form(None),
    face_h: float | None = Form(None),
) -> Template:
    label = label.strip()[:40]
    if not label:
        raise ApiError("INVALID_REQUEST", 400, "テンプレート名を入力してください。")
    rgb = decode(image.file.read())
    h, w = rgb.shape[:2]
    box = (face_x, face_y, face_w, face_h)
    if all(v is not None for v in box):
        x, y, bw, bh = box
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ApiError("INVALID_REQUEST", 400, "顔の範囲は画像の内側で指定してください。")
        face = np.array([(x + bw / 2) * w, (y + bh / 2) * h, bw * w / 2, bh * h / 2], np.float32)
        slot = slot_from_face(face)
    else:
        slot, face = _detect_face(request, rgb)
    color_match = not all(v is not None for v in box)  # 写っている人物の肌色に合わせるのは自動検出のときだけ
    if face[2] * 2 < 60 or face[3] * 2 < 60:
        raise ApiError("INVALID_REQUEST", 400, "顔の範囲が小さすぎます（60px 以上にしてください）。")

    tid = f"{CUSTOM_PREFIX}{uuid.uuid4().hex[:8]}"
    t = save_custom_template(TEMPLATE_DIR, tid, era, label, rgb, slot, face, color_match)
    request.app.state.templates[tid] = t
    return t.info


@router.delete("/templates/{template_id}", status_code=204)
def delete_template(template_id: str, request: Request) -> None:
    """画面から追加したテンプレートだけ削除できる"""
    if template_id not in request.app.state.templates:
        raise ApiError("TEMPLATE_NOT_FOUND", 404)
    if not template_id.startswith(CUSTOM_PREFIX):
        raise ApiError("INVALID_REQUEST", 400, "標準のテンプレートは削除できません。")
    del request.app.state.templates[template_id]
    shutil.rmtree(TEMPLATE_DIR / template_id, ignore_errors=True)
