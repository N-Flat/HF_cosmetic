"""レスポンスの型。Next.js 側の types.ts と1対1（勝手に変えない）"""
from typing import Literal

from pydantic import BaseModel

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


class ImageSize(BaseModel):
    width: int
    height: int


class Face(BaseModel):
    bbox: tuple[float, float, float, float]
    landmarks: list[Vec3]  # 478
    transform: list[float]  # 16
    blendshapes: dict[str, float]


class Anchor(BaseModel):
    tri: tuple[int, int, int]
    bary: Vec3


class Template(BaseModel):
    id: str
    era: Literal["past", "present", "future"]
    label: str
    thumbnail: str
    custom: bool = False  # 画面から追加したもの（削除できる）


class TemplatesResponse(BaseModel):
    templates: list[Template]


class Feature(BaseModel):
    id: str
    type: Literal["mole", "freckle", "smile_line", "other"]
    xy: Vec2
    uv: Vec2
    anchor: Anchor
    size: float
    confidence: float


class AnalyzeResponse(BaseModel):
    request_id: str
    image: ImageSize
    face: Face
    features: list[Feature]
    warnings: list[str]
    timings_ms: dict[str, int]


class CompositeResponse(BaseModel):
    request_id: str
    template_id: str
    image: str  # data URL
    width: int
    height: int
    timings_ms: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    models: dict[str, str]


class VerifyFace(BaseModel):
    bbox: tuple[float, float, float, float]
    warnings: list[str]


class VerifyResponse(BaseModel):
    request_id: str
    similarity: float  # 一致度 0〜100(%)
    threshold: float  # これ以上で同一人物とみなす
    same_person: bool
    method: str
    rms: float  # 形状の差（小さいほど似ている）
    faces: list[VerifyFace]  # [1枚目, 2枚目]
    timings_ms: dict[str, int]


# --- 顔認証（YuNet + SFace） ---

class DetectedFace(BaseModel):
    bbox: tuple[float, float, float, float]  # [x, y, w, h]（画像に対する 0〜1）
    score: float  # 検出の信頼度
    warnings: list[str]


class FaceEnrollResponse(BaseModel):
    request_id: str
    user_id: str
    images: int  # 登録した枚数
    replaced: bool  # 同じ user_id の登録を置き換えたか
    min_pair_score: float  # 登録写真どうしの類似度の最小値（写真がそろっているかの目安）
    faces: list[DetectedFace]
    timings_ms: dict[str, int]


class FaceCandidate(BaseModel):
    user_id: str
    score: float  # コサイン類似度（-1〜1、大きいほど似ている）


class FaceIdentifyResponse(BaseModel):
    request_id: str
    matched: bool
    user_id: str | None  # 一致した人（matched=False なら null）
    score: float  # 1位の類似度（登録0人なら 0）
    threshold: float
    candidates: list[FaceCandidate]  # 類似度の高い順
    face: DetectedFace
    timings_ms: dict[str, int]


class FaceVerifyResponse(BaseModel):
    request_id: str
    user_id: str
    matched: bool
    score: float
    threshold: float
    face: DetectedFace
    timings_ms: dict[str, int]
