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
