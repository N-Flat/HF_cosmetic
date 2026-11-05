"""FastAPI アプリ本体。モデル・メッシュ・テンプレートは起動時に1回だけ読む"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, MODEL_PATH, OBJ_PATH, TEMPLATE_DIR
from app.core.composite import load_templates
from app.core.geometry import load_canonical_mesh
from app.core.landmarker import Landmarker
from app.errors import register_error_handlers
from app.routers import analyze, composite, health, templates, verify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("cosme-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 読み込みに失敗しても起動はする（/health が MODEL_NOT_READY を返す）
    try:
        app.state.landmarker = Landmarker(MODEL_PATH)
    except Exception:
        log.exception("failed to load landmarker: %s", MODEL_PATH)
        app.state.landmarker = None
    try:
        app.state.mesh = load_canonical_mesh(OBJ_PATH)
    except Exception:
        log.exception("failed to load mesh: %s", OBJ_PATH)
        app.state.mesh = None
    app.state.templates = load_templates(TEMPLATE_DIR)
    log.info("templates: %s", list(app.state.templates))
    yield
    if app.state.landmarker is not None:
        app.state.landmarker.close()


app = FastAPI(title="COSME2126 Python AI API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
register_error_handlers(app)

app.include_router(health.router)
app.include_router(templates.router)
app.include_router(analyze.router)
app.include_router(composite.router)
app.include_router(verify.router)
