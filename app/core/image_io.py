import io

import numpy as np
from PIL import Image, ImageOps

from app.config import MAX_BYTES, MAX_SIDE
from app.errors import ApiError


def decode(data: bytes) -> np.ndarray:
    if len(data) > MAX_BYTES:
        raise ApiError("IMAGE_TOO_LARGE", 413)
    try:
        im = Image.open(io.BytesIO(data))
        if im.format not in ("JPEG", "PNG"):
            raise ValueError(im.format)
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    except Exception:
        raise ApiError("INVALID_IMAGE", 400)
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.ascontiguousarray(im)  # RGB
    # NOTE: never flip horizontally
