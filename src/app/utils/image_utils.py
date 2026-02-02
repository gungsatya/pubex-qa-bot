from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def validate_image_bytes(img_bytes: bytes, page_no: int) -> bool:
    try:
        Image.open(io.BytesIO(img_bytes)).verify()
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Slide %s: image tidak valid, dilewati.", page_no)
        return False


def load_slide_image_bytes(image_path: str | None) -> bytes | None:
    if not image_path:
        return None
    try:
        return Path(image_path).read_bytes()
    except OSError:
        logger.warning("Gagal membaca image slide dari path: %s", image_path)
        return None


def downscale_png(img_bytes: bytes, max_w: int = 1280) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if img.width <= max_w:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    new_h = int(img.height * (max_w / img.width))
    img = img.resize((max_w, new_h), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
