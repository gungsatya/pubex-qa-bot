from __future__ import annotations

import base64
import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def validate_image_bytes(img_bytes: bytes, page_no: int) -> bool:
    try:
        Image.open(io.BytesIO(img_bytes)).verify()
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Slide %s: image tidak valid, dilewati.", page_no)
        return False


def decode_image_base64(content_base_64: str | None) -> bytes | None:
    if not content_base_64:
        return None
    try:
        return base64.b64decode(content_base_64)
    except (ValueError, TypeError):
        return None
