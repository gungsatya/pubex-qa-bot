from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_slide_image_bytes(image_path: str | None) -> bytes | None:
    if not image_path:
        return None
    try:
        return Path(image_path).read_bytes()
    except OSError:
        logger.warning("Gagal membaca image slide dari path: %s", image_path)
        return None
