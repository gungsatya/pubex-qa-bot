import logging
from typing import Optional

import requests

from src.app.config import (
    TELEGRAM_API_BASE_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_DISABLE_NOTIFICATION,
)

logger = logging.getLogger(__name__)


def send_telegram_message(
    text: str,
    *,
    disable_notification: Optional[bool] = None,
    timeout_seconds: int = 15,
) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram config belum lengkap, lewati kirim pesan.")
        return False

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_notification": (
            TELEGRAM_DISABLE_NOTIFICATION
            if disable_notification is None
            else disable_notification
        ),
    }
    url = f"{TELEGRAM_API_BASE_URL.rstrip('/')}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json=payload, timeout=timeout_seconds)
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gagal kirim pesan Telegram: %s", exc)
        return False
