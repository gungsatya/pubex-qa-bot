from __future__ import annotations

from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import OLLAMA


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=OLLAMA.retry_total,
        backoff_factor=OLLAMA.retry_backoff,
        status_forcelist=OLLAMA.retry_status_forcelist,
        allowed_methods=OLLAMA.retry_allowed_methods,
    )
    adapter = HTTPAdapter(
        pool_connections=OLLAMA.pool_connections,
        pool_maxsize=OLLAMA.pool_maxsize,
        max_retries=retries,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def build_url(path: str) -> str:
    return urljoin(f"{OLLAMA.base_url.rstrip('/')}/", path.lstrip("/"))


SESSION = _build_session()
