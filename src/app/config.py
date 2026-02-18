from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(code.strip()) for code in value.split(",") if code.strip())


def _parse_method_set(value: str) -> frozenset[str]:
    return frozenset(method.strip().upper() for method in value.split(",") if method.strip())


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


DATABASE_URL = os.getenv("DATABASE_URL")


@dataclass(frozen=True)
class OllamaHTTPConfig:
    base_url: str
    timeout_seconds: int
    chat_path: str
    embed_path: str
    retry_total: int
    retry_backoff: float
    retry_status_forcelist: tuple[int, ...]
    retry_allowed_methods: frozenset[str]
    pool_connections: int
    pool_maxsize: int


@dataclass(frozen=True)
class ModelGenerationConfig:
    model: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float


@dataclass(frozen=True)
class DoclingConfig:
    api_base_url: str
    api_timeout_seconds: int
    preset: str
    batch_size: int
    page_break_placeholder: str
    image_dpi: int


@dataclass(frozen=True)
class EmbeddingLLMTypeConfig:
    model: str
    batch_size: int


_RETRY_TOTAL = int(
    _env(
        "OLLAMA_HTTP_RETRY_TOTAL",
        "2",
    )
)
_RETRY_BACKOFF = float(
    _env(
        "OLLAMA_HTTP_RETRY_BACKOFF",
        "0.3",
    )
)
_RETRY_STATUS_FORCELIST = _parse_int_tuple(
    _env(
        "OLLAMA_HTTP_RETRY_STATUS_FORCELIST",
        "429,500,502,503,504",
    )
)
_RETRY_ALLOWED_METHODS = _parse_method_set(
    _env(
        "OLLAMA_HTTP_RETRY_ALLOWED_METHODS",
        "POST",
    )
)

OLLAMA = OllamaHTTPConfig(
    base_url=_env(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    ),
    timeout_seconds=int(
        _env(
            "OLLAMA_TIMEOUT_SECONDS",
            "300",
        )
    ),
    chat_path=_env(
        "OLLAMA_CHAT_PATH",
        "/api/chat",
    ),
    embed_path=_env(
        "OLLAMA_EMBED_PATH",
        "/api/embed",
    ),
    retry_total=_RETRY_TOTAL,
    retry_backoff=_RETRY_BACKOFF,
    retry_status_forcelist=_RETRY_STATUS_FORCELIST,
    retry_allowed_methods=_RETRY_ALLOWED_METHODS,
    pool_connections=int(
        _env(
            "OLLAMA_POOL_CONNECTIONS",
            "16",
        )
    ),
    pool_maxsize=int(
        _env(
            "OLLAMA_POOL_MAXSIZE",
            "16",
        )
    ),
)

DOCLING = DoclingConfig(
    api_base_url=_env("DOCLING_API_BASE_URL", "http://localhost:8081"),
    api_timeout_seconds=int(_env("DOCLING_API_TIMEOUT_SECONDS", "300")),
    preset=_env("DOCLING_PRESET", "qwen"),
    batch_size=int(_env("DOCLING_BATCH_SIZE", "1")),
    page_break_placeholder=_env("DOCLING_PAGE_BREAK_PLACEHOLDER", "<!-- page break -->"),
    image_dpi=int(_env("DOCLING_IMAGE_DPI", "175")),
)

LLM = ModelGenerationConfig(
    model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
    max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "2048")),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    top_p=float(os.getenv("LLM_TOP_P", "0.8")),
    top_k=int(os.getenv("LLM_TOP_K", "30")),
    repeat_penalty=float(os.getenv("LLM_REPEAT_PENALTY", "1.3")),
)

EMBEDDING_LLM = EmbeddingLLMTypeConfig(
    model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    batch_size=int(os.getenv("EMBED_BATCH_SIZE", "10")),
)
