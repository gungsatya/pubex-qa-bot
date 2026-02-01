from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from app.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TIMEOUT,
    DEFAULT_OLLAMA_TEST_OUTPUT_DIR,
    DEFAULT_VLM_MODEL,
)
from app.core.ingestion_pipeline import PROMPT_PUBEX
from app.utils.image_utils import validate_image_bytes


from llama_index.core.prompts import PromptTemplate


logger = logging.getLogger(__name__)

SESSION = requests.Session()


def _safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return cleaned or "output"


def _build_markdown_header(
    *,
    model: str,
    options: dict[str, Any] | None,
    elapsed_seconds: float,
    image_path: Path,
    prompt: str,
    response_meta: dict[str, Any],
) -> str:
    options_text = json.dumps(options or {}, ensure_ascii=True, sort_keys=True)
    response_text = json.dumps(response_meta, ensure_ascii=True, sort_keys=True, indent=2)
    header_lines = [
        "# Ollama Chat Test",
        "",
        f"- Model: {model}",
        f"- Options: {options_text}",
        f"- Elapsed Seconds: {elapsed_seconds:.2f}",
        f"- Image: {image_path}",
        f"- Prompt: {prompt}",
        "",
        "## Ollama Response Meta",
        "```json",
        response_text,
        "```",
        "",
        "---",
        "",
    ]
    return "\n".join(header_lines)


def run_ollama_chat_test(
    *,
    image_path: str,
    model: str | None = None,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    output_dir: str | Path | None = None,
) -> Path:
    model = model or DEFAULT_VLM_MODEL
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image tidak ditemukan: {path}")

    image_bytes = path.read_bytes()
    if not validate_image_bytes(image_bytes, 1):
        raise ValueError(f"Image tidak valid atau rusak: {path}")

    prompt = PROMPT_PUBEX.format(
                        document_name="-",
                        issuer_name="-",
                        issuer_code="-",
                        document_year="-",
                        document_metadata="-",
                        slide_no="-",
                        total_pages="-",
                    )
    options: dict[str, Any] = {
        "temperature": 0.2,
        "num_predict": 768,      # batasi panjang output
        "top_p": 0.8,            # sampling lebih ketat
        "top_k": 30,             # sampling lebih ketat
        "repeat_penalty": 1.2,   # hukum pengulangan
    }

    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }
        ],
    }
    payload["options"] = options

    start = time.monotonic()
    try:
        response = SESSION.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=DEFAULT_OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ollama chat error: {exc}") from exc
    elapsed = time.monotonic() - start

    try:
        content = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response from Ollama: {data}") from exc

    response_meta = {k: v for k, v in data.items() if k != "message"}

    output_root = Path(output_dir) if output_dir else Path(DEFAULT_OLLAMA_TEST_OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = _safe_filename(model)
    output_path = output_root / f"{timestamp}_{safe_model}.md"

    header = _build_markdown_header(
        model=model,
        options=options,
        elapsed_seconds=elapsed,
        image_path=path,
        prompt=prompt,
        response_meta=response_meta,
    )
    output_text = header + content.strip() + "\n"
    output_path.write_text(output_text, encoding="utf-8")

    logger.info("Ollama chat test saved: %s (%.2fs)", output_path, elapsed)
    return output_path
