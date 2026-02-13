from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
from typing import Optional

from PIL import Image

from app.config import OLLAMA, VLM
from app.core.ollama_client import SESSION, build_url


def _to_image_bytes(
    *,
    image: Optional[Image.Image] = None,
    image_bytes: Optional[bytes] = None,
    image_path: Optional[str] = None,
) -> bytes:
    if image_bytes is not None:
        return image_bytes
    if image is not None:
        buff = BytesIO()
        image.convert("RGB").save(buff, format="PNG")
        return buff.getvalue()
    if image_path is not None:
        return Path(image_path).expanduser().read_bytes()
    raise ValueError("Tidak ada input gambar: butuh salah satu dari image / image_bytes / image_path.")


def generate_vlm(
    *,
    model_id: str,
    system_prompt: str = "",
    image: Optional[Image.Image] = None,
    image_bytes: Optional[bytes] = None,
    image_path: Optional[str] = None,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    gen_kwargs = dict(gen_kwargs or {})
    raw_image = _to_image_bytes(image=image, image_bytes=image_bytes, image_path=image_path)
    image_b64 = base64.b64encode(raw_image).decode("ascii")

    messages: list[Dict[str, Any]] = []
    # if system_prompt:
    #     messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": system_prompt,
            "images": [image_b64],
        }
    )

    options: Dict[str, Any] = {
        "num_predict": int(gen_kwargs.pop("max_new_tokens", VLM.max_new_tokens)),
        "temperature": float(gen_kwargs.pop("temperature", VLM.temperature)),
        "top_p": float(gen_kwargs.pop("top_p", VLM.top_p)),
        "top_k": int(gen_kwargs.pop("top_k", VLM.top_k)),
        "repeat_penalty": float(
            gen_kwargs.pop("repetition_penalty", VLM.repeat_penalty)
        ),
    }
    options.update(gen_kwargs)

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    response = SESSION.post(
        build_url(OLLAMA.chat_path),
        json=payload,
        timeout=OLLAMA.timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()

    try:
        content = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Ollama response: {data}") from exc
    return (content or "").strip()
