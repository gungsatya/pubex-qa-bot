from __future__ import annotations

from typing import Any, Dict
from typing import Optional

from app.config import LLM, OLLAMA
from app.core.ollama_client import SESSION, build_url


def generate_llm(
    *,
    model_id: str,
    system_prompt: str = "",
    prompt: str,
    gen_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    gen_kwargs = dict(gen_kwargs or {})

    messages: list[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    options: Dict[str, Any] = {
        "num_predict": int(gen_kwargs.pop("max_new_tokens", LLM.max_new_tokens)),
        "temperature": float(gen_kwargs.pop("temperature", LLM.temperature)),
        "top_p": float(gen_kwargs.pop("top_p", LLM.top_p)),
        "top_k": int(gen_kwargs.pop("top_k", LLM.top_k)),
        "repeat_penalty": float(
            gen_kwargs.pop("repetition_penalty", LLM.repeat_penalty)
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
