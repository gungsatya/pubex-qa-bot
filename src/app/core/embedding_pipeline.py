from __future__ import annotations

import logging
import math
import os
from typing import Iterable, List, Tuple

import requests
from sqlalchemy import select, func

from app.db.models import Slide, Document, EMBEDDING_DIM
from app.db.session import get_session
from app.config import DEFAULT_LLAMA_CPP_BASE_URL, DEFAULT_LLAMA_CPP_TIMEOUT
from src.data.enums import DocumentStatusEnum

logger = logging.getLogger(__name__)
SESSION = requests.Session()

DEFAULT_EMBED_MODEL = os.getenv("LLAMA_CPP_EMBED_MODEL", "embedding")
DEFAULT_LLAMA_CPP_EMBED_BASE_URL = os.getenv(
    "LLAMA_CPP_EMBED_BASE_URL", DEFAULT_LLAMA_CPP_BASE_URL
)
DEFAULT_EMBED_BATCH = int(os.getenv("EMBED_BATCH_SIZE", "10"))


def _fetch_embeddings(
    *,
    texts: List[str],
    model_name: str,
    base_url: str,
    timeout: int,
) -> List[List[float]]:
    payload = {
        "model": model_name,
        "input": texts,
        "encoding_format": "float",
    }
    response = SESSION.post(
        f"{base_url.rstrip('/')}/v1/embeddings",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    items = data.get("data")
    if not isinstance(items, list):
        raise RuntimeError(f"Unexpected embeddings response: {data}")
    try:
        items_sorted = sorted(items, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items_sorted]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid embeddings payload: {data}") from exc


def _get_slides_to_embed(limit: int | None) -> List[Slide]:
    with get_session() as session:
        stmt = (
            select(Slide)
            .where(Slide.content_text.isnot(None))
            .where(Slide.content_text_vector.is_(None))
            .order_by(Slide.created_at.asc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = session.execute(stmt)
        return result.scalars().all()


def _chunk(items: List[Slide], size: int) -> Iterable[List[Slide]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _prepare_texts(batch: List[Slide]) -> Tuple[List[Slide], List[str]]:
    pairs: List[Tuple[Slide, str]] = []
    for slide in batch:
        text = (slide.content_text or "").strip()
        if text:
            pairs.append((slide, text))
    if not pairs:
        return [], []
    slides, texts = zip(*pairs)
    return list(slides), list(texts)


def _is_valid_vector(vector: List[float]) -> bool:
    if not vector:
        return False
    for value in vector:
        if not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)):
            return False
    return True


def _update_document_status(session, document_ids: List[str]) -> None:
    if not document_ids:
        return
    stmt = (
        select(Document.id)
        .where(Document.id.in_(document_ids))
        .where(
            select(func.count(Slide.id))
            .where(Slide.document_id == Document.id)
            .where(Slide.content_text_vector.is_(None))
            .correlate(Document)
            .scalar_subquery()
            == 0
        )
    )
    ready_doc_ids = [row[0] for row in session.execute(stmt).all()]
    if not ready_doc_ids:
        return
    session.query(Document).filter(Document.id.in_(ready_doc_ids)).update(
        {Document.status: DocumentStatusEnum.EMBEDDED.id},
        synchronize_session=False,
    )


def run_embedding_pipeline(
    *,
    limit: int | None = None,
    model_name: str = DEFAULT_EMBED_MODEL,
    base_url: str = DEFAULT_LLAMA_CPP_EMBED_BASE_URL,
    batch_size: int = DEFAULT_EMBED_BATCH,
) -> None:
    slides = _get_slides_to_embed(limit)
    if not slides:
        logger.info("Tidak ada slide untuk di-embed.")
        print("Tidak ada slide untuk di-embed.")
        return

    for batch in _chunk(slides, batch_size):
        slides_with_text, texts = _prepare_texts(batch)
        if not texts:
            logger.warning("Batch kosong: tidak ada content_text.")
            continue

        embeddings: List[List[float]] = []
        try:
            embeddings = _fetch_embeddings(
                texts=texts,
                model_name=model_name,
                base_url=base_url,
                timeout=DEFAULT_LLAMA_CPP_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Gagal generate embedding batch, fallback per item: %s", exc
            )
            embeddings = []
            for text in texts:
                try:
                    embeddings.extend(
                        _fetch_embeddings(
                            texts=[text],
                            model_name=model_name,
                            base_url=base_url,
                            timeout=DEFAULT_LLAMA_CPP_TIMEOUT,
                        )
                    )
                except Exception as per_exc:  # noqa: BLE001
                    logger.exception("Gagal embed item, dilewati: %s", per_exc)
                    embeddings.append([])

        if len(embeddings) != len(slides_with_text):
            logger.error(
                "Jumlah embedding tidak cocok (got=%s, expected=%s).",
                len(embeddings),
                len(slides_with_text),
            )
            continue

        doc_ids: List[str] = []
        with get_session() as session:
            for slide, vector in zip(slides_with_text, embeddings):
                if not _is_valid_vector(vector):
                    logger.warning(
                        "Embedding tidak valid untuk slide %s (NaN/inf/kosong).",
                        slide.id,
                    )
                    continue
                if len(vector) != EMBEDDING_DIM:
                    logger.error(
                        "Dimensi embedding tidak sesuai untuk slide %s (got=%s).",
                        slide.id,
                        len(vector),
                    )
                    continue
                db_slide = session.get(Slide, slide.id)
                if not db_slide:
                    continue
                db_slide.content_text_vector = vector
                doc_ids.append(db_slide.document_id)

            _update_document_status(session, doc_ids)
            session.commit()
