from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from sqlalchemy import select, func

from app.utils.document_utils import pdf_to_images, count_pdf_pages
from app.utils.image_utils import downscale_png, validate_image_bytes
from app.db.models import Document, Slide
from app.db.session import get_session
from app.core.ollama_vlm import generate_vlm
from app.config import LLM, VLM
from src.data.enums import DocumentStatusEnum

logger = logging.getLogger(__name__)

PROMPT_VLM_SINGLE = """You will be provided with an image of a PDF page or slide.

Your task is to convert ALL visible content into structured Markdown format.

You MUST act as a visual transcription and formatting system, NOT as an analyst.

====================================
PRIMARY OBJECTIVE
====================================

- Extract ONLY text, numbers, tables, and labels that are clearly visible.
- Preserve original meaning and wording.
- Do NOT summarize, interpret, explain, or rephrase.
- Do NOT add any new information.
- Do NOT guess missing or unclear content.
- If content is unreadable -> omit it.

====================================
GENERAL RULES
====================================

1. All output MUST be in valid Markdown.
2. Maintain original language as shown.
3. Keep spelling, capitalization, and numeric values unchanged.
4. Do NOT add commentary, opinions, or conclusions.
5. Do NOT mention document format (PDF, slide, page, image, etc).
6. Do NOT describe layout or positions.

====================================
TITLE HANDLING
====================================

If a clear title exists:

- Write it as:

## {Title}

If no title exists:

- Do NOT create one.

====================================
TEXT & LISTS
====================================

- Paragraphs -> normal Markdown text
- Bullet points -> Markdown lists
- Numbered items -> ordered lists

Example:

- Item A
- Item B

1. Step One
2. Step Two

====================================
TABLES
====================================

If content appears in tabular form, convert it into Markdown tables.

Example:

| Column A | Column B |
|----------|----------|
| Value 1  | Value 2  |

====================================
CHARTS / GRAPHS
====================================

If charts or graphs are present:

- Convert all visible numeric data into tables OR lists.
- Include labels, units, and legends if visible.
- Do NOT interpret trends.

Example (Table):

| Year | Revenue |
|------|----------|
| 2023 | 120      |
| 2024 | 150      |

Example (List):

- 2023: 120
- 2024: 150

====================================
INFOGRAPHICS / MAPS / DIAGRAMS
====================================

If infographics, maps, or diagrams are present:

- Convert visible data into structured lists or tables.
- Include names, markers, labels, and values.
- Do NOT explain meaning.

Example (List):

- Factory: Jakarta
- Warehouse: Surabaya
- Port: Belawan

====================================
IMAGES WITH TEXT
====================================

- Extract all readable text inside images.
- Ignore decorative or unreadable elements.

====================================
OUTPUT FORMAT
====================================

If title exists:

## {Title}

{Markdown content}

If no title:

{Markdown content only}

====================================
STRICT PROHIBITIONS
====================================

No interpretation
No summarization
No rewording
No hallucination
No inference
No explanation

Only transcribe and format what is visible.
"""

def _build_pubex_prompts() -> tuple[str, str]:
    system_prompt = PROMPT_VLM_SINGLE
    prompt = ""
    return system_prompt, prompt


def _prepare_vlm_image_bytes(image_bytes: bytes) -> bytes:
    return downscale_png(image_bytes, max_w=VLM.image_max_w)


def _call_vlm(
    *,
    model: str,
    system_prompt: str,
    image_bytes: bytes,
    temperature: float = VLM.temperature,
) -> str:
    try:
        image_bytes = _prepare_vlm_image_bytes(image_bytes)
        return generate_vlm(
            model_id=model,
            system_prompt=system_prompt.strip(),
            image_bytes=image_bytes,
            gen_kwargs={
                "temperature": temperature,
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"VLM error (Ollama): {exc}") from exc


def _is_kosong_marker(content: str) -> bool:
    return (content or "").strip() == "[[KOSONG]]"


def _extract_pubex_single_prompt(
    *,
    model: str,
    image_bytes: bytes,
) -> str:
    system_prompt, prompt = _build_pubex_prompts()
    raw_result = _call_vlm(
        model=model,
        system_prompt=system_prompt,
        image_bytes=image_bytes,
    )
    if _is_kosong_marker(raw_result):
        return ""
    return raw_result


def _store_slide(
    *,
    session,
    document_id: str,
    slide_no: int,
    total_pages: int,
    pdf_path: Path,
    slide_dir: Path,
    image_bytes: bytes,
    content_md: str,
    model: str,
    text_model: str,
    dpi: int,
    note: str | None,
    extractor_method: str,
    start_at: datetime,
    end_at: datetime,
) -> None:
    image_path = slide_dir / f"{slide_no:03}.png"
    image_path.write_bytes(image_bytes)

    slide_metadata = {
        "slide_no": slide_no,
        "total_pages": total_pages,
        "file_path": str(pdf_path),
        "image_mime": "image/png",
        "dpi": dpi,
        "vlm_model": model,
        "llm_model": text_model,
        "extractor_method": extractor_method,
    }
    if note:
        slide_metadata["ingestion_note"] = note

    slide = Slide(
        document_id=document_id,
        content_text=content_md,
        image_path=str(image_path),
        slide_metadata=slide_metadata,
        ingestion_start_at=start_at,
        ingestion_end_at=end_at,
    )
    session.add(slide)


def _get_downloaded_documents(
    limit: int | None,
    document_ids: Sequence[str] | None = None,
) -> Iterable[Document]:
    with get_session() as session:
        stmt = select(Document)
        if document_ids:
            stmt = stmt.where(Document.id.in_(document_ids))
        else:
            stmt = stmt.where(
                Document.status.in_(
                    [
                        DocumentStatusEnum.DOWNLOADED.id,
                        DocumentStatusEnum.FAILED_PARSED.id,
                    ]
                )
            )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = session.execute(stmt)
        docs = result.scalars().all()
        return docs


def _count_existing_slides(
    session,
    document_id: str,
    model: str | None = None,
) -> int:
    stmt = select(func.count(Slide.id)).where(Slide.document_id == document_id)
    if model:
        stmt = stmt.where(Slide.slide_metadata["vlm_model"].astext == model)
    return int(session.execute(stmt).scalar() or 0)


def _delete_existing_slides(
    session,
    document_id: str,
    model: str | None = None,
) -> int:
    query = session.query(Slide).filter(Slide.document_id == document_id)
    if model:
        query = query.filter(Slide.slide_metadata["vlm_model"].astext == model)
    deleted = query.delete(synchronize_session=False)
    return int(deleted or 0)


def _infer_document_type(doc: Document) -> str:
    collection_metadata = doc.collection.collection_metadata if doc.collection else None
    if isinstance(collection_metadata, dict):
        doc_type = collection_metadata.get("type")
        if isinstance(doc_type, str) and doc_type.strip():
            return doc_type.strip().lower()
    if isinstance(doc.document_metadata, dict):
        doc_type = doc.document_metadata.get("type")
        if isinstance(doc_type, str) and doc_type.strip():
            return doc_type.strip().lower()
    return "unknown"


def _process_document(
    *,
    document_id: str,
    pdf_path: Path,
    model: str,
    text_model: str,
    dpi: int,
    note: str | None = None,
    overwrite_mode: str = "document",
    update_doc_status: bool = True,
) -> bool:
    total_pages = count_pdf_pages(pdf_path)
    if total_pages == 0:
        logger.warning("PDF kosong: %s", pdf_path)
        return False

    success = True
    with get_session() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            logger.warning("Dokumen tidak ditemukan: %s", document_id)
            return False
        document_type = _infer_document_type(doc)

        if overwrite_mode not in {"document", "model", "none"}:
            raise ValueError(
                "overwrite_mode harus salah satu dari: document, model, none"
            )

        if overwrite_mode == "document":
            existing = _count_existing_slides(session, document_id)
            if existing > 0:
                logger.info(
                    "Overwrite %s: hapus %s slide lama sebelum ingest ulang.",
                    document_id,
                    existing,
                )
                _delete_existing_slides(session, document_id)
                session.commit()
        elif overwrite_mode == "model":
            existing = _count_existing_slides(session, document_id, model)
            if existing > 0:
                logger.info(
                    "Overwrite %s (model=%s): hapus %s slide lama sebelum ingest ulang.",
                    document_id,
                    model,
                    existing,
                )
                _delete_existing_slides(session, document_id, model)
                session.commit()

        slide_dir = pdf_path.parent / "slides" / document_id
        slide_dir.mkdir(parents=True, exist_ok=True)

        for slide_no, img_bytes in pdf_to_images(pdf_path, dpi=dpi):
            logger.info(
                "Mulai proses slide %s/%s doc_id=%s (%s)",
                slide_no,
                total_pages,
                document_id,
                pdf_path.name,
            )
            if not validate_image_bytes(img_bytes, slide_no):
                logger.warning(
                    "Validasi image gagal untuk slide %s/%s doc_id=%s",
                    slide_no,
                    total_pages,
                    document_id,
                )
                success = False
                continue

            start_at = datetime.now(timezone.utc)

            try:
                if document_type == "pubex":
                    raw_md = _extract_pubex_single_prompt(
                        model=model,
                        image_bytes=img_bytes,
                    )
                    content_md = raw_md
                    extractor_method = "pubex_single_prompt_ollama"
                else:
                    content_md = ""
                    extractor_method = "unsupported_document_type"
                end_at = datetime.now(timezone.utc)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Gagal memanggil VLM untuk doc_id=%s slide=%s: %s",
                    document_id,
                    slide_no,
                    exc,
                )
                success = False
                continue

            _store_slide(
                session=session,
                document_id=document_id,
                slide_no=slide_no,
                total_pages=total_pages,
                pdf_path=pdf_path,
                slide_dir=slide_dir,
                image_bytes=img_bytes,
                content_md=content_md,
                model=model,
                text_model=text_model,
                dpi=dpi,
                note=note,
                extractor_method=extractor_method,
                start_at=start_at,
                end_at=end_at,
            )
            logger.info(
                "Selesai proses slide %s/%s doc_id=%s",
                slide_no,
                total_pages,
                document_id,
            )

        if update_doc_status:
            doc.status = (
                DocumentStatusEnum.PARSED.id
                if success
                else DocumentStatusEnum.FAILED_PARSED.id
            )
        session.commit()

    return success


def run_ingestion(
    *,
    limit: int | None = None,
    document_ids: Sequence[str] | None = None,
    note: str | None = None,
    model: str = VLM.model,
    text_model: str = LLM.model,
    dpi: int = VLM.ingestion_pdf_dpi,
    overwrite_mode: str = "document",
    update_doc_status: bool = True,
) -> None:
    document_ids = [
        doc_id.strip()
        for doc_id in (document_ids or [])
        if doc_id and doc_id.strip()
    ]
    note = note.strip() if note else None

    if document_ids:
        # Explicit document selection should not mutate document lifecycle state.
        update_doc_status = False

    logger.info(
        "Konfigurasi ingestion: vlm_model=%s, llm_model=%s",
        model,
        text_model,
    )

    start_at = datetime.now(timezone.utc)
    docs = _get_downloaded_documents(limit, document_ids=document_ids)
    if not docs:
        if document_ids:
            logger.info("Tidak ada dokumen untuk id yang diberikan: %s", ", ".join(document_ids))
            print("Tidak ada dokumen untuk id yang diberikan.")
        else:
            logger.info("Tidak ada dokumen berstatus downloaded.")
            print("Tidak ada dokumen berstatus downloaded.")
        return

    for doc in docs:
        pdf_path = Path(doc.file_path)
        if not pdf_path.is_file():
            logger.error("File PDF tidak ditemukan: %s", pdf_path)
            if update_doc_status:
                with get_session() as session:
                    db_doc = session.get(Document, doc.id)
                    if db_doc:
                        db_doc.status = DocumentStatusEnum.FAILED_PARSED.id
                        session.commit()
            continue

        logger.info("Memproses doc_id=%s (%s)", doc.id, pdf_path.name)
        _process_document(
            document_id=doc.id,
            pdf_path=pdf_path,
            model=model,
            text_model=text_model,
            dpi=dpi,
            note=note,
            overwrite_mode=overwrite_mode,
            update_doc_status=update_doc_status,
        )

    end_at = datetime.now(timezone.utc)
    total_seconds = (end_at - start_at).total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    logger.info("Durasi ingestion total: %s menit %s detik", minutes, seconds)
    print(f"Durasi ingestion total: {minutes} menit {seconds} detik")
