from __future__ import annotations

import logging
from importlib import metadata as importlib_metadata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from sqlalchemy import func, select

from app.db.models import Document, Slide
from app.db.session import get_session
from app.config import DOCLING, LLM
from app.utils.document_utils import count_pdf_pages, pdf_to_png_images
from src.data.enums import DocumentStatusEnum

logger = logging.getLogger(__name__)

DEFAULT_PAGE_BREAK_PLACEHOLDER = "<!-- page break -->"
EXTRACTOR_METHOD = "docling_markdown_pagebreak"


def _get_docling_version() -> str:
    try:
        return importlib_metadata.version("docling")
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def _raise_docling_incompatible(exc: Exception) -> None:
    version = _get_docling_version()
    raise RuntimeError(
        "Docling terpasang tetapi tidak kompatibel dengan pipeline ini "
        f"(version={version}). "
        "Dibutuhkan API Docling VLM (`docling.datamodel.pipeline_options`, "
        "`docling.pipeline.vlm_pipeline`). "
        "Upgrade dependency: `pip install \"docling>=2,<3\"`."
    ) from exc


def _load_docling_converter(
    *,
    base_url: str,
    timeout_seconds: int,
    preset: str,
    batch_size: int,
) -> Any:
    try:
        try:
            from docling.datamodel.base_models import InputFormat
        except ImportError:
            from docling.datamodel.document import InputFormat  # fallback beberapa versi
        from docling.datamodel.pipeline_options import (
            VlmConvertOptions,
            VlmPipelineOptions,
        )
        from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.models.inference_engines.vlm import VlmEngineType
        from docling.pipeline.vlm_pipeline import VlmPipeline
    except ImportError as exc:
        # Kasus umum:
        # - docling belum terpasang
        # - docling lama (mis. 1.x) tanpa API VLM
        # - import side-effect gagal karena dependency lama
        if _get_docling_version() == "not_installed":
            raise RuntimeError(
                "Docling belum terpasang. Jalankan `pip install -r requirements.txt`."
            ) from exc
        _raise_docling_incompatible(exc)

    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    convert_options = VlmConvertOptions.from_preset(
        preset,
        engine_options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API,
            url=endpoint,
            timeout=timeout_seconds,
        ),
    )
    convert_options.batch_size = max(1, int(batch_size))

    pipeline_options = VlmPipelineOptions(
        vlm_options=convert_options,
        enable_remote_services=True,
    )
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        )
    }
    return DocumentConverter(
        allowed_formats=format_options.keys(),
        format_options=format_options,
    )


def _convert_pdf_to_markdown(
    *,
    converter: Any,
    pdf_path: Path,
    page_break_placeholder: str,
) -> str:
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown(
        page_break_placeholder=page_break_placeholder,
    )


def _split_markdown_by_page_break(
    markdown: str,
    page_break_placeholder: str,
) -> list[str]:
    if page_break_placeholder and page_break_placeholder in markdown:
        chunks = markdown.split(page_break_placeholder)
    else:
        chunks = [markdown]
    # Pertahankan urutan halaman, termasuk halaman kosong.
    return [chunk.strip() for chunk in chunks] or [""]


def _align_markdown_pages(
    pages_md: list[str],
    target_page_count: int,
) -> list[str]:
    """
    Samakan jumlah chunk markdown dengan jumlah halaman PDF agar 1 slide = 1 halaman image.
    """
    if target_page_count <= 0:
        return pages_md or [""]

    current_count = len(pages_md)
    if current_count == target_page_count:
        return pages_md

    if current_count < target_page_count:
        return pages_md + [""] * (target_page_count - current_count)

    # current_count > target_page_count:
    # simpan konten berlebih ke halaman terakhir agar data tidak hilang.
    if target_page_count == 1:
        return ["\n\n".join(pages_md)]
    return pages_md[: target_page_count - 1] + ["\n\n".join(pages_md[target_page_count - 1 :])]


def _store_slide(
    *,
    session: Any,
    document_id: str,
    slide_no: int,
    total_pages: int,
    pdf_path: Path,
    page_break_placeholder: str,
    document_markdown_path: Path,
    slide_markdown_path: Path,
    content_md: str,
    image_path: Path,
    image_dpi: int,
    model: str,
    docling_base_url: str,
    text_model: str,
    note: str | None,
    start_at: datetime,
    end_at: datetime,
) -> None:
    slide_metadata = {
        "slide_no": slide_no,
        "total_pages": total_pages,
        "file_path": str(pdf_path),
        "page_break_placeholder": page_break_placeholder,
        "docling_preset": model,
        "docling_base_url": docling_base_url,
        "image_mime": "image/png",
        "image_dpi": image_dpi,
        "llm_model": text_model,
        "extractor_method": EXTRACTOR_METHOD,
        "document_markdown_path": str(document_markdown_path),
        "slide_markdown_path": str(slide_markdown_path),
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
    session: Any,
    document_id: str,
    model: str | None = None,
) -> int:
    stmt = select(func.count(Slide.id)).where(Slide.document_id == document_id)
    if model:
        stmt = stmt.where(Slide.slide_metadata["docling_preset"].astext == model)
    return int(session.execute(stmt).scalar() or 0)


def _delete_existing_slides(
    session: Any,
    document_id: str,
    model: str | None = None,
) -> int:
    query = session.query(Slide).filter(Slide.document_id == document_id)
    if model:
        query = query.filter(Slide.slide_metadata["docling_preset"].astext == model)
    deleted = query.delete(synchronize_session=False)
    return int(deleted or 0)


def _process_document(
    *,
    document_id: str,
    pdf_path: Path,
    converter: Any,
    page_break_placeholder: str,
    model: str,
    docling_base_url: str,
    image_dpi: int,
    text_model: str,
    note: str | None = None,
    overwrite_mode: str = "document",
    update_doc_status: bool = True,
) -> bool:
    total_pdf_pages = count_pdf_pages(pdf_path)
    if total_pdf_pages == 0:
        logger.warning("PDF kosong: %s", pdf_path)
        return False

    success = True
    with get_session() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            logger.warning("Dokumen tidak ditemukan: %s", document_id)
            return False

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

        conversion_start_at = datetime.now(timezone.utc)
        try:
            markdown = _convert_pdf_to_markdown(
                converter=converter,
                pdf_path=pdf_path,
                page_break_placeholder=page_break_placeholder,
            ) or ""
            conversion_end_at = datetime.now(timezone.utc)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Gagal convert PDF ke markdown via Docling untuk doc_id=%s: %s",
                document_id,
                exc,
            )
            success = False
            conversion_end_at = datetime.now(timezone.utc)
            markdown = ""

        if success:
            try:
                document_markdown_path = slide_dir / "document.md"
                document_markdown_path.write_text(markdown, encoding="utf-8")
                pages_md = _split_markdown_by_page_break(
                    markdown,
                    page_break_placeholder=page_break_placeholder,
                )
                total_pages = total_pdf_pages
                if len(pages_md) != total_pdf_pages:
                    logger.warning(
                        "Jumlah halaman dari markdown (%s) berbeda dengan PDF (%s) doc_id=%s",
                        len(pages_md),
                        total_pdf_pages,
                        document_id,
                    )
                pages_md = _align_markdown_pages(
                    pages_md=pages_md,
                    target_page_count=total_pages,
                )

                page_images: dict[int, bytes] = dict(
                    pdf_to_png_images(pdf_path, dpi=image_dpi)
                )
                if len(page_images) != total_pages:
                    logger.warning(
                        "Jumlah image halaman (%s) berbeda dengan PDF (%s) doc_id=%s",
                        len(page_images),
                        total_pages,
                        document_id,
                    )

                conversion_duration_seconds = max(
                    (conversion_end_at - conversion_start_at).total_seconds(),
                    0.0,
                )
                per_slide_seconds = (
                    conversion_duration_seconds / total_pages
                    if total_pages > 0
                    else 0.0
                )

                for slide_no, page_md in enumerate(
                    pages_md[:total_pages],
                    start=1,
                ):
                    logger.info(
                        "Mulai simpan slide %s/%s doc_id=%s (%s)",
                        slide_no,
                        total_pages,
                        document_id,
                        pdf_path.name,
                    )
                    slide_markdown_path = slide_dir / f"{slide_no:03}.md"
                    slide_markdown_path.write_text(page_md, encoding="utf-8")
                    image_path = slide_dir / f"{slide_no:03}.png"

                    page_image = page_images.get(slide_no)
                    if page_image is None:
                        logger.error(
                            "Image halaman %s tidak ditemukan doc_id=%s. Slide dilewati.",
                            slide_no,
                            document_id,
                        )
                        success = False
                        continue
                    image_path.write_bytes(page_image)

                    slide_start_at = conversion_start_at + timedelta(
                        seconds=per_slide_seconds * (slide_no - 1)
                    )
                    slide_end_at = slide_start_at + timedelta(seconds=per_slide_seconds)

                    _store_slide(
                        session=session,
                        document_id=document_id,
                        slide_no=slide_no,
                        total_pages=total_pages,
                        pdf_path=pdf_path,
                        page_break_placeholder=page_break_placeholder,
                        document_markdown_path=document_markdown_path,
                        slide_markdown_path=slide_markdown_path,
                        content_md=page_md,
                        image_path=image_path,
                        image_dpi=image_dpi,
                        model=model,
                        docling_base_url=docling_base_url,
                        text_model=text_model,
                        note=note,
                        start_at=slide_start_at,
                        end_at=slide_end_at,
                    )
                    logger.info(
                        "Selesai simpan slide %s/%s doc_id=%s",
                        slide_no,
                        total_pages,
                        document_id,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Gagal memproses hasil markdown untuk doc_id=%s: %s",
                    document_id,
                    exc,
                )
                success = False

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
    model: str = DOCLING.preset,
    text_model: str = LLM.model,
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

    page_break_placeholder = (
        DOCLING.page_break_placeholder.strip()
        if DOCLING.page_break_placeholder
        else DEFAULT_PAGE_BREAK_PLACEHOLDER
    )
    model = model.strip() or DOCLING.preset

    logger.info(
        "Konfigurasi ingestion: docling_preset=%s, docling_api_base_url=%s, image_dpi=%s, llm_model=%s",
        model,
        DOCLING.api_base_url,
        DOCLING.image_dpi,
        text_model,
    )

    try:
        converter = _load_docling_converter(
            base_url=DOCLING.api_base_url,
            timeout_seconds=DOCLING.api_timeout_seconds,
            preset=model,
            batch_size=DOCLING.batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gagal memuat DocumentConverter Docling: %s", exc)
        print(f"Gagal memuat DocumentConverter Docling: {exc}")
        return

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
            converter=converter,
            page_break_placeholder=page_break_placeholder,
            model=model,
            docling_base_url=DOCLING.api_base_url,
            image_dpi=DOCLING.image_dpi,
            text_model=text_model,
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
