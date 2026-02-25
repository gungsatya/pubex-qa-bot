from __future__ import annotations

import logging
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image
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
        try:
            from docling.document_converter import ImageFormatOption
        except ImportError:
            ImageFormatOption = None
        from docling.models.inference_engines.vlm import VlmEngineType
        from docling.pipeline.vlm_pipeline import VlmPipeline
    except ImportError as exc:
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
        ),
    }
    if hasattr(InputFormat, "IMAGE"):
        image_format_option_cls = ImageFormatOption or PdfFormatOption
        format_options[InputFormat.IMAGE] = image_format_option_cls(
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        )
    return DocumentConverter(
        allowed_formats=format_options.keys(),
        format_options=format_options,
    )


def iter_pdf_pages_to_png_paths(
    *,
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    max_width_px: int | None,
) -> Iterable[tuple[int, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for page_no, png_bytes in pdf_to_png_images(pdf_path, dpi=dpi):
        png_path = out_dir / f"{page_no:03}.png"
        png_path.write_bytes(png_bytes)

        if max_width_px and max_width_px > 0:
            with Image.open(png_path) as im:
                if im.width > max_width_px:
                    scale = max_width_px / float(im.width)
                    new_h = max(1, int(im.height * scale))
                    resized = im.resize((max_width_px, new_h))
                    try:
                        resized.save(png_path, format="PNG", optimize=True)
                    finally:
                        resized.close()

        yield page_no, png_path


def _convert_image_to_markdown_timed(
    *,
    converter: Any,
    image_path: Path,
    page_break_placeholder: str,
) -> tuple[str, datetime, datetime, float]:
    start_at = datetime.now(timezone.utc)
    result = converter.convert(str(image_path))
    markdown = result.document.export_to_markdown(
        page_break_placeholder=page_break_placeholder,
    ) or ""
    end_at = datetime.now(timezone.utc)
    convert_seconds = max((end_at - start_at).total_seconds(), 0.0)

    cleaned_md = markdown.strip()
    if page_break_placeholder and page_break_placeholder in cleaned_md:
        chunks = [chunk.strip() for chunk in cleaned_md.split(page_break_placeholder)]
        cleaned_md = "\n\n".join(chunk for chunk in chunks if chunk).strip()

    return cleaned_md, start_at, end_at, convert_seconds


def _store_slide(
    *,
    session: Any,
    document_id: str,
    slide_no: int,
    total_pages: int,
    pdf_path: Path,
    content_md: str,
    image_path: Path,
    image_dpi: int,
    image_max_width_px: int | None,
    model: str,
    docling_base_url: str,
    convert_seconds: float,
    status: str,
    error: str | None,
    note: str | None,
    start_at: datetime,
    end_at: datetime,
) -> None:
    slide_metadata = {
        "slide_no": slide_no,
        "total_pages": total_pages,
        "file_path": str(pdf_path),
        "docling_preset": model,
        "docling_base_url": docling_base_url,
        "image_mime": "image/png",
        "image_dpi": image_dpi,
        "convert_seconds": convert_seconds,
        "status": status,
    }
    if image_max_width_px is not None:
        slide_metadata["image_max_width_px"] = image_max_width_px
    if error:
        slide_metadata["error"] = error
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
    image_max_width_px: int | None,
    text_model: str,
    note: str | None = None,
    overwrite_mode: str = "document",
    update_doc_status: bool = True,
) -> bool:
    total_pdf_pages = count_pdf_pages(pdf_path)
    if total_pdf_pages == 0:
        logger.warning("PDF kosong: %s", pdf_path)
        return False

    success_all = True
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
        doc.file_md_path = None

        processed_slides = 0
        try:
            for slide_no, image_path in iter_pdf_pages_to_png_paths(
                pdf_path=pdf_path,
                out_dir=slide_dir,
                dpi=image_dpi,
                max_width_px=image_max_width_px,
            ):
                processed_slides += 1
                logger.info(
                    "Mulai proses slide %s/%s doc_id=%s (%s)",
                    slide_no,
                    total_pdf_pages,
                    document_id,
                    pdf_path.name,
                )
                try:
                    content_md, start_at, end_at, convert_seconds = _convert_image_to_markdown_timed(
                        converter=converter,
                        image_path=image_path,
                        page_break_placeholder=page_break_placeholder,
                    )
                    _store_slide(
                        session=session,
                        document_id=document_id,
                        slide_no=slide_no,
                        total_pages=total_pdf_pages,
                        pdf_path=pdf_path,
                        content_md=content_md,
                        image_path=image_path,
                        image_dpi=image_dpi,
                        image_max_width_px=image_max_width_px,
                        model=model,
                        docling_base_url=docling_base_url,
                        convert_seconds=convert_seconds,
                        status="SUCCESS",
                        error=None,
                        note=note,
                        start_at=start_at,
                        end_at=end_at,
                    )
                    session.commit()
                except Exception as exc:  # noqa: BLE001
                    success_all = False
                    logger.exception(
                        "Gagal convert image->markdown doc_id=%s slide=%s: %s",
                        document_id,
                        slide_no,
                        exc,
                    )
                    session.rollback()
                    now = datetime.now(timezone.utc)
                    try:
                        _store_slide(
                            session=session,
                            document_id=document_id,
                            slide_no=slide_no,
                            total_pages=total_pdf_pages,
                            pdf_path=pdf_path,
                            content_md="",
                            image_path=image_path,
                            image_dpi=image_dpi,
                            image_max_width_px=image_max_width_px,
                            model=model,
                            docling_base_url=docling_base_url,
                            convert_seconds=0.0,
                            status="FAILED",
                            error=str(exc),
                            note=note,
                            start_at=now,
                            end_at=now,
                        )
                        session.commit()
                    except Exception as save_exc:  # noqa: BLE001
                        logger.exception(
                            "Gagal simpan slide gagal doc_id=%s slide=%s: %s",
                            document_id,
                            slide_no,
                            save_exc,
                        )
                        session.rollback()
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Gagal ekstraksi page image PDF doc_id=%s: %s",
                document_id,
                exc,
            )
            success_all = False
            session.rollback()

        if processed_slides != total_pdf_pages:
            logger.warning(
                "Jumlah slide terproses (%s) berbeda dengan total PDF pages (%s) doc_id=%s",
                processed_slides,
                total_pdf_pages,
                document_id,
            )
            success_all = False

        if update_doc_status:
            doc.status = (
                DocumentStatusEnum.PARSED.id
                if success_all
                else DocumentStatusEnum.FAILED_PARSED.id
            )
        session.commit()

    return success_all


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
    image_max_width_px = DOCLING.image_max_width_px

    logger.info(
        "Konfigurasi ingestion: docling_preset=%s, docling_api_base_url=%s, image_dpi=%s, image_max_width_px=%s, llm_model=%s",
        model,
        DOCLING.api_base_url,
        DOCLING.image_dpi,
        image_max_width_px,
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
            image_max_width_px=image_max_width_px,
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
