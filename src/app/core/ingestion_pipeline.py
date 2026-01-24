from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF
import requests
from PIL import Image
from sqlalchemy import select, func

from app.db.models import Document, Slide
from app.db.session import get_session
from src.data.enums import DocumentStatusEnum

try:
    from llama_index.core.prompts import PromptTemplate
except ImportError as exc:  # pragma: no cover - guarded for runtime safety
    raise RuntimeError(
        "LlamaIndex tidak ditemukan. Install dengan 'pip install llama-index'."
    ) from exc

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = os.getenv(
    "OLLAMA_URL", "http://localhost:11434/v1/chat/completions"
)
DEFAULT_OLLAMA_MODEL = os.getenv(
    "OLLAMA_VLM_MODEL", "qwen3-vl:2b-instruct-q4_K_M"
)
DEFAULT_DPI = int(os.getenv("INGESTION_PDF_DPI", "144"))

PROMPT_TEMPLATE = PromptTemplate(
    "Anda adalah financial analyst.\n"
    "Gambar berikut adalah satu slide dari dokumen keuangan.\n"
    "Slide {slide_no} dari {total_pages}.\n\n"
    "Aturan:\n"
    "- Fokus hanya pada konten yang terlihat pada gambar.\n"
    "- Jangan menebak atau menambah asumsi di luar gambar.\n"
    "- Jika teks, angka, atau grafik tidak terbaca, tulis 'Tidak terbaca'.\n"
    "- Jika ada tabel atau grafik, ubah ke tabel Markdown.\n"
    "- Jika ada beberapa tabel, beri judul singkat tiap tabel.\n\n"
    "Format keluaran (Markdown, bahasa Indonesia):\n"
    "### Ringkasan\n"
    "- (1-3 kalimat ringkas)\n\n"
    "### Poin Utama\n"
    "- ...\n\n"
    "### Tabel\n"
    "- (gunakan tabel Markdown jika ada tabel/grafik)\n"
)


def _pdf_to_images(pdf_path: Path, dpi: int) -> List[Tuple[int, bytes]]:
    doc = fitz.open(pdf_path)
    images: List[Tuple[int, bytes]] = []

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        images.append((page_index + 1, img_bytes))

    doc.close()
    return images


def _image_bytes_to_data_url(img_bytes: bytes) -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _call_ollama_vlm(
    *,
    ollama_url: str,
    model: str,
    prompt: str,
    image_data_url: str,
    temperature: float = 0.2,
    timeout: int = 300,
) -> str:
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }

    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response from Ollama: {data}") from exc


def _validate_image(img_bytes: bytes, page_no: int) -> bool:
    try:
        Image.open(io.BytesIO(img_bytes)).verify()
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Slide %s: image tidak valid, dilewati.", page_no)
        return False


def _get_downloaded_documents(limit: int | None) -> Iterable[Document]:
    with get_session() as session:
        stmt = select(Document).where(
            Document.status == DocumentStatusEnum.DOWNLOADED.id
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = session.execute(stmt)
        docs = result.scalars().all()
        return docs


def _count_existing_slides(session, document_id: str) -> int:
    stmt = select(func.count(Slide.id)).where(Slide.document_id == document_id)
    return int(session.execute(stmt).scalar() or 0)


def _process_document(
    *,
    document_id: str,
    pdf_path: Path,
    ollama_url: str,
    model: str,
    dpi: int,
) -> bool:
    images = _pdf_to_images(pdf_path, dpi=dpi)
    total_pages = len(images)
    if total_pages == 0:
        logger.warning("PDF kosong: %s", pdf_path)
        return False

    success = True
    with get_session() as session:
        doc = session.get(Document, document_id)
        if doc is None:
            logger.warning("Dokumen tidak ditemukan: %s", document_id)
            return False

        existing = _count_existing_slides(session, document_id)
        if existing > 0:
            logger.info("Skip %s: slides sudah ada (%s).", document_id, existing)
            return True

        for slide_no, img_bytes in images:
            if not _validate_image(img_bytes, slide_no):
                success = False
                continue

            data_url = _image_bytes_to_data_url(img_bytes)
            prompt = PROMPT_TEMPLATE.format(
                slide_no=slide_no,
                total_pages=total_pages,
            )

            try:
                content_md = _call_ollama_vlm(
                    ollama_url=ollama_url,
                    model=model,
                    prompt=prompt,
                    image_data_url=data_url,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Gagal memanggil VLM untuk doc_id=%s slide=%s: %s",
                    document_id,
                    slide_no,
                    exc,
                )
                success = False
                continue

            slide = Slide(
                document_id=document_id,
                content_text=content_md.strip(),
                content_base_64=base64.b64encode(img_bytes).decode("ascii"),
                slide_metadata={
                    "slide_no": slide_no,
                    "total_pages": total_pages,
                    "file_path": str(pdf_path),
                    "image_mime": "image/png",
                    "dpi": dpi,
                },
            )
            session.add(slide)

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
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    dpi: int = DEFAULT_DPI,
) -> None:
    docs = _get_downloaded_documents(limit)
    if not docs:
        logger.info("Tidak ada dokumen berstatus downloaded.")
        print("Tidak ada dokumen berstatus downloaded.")
        return

    for doc in docs:
        pdf_path = Path(doc.file_path)
        if not pdf_path.is_file():
            logger.error("File PDF tidak ditemukan: %s", pdf_path)
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
            ollama_url=ollama_url,
            model=model,
            dpi=dpi,
        )
