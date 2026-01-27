from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
import json
import re
import tempfile
from typing import Iterable
from datetime import datetime, timezone

from ollama import chat
from sqlalchemy import select, func

from app.utils.document_utils import pdf_to_images
from app.utils.image_utils import validate_image_bytes
from app.db.models import Document, Slide
from app.db.session import get_session
from src.app.config import DEFAULT_DPI, DEFAULT_VLM_MODEL
from src.data.enums import DocumentStatusEnum

try:
    from llama_index.core.prompts import PromptTemplate
except ImportError as exc:  # pragma: no cover - guarded for runtime safety
    raise RuntimeError(
        "LlamaIndex tidak ditemukan. Install dengan 'pip install llama-index'."
    ) from exc

logger = logging.getLogger(__name__)

PROMPT = PromptTemplate(
    "Anda adalah Analis Keuangan yang mengekstraksi isi slide untuk sistem Tanya Jawab berbasis RAG. "
    "Output harus faktual, self-contained, relevan finansial/bisnis, dan hanya berupa Markdown.\n\n"

    "ATURAN UTAMA:\n"
    "- Analisis hanya berdasarkan teks, angka, tabel, grafik, atau daftar yang terlihat.\n"
    "- Abaikan visual non-informatif: foto manusia, ekspresi, suasana, pakaian, ikon, dekorasi, warna, dan estetika.\n"
    "- Tidak boleh menebak, menambah, menyimpulkan, atau beropini.\n"
    "- Jika bagian tidak terbaca/tidak jelas → lewati tanpa menebak.\n"
    "- Dilarang memasukkan metadata atau JSON ke dalam output.\n\n"

    "OUTPUT MARKDOWN WAJIB:\n"
    "## Judul Slide\n"
    "- Jika ada judul di slide, gunakan apa adanya.\n"
    "- Jika tidak ada, buat frasa deskriptif ≤10 kata tanpa opini.\n\n"

    "### Ringkasan\n"
    "- Maksimal 7 kalimat (atau bullet).\n"
    "- Hanya informasi finansial/bisnis/operasional atau informasi presentasi.\n"
    "- Dilarang mendeskripsikan visual non-informatif.\n"
    "- Jika slide adalah cover/poster tanpa konten: tulis 'Slide pembuka. Tidak terdapat konten finansial atau bisnis.'\n\n"

    "### Konten\n"
    "- Bagian 'Konten' wajib berisi pengulangan isi utama slide (teks, angka, tabel, grafik, daftar) dalam bentuk Markdown.\n"
    "- Jika tidak ada konten finansial/bisnis yang terbaca, tulis: 'Tidak terdapat konten finansial atau bisnis yang dapat dibaca.' tanpa membuat segmen lain.\n"
    "- Jika hanya terdapat 1 jenis data (misalnya hanya paragraf, hanya daftar, atau hanya tabel), tampilkan langsung tanpa heading tambahan.\n"
    "- Jika terdapat >1 jenis data (misalnya tabel + daftar atau grafik + tabel), buat segmen per jenis menggunakan heading 4 (####).\n"
    "- Dilarang membuat segmen untuk konten yang tidak ada (misalnya 'Tidak ada tabel').\n"
    "- Untuk tabel: tulis ulang dalam tabel Markdown sesuai struktur dan label yang terlihat.\n"
    "- Untuk grafik: jika dapat dibaca, konversi ke tabel; jika tidak, jelaskan label/sumbu/kategori tanpa interpretasi.\n"
    "- Untuk daftar: tulis ulang sebagai bullet atau numbering mengikuti struktur asli.\n"
    "- Untuk disclaimer: tulis ulang isi teksnya.\n\n"

    "MODE RAG:\n"
    "- Hasil harus bisa dipahami tanpa melihat slide.\n"
    "- Hindari referensi relatif: 'slide ini', 'di atas', 'lihat berikut'.\n"
    "- Sebutkan eksplisit nama emiten, periode, dan satuan jika muncul (miliar, triliun, %, USD, IDR).\n"
    "- Prioritaskan data: finansial, bisnis, operasional, ESG, tata kelola.\n\n"

    "ANTI-HALUSINASI:\n"
    "- Jika tidak terlihat → jangan ditulis.\n"
    "- Jika angka tidak jelas → lewati tanpa mengisi.\n"
    "- Tidak boleh menambah narasi konteks, lokasi, atau interpretasi yang tidak tertulis.\n\n"

    "METADATA DOKUMEN:\n"
    "Dokumen: {document_name}\n"
    "Jenis: {document_type}\n"
    "Emiten: {issuer_name} ({issuer_code})\n"
    "Tahun: {document_year}\n"
    "Metadata: {document_metadata}\n"
    "Slide: {slide_no} / {total_pages}\n\n"
    "{slide_content}"
)


def _call_vlm(
    *,
    model: str,
    prompt: str,
    image_path: str,
    temperature: float = 0.2,
) -> str:
    try:
        response = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                },
            ],
            options={
                "temperature": temperature,
                "stream": False,
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"VLM error (ollama): {exc}") from exc
    try:
        return response["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response from VLM: {response}") from exc


def _get_downloaded_documents(limit: int | None) -> Iterable[Document]:
    with get_session() as session:
        stmt = select(Document).where(
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


def _count_existing_slides(session, document_id: str) -> int:
    stmt = select(func.count(Slide.id)).where(Slide.document_id == document_id)
    return int(session.execute(stmt).scalar() or 0)


def _extract_year(doc: Document, pdf_path: Path) -> str:
    if doc.publish_at:
        return str(doc.publish_at.year)
    candidates = f"{doc.name or ''} {pdf_path.name}"
    match = re.search(r"(19|20)\\d{2}", candidates)
    if match:
        return match.group(0)
    return "Tidak diketahui"


def _infer_document_type(doc: Document, pdf_path: Path) -> str:
    name = f"{doc.name or ''} {pdf_path.name}".lower()
    if "laporan keuangan" in name or "financial statement" in name:
        return "laporan keuangan"
    if "annual report" in name:
        return "laporan tahunan"
    if "public expose" in name or "pubex" in name:
        return "dokumen Public Expose (Pubex)"
    return "dokumen keuangan"


def _process_document(
    *,
    document_id: str,
    pdf_path: Path,
    model: str,
    dpi: int,
) -> bool:
    images = pdf_to_images(pdf_path, dpi=dpi)
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
        issuer_name = doc.issuer.name if doc.issuer else "Tidak diketahui"
        issuer_code = doc.issuer.code if doc.issuer else "Tidak diketahui"
        document_name = doc.name or pdf_path.name
        document_year = _extract_year(doc, pdf_path)
        document_type = _infer_document_type(doc, pdf_path)
        document_metadata = (
            json.dumps(doc.document_metadata, ensure_ascii=True, sort_keys=True)
            if doc.document_metadata
            else "-"
        )

        existing = _count_existing_slides(session, document_id)
        if existing > 0:
            logger.info(
                "Overwrite %s: hapus %s slide lama sebelum ingest ulang.",
                document_id,
                existing,
            )
            session.query(Slide).filter(Slide.document_id == document_id).delete(
                synchronize_session=False
            )
            session.commit()

        for slide_no, img_bytes in images:
            if not validate_image_bytes(img_bytes, slide_no):
                success = False
                continue

            prompt = PROMPT.format(
                document_name=document_name,
                issuer_name=issuer_name,
                issuer_code=issuer_code,
                document_year=document_year,
                document_type=document_type,
                document_metadata=document_metadata,
                slide_no=slide_no,
                total_pages=total_pages,
                slide_content="Gambar terlampir.",
            )

            tmp_path = None
            start_at = datetime.now(timezone.utc)
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp_file:
                    tmp_file.write(img_bytes)
                    tmp_path = tmp_file.name

                content_md = _call_vlm(
                    model=model,
                    prompt=prompt,
                    image_path=tmp_path,
                )
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
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        logger.debug("Gagal menghapus file sementara: %s", tmp_path)

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
                ingestion_start_at=start_at,
                ingestion_end_at=end_at,
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
    model: str = DEFAULT_VLM_MODEL,
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
            model=model,
            dpi=dpi,
        )
