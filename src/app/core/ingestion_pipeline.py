from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
import json
import re
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

DEFAULT_VLM_BASE_URL = os.getenv("VLM_BASE_URL") or os.getenv(
    "OLLAMA_URL", "http://localhost:11434"
)
DEFAULT_VLM_MODEL = os.getenv("VLM_MODEL") or os.getenv(
    "OLLAMA_VLM_MODEL", "qwen3-vl:2b-instruct-q4_K_M"
)
DEFAULT_VLM_API_KEY = os.getenv("VLM_API_KEY")
DEFAULT_VLM_IMAGE_SUPPORT = (
    os.getenv("VLM_IMAGE_SUPPORT", "").lower() in {"1", "true", "yes"}
)
DEFAULT_DPI = int(os.getenv("INGESTION_PDF_DPI", "144"))

SYSTEM_PROMPT = PromptTemplate(
    "Anda berperan sebagai Analis Keuangan yang bertugas mengekstraksi isi slide untuk keperluan RAG indexing.\n"
    "Output akan digunakan oleh sistem QA berbasis pgvector, sehingga harus faktual, self-contained, dan bebas opini.\n\n"

    "=== ATURAN UMUM ===\n"
    "- Analisis hanya berdasarkan teks, angka, tabel, grafik, dan daftar yang terlihat.\n"
    "- Abaikan elemen non-informatif seperti ornamen, dekorasi, ikon/logo, foto manusia, ekspresi, suasana, warna latar, dan estetika visual.\n"
    "- Dilarang menebak atau menambah informasi yang tidak tertulis secara eksplisit.\n"
    "- Jika terdapat bagian yang tidak terbaca, kosong, atau terpotong, lewati tanpa menebak.\n"
    "- Gunakan bahasa formal, faktual, deskriptif, dan tanpa opini.\n"
    "- Dilarang menggunakan format JSON, objek, atau struktur key-value.\n\n"

    "=== MODE RAG ALIGNMENT ===\n"
    "- Output harus dapat berdiri sendiri (self-contained) tanpa melihat slide.\n"
    "- Hindari frasa seperti 'slide ini', 'gambar di atas', 'lihat di bawah', atau referensi relatif lainnya.\n"
    "- Jika terdapat nama perusahaan, periode, satuan (miliar, triliun, persen, USD, IDR), tulis secara eksplisit.\n"
    "- Prioritaskan data finansial, bisnis, operasional, dan tatakelola.\n"
    "- Jika slide berupa cover/poster non-informasional, tuliskan sebagai slide pembuka dan jangan mendeskripsikan foto/visual.\n\n"

    "=== FORMAT OUTPUT MARKDOWN WAJIB ===\n"
    "## Judul Slide\n\n"
    "### Ringkasan\n"
    "- Maksimal 7 kalimat, boleh menggunakan bullet.\n"
    "- Hanya memuat konten yang relevan secara finansial, bisnis, operasional, atau informasi presentasi.\n"
    "- Dilarang menjelaskan visual (misalnya foto, ekspresi, pakaian, suasana, warna latar, tata letak).\n"
    "- Jika slide merupakan cover/poster tanpa konten finansial: tulis 'Slide pembuka. Tidak terdapat konten finansial atau bisnis.'\n\n"
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

    "=== MODE FINANCIAL PRIORITY ALIGNMENT ===\n"
    "- Prioritaskan ekstraksi kategori berikut jika tersedia:\n"
    "  * metrik finansial: pendapatan, laba, EBITDA, margin, capex, dividen, guidance\n"
    "  * komparatif: YoY, QoQ, persen, rasio\n"
    "  * entitas: perusahaan, unit bisnis, produk\n"
    "  * periode: tahun, kuartal, bulan\n"
    "  * operasional: produksi, kapasitas, volume, ESG\n"
    "  * event: Public Expose, RUPS, Analyst Meeting\n\n"

    "=== MODE ANTI-HALLUCINATION ===\n"
    "- Jika informasi tidak terlihat → jangan tulis.\n"
    "- Jika angka/rincian tidak jelas → lewati tanpa mengisi.\n"
    "- Dilarang menulis opini atau spekulasi.\n"
    "- Dilarang mengarang konteks.\n"
)

USER_PROMPT = PromptTemplate(
    "Dokumen: {document_name}\n"
    "Jenis Dokumen: {document_type}\n"
    "Emiten: {issuer_name} ({issuer_code})\n"
    "Tahun: {document_year}\n"
    "Metadata Dokumen: {document_metadata}\n"
    "Slide: {slide_no} dari {total_pages}\n\n"
    "{slide_content}"
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


def _resolve_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions") or base_url.endswith(
        "/v1/chat/completions"
    ):
        return base_url
    if "localhost" in base_url or "127.0.0.1" in base_url:
        return f"{base_url}/v1/chat/completions"
    return f"{base_url}/chat/completions"


def _is_local_endpoint(base_url: str) -> bool:
    return "localhost" in base_url or "127.0.0.1" in base_url


def _resolve_api_key(base_url: str) -> str:
    if DEFAULT_VLM_API_KEY:
        return DEFAULT_VLM_API_KEY
    if _is_local_endpoint(base_url):
        return "ollama"
    raise RuntimeError(
        "VLM_API_KEY belum di-set untuk endpoint VLM non-lokal."
    )


def _call_vlm(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    temperature: float = 0.2,
    timeout: int = 600,
) -> str:
    endpoint = _resolve_endpoint(base_url)
    api_key = _resolve_api_key(base_url)
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "ollama":
        headers["Authorization"] = f"Bearer {api_key}"
    supports_images = _is_local_endpoint(base_url) or DEFAULT_VLM_IMAGE_SUPPORT
    if supports_images:
        content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    else:
        logger.warning(
            "Endpoint %s tidak mendukung image_url. "
            "Kirim prompt teks saja. Set VLM_IMAGE_SUPPORT=true jika endpoint mendukung image.",
            endpoint,
        )
        content = f"image: {image_data_url}\n\nprompt: {user_prompt}"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": content,
            }
        ],
         "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "num_predict": 256
        }
    }
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(
            f"VLM error {resp.status_code} for {endpoint}: {resp.text}"
        )
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response from VLM: {data}") from exc


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
    base_url: str,
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
            if not _validate_image(img_bytes, slide_no):
                success = False
                continue

            data_url = _image_bytes_to_data_url(img_bytes)

            system_prompt = SYSTEM_PROMPT.format()
            user_prompt = USER_PROMPT.format(
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

            try:
                content_md = _call_vlm(
                    base_url=base_url,
                    model=model,
                    user_prompt=user_prompt,
                    image_data_url=data_url,
                    system_prompt=system_prompt,
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
    base_url: str = DEFAULT_VLM_BASE_URL,
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
            base_url=base_url,
            model=model,
            dpi=dpi,
        )
