from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
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

DEFAULT_VLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or os.getenv(
    "OLLAMA_URL", "http://localhost:11434"
)
DEFAULT_VLM_MODEL = os.getenv("DEEPSEEK_VLM_MODEL") or os.getenv(
    "OLLAMA_VLM_MODEL", "qwen3-vl:2b-instruct-q4_K_M"
)
DEFAULT_VLM_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("VLM_API_KEY")
DEFAULT_VLM_IMAGE_SUPPORT = (
    os.getenv("VLM_IMAGE_SUPPORT", "").lower() in {"1", "true", "yes"}
)
DEFAULT_DPI = int(os.getenv("INGESTION_PDF_DPI", "144"))

SYSTEM_PROMPT = PromptTemplate(
    "Anda berperan sebagai Analis Keuangan.\n\n"
    "Tugas Utama:\n"
    "- Melakukan analisis berbasis konten visual dari sebuah slide presentasi.\n"
    "- Analisis hanya boleh merujuk pada teks, angka, tabel, grafik, dan daftar yang terlihat pada slide.\n"
    "- Dilarang menambahkan asumsi, inferensi ekonomi, atau opini di luar informasi eksplisit pada slide.\n"
    "- Abaikan elemen non-informatif seperti ornamen, dekorasi, ikon/logo, dan latar visual.\n"
    "- Jika terdapat elemen yang tidak terbaca, kosong, atau terpotong, lewati tanpa menebak.\n"
    "- Gunakan gaya deskriptif formal, faktual, akurat, dan tanpa opini.\n\n"
    "Catatan Penggunaan:\n"
    "Output ini ditujukan untuk calon investor yang membutuhkan pembacaan cepat, padat, dan aktual.\n\n"
    "Format Keluaran (Markdown, Bahasa Indonesia):\n\n"
    "## (Judul Slide)\n"
    "Aturan Judul:\n"
    "- Jika slide memiliki judul eksplisit, gunakan judul tersebut apa adanya.\n"
    "- Jika tidak ada judul eksplisit, buat frasa deskriptif ringkas (≤ 10 kata) yang menggambarkan konten tanpa opini.\n\n"
    "### Ringkasan\n"
    "Aturan Ringkasan:\n"
    "- Berisi 3 hingga 7 kalimat faktual yang menggambarkan isi slide.\n"
    "- Boleh memanfaatkan bullet jika diperlukan untuk kejelasan.\n"
    "- Tidak ada penjelasan dengan makna yang berulang.\n"
    "- Tidak boleh menambah interpretasi, asumsi, maupun penilaian.\n\n"
    "### Konten\n"
    "Aturan Konten:\n"
    "- Konten wajib dipisahkan berdasarkan jenis data pada slide.\n"
    "- Jika terdapat lebih dari satu jenis (misalnya tabel, grafik, dan daftar), setiap jenis harus disajikan pada segmen terpisah.\n"
    "- Setiap segmen menggunakan heading 4 (####) dengan nama sesuai jenis konten, misalnya: '#### Tabel Penjualan', '#### Grafik Pertumbuhan', atau '#### Daftar Poin'.\n"
    "- Untuk tabel: tulis ulang ke tabel Markdown sesuai struktur yang terlihat.\n"
    "- Untuk grafik: jika dapat dikonversi ke tabel angka, lakukan; jika tidak memungkinkan, gambarkan secara tekstual yang menjelaskan sumbu, label, dan nilai yang terlihat.\n"
    "- Untuk daftar/bullet: tulis ulang dalam format bullet Markdown.\n"
    "- Jika terdapat konten yang tidak dapat dibaca atau kosong, lewati tanpa membuat spekulasi.'\n"
    "- Selain dari tabel, grafik, dan daftar, jika terdapat gambar (gambar manusia, tumbuhan, dsb) jangan buat penjelasan tambahan.\n\n"
)


USER_PROMPT = PromptTemplate(
    "Dokumen: {document_name}\n"
    "Jenis Dokumen: {document_type}\n"
    "Emiten: {issuer_name} ({issuer_code})\n"
    "Tahun: {document_year}\n"
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
        "DEEPSEEK_API_KEY belum di-set untuk endpoint VLM non-lokal."
    )


def _call_vlm(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    temperature: float = 0.2,
    timeout: int = 300,
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
        "temperature": temperature,
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
