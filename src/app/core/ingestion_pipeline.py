from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
import json
import re
from typing import Iterable
from datetime import datetime, timezone

import requests
from sqlalchemy import select, func

from app.utils.document_utils import pdf_to_images, count_pdf_pages
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
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
SESSION = requests.Session()

PROMPT_PUBEX = PromptTemplate(
    "Anda adalah Konsultan Keuangan dan Kebijakan Perusahaan.\n"
    "Tugas Anda adalah mengekstraksi isi slide presentasi Public Expose (Pubex) untuk sistem Tanya Jawab berbasis RAG.\n\n"
    
    "*ATURAN UMUM:*\n"
    "- Analisis HANYA berdasarkan pada konten slide yang terlihat, termasuk teks, angka, tabel, grafik, atau daftar.\n"
    "- ABAIKAN visual non-informatif: foto manusia, ekspresi, suasana, pakaian, ikon, dekorasi, warna, dan estetika.\n"
    "- TIDAK BOLEH menebak, menambah, menyimpulkan, atau beropini.\n"
    "- Jika bagian tidak terbaca/tidak jelas → lewati tanpa menebak.\n"
    "- Hasil harus bisa dipahami tanpa melihat slide.\n"
    "- Hindari referensi relatif: 'slide ini', 'di atas', 'lihat berikut'.\n"
    "- Sebutkan eksplisit nama emiten, periode, dan satuan jika muncul (miliar, triliun, %, USD, IDR).\n"
    "- Prioritaskan data: finansial, bisnis, operasional, ESG, tata kelola.\n"
    "- DILARANG menulis kalimat generik yang tidak muncul di slide, seperti: 'Laporan ini mencakup informasi keuangan, operasional, dan rencana bisnis' atau 'Data ini disajikan dalam bentuk laporan yang terstruktur dan dapat diakses oleh investor'.\n"
    "- Jika konten pada slide sangat sedikit (misalnya hanya judul, logo, atau 1 kalimat singkat),  cukup tulis ulang apa yang ada tanpa menambah penjelasan tambahan."
    "- OUTPUT berupa Markdown berbahasa Indonesia.\n"
    "- Gunakan gaya bahasa informatif, formal dan jelas.\n\n"
    
    "*OUTPUT MARKDOWN WAJIB:*\n"
    "- Struktur output harus terdiri dari dua bagian utama: (Judul Slide), dan Konten.\n"
    "- Output HARUS mengikuti aturan umum.\n"
    "- (Judul slide) disesuaikan dengan aturan yang telah ditentukan.\n\n"
    
    "Aturan Output :\n"
    "## (Judul Slide) (Heading 2)\n"
    "-Ganti (Judul Slide) dengan judul yang sesuai berdasarkan aturan berikut:\n"
    "   - Jika slide memiliki judul eksplisit, gunakan teks judul tersebut secara apa adanya.\n"
    "   - Jika tidak terdapat judul eksplisit, buat frasa deskriptif ≤10 kata yang hanya menggambarkan isi slide tanpa opini, interpretasi, atau penambahan informasi.\n\n"
    
    "### Konten (Heading 3)\n"
    "- Hanya berisi informasi finansial/bisnis yang berasal dari paragraf, tabel, grafik, daftar pada slide\n"
    "- Jika tidak ada konten finansial/bisnis yang terbaca, tulis: 'Tidak terdapat konten finansial atau bisnis yang dapat dibaca.'.\n"
    "- Jika hanya terdapat 1 jenis data (misalnya hanya paragraf, hanya daftar, atau hanya tabel), tampilkan langsung tanpa membuat segmen.\n"
    "- Jika terdapat >1 jenis data (misalnya ada 2 tabel, 1 daftar, 3 grafik; atau tabel, daftar dan grafik masing-masing satu), buat segmen per jenis dengan judul sesuai topik konten menggunakan heading 4 (####).\n"
    "- DILARANG membuat segmen jika tidak ada konten yang ditampilkan.\n"
    "- Untuk tabel: tulis ulang dalam tabel Markdown sesuai struktur dan label yang terlihat.\n"
    "- Untuk grafik: jika dapat dibaca, konversi ke tabel; jika tidak, jelaskan label/sumbu/kategori tanpa interpretasi.\n"
    "- Untuk daftar: tulis ulang sebagai bullet atau numbering mengikuti struktur asli.\n"
    "- Untuk paragraf: tulis ulang isi teksnya secara apa adanya, tanpa menambah kalimat baru.\n"
    "- Jika paragraf hanya berisi judul atau frasa pendek, cukup salin apa adanya.\n"
    "- Untuk infografis geografis dan peta sebaran:\n"
    "   - Jika terdapat peta geografis atau diagram penyebaran lokasi, identifikasi informasi geografis yang eksplisit terlihat.\n"
    "   - Informasi yang dapat diekstraksi mencakup nama kota, kabupaten, provinsi, pulau, negara, fasilitas, unit bisnis, pelabuhan, tambang, perkebunan, kantor cabang, gudang, atau area operasional.\n"
    "   - Jika terdapat angka (misalnya luas, kapasitas, tonase, hektare, km, unit), tuliskan angkanya sesuai yang terlihat.\n"
    "   - Jika terdapat label ikon (contoh: ikon pabrik, kapal, rig, truk, menara BTS), tuliskan labelnya tanpa menebak maknanya.\n"
    "   - Jika terdapat koneksi garis (misalnya rute logistik atau alur distribusi), tuliskan daftar rute yang terlihat tanpa interpretasi.\n"
    "   - Jika terdapat cluster wilayah, kelompokkan berdasarkan wilayah administratif yang terlihat (misalnya provinsi → kota, atau negara → kota).\n"
    "   - Jika informasi tidak terbaca atau ambigu, lewati tanpa menebak.\n\n"

    "INFORMASI DOKUMEN:\n"
    "- Dokumen: {document_name}\n"
    "- Emiten: {issuer_name} ({issuer_code})\n"
    "- Tahun: {document_year}\n"
    "- Slide: {slide_no} / {total_pages}\n\n"
)


def _call_vlm(
    *,
    model: str,
    prompt: str,
    image_bytes: bytes,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    temperature: float = 0.2,
) -> str:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 768,      # batasi panjang output
                "top_p": 0.8,            # sampling lebih ketat
                "repeat_penalty": 1.2,   # hukum pengulangan
            },
        }
        response = SESSION.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=DEFAULT_OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"VLM error (ollama generate): {exc}") from exc

    try:
        return data["response"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response from VLM: {data}") from exc


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
    collection_metadata = doc.collection.collection_metadata if doc.collection else None
    if isinstance(collection_metadata, dict):
        year = collection_metadata.get("year")
        if isinstance(year, int):
            return str(year)
        if isinstance(year, str) and year.strip():
            return year.strip()
    if isinstance(doc.document_metadata, dict):
        year = doc.document_metadata.get("year")
        if isinstance(year, int):
            return str(year)
        if isinstance(year, str) and year.strip():
            return year.strip()
    candidates = f"{doc.name or ''} {pdf_path.name}"
    match = re.search(r"(19|20)\\d{2}", candidates)
    if match:
        return match.group(0)
    return "Tidak diketahui"


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
    dpi: int,
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
        issuer_name = doc.issuer.name if doc.issuer else "Tidak diketahui"
        issuer_code = doc.issuer.code if doc.issuer else "Tidak diketahui"
        document_name = doc.name or pdf_path.name
        document_year = _extract_year(doc, pdf_path)
        document_type = _infer_document_type(doc)
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

            if document_type == "pubex":
                try:
                    prompt = PROMPT_PUBEX.format(
                        document_name=document_name,
                        issuer_name=issuer_name,
                        issuer_code=issuer_code,
                        document_year=document_year,
                        document_metadata=document_metadata,
                        slide_no=slide_no,
                        total_pages=total_pages,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Gagal format prompt untuk doc_id=%s slide=%s: %s",
                        document_id,
                        slide_no,
                        exc,
                    )
                    success = False
                    continue
            else:
                prompt = ""

            start_at = datetime.now(timezone.utc)
            try:
                content_md = _call_vlm(
                    model=model,
                    prompt=prompt,
                    image_bytes=img_bytes,
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

            image_path = slide_dir / f"{slide_no:03}.png"
            image_path.write_bytes(img_bytes)

            slide = Slide(
                document_id=document_id,
                content_text=content_md.strip(),
                image_path=str(image_path),
                slide_metadata={
                    "slide_no": slide_no,
                    "total_pages": total_pages,
                    "file_path": str(pdf_path),
                    "image_mime": "image/png",
                    "dpi": dpi,
                    "vlm_model": model,
                },
                ingestion_start_at=start_at,
                ingestion_end_at=end_at,
            )
            session.add(slide)
            logger.info(
                "Selesai proses slide %s/%s doc_id=%s",
                slide_no,
                total_pages,
                document_id,
            )

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
    start_at = datetime.now(timezone.utc)
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

    end_at = datetime.now(timezone.utc)
    total_seconds = (end_at - start_at).total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    logger.info("Durasi ingestion total: %s menit %s detik", minutes, seconds)
    print(f"Durasi ingestion total: {minutes} menit {seconds} detik")
