from __future__ import annotations

import base64
import logging
from pathlib import Path
import json
import re
from typing import Iterable
from datetime import datetime, timezone

import requests
from sqlalchemy import select, func

from app.utils.document_utils import pdf_to_images, count_pdf_pages
from app.utils.image_utils import downscale_png, validate_image_bytes
from app.db.models import Document, Slide
from app.db.session import get_session
from src.app.config import (
    DEFAULT_DPI,
    DEFAULT_LLAMA_CPP_BASE_URL,
    DEFAULT_LLAMA_CPP_TIMEOUT,
    DEFAULT_VLM_MODEL,
)
from src.data.enums import DocumentStatusEnum
from src.app.utils.telegram import send_telegram_message

try:
    from llama_index.core.prompts import PromptTemplate
except ImportError as exc:  # pragma: no cover - guarded for runtime safety
    raise RuntimeError(
        "LlamaIndex tidak ditemukan. Install dengan 'pip install llama-index'."
    ) from exc

logger = logging.getLogger(__name__)

SESSION = requests.Session()

PROMPT_PUBEX = PromptTemplate(
    "Anda adalah Konsultan Keuangan yang mengekstraksi isi slide Public Expose (Pubex) "
    "menjadi teks Markdown untuk sistem Tanya Jawab berbasis RAG.\n\n"

    "ATURAN UTAMA:\n"
    "- Ekstraksi HANYA dari konten yang terlihat: teks, angka, tabel, grafik, daftar, diagram, atau peta.\n"
    "- Abaikan visual non-informatif: foto manusia, ekspresi, ikon dekoratif, ornamen, warna, latar, estetika.\n"
    "- Jangan menebak, menyimpulkan, atau menambah informasi.\n"
    "- Jika tidak terbaca atau ambigu, lewati.\n"
    "- Hindari frasa presentasi seperti: 'pada slide ini', 'dapat dilihat', 'berikut ini', atau kalimat generik.\n"
    "- Jika konten sangat sedikit, salin apa adanya tanpa narasi tambahan.\n"
    "- Dilarang menambah informasi dokumen dalam hasil ekstraksi.\n"
    "- Output WAJIB Markdown berbahasa Indonesia.\n\n"

    "STRUKTUR OUTPUT:\n"
    "- Hanya berisi Judul Slide (Heading 2) dan Konten (Jika Ada).\n"
    "- Jangan menambahkan label seperti 'Konten:' atau heading lain di luar aturan.\n"
    "- Gunakan Heading 3 (###) hanya jika terdapat lebih dari satu segmen konten.\n\n"

    "## Judul Slide\n"
    "- Jika ada judul eksplisit, salin apa adanya.\n"
    "- Jika tidak ada, buat judul deskriptif ≤10 kata tanpa opini.\n\n"

    "### Konten\n"
    "- Tulis ulang isi slide secara faktual dalam bentuk:\n"
    "  - tabel Markdown (jika tabel),\n"
    "  - bullet/numbering (jika daftar),\n"
    "  - paragraf pendek (jika teks).\n"
    "- Untuk grafik: konversi ke tabel jika angka terbaca; jika tidak, tulis label/kategori saja.\n"
    "- Untuk peta/infografis geografis: tulis lokasi, fasilitas, unit operasional, dan angka yang tertulis.\n"
    "- Jangan membuat segmen untuk konten yang tidak ada.\n"
    "- Jika tidak ada konten yang dapat dibaca, lewati tanpa memberi opini.\n\n"

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
    base_url: str = DEFAULT_LLAMA_CPP_BASE_URL,
    temperature: float = 0.2,
) -> str:
    try:
        image_bytes = downscale_png(image_bytes, max_w=1280)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": 768,  # batasi panjang output
            "top_p": 0.8,  # sampling lebih ketat
            "top_k": 30,  # sampling lebih ketat
            "repeat_penalty": 1.2,  # hukum pengulangan
        }
        response = SESSION.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=DEFAULT_LLAMA_CPP_TIMEOUT,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"llama.cpp HTTP {response.status_code}: {response.text}"
            )
            
        data = response.json()
        
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"VLM error (llama.cpp chat): {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response from llama.cpp: {data}") from exc


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
    model: str = DEFAULT_VLM_MODEL,
    dpi: int = DEFAULT_DPI,
    overwrite_mode: str = "document",
    update_doc_status: bool = True,
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
            overwrite_mode=overwrite_mode,
            update_doc_status=update_doc_status,
        )

    end_at = datetime.now(timezone.utc)
    total_seconds = (end_at - start_at).total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    logger.info("Durasi ingestion total: %s menit %s detik", minutes, seconds)
    print(f"Durasi ingestion total: {minutes} menit {seconds} detik")


def run_ingestion_multi_model(
    *,
    models: Iterable[str],
    limit: int | None = None,
    dpi: int = DEFAULT_DPI,
    overwrite_mode: str = "model",
    update_doc_status: bool = False,
) -> None:
    model_list = [m.strip() for m in models if m and m.strip()]
    if not model_list:
        logger.info("Tidak ada model yang diberikan untuk ingestion multi-model.")
        print("Tidak ada model yang diberikan untuk ingestion multi-model.")
        return

    start_at = datetime.now(timezone.utc)
    docs = _get_downloaded_documents(limit)
    if not docs:
        logger.info("Tidak ada dokumen berstatus downloaded.")
        print("Tidak ada dokumen berstatus downloaded.")
        return

    send_telegram_message(
        (
            "Mulai komparasi ingestion.\n"
            f"Model: {', '.join(model_list)}\n"
            f"Dokumen: {len(docs)}"
        ),
    )

    comparison_stats: dict[str, dict[str, int]] = {
        model: {"docs_success": 0, "slides": 0} for model in model_list
    }

    for doc in docs:
        pdf_path = Path(doc.file_path)
        if not pdf_path.is_file():
            logger.error("File PDF tidak ditemukan: %s", pdf_path)
            with get_session() as session:
                db_doc = session.get(Document, doc.id)
                if db_doc and update_doc_status:
                    db_doc.status = DocumentStatusEnum.FAILED_PARSED.id
                    session.commit()
            continue

        per_doc_counts: dict[str, int] = {}
        for model in model_list:
            logger.info(
                "Memproses doc_id=%s (%s) dengan model=%s",
                doc.id,
                pdf_path.name,
                model,
            )
            success = _process_document(
                document_id=doc.id,
                pdf_path=pdf_path,
                model=model,
                dpi=dpi,
                overwrite_mode=overwrite_mode,
                update_doc_status=update_doc_status,
            )
            with get_session() as session:
                per_doc_counts[model] = _count_existing_slides(
                    session, doc.id, model
                )
            if success:
                comparison_stats[model]["docs_success"] += 1
            comparison_stats[model]["slides"] += per_doc_counts[model]

        doc_name = doc.name or pdf_path.name
        per_model_text = ", ".join(
            f"{model}={per_doc_counts.get(model, 0)}" for model in model_list
        )
        send_telegram_message(
            (
                "Progress komparasi ingestion (tanpa notif).\n"
                f"Dokumen: {doc_name}\n"
                f"Slides per model: {per_model_text}"
            ),
            disable_notification=True,
        )

    end_at = datetime.now(timezone.utc)
    total_seconds = (end_at - start_at).total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    logger.info("Durasi ingestion total: %s menit %s detik", minutes, seconds)
    print(f"Durasi ingestion total: {minutes} menit {seconds} detik")

    comparison_lines = [
        "Selesai komparasi ingestion.",
        f"Dokumen diproses: {len(docs)}",
        f"Durasi total: {minutes} menit {seconds} detik",
        "Perbandingan hasil:",
    ]
    for model in model_list:
        stats = comparison_stats[model]
        comparison_lines.append(
            f"- {model}: docs_success={stats['docs_success']}, slides={stats['slides']}"
        )
    send_telegram_message("\n".join(comparison_lines))
