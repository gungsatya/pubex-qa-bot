# src/app/core/pdf_downloader.py

from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cloudscraper
import requests
from tqdm import tqdm

from app.utils.document_utils import (
    compute_checksum,
    get_pdf_page_count,
    sanitize_filename,
)
from app.db.session import get_session
from app.db.repositories import (
    IssuerRepository,
    CollectionRepository,
    DocumentStatusRepository,
    DocumentRepository,
)

logger = logging.getLogger(__name__)

IDX_API_URL = "https://www.idx.co.id/primary/ListedCompany/GetProfileAnnouncement"


def _build_keyword_for_type(doc_type: str) -> str:
    doc_type = doc_type.lower()
    if doc_type == "pubex":
        return "penyampaian materi public expose"
    elif doc_type == "financial_report":
        # silakan sesuaikan lagi keyword-nya
        return "laporan keuangan"
    return doc_type


def _build_idx_params(issuer_code: str, year: int, doc_type: str) -> dict:
    """Bangun parameter untuk API IDX berdasarkan kode emiten, tahun, dan tipe dokumen."""
    keyword = _build_keyword_for_type(doc_type)
    return {
        "KodeEmiten": issuer_code,
        "indexFrom": 0,
        "pageSize": 10,
        "dateFrom": f"{year}0101",
        "dateTo": f"{year}1231",
        "lang": "id",
        "keyword": keyword,
    }


def _parse_publish_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning("Format TglPengumuman tidak valid: %s", value)
        return None


def _extract_pengumuman_metadata(pengumuman: Dict) -> Dict:
    return {
        "Id2": pengumuman.get("Id2"),
        "NoPengumuman": pengumuman.get("NoPengumuman"),
        "JudulPengumuman": pengumuman.get("JudulPengumuman"),
        "PerihalPengumuman": pengumuman.get("PerihalPengumuman"),
        "Kode_Emiten": pengumuman.get("Kode_Emiten"),
        "JenisPengumuman": pengumuman.get("JenisPengumuman"),
    }


def _fetch_attachments_for_issuer(
    scraper: cloudscraper.CloudScraper,
    issuer_code: str,
    year: int,
    doc_type: str,
) -> List[Dict]:
    """
    Panggil API IDX untuk satu issuer & tahun & tipe dokumen.
    Return list of attachment info dicts.
    """
    params = _build_idx_params(issuer_code, year, doc_type)
    resp = scraper.get(IDX_API_URL, params=params, timeout=30)

    if resp.status_code != 200:
        logger.warning(
            "Gagal fetch IDX untuk %s (%s, %s): status %s / %s",
            issuer_code,
            doc_type,
            year,
            resp.status_code,
            resp.text[:200],
        )
        return []

    data = resp.json()
    replies = data.get("Replies", []) or []

    attachments: List[Dict] = []

    for reply in replies:
        pengumuman = reply.get("pengumuman") or {}
        publish_at = _parse_publish_at(pengumuman.get("TglPengumuman"))
        pengumuman_metadata = _extract_pengumuman_metadata(pengumuman)
        raw_attachments = reply.get("attachments") or []
        filtered = [att for att in raw_attachments if att.get("IsAttachment")]

        for att in filtered:
            full_save_path = att.get("FullSavePath")
            if not full_save_path:
                continue

            download_url = f"https://idx.co.id{full_save_path}"
            original_filename = att.get("OriginalFilename") or os.path.basename(
                full_save_path
            )
            safe_filename = sanitize_filename(original_filename)
            attachments.append(
                {
                    "download_url": download_url,
                    "file_name": safe_filename,
                    "original_filename": original_filename,
                    "publish_at": publish_at,
                    "pengumuman_metadata": pengumuman_metadata,
                }
            )

    return attachments


def _download_file(scraper: cloudscraper.CloudScraper, url: str, dest_path: Path) -> None:
    """Download satu file ke dest_path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    resp = scraper.get(url, stream=True, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Status code {resp.status_code} untuk URL {url}")

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            f.write(chunk)


def download_all_from_idx_for_year(
    year: int,
    doc_type: str,
    base_dir: Path,
) -> None:
    """
    Mekanisme utama:
    - Pilih type: 'pubex' atau 'financial_report'
    - Buat / ambil collection untuk (type, year)
    - Ambil semua issuer via IssuerRepository
    - Untuk setiap issuer:
        - panggil API IDX
        - download file (kalau belum ada)
        - hitung checksum
        - kalau checksum belum ada di documents → insert row dengan status 'downloaded'
    """
    doc_type = doc_type.lower()
    logger.info("Mulai download IDX (%s) untuk tahun %s", doc_type, year)

    # Struktur folder: src/data/documents/<type>/<year>/<KODE>/
    base_dir.mkdir(parents=True, exist_ok=True)
    type_dir = base_dir / doc_type
    year_dir = type_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    scraper = cloudscraper.create_scraper()

    with get_session() as session:
        issuer_repo = IssuerRepository(session)
        collection_repo = CollectionRepository(session)
        doc_status_repo = DocumentStatusRepository(session)
        doc_repo = DocumentRepository(session)

        collection = collection_repo.get_or_create_for_type_year(doc_type, year)
        downloaded_status = doc_status_repo.get_by_name("downloaded")
        if not downloaded_status:
            raise RuntimeError(
                "document_status 'downloaded' tidak ditemukan. "
                "Isi dulu tabel document_status."
            )

        issuers = issuer_repo.get_all()

        if not issuers:
            msg = "Tidak ada issuer di tabel issuers."
            logger.warning(msg)
            print(msg)
            return

        # Progress bar per issuer
        for issuer in tqdm(
            issuers,
            desc=f"{doc_type.upper()} {year}",
            unit="issuer",
        ):
            code = (issuer.code or "").strip()
            if not code:
                continue

            try:
                attachments = _fetch_attachments_for_issuer(
                    scraper, code, year, doc_type
                )
            except requests.exceptions.RequestException as exc:
                logger.exception("Error request IDX untuk %s: %s", code, exc)
                continue

            if not attachments:
                logger.info(
                    "Tidak ada %s untuk %s tahun %s",
                    doc_type,
                    code,
                    year,
                )
                time.sleep(random.uniform(0.5, 1.0))
                continue

            emiten_dir = year_dir / code
            emiten_dir.mkdir(parents=True, exist_ok=True)

            for attachment in attachments:
                download_url = attachment["download_url"]
                file_name = attachment["file_name"]
                original_filename = attachment["original_filename"]
                publish_at = attachment["publish_at"]
                pengumuman_metadata = attachment["pengumuman_metadata"]
                dest_path = emiten_dir / file_name

                # Skip kalau file sudah ada → compute checksum & cek DB
                if dest_path.exists():
                    checksum_existing = compute_checksum(dest_path)
                    existing_doc = doc_repo.get_by_checksum(checksum_existing)
                    if existing_doc:
                        logger.info(
                            "SKIP: File %s (%s) sudah ada dan ter-registrasi di DB (id=%s).",
                            file_name,
                            code,
                            existing_doc.id,
                        )
                        continue
                    # else: file ada tapi belum ada di DB → kita treat sebagai asset baru

                # Download file
                try:
                    logger.info("Download %s", original_filename)
                    _download_file(scraper, download_url, dest_path)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Gagal download file %s untuk %s: %s",
                        file_name,
                        code,
                        exc,
                    )
                    continue

                # Compute checksum
                checksum = compute_checksum(dest_path)

                # Check DB again after download
                existing_doc = doc_repo.get_by_checksum(checksum)
                if existing_doc:
                    logger.info(
                        "SKIP: File %s (%s) checksum sudah ada di DB (id=%s).",
                        file_name,
                        code,
                        existing_doc.id,
                    )
                    continue

                # Build pretty name & metadata
                pretty_type = "Public Expose" if doc_type == "pubex" else "Laporan Keuangan"
                doc_name = f"{pretty_type} {code} {year} - {original_filename}"

                page_count = get_pdf_page_count(dest_path)
                metadata = {
                    "source": "IDX",
                    "type": "pubex" if doc_type == "pubex" else "finrep",
                    "year": year,
                    "collection_name": collection.name,
                    "pages": page_count,
                    "issuer_code": code,
                    "filename": original_filename,
                    **pengumuman_metadata,
                }

                # Insert to documents (ONE DOCUMENT)
                doc = doc_repo.create_document(
                    collection_code=collection.code,
                    issuer_code=code,
                    checksum=checksum,
                    name=doc_name,
                    file_path=str(dest_path),
                    publish_at=publish_at,
                    status_id=downloaded_status.id,
                    metadata=metadata,
                )

                session.commit()  # <── COMMIT SATU DOKUMEN
                logger.info("INSERT DB: document id=%s path=%s", doc.id, dest_path)

            # Delay kecil antar issuer untuk sopan ke IDX
            time.sleep(random.uniform(0.8, 1.5))

        session.commit()

    logger.info("Selesai download %s tahun %s", doc_type, year)
