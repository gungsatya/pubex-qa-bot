# src/app/core/pdf_downloader.py

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import cloudscraper
import requests
from tqdm import tqdm

from app.db.session import get_session
from app.db.repositories import (
    IssuerRepository,
    CollectionRepository,
    DocumentStatusRepository,
    DocumentRepository,
)
from src.data.enums import DocumentStatusEnum

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


def _fetch_attachments_for_issuer(
    scraper: cloudscraper.CloudScraper,
    issuer_code: str,
    year: int,
    doc_type: str,
) -> List[Tuple[str, str]]:
    """
    Panggil API IDX untuk satu issuer & tahun & tipe dokumen.
    Return list (download_url, file_name).
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

    attachments: List[Tuple[str, str]] = []

    for reply in replies:
        raw_attachments = reply.get("attachments") or []
        filtered = [att for att in raw_attachments if att.get("IsAttachment")]

        for att in filtered:
            full_save_path = att.get("FullSavePath")
            if not full_save_path:
                continue

            download_url = f"https://idx.co.id{full_save_path}"
            file_name = os.path.basename(full_save_path)
            attachments.append((download_url, file_name))

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


def _compute_checksum(path: Path) -> str:
    """Hitung checksum SHA256 dari file lokal."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


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

            for download_url, file_name in attachments:
                dest_path = emiten_dir / file_name

                # Skip kalau file sudah ada → compute checksum & cek DB
                if dest_path.exists():
                    checksum_existing = _compute_checksum(dest_path)
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
                    logger.info("Download %s -> %s", download_url, dest_path)
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
                checksum = _compute_checksum(dest_path)

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
                doc_name = f"{pretty_type} {code} {year} - {file_name}"

                metadata = {
                    "source": "IDX",
                    "type": doc_type,
                    "year": year,
                    "issuer_code": code,
                    "filename": file_name,
                }

                # Insert to documents (ONE DOCUMENT)
                doc = doc_repo.create_document(
                    collection_code=collection.code,
                    issuer_code=code,
                    checksum=checksum,
                    name=doc_name,
                    file_path=str(dest_path),
                    status_id=DocumentStatusEnum.DOWNLOADED.id,
                    metadata=metadata,
                )

                session.commit()  # <── COMMIT SATU DOKUMEN
                logger.info("INSERT DB: document id=%s path=%s", doc.id, dest_path)

                dest_path = emiten_dir / file_name

                # Kalau file sudah ada → cek checksum & DB
                if dest_path.exists():
                    checksum_existing = _compute_checksum(dest_path)
                    existing_doc = doc_repo.get_by_checksum(checksum_existing)
                    if existing_doc:
                        logger.info(
                            "File %s (%s) sudah terdaftar dengan checksum yang sama, skip.",
                            file_name,
                            code,
                        )
                        continue  # tidak usah download lagi

                # Download (overwrite atau baru)
                try:
                    logger.info("Download %s -> %s", download_url, dest_path)
                    _download_file(scraper, download_url, dest_path)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Gagal download file %s untuk %s: %s",
                        file_name,
                        code,
                        exc,
                    )
                    continue

                # Hitung checksum file yang baru di-download
                checksum = _compute_checksum(dest_path)

                # Cek di DB: kalau checksum sudah ada, tidak usah insert dokumen baru
                existing_doc = doc_repo.get_by_checksum(checksum)
                if existing_doc:
                    logger.info(
                        "Checksum %s sudah ada di documents (id=%s), skip insert.",
                        checksum,
                        existing_doc.id,
                    )
                    continue

                # Buat nama dokumen yang manusiawi
                # contoh: "PUBEX ABMM 2023 - <nama_file>.pdf"
                pretty_type = "Public Expose" if doc_type == "pubex" else "Laporan Keuangan"
                doc_name = f"{pretty_type} {code} {year} - {file_name}"

                metadata = {
                    "source": "IDX",
                    "type": doc_type,
                    "year": year,
                    "issuer_code": code,
                    "filename": file_name,
                }

                doc = doc_repo.create_document(
                    collection_code=collection.code,
                    issuer_code=code,
                    checksum=checksum,
                    name=doc_name,
                    file_path=str(dest_path),
                    status_id=downloaded_status.id,
                    metadata=metadata,
                )
                logger.info(
                    "Dokumen tersimpan di DB: id=%s, path=%s",
                    doc.id,
                    dest_path,
                )

            # Delay kecil antar issuer untuk sopan ke IDX
            time.sleep(random.uniform(0.8, 1.5))

        session.commit()

    logger.info("Selesai download %s tahun %s", doc_type, year)
