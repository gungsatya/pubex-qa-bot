from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Pastikan nama file aman dan tidak berisi separator path."""
    base_name = Path(name).name
    sanitized = base_name.replace("/", "_").replace("\\", "_").strip()
    return sanitized or "document.pdf"


def compute_checksum(path: Path) -> str:
    """Hitung checksum SHA256 dari file lokal."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_pdf_page_count(path: Path) -> int | None:
    """Hitung jumlah halaman PDF, return None jika gagal."""
    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:  # noqa: BLE001
        logger.exception("Gagal menghitung jumlah halaman untuk %s", path)
        return None


def count_pdf_pages(pdf_path: Path) -> int:
    """
    Mengembalikan jumlah halaman PDF tanpa merender semuanya.
    Hemat memori, hanya baca metadata.
    """
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def pdf_to_png_images(
    pdf_path: Path,
    dpi: int = 150,
) -> Generator[tuple[int, bytes], None, None]:
    """
    Render PDF menjadi PNG per halaman.

    Yields:
        (page_no, png_bytes)
    """
    zoom = dpi / 72.0  # PDF default 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            yield page_index + 1, pix.tobytes("png")
