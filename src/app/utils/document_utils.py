from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Generator, Tuple

import fitz  # PyMuPDF
from PIL import Image
import io

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


def pdf_to_images(
    pdf_path: Path,
    dpi: int = 150,
) -> Generator[Tuple[int, bytes], None, None]:
    """
    Generator: mengubah PDF menjadi image per halaman, satu per satu.

    Yields:
        (slide_no, image_bytes_png)
    """
    zoom = dpi / 72.0  # PDF default 72 dpi
    mat = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Konversi ke PNG bytes
            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples,
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            slide_no = page_index + 1
            yield slide_no, img_bytes
