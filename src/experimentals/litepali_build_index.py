#!/usr/bin/env python3
"""
Build LitePali index dari kumpulan PDF di sebuah folder.

Alur:
- Ambil semua file .pdf dari input_dir
- Setiap PDF dipecah per halaman → disimpan sebagai JPG
- Setiap JPG didaftarkan ke LitePali sebagai ImageFile
- LitePali memproses semua image dan menyimpan index

Contoh pakai:

    python build_litepali_index_from_pdfs.py \
        --input-pdfs ./pdfs \
        --images-dir ./images \
        --index-dir ./indexes

Nanti:
- Gambar-gambar ada di ./images
- Index LitePali ada di ./indexes/litepali_index
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from litepali import LitePali, ImageFile
from tqdm import tqdm

logger = logging.getLogger("pdf_to_litepali")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


def pdf_to_images(
    pdf_path: Path,
    images_root: Path,
    dpi: int = 200,
) -> List[Path]:
    """
    Konversi 1 file PDF menjadi beberapa file JPG (per halaman).

    images_root / <pdf_stem> / <pdf_stem>_p001.jpg, dst.
    """
    pdf_path = pdf_path.resolve()
    out_dir = images_root.resolve() / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Konversi PDF → image: %s", pdf_path.name)
    # pdf2image akan mengembalikan list PIL.Image
    pages = convert_from_path(str(pdf_path), dpi=dpi)

    image_paths: List[Path] = []
    for i, page in enumerate(pages, start=1):
        img_name = f"{pdf_path.stem}_p{i:03d}.jpg"
        img_path = out_dir / img_name
        # Simpan sebagai JPEG
        page.save(img_path, "JPEG")
        image_paths.append(img_path)

    logger.info(
        "Selesai konversi %s → %d halaman (image disimpan di %s)",
        pdf_path.name,
        len(image_paths),
        out_dir,
    )
    return image_paths


def build_litepali_index(
    input_pdfs_dir: Path,
    images_dir: Path,
    index_dir: Path,
    dpi: int = 200,
) -> None:
    """
    Baca semua PDF di input_pdfs_dir, convert ke image, dan build LitePali index.
    """
    input_pdfs_dir = input_pdfs_dir.resolve()
    images_dir = images_dir.resolve()
    index_dir = index_dir.resolve()

    if not input_pdfs_dir.exists():
        raise FileNotFoundError(f"Folder PDF tidak ditemukan: {input_pdfs_dir}")

    pdf_files = sorted(input_pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"Tidak ditemukan file .pdf di folder: {input_pdfs_dir}")

    logger.info("Menemukan %d file PDF di %s", len(pdf_files), input_pdfs_dir)

    litepali = LitePali(device="cpu")

    # document_id bisa sekadar counter per file PDF
    document_id = 1

    for pdf_path in tqdm(pdf_files, desc="Memproses PDF", unit="file"):
        # 1) PDF → images
        page_images = pdf_to_images(pdf_path, images_dir, dpi=dpi)

        # 2) Daftarkan setiap image ke LitePali
        for page_idx, img_path in enumerate(page_images, start=1):
            litepali.add(
                ImageFile(
                    path=str(img_path),
                    document_id="%04d" % document_id,
                    page_id="%04d" % page_idx,
                    metadata={
                        "filename": pdf_path.name,
                        "stem": pdf_path.stem,
                        "page": page_idx,
                    },
                )
            )

        document_id += 1

    logger.info("Mulai proses indexing LitePali (ini bagian yang berat / model jalan)...")
    litepali.process(batch_size=4)
    logger.info("Indexing selesai.")

    # 3) Simpan index ke disk
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "litepali_index"
    litepali.save_index(str(index_path))

    logger.info("Index disimpan ke: %s", index_path)
    logger.info("Selesai. Kamu bisa load index ini nanti untuk search.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Bangun LitePali index dari PDF dalam sebuah folder.\n\n"
            "Contoh:\n"
            "  python build_litepali_index_from_pdfs.py "
            "--input-pdfs ./pdfs --images-dir ./images --index-dir ./indexes"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "--input-pdfs",
        type=str,
        help="Folder berisi file-file PDF.",
        default="./src/data/documents",
    )
    p.add_argument(
        "--images-dir",
        type=str,
        default="./src/data/images",
        help="Folder output untuk menyimpan image hasil split PDF (default: ./images).",
    )
    p.add_argument(
        "--index-dir",
        type=str,
        default="./src/data/indexes",
        help="Folder output untuk menyimpan index LitePali (default: ./indexes).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI untuk konversi PDF → image (default: 200).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Aktifkan log level DEBUG.",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    input_pdfs_dir = Path(args.input_pdfs)
    images_dir = Path(args.images_dir)
    index_dir = Path(args.index_dir)

    build_litepali_index(
        input_pdfs_dir=input_pdfs_dir,
        images_dir=images_dir,
        index_dir=index_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
