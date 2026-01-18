#!/usr/bin/env python3
"""
Balanced Docling pipeline:
- Input:  1 PDF atau 1 folder berisi banyak PDF
- Output: 
    - <doc_id>.slides.jsonl  → 1 baris per "slide"/halaman, siap untuk RAG
    - <doc_id>.md            → markdown gabungan semua halaman

Fitur:
- Page-level split pakai page_break_placeholder Docling
- Heuristik "diagram-heavy" pakai panjang teks + placeholder gambar
- Integrasi VLM opsional untuk ringkasan diagram (via endpoint OpenAI-compatible)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import requests
from docling.document_converter import DocumentConverter
from tqdm import tqdm


# ---------------------------
# Logging
# ---------------------------

logger = logging.getLogger("docling_balanced")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


# ---------------------------
# Data structure
# ---------------------------

@dataclass
class SlideRecord:
    doc_id: str
    slide_no: int
    title: str
    body_md: str
    diagram_summary: Optional[str]
    is_diagram_heavy: bool
    source_file: str
    page_index: int


# ---------------------------
# Heuristik & utilitas
# ---------------------------

PAGE_BREAK_TOKEN = "[[[DOC_PAGE_BREAK]]]"


def export_markdown_per_page(doc) -> List[str]:
    """
    Gunakan page_break_placeholder Docling untuk membagi markdown per halaman.

    Kita minta Docling menaruh token khusus di setiap batas halaman,
    lalu split string markdown berdasarkan token tersebut.
    """
    full_md: str = doc.export_to_markdown(
        page_break_placeholder=PAGE_BREAK_TOKEN,
        image_placeholder="<!-- image -->",  # default, eksplisitkan saja
    )
    chunks = full_md.split(PAGE_BREAK_TOKEN)
    # Buang chunk kosong di depan/belakang jika ada
    return [c.strip() for c in chunks if c.strip()]


def is_diagram_heavy(md_text: str, image_count: int) -> bool:
    """
    Heuristik sederhana:
    - kalau teks sedikit tapi gambar banyak → kemungkinan diagram/chart heavy.
    """
    text_len = len(md_text.strip())
    if image_count >= 1 and text_len < 400:
        return True
    return False


# ---------------------------
# VLM (opsional)
# ---------------------------

def summarize_slide_with_vlm(
    md_text: str,
    *,
    slide_no: int,
    doc_id: str,
    timeout: int = 60,
) -> Optional[str]:
    """
    Ringkasan diagram via VLM (opsional).

    Konfigurasi lewat environment variable:
    - VLM_ENDPOINT   → contoh: http://localhost:11434/v1/chat/completions (Ollama)
                        atau  https://api.openai.com/v1/chat/completions
    - VLM_MODEL_NAME → contoh: qwen2.5-vl, gpt-4.1-mini, dsb.
    - VLM_API_KEY    → kalau endpoint butuh Authorization (OpenAI, dsb.)

    Kalau tidak diset, fungsi ini akan mengembalikan None dan tidak mengganggu pipeline.
    """
    endpoint = os.getenv("VLM_ENDPOINT")
    model_name = os.getenv("VLM_MODEL_NAME")
    api_key = os.getenv("VLM_API_KEY")

    if not endpoint or not model_name:
        logger.debug(
            "VLM_ENDPOINT atau VLM_MODEL_NAME belum di-set, skip VLM summary "
            f"(doc_id={doc_id}, slide={slide_no})"
        )
        return None

    system_prompt = (
        "Kamu adalah asisten yang merangkum isi slide presentasi berbasis bisnis "
        "dalam bahasa Indonesia. Fokus pada penjelasan grafik, diagram, dan angka, "
        "bukan formatting markdown. Jawaban singkat, 2–4 kalimat saja."
    )
    user_prompt = (
        f"Ini adalah isi suatu slide presentasi dalam format markdown.\n"
        f"Dokumen: {doc_id}, Slide: {slide_no}.\n"
        f"Jelaskan inti informasi visual (grafik/diagram) yang kemungkinan muncul "
        f"di slide ini, gunakan bahasa Indonesia yang ringkas:\n\n"
        f"{md_text}"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.3,
    }

    headers = {"Content-Type": "application/json"}
    # Kalau API pakai Bearer token (misal OpenAI)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-compatible: choices[0].message.content
        choices = data.get("choices")
        if not choices:
            logger.warning("Respons VLM tanpa 'choices', skip summary.")
            return None

        content = choices[0].get("message", {}).get("content")
        if not content:
            logger.warning("Respons VLM tanpa 'message.content', skip summary.")
            return None

        summary = content.strip()
        logger.debug(
            "VLM summary ok (doc_id=%s, slide=%s): %.60s...",
            doc_id,
            slide_no,
            summary.replace("\n", " "),
        )
        return summary

    except Exception as exc:
        logger.warning(
            "Gagal memanggil VLM untuk doc_id=%s slide=%s: %s",
            doc_id,
            slide_no,
            exc,
        )
        return None


# ---------------------------
# Core: proses 1 PDF
# ---------------------------

def process_pdf_balanced(
    pdf_path: Path,
    out_dir: Path,
    *,
    doc_id: Optional[str] = None,
    use_vlm: bool = False,
    overwrite_cache: bool = False,
) -> List[SlideRecord]:
    pdf_path = pdf_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if doc_id is None:
        doc_id = pdf_path.stem

    jsonl_path = out_dir / f"{doc_id}.slides.jsonl"
    md_path = out_dir / f"{doc_id}.md"

    if jsonl_path.exists() and not overwrite_cache:
        logger.info(
            "[SKIP] %s (cache ditemukan: %s)",
            pdf_path.name,
            jsonl_path.name,
        )
        slides: List[SlideRecord] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            slides.append(SlideRecord(**raw))
        return slides

    logger.info("[INFO] Mengonversi %s dengan Docling...", pdf_path.name)

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Bagi markdown per halaman
    page_chunks = export_markdown_per_page(doc)
    logger.info(
        "[INFO] Dokumen %s → %d halaman (berdasarkan page_break_placeholder).",
        doc_id,
        len(page_chunks),
    )

    all_slides: List[SlideRecord] = []
    all_md_parts: List[str] = []

    for page_index, page_md in enumerate(page_chunks):
        slide_no = page_index + 1
        page_md_stripped = page_md.strip()
        if not page_md_stripped:
            # Halaman kosong
            continue

        # Hitung jumlah placeholder gambar di markdown
        image_count = page_md_stripped.count("<!-- image -->")
        heavy = is_diagram_heavy(page_md_stripped, image_count)

        # Ambil judul sederhana: baris pertama yang diawali '#'
        title: Optional[str] = None
        for line in page_md_stripped.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith("#"):
                # Buang leading '#' dan spasi
                title = line_stripped.lstrip("#").strip()
                if title:
                    break

        if not title:
            title = f"Slide {slide_no}"

        diagram_summary: Optional[str] = None
        if use_vlm and heavy:
            diagram_summary = summarize_slide_with_vlm(
                page_md_stripped,
                slide_no=slide_no,
                doc_id=doc_id,
            )

        slide = SlideRecord(
            doc_id=doc_id,
            slide_no=slide_no,
            title=title,
            body_md=page_md_stripped,
            diagram_summary=diagram_summary,
            is_diagram_heavy=heavy,
            source_file=str(pdf_path),
            page_index=page_index,
        )
        all_slides.append(slide)

        # Untuk file .md gabungan
        header_comment = f"<!-- Slide {slide_no}: {title} -->"
        all_md_parts.append(f"{header_comment}\n\n{page_md_stripped}\n")

    # Simpan JSONL (1 baris per slide)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for slide in all_slides:
            f.write(json.dumps(asdict(slide), ensure_ascii=False) + "\n")

    # Simpan markdown gabungan (opsional, tapi berguna)
    md_path.write_text("\n\n".join(all_md_parts), encoding="utf-8")

    logger.info(
        "[DONE] %s → %d slide → %s, %s",
        pdf_path.name,
        len(all_slides),
        jsonl_path.name,
        md_path.name,
    )
    return all_slides


# ---------------------------
# Batch processing
# ---------------------------

def discover_pdfs(input_path: Path) -> List[Path]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"File bukan PDF: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Path tidak ditemukan: {input_path}")

    pdfs = sorted(p for p in input_path.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"Tidak ditemukan file PDF di folder: {input_path}")
    return pdfs


def process_many_pdfs(
    input_path: Path,
    out_dir: Path,
    *,
    use_vlm: bool = False,
    overwrite_cache: bool = False,
) -> None:
    pdf_files = discover_pdfs(input_path)
    logger.info("[INFO] Menemukan %d file PDF.", len(pdf_files))

    for pdf in tqdm(pdf_files, desc="Memproses PDF", unit="file"):
        try:
            process_pdf_balanced(
                pdf,
                out_dir,
                use_vlm=use_vlm,
                overwrite_cache=overwrite_cache,
            )
        except Exception as exc:
            logger.error("Gagal memproses %s: %s", pdf.name, exc)


# ---------------------------
# CLI
# ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Balanced Docling pipeline: PDF → per-slide JSONL + markdown.\n\n"
            "Contoh:\n"
            "  python docling_balanced_pipeline.py mydoc.pdf -o out/\n"
            "  python docling_balanced_pipeline.py ./pdfs -o out/ --use-vlm\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "input_path",
        type=str,
        help="Path ke 1 file PDF atau folder berisi PDF.",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="out_docling",
        help="Folder output untuk .slides.jsonl dan .md (default: %(default)s).",
    )
    p.add_argument(
        "--use-vlm",
        action="store_true",
        help=(
            "Aktifkan ringkasan diagram via VLM (OpenAI-compatible endpoint).\n"
            "Konfigurasi: VLM_ENDPOINT, VLM_MODEL_NAME, VLM_API_KEY (opsional)."
        ),
    )
    p.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Ignore cache dan proses ulang semua dokumen.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Set log level ke DEBUG.",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Log level di-set ke DEBUG")

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)

    process_many_pdfs(
        input_path=input_path,
        out_dir=out_dir,
        use_vlm=args.use_vlm,
        overwrite_cache=args.overwrite_cache,
    )


if __name__ == "__main__":
    main()
