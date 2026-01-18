#!/usr/bin/env python3
"""
pdf_vlm_extractor.py

Script untuk:
- Membaca file PDF
- Mengubah tiap halaman menjadi image
- Mengirim image ke Ollama VLM (OpenAI compatible /v1/chat/completions)
- Menghasilkan penjelasan konten halaman dalam bahasa Indonesia
- Menggabungkan hasil ke dalam satu file Markdown

Dependensi:
    pip install pymupdf pillow requests

Contoh pemakaian:
    python pdf_vlm_extractor.py input.pdf -o output.md \
        --model qwen3-vl:latest \
        --ollama-url http://localhost:11434/v1/chat/completions
"""

import argparse
import base64
import io
import os
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image
import requests


def pdf_to_images(pdf_path: str, dpi: int = 144) -> List[Tuple[int, bytes]]:
    """
    Mengubah tiap halaman PDF menjadi image PNG (bytes).

    :param pdf_path: Path ke file PDF
    :param dpi: Resolusi render (semakin besar, semakin tajam & berat)
    :return: List tuple (page_number, image_bytes)
    """
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


def image_bytes_to_data_url(img_bytes: bytes) -> str:
    """
    Mengubah bytes image menjadi data URL base64.

    :param img_bytes: Image dalam bentuk bytes
    :return: String "data:image/png;base64,..."
    """
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_prompt(page_num: int, total_pages: int, language: str = "id") -> str:
    """
    Membuat prompt untuk VLM agar menjelaskan isi halaman.

    :param page_num: Nomor halaman
    :param total_pages: Total halaman
    :param language: Kode bahasa ("id" atau "en")
    :return: Prompt teks
    """
    if language.lower().startswith("id"):
        return (
            "Kamu adalah analis dokumen keuangan yang ahli.\n"
            "Gambar yang disertakan adalah satu halaman dari dokumen presentasi / laporan (misalnya Public Expose).\n\n"
            f"Halaman ini adalah halaman {page_num} dari total {total_pages} halaman.\n\n"
            "Tugasmu:\n"
            "1. Jelaskan secara ringkas tetapi jelas apa isi utama halaman ini.\n"
            "2. Uraikan poin-poin penting dalam bentuk bullet.\n"
            "3. Jika ada angka, grafik, atau tabel, jelaskan maknanya secara deskriptif.\n"
            "4. Gunakan bahasa Indonesia baku dan mudah dipahami oleh investor ritel.\n\n"
            "Format keluaran (gunakan Markdown):\n"
            "### Ringkasan\n"
            "- (1–3 kalimat ringkasan)\n\n"
            "### Poin Utama\n"
            "- ...\n"
            "- ...\n\n"
            "### Detail Tambahan\n"
            "- ... (opsional, hanya jika ada detail yang penting)\n"
        )
    else:
        return (
            "You are an expert financial document analyst.\n"
            "The attached image is one page of a slide deck / report.\n\n"
            f"This is page {page_num} out of {total_pages} pages.\n\n"
            "Tasks:\n"
            "1. Briefly summarize the main content of this page.\n"
            "2. List key points as bullets.\n"
            "3. If there are numbers, charts, or tables, describe their meaning.\n"
            "4. Use clear, simple language.\n\n"
            "Output format (Markdown):\n"
            "### Summary\n"
            "- ...\n\n"
            "### Key Points\n"
            "- ...\n\n"
            "### Additional Details\n"
            "- ... (optional)\n"
        )


def call_ollama_vlm(
    ollama_url: str,
    model: str,
    prompt: str,
    image_data_url: str,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    """
    Memanggil Ollama VLM dengan endpoint /v1/chat/completions (OpenAI compatible).

    :param ollama_url: URL lengkap endpoint, misal "http://localhost:11434/v1/chat/completions"
    :param model: Nama model Ollama, misal "qwen3-vl:latest"
    :param prompt: Teks prompt
    :param image_data_url: Data URL base64 untuk gambar
    :param temperature: Suhu sampling
    :param timeout: Timeout request (detik)
    :return: Teks hasil dari model
    """
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }

    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Format OpenAI-compatible: choices[0].message.content
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response from Ollama: {data}") from e


def process_pdf_with_vlm(
    pdf_path: str,
    ollama_url: str,
    model: str,
    language: str = "id",
    dpi: int = 144,
    max_pages: int | None = None,
) -> str:
    """
    Proses PDF dengan VLM dan kembalikan teks Markdown gabungan.

    :param pdf_path: Path ke PDF
    :param ollama_url: URL endpoint Ollama /v1/chat/completions
    :param model: Nama model VLM di Ollama
    :param language: "id" atau "en"
    :param dpi: Resolusi render PDF -> image
    :param max_pages: Batas maksimal halaman (untuk testing). None = semua halaman.
    :return: Markdown gabungan dari semua halaman
    """
    images = pdf_to_images(pdf_path, dpi=dpi)
    total_pages = len(images)
    if max_pages is not None:
        images = images[:max_pages]

    md_sections: List[str] = []
    md_sections.append(f"# Ringkasan Dokumen\n\n")
    md_sections.append(f"_Sumber_: `{os.path.basename(pdf_path)}`\n")
    md_sections.append(f"_Total halaman diolah_: {len(images)} dari {total_pages}\n\n")

    for page_num, img_bytes in images:
        print(f"[INFO] Memproses halaman {page_num}/{total_pages}...")

        # Pastikan image valid (opsional)
        # Hanya untuk sanity check, tidak wajib
        try:
            Image.open(io.BytesIO(img_bytes)).verify()
        except Exception:
            print(f"[WARN] Halaman {page_num}: image tidak valid, lanjut.")
            continue

        data_url = image_bytes_to_data_url(img_bytes)
        prompt = build_prompt(page_num, total_pages, language=language)

        try:
            content_md = call_ollama_vlm(
                ollama_url=ollama_url,
                model=model,
                prompt=prompt,
                image_data_url=data_url,
            )
        except Exception as e:
            print(f"[ERROR] Gagal memanggil VLM untuk halaman {page_num}: {e}")
            continue

        md_sections.append(f"## Halaman {page_num}\n\n")
        md_sections.append(content_md.strip() + "\n\n")

    return "".join(md_sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ekstraksi & penjelasan konten PDF per halaman dengan Ollama VLM."
    )
    parser.add_argument("pdf_path", help="Path ke file PDF yang akan diproses.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path file output Markdown (jika tidak diisi, hasil akan dicetak ke stdout).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/v1/chat/completions",
        help="URL endpoint Ollama /v1/chat/completions "
        "(default: http://localhost:11434/v1/chat/completions)",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl:2b-instruct-q4_K_M",
        help="Nama model Ollama VLM (default: qwen3-vl:2b-instruct-q4_K_M).",
    )
    parser.add_argument(
        "--language",
        default="id",
        choices=["id", "en"],
        help="Bahasa penjelasan (id/en), default: id.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=144,
        help="DPI untuk render PDF -> image (default: 144).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Batas jumlah halaman yang diproses (untuk testing). Default: semua.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.pdf_path):
        raise FileNotFoundError(f"File PDF tidak ditemukan: {args.pdf_path}")

    markdown = process_pdf_with_vlm(
        pdf_path=args.pdf_path,
        ollama_url=args.ollama_url,
        model=args.model,
        language=args.language,
        dpi=args.dpi,
        max_pages=args.max_pages,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"[INFO] Hasil disimpan ke: {args.output}")
    else:
        # Cetak ke stdout
        print(markdown)


if __name__ == "__main__":
    main()
