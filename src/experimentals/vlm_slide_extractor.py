#!/usr/bin/env python3
"""
pdf_vlm_extractor.py

Script untuk:
- Membaca file PDF
- Mengubah tiap halaman menjadi image
 - Mengirim image ke VLM via request OpenAI-compatible API
- Menghasilkan penjelasan konten halaman dalam bahasa Indonesia
- Menggabungkan hasil ke dalam satu file Markdown

Dependensi:
    pip install pymupdf pillow requests

Contoh pemakaian:
    python pdf_vlm_extractor.py input.pdf -o output.md \
        --model deepseek-vl:7b \
        --base-url https://api.deepseek.com
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


def _resolve_api_key(api_key: str | None, base_url: str) -> str:
    if api_key:
        return api_key
    if _is_local_endpoint(base_url):
        return "ollama"
    raise RuntimeError("DEEPSEEK_API_KEY belum di-set untuk endpoint non-lokal.")


def call_vlm(
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    image_data_url: str,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    """
    Memanggil VLM dengan API OpenAI-compatible.

    :param base_url: Base URL endpoint, misal "https://api.deepseek.com"
    :param api_key: API key untuk endpoint (opsional untuk lokal)
    :param model: Nama model VLM, misal "deepseek-vl:7b"
    :param prompt: Teks prompt
    :param image_data_url: Data URL base64 untuk gambar
    :param temperature: Suhu sampling
    :param timeout: Timeout request (detik)
    :return: Teks hasil dari model
    """
    endpoint = _resolve_endpoint(base_url)
    api_key = _resolve_api_key(api_key, base_url)
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "ollama":
        headers["Authorization"] = f"Bearer {api_key}"
    supports_images = _is_local_endpoint(base_url) or os.getenv(
        "VLM_IMAGE_SUPPORT", ""
    ).lower() in {"1", "true", "yes"}
    if supports_images:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    else:
        print(
            "[WARN] Endpoint tidak mendukung image_url. "
            "Kirim prompt teks saja. Set VLM_IMAGE_SUPPORT=true jika didukung."
        )
        content = prompt
    payload = {
        "model": model,
        "temperature": temperature,
        "stream": False,
        "messages": [
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
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response from VLM: {data}") from e


def process_pdf_with_vlm(
    pdf_path: str,
    base_url: str,
    api_key: str | None,
    model: str,
    language: str = "id",
    dpi: int = 144,
    max_pages: int | None = None,
) -> str:
    """
    Proses PDF dengan VLM dan kembalikan teks Markdown gabungan.

    :param pdf_path: Path ke PDF
    :param base_url: Base URL endpoint OpenAI-compatible
    :param api_key: API key untuk endpoint (opsional untuk lokal)
    :param model: Nama model VLM
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
            content_md = call_vlm(
                base_url=base_url,
                api_key=api_key,
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
        description="Ekstraksi & penjelasan konten PDF per halaman dengan VLM (OpenAI-compatible)."
    )
    parser.add_argument("pdf_path", help="Path ke file PDF yang akan diproses.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path file output Markdown (jika tidak diisi, hasil akan dicetak ke stdout).",
    )
    parser.add_argument(
        "--base-url",
        "--ollama-url",
        dest="base_url",
        default=os.getenv("DEEPSEEK_BASE_URL", "http://localhost:11434"),
        help="Base URL endpoint OpenAI-compatible "
        "(default: DEEPSEEK_BASE_URL atau http://localhost:11434)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DEEPSEEK_API_KEY"),
        help="API key untuk endpoint (default: DEEPSEEK_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default="deepseek-vl:7b",
        help="Nama model VLM (default: deepseek-vl:7b).",
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
        base_url=args.base_url,
        api_key=args.api_key,
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
