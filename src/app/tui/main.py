from __future__ import annotations
import os
from pathlib import Path
from app.core.pdf_downloader import download_all_from_idx_for_year
from app.core.ingestion_pipeline import (
    run_ingestion as run_ingestion_pipeline,
    run_ingestion_multi_model as run_ingestion_multi_model_pipeline,
)
from app.core.embedding_pipeline import run_embedding_pipeline
from app.core.ollama_test_pipeline import run_ollama_chat_test as run_ollama_chat_test_pipeline
import sys
import logging
from typing import Callable, Dict

from app.logging_config import setup_logging
from app.config import DEFAULT_OLLAMA_TEST_OUTPUT_DIR, DEFAULT_VLM_MODEL
logger = logging.getLogger(__name__)

# ===== ANSI Colors =====
RESET = "\033[0m"
BOLD = "\033[1m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

def _ask_doc_type() -> str | None:
    print("\nPilih jenis dokumen:")
    print("  [1] Pubex (Public Expose)")
    print("  [2] Laporan Keuangan")
    choice = input("Pilihan [1-2]: ").strip()

    if choice == "1":
        return "pubex"
    if choice == "2":
        return "financial_report"

    print("Pilihan tidak valid.")
    return None

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pause(msg="\nTekan ENTER untuk kembali ke menu..."):
    input(msg)


def print_header():
    clear_screen()

    # ===== ASCII ART BANNER =====
    print(f"""{BOLD}{RED}
.______    __    __  .______    __________   ___ 
|   _  \  |  |  |  | |   _  \  |   ____\  \ /  / 
|  |_)  | |  |  |  | |  |_)  | |  |__   \  V  /  
|   ___/  |  |  |  | |   _  <  |   __|   >   <   
|  |      |  `--'  | |  |_)  | |  |____ /  .  \  
| _|       \______/  |______/  |_______/__/ \__\ 
                                                                                                    
{BOLD}{WHITE}                                       
▄█████▄ ▄████▄     █████▄ ▄████▄ ██████ 
██ ▄ ██ ██▄▄██ ▄▄▄ ██▄▄██ ██  ██   ██   
▀█████▀ ██  ██     ██▄▄█▀ ▀████▀   ██   
     ▀▀                                 
{RESET}""")

    # ===== INFO BOX =====
    print(f"{CYAN}" + "─" * 62 + f"{RESET}")
    print(f"{CYAN}{RESET}  by Satya Wibawa")
    print(f"{CYAN}{RESET}  version 0.1")
    print(f"{CYAN}" + "─" * 62 + f"{RESET}\n")


def print_menu():
    print(f"{BOLD}Main Menu{RESET}\n")
    print(f"  [1] Download Dokumen{RESET}")
    print(f"  [2] Ingestion Pipeline{RESET}")
    print(f"  [3] Embedding Pipeline{RESET}")
    print(f"  [4] Run QA-Bot (Chainlit){RESET}")
    print(f"  [5] Ingestion Multi-Model (Compare){RESET}")
    print(f"  [6] Ollama Chat Test (Image Prompt){RESET}")
    print(f"  [0] {RED}Keluar{RESET}\n")


def download_documents():
    print(f"\n{YELLOW}Download Dokumen dari IDX{RESET}")

    doc_type = _ask_doc_type()
    if not doc_type:
        return

    year_text = input("Masukkan tahun dokumen (mis. 2023): ").strip()
    if not year_text.isdigit():
        print(f"{RED}Tahun harus berupa angka.{RESET}")
        return

    year = int(year_text)

    base_dir = Path("src/data/documents")

    print(
        f"\nMulai download {doc_type} tahun {year} "
        f"ke folder: {base_dir}/{doc_type}/{year}"
    )
    download_all_from_idx_for_year(year, doc_type, base_dir)
    print(f"\n{GREEN}Proses download selesai (cek log & DB).{RESET}")



def run_ingestion():
    print(f"\n{YELLOW}Ingestion Pipeline{RESET}")
    limit_text = input(
        "Limit dokumen (kosong = semua, contoh 10): "
    ).strip()
    if limit_text:
        if not limit_text.isdigit():
            print(f"{RED}Limit harus berupa angka.{RESET}")
            return
        limit = int(limit_text)
    else:
        limit = None

    print(
        f"{CYAN}Mulai ingestion dokumen berstatus downloaded..."
        f"{RESET}"
    )
    logger.info("Ingestion started (limit=%s).", limit)
    run_ingestion_pipeline(limit=limit)


def run_ingestion_multi_model():
    print(f"\n{YELLOW}Ingestion Multi-Model (Compare){RESET}")
    limit_text = input(
        "Limit dokumen (kosong = semua, contoh 10): "
    ).strip()
    if limit_text:
        if not limit_text.isdigit():
            print(f"{RED}Limit harus berupa angka.{RESET}")
            return
        limit = int(limit_text)
    else:
        limit = None

    models_text = input(
        "Masukkan daftar model (pisahkan dengan koma): "
    ).strip()
    models = [m.strip() for m in models_text.split(",") if m.strip()]
    if not models:
        print(f"{RED}Model tidak boleh kosong.{RESET}")
        return

    print(
        f"{CYAN}Mulai ingestion multi-model untuk: {', '.join(models)}..."
        f"{RESET}"
    )
    logger.info("Ingestion multi-model started (limit=%s, models=%s).", limit, models)
    run_ingestion_multi_model_pipeline(limit=limit, models=models)


def run_chainlit():
    print(f"\n{BLUE}Menjalankan Chainlit...{RESET}")
    logger.info("Chainlit stub called.")
    print("Belum diimplementasikan.")

def run_embedding():
    print(f"\n{YELLOW}Embedding Pipeline{RESET}")
    limit_text = input(
        "Limit slide (kosong = semua, contoh 100): "
    ).strip()
    if limit_text:
        if not limit_text.isdigit():
            print(f"{RED}Limit harus berupa angka.{RESET}")
            return
        limit = int(limit_text)
    else:
        limit = None

    print(f"{CYAN}Mulai embedding content_text...{RESET}")
    logger.info("Embedding started (limit=%s).", limit)
    run_embedding_pipeline(limit=limit)


def run_ollama_chat_test():
    print(f"\n{YELLOW}Ollama Chat Test (Image Prompt){RESET}")

    image_path_text = input("Path image: ").strip()
    # if not image_path:
    #     print(f"{RED}Path image tidak boleh kosong.{RESET}")
    #     return
    image_path = image_path_text or "/home/gungsatya/Projects/pubex-qa-bot/src/data/documents/pubex/2025/AALI/slides/2097b41d-112b-48b6-96d9-72e7412aaa20/005.png"

    model_text = input(f"Model (default: {DEFAULT_VLM_MODEL}): ").strip()
    model = model_text or DEFAULT_VLM_MODEL

    output_dir_text = input(
        f"Output folder (default: {DEFAULT_OLLAMA_TEST_OUTPUT_DIR}): "
    ).strip()
    output_dir = output_dir_text or None

    try:
        output_path = run_ollama_chat_test_pipeline(
            image_path=image_path,
            model=model,
            output_dir=output_dir,
        )
    except Exception as exc:
        logger.exception("Ollama chat test error: %s", exc)
        print(f"{RED}Error: {exc}{RESET}")
        return

    print(f"{GREEN}Selesai. Output disimpan di:{RESET} {output_path}")


def main_loop():
    actions: Dict[str, Callable[[], None]] = {
        "1": download_documents,
        "2": run_ingestion,
        "3": run_embedding,
        "4": run_chainlit,
        "5": run_ingestion_multi_model,
        "6": run_ollama_chat_test,
    }

    while True:
        print_header()
        print_menu()
        choice = input("> Pilih menu [0-6]: ").strip()

        if choice == "0":
            print(f"\n{RED}Keluar dari QA-Bot. Bye!{RESET}")
            break

        action = actions.get(choice)
        if not action:
            print(f"\n{RED}Pilihan tidak dikenal.{RESET}")
            pause()
            continue

        try:
            action()
        except Exception as exc:
            logger.exception("CLI error: %s", exc)
            print(f"{RED}Error: {exc}{RESET}")

        pause()


def main():
    setup_logging()
    logger.info("Starting QA-Bot CLI")
    try:
        main_loop()
    except KeyboardInterrupt:
        print(f"\n{RED}Interrupted by user.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
