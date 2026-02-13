from __future__ import annotations
import os
import subprocess
from pathlib import Path
from app.core.pdf_downloader import download_all_from_idx_for_year
from app.core.ingestion_pipeline import run_ingestion as run_ingestion_pipeline
from app.core.embedding_pipeline import run_embedding_pipeline
from app.config import VLM
import sys
import logging
from typing import Callable, Dict

from app.logging_config import setup_logging
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
    banner_main = r"""
.______    __    __  .______    __________   ___ 
|   _  \  |  |  |  | |   _  \  |   ____\  \ /  / 
|  |_)  | |  |  |  | |  |_)  | |  |__   \  V  /  
|   ___/  |  |  |  | |   _  <  |   __|   >   <   
|  |      |  `--'  | |  |_)  | |  |____ /  .  \  
| _|       \______/  |______/  |_______/__/ \__\ 
"""
    banner_sub = r"""
                                       
▄█████▄ ▄████▄     █████▄ ▄████▄ ██████ 
██ ▄ ██ ██▄▄██ ▄▄▄ ██▄▄██ ██  ██   ██   
▀█████▀ ██  ██     ██▄▄█▀ ▀████▀   ██   
     ▀▀                                 
"""
    print(f"{BOLD}{RED}{banner_main}{BOLD}{WHITE}{banner_sub}{RESET}")

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
    document_ids_text = input(
        "Document ID (opsional, bisa lebih dari 1 pisahkan koma): "
    ).strip()
    document_ids = [
        doc_id.strip() for doc_id in document_ids_text.split(",") if doc_id.strip()
    ]

    if document_ids:
        limit = None
        print(f"{CYAN}Mulai ingestion dokumen berdasarkan Document ID...{RESET}")
    else:
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
            f"{CYAN}Mulai ingestion dokumen berstatus downloaded dan failed to ingest...{RESET}"
        )
    note = input("Note ingestion (opsional): ").strip() or None

    logger.info(
        "Ingestion started (limit=%s, document_ids=%s, note=%s, vlm_model=%s).",
        limit,
        document_ids,
        note,
        VLM.model,
    )
    run_ingestion_pipeline(
        limit=limit,
        document_ids=document_ids,
        note=note,
    )


def run_chainlit():
    app_path = Path("src/app/chainlit/app.py")
    if not app_path.exists():
        print(f"{RED}Aplikasi Chainlit tidak ditemukan: {app_path}{RESET}")
        return

    host = input("Host Chainlit [0.0.0.0]: ").strip() or "0.0.0.0"
    port = input("Port Chainlit [8000]: ").strip() or "8000"
    if not port.isdigit():
        print(f"{RED}Port harus berupa angka.{RESET}")
        return

    cmd = [
        "python",
        "-m",
        "chainlit",
        "run",
        str(app_path),
        "--host",
        host,
        "--port",
        port,
    ]
    logger.info("Menjalankan Chainlit dengan command: %s", " ".join(cmd))
    print(f"\n{BLUE}Menjalankan Chainlit di http://{host}:{port}{RESET}")
    subprocess.run(cmd, check=False)

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


def main_loop():
    actions: Dict[str, Callable[[], None]] = {
        "1": download_documents,
        "2": run_ingestion,
        "3": run_embedding,
        "4": run_chainlit,
    }

    while True:
        print_header()
        print_menu()
        choice = input("> Pilih menu [0-4]: ").strip()

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
