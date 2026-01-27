# src/app/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # mengarah ke `src/..`
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

DATABASE_URL = os.getenv("DATABASE_URL")
DEFAULT_VLM_MODEL = os.getenv("VLM_MODEL") or "qwen3-vl:2b-instruct-q4_K_M"
DEFAULT_DPI = int(os.getenv("INGESTION_PDF_DPI", "144"))