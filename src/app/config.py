# src/app/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # mengarah ke `src/..`
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

DATABASE_URL = os.getenv("DATABASE_URL")

DEFAULT_LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080")
DEFAULT_LLAMA_CPP_TIMEOUT = int(os.getenv("LLAMA_CPP_TIMEOUT_SECONDS", "300"))
DEFAULT_VLM_MODEL = os.getenv("VLM_MODEL") or "vlm-model"
DEFAULT_DPI = int(os.getenv("INGESTION_PDF_DPI", "120"))
DEFAULT_LLAMA_CPP_TEST_OUTPUT_DIR = os.getenv(
    "LLAMA_CPP_TEST_OUTPUT_DIR",
    str(BASE_DIR / "src/data/llama_cpp_tests"),
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_API_BASE_URL = os.getenv("TELEGRAM_API_BASE_URL") or "https://api.telegram.org"
TELEGRAM_DISABLE_NOTIFICATION = (
    os.getenv("TELEGRAM_DISABLE_NOTIFICATION", "false").strip().lower()
    in {"1", "true", "yes", "y", "on"}
)
