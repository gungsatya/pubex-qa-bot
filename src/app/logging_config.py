import logging
import logging.config
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env jika ada
load_dotenv()

# Baca konfigurasi level dari ENV
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", None)

# Optional file output
LOG_DIR = Path("logs")
if LOG_FILE:
    LOG_DIR.mkdir(exist_ok=True)
    LOG_FILE = LOG_DIR / LOG_FILE


def get_logging_config():
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "level": LOG_LEVEL,
        },
    }

    if LOG_FILE:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "formatter": "file",
            "filename": str(LOG_FILE),
            "level": LOG_LEVEL,
        }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "file": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(module)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": handlers,
        "root": {
            "level": LOG_LEVEL,
            "handlers": list(handlers.keys()),
        },
    }


def setup_logging():
    config = get_logging_config()
    logging.config.dictConfig(config)
