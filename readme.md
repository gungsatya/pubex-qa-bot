# Pubex QA Bot

Mekanisme utama pada proyek ini:

1. `TUI` (`src/app/tui/main.py`)
2. `Download` dokumen IDX (`src/app/core/pdf_downloader.py`)
3. `Ingestion` slide (`src/app/core/ingestion_pipeline.py`)
4. `Embedding` (`src/app/core/embedding_pipeline.py`)
5. `Chainlit` app (`src/app/chainlit/app.py`)

## Setup Ollama

1. Jalankan script helper untuk menarik model:

```bash
./start_ollama.sh --pull-only
```

2. Jalankan server Ollama:

```bash
ollama serve
```

Default endpoint yang digunakan aplikasi: `http://127.0.0.1:11434`.

## Menjalankan TUI

```bash
PYTHONPATH=src python -m app.tui.main
```

## Menjalankan Chainlit langsung

```bash
python -m chainlit run src/app/chainlit/app.py --host 0.0.0.0 --port 8000
```
