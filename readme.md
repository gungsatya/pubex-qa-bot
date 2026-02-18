# Pubex QA Bot

Mekanisme utama pada proyek ini:

1. `TUI` (`src/app/tui/main.py`)
2. `Download` dokumen IDX (`src/app/core/pdf_downloader.py`)
3. `Ingestion` slide berbasis Docling (`src/app/core/ingestion_pipeline.py`)  
   Alur: `PDF -> Markdown (Docling) -> split per page break -> simpan ke slides`
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

## Setup Docling Ingestion

Ingestion menggunakan Docling API endpoint dengan default:

- `DOCLING_API_BASE_URL=http://localhost:8081`
- `DOCLING_PRESET=granite_vision`
- `DOCLING_PAGE_BREAK_PLACEHOLDER=<!-- page break -->`
- `DOCLING_IMAGE_DPI=175` (untuk render image per slide)

Pastikan endpoint `${DOCLING_API_BASE_URL}/v1/chat/completions` tersedia sebelum menjalankan ingestion.
Gunakan Docling major versi 2 (`docling>=2,<3`) agar API VLM yang dipakai pipeline tersedia.

## Menjalankan TUI

```bash
PYTHONPATH=src python -m app.tui.main
```

## Menjalankan Chainlit langsung

```bash
python -m chainlit run src/app/chainlit/app.py --host 0.0.0.0 --port 8000
```
