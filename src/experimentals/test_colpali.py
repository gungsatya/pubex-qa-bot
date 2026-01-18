import os
from io import BytesIO
import base64
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import torch
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import lancedb
import numpy as np
from ollama import chat as ollama_chat

from colpali_engine.models import ColQwen2, ColQwen2Processor
import ollama



# -----------------------------
# Konfigurasi global
# -----------------------------
LANCEDB_URI = "data/lancedb_colqwen"
TABLE_NAME = "pdf_pages"
IMAGES_DIR = "data/page_images"

Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

def get_lancedb_table():
    db = lancedb.connect(LANCEDB_URI)
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return None

INITIAL_DB_READY = get_lancedb_table() is not None

# Kalau mau aman, set USE_GPU = False dulu
USE_GPU = False  # ubah ke True kalau mau coba lagi pakai CUDA

if USE_GPU and torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16  # lebih hemat dari bfloat16
else:
    device = "cpu"
    torch_dtype = torch.float32

print(f"[INFO] Loading ColQwen2 on {device} ({torch_dtype})")

model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0",
    torch_dtype=torch_dtype,
)
model.to(device)
model.eval()

processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")


# -----------------------------
# Util image ↔ base64 (kalau suatu saat mau pakai)
# -----------------------------
def encode_image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# -----------------------------
# LanceDB helpers
# -----------------------------


def create_or_overwrite_table(records: List[dict]):
    db = lancedb.connect(LANCEDB_URI)
    table = db.create_table(TABLE_NAME, data=records, mode="overwrite")
    return table


# -----------------------------
# Ollama LLM helper (vision/text)
# -----------------------------
def query_ollama_vision(query: str, images, model_name: str) -> str:
    """
    Panggil Ollama lokal (http://localhost:11434) dengan model vision,
    misal: llama3.2-vision, llava, qwen2-vl, dll.

    images: list[(PIL.Image, caption)]
    """
    if not model_name:
        return "Masukkan nama model Ollama (mis. 'llama3.2-vision' atau 'llava')."

    # Simpan gambar ke file sementara dan kirim path ke Ollama
    img_paths = []
    for i, (img, _) in enumerate(images):
        tmp_path = f"/tmp/colqwen_page_{i}.jpg"
        img.save(tmp_path, format="JPEG")
        img_paths.append(tmp_path)

    system_prompt = (
        "Kamu adalah asisten yang menjawab pertanyaan berdasarkan halaman-halaman PDF "
        "yang diberikan (sebagai gambar). Gunakan hanya informasi dari halaman tersebut. "
        "Sebutkan nomor halaman pada jawaban jika terlihat. Jawab dalam bahasa yang sama "
        "dengan query pengguna."
    )

    try:
        res = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                    "images": img_paths,
                },
            ],
        )

        # Akses konten dengan cara yang benar
        content = res["message"]["content"]

        # Bersihkan token spesial yang bocor
        for tok in ("<|im_start|>", "<|im_end|>", "<|begin_of_text|>", "<|end_of_text|>"):
            content = content.replace(tok, "")

        content = content.strip()

        if not content:
            return "Model hanya mengembalikan token spesial tanpa jawaban. Coba query lain atau model lain."

        return content

    except Exception as e:
        return f"Gagal memanggil Ollama: {e}"



# -----------------------------
# ColQwen2 encoding helpers
# -----------------------------
def convert_files(files) -> List[Image.Image]:
    images: List[Image.Image] = []
    for f in files:
        # pdf2image butuh poppler terinstall di OS (sudo apt install poppler-utils)
        imgs = convert_from_path(f, thread_count=4)
        images.extend(imgs)

    if len(images) >= 150:
        raise gr.Error("Jumlah total halaman (gambar) harus kurang dari 150.")
    return images


def encode_images_to_vectors(images: List[Image.Image], batch_size: int = 2) -> np.ndarray:
    """
    Encode PDF pages (PIL images) menjadi vektor (1 vektor per halaman) menggunakan ColQwen2.

    ColQwen2 (ColPali) output multi-vector (B, S, D).
    Di sini kita lakukan mean-pooling sepanjang S → (B, D) agar bisa disimpan di LanceDB.
    """
    global model, processor, device

    if device != str(model.device):
        model.to(device)

    dataloader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x).to(model.device),
    )

    vectors = []

    for batch_doc in tqdm(dataloader, desc="Encoding pages"):
        with torch.no_grad():
            # pastikan semua tensor di device
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)  # (B, S, D) multi-vector

            if isinstance(embeddings_doc, torch.Tensor):
                multi_vec = embeddings_doc
            else:
                # fallback kalau model return ModelOutput
                multi_vec = embeddings_doc.embeddings  # type: ignore[attr-defined]

            # Mean pooling → 1 vektor per halaman
            pooled = multi_vec.mean(dim=1)  # (B, D)

        vectors.append(pooled.cpu())

    vectors = torch.cat(vectors, dim=0).numpy().astype("float32")
    return vectors


def encode_query_to_vector(query: str) -> np.ndarray:
    """
    Encode query text menjadi 1 vektor menggunakan ColQwen2 (mean-pooling).
    """
    global model, processor, device

    if device != str(model.device):
        model.to(device)

    with torch.no_grad():
        batch_query = processor.process_queries([query]).to(model.device)
        embeddings_query = model(**batch_query)  # (1, S, D)

        if isinstance(embeddings_query, torch.Tensor):
            multi_vec = embeddings_query
        else:
            multi_vec = embeddings_query.embeddings  # type: ignore[attr-defined]

        pooled = multi_vec.mean(dim=1)[0]  # (D,)

    return pooled.cpu().numpy().astype("float32")


# -----------------------------
# Gradio callback functions
# -----------------------------
def index(files):
    """
    1. Convert PDF → list of images
    2. Simpan image ke disk (IMAGES_DIR)
    3. Encode ke vektor dengan ColQwen2
    4. Simpan ke LanceDB (overwrite table)
    """
    if not files:
        raise gr.Error("Silakan upload minimal satu PDF.")

    all_images: List[Image.Image] = []
    image_paths: List[str] = []
    pages: List[int] = []

    for f in files:
        # f = filepath string karena type="filepath"
        pdf_path = Path(f)
        pdf_name = pdf_path.stem  # nama file tanpa .pdf

        # Konversi setiap page
        pdf_images = convert_from_path(str(pdf_path), thread_count=4)

        for i, img in enumerate(pdf_images, start=1):
            # Simpan image ke disk
            out_path = Path(IMAGES_DIR) / f"{pdf_name}_page_{i}.jpg"
            img.save(out_path, format="JPEG")

            all_images.append(img)
            image_paths.append(str(out_path))
            pages.append(i)

    if len(all_images) >= 150:
        raise gr.Error("Jumlah total halaman (gambar) harus kurang dari 150.")

    # Encode semua image menjadi vektor
    vectors = encode_images_to_vectors(all_images, batch_size=2)

    # Normalisasi (cosine)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    normed_vectors = (vectors / norms).astype("float32")

    records = []
    for idx, vec in enumerate(normed_vectors):
        records.append(
            {
                "vector": vec.tolist(),
                "image_path": image_paths[idx],
                "page": int(pages[idx]),
            }
        )

    create_or_overwrite_table(records)

    msg = f"Berhasil mengindeks {len(all_images)} halaman ke LanceDB."
    # db_ready=True, imgs tidak perlu dipakai lagi (bisa dikosongkan)
    return msg, True, []



def search(query: str, k: int, db_ready: bool, ollama_model_name: str):
    """
    1. Encode query → vektor
    2. Search di LanceDB (ANN cosine)
    3. Load gambar dari disk berdasarkan image_path
    4. Panggil Ollama untuk generate jawaban berdasarkan halaman tersebut
    """
    if not db_ready:
        raise gr.Error("Dokumen belum diindeks. Silakan klik 'Index documents' dulu.")

    if not query:
        raise gr.Error("Masukkan query.")

    table = get_lancedb_table()
    if table is None:
        raise gr.Error("Tabel LanceDB tidak ditemukan. Silakan index ulang dokumen.")

    k = int(k)
    if k < 1:
        k = 1

    # 1. Encode query
    q_vec = encode_query_to_vector(query)

    # 2. Vector search di LanceDB
    results = (
        table.search(q_vec.tolist())
        .metric("cosine")
        .limit(k)
        .to_list()
    )

    if not results:
        return [], "Tidak ada halaman yang relevan ditemukan."

    # 3. Load images dari disk sesuai image_path
    top_images = []
    for row in results:
        image_path = row.get("image_path")
        page_no = row.get("page", None)

        if image_path is None:
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            # Kalau file hilang/korup, skip
            continue

        caption = f"Page {page_no}" if page_no is not None else "Page ?"
        top_images.append((img, caption))

    if not top_images:
        return [], "Gagal memuat gambar dari disk. Coba re-index dokumen."

    # 4. Panggil Ollama
    answer = query_ollama_vision(query, top_images, ollama_model_name)

    return top_images, answer


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 ColQwen2 + LanceDB + Ollama RAG")
    gr.Markdown(
        """
Demo ini:
1️⃣ Mengubah PDF menjadi gambar halaman  
2️⃣ Mengindeks embedding gambar dengan **ColQwen2 (ColPali)** ke **LanceDB**  
3️⃣ Mencari halaman paling relevan dengan query  
4️⃣ Menggunakan **Ollama** (model vision) untuk menjawab berdasarkan halaman tersebut  

💡 Pastikan:
- `ollama` sudah jalan (`ollama serve` otomatis kalau app-nya aktif)
- Model vision sudah dipull, misalnya: `ollama pull llama3.2-vision`
"""
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 1️⃣ Upload & Index PDFs")
            file = gr.File(
                file_types=[".pdf"],
                file_count="multiple",
                label="Upload PDFs",
                type="filepath"
            )

            convert_button = gr.Button("🔄 Index documents")
            message = gr.Textbox("Files not yet uploaded", label="Status")

            ollama_model = gr.Textbox(
                placeholder="contoh: llama3.2-vision atau llava",
                label="Nama model Ollama (vision)",
                value="qwen3-vl:2b-instruct-q4_K_M",
            )

            db_ready = gr.State(value=INITIAL_DB_READY)
            imgs = gr.State(value=[])

        with gr.Column(scale=3):
            gr.Markdown("## 2️⃣ Search & Ask")
            query = gr.Textbox(placeholder="Tulis pertanyaan di sini", label="Query")
            k = gr.Slider(minimum=1, maximum=10, step=1, label="Jumlah halaman yang di-retrieve", value=5)

    search_button = gr.Button("🔍 Search", variant="primary")
    output_gallery = gr.Gallery(label="Retrieved Pages", height=600, show_label=True)
    output_text = gr.Textbox(label="AI Response", lines=8)

    # Wiring
    convert_button.click(
        fn=index,
        inputs=[file],
        outputs=[message, db_ready, imgs],  # imgs tetap diisi [], tidak dipakai
    )

    search_button.click(
        fn=search,
        inputs=[query, k, db_ready, ollama_model],
        outputs=[output_gallery, output_text],
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True)
