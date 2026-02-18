Struktur Tabel pada aplikasi ini adalah sebagai berikut.

## `listing_boards`

Tabel untuk papan pencatatan emiten (mis. Main Board, Development Board, dsb.).

| Kolom  | Tipe (PostgreSQL) | Konstraint                   | Keterangan                                                                                   |
| ------ | ----------------- | ---------------------------- | -------------------------------------------------------------------------------------------- |
| `code` | `CHAR(10)`        | **PK**, `UNIQUE`, `NOT NULL` | Kode papan pencatatan. Bisa diselaraskan dengan kode resmi bursa (mis. “U”, “UEB”, dst.). |
| `name` | `VARCHAR`         | `NOT NULL`                   | Nama papan pencatatan. Contoh: “Utama”, “Utama-Ekonomi Baru”, dst.                            |


## `issuers`

Tabel master emiten yang menerbitkan dokumen (pubex, laporan keuangan, dll.)

| Kolom                | Tipe (PostgreSQL) | Konstraint                              | Keterangan                                                               |
| -------------------- | ----------------- | --------------------------------------- | ------------------------------------------------------------------------ |
| `code`               | `CHAR(10)`        | **PK**, `UNIQUE`, `NOT NULL`            | Kode emiten. Biasanya sama dengan ticker di bursa (mis. “ABMM”, “TLKM”). |
| `listing_board_code` | `CHAR(10)`        | `FK → listing_boards(code)`, `NOT NULL` | Mengindikasikan emiten ini tercatat di papan yang mana.                  |
| `name`               | `VARCHAR`         | `NOT NULL`                              | Nama lengkap emiten. Contoh: “PT ABM Investama Tbk”.                     |

## `collections`

Tabel untuk kelompok dokumen tematik, misalnya “Pubex 2025”, “Laporan Keuangan 2025”.

| Kolom      | Tipe (PostgreSQL) | Konstraint                   | Keterangan                                                                                                                                     |
| ---------- | ----------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `code`     | `VARCHAR(50)`     | **PK**, `UNIQUE`, `NOT NULL` | Kode koleksi, misalnya `PUBEX_2025`, `FS_2025`.                                                                                                |
| `name`     | `VARCHAR`         | `NOT NULL`                   | Nama koleksi, misalnya “Public Expose 2025”, “Laporan Keuangan 2025”.                                                                          |
| `metadata` | `JSONB`           | `NULL` diperbolehkan         | Metadata fleksibel untuk koleksi: bisa simpan tahun, jenis dokumen, versi pipeline ingest, dsb. Contoh: `{"year": 2025, "category": "pubex"}`. |

## `document_status`

Tabel referensi status dokumen (enum-ish) untuk melacak lifecycle ingestion/pipeline.

| Kolom  | Tipe (PostgreSQL) | Konstraint                   | Keterangan                                                                                                                                    |
| ------ | ----------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`   | `SMALLINT`        | **PK**, `UNIQUE`, `NOT NULL` | ID status (setara **tinyint** dalam desain awal). Nilai bisa dipakai seperti 1: downloaded, 2: parsed, 3: embedded, 4: ready, 5: failed, dll. |
| `name` | `VARCHAR`         | `NOT NULL`                   | Nama status, human-readable. Contoh: “downloaded”, “ingested”, “embedded”, “ready”, “failed”.                                                 |


## `documents`

Tabel utama untuk menyimpan satu entitas dokumen (satu file pubex / laporan keuangan / dsb.)

| Kolom             | Tipe (PostgreSQL) | Konstraint                                                   | Keterangan                                                                                                                                                         |
| ----------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `id`              | `UUID`            | **PK**, `DEFAULT gen_random_uuid()` (opsional), `NOT NULL`   | Identitas unik dokumen. Baik untuk ingestion paralel dan integrasi sistem lain.                                                                                    |
| `collection_code` | `VARCHAR(50)`     | `FK → collections(code)`, `NOT NULL`                         | Mengelompokkan dokumen ke dalam koleksi, misalnya `PUBEX_2025` atau `FS_2025`.                                                                                     |
| `issuer_code`     | `CHAR(10)`        | `FK → issuers(code)`, `NOT NULL`                             | Menandai dokumen ini milik emiten mana (contoh: “ABMM”).                                                                                                           |
| `checksum`        | `TEXT`            | `UNIQUE` (opsional), `NOT NULL` disarankan                   | Fingerprint file, berguna untuk deduplikasi (mis. hash MD5/SHA256).                                                                                                |
| `name`            | `VARCHAR`         | `NOT NULL`                                                   | Nama dokumen yang lebih manusiawi. Contoh: “Public Expose ABMM 2025”, “Laporan Keuangan Tahunan 2025”.                                                             |
| `publish_at`      | `TIMESTAMPTZ`     | `NULL` bisa, tapi `NOT NULL` dianjurkan                      | Tanggal publikasi resmi dokumen. Berguna untuk filter temporal (“tampilkan pubex 2023–2025”).                                                                      |
| `status`          | `SMALLINT`        | `FK → document_status(id)`, `NOT NULL`                       | Status lifecycle dokumen (downloaded, parsed, embedded, ready, dsb.).                                                                                              |
| `file_path`       | `VARCHAR`         | `NOT NULL`                                                   | Lokasi file di sistem (mis. path lokal atau URL ke object storage). Contoh: `src/data/documents/abmm_pubex_2025.pdf`.                                              |
| `metadata`        | `JSONB`           | `NULL` diperbolehkan                                         | Field fleksibel untuk informasi tambahan: sumber URL, tipe dokumen, versi parser, catatan regulatory, dsb. Contoh: `{"source": "IDX", "lang": "id", "pages": 45}`. |
| `created_at`      | `TIMESTAMPTZ`     | `NOT NULL`, `DEFAULT now()`                                  | Waktu saat dokumen pertama kali dicatat di sistem (waktu ingest).                                                                                                  |
| `updated_at`      | `TIMESTAMPTZ`     | `NOT NULL`, `DEFAULT now()` + trigger auto-update (opsional) | Waktu saat dokumen terakhir di-update (status berubah, metadata diubah, dsb.).                                                                                     |

## `slides`

Tabel untuk menyimpan hasil ekstraksi dokumen per slide/halaman. Ini yang jadi basis RAG.

| Kolom                 | Tipe (PostgreSQL)   | Konstraint                                                   | Keterangan                                                                                                                                                                                                   |
| --------------------- | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `id`                  | `UUID`              | **PK**, `DEFAULT gen_random_uuid()` (opsional), `NOT NULL`   | ID unik per slide/halaman.                                                                                                                                                                                   |
| `document_id`         | `UUID`              | `FK → documents(id)`, `NOT NULL`                             | Menandakan slide ini berasal dari dokumen yang mana. Relasi 1 dokumen : N slides.                                                                                                                            |
| `content_text`        | `TEXT`              | `NULL` diperbolehkan                                         | Hasil ekstraksi teks dari slide. Bisa berasal dari OCR, Docling, atau parser PDF biasa. Ini yang akan di-embedding.                                                                                          |
| `content_text_vector` | `VECTOR` (pgvector) | `NULL` diperbolehkan, index disarankan                       | Representasi embedding dari `content_text`. Tipe `vector` berasal dari ekstensi **pgvector** (`CREATE EXTENSION vector;`). Dimensi ditentukan oleh model (mis. `vector(1024)`).                              |
| `content_base_64`     | `TEXT`              | `NULL` diperbolehkan                                         | Representasi slide sebagai image yang di-encode base64. Berguna untuk audit visual atau front-end yang ingin menampilkan slide asli. Bila storage jadi berat, bisa diganti jadi `image_path`.               |
| `metadata`            | `JSONB`             | `NULL` diperbolehkan                                         | Metadata per slide: nomor slide, nomor chunk dsb. Contoh: `{"slide_no": 12, "chunk_index": 0 }`.                            |
| `created_at`          | `TIMESTAMPTZ`       | `NOT NULL`, `DEFAULT now()`                                  | Waktu slide pertama kali disimpan ke DB (saat pipeline ekstraksi berjalan).                                                                                                                                  |
| `updated_at`          | `TIMESTAMPTZ`       | `NOT NULL`, `DEFAULT now()` + trigger auto-update (opsional) | Waktu terakhir slide diubah (mis. re-embedding dengan model baru).                                                                                                                                           |
