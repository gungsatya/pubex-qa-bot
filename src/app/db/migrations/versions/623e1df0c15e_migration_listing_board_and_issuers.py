"""migration listing_board and issuers

Revision ID: 623e1df0c15e
Revises: a685bb03e168
Create Date: 2026-01-18 23:55:39.537807

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import os
import csv


# revision identifiers, used by Alembic.
revision: str = '623e1df0c15e'
down_revision: Union[str, None] = 'a685bb03e168'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _get_csv_path() -> str:
    """
    Lokasi file CSV relatif terhadap file migration ini:

    migration:  src/app/db/migrations/versions/xxxx.py
    csv:        src/data/daftarsaham.csv

    Jadi: .. (migrations) -> .. (db) -> .. (app) -> .. (src) -> data/daftarsaham.csv
    """
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.abspath(
        os.path.join(
            current_dir,
            "..",  # migrations
            "..",  # db
            "..",  # app
            "..",  # src
            "data",
            "daftarsaham.csv",
        )
    )
    return csv_path


def upgrade() -> None:
    connection = op.get_bind()

    # 1) Mapping Papan Pencatatan → listing_board_code
    #    Silakan sesuaikan kalau kamu mau kode lain.
    board_name_to_code = {
        "Utama": "U",
        "Pengembangan": "P",
        "Akselerasi": "A",
        "Ekonomi Baru": "EB",
        "Pemantauan Khusus": "PK",
    }

    # 2) Siapkan "table" tiruan untuk bulk_insert
    listing_boards_table = sa.table(
        "listing_boards",
        sa.column("code", sa.CHAR(length=10)),
        sa.column("name", sa.String()),
    )

    issuers_table = sa.table(
        "issuers",
        sa.column("code", sa.CHAR(length=10)),
        sa.column("listing_board_code", sa.CHAR(length=10)),
        sa.column("name", sa.String()),
    )

    csv_path = _get_csv_path()
    if not os.path.exists(csv_path):
        raise RuntimeError(
            f"CSV daftar saham tidak ditemukan di: {csv_path}. "
            "Pastikan kamu sudah mengekspor daftarsaham.xlsx ke src/data/daftarsaham.csv "
            "dengan header: Kode, Nama Perusahaan, Papan Pencatatan."
        )

    listing_boards_seen = set()
    issuers_rows = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("Kode") or "").strip()
            name = (row.get("Nama Perusahaan") or "").strip()
            board_name = (row.get("Papan Pencatatan") or "").strip()

            if not code or not name or not board_name:
                # Lewatkan baris yang tidak lengkap
                continue

            # Normalisasi nama papan pencatatan ke code
            if board_name not in board_name_to_code:
                raise RuntimeError(
                    f"Papan Pencatatan tidak dikenal: '{board_name}' untuk kode '{code}'. "
                    "Tambahkan mapping-nya di board_name_to_code di migration ini."
                )

            board_code = board_name_to_code[board_name]

            # Simpan listing_board unik (name + code)
            listing_boards_seen.add((board_code, board_name))

            # Siapkan row issuers
            issuers_rows.append(
                {
                    "code": code,
                    "listing_board_code": board_code,
                    "name": name,
                }
            )

    # 3) Insert ke listing_boards (distinct by papan pencatatan)
    listing_boards_rows = [
        {"code": code, "name": name} for (code, name) in sorted(listing_boards_seen)
    ]

    if listing_boards_rows:
        op.bulk_insert(listing_boards_table, listing_boards_rows)

    # 4) Insert ke issuers (semua row saham)
    if issuers_rows:
        op.bulk_insert(issuers_table, issuers_rows)


def downgrade() -> None:
    # Untuk downgrade yang simpel, kita hapus data hasil seed ini.
    # Kalau di environment kamu sudah ada data lain,
    # bisa dipersempit pakai subquery/WHERE dari CSV.
    connection = op.get_bind()

    # Hapus semua issuers
    connection.execute(sa.text("DELETE FROM issuers;"))

    # Hapus semua listing_boards
    connection.execute(sa.text("DELETE FROM listing_boards;"))
