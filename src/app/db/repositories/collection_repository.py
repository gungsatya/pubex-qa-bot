from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Collection


class CollectionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_code(self, code: str) -> Optional[Collection]:
        stmt = select(Collection).where(Collection.code == code)
        result = self.session.execute(stmt)
        return result.scalars().first()

    def get_or_create_for_type_year(self, doc_type: str, year: int) -> Collection:
        """
        doc_type: 'pubex' atau 'financial_report'
        -> code  : PUBEX_2025, FINREP_2025
        -> name  : 'Public Expose 2025', 'Laporan Keuangan 2025'
        -> metadata: {"type": doc_type, "year": year}
        """
        doc_type = doc_type.lower()

        if doc_type == "pubex":
            code = f"PUBEX_{year}"
            name = f"Public Expose {year}"
        elif doc_type == "financial_report":
            code = f"FINREP_{year}"
            name = f"Laporan Keuangan {year}"
        else:
            code = f"{doc_type.upper()}_{year}"
            name = f"{doc_type.title()} {year}"

        collection = self.get_by_code(code)
        if collection:
            return collection

        collection = Collection(
            code=code,
            name=name,
            metadata={"type": doc_type, "year": year},
        )
        self.session.add(collection)
        self.session.flush()  # supaya collection.code/uuid langsung terisi bila perlu
        return collection
