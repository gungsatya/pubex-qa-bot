from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Issuer


class IssuerRepository:
    """Repository untuk operasi terkait tabel issuers."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_all(self) -> List[Issuer]:
        """Ambil semua issuer, diurut berdasarkan code."""
        stmt = select(Issuer).order_by(Issuer.code)
        result = self.session.execute(stmt)
        return list(result.scalars().all())

    def get_by_code(self, code: str) -> Optional[Issuer]:
        stmt = select(Issuer).where(Issuer.code == code)
        result = self.session.execute(stmt)
        return result.scalars().first()
