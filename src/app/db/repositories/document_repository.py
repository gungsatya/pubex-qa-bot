from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_checksum(self, checksum: str) -> Optional[Document]:
        stmt = select(Document).where(Document.checksum == checksum)
        result = self.session.execute(stmt)
        return result.scalars().first()

    def create_document(
        self,
        *,
        collection_code: str,
        issuer_code: str,
        checksum: str,
        name: str,
        file_path: str,
        file_md_path: Optional[str] = None,
        publish_at: Optional[datetime] = None,
        status_id: int,
        metadata: dict | None = None,
    ) -> Document:
        doc = Document(
            collection_code=collection_code,
            issuer_code=issuer_code,
            checksum=checksum,
            name=name,
            publish_at=publish_at,
            status=status_id,
            file_path=file_path,
            file_md_path=file_md_path,
            document_metadata=metadata or {},
        )
        self.session.add(doc)
        self.session.flush()
        return doc
