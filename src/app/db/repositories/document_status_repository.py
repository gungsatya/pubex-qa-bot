from typing import Optional, Union
from sqlalchemy import select
from app.db.models import DocumentStatus
from src.data.enums import DocumentStatusEnum


class DocumentStatusRepository:
    def __init__(self, session):
        self.session = session

    def get_by_enum(
        self, status: DocumentStatusEnum
    ) -> Optional[DocumentStatus]:
        stmt = select(DocumentStatus).where(DocumentStatus.id == status.value)
        result = self.session.execute(stmt)
        return result.scalars().first()

    def get_by_id(
        self, status_id: int
    ) -> Optional[DocumentStatus]:
        stmt = select(DocumentStatus).where(DocumentStatus.id == status_id)
        result = self.session.execute(stmt)
        return result.scalars().first()

    def get_by_name(
        self, name: str
    ) -> Optional[DocumentStatus]:
        stmt = select(DocumentStatus).where(DocumentStatus.name == name)
        result = self.session.execute(stmt)
        return result.scalars().first()
