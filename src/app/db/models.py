# src/app/db/models.py

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    String,
    Text,
    ForeignKey,
    SmallInteger,
    CHAR,
    text,
)
from sqlalchemy.dialects.postgresql import (
    JSONB,
    UUID,
    TIMESTAMP,
    VARCHAR,
)
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column

from pgvector.sqlalchemy import Vector

Base = declarative_base()

# Sesuaikan dengan dimensi embedding yang kamu pakai
EMBEDDING_DIM = 1024


class ListingBoard(Base):
    __tablename__ = "listing_boards"

    code: Mapped[str] = mapped_column(
        CHAR(10),
        primary_key=True,
        unique=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    issuers: Mapped[List["Issuer"]] = relationship(
        "Issuer",
        back_populates="listing_board",
        cascade="all, delete-orphan",
    )


class Issuer(Base):
    __tablename__ = "issuers"

    code: Mapped[str] = mapped_column(
        CHAR(10),
        primary_key=True,
        unique=True,
        nullable=False,
    )
    listing_board_code: Mapped[str] = mapped_column(
        CHAR(10),
        ForeignKey("listing_boards.code"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    listing_board: Mapped["ListingBoard"] = relationship(
        "ListingBoard",
        back_populates="issuers",
    )
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="issuer",
        cascade="all, delete-orphan",
    )


class Collection(Base):
    __tablename__ = "collections"

    code: Mapped[str] = mapped_column(
        VARCHAR(50),
        primary_key=True,
        unique=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    # Kolom DB tetap "metadata", attribute Python = collection_metadata
    collection_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="collection",
        cascade="all, delete-orphan",
    )


class DocumentStatus(Base):
    __tablename__ = "document_status"

    id: Mapped[int] = mapped_column(
        SmallInteger,
        primary_key=True,
        unique=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="status_ref",
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        nullable=False,
        server_default=text("gen_random_uuid()"),
    )
    collection_code: Mapped[str] = mapped_column(
        VARCHAR(50),
        ForeignKey("collections.code"),
        nullable=False,
    )
    issuer_code: Mapped[str] = mapped_column(
        CHAR(10),
        ForeignKey("issuers.code"),
        nullable=False,
    )
    checksum: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        unique=True,
    )
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    publish_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    status: Mapped[int] = mapped_column(
        SmallInteger,
        ForeignKey("document_status.id"),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    # Kolom DB tetap "metadata", attribute Python = document_metadata
    document_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
        server_onupdate=text("NOW()"),
    )

    collection: Mapped["Collection"] = relationship(
        "Collection",
        back_populates="documents",
    )
    issuer: Mapped["Issuer"] = relationship(
        "Issuer",
        back_populates="documents",
    )
    status_ref: Mapped["DocumentStatus"] = relationship(
        "DocumentStatus",
        back_populates="documents",
    )
    slides: Mapped[List["Slide"]] = relationship(
        "Slide",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class Slide(Base):
    __tablename__ = "slides"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        nullable=False,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.id"),
        nullable=False,
    )
    content_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    content_text_vector: Mapped[Optional[list]] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=True,
    )
    image_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    # Kolom DB tetap "metadata", attribute Python = slide_metadata
    slide_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )
    ingestion_start_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    ingestion_end_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
        server_onupdate=text("NOW()"),
    )

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="slides",
    )
