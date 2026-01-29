from typing import List, Dict, Any

import streamlit as st
from sqlalchemy import select, func, cast, Integer

from app.db.models import Slide, Document
from app.db.session import get_session
from app.utils.image_utils import load_slide_image_bytes

PAGE_SIZE_DEFAULT = 10


def _count_slides(document_id: str | None = None) -> int:
    with get_session() as session:
        stmt = select(func.count(Slide.id))
        if document_id:
            stmt = stmt.where(Slide.document_id == document_id)
    return int(session.execute(stmt).scalar() or 0)


def _fetch_document_options() -> List[Dict[str, str]]:
    with get_session() as session:
        rows = session.execute(
            select(Document.id, Document.name)
            .join(Slide, Slide.document_id == Document.id)
            .distinct(Document.id, Document.name)
            .order_by(Document.name.asc())
        ).all()
    options: List[Dict[str, str]] = []
    for row in rows:
        options.append({"id": str(row.id), "name": row.name})
    return options


def _fetch_slides(
    *,
    limit: int,
    offset: int,
    document_id: str | None = None,
) -> List[Dict[str, Any]]:
    with get_session() as session:
        slide_no_expr = cast(Slide.slide_metadata["slide_no"].astext, Integer)
        stmt = (
            select(
                Slide.id,
                Slide.document_id,
                Slide.content_text,
                Slide.image_path,
                Slide.slide_metadata,
                Slide.ingestion_start_at,
                Slide.ingestion_end_at,
                Slide.created_at,
                Document.name,
            )
            .join(Document, Slide.document_id == Document.id)
            .order_by(
                Slide.document_id.asc(),
                slide_no_expr.asc().nullslast(),
                Slide.id.asc(),
            )
            .limit(limit)
            .offset(offset)
        )
        if document_id:
            stmt = stmt.where(Slide.document_id == document_id)

        rows = session.execute(stmt).all()

    slides: List[Dict[str, Any]] = []
    for row in rows:
        slides.append(
            {
                "id": row.id,
                "document_id": row.document_id,
                "document_name": row.name,
                "content_text": row.content_text,
                "image_path": row.image_path,
                "slide_metadata": row.slide_metadata,
                "ingestion_start_at": row.ingestion_start_at,
                "ingestion_end_at": row.ingestion_end_at,
                "created_at": row.created_at,
            }
        )
    return slides


def _init_state() -> None:
    if "page_size" not in st.session_state:
        st.session_state.page_size = PAGE_SIZE_DEFAULT
    if "shown_count" not in st.session_state:
        st.session_state.shown_count = PAGE_SIZE_DEFAULT
    if "document_id" not in st.session_state:
        st.session_state.document_id = ""
    if "last_document_id" not in st.session_state:
        st.session_state.last_document_id = ""


def main() -> None:
    st.set_page_config(page_title="Slide Viewer", layout="wide")
    _init_state()

    st.title("Slide Viewer")
    st.caption("Menampilkan content_text, metadata, dan image per slide (lazy load 10 row).")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        options = _fetch_document_options()
        labels = ["(Semua dokumen)"] + [
            f"{opt['name']} | {opt['id']}" for opt in options
        ]
        values = [""] + [opt["id"] for opt in options]
        try:
            current_index = values.index(st.session_state.document_id)
        except ValueError:
            current_index = 0
        selection = st.selectbox(
            "Pilih dokumen",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            index=current_index,
        )
        document_id = values[selection]
    with col_b:
        page_size = st.number_input(
            "Rows per load",
            min_value=1,
            max_value=50,
            value=st.session_state.page_size,
            step=1,
        )
    with col_c:
        if st.button("Reset"):
            st.session_state.document_id = document_id
            st.session_state.page_size = int(page_size)
            st.session_state.shown_count = st.session_state.page_size
            st.rerun()

    st.session_state.document_id = document_id
    st.session_state.page_size = int(page_size)
    if st.session_state.last_document_id != document_id:
        st.session_state.shown_count = st.session_state.page_size
        st.session_state.last_document_id = document_id

    total = _count_slides(document_id=document_id or None)
    st.write(f"Total slides: {total}")

    slides = _fetch_slides(
        limit=st.session_state.shown_count,
        offset=0,
        document_id=document_id or None,
    )

    if not slides:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    for slide in slides:
        st.markdown("---")
        st.subheader(f"Slide {slide['id']}")

        left_col, right_col = st.columns([2, 1])
        with left_col:
            ingestion_start = slide.get("ingestion_start_at")
            ingestion_end = slide.get("ingestion_end_at")
            if ingestion_start and ingestion_end:
                duration_seconds = (ingestion_end - ingestion_start).total_seconds()
                st.markdown(
                    f"**ingestion_time**: {duration_seconds:.2f} detik"
                )
            else:
                st.markdown("**ingestion_time**: (tidak tersedia)")

            vlm_model = (slide.get("slide_metadata") or {}).get("vlm_model")
            st.markdown(f"**vlm_model**: {vlm_model or '(tidak tersedia)'}")

            st.markdown("**metadata**")
            st.json(slide["slide_metadata"] or {})

            st.markdown("**content_text**")
            st.markdown(slide["content_text"] or "(kosong)")

        with right_col:
            img_bytes = load_slide_image_bytes(slide["image_path"])
            if img_bytes:
                st.image(img_bytes, caption="Slide Image", width=500)
            else:
                st.warning("Image tidak tersedia.")

    shown = min(len(slides), total)
    st.write(f"Menampilkan {shown} dari {total} slide.")

    if shown < total:
        if st.button("Load next"):
            st.session_state.shown_count += st.session_state.page_size
            st.rerun()


if __name__ == "__main__":
    main()
