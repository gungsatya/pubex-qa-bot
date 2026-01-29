from __future__ import annotations

from typing import List, Dict, Any

import streamlit as st
from sqlalchemy import select, func, cast, Integer

from app.db.models import Slide, Document
from app.db.session import get_session
from app.utils.image_utils import load_slide_image_bytes


def _fetch_document_options() -> List[Dict[str, str]]:
    with get_session() as session:
        rows = session.execute(
            select(Document.id, Document.name).order_by(Document.name.asc())
        ).all()
    options: List[Dict[str, str]] = []
    for row in rows:
        options.append({"id": str(row.id), "name": row.name})
    return options


def _fetch_models(document_id: str | None) -> List[str]:
    with get_session() as session:
        model_expr = Slide.slide_metadata["vlm_model"].astext
        stmt = select(func.distinct(model_expr)).where(model_expr.isnot(None))
        if document_id:
            stmt = stmt.where(Slide.document_id == document_id)
        rows = session.execute(stmt).scalars().all()
    return sorted([row for row in rows if row])


def _fetch_slide_numbers(document_id: str) -> List[int]:
    with get_session() as session:
        slide_no_expr = cast(Slide.slide_metadata["slide_no"].astext, Integer)
        stmt = (
            select(func.distinct(slide_no_expr))
            .where(Slide.document_id == document_id)
            .where(slide_no_expr.isnot(None))
            .order_by(slide_no_expr.asc())
        )
        rows = session.execute(stmt).scalars().all()
    return [int(row) for row in rows if row is not None]


def _fetch_slides_for_compare(
    *,
    document_id: str,
    slide_no: int,
    models: List[str],
) -> List[Dict[str, Any]]:
    with get_session() as session:
        slide_no_expr = cast(Slide.slide_metadata["slide_no"].astext, Integer)
        model_expr = Slide.slide_metadata["vlm_model"].astext
        stmt = (
            select(
                Slide.id,
                Slide.content_text,
                Slide.image_path,
                Slide.slide_metadata,
                Slide.ingestion_start_at,
                Slide.ingestion_end_at,
                model_expr.label("vlm_model"),
            )
            .where(Slide.document_id == document_id)
            .where(slide_no_expr == slide_no)
        )
        if models:
            stmt = stmt.where(model_expr.in_(models))
        rows = session.execute(stmt).all()

    slides: List[Dict[str, Any]] = []
    for row in rows:
        slides.append(
            {
                "id": row.id,
                "content_text": row.content_text,
                "image_path": row.image_path,
                "slide_metadata": row.slide_metadata,
                "ingestion_start_at": row.ingestion_start_at,
                "ingestion_end_at": row.ingestion_end_at,
                "vlm_model": row.vlm_model,
            }
        )
    return slides


def _fetch_model_stats(
    *,
    document_id: str,
    models: List[str],
) -> List[Dict[str, Any]]:
    with get_session() as session:
        model_expr = Slide.slide_metadata["vlm_model"].astext
        stmt = (
            select(model_expr, Slide.ingestion_start_at, Slide.ingestion_end_at)
            .where(Slide.document_id == document_id)
            .where(model_expr.isnot(None))
        )
        if models:
            stmt = stmt.where(model_expr.in_(models))
        rows = session.execute(stmt).all()

    bucket: Dict[str, List[float]] = {}
    for model, start_at, end_at in rows:
        if not model:
            continue
        if start_at and end_at:
            duration = (end_at - start_at).total_seconds()
            bucket.setdefault(model, []).append(duration)

    stats: List[Dict[str, Any]] = []
    for model, durations in bucket.items():
        if not durations:
            continue
        avg_seconds = sum(durations) / len(durations)
        stats.append(
            {
                "model": model,
                "slides": len(durations),
                "avg_seconds": round(avg_seconds, 2),
                "min_seconds": round(min(durations), 2),
                "max_seconds": round(max(durations), 2),
            }
        )
    return sorted(stats, key=lambda item: item["model"])


def main() -> None:
    st.set_page_config(page_title="VLM Comparison", layout="wide")
    st.title("VLM Comparison")
    st.caption("Bandingkan hasil ingestion antar model untuk slide yang sama.")

    options = _fetch_document_options()
    labels = ["(Pilih dokumen)"] + [f"{opt['name']} | {opt['id']}" for opt in options]
    values = [""] + [opt["id"] for opt in options]

    selection = st.selectbox(
        "Pilih dokumen",
        options=list(range(len(labels))),
        format_func=lambda i: labels[i],
        index=0,
    )
    document_id = values[selection]

    if not document_id:
        st.info("Silakan pilih dokumen untuk memulai perbandingan.")
        return

    models = _fetch_models(document_id)
    if not models:
        st.warning("Belum ada hasil ingestion untuk dokumen ini.")
        return

    selected_models = st.multiselect(
        "Pilih model",
        options=models,
        default=models,
    )

    slide_numbers = _fetch_slide_numbers(document_id)
    if not slide_numbers:
        st.warning("Slide belum tersedia untuk dokumen ini.")
        return

    slide_no = st.selectbox("Pilih slide", options=slide_numbers)

    stats = _fetch_model_stats(document_id=document_id, models=selected_models)
    if stats:
        st.subheader("Ringkasan Performa")
        st.dataframe(stats, use_container_width=True)

    slides = _fetch_slides_for_compare(
        document_id=document_id,
        slide_no=int(slide_no),
        models=selected_models,
    )

    if not slides:
        st.info("Tidak ada output untuk slide ini pada model yang dipilih.")
        return

    st.subheader(f"Slide {slide_no}")
    img_bytes = load_slide_image_bytes(slides[0]["image_path"])
    if img_bytes:
        st.image(img_bytes, caption=f"Slide {slide_no}", use_container_width=True)
    else:
        st.warning("Image tidak tersedia.")

    slide_by_model = {slide["vlm_model"]: slide for slide in slides}

    if not selected_models:
        st.info("Tidak ada model yang dipilih.")
        return

    cols = st.columns(len(selected_models))
    for idx, model in enumerate(selected_models):
        with cols[idx]:
            st.markdown(f"### {model}")
            slide = slide_by_model.get(model)
            if not slide:
                st.warning("Tidak ada output untuk model ini.")
                continue

            ingestion_start = slide.get("ingestion_start_at")
            ingestion_end = slide.get("ingestion_end_at")
            if ingestion_start and ingestion_end:
                duration_seconds = (ingestion_end - ingestion_start).total_seconds()
                st.markdown(f"**ingestion_time**: {duration_seconds:.2f} detik")
            else:
                st.markdown("**ingestion_time**: (tidak tersedia)")

            st.markdown("**metadata**")
            st.json(slide.get("slide_metadata") or {})

            st.markdown("**content_text**")
            st.markdown(slide.get("content_text") or "(kosong)")


if __name__ == "__main__":
    main()
