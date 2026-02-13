from __future__ import annotations

import chainlit as cl


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "Pubex QA Bot siap.\\n"
            "Mekanisme yang tersedia saat ini: download, ingestion, embedding, dan interface Chainlit."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    user_text = (message.content or "").strip()
    if not user_text:
        await cl.Message(content="Pesan kosong, silakan kirim pertanyaan.").send()
        return
    await cl.Message(
        content=(
            "Mekanisme QA di Chainlit sudah aktif, namun flow tanya jawab belum dihubungkan ke retrieval pipeline.\\n\\n"
            f"Pesan Anda: {user_text}"
        )
    ).send()
