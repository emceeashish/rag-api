"""Prompt template for RAG generation."""

SYSTEM_PROMPT = (
    "Answer using only the provided context. "
    "If the answer is not in the context, say 'I don't know'."
)


def build_messages(context_chunks: list[str], question: str) -> list[dict[str, str]]:
    """Build OpenAI Chat Completions messages from retrieved context and user question."""
    context = "\n\n".join(context_chunks)
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
