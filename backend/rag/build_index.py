"""Build the FAISS semantic index from the knowledge base.

    python -m backend.rag.build_index

Optional: only needed for free-text symptom search; direct lookup works without it.
"""
from __future__ import annotations

import logging

from backend.config import settings
from backend.rag.retriever import KnowledgeBase
from backend.rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)


def main() -> None:
    kb = KnowledgeBase(settings.path(settings.kb_path))
    VectorStore.build(kb, settings.path(settings.rag_index))
    print(f"✅ Semantic index built at {settings.path(settings.rag_index)}")


if __name__ == "__main__":
    main()
