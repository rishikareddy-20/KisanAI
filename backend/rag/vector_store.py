"""Optional FAISS semantic index over the knowledge base.

Needs sentence-transformers and faiss-cpu; the retriever falls back to direct
lookup / keyword search when they're missing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger("kisanai.rag.vector_store")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_META_FILE = "chunks.json"
_INDEX_FILE = "index.faiss"


@dataclass
class Chunk:
    disease_key: str
    text: str


def _kb_to_chunks(kb) -> List[Chunk]:
    """Flatten each disease's English content into a searchable chunk."""
    chunks: List[Chunk] = []
    for key, entry in kb.entries.items():
        content = entry.localized("en")
        parts = [content.get("name", key), content.get("summary", "")]
        for field in ("symptoms", "organic", "chemical", "prevention"):
            parts.extend(content.get(field, []) or [])
        chunks.append(Chunk(disease_key=key, text=" ".join(p for p in parts if p)))
    return chunks


class VectorStore:
    def __init__(self, index, chunks: List[Chunk], model, kb) -> None:
        self._index = index
        self._chunks = chunks
        self._model = model
        self._kb = kb

    @classmethod
    def build(cls, kb, out_dir: Path) -> "VectorStore":
        import faiss  # type: ignore
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore

        chunks = _kb_to_chunks(kb)
        model = SentenceTransformer(EMBED_MODEL)
        embeddings = model.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine via inner product on normalized vectors
        index.add(embeddings)

        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(out_dir / _INDEX_FILE))
        with (out_dir / _META_FILE).open("w", encoding="utf-8") as fh:
            json.dump([c.__dict__ for c in chunks], fh, ensure_ascii=False)
        logger.info("Built semantic index with %d chunks at %s", len(chunks), out_dir)
        return cls(index, chunks, model, kb)

    @classmethod
    def load(cls, index_dir: Path, kb) -> "VectorStore":
        index_path = index_dir / _INDEX_FILE
        meta_path = index_dir / _META_FILE
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No FAISS index in {index_dir}")

        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        index = faiss.read_index(str(index_path))
        with meta_path.open("r", encoding="utf-8") as fh:
            chunks = [Chunk(**c) for c in json.load(fh)]
        model = SentenceTransformer(EMBED_MODEL)
        return cls(index, chunks, model, kb)

    def search(self, query: str, top_k: int = 3):
        import numpy as np

        vec = self._model.encode([query], normalize_embeddings=True).astype("float32")
        _scores, idx = self._index.search(vec, top_k)
        results = []
        for i in idx[0]:
            if 0 <= i < len(self._chunks):
                entry = self._kb.get(self._chunks[i].disease_key)
                if entry is not None:
                    results.append(entry)
        return results
