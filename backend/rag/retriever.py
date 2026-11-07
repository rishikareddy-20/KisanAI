"""Knowledge-base loading and retrieval.

Two paths: direct lookup by predicted class key (primary), and optional
semantic search over a FAISS index, which falls back to keyword matching.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("kisanai.rag")

_DEFAULT_LANG = "en"


@dataclass
class DiseaseEntry:
    key: str
    crop: str
    healthy: bool
    translations: Dict[str, dict]

    def localized(self, lang: str) -> dict:
        """Entry content in `lang`, falling back to English."""
        return self.translations.get(lang) or self.translations.get(_DEFAULT_LANG, {})


class KnowledgeBase:
    """Loads and indexes the curated treatment knowledge base."""

    def __init__(self, kb_path: Path, label_map_path: Optional[Path] = None) -> None:
        self.kb_path = kb_path
        with kb_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.languages: List[str] = raw.get("languages", [_DEFAULT_LANG])
        self.labels: dict = raw.get("labels", {})
        self.entries: Dict[str, DiseaseEntry] = {
            key: DiseaseEntry(
                key=key,
                crop=val.get("crop", ""),
                healthy=val.get("healthy", False),
                translations=val.get("translations", {}),
            )
            for key, val in raw.get("classes", {}).items()
        }

        # Optional alias map (raw model label -> canonical KB key).
        # Defaults to a sibling label_map.json if present.
        self.aliases: Dict[str, Optional[str]] = {}
        if label_map_path is None:
            label_map_path = kb_path.parent / "label_map.json"
        if label_map_path.exists():
            with label_map_path.open("r", encoding="utf-8") as fh:
                self.aliases = json.load(fh).get("map", {})
            logger.info("Loaded %d label aliases from %s", len(self.aliases), label_map_path)

    def resolve(self, disease_key: str) -> Optional[str]:
        """Resolve a raw label to a KB key. None means "no KB entry";
        unmapped labels pass through unchanged."""
        if disease_key in self.aliases:
            return self.aliases[disease_key]  # KB key or None
        return disease_key

    def get(self, disease_key: str) -> Optional[DiseaseEntry]:
        return self.entries.get(disease_key)

    def keys(self) -> List[str]:
        return list(self.entries.keys())

    def section_labels(self, lang: str) -> dict:
        section = self.labels.get("section", {})
        return section.get(lang) or section.get(_DEFAULT_LANG, {})

    def message(self, name: str, lang: str) -> str:
        """Shared message (uncertain/healthy/safety) in `lang`."""
        block = self.labels.get(name, {})
        return block.get(lang) or block.get(_DEFAULT_LANG, "")


class Retriever:
    """Retrieves KB context for the advisory layer."""

    def __init__(self, kb: KnowledgeBase, index_dir: Optional[Path] = None) -> None:
        self.kb = kb
        self.index_dir = index_dir
        self._semantic = None  # lazily loaded, if present
        if index_dir is not None:
            self._try_load_semantic(index_dir)

    def retrieve_for_disease(self, disease_key: str) -> Optional[DiseaseEntry]:
        resolved = self.kb.resolve(disease_key)
        if resolved is None:
            return None
        return self.kb.get(resolved)

    def search(self, query: str, top_k: int = 3) -> List[DiseaseEntry]:
        if self._semantic is not None:
            try:
                return self._semantic.search(query, top_k)
            except Exception as exc:  # pragma: no cover
                logger.warning("Semantic search failed (%s); using keyword fallback.", exc)
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[DiseaseEntry]:
        q = query.lower()
        scored = []
        for entry in self.kb.entries.values():
            text = json.dumps(entry.localized(_DEFAULT_LANG), ensure_ascii=False).lower()
            score = sum(1 for token in q.split() if token in text)
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def _try_load_semantic(self, index_dir: Path) -> None:
        try:
            from .vector_store import VectorStore
        except Exception as exc:  # pragma: no cover
            logger.info("Vector store module unavailable (%s).", exc)
            return
        try:
            self._semantic = VectorStore.load(index_dir, self.kb)
            logger.info("Loaded semantic RAG index from %s", index_dir)
        except FileNotFoundError:
            logger.info("No semantic index at %s; using direct lookup only.", index_dir)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load semantic index (%s).", exc)
