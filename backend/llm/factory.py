"""Select the advisory provider. One entry point so the provider can be swapped later."""
from __future__ import annotations

from backend.rag.retriever import KnowledgeBase

from .base import AdvisoryProvider
from .template_provider import TemplateProvider


def get_provider(kb: KnowledgeBase) -> AdvisoryProvider:
    return TemplateProvider(kb)
