"""Advisory provider interface. Every provider returns the same TreatmentPlan structure."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from backend.rag.retriever import DiseaseEntry, KnowledgeBase


@dataclass
class AdvisoryRequest:
    disease_key: str
    confidence: float
    uncertain: bool
    crop: Optional[str]
    language: str
    entry: Optional[DiseaseEntry]  # None if unknown class


@dataclass
class Section:
    label: str
    items: List[str]


@dataclass
class TreatmentPlan:
    title: str
    language: str
    summary: str
    sections: List[Section] = field(default_factory=list)
    uncertain_note: Optional[str] = None
    safety_note: Optional[str] = None
    provider: str = "template"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "language": self.language,
            "summary": self.summary,
            "sections": [{"label": s.label, "items": s.items} for s in self.sections],
            "uncertain_note": self.uncertain_note,
            "safety_note": self.safety_note,
            "provider": self.provider,
        }


class AdvisoryProvider(ABC):
    """Base class for advisory generation."""

    name = "base"

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    @abstractmethod
    def generate(self, req: AdvisoryRequest) -> TreatmentPlan:
        """Produce a treatment plan for the given diagnosis."""
        raise NotImplementedError
