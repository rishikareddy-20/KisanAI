"""Offline template advisory provider.

Builds a treatment plan straight from the KB, no external model. Dosages are
taken verbatim from the KB (never invented). Output is localized to any KB
language, falling back to English content under localized section headers.
"""
from __future__ import annotations

from typing import List

from .base import AdvisoryProvider, AdvisoryRequest, Section, TreatmentPlan

# Order in which KB fields are rendered as sections.
_SECTION_FIELDS = ["symptoms", "organic", "chemical", "prevention"]


class TemplateProvider(AdvisoryProvider):
    name = "template"

    def generate(self, req: AdvisoryRequest) -> TreatmentPlan:
        lang = req.language
        labels = self.kb.section_labels(lang)
        entry = req.entry

        # Unknown class: return the uncertain response.
        if entry is None:
            return self._uncertain_plan(req, title=req.disease_key)

        content = entry.localized(lang)
        title = content.get("name", entry.key)

        # Healthy leaf: no treatment.
        if entry.healthy:
            return TreatmentPlan(
                title=title,
                language=lang,
                summary=self.kb.message("healthy", lang),
                provider=self.name,
            )

        # Uncertain prediction: show the plan but lead with the caution note.
        if req.uncertain:
            plan = self._build_full_plan(req, entry, content, labels, title)
            plan.uncertain_note = self.kb.message("uncertain", lang)
            return plan

        return self._build_full_plan(req, entry, content, labels, title)

    def _build_full_plan(self, req, entry, content, labels, title) -> TreatmentPlan:
        lang = req.language
        sections: List[Section] = []
        for field in _SECTION_FIELDS:
            items = content.get(field) or []
            if items:
                sections.append(Section(label=labels.get(field, field), items=list(items)))

        return TreatmentPlan(
            title=title,
            language=lang,
            summary=content.get("summary", ""),
            sections=sections,
            safety_note=self.kb.message("safety", lang),
            provider=self.name,
        )

    def _uncertain_plan(self, req, title) -> TreatmentPlan:
        return TreatmentPlan(
            title=title,
            language=req.language,
            summary="",
            uncertain_note=self.kb.message("uncertain", req.language),
            provider=self.name,
        )
