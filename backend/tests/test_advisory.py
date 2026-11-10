"""Tests for the KB + template advisory layer. Run: pytest backend/tests"""
from __future__ import annotations

from backend.config import settings
from backend.llm.base import AdvisoryRequest
from backend.llm.template_provider import TemplateProvider
from backend.rag.retriever import KnowledgeBase, Retriever


def _setup():
    kb = KnowledgeBase(settings.path(settings.kb_path))
    retriever = Retriever(kb, index_dir=None)  # direct lookup
    provider = TemplateProvider(kb)
    return kb, retriever, provider


def _make_req(kb, retriever, disease_key, language="en", uncertain=False, confidence=0.9):
    entry = retriever.retrieve_for_disease(disease_key)
    return AdvisoryRequest(
        disease_key=disease_key,
        confidence=confidence,
        uncertain=uncertain,
        crop=None,
        language=language,
        entry=entry,
    )


def test_kb_loads_classes():
    kb, _, _ = _setup()
    assert len(kb.keys()) > 0
    assert "Tomato___Late_blight" in kb.keys()


def test_known_disease_plan_has_sections():
    kb, retriever, provider = _setup()
    plan = provider.generate(_make_req(kb, retriever, "Tomato___Late_blight"))
    labels = {s.label for s in plan.sections}
    assert plan.title
    assert plan.summary
    assert plan.sections  # symptoms/organic/chemical/prevention present
    assert plan.safety_note
    # each section must have at least one actionable item
    assert all(s.items for s in plan.sections)


def test_healthy_class_needs_no_treatment():
    kb, retriever, provider = _setup()
    plan = provider.generate(_make_req(kb, retriever, "Tomato___healthy"))
    assert plan.sections == []
    assert plan.summary  # the "healthy, no treatment needed" message


def test_uncertain_prediction_flags_note():
    kb, retriever, provider = _setup()
    plan = provider.generate(
        _make_req(kb, retriever, "Potato___Late_blight", uncertain=True, confidence=0.3)
    )
    assert plan.uncertain_note


def test_unknown_class_is_safe():
    kb, retriever, provider = _setup()
    plan = provider.generate(_make_req(kb, retriever, "NotARealClass___xyz"))
    assert plan.uncertain_note
    assert plan.sections == []


def test_hindi_localization():
    kb, retriever, provider = _setup()
    en_plan = provider.generate(_make_req(kb, retriever, "Tomato___Late_blight", language="en"))
    hi_plan = provider.generate(_make_req(kb, retriever, "Tomato___Late_blight", language="hi"))
    assert hi_plan.language == "hi"
    assert hi_plan.sections
    assert hi_plan.title
    # Hindi, not an English fallback.
    assert hi_plan.title != en_plan.title
    assert hi_plan.summary != en_plan.summary


def test_all_disease_classes_have_hindi():
    """Every non-healthy class must carry Hindi content, not an English fallback."""
    kb, retriever, provider = _setup()
    for key, entry in kb.entries.items():
        if entry.healthy:
            continue
        en = entry.localized("en")
        hi = entry.localized("hi")
        assert hi and hi is not en, f"{key} has no Hindi translation"
        assert hi.get("summary") and hi.get("summary") != en.get("summary"), key