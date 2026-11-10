"""KisanAI FastAPI app. Wires vision -> rag -> llm -> tts.

Run: uvicorn backend.api.main:app --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.llm import get_provider
from backend.llm.base import AdvisoryRequest
from backend.rag.retriever import KnowledgeBase, Retriever
from backend.tts.synthesizer import AUDIO_CACHE_DIR, get_synthesizer
from backend.vision import DiseaseClassifier

from .schemas import (
    ClassificationResult,
    DiagnoseResponse,
    DiseaseInfo,
    HealthResponse,
    TreatmentPlanModel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kisanai.api")

app = FastAPI(
    title="KisanAI",
    description="Crop disease diagnosis + agronomy advisor",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load components once at startup.
kb = KnowledgeBase(settings.path(settings.kb_path))
retriever = Retriever(kb, index_dir=settings.path(settings.rag_index))
classifier = DiseaseClassifier(
    model_path=settings.path(settings.tflite_model),
    labels_path=settings.path(settings.labels),
    confidence_threshold=settings.confidence_threshold,
)
advisor = get_provider(kb)
synthesizer = get_synthesizer()

# Serve cached TTS audio.
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(AUDIO_CACHE_DIR)), name="audio")

_SUPPORTED_LANGS = {"en", "hi"}


def _validate_lang(language: str) -> str:
    lang = (language or "en").lower().strip()
    return lang if lang in _SUPPORTED_LANGS else "en"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        advisory_provider=advisor.name,
        tts_provider=synthesizer.provider,
        vision_is_mock=classifier.is_mock,
        languages=sorted(_SUPPORTED_LANGS),
        num_classes=len(kb.keys()),
    )


@app.get("/diseases", response_model=list[DiseaseInfo])
def diseases() -> list[DiseaseInfo]:
    out = []
    for key, entry in kb.entries.items():
        out.append(
            DiseaseInfo(
                key=key,
                crop=entry.crop,
                healthy=entry.healthy,
                name=entry.localized("en").get("name", key),
            )
        )
    return out


@app.post("/classify", response_model=ClassificationResult)
async def classify(file: UploadFile = File(...)) -> ClassificationResult:
    image_bytes = await _read_image(file)
    pred = classifier.predict(image_bytes)
    return ClassificationResult(
        disease=pred.disease,
        confidence=pred.confidence,
        uncertain=pred.uncertain,
        is_mock=pred.is_mock,
        top_k=pred.top_k,
    )


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(
    file: UploadFile = File(...),
    language: str = Form("en"),
    crop: str | None = Form(None),
) -> DiagnoseResponse:
    lang = _validate_lang(language)
    image_bytes = await _read_image(file)

    # Vision
    pred = classifier.predict(image_bytes)

    # RAG
    entry = retriever.retrieve_for_disease(pred.disease)

    # Advisory
    plan = advisor.generate(
        AdvisoryRequest(
            disease_key=pred.disease,
            confidence=pred.confidence,
            uncertain=pred.uncertain,
            crop=crop,
            language=lang,
            entry=entry,
        )
    )

    # TTS (optional)
    audio_url = synthesizer.synthesize(_plan_to_speech(plan), lang)

    return DiagnoseResponse(
        classification=ClassificationResult(
            disease=pred.disease,
            confidence=pred.confidence,
            uncertain=pred.uncertain,
            is_mock=pred.is_mock,
            top_k=pred.top_k,
        ),
        plan=TreatmentPlanModel(**plan.to_dict()),
        audio_url=audio_url,
    )


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    return data


def _plan_to_speech(plan) -> str:
    """Flatten a plan into a single string for TTS."""
    parts = [plan.title, plan.summary]
    if plan.uncertain_note:
        parts.append(plan.uncertain_note)
    for section in plan.sections:
        parts.append(section.label + ".")
        parts.extend(section.items)
    if plan.safety_note:
        parts.append(plan.safety_note)
    return " ".join(p for p in parts if p)
