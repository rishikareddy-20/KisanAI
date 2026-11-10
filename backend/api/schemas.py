"""Pydantic response models for the API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class TopKItem(BaseModel):
    disease: str
    confidence: float


class ClassificationResult(BaseModel):
    disease: str
    confidence: float
    uncertain: bool
    is_mock: bool
    top_k: List[TopKItem]


class PlanSection(BaseModel):
    label: str
    items: List[str]


class TreatmentPlanModel(BaseModel):
    title: str
    language: str
    summary: str
    sections: List[PlanSection]
    uncertain_note: Optional[str] = None
    safety_note: Optional[str] = None
    provider: str


class DiagnoseResponse(BaseModel):
    classification: ClassificationResult
    plan: TreatmentPlanModel
    audio_url: Optional[str] = None


class DiseaseInfo(BaseModel):
    key: str
    crop: str
    healthy: bool
    name: str


class HealthResponse(BaseModel):
    status: str
    advisory_provider: str
    tts_provider: str
    vision_is_mock: bool
    languages: List[str]
    num_classes: int
