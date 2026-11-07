"""Advisory layer: turns a diagnosis + KB context into a treatment plan, offline."""
from .base import AdvisoryProvider, TreatmentPlan
from .factory import get_provider

__all__ = ["AdvisoryProvider", "TreatmentPlan", "get_provider"]
