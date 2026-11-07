"""Text-to-speech synthesis.

NullSynthesizer is the default and returns None (the frontend can still use the
browser Web Speech API). GttsSynthesizer renders an MP3 per language via gTTS
(needs internet); enable with KISANAI_TTS_PROVIDER=gtts. Audio is cached under
models/tts_cache and served by the API.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from backend.config import ROOT_DIR, settings

logger = logging.getLogger("kisanai.tts")

_CACHE_DIR = ROOT_DIR / "models" / "tts_cache"
_GTTS_LANG = {"en": "en", "hi": "hi"}


class NullSynthesizer:
    provider = "none"

    def synthesize(self, text: str, language: str) -> Optional[str]:
        return None


class GttsSynthesizer:
    provider = "gtts"

    def synthesize(self, text: str, language: str) -> Optional[str]:
        if not text.strip():
            return None
        try:
            from gtts import gTTS  # type: ignore
        except ImportError:
            logger.warning("gTTS not installed; skipping audio synthesis.")
            return None

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(f"{language}:{text}".encode("utf-8")).hexdigest()[:16]
        out_path = _CACHE_DIR / f"{digest}.mp3"
        if not out_path.exists():
            try:
                gTTS(text=text, lang=_GTTS_LANG.get(language, "en")).save(str(out_path))
            except Exception as exc:  # pragma: no cover - network/runtime
                logger.warning("gTTS synthesis failed (%s).", exc)
                return None
        # URL path served by the API static mount.
        return f"/audio/{out_path.name}"


def get_synthesizer():
    if settings.tts_provider.lower().strip() == "gtts":
        return GttsSynthesizer()
    return NullSynthesizer()


AUDIO_CACHE_DIR = _CACHE_DIR
