"""L1 Translation entry point.

Routes to LLM translation (api_key present) or NLLB fallback (no key).
English input is passed through without calling any model.
"""
from __future__ import annotations

from app.safe_schemas import TranslationReport
from .llm_translator import llm_translate
from .nllb_translator import nllb_translate


def _is_english(text: str) -> bool:
    """Detect whether text is English.

    Tries langdetect first; falls back to ASCII-ratio heuristic when the
    library is not installed.  Returns True for empty / whitespace-only input.
    """
    if not text.strip():
        return True
    try:
        from langdetect import detect  # optional dependency
        return detect(text) == "en"
    except ImportError:
        ascii_count = sum(1 for c in text if ord(c) < 128)
        return ascii_count / len(text) > 0.90
    except Exception:
        # Detection failed → conservatively attempt translation
        return False


def translate(
    text: str,
    api_key: str | None = None,
    provider: str | None = None,  # "openai" | "anthropic" | None
) -> TranslationReport:
    """L1 translation entry point.

    - English input  → TranslationReport(translation_applied=False)
    - api_key present → LLM translation
    - api_key absent  → NLLB-200 fallback
    """
    if _is_english(text):
        return TranslationReport(
            translation_applied=False,
            primary_language="en",
        )
    if api_key:
        return llm_translate(text, api_key=api_key, provider=provider)
    return nllb_translate(text)
