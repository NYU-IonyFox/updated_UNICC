"""NLLB-200 fallback translator (facebook/nllb-200-distilled-600M).

Used when no LLM API key is provided.  The model is loaded lazily to avoid
eager import of torch/transformers at process startup (which can segfault in
some environments).

Confidence is computed as the mean token probability across the generated
sequence; values below 0.60 set confidence_warning=True and values below
0.40 additionally set multilingual_jailbreak_suspected=True.
"""
from __future__ import annotations

import math
from typing import Any

from app.safe_schemas import TranslationReport

_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Unicode range → NLLB language code (first match wins)
_SCRIPT_RANGES: list[tuple[tuple[int, int], str]] = [
    ((0x0600, 0x06FF), "arb_Arab"),   # Arabic
    ((0x0400, 0x04FF), "rus_Cyrl"),   # Cyrillic → Russian
    ((0x4E00, 0x9FFF), "zho_Hans"),   # CJK ideographs → Simplified Chinese
    ((0x3040, 0x30FF), "jpn_Jpan"),   # Hiragana / Katakana → Japanese
    ((0x0900, 0x097F), "hin_Deva"),   # Devanagari → Hindi
    ((0xAC00, 0xD7AF), "kor_Hang"),   # Hangul → Korean
    ((0x0590, 0x05FF), "heb_Hebr"),   # Hebrew
    ((0x0E00, 0x0E7F), "tha_Thai"),   # Thai
]


def _detect_src_lang(text: str) -> str:
    """Rough script-based source-language detection.

    Returns the NLLB language code for the first recognised non-Latin script
    found in *text*.  Falls back to "fra_Latn" (French / generic Latin) when
    no distinctive script characters are present.
    """
    for (lo, hi), lang_code in _SCRIPT_RANGES:
        if any(lo <= ord(c) <= hi for c in text):
            return lang_code
    return "fra_Latn"


def _load_model() -> tuple[Any, Any]:
    """Lazy-load NLLB model and tokenizer.

    Importing transformers here (not at module level) prevents accidental eager
    loading when the translation layer is imported but NLLB is not needed.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
    return model, tokenizer


def _run_translation(
    text: str,
    src_lang: str,
    model: Any,
    tokenizer: Any,
) -> tuple[str, float]:
    """Run NLLB beam-search translation.

    Returns ``(translated_text, confidence_score)`` where confidence is the
    mean token probability (0–1) of the best beam sequence.
    """
    import torch  # noqa: PLC0415

    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    eng_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=eng_token_id,
            num_beams=4,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=512,
        )

    translated = tokenizer.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )[0]

    # Mean token probability as confidence proxy
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    token_probs = [math.exp(float(s)) for s in transition_scores[0].tolist()]
    confidence = sum(token_probs) / len(token_probs) if token_probs else 0.5

    return translated, float(min(max(confidence, 0.0), 1.0))


def nllb_translate(text: str) -> TranslationReport:
    """Translate *text* using NLLB-200.

    Returns a TranslationReport with NLLB-mode fields; LLM fields are None.
    Falls back to confidence=0.0 on any model error so scoring can continue.
    """
    src_lang = _detect_src_lang(text)

    try:
        model, tokenizer = _load_model()
        _translated, confidence = _run_translation(text, src_lang, model, tokenizer)
    except Exception:
        confidence = 0.0

    return TranslationReport(
        translation_applied=True,
        primary_language=src_lang,
        confidence_numeric=confidence,
        confidence_warning=confidence < 0.60,
        multilingual_jailbreak_suspected=confidence < 0.40,
    )
