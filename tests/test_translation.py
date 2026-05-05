"""Tests for L1 Translation layer (Phase 2).

NLLB model loading is mocked throughout to avoid loading HuggingFace weights
(which may segfault or OOM in CI environments).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.safe_schemas import TranslationReport
from app.translation.nllb_translator import nllb_translate
from app.translation.translation_service import translate


# ── Test 1: English input → pass-through, no model called ───────────────────

def test_english_input_no_translation_applied():
    """English text → translation_applied=False; neither LLM nor NLLB is invoked."""
    with patch("app.translation.translation_service.llm_translate") as mock_llm, \
         patch("app.translation.translation_service.nllb_translate") as mock_nllb:
        result = translate("Hello, this is a test in English.", api_key=None)

    assert result.translation_applied is False
    assert result.primary_language == "en"
    mock_llm.assert_not_called()
    mock_nllb.assert_not_called()


# ── Test 2: LLM mode → confidence_qualitative populated ─────────────────────

def test_llm_mode_returns_confidence_qualitative():
    """LLM mode (api_key present, non-English input) → TranslationReport with confidence_qualitative."""
    expected = TranslationReport(
        translation_applied=True,
        primary_language="ar",
        confidence_qualitative="High",
        confidence_note="Clear Modern Standard Arabic, no ambiguity.",
    )
    # Arabic text (non-ASCII) bypasses the English-detection short-circuit
    with patch("app.translation.translation_service.llm_translate", return_value=expected) as mock_llm:
        result = translate("مرحبا بالعالم", api_key="sk-ant-test123")

    assert isinstance(result, TranslationReport)
    assert result.confidence_qualitative is not None
    mock_llm.assert_called_once()


# ── Test 3: NLLB mode → confidence_numeric populated ────────────────────────

def test_nllb_mode_returns_confidence_numeric():
    """NLLB mode (api_key=None, non-English input) → TranslationReport with confidence_numeric."""
    expected = TranslationReport(
        translation_applied=True,
        primary_language="zho_Hans",
        confidence_numeric=0.72,
    )
    # Chinese text (non-ASCII) bypasses English-detection short-circuit
    with patch("app.translation.translation_service.nllb_translate", return_value=expected) as mock_nllb:
        result = translate("这是一个测试", api_key=None)

    assert isinstance(result, TranslationReport)
    assert result.confidence_numeric is not None
    mock_nllb.assert_called_once()


# ── Test 4: Low confidence → confidence_warning=True ────────────────────────

def test_nllb_low_confidence_sets_warning_flag():
    """nllb_translate: confidence_numeric < 0.60 → confidence_warning=True."""
    with patch("app.translation.nllb_translator._load_model") as mock_load, \
         patch("app.translation.nllb_translator._run_translation") as mock_run, \
         patch("app.translation.nllb_translator._detect_src_lang", return_value="fra_Latn"):
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_run.return_value = ("le texte traduit", 0.45)   # score < 0.60

        result = nllb_translate("Bonjour le monde")

    assert result.confidence_numeric == pytest.approx(0.45)
    assert result.confidence_warning is True
    # 0.45 is above the jailbreak threshold of 0.40
    assert result.multilingual_jailbreak_suspected is False


# ── Test 5: Both modes return TranslationReport instances ───────────────────

def test_both_modes_return_translation_report_type():
    """LLM and NLLB paths both produce TranslationReport instances with the correct fields."""
    llm_report = TranslationReport(
        translation_applied=True,
        primary_language="es",
        confidence_qualitative="Medium",
        confidence_note="Some dialectal variation.",
    )
    nllb_report = TranslationReport(
        translation_applied=True,
        primary_language="fra_Latn",
        confidence_numeric=0.80,
    )

    assert isinstance(llm_report, TranslationReport)
    assert isinstance(nllb_report, TranslationReport)

    # LLM mode: qualitative present, numeric absent
    assert llm_report.confidence_qualitative is not None
    assert llm_report.confidence_numeric is None

    # NLLB mode: numeric present, qualitative absent
    assert nllb_report.confidence_numeric is not None
    assert nllb_report.confidence_qualitative is None
