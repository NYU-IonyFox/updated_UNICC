"""Tests for app/safe_schemas.py — Phase 1 validation."""

from app.safe_schemas import (
    ArbitrationResult,
    DimensionScore,
    EvidenceBundle,
    ExpertOutput,
    SAFEEvaluationResponse,
    TranslationReport,
)


def _make_translation_report_llm() -> TranslationReport:
    return TranslationReport(
        translation_applied=True,
        primary_language="fr",
        confidence_qualitative="High",
        confidence_note="Detected French with high confidence.",
        language_segments=[{"lang": "fr", "text": "Bonjour"}],
    )


def _make_translation_report_nllb() -> TranslationReport:
    return TranslationReport(
        translation_applied=True,
        primary_language="ar",
        confidence_numeric=0.82,
        confidence_warning=False,
        multilingual_jailbreak_suspected=False,
    )


def _make_dimension_score() -> DimensionScore:
    return DimensionScore(
        name="jailbreak_resistance",
        display_name="Jailbreak Resistance",
        tier="CORE",
        level="HIGH",
        evidence_quality="Strong",
        regulatory_anchor="OWASP LLM01:2025",
        reason="Model responded to a known DAN jailbreak prompt.",
    )


# ── 1. Import smoke test ────────────────────────────────────────────────────

def test_imports():
    assert DimensionScore is not None
    assert ExpertOutput is not None
    assert TranslationReport is not None
    assert EvidenceBundle is not None
    assert ArbitrationResult is not None
    assert SAFEEvaluationResponse is not None


# ── 2. DimensionScore instantiation ────────────────────────────────────────

def test_dimension_score_instantiation():
    ds = _make_dimension_score()
    assert ds.name == "jailbreak_resistance"
    assert ds.tier == "CORE"
    assert ds.level == "HIGH"
    assert ds.evidence_quality == "Strong"


# ── 3. TranslationReport — LLM mode ────────────────────────────────────────

def test_translation_report_llm_mode():
    tr = _make_translation_report_llm()
    assert tr.confidence_qualitative == "High"
    assert tr.confidence_note is not None
    assert tr.language_segments is not None
    assert tr.confidence_numeric is None


# ── 4. TranslationReport — NLLB mode ───────────────────────────────────────

def test_translation_report_nllb_mode():
    tr = _make_translation_report_nllb()
    assert tr.confidence_numeric == 0.82
    assert tr.confidence_qualitative is None
    assert tr.language_segments is None
    assert tr.confidence_warning is False


def test_translation_report_nllb_low_confidence_warning():
    tr = TranslationReport(
        translation_applied=True,
        primary_language="zh",
        confidence_numeric=0.45,
        confidence_warning=True,
    )
    assert tr.confidence_warning is True


# ── 5. ExpertOutput ─────────────────────────────────────────────────────────

def test_expert_output_instantiation():
    eo = ExpertOutput(
        id="expert_adversarial_security",
        overall="HIGH",
        triggered_dimensions=[_make_dimension_score()],
    )
    assert eo.overall == "HIGH"
    assert isinstance(eo.triggered_dimensions, list)
    assert len(eo.triggered_dimensions) == 1


def test_expert_output_triggered_dimensions_is_list():
    eo = ExpertOutput(id="expert_content_safety", overall="LOW")
    assert isinstance(eo.triggered_dimensions, list)


# ── 6. EvidenceBundle ───────────────────────────────────────────────────────

def test_evidence_bundle_instantiation():
    eb = EvidenceBundle(
        input_type="conversation",
        translation_report=_make_translation_report_llm(),
        content={"turns": 5},
    )
    assert eb.input_type == "conversation"
    assert eb.live_attack_results is None


# ── 7. ArbitrationResult ────────────────────────────────────────────────────

def test_arbitration_result_reject():
    ar = ArbitrationResult(
        verdict="REJECT",
        primary_reason={
            "rule": "R1",
            "expert_id": "expert_adversarial_security",
            "dimension": "jailbreak_resistance",
            "tier": "CORE",
            "level": "HIGH",
        },
        additional_findings=["prompt_injection_robustness also HIGH"],
        convergent_risk_note="",
    )
    assert ar.verdict == "REJECT"
    assert ar.primary_reason["rule"] == "R1"


def test_arbitration_result_approve():
    ar = ArbitrationResult(verdict="APPROVE")
    assert ar.verdict == "APPROVE"
    assert ar.additional_findings == []


# ── 8. SAFEEvaluationResponse ───────────────────────────────────────────────

def test_safe_evaluation_response_instantiation():
    resp = SAFEEvaluationResponse(
        evaluation_id="eval-001",
        timestamp="2026-05-04T12:00:00Z",
        safe_version="1.0.0",
        verdict="HOLD",
        translation_report=_make_translation_report_nllb(),
    )
    assert resp.evaluation_id == "eval-001"
    assert resp.verdict == "HOLD"
    assert isinstance(resp.experts, list)
    assert isinstance(resp.additional_findings, list)
    assert isinstance(resp.recommendations, list)


def test_safe_evaluation_response_with_experts():
    expert = ExpertOutput(
        id="expert_governance_un",
        overall="MEDIUM",
        triggered_dimensions=[
            DimensionScore(
                name="regulatory_compliance",
                display_name="Regulatory Compliance",
                tier="CORE",
                level="MEDIUM",
                evidence_quality="Partial",
                regulatory_anchor="EU AI Act Art. 9",
                reason="Incomplete risk management documentation.",
            )
        ],
    )
    resp = SAFEEvaluationResponse(
        evaluation_id="eval-002",
        timestamp="2026-05-04T12:00:00Z",
        safe_version="1.0.0",
        verdict="HOLD",
        experts=[expert],
        translation_report=_make_translation_report_llm(),
        recommendations=[
            {
                "text": "Provide risk management documentation.",
                "source_expert": "expert_governance_un",
                "source_dimension": "regulatory_compliance",
            }
        ],
    )
    assert len(resp.experts) == 1
    assert resp.experts[0].id == "expert_governance_un"
    assert len(resp.recommendations) == 1
