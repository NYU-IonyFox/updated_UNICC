"""
Phase 6 tests — output layer: PDF and JSON report generation.
All tests use EXECUTION_MODE=rules; no LLM is loaded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["EXECUTION_MODE"] = "rules"

import pytest

from app.reporting.report_service import generate_reports
from app.reporting.json_exporter import export_json
from app.safe_schemas import (
    EvidenceBundle,
    ExpertOutput,
    SAFEEvaluationResponse,
    TranslationReport,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _mock_response(
    verdict: str = "APPROVE",
    evaluation_id: str = "test-eval-001",
    experts: list | None = None,
) -> SAFEEvaluationResponse:
    return SAFEEvaluationResponse(
        evaluation_id=evaluation_id,
        timestamp="2026-01-01T00:00:00+00:00",
        safe_version="1.2",
        verdict=verdict,
        primary_reason={
            "rule": "Rule 6: All Experts = LOW → PASS",
            "expert_summary": {
                "expert_1_security": "LOW",
                "expert_2_content": "LOW",
                "expert_3_governance": "LOW",
            },
        },
        additional_findings=[],
        submission_context={"input_type": "conversation", "target_name": "Test Agent"},
        experts=experts or [
            ExpertOutput(id="expert_adversarial_security", overall="LOW", triggered_dimensions=[]),
            ExpertOutput(id="expert_content_safety",       overall="LOW", triggered_dimensions=[]),
            ExpertOutput(id="expert_governance_un",        overall="LOW", triggered_dimensions=[]),
        ],
        recommendations=[],
        translation_report=TranslationReport(
            translation_applied=False,
            primary_language="eng_Latn",
        ),
    )


def _reject_response() -> SAFEEvaluationResponse:
    return _mock_response(verdict="REJECT", evaluation_id="test-eval-002")


def _hold_response() -> SAFEEvaluationResponse:
    return _mock_response(verdict="HOLD", evaluation_id="test-eval-003")


# ---------------------------------------------------------------------------
# 1. generate_reports() creates both files
# ---------------------------------------------------------------------------

def test_generate_reports_creates_pdf(tmp_path):
    paths = generate_reports(_mock_response(), output_dir=str(tmp_path))
    assert Path(paths["pdf_path"]).exists()


def test_generate_reports_creates_json(tmp_path):
    paths = generate_reports(_mock_response(), output_dir=str(tmp_path))
    assert Path(paths["json_path"]).exists()


def test_generate_reports_creates_output_dir(tmp_path):
    nested = tmp_path / "nested" / "reports"
    generate_reports(_mock_response(), output_dir=str(nested))
    assert nested.exists()


def test_generate_reports_file_naming(tmp_path):
    paths = generate_reports(_mock_response(), output_dir=str(tmp_path))
    assert "test-eval-001" in paths["pdf_path"]
    assert "APPROVE" in paths["pdf_path"]
    assert "test-eval-001" in paths["json_path"]
    assert "APPROVE" in paths["json_path"]


def test_generate_reports_reject_naming(tmp_path):
    paths = generate_reports(_reject_response(), output_dir=str(tmp_path))
    assert "_REJECT.pdf" in paths["pdf_path"]
    assert "_REJECT.json" in paths["json_path"]


# ---------------------------------------------------------------------------
# 2. Generated JSON is parseable and contains required fields
# ---------------------------------------------------------------------------

def test_json_contains_verdict(tmp_path):
    paths = generate_reports(_mock_response(), output_dir=str(tmp_path))
    with open(paths["json_path"], encoding="utf-8") as f:
        data = json.load(f)
    assert data["verdict"] == "APPROVE"


def test_json_contains_evaluation_id(tmp_path):
    paths = generate_reports(_mock_response(), output_dir=str(tmp_path))
    with open(paths["json_path"], encoding="utf-8") as f:
        data = json.load(f)
    assert data["evaluation_id"] == "test-eval-001"


def test_json_contains_experts(tmp_path):
    paths = generate_reports(_mock_response(), output_dir=str(tmp_path))
    with open(paths["json_path"], encoding="utf-8") as f:
        data = json.load(f)
    assert "experts" in data
    assert isinstance(data["experts"], list)


def test_json_is_utf8(tmp_path):
    paths = generate_reports(_mock_response(), output_dir=str(tmp_path))
    raw = Path(paths["json_path"]).read_bytes()
    decoded = raw.decode("utf-8")  # must not raise
    assert len(decoded) > 0


# ---------------------------------------------------------------------------
# 3. Generated PDF exists and has non-zero size
# ---------------------------------------------------------------------------

def test_pdf_nonzero_size(tmp_path):
    paths = generate_reports(_mock_response(), output_dir=str(tmp_path))
    assert Path(paths["pdf_path"]).stat().st_size > 0


def test_pdf_reject_nonzero_size(tmp_path):
    paths = generate_reports(_reject_response(), output_dir=str(tmp_path))
    assert Path(paths["pdf_path"]).stat().st_size > 0


def test_pdf_hold_nonzero_size(tmp_path):
    paths = generate_reports(_hold_response(), output_dir=str(tmp_path))
    assert Path(paths["pdf_path"]).stat().st_size > 0


# ---------------------------------------------------------------------------
# 4. GET /report/{evaluation_id} returns 200 when file exists
# ---------------------------------------------------------------------------

def test_get_report_returns_200(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main_module
    from app.main import app

    paths = generate_reports(_mock_response(), output_dir=str(tmp_path))
    monkeypatch.setattr(main_module, "REPORTS_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/report/test-eval-001")
    assert resp.status_code == 200


def test_get_report_content_type_pdf(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main_module
    from app.main import app

    generate_reports(_mock_response(), output_dir=str(tmp_path))
    monkeypatch.setattr(main_module, "REPORTS_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/report/test-eval-001")
    assert "pdf" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# 5. GET /report/{evaluation_id} returns 404 when not found
# ---------------------------------------------------------------------------

def test_get_report_returns_404_when_missing(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main_module
    from app.main import app

    monkeypatch.setattr(main_module, "REPORTS_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/report/nonexistent-evaluation-id")
    assert resp.status_code == 404


def test_get_report_404_has_detail(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main_module
    from app.main import app

    monkeypatch.setattr(main_module, "REPORTS_DIR", str(tmp_path))

    client = TestClient(app)
    resp = client.get("/report/no-such-id")
    body = resp.json()
    assert "detail" in body


# ---------------------------------------------------------------------------
# 6. PDF failure in run_evaluation() → result still returned, report_path=""
# ---------------------------------------------------------------------------

def test_pdf_failure_doesnt_crash_run_evaluation(monkeypatch):
    import app.orchestrator as orch
    from app.orchestrator import run_evaluation

    def _always_fail(*args, **kwargs):
        raise RuntimeError("Simulated PDF generation failure")

    monkeypatch.setattr(orch, "generate_reports", _always_fail)

    bundle = EvidenceBundle(
        input_type="conversation",
        translation_report=TranslationReport(
            translation_applied=False,
            primary_language="eng_Latn",
        ),
    )
    result = run_evaluation(bundle)
    assert isinstance(result, SAFEEvaluationResponse)
    assert result.report_path == ""


def test_pdf_failure_verdict_still_correct(monkeypatch):
    import app.orchestrator as orch
    from app.orchestrator import run_evaluation

    monkeypatch.setattr(orch, "generate_reports", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))

    bundle = EvidenceBundle(
        input_type="conversation",
        translation_report=TranslationReport(
            translation_applied=False,
            primary_language="eng_Latn",
        ),
    )
    result = run_evaluation(bundle)
    assert result.verdict in ("APPROVE", "HOLD", "REJECT")
