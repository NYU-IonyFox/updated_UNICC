"""
Phase 7 tests — HTML frontend: index, result page, and /submit endpoint.
All tests use EXECUTION_MODE=rules; no LLM is loaded.
"""
from __future__ import annotations

import json
import os

os.environ["EXECUTION_MODE"] = "rules"

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. GET / returns 200 and contains "SAFE"
# ---------------------------------------------------------------------------

def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_index_contains_safe(client):
    resp = client.get("/")
    assert "SAFE" in resp.text


# ---------------------------------------------------------------------------
# 2. GET /result returns 200
# ---------------------------------------------------------------------------

def test_result_page_returns_200(client):
    resp = client.get("/result")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. POST /submit returns a SAFEEvaluationResponse with a valid verdict
# ---------------------------------------------------------------------------

def test_submit_behavior_only_returns_verdict(client):
    conv = json.dumps([{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}])
    resp = client.post("/submit", data={
        "evaluation_mode": "behavior_only",
        "agent_name": "Test Agent",
        "description": "A test agent",
        "domain": "Other",
        "high_autonomy": "false",
        "selected_policies": json.dumps(["eu_ai_act"]),
        "conversation": conv,
        "source_type": "github_url",
        "github_url": "",
        "local_path": "",
        "repo_description": "",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] in ("APPROVE", "HOLD", "REJECT")


def test_submit_returns_evaluation_id(client):
    resp = client.post("/submit", data={
        "evaluation_mode": "behavior_only",
        "agent_name": "ID Test",
        "description": "",
        "domain": "Other",
        "high_autonomy": "false",
        "selected_policies": json.dumps(["eu_ai_act"]),
        "conversation": json.dumps([]),
        "source_type": "github_url",
        "github_url": "",
        "local_path": "",
        "repo_description": "",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "evaluation_id" in body
    assert body["evaluation_id"]
