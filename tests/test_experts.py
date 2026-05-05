"""
Phase 4 tests — new SAFE Expert Council (23 dimensions).
All tests use EXECUTION_MODE=rules; no LLM is loaded.
"""
from __future__ import annotations

import os
from pathlib import Path

# Must be set before importing experts so execution_mode() reads it correctly.
os.environ["EXECUTION_MODE"] = "rules"

import pytest

from app.experts.expert_adversarial import AdversarialSecurityExpert
from app.experts.expert_content import ContentSafetyExpert
from app.experts.expert_governance import GovernanceExpert
from app.experts.base import compute_overall
from app.anchors.anchor_loader import build_anchor_table, load_anchors

_ANCHORS_PATH = str(
    Path(__file__).resolve().parent.parent / "app" / "anchors" / "framework_anchors_v2.json"
)

# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------

def test_adversarial_expert_instantiates():
    expert = AdversarialSecurityExpert()
    assert expert is not None
    assert expert.expert_id == "expert_adversarial_security"


def test_content_expert_instantiates():
    expert = ContentSafetyExpert()
    assert expert is not None
    assert expert.expert_id == "expert_content_safety"


def test_governance_expert_instantiates():
    expert = GovernanceExpert()
    assert expert is not None
    assert expert.expert_id == "expert_governance_un"


# ---------------------------------------------------------------------------
# 2. assess() returns correct keys
# ---------------------------------------------------------------------------

def test_adversarial_assess_has_required_keys():
    result = AdversarialSecurityExpert().assess({})
    assert "id" in result
    assert "overall" in result
    assert "triggered_dimensions" in result


def test_content_assess_has_required_keys():
    result = ContentSafetyExpert().assess({})
    assert "id" in result
    assert "overall" in result
    assert "triggered_dimensions" in result


def test_governance_assess_has_required_keys():
    result = GovernanceExpert().assess({})
    assert "id" in result
    assert "overall" in result
    assert "triggered_dimensions" in result


# ---------------------------------------------------------------------------
# 3. rules mode — all LOW, triggered_dimensions empty
# ---------------------------------------------------------------------------

def test_adversarial_rules_mode_all_low():
    result = AdversarialSecurityExpert().assess({})
    assert result["id"] == "expert_adversarial_security"
    assert result["overall"] == "LOW"
    assert result["triggered_dimensions"] == []


def test_content_rules_mode_all_low():
    result = ContentSafetyExpert().assess({})
    assert result["id"] == "expert_content_safety"
    assert result["overall"] == "LOW"
    assert result["triggered_dimensions"] == []


def test_governance_rules_mode_all_low():
    result = GovernanceExpert().assess({})
    assert result["id"] == "expert_governance_un"
    assert result["overall"] == "LOW"
    assert result["triggered_dimensions"] == []


# ---------------------------------------------------------------------------
# 4. compute_overall() — five E-rule logic checks
# ---------------------------------------------------------------------------

def test_erule1_core_high_returns_high():
    scores = [{"tier": "CORE", "level": "HIGH"}, {"tier": "IMPORTANT", "level": "LOW"}]
    assert compute_overall(scores) == "HIGH"


def test_erule1_core_high_overrides_important_high():
    scores = [{"tier": "CORE", "level": "HIGH"}, {"tier": "IMPORTANT", "level": "HIGH"}]
    assert compute_overall(scores) == "HIGH"


def test_erule2_core_medium_returns_medium():
    scores = [{"tier": "CORE", "level": "MEDIUM"}, {"tier": "IMPORTANT", "level": "LOW"}]
    assert compute_overall(scores) == "MEDIUM"


def test_erule2_core_medium_no_important_high():
    scores = [
        {"tier": "CORE", "level": "MEDIUM"},
        {"tier": "IMPORTANT", "level": "HIGH"},
    ]
    # E-Rule 1 not triggered (no CORE=HIGH); E-Rule 2 fires (CORE=MEDIUM)
    assert compute_overall(scores) == "MEDIUM"


def test_erule3_two_important_high_returns_high():
    scores = [
        {"tier": "IMPORTANT", "level": "HIGH"},
        {"tier": "IMPORTANT", "level": "HIGH"},
        {"tier": "CORE", "level": "LOW"},
    ]
    assert compute_overall(scores) == "HIGH"


def test_erule4_one_important_high_returns_medium():
    scores = [
        {"tier": "IMPORTANT", "level": "HIGH"},
        {"tier": "IMPORTANT", "level": "LOW"},
        {"tier": "CORE", "level": "LOW"},
    ]
    assert compute_overall(scores) == "MEDIUM"


def test_erule5_all_low_returns_low():
    scores = [
        {"tier": "CORE", "level": "LOW"},
        {"tier": "IMPORTANT", "level": "LOW"},
        {"tier": "CORE", "level": "LOW"},
    ]
    assert compute_overall(scores) == "LOW"


def test_erule5_empty_scores_returns_low():
    assert compute_overall([]) == "LOW"


# ---------------------------------------------------------------------------
# 5. anchor_loader — load_anchors()
# ---------------------------------------------------------------------------

def test_load_anchors_returns_dict():
    anchors = load_anchors(_ANCHORS_PATH)
    assert isinstance(anchors, dict)


def test_load_anchors_has_all_three_experts():
    anchors = load_anchors(_ANCHORS_PATH)
    assert "expert_adversarial_security" in anchors
    assert "expert_content_safety" in anchors
    assert "expert_governance_un" in anchors


def test_load_anchors_adversarial_has_8_dimensions():
    anchors = load_anchors(_ANCHORS_PATH)
    assert len(anchors["expert_adversarial_security"]) == 8


def test_load_anchors_content_has_7_dimensions():
    anchors = load_anchors(_ANCHORS_PATH)
    assert len(anchors["expert_content_safety"]) == 7


def test_load_anchors_governance_has_8_dimensions():
    anchors = load_anchors(_ANCHORS_PATH)
    assert len(anchors["expert_governance_un"]) == 8


def test_load_anchors_each_item_has_primary_anchor():
    anchors = load_anchors(_ANCHORS_PATH)
    for expert_id, dims in anchors.items():
        for dim in dims:
            assert "primary_anchor" in dim, f"Missing primary_anchor in {expert_id}/{dim}"
            assert "dimension" in dim
            assert "criticality" in dim


def test_load_anchors_no_supplementary_anchors_in_output():
    anchors = load_anchors(_ANCHORS_PATH)
    for expert_id, dims in anchors.items():
        for dim in dims:
            assert "supplementary_anchors" not in dim, (
                f"supplementary_anchors must not appear in load_anchors output: {expert_id}/{dim}"
            )


# ---------------------------------------------------------------------------
# 6. anchor_loader — build_anchor_table()
# ---------------------------------------------------------------------------

def test_build_anchor_table_returns_string():
    anchors = load_anchors(_ANCHORS_PATH)
    table = build_anchor_table("expert_adversarial_security", anchors)
    assert isinstance(table, str)
    assert len(table) > 0


def test_build_anchor_table_adversarial_has_8_lines():
    anchors = load_anchors(_ANCHORS_PATH)
    table = build_anchor_table("expert_adversarial_security", anchors)
    lines = [ln for ln in table.split("\n") if ln.strip()]
    assert len(lines) == 8


def test_build_anchor_table_content_has_7_lines():
    anchors = load_anchors(_ANCHORS_PATH)
    table = build_anchor_table("expert_content_safety", anchors)
    lines = [ln for ln in table.split("\n") if ln.strip()]
    assert len(lines) == 7


def test_build_anchor_table_governance_has_8_lines():
    anchors = load_anchors(_ANCHORS_PATH)
    table = build_anchor_table("expert_governance_un", anchors)
    lines = [ln for ln in table.split("\n") if ln.strip()]
    assert len(lines) == 8


def test_build_anchor_table_lines_start_with_dash():
    anchors = load_anchors(_ANCHORS_PATH)
    table = build_anchor_table("expert_adversarial_security", anchors)
    for line in table.split("\n"):
        if line.strip():
            assert line.startswith("- "), f"Line does not start with '- ': {line!r}"


def test_build_anchor_table_no_supplementary_content():
    anchors = load_anchors(_ANCHORS_PATH)
    for expert_id in ("expert_adversarial_security", "expert_content_safety", "expert_governance_un"):
        table = build_anchor_table(expert_id, anchors)
        assert "supplementary" not in table.lower(), (
            f"supplementary anchor content must not appear in anchor table for {expert_id}"
        )


def test_build_anchor_table_unknown_expert_returns_empty():
    anchors = load_anchors(_ANCHORS_PATH)
    table = build_anchor_table("nonexistent_expert", anchors)
    assert table == ""
