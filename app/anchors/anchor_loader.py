from __future__ import annotations

import json
from pathlib import Path

_EXPERT_KEY_MAP = {
    "expert_1_security": "expert_adversarial_security",
    "expert_2_content": "expert_content_safety",
    "expert_3_governance": "expert_governance_un",
}


def load_anchors(json_path: str) -> dict:
    """Read framework_anchors_v2.json and return primary anchors keyed by expert_id.

    Returns:
        {
          "expert_adversarial_security": [
            {"dimension": str, "criticality": str, "primary_anchor": {...}},
            ...
          ],
          ...
        }
    Supplementary anchors are NOT included — audit log use only.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    result: dict = {}
    for json_key, expert_id in _EXPERT_KEY_MAP.items():
        if json_key not in data:
            continue
        dims = data[json_key].get("dimensions", [])
        result[expert_id] = [
            {
                "dimension": d["dimension"],
                "criticality": d["criticality"],
                "primary_anchor": d["primary_anchor"],
            }
            for d in dims
        ]
    return result


def build_anchor_table(expert_id: str, anchors: dict) -> str:
    """Format primary anchors for injection into the {{anchor_table}} placeholder.

    Each line: "- <dimension>: <framework> § <section> — <provision>"
    Supplementary anchors are never included.
    """
    lines = []
    for item in anchors.get(expert_id, []):
        dim = item["dimension"]
        pa = item["primary_anchor"]
        framework = pa.get("framework", "")
        section = pa.get("section", "")
        provision = pa.get("provision", "")
        lines.append(f"- {dim}: {framework} § {section} — {provision}")
    return "\n".join(lines)
