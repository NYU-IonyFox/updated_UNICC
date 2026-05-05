from __future__ import annotations

import json

from app.safe_schemas import SAFEEvaluationResponse


def export_json(response: SAFEEvaluationResponse, path: str) -> str:
    """Serialize SAFEEvaluationResponse to a UTF-8 JSON file. Returns the path."""
    data = response.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path
