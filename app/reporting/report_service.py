from __future__ import annotations

from pathlib import Path

from app.reporting.json_exporter import export_json
from app.reporting.pdf_generator import generate_pdf
from app.safe_schemas import SAFEEvaluationResponse


def generate_reports(
    response: SAFEEvaluationResponse,
    output_dir: str = "reports",
) -> dict:
    """Generate PDF and JSON reports for a completed evaluation.

    Creates *output_dir* if it does not exist.

    Returns:
        {"pdf_path": str, "json_path": str}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = f"{response.evaluation_id}_{response.verdict}"
    pdf_path = str(out / f"{base}.pdf")
    json_path = str(out / f"{base}.json")

    generate_pdf(response, pdf_path)
    export_json(response, json_path)

    return {"pdf_path": pdf_path, "json_path": json_path}
