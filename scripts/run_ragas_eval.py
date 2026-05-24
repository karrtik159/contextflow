"""
Run RAGAS evaluation against a local or remote FastAPI backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evals.ragas_framework import (
    FastAPIRagTarget,
    evaluate_cases,
    load_eval_cases,
    materialize_responses,
    summarise_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for the ContextFlow RAG pipeline.")
    parser.add_argument(
        "--dataset",
        default="tests/test_eval/fixtures/ragas_cases.json",
        help="Path to a JSON dataset of RAGAS eval cases.",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8000",
        help="FastAPI base URL used to materialize missing responses.",
    )
    parser.add_argument(
        "--output",
        default="tests/test_eval/last_ragas_summary.json",
        help="Where to write the summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = load_eval_cases(args.dataset)
    cases = materialize_responses(cases, FastAPIRagTarget(args.api_base_url))
    result = evaluate_cases(cases)
    summary = summarise_result(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
