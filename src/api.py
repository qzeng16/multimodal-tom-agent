import json
from pathlib import Path
from typing import Dict, Optional

from fastapi import (
    FastAPI,
    HTTPException,
)


app = FastAPI(
    title=(
        "Multimodal Theory-of-Mind Agent"
    ),
    version="1.0.0",
)


BENCHMARK_PATH = Path(
    "data/processed/benchmark.jsonl"
)

RETRIEVAL_PATH = Path(
    "data/processed/retrieval.jsonl"
)

REASONING_PATH = Path(
    "data/processed/reasoning.jsonl"
)

VERIFICATION_PATH = Path(
    "data/processed/verification.jsonl"
)


def load_index(
    path: Path,
) -> Dict[str, dict]:

    results = {}

    if not path.exists():
        return results

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            record = json.loads(
                line
            )

            sample_id = (
                record.get(
                    "sample_id"
                )
            )

            if sample_id:
                results[
                    sample_id
                ] = record

    return results


def build_sample_response(
    sample_id: str,
) -> Optional[dict]:

    benchmark = load_index(
        BENCHMARK_PATH
    )

    if sample_id not in benchmark:
        return None

    retrieval = load_index(
        RETRIEVAL_PATH
    )

    reasoning = load_index(
        REASONING_PATH
    )

    verification = load_index(
        VERIFICATION_PATH
    )

    return {
        "sample": benchmark[
            sample_id
        ],
        "retrieval": retrieval.get(
            sample_id
        ),
        "reasoning": reasoning.get(
            sample_id
        ),
        "verification": (
            verification.get(
                sample_id
            )
        ),
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get(
    "/samples/{sample_id}"
)
def get_sample(
    sample_id: str,
):

    result = (
        build_sample_response(
            sample_id
        )
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown sample: "
                f"{sample_id}"
            ),
        )

    return result