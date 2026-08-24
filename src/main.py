import argparse
import json
from pathlib import Path
from typing import Optional


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


def load_jsonl_record(
    path: Path,
    sample_id: str,
) -> Optional[dict]:

    if not path.exists():
        return None

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

            if (
                record.get(
                    "sample_id"
                )
                == sample_id
            ):
                return record

    return None


def inspect_sample(
    sample_id: str,
) -> None:

    benchmark = (
        load_jsonl_record(
            BENCHMARK_PATH,
            sample_id,
        )
    )

    if benchmark is None:
        raise ValueError(
            f"Sample not found: "
            f"{sample_id}"
        )

    retrieval = (
        load_jsonl_record(
            RETRIEVAL_PATH,
            sample_id,
        )
    )

    reasoning = (
        load_jsonl_record(
            REASONING_PATH,
            sample_id,
        )
    )

    verification = (
        load_jsonl_record(
            VERIFICATION_PATH,
            sample_id,
        )
    )

    print(
        f"Sample: {sample_id}"
    )

    print(
        f"Question type: "
        f"{benchmark['question_type']}"
    )

    print()
    print(
        benchmark[
            "question"
        ]
    )

    print()
    print(
        f"Gold answer: "
        f"{benchmark['gold_answer']}"
    )

    if retrieval:
        print()
        print(
            "Retrieved evidence:"
        )

        for item in (
            retrieval[
                "evidence"
            ]
        ):
            print(
                f"[{item['rank']}] "
                f"{item['source']} "
                f"score="
                f"{item['retrieval_score']:.3f}"
            )

            print(
                f"  {item['content']}"
            )

    if reasoning:
        print()
        print(
            f"Prediction: "
            f"{reasoning['prediction']}"
        )

        print(
            f"Parse OK: "
            f"{reasoning['parse_ok']}"
        )

        print()
        print(
            "Reasoning:"
        )

        print(
            reasoning[
                "reasoning_summary"
            ]
        )

    if verification:
        print()
        print(
            f"Confidence: "
            f"{verification['confidence']:.3f}"
        )

        print(
            f"Status: "
            f"{verification['verification_status']}"
        )

        print(
            f"Abstained: "
            f"{verification['abstained']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect outputs from the "
            "Multimodal ToM Agent."
        )
    )

    parser.add_argument(
        "--sample-id",
        required=True,
    )

    args = parser.parse_args()

    inspect_sample(
        args.sample_id
    )


if __name__ == "__main__":
    main()