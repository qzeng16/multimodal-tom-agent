import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from schemas import (
    EpisodeInput,
    ReasoningResult,
    RetrievalResult,
    VerificationResult,
)


BENCHMARK_PATH = Path(
    "data/processed/benchmark.jsonl"
)

REASONING_PATH = Path(
    "data/processed/reasoning.jsonl"
)

VERIFICATION_PATH = Path(
    "data/processed/verification.jsonl"
)

RETRIEVAL_PATH = Path(
    "data/processed/retrieval.jsonl"
)

SUMMARY_PATH = Path(
    "data/processed/evaluation_summary.json"
)

ERROR_PATH = Path(
    "data/processed/error_analysis.jsonl"
)


def load_benchmark() -> Dict[str, EpisodeInput]:
    results = {}

    with open(
        BENCHMARK_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            sample = EpisodeInput(
                **json.loads(line)
            )

            results[
                sample.sample_id
            ] = sample

    return results


def load_reasoning() -> Dict[str, ReasoningResult]:
    results = {}

    with open(
        REASONING_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            result = ReasoningResult(
                **json.loads(line)
            )

            results[
                result.sample_id
            ] = result

    return results


def load_verification() -> Dict[
    str,
    VerificationResult,
]:
    results = {}

    with open(
        VERIFICATION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            result = VerificationResult(
                **json.loads(line)
            )

            results[
                result.sample_id
            ] = result

    return results


def load_retrieval() -> Dict[str, RetrievalResult]:
    results = {}

    with open(
        RETRIEVAL_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            result = RetrievalResult(
                **json.loads(line)
            )

            results[
                result.sample_id
            ] = result

    return results


def safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def classify_case(
    correct: bool,
    abstained: bool,
) -> str:

    if correct and not abstained:
        return "correct_accepted"

    if correct and abstained:
        return "correct_but_abstained"

    if (
        not correct
        and abstained
    ):
        return "error_caught_by_verifier"

    return "overconfident_wrong"


def compute_threshold_ablation(
    records: List[dict],
) -> List[dict]:

    thresholds = [
        0.50,
        0.58,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    results = []

    for threshold in thresholds:

        accepted = []

        for record in records:

            verification = record[
                "verification"
            ]

            reasoning = record[
                "reasoning"
            ]

            accept = (
                reasoning.parse_ok
                and verification.prediction
                is not None
                and (
                    verification
                    .support_margin
                    >= -0.05
                )
                and (
                    verification
                    .mapping_score
                    >= 0.35
                )
                and (
                    verification
                    .confidence
                    >= threshold
                )
            )

            if accept:
                accepted.append(
                    record
                )

        correct = sum(
            1
            for record in accepted
            if record["correct"]
        )

        results.append(
            {
                "threshold": threshold,
                "coverage": safe_divide(
                    len(accepted),
                    len(records),
                ),
                "selective_accuracy": (
                    safe_divide(
                        correct,
                        len(accepted),
                    )
                ),
                "accepted": len(
                    accepted
                ),
            }
        )

    return results


def modality_statistics(
    records: List[dict],
) -> dict:

    total_retrieved = 0
    visual_retrieved = 0
    text_retrieved = 0

    samples_with_visual = 0
    samples_with_text = 0

    used_visual = 0
    used_text = 0

    for record in records:

        retrieval = record[
            "retrieval"
        ]

        reasoning = record[
            "reasoning"
        ]

        sources_by_id = {
            item.evidence_id:
            item.source
            for item
            in retrieval.evidence
        }

        retrieved_sources = [
            item.source
            for item
            in retrieval.evidence
        ]

        total_retrieved += len(
            retrieved_sources
        )

        visual_retrieved += (
            retrieved_sources.count(
                "visual"
            )
        )

        text_retrieved += (
            retrieved_sources.count(
                "text"
            )
        )

        if (
            "visual"
            in retrieved_sources
        ):
            samples_with_visual += 1

        if (
            "text"
            in retrieved_sources
        ):
            samples_with_text += 1

        used_sources = {
            sources_by_id[
                evidence_id
            ]
            for evidence_id
            in reasoning.used_evidence_ids
            if evidence_id
            in sources_by_id
        }

        if "visual" in used_sources:
            used_visual += 1

        if "text" in used_sources:
            used_text += 1

    return {
        "retrieved_visual_fraction": (
            safe_divide(
                visual_retrieved,
                total_retrieved,
            )
        ),
        "retrieved_text_fraction": (
            safe_divide(
                text_retrieved,
                total_retrieved,
            )
        ),
        "samples_with_visual_retrieval": (
            safe_divide(
                samples_with_visual,
                len(records),
            )
        ),
        "samples_with_text_retrieval": (
            safe_divide(
                samples_with_text,
                len(records),
            )
        ),
        "samples_using_visual_evidence": (
            safe_divide(
                used_visual,
                len(records),
            )
        ),
        "samples_using_text_evidence": (
            safe_divide(
                used_text,
                len(records),
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    benchmark = load_benchmark()
    reasoning = load_reasoning()
    verification = (
        load_verification()
    )
    retrieval = load_retrieval()

    common_ids = [
        sample_id
        for sample_id
        in reasoning.keys()
        if (
            sample_id
            in benchmark
            and sample_id
            in verification
            and sample_id
            in retrieval
        )
    ]

    if args.max_samples is not None:
        common_ids = common_ids[
            :args.max_samples
        ]

    records = []

    for sample_id in common_ids:

        sample = benchmark[
            sample_id
        ]

        reasoning_result = reasoning[
            sample_id
        ]

        verification_result = (
            verification[
                sample_id
            ]
        )

        retrieval_result = retrieval[
            sample_id
        ]

        correct = (
            reasoning_result.prediction
            == sample.gold_answer
        )

        records.append(
            {
                "sample": sample,
                "reasoning": (
                    reasoning_result
                ),
                "verification": (
                    verification_result
                ),
                "retrieval": (
                    retrieval_result
                ),
                "correct": correct,
            }
        )

    total = len(records)

    if total == 0:
        print(
            "No aligned evaluation "
            "samples found."
        )
        return

    correct_count = sum(
        1
        for record in records
        if record["correct"]
    )

    parsed_count = sum(
        1
        for record in records
        if (
            record["reasoning"]
            .parse_ok
        )
    )

    accepted_records = [
        record
        for record in records
        if not (
            record[
                "verification"
            ].abstained
        )
    ]

    accepted_correct = sum(
        1
        for record
        in accepted_records
        if record["correct"]
    )

    abstained_count = (
        total
        - len(accepted_records)
    )

    correct_confidences = [
        record[
            "verification"
        ].confidence
        for record in records
        if record["correct"]
    ]

    wrong_confidences = [
        record[
            "verification"
        ].confidence
        for record in records
        if not record["correct"]
    ]

    per_type = defaultdict(
        lambda: {
            "count": 0,
            "correct": 0,
            "accepted": 0,
            "accepted_correct": 0,
        }
    )

    error_counts = defaultdict(
        int
    )

    errors = []

    for record in records:

        sample = record[
            "sample"
        ]

        reasoning_result = record[
            "reasoning"
        ]

        verification_result = (
            record[
                "verification"
            ]
        )

        question_type = (
            sample.question_type
        )

        stats = per_type[
            question_type
        ]

        stats["count"] += 1

        if record["correct"]:
            stats[
                "correct"
            ] += 1

        if not verification_result.abstained:
            stats[
                "accepted"
            ] += 1

            if record["correct"]:
                stats[
                    "accepted_correct"
                ] += 1

        category = classify_case(
            correct=record["correct"],
            abstained=(
                verification_result
                .abstained
            ),
        )

        error_counts[
            category
        ] += 1

        if (
            not record["correct"]
            or verification_result
            .abstained
        ):
            errors.append(
                {
                    "sample_id": (
                        sample.sample_id
                    ),
                    "question_type": (
                        sample.question_type
                    ),
                    "gold_answer": (
                        sample.gold_answer
                    ),
                    "prediction": (
                        reasoning_result
                        .prediction
                    ),
                    "confidence": (
                        verification_result
                        .confidence
                    ),
                    "mapping_score": (
                        verification_result
                        .mapping_score
                    ),
                    "support_margin": (
                        verification_result
                        .support_margin
                    ),
                    "status": (
                        verification_result
                        .verification_status
                    ),
                    "abstained": (
                        verification_result
                        .abstained
                    ),
                    "category": category,
                    "used_evidence_ids": (
                        reasoning_result
                        .used_evidence_ids
                    ),
                    "reasoning": (
                        reasoning_result
                        .reasoning_summary
                    ),
                }
            )

    per_type_output = {}

    for question_type, stats in (
        per_type.items()
    ):
        per_type_output[
            question_type
        ] = {
            "count": stats[
                "count"
            ],
            "accuracy": safe_divide(
                stats["correct"],
                stats["count"],
            ),
            "coverage": safe_divide(
                stats["accepted"],
                stats["count"],
            ),
            "selective_accuracy": (
                safe_divide(
                    stats[
                        "accepted_correct"
                    ],
                    stats[
                        "accepted"
                    ],
                )
            ),
        }

    avg_correct_confidence = (
        sum(
            correct_confidences
        )
        / len(
            correct_confidences
        )
        if correct_confidences
        else 0.0
    )

    avg_wrong_confidence = (
        sum(
            wrong_confidences
        )
        / len(
            wrong_confidences
        )
        if wrong_confidences
        else 0.0
    )

    summary = {
        "evaluation_scope": (
            "pilot_subset"
            if total < 900
            else "full_benchmark"
        ),
        "evaluated_samples": total,

        "reasoning_accuracy": (
            safe_divide(
                correct_count,
                total,
            )
        ),

        "parse_rate": (
            safe_divide(
                parsed_count,
                total,
            )
        ),

        "coverage": (
            safe_divide(
                len(
                    accepted_records
                ),
                total,
            )
        ),

        "abstention_rate": (
            safe_divide(
                abstained_count,
                total,
            )
        ),

        "selective_accuracy": (
            safe_divide(
                accepted_correct,
                len(
                    accepted_records
                ),
            )
        ),

        "average_confidence_correct": (
            avg_correct_confidence
        ),

        "average_confidence_wrong": (
            avg_wrong_confidence
        ),

        "per_question_type": (
            per_type_output
        ),

        "error_categories": dict(
            error_counts
        ),

        "threshold_ablation": (
            compute_threshold_ablation(
                records
            )
        ),

        "modality_usage": (
            modality_statistics(
                records
            )
        ),
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    with open(
        ERROR_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for error in errors:
            f.write(
                json.dumps(
                    error
                )
                + "\n"
            )

    print(
        "Evaluation complete."
    )

    print()
    print(
        f"Evaluated samples: "
        f"{total}"
    )

    print(
        f"Reasoning accuracy: "
        f"{summary['reasoning_accuracy']:.3f}"
    )

    print(
        f"Parse rate: "
        f"{summary['parse_rate']:.3f}"
    )

    print(
        f"Coverage: "
        f"{summary['coverage']:.3f}"
    )

    print(
        f"Selective accuracy: "
        f"{summary['selective_accuracy']:.3f}"
    )

    print(
        f"Abstention rate: "
        f"{summary['abstention_rate']:.3f}"
    )

    print()
    print(
        "Error categories:"
    )

    for category, count in (
        summary[
            "error_categories"
        ].items()
    ):
        print(
            f"  {category}: "
            f"{count}"
        )

    print()
    print(
        "Threshold ablation:"
    )

    for result in (
        summary[
            "threshold_ablation"
        ]
    ):
        print(
            f"  threshold="
            f"{result['threshold']:.2f} "
            f"coverage="
            f"{result['coverage']:.3f} "
            f"selective_accuracy="
            f"{result['selective_accuracy']:.3f}"
        )

    print()
    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Errors: "
        f"{ERROR_PATH}"
    )


if __name__ == "__main__":
    main()