import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer

from schemas import (
    EpisodeInput,
    ReasoningResult,
    RetrievalResult,
    VerificationResult,
)


MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
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

OUTPUT_PATH = Path(
    "data/processed/verification.jsonl"
)


def select_device(
    requested: str,
) -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def load_benchmark() -> Dict[str, EpisodeInput]:
    samples = {}

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

            samples[sample.sample_id] = sample

    return samples


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

            results[result.sample_id] = result

    return results


def load_reasoning() -> List[ReasoningResult]:
    results = []

    with open(
        REASONING_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            results.append(
                ReasoningResult(
                    **json.loads(line)
                )
            )

    return results


def clean_choice(
    choice: str,
) -> Tuple[str, str]:

    match = re.match(
        r"\s*([ABC])\)\s*(.+)",
        choice,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Invalid choice: {choice}"
        )

    return (
        match.group(1).upper(),
        match.group(2).strip(),
    )


def parse_answer_text(
    raw_output: str,
) -> Optional[str]:

    match = re.search(
        r"ANSWER\s*:\s*(.+)",
        raw_output,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return (
        match.group(1)
        .splitlines()[0]
        .strip()
    )


def cosine_scores(
    model,
    query: str,
    texts: List[str],
) -> List[float]:

    embeddings = model.encode(
        [query] + texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = embeddings[0]
    text_embeddings = embeddings[1:]

    scores = torch.matmul(
        text_embeddings,
        query_embedding,
    )

    return (
        scores.detach()
        .cpu()
        .tolist()
    )


def recover_mapping_score(
    reasoning: ReasoningResult,
    sample: EpisodeInput,
    model,
) -> float:

    if reasoning.mapping_score is not None:
        return float(
            reasoning.mapping_score
        )

    answer_text = reasoning.answer_text

    if answer_text is None:
        answer_text = parse_answer_text(
            reasoning.raw_output
        )

    if not answer_text:
        return 0.0

    choices = [
        clean_choice(choice)
        for choice in sample.choices
    ]

    labels = [
        label
        for label, _ in choices
    ]

    texts = [
        text
        for _, text in choices
    ]

    scores = cosine_scores(
        model=model,
        query=answer_text,
        texts=texts,
    )

    if reasoning.prediction not in labels:
        return 0.0

    index = labels.index(
        reasoning.prediction
    )

    return float(
        scores[index]
    )


def compute_choice_support(
    sample: EpisodeInput,
    retrieval: RetrievalResult,
    model,
) -> Dict[str, float]:

    choices = [
        clean_choice(choice)
        for choice in sample.choices
    ]

    evidence_texts = [
        item.content
        for item in retrieval.evidence
    ]

    support = {}

    for label, choice_text in choices:

        scores = cosine_scores(
            model=model,
            query=choice_text,
            texts=evidence_texts,
        )

        support[label] = (
            max(scores)
            if scores
            else 0.0
        )

    return support


def compute_evidence_score(
    reasoning: ReasoningResult,
    retrieval: RetrievalResult,
) -> float:

    retrieval_by_id = {
        item.evidence_id: item
        for item in retrieval.evidence
    }

    selected = [
        retrieval_by_id[evidence_id]
        for evidence_id
        in reasoning.used_evidence_ids
        if evidence_id in retrieval_by_id
    ]

    if not selected:
        selected = retrieval.evidence[:2]

    if not selected:
        return 0.0

    score = sum(
        item.retrieval_score
        for item in selected
    ) / len(selected)

    return float(score)


def sigmoid(
    value: float,
) -> float:
    return 1.0 / (
        1.0 + math.exp(-value)
    )


def clamp(
    value: float,
) -> float:
    return max(
        0.0,
        min(1.0, value),
    )


def build_verification(
    reasoning: ReasoningResult,
    sample: EpisodeInput,
    retrieval: RetrievalResult,
    model,
) -> VerificationResult:

    mapping_score = (
        recover_mapping_score(
            reasoning=reasoning,
            sample=sample,
            model=model,
        )
    )

    support = compute_choice_support(
        sample=sample,
        retrieval=retrieval,
        model=model,
    )

    prediction = reasoning.prediction

    if (
        prediction is None
        or prediction not in support
    ):
        selected_support = 0.0
        alternative_support = max(
            support.values()
        )
    else:
        selected_support = support[
            prediction
        ]

        alternatives = [
            score
            for label, score
            in support.items()
            if label != prediction
        ]

        alternative_support = max(
            alternatives
        )

    support_margin = (
        selected_support
        - alternative_support
    )

    evidence_score = (
        compute_evidence_score(
            reasoning=reasoning,
            retrieval=retrieval,
        )
    )

    mapping_component = clamp(
        (mapping_score + 1.0) / 2.0
    )

    support_component = clamp(
        (selected_support + 1.0)
        / 2.0
    )

    evidence_component = clamp(
        (evidence_score + 1.0)
        / 2.0
    )

    margin_component = sigmoid(
        5.0 * support_margin
    )

    citation_component = (
        1.0
        if reasoning.used_evidence_ids
        else 0.5
    )

    confidence = (
        0.35 * mapping_component
        + 0.30 * support_component
        + 0.15 * margin_component
        + 0.10 * evidence_component
        + 0.10 * citation_component
    )

    confidence = clamp(
        confidence
    )

    if (
        not reasoning.parse_ok
        or prediction is None
    ):
        status = "uncertain"
        abstained = True

    elif support_margin < -0.05:
        status = "conflicted"
        abstained = True

    elif (
        mapping_score < 0.35
        or confidence < 0.58
    ):
        status = "uncertain"
        abstained = True

    else:
        status = "supported"
        abstained = False

    return VerificationResult(
        sample_id=sample.sample_id,
        episode_id=sample.episode_id,
        prediction=prediction,
        confidence=confidence,
        mapping_score=mapping_score,
        selected_support=(
            selected_support
        ),
        alternative_support=(
            alternative_support
        ),
        support_margin=(
            support_margin
        ),
        evidence_score=evidence_score,
        verification_status=status,
        abstained=abstained,
        used_evidence_ids=(
            reasoning.used_evidence_ids
        ),
    )


def show_sanity_check(
    result: VerificationResult,
    sample: EpisodeInput,
):
    print()
    print("Sanity check:")

    print(
        f"Sample: "
        f"{result.sample_id}"
    )

    print(
        f"Prediction: "
        f"{result.prediction}"
    )

    print(
        f"Gold answer: "
        f"{sample.gold_answer}"
    )

    print(
        f"Confidence: "
        f"{result.confidence:.3f}"
    )

    print(
        f"Mapping score: "
        f"{result.mapping_score:.3f}"
    )

    print(
        f"Selected support: "
        f"{result.selected_support:.3f}"
    )

    print(
        f"Alternative support: "
        f"{result.alternative_support:.3f}"
    )

    print(
        f"Support margin: "
        f"{result.support_margin:.3f}"
    )

    print(
        f"Status: "
        f"{result.verification_status}"
    )

    print(
        f"Abstained: "
        f"{result.abstained}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    device = select_device(
        args.device
    )

    print(
        f"Using device: {device}"
    )

    benchmark = load_benchmark()
    retrieval = load_retrieval()
    reasoning = load_reasoning()

    if args.max_samples is not None:
        reasoning = reasoning[
            :args.max_samples
        ]

    print(
        f"Reasoning samples: "
        f"{len(reasoning)}"
    )

    print(
        "Loading verifier model..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    print(
        "Verifier model loaded."
    )

    first_result = None
    first_sample = None
    processed = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:

        for item in reasoning:

            sample = benchmark.get(
                item.sample_id
            )

            retrieval_result = (
                retrieval.get(
                    item.sample_id
                )
            )

            if (
                sample is None
                or retrieval_result is None
            ):
                continue

            result = build_verification(
                reasoning=item,
                sample=sample,
                retrieval=(
                    retrieval_result
                ),
                model=model,
            )

            output_file.write(
                result.model_dump_json()
                + "\n"
            )

            processed += 1

            if first_result is None:
                first_result = result
                first_sample = sample

    print()
    print(
        "Verification complete."
    )

    print(
        f"Processed samples: "
        f"{processed}"
    )

    print(
        f"Output: "
        f"{OUTPUT_PATH}"
    )

    if (
        first_result is not None
        and first_sample is not None
    ):
        show_sanity_check(
            result=first_result,
            sample=first_sample,
        )


if __name__ == "__main__":
    main()