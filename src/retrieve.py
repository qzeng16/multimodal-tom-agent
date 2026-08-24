import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from sentence_transformers import (
    SentenceTransformer,
)

from schemas import (
    EpisodeEvidence,
    EpisodeInput,
    RetrievedEvidenceItem,
    RetrievalResult,
)


MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

BENCHMARK_PATH = Path(
    "data/processed/benchmark.jsonl"
)

EVIDENCE_PATH = Path(
    "data/processed/evidence.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/retrieval.jsonl"
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


def load_samples() -> List[EpisodeInput]:
    samples = []

    with open(
        BENCHMARK_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            samples.append(
                EpisodeInput(
                    **json.loads(line)
                )
            )

    return samples


def load_evidence() -> Dict[
    str,
    EpisodeEvidence,
]:
    evidence_by_episode = {}

    with open(
        EVIDENCE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            episode = EpisodeEvidence(
                **json.loads(line)
            )

            evidence_by_episode[
                episode.episode_id
            ] = episode

    return evidence_by_episode


def group_samples_by_episode(
    samples: List[EpisodeInput],
):
    grouped = defaultdict(list)

    for sample in samples:
        grouped[
            sample.episode_id
        ].append(sample)

    return grouped


def load_embedding_model(
    device: str,
):
    print(
        f"Loading embedding model "
        f"on {device}..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    print(
        "Embedding model loaded."
    )

    return model


def encode_evidence(
    model,
    evidence,
):
    texts = [
        item.content
        for item in evidence
    ]

    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings


def encode_query(
    model,
    question: str,
):
    embedding = model.encode(
        question,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding


def compute_scores(
    query_embedding,
    evidence_embeddings,
) -> List[float]:

    if query_embedding.dim() == 1:
        query_embedding = (
            query_embedding.unsqueeze(0)
        )

    scores = torch.matmul(
        evidence_embeddings,
        query_embedding.T,
    ).squeeze(-1)

    return (
        scores
        .detach()
        .cpu()
        .tolist()
    )


def balanced_top_k(
    evidence,
    scores,
    top_k: int,
    balance_modalities: bool,
):
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    selected = ranked_indices[
        :top_k
    ]

    if (
        not balance_modalities
        or top_k < 2
    ):
        return selected

    available_sources = {
        item.source
        for item in evidence
    }

    required_sources = [
        source
        for source in (
            "visual",
            "text",
        )
        if source in available_sources
    ]

    for required_source in required_sources:

        if any(
            evidence[index].source
            == required_source
            for index in selected
        ):
            continue

        replacement = next(
            (
                index
                for index
                in ranked_indices
                if evidence[index].source
                == required_source
                and index not in selected
            ),
            None,
        )

        if replacement is None:
            continue

        removable = sorted(
            selected,
            key=lambda index: scores[index],
        )

        index_to_remove = None

        for candidate in removable:

            candidate_source = (
                evidence[candidate].source
            )

            same_source_count = sum(
                evidence[index].source
                == candidate_source
                for index in selected
            )

            if same_source_count > 1:
                index_to_remove = candidate
                break

        if index_to_remove is None:
            index_to_remove = removable[0]

        selected.remove(
            index_to_remove
        )

        selected.append(
            replacement
        )

    selected.sort(
        key=lambda index: scores[index],
        reverse=True,
    )

    return selected


def build_result(
    sample: EpisodeInput,
    episode_evidence: EpisodeEvidence,
    evidence_embeddings,
    model,
    top_k: int,
    balance_modalities: bool,
):
    query_embedding = encode_query(
        model=model,
        question=sample.question,
    )

    scores = compute_scores(
        query_embedding=query_embedding,
        evidence_embeddings=(
            evidence_embeddings
        ),
    )

    actual_top_k = min(
        top_k,
        len(
            episode_evidence.evidence
        ),
    )

    selected_indices = balanced_top_k(
        evidence=(
            episode_evidence.evidence
        ),
        scores=scores,
        top_k=actual_top_k,
        balance_modalities=(
            balance_modalities
        ),
    )

    retrieved = []

    for rank, index in enumerate(
        selected_indices,
        start=1,
    ):
        item = (
            episode_evidence
            .evidence[index]
        )

        retrieved.append(
            RetrievedEvidenceItem(
                evidence_id=(
                    item.evidence_id
                ),
                source=item.source,
                content=item.content,
                timestamp=(
                    item.timestamp
                ),
                frame_path=(
                    item.frame_path
                ),
                retrieval_score=(
                    float(
                        scores[index]
                    )
                ),
                rank=rank,
            )
        )

    return RetrievalResult(
        sample_id=sample.sample_id,
        episode_id=sample.episode_id,
        question_type=(
            sample.question_type
        ),
        question=sample.question,
        top_k=actual_top_k,
        modality_balanced=(
            balance_modalities
        ),
        evidence=retrieved,
    )


def show_sanity_check(
    result: RetrievalResult,
):
    print()
    print("Sanity check:")
    print(
        f"Sample: "
        f"{result.sample_id}"
    )

    print(
        f"Question type: "
        f"{result.question_type}"
    )

    print()
    print("Top evidence:")

    for item in result.evidence:
        print(
            f"[{item.rank}] "
            f"{item.source.upper()} "
            f"score="
            f"{item.retrieval_score:.3f}"
        )

        print(
            f"    {item.content}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

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
        "--no-balance",
        action="store_true",
    )

    args = parser.parse_args()

    device = select_device(
        args.device
    )

    balance_modalities = (
        not args.no_balance
    )

    print(
        f"Using device: {device}"
    )

    print(
        "Modality balancing: "
        f"{balance_modalities}"
    )

    samples = load_samples()

    if args.max_samples is not None:
        samples = samples[
            :args.max_samples
        ]

    evidence_by_episode = (
        load_evidence()
    )

    grouped_samples = (
        group_samples_by_episode(
            samples
        )
    )

    print(
        f"Samples to process: "
        f"{len(samples)}"
    )

    print(
        f"Episodes involved: "
        f"{len(grouped_samples)}"
    )

    model = load_embedding_model(
        device
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = 0
    first_result = None

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as output_file:

        for episode_id, episode_samples in (
            grouped_samples.items()
        ):
            episode_evidence = (
                evidence_by_episode.get(
                    episode_id
                )
            )

            if episode_evidence is None:
                print(
                    f"Warning: no evidence "
                    f"for episode "
                    f"{episode_id}"
                )
                continue

            if not (
                episode_evidence.evidence
            ):
                print(
                    f"Warning: empty evidence "
                    f"for episode "
                    f"{episode_id}"
                )
                continue

            evidence_embeddings = (
                encode_evidence(
                    model=model,
                    evidence=(
                        episode_evidence
                        .evidence
                    ),
                )
            )

            for sample in episode_samples:

                result = build_result(
                    sample=sample,
                    episode_evidence=(
                        episode_evidence
                    ),
                    evidence_embeddings=(
                        evidence_embeddings
                    ),
                    model=model,
                    top_k=args.top_k,
                    balance_modalities=(
                        balance_modalities
                    ),
                )

                output_file.write(
                    result.model_dump_json()
                    + "\n"
                )

                processed += 1

                if first_result is None:
                    first_result = result

                if (
                    processed % 100 == 0
                ):
                    print(
                        f"Processed "
                        f"{processed}/"
                        f"{len(samples)} "
                        f"samples"
                    )

    print()
    print(
        "Retrieval complete."
    )

    print(
        f"Processed samples: "
        f"{processed}"
    )

    print(
        f"Output: "
        f"{OUTPUT_PATH}"
    )

    if first_result:
        show_sanity_check(
            first_result
        )


if __name__ == "__main__":
    main()