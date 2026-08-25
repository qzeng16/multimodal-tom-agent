import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_PATH = ROOT / "data/processed/benchmark.jsonl"
RETRIEVAL_PATH = ROOT / "data/processed/retrieval.jsonl"
REASONING_PATH = ROOT / "data/processed/reasoning.jsonl"
VERIFICATION_PATH = ROOT / "data/processed/verification.jsonl"

OUTPUT_PATH = Path(__file__).parent / "demo_samples.json"


def load_index(path):
    results = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            sample_id = item.get("sample_id")

            if sample_id:
                results[sample_id] = item

    return results


def main():
    benchmark = load_index(BENCHMARK_PATH)
    retrieval = load_index(RETRIEVAL_PATH)
    reasoning = load_index(REASONING_PATH)
    verification = load_index(VERIFICATION_PATH)

    sample_ids = [
        sample_id
        for sample_id in reasoning
        if (
            sample_id in benchmark
            and sample_id in retrieval
            and sample_id in verification
        )
    ]

    demo_samples = []

    for sample_id in sample_ids[:10]:
        sample = benchmark[sample_id].copy()

        # The demo does not need local machine paths.
        sample.pop("video_path", None)

        demo_samples.append(
            {
                "sample_id": sample_id,
                "sample": sample,
                "retrieval": retrieval[sample_id],
                "reasoning": reasoning[sample_id],
                "verification": verification[sample_id],
            }
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            demo_samples,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Exported {len(demo_samples)} demo samples."
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()