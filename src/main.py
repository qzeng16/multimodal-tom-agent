import json
from pathlib import Path

from schemas import EpisodeInput


BENCHMARK_PATH = Path(
    "data/processed/benchmark.jsonl"
)


def load_first_sample() -> EpisodeInput:
    with open(
        BENCHMARK_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        line = f.readline()

    return EpisodeInput(
        **json.loads(line)
    )


def main():
    sample = load_first_sample()

    print(
        "Benchmark loaded successfully."
    )

    print()
    print(
        f"Sample ID: "
        f"{sample.sample_id}"
    )

    print(
        f"Episode ID: "
        f"{sample.episode_id}"
    )

    print(
        f"Question type: "
        f"{sample.question_type}"
    )

    print()
    print(
        f"Question:\n"
        f"{sample.question}"
    )

    print()
    print(
        f"Gold answer: "
        f"{sample.gold_answer}"
    )


if __name__ == "__main__":
    main()