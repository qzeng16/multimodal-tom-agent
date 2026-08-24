import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set

import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
)

from schemas import (
    EpisodeEvidence,
    EpisodeInput,
    EvidenceItem,
)


MODEL_NAME = "Salesforce/blip-image-captioning-base"

BENCHMARK_PATH = Path(
    "data/processed/benchmark.jsonl"
)

FRAME_MANIFEST_PATH = Path(
    "data/processed/frame_manifest.json"
)

OUTPUT_PATH = Path(
    "data/processed/evidence.jsonl"
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


def load_unique_episodes() -> Dict[str, EpisodeInput]:
    episodes = {}

    with open(
        BENCHMARK_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            sample = EpisodeInput(
                **json.loads(line)
            )

            if sample.episode_id not in episodes:
                episodes[
                    sample.episode_id
                ] = sample

    return episodes


def load_frame_manifest() -> Dict:
    with open(
        FRAME_MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def split_context(
    text: str,
) -> List[str]:
    """
    Split textual episode context into
    sentence-level evidence units.
    """

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def build_text_evidence(
    episode_id: str,
    context_text: str,
) -> List[EvidenceItem]:
    sentences = split_context(
        context_text
    )

    evidence = []

    for index, sentence in enumerate(
        sentences
    ):
        evidence.append(
            EvidenceItem(
                evidence_id=(
                    f"{episode_id}"
                    f"_text_{index:02d}"
                ),
                source="text",
                content=sentence,
            )
        )

    return evidence


def load_blip(
    device: str,
):
    print(
        f"Loading BLIP model "
        f"on {device}..."
    )

    processor = (
        BlipProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        BlipForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            use_safetensors=True,
        )
    )

    model.to(device)
    model.eval()

    print("BLIP loaded.")

    return processor, model


def caption_frame(
    frame_path: str,
    processor,
    model,
    device: str,
) -> str:
    image = Image.open(
        frame_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True,
    )

    return caption.strip()


def build_visual_evidence(
    episode_id: str,
    frame_records: List[Dict],
    processor,
    model,
    device: str,
) -> List[EvidenceItem]:

    evidence = []

    for index, frame in enumerate(
        frame_records
    ):
        frame_path = frame["path"]

        if not Path(
            frame_path
        ).exists():
            print(
                f"Warning: missing frame "
                f"{frame_path}"
            )
            continue

        try:
            caption = caption_frame(
                frame_path=frame_path,
                processor=processor,
                model=model,
                device=device,
            )

        except Exception as exc:
            print(
                f"Warning: failed to caption "
                f"{frame_path}: {exc}"
            )
            continue

        evidence.append(
            EvidenceItem(
                evidence_id=(
                    f"{episode_id}"
                    f"_visual_{index:02d}"
                ),
                source="visual",
                content=caption,
                timestamp=frame.get(
                    "timestamp"
                ),
                frame_path=frame_path,
            )
        )

    return evidence


def load_completed_episode_ids(
    path: Path,
) -> Set[str]:
    completed = set()

    if not path.exists():
        return completed

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            completed.add(
                str(
                    data["episode_id"]
                )
            )

    return completed


def show_sanity_check(
    result: EpisodeEvidence,
) -> None:

    visual = [
        item
        for item in result.evidence
        if item.source == "visual"
    ]

    text = [
        item
        for item in result.evidence
        if item.source == "text"
    ]

    print()
    print("Sanity check:")

    print(
        f"Episode: "
        f"{result.episode_id}"
    )

    print(
        f"Visual evidence: "
        f"{len(visual)}"
    )

    print(
        f"Text evidence: "
        f"{len(text)}"
    )

    if visual:
        print()
        print(
            "First visual evidence:"
        )
        print(
            visual[0].content
        )

    if text:
        print()
        print(
            "First text evidence:"
        )
        print(
            text[0].content
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-episodes",
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
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    device = select_device(
        args.device
    )

    print(
        f"Using device: {device}"
    )

    episodes = load_unique_episodes()

    manifest = load_frame_manifest()

    print(
        f"Unique episodes: "
        f"{len(episodes)}"
    )

    completed = set()

    if args.resume:
        completed = (
            load_completed_episode_ids(
                OUTPUT_PATH
            )
        )

        print(
            f"Already completed: "
            f"{len(completed)}"
        )

    remaining = [
        episode
        for episode_id, episode
        in episodes.items()
        if episode_id not in completed
    ]

    if args.max_episodes is not None:
        remaining = remaining[
            :args.max_episodes
        ]

    if not remaining:
        print(
            "No episodes to process."
        )
        return

    processor, model = load_blip(
        device
    )

    mode = (
        "a"
        if args.resume
        else "w"
    )

    first_result = None

    with open(
        OUTPUT_PATH,
        mode,
        encoding="utf-8",
    ) as output_file:

        total = len(remaining)

        for index, episode in enumerate(
            remaining,
            start=1,
        ):
            episode_id = (
                episode.episode_id
            )

            frame_records = (
                manifest.get(
                    episode_id,
                    [],
                )
            )

            visual_evidence = (
                build_visual_evidence(
                    episode_id=episode_id,
                    frame_records=frame_records,
                    processor=processor,
                    model=model,
                    device=device,
                )
            )

            text_evidence = (
                build_text_evidence(
                    episode_id=episode_id,
                    context_text=(
                        episode.context_text
                    ),
                )
            )

            result = EpisodeEvidence(
                episode_id=episode_id,
                evidence=(
                    visual_evidence
                    + text_evidence
                ),
            )

            output_file.write(
                result.model_dump_json()
                + "\n"
            )

            output_file.flush()

            if first_result is None:
                first_result = result

            print(
                f"Processed "
                f"{index}/{total}: "
                f"episode {episode_id} "
                f"("
                f"{len(visual_evidence)} visual, "
                f"{len(text_evidence)} text"
                f")"
            )

    print()
    print(
        "Evidence extraction complete."
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    if first_result:
        show_sanity_check(
            first_result
        )


if __name__ == "__main__":
    main()