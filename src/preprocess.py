import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import cv2

from huggingface_hub import snapshot_download

from schemas import EpisodeInput


REPO_ID = "SCAI-JHU/MUMA-TOM-BENCHMARK"

RAW_DIR = Path("data/raw/muma_tom")
PROCESSED_DIR = Path("data/processed")

BENCHMARK_PATH = PROCESSED_DIR / "benchmark.jsonl"
FRAME_ROOT = PROCESSED_DIR / "frames"
FRAME_MANIFEST_PATH = PROCESSED_DIR / "frame_manifest.json"


def download_dataset() -> None:
    """
    Download only the benchmark files we need:
    - questions.json
    - texts.json
    - RGB videos

    We intentionally skip the training set and other large assets.
    """

    print("Downloading MuMA-ToM benchmark...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[
            "questions.json",
            "texts.json",
            "videos/*.mp4",
        ],
        local_dir=RAW_DIR,
    )

    print("Dataset download complete.")


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_choices(question: str) -> List[str]:
    """
    Extract multiple-choice options from a question.

    Example:
        A) option one
        B) option two
        C) option three
    """

    matches = re.findall(
        r"^\s*([A-Z])\)\s*(.+)$",
        question,
        flags=re.MULTILINE,
    )

    return [
        f"{letter}) {text.strip()}"
        for letter, text in matches
    ]


def parse_answer_letter(answer: str) -> str:
    match = re.match(
        r"\s*([A-Z])\)",
        answer,
    )

    if not match:
        raise ValueError(
            f"Could not parse answer: {answer}"
        )

    return match.group(1)


def normalize_benchmark() -> int:
    """
    Convert the original MuMA-ToM JSON structure into
    one normalized JSONL record per question.
    """

    questions_path = RAW_DIR / "questions.json"
    texts_path = RAW_DIR / "texts.json"

    questions_data = load_json(questions_path)
    texts_data = load_json(texts_path)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with open(
        BENCHMARK_PATH,
        "w",
        encoding="utf-8",
    ) as out_file:

        for episode_id, episode_data in questions_data.items():

            context_text = texts_data.get(
                episode_id,
                episode_data.get(
                    "description",
                    "",
                ),
            )

            video_path = (
                RAW_DIR
                / "videos"
                / f"video_{episode_id}.mp4"
            )

            questions = episode_data["questions"]
            answers = episode_data["answers"]
            labels = episode_data["labels"]

            for question_id, question in questions.items():

                answer_text = answers[question_id]
                answer_letter = parse_answer_letter(
                    answer_text
                )

                sample = EpisodeInput(
                    sample_id=(
                        f"{episode_id}_{question_id}"
                    ),
                    episode_id=episode_id,
                    question_id=question_id,
                    video_path=str(video_path),
                    context_text=context_text,
                    question=question,
                    question_type=labels[
                        question_id
                    ],
                    choices=parse_choices(
                        question
                    ),
                    gold_answer=answer_letter,
                    gold_answer_text=answer_text,
                )

                out_file.write(
                    sample.model_dump_json()
                    + "\n"
                )

                count += 1

    return count


def sample_video_frames(
    video_path: Path,
    output_dir: Path,
    num_frames: int,
) -> List[Dict]:
    """
    Uniformly sample frames across a video.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        print(
            f"Warning: could not open "
            f"{video_path}"
        )
        return []

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if total_frames <= 0:
        cap.release()
        return []

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if num_frames == 1:
        frame_indices = [
            total_frames // 2
        ]
    else:
        frame_indices = [
            int(
                i
                * (total_frames - 1)
                / (num_frames - 1)
            )
            for i in range(
                num_frames
            )
        ]

    frame_records = []

    for i, frame_index in enumerate(
        frame_indices
    ):
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = cap.read()

        if not success:
            continue

        frame_name = (
            f"frame_{i:02d}.jpg"
        )

        frame_path = (
            output_dir
            / frame_name
        )

        cv2.imwrite(
            str(frame_path),
            frame,
        )

        timestamp = (
            frame_index / fps
            if fps > 0
            else None
        )

        frame_records.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "path": str(
                    frame_path
                ),
            }
        )

    cap.release()

    return frame_records


def preprocess_videos(
    num_frames: int,
) -> Dict:
    """
    Sample frames from every benchmark video.
    """

    video_dir = (
        RAW_DIR / "videos"
    )

    videos = sorted(
        video_dir.glob(
            "video_*.mp4"
        )
    )

    manifest = {}

    print(
        f"Processing {len(videos)} videos..."
    )

    for index, video_path in enumerate(
        videos,
        start=1,
    ):
        episode_id = (
            video_path.stem
            .replace(
                "video_",
                "",
            )
        )

        output_dir = (
            FRAME_ROOT
            / episode_id
        )

        frames = sample_video_frames(
            video_path=video_path,
            output_dir=output_dir,
            num_frames=num_frames,
        )

        manifest[episode_id] = frames

        if (
            index % 25 == 0
            or index == len(videos)
        ):
            print(
                f"Processed "
                f"{index}/"
                f"{len(videos)} videos"
            )

    with open(
        FRAME_MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    return manifest


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--skip-download",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.skip_download:
        download_dataset()

    print(
        "\nNormalizing benchmark..."
    )

    sample_count = (
        normalize_benchmark()
    )

    print(
        f"Created "
        f"{sample_count} "
        f"question samples."
    )

    print(
        "\nSampling video frames..."
    )

    manifest = preprocess_videos(
        num_frames=(
            args.frames_per_video
        )
    )

    total_frames = sum(
        len(frames)
        for frames in manifest.values()
    )

    print()
    print(
        "Preprocessing complete."
    )

    print(
        f"Question samples: "
        f"{sample_count}"
    )

    print(
        f"Videos: "
        f"{len(manifest)}"
    )

    print(
        f"Sampled frames: "
        f"{total_frames}"
    )

    print(
        f"Benchmark: "
        f"{BENCHMARK_PATH}"
    )

    print(
        f"Frame manifest: "
        f"{FRAME_MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()