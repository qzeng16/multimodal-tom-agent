import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from schemas import (
    EpisodeInput,
    ReasoningResult,
    RetrievalResult,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAPPER_MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

BENCHMARK_PATH = Path(
    "data/processed/benchmark.jsonl"
)

RETRIEVAL_PATH = Path(
    "data/processed/retrieval.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/reasoning.jsonl"
)


TOM_INSTRUCTIONS = {
    "belief": (
        "Infer what the relevant person believed at the "
        "specified time. Distinguish the person's belief "
        "from objective reality. Pay attention to what "
        "information the person observed or was told and "
        "to the temporal order of events."
    ),

    "social_goal": (
        "Infer the person's social goal from their actions, "
        "dialogue, cooperation, and interaction outcome. "
        "Focus on what interpersonal objective best explains "
        "the observed behavior."
    ),

    "belief_of_goal": (
        "Infer one person's belief about another person's "
        "goal. This is a nested mental-state problem. "
        "Do not confuse the other person's actual goal with "
        "what the target person believes that goal to be."
    ),
}


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

            samples[
                sample.sample_id
            ] = sample

    return samples


def load_retrieval() -> List[RetrievalResult]:
    results = []

    with open(
        RETRIEVAL_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            results.append(
                RetrievalResult(
                    **json.loads(line)
                )
            )

    return results


def load_completed_sample_ids(
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
                data["sample_id"]
            )

    return completed


def load_reasoning_model(
    model_name: str,
    device: str,
):
    print(
        f"Loading reasoning model "
        f"on {device}..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name
        )
    )

    if device in (
        "mps",
        "cuda",
    ):
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_name,
            dtype=dtype,
            use_safetensors=True,
        )
    )

    model.to(device)
    model.eval()

    print(
        "Reasoning model loaded."
    )

    return tokenizer, model


def load_mapper():
    print(
        "Loading option-mapping model..."
    )

    model = SentenceTransformer(
        MAPPER_MODEL_NAME,
        device="cpu",
    )

    print(
        "Option-mapping model loaded."
    )

    return model


def format_evidence(
    retrieval: RetrievalResult,
) -> str:
    lines = []

    for item in retrieval.evidence:

        timestamp = ""

        if item.timestamp is not None:
            timestamp = (
                f" timestamp="
                f"{item.timestamp:.2f}s"
            )

        lines.append(
            f"[{item.evidence_id}] "
            f"source={item.source} "
            f"retrieval_score="
            f"{item.retrieval_score:.3f}"
            f"{timestamp}\n"
            f"{item.content}"
        )

    return "\n\n".join(lines)


def build_prompt(
    sample: EpisodeInput,
    retrieval: RetrievalResult,
) -> str:

    instruction = TOM_INSTRUCTIONS[
        sample.question_type
    ]

    evidence_text = format_evidence(
        retrieval
    )

    return f"""
You are solving a Theory-of-Mind reasoning task.

Question type:
{sample.question_type}

Reasoning rule:
{instruction}

Use only the supplied evidence.
Do not use the gold answer or outside knowledge.

Question:
{sample.question}

Retrieved evidence:
{evidence_text}

Reason about the mental state first.

IMPORTANT:
Do NOT return the option letter A, B, or C as your answer.
Instead, write the actual answer content in your own words.
For example, if the correct option is about a remote control,
write "remote control inside the cabinet" rather than "B".

Return exactly these three fields:

ANSWER: <short answer content>
EVIDENCE: <comma-separated evidence IDs>
REASONING: <brief explanation>

Your reasoning must distinguish observed facts from mental
states such as beliefs and goals.
""".strip()


def generate_response(
    tokenizer,
    model,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful multimodal "
                "Theory-of-Mind reasoning system."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    input_length = (
        inputs["input_ids"].shape[1]
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=(
                tokenizer.eos_token_id
            ),
        )

    generated_tokens = output[
        0,
        input_length:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


def parse_answer_text(
    text: str,
) -> Optional[str]:

    match = re.search(
        r"ANSWER\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    answer = (
        match.group(1)
        .splitlines()[0]
        .strip()
    )

    return answer or None


def parse_evidence_ids(
    text: str,
) -> List[str]:

    match = re.search(
        r"EVIDENCE\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return []

    line = (
        match.group(1)
        .splitlines()[0]
    )

    candidates = [
        item.strip()
        for item in line.split(",")
    ]

    return [
        item
        for item in candidates
        if item
    ]


def parse_reasoning(
    text: str,
) -> str:

    match = re.search(
        r"REASONING\s*:\s*(.+)",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if not match:
        return text.strip()

    return match.group(1).strip()


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
            f"Invalid choice format: "
            f"{choice}"
        )

    label = (
        match.group(1)
        .upper()
    )

    text = (
        match.group(2)
        .strip()
    )

    return label, text


def normalize_text(
    text: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def map_answer_to_option(
    answer_text: str,
    choices: List[str],
    mapper,
) -> Tuple[
    Optional[str],
    float,
]:

    # If the model accidentally returns only
    # a single option letter, handle it.
    letter_match = re.fullmatch(
        r"\s*([ABC])\s*",
        answer_text,
        flags=re.IGNORECASE,
    )

    if letter_match:
        return (
            letter_match.group(1).upper(),
            1.0,
        )

    # Remove an accidental leading label,
    # but trust the answer content rather
    # than that label.
    cleaned_answer = re.sub(
        r"^\s*[ABC][\)\.\:\-]\s*",
        "",
        answer_text,
        flags=re.IGNORECASE,
    ).strip()

    parsed_choices = [
        clean_choice(choice)
        for choice in choices
    ]

    labels = [
        item[0]
        for item in parsed_choices
    ]

    choice_texts = [
        item[1]
        for item in parsed_choices
    ]

    normalized_answer = normalize_text(
        cleaned_answer
    )

    # Fast lexical mapping when the answer
    # content clearly matches one option.
    lexical_matches = []

    for index, choice_text in enumerate(
        choice_texts
    ):
        normalized_choice = normalize_text(
            choice_text
        )

        if (
            normalized_answer
            in normalized_choice
            or normalized_choice
            in normalized_answer
        ):
            lexical_matches.append(index)

    if len(lexical_matches) == 1:
        index = lexical_matches[0]

        return (
            labels[index],
            1.0,
        )

    # Otherwise use semantic similarity.
    embeddings = mapper.encode(
        [
            cleaned_answer,
            *choice_texts,
        ],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    answer_embedding = (
        embeddings[0]
    )

    choice_embeddings = (
        embeddings[1:]
    )

    scores = torch.matmul(
        choice_embeddings,
        answer_embedding,
    )

    best_index = int(
        torch.argmax(scores).item()
    )

    best_score = float(
        scores[best_index].item()
    )

    return (
        labels[best_index],
        best_score,
    )


def build_reasoning_result(
    sample: EpisodeInput,
    retrieval: RetrievalResult,
    raw_output: str,
    answer_text: Optional[str],
    prediction: Optional[str],
) -> ReasoningResult:

    evidence_ids = parse_evidence_ids(
        raw_output
    )

    reasoning = parse_reasoning(
        raw_output
    )

    valid_evidence_ids = {
        item.evidence_id
        for item in retrieval.evidence
    }

    evidence_ids = [
        evidence_id
        for evidence_id in evidence_ids
        if evidence_id
        in valid_evidence_ids
    ]

    parse_ok = (
        answer_text is not None
        and prediction is not None
    )

    return ReasoningResult(
        sample_id=sample.sample_id,
        episode_id=sample.episode_id,
        question_type=(
            sample.question_type
        ),
        prediction=prediction,
        used_evidence_ids=(
            evidence_ids
        ),
        reasoning_summary=reasoning,
        parse_ok=parse_ok,
        raw_output=raw_output,
    )


def show_sanity_check(
    result: ReasoningResult,
    sample: EpisodeInput,
    answer_text: Optional[str],
    mapping_score: float,
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
    print(
        f"Model answer content: "
        f"{answer_text}"
    )

    print(
        f"Mapped prediction: "
        f"{result.prediction}"
    )

    print(
        f"Mapping score: "
        f"{mapping_score:.3f}"
    )

    print(
        f"Gold answer: "
        f"{sample.gold_answer}"
    )

    print(
        f"Parse OK: "
        f"{result.parse_ok}"
    )

    print()
    print(
        "Used evidence:"
    )

    for evidence_id in (
        result.used_evidence_ids
    ):
        print(
            f"  {evidence_id}"
        )

    print()
    print(
        "Reasoning:"
    )

    print(
        result.reasoning_summary
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
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
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
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

    benchmark = load_benchmark()

    retrieval_results = (
        load_retrieval()
    )

    completed = set()

    if args.resume:
        completed = (
            load_completed_sample_ids(
                OUTPUT_PATH
            )
        )

        print(
            f"Already completed: "
            f"{len(completed)}"
        )

    remaining = [
        retrieval
        for retrieval
        in retrieval_results
        if retrieval.sample_id
        not in completed
    ]

    if args.max_samples is not None:
        remaining = remaining[
            :args.max_samples
        ]

    print(
        f"Samples to process: "
        f"{len(remaining)}"
    )

    if not remaining:
        print(
            "No samples to process."
        )
        return

    tokenizer, model = (
        load_reasoning_model(
            model_name=args.model,
            device=device,
        )
    )

    mapper = load_mapper()

    mode = (
        "a"
        if args.resume
        else "w"
    )

    first_result = None
    first_sample = None
    first_answer_text = None
    first_mapping_score = 0.0

    with open(
        OUTPUT_PATH,
        mode,
        encoding="utf-8",
    ) as output_file:

        total = len(remaining)

        for index, retrieval in enumerate(
            remaining,
            start=1,
        ):

            sample = benchmark.get(
                retrieval.sample_id
            )

            if sample is None:
                print(
                    f"Warning: missing benchmark "
                    f"sample "
                    f"{retrieval.sample_id}"
                )
                continue

            prompt = build_prompt(
                sample=sample,
                retrieval=retrieval,
            )

            raw_output = generate_response(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                device=device,
                max_new_tokens=(
                    args.max_new_tokens
                ),
            )

            answer_text = (
                parse_answer_text(
                    raw_output
                )
            )

            prediction = None
            mapping_score = 0.0

            if answer_text is not None:
                (
                    prediction,
                    mapping_score,
                ) = map_answer_to_option(
                    answer_text=answer_text,
                    choices=sample.choices,
                    mapper=mapper,
                )

            result = build_reasoning_result(
                sample=sample,
                retrieval=retrieval,
                raw_output=raw_output,
                answer_text=answer_text,
                prediction=prediction,
            )

            output_file.write(
                result.model_dump_json()
                + "\n"
            )

            output_file.flush()

            if first_result is None:
                first_result = result
                first_sample = sample
                first_answer_text = (
                    answer_text
                )
                first_mapping_score = (
                    mapping_score
                )

            print(
                f"Processed "
                f"{index}/{total}: "
                f"{sample.sample_id} "
                f"prediction="
                f"{result.prediction} "
                f"mapping="
                f"{mapping_score:.3f} "
                f"parse_ok="
                f"{result.parse_ok}"
            )

    print()
    print(
        "Reasoning complete."
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
            answer_text=(
                first_answer_text
            ),
            mapping_score=(
                first_mapping_score
            ),
        )


if __name__ == "__main__":
    main()