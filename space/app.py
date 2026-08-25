import json
from pathlib import Path

import gradio as gr


DATA_PATH = (
    Path(__file__).parent
    / "demo_samples.json"
)


with open(
    DATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    DEMO_SAMPLES = json.load(f)


SAMPLE_INDEX = {
    item["sample_id"]: item
    for item in DEMO_SAMPLES
}

SAMPLE_IDS = list(
    SAMPLE_INDEX.keys()
)


def render_sample(sample_id):
    item = SAMPLE_INDEX[sample_id]

    sample = item["sample"]
    retrieval = item["retrieval"]
    reasoning = item["reasoning"]
    verification = item["verification"]

    question_text = (
        f"### Question\n\n"
        f"**Type:** `{sample['question_type']}`\n\n"
        f"{sample['question']}"
    )

    evidence_rows = []

    for evidence in retrieval["evidence"]:
        evidence_rows.append(
            [
                evidence["rank"],
                evidence["source"],
                round(
                    evidence[
                        "retrieval_score"
                    ],
                    3,
                ),
                evidence["content"],
            ]
        )

    prediction = reasoning.get(
        "prediction"
    )

    gold = sample.get(
        "gold_answer"
    )

    confidence = verification.get(
        "confidence",
        0.0,
    )

    status = verification.get(
        "verification_status"
    )

    abstained = verification.get(
        "abstained"
    )

    result_text = (
        "### Final Result\n\n"
        f"**Prediction:** `{prediction}`  \n"
        f"**Gold Answer:** `{gold}`  \n"
        f"**Confidence:** "
        f"`{confidence:.3f}`  \n"
        f"**Verification:** "
        f"`{status}`  \n"
        f"**Abstained:** "
        f"`{abstained}`"
    )

    reasoning_text = (
        reasoning.get(
            "reasoning_summary",
            "",
        )
    )

    return (
        question_text,
        evidence_rows,
        reasoning_text,
        result_text,
    )


initial_id = SAMPLE_IDS[0]

(
    initial_question,
    initial_evidence,
    initial_reasoning,
    initial_result,
) = render_sample(initial_id)


with gr.Blocks(
    title=(
        "Multimodal Theory-of-Mind "
        "Reasoning Agent"
    )
) as demo:

    gr.Markdown(
        """
# 🧠 Multimodal Theory-of-Mind Reasoning Agent

An evidence-grounded multimodal reasoning demo built on the
**MuMA-ToM** benchmark.

The pipeline combines:

**Video/Text Evidence → Semantic Retrieval → ToM Reasoning → Verification → Confidence & Abstention**

This public demo uses precomputed multimodal evidence and
reasoning outputs from a pilot evaluation subset.
"""
    )

    sample_dropdown = gr.Dropdown(
        choices=SAMPLE_IDS,
        value=initial_id,
        label="Select Benchmark Sample",
    )

    question_output = gr.Markdown(
        value=initial_question
    )

    gr.Markdown(
        "### Retrieved Multimodal Evidence"
    )

    evidence_output = gr.Dataframe(
        headers=[
            "Rank",
            "Modality",
            "Retrieval Score",
            "Evidence",
        ],
        datatype=[
            "number",
            "str",
            "number",
            "str",
        ],
        value=initial_evidence,
        interactive=False,
        wrap=True,
    )

    gr.Markdown(
        "### Theory-of-Mind Reasoning"
    )

    reasoning_output = gr.Textbox(
        value=initial_reasoning,
        lines=6,
        interactive=False,
        show_label=False,
    )

    result_output = gr.Markdown(
        value=initial_result
    )

    gr.Markdown(
        """
---

### Architecture

`MuMA-ToM video + text`
→ `BLIP visual evidence`
→ `MiniLM retrieval`
→ `Qwen2.5 ToM reasoning`
→ `semantic option mapping`
→ `verification`
→ `confidence / abstention`

The full project includes preprocessing, multimodal evidence
extraction, retrieval over 900 benchmark questions, structured
reasoning, verification, error analysis, FastAPI, tests, and CI.
"""
    )

    sample_dropdown.change(
        fn=render_sample,
        inputs=sample_dropdown,
        outputs=[
            question_output,
            evidence_output,
            reasoning_output,
            result_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()