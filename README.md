# Multimodal Theory-of-Mind Reasoning Agent

**Live Demo:** https://huggingface.co/spaces/qzeng16/multimodal-tom-agent

**GitHub:** https://github.com/qzeng16/multimodal-tom-agent

A multimodal reasoning pipeline for inferring beliefs and social goals from video and textual evidence.

The project uses the MuMA-ToM benchmark and combines video frame understanding, semantic evidence retrieval, Theory-of-Mind reasoning, verification, confidence estimation, and selective abstention.

## Overview

The system is designed to answer Theory-of-Mind questions such as:

- What does a person believe?
- What social goal explains a person's behavior?
- What does one person believe another person's goal is?

Instead of sending an entire video directly to a language model, the system builds an explicit multimodal evidence pipeline.

~~~text
Video
  ↓
Uniform frame sampling
  ↓
BLIP image captioning
  ↓
Visual evidence
                \
                 → Unified evidence pool
                /
Text context
  ↓
Sentence evidence
                ↓
SentenceTransformer retrieval
                ↓
Modality-balanced Top-K evidence
                ↓
Qwen2.5 Theory-of-Mind reasoning
                ↓
Semantic option mapping
                ↓
Verification
                ↓
Confidence + selective abstention
~~~

## Dataset

The project uses the MuMA-ToM benchmark from the Johns Hopkins Social and Cognitive AI Lab.

The preprocessing pipeline converts the benchmark into:

- 900 question samples
- 225 video episodes
- uniformly sampled video frames
- normalized question records
- structured multimodal evidence

The benchmark contains three Theory-of-Mind question types:

- `belief`
- `social_goal`
- `belief_of_goal`

Raw videos and generated intermediate artifacts are excluded from Git.

## Project Pipeline

The project is organized into the following stages:

1. Dataset preprocessing
2. Video frame sampling
3. Multimodal evidence extraction
4. Evidence retrieval
5. Theory-of-Mind reasoning
6. Verification and confidence estimation
7. Evaluation and error analysis
8. CLI, API, tests, and documentation
9. Hugging Face Spaces deployment

## 1. Dataset Preprocessing

The original MuMA-ToM structure is normalized into one record per question.

Each sample contains:

- `sample_id`
- `episode_id`
- `question_id`
- question type
- multiple-choice question
- gold answer
- textual episode context
- video path

For example:

~~~text
episode_id  = 4005
question_id = 1
sample_id   = 4005_1
~~~

An episode may contain multiple Theory-of-Mind questions, so video preprocessing is performed once per episode and reused across questions.

## 2. Video Frame Sampling

Each benchmark video is uniformly sampled into eight representative frames.

This provides a lightweight visual representation without requiring full video-language inference over every frame.

Example structure:

~~~text
data/processed/frames/4005/
├── frame_00.jpg
├── frame_01.jpg
├── frame_02.jpg
├── frame_03.jpg
├── frame_04.jpg
├── frame_05.jpg
├── frame_06.jpg
└── frame_07.jpg
~~~

A frame manifest stores the original video position and timestamp for every sampled frame.

## 3. Multimodal Evidence Extraction

Visual evidence is extracted using:

`Salesforce/blip-image-captioning-base`

Each sampled frame is converted into a textual visual description.

Example:

~~~text
a room with a couch, a table and a television
~~~

Textual episode descriptions are split into sentence-level evidence units.

Both visual and textual evidence are stored using the same schema.

Example:

~~~json
{
  "evidence_id": "4005_visual_00",
  "source": "visual",
  "content": "a room with a couch, a table and a television",
  "timestamp": 0.0
}
~~~

Text evidence uses the same representation:

~~~json
{
  "evidence_id": "4005_text_01",
  "source": "text",
  "content": "Jessica asked Michael where the remote control might be."
}
~~~

This creates a unified evidence store that can later be searched independently of modality.

## 4. Evidence Retrieval

Evidence retrieval uses:

`sentence-transformers/all-MiniLM-L6-v2`

The system embeds:

- the Theory-of-Mind question
- all visual evidence
- all textual evidence

Cosine similarity is used to rank evidence by semantic relevance.

The system uses modality-balanced Top-K retrieval.

Without modality balancing, textual evidence can dominate the ranking because textual descriptions often have stronger lexical overlap with the question.

The balancing rule ensures that visual evidence remains represented in the retrieved set when visual evidence is available.

Example retrieval result:

~~~text
Top evidence:

[1] TEXT score=0.684
Jessica then moved to the living room and asked,
"Do you have any idea where the remote control might be?"

[2] TEXT score=0.506
Jessica walked into the kitchen while Michael stayed silent.

[3] TEXT score=0.500
Jessica and Michael completed their tasks without further communication.

...

[6] VISUAL score=0.261
a room with a couch, a table and a television
~~~

The retrieval pipeline was run across all 900 benchmark question samples.

## 5. Theory-of-Mind Reasoning

Theory-of-Mind reasoning uses:

`Qwen/Qwen2.5-1.5B-Instruct`

The reasoning stage receives:

- question type
- question and options
- retrieved multimodal evidence
- a Theory-of-Mind-specific reasoning instruction

Different reasoning instructions are used for each question type.

### Belief

The model must distinguish:

~~~text
objective world state
        ≠
what the person believes
~~~

It is instructed to consider:

- what information the person observed
- what information the person was told
- temporal order
- information availability

### Social Goal

The model reasons about the interpersonal objective that best explains observed actions, dialogue, cooperation, and outcomes.

### Belief of Goal

This is treated as a nested mental-state problem:

~~~text
person B's actual goal
        ≠
person A's belief about B's goal
~~~

The model must reason about one person's representation of another person's intention rather than the true intention itself.

## Semantic Option Mapping

During initial testing, the language model could derive the correct answer content but occasionally output the wrong option letter.

For example, the model reasoned that:

~~~text
Michael believed there was a remote control inside the cabinet.
~~~

but originally returned:

~~~text
A
~~~

even though the remote-control answer corresponded to option `B`.

To separate reasoning from option formatting, the final system asks the language model to output answer content rather than an A/B/C label.

Example:

~~~text
ANSWER:
Michael believed there was a remote control inside the cabinet.
~~~

The answer content is then mapped to the benchmark options using semantic similarity with MiniLM.

This creates the pipeline:

~~~text
Theory-of-Mind reasoning
        ↓
answer content
        ↓
semantic similarity
        ↓
A / B / C
~~~

For sample `4005_1`, the corrected pipeline produced:

~~~text
Model answer content:
Michael believed that there was a remote control
inside the cabinet in the living room.

Mapped prediction: B
Mapping score: 0.866
Gold answer: B
Parse OK: True
~~~

## 6. Verification, Confidence, and Abstention

The reasoning model does not automatically trust every prediction.

A separate verification stage evaluates whether the prediction is sufficiently supported by retrieved evidence.

The verifier combines several signals:

- semantic option-mapping score
- evidence support for the predicted option
- evidence support for competing options
- support margin
- retrieval quality
- whether the reasoning model cited evidence

The system computes:

~~~text
selected_support
alternative_support

support_margin =
selected_support - alternative_support
~~~

A confidence score combines these signals.

The verifier then assigns one of three states:

- `supported`
- `uncertain`
- `conflicted`

Low-confidence or conflicting outputs can be rejected through selective abstention.

Example:

~~~text
Sample: 4005_1
Prediction: B
Gold answer: B
Confidence: 0.836
Mapping score: 0.866
Selected support: 0.657
Alternative support: 0.624
Support margin: 0.033
Status: supported
Abstained: False
~~~

## 7. Evaluation

The evaluation framework measures:

- reasoning accuracy
- parse success rate
- coverage
- selective accuracy
- abstention rate
- per-question-type performance
- confidence threshold ablation
- modality usage
- error categories

The current reasoning experiment is a small pilot subset used to validate the complete pipeline.

It should not be interpreted as full MuMA-ToM benchmark performance.

The evaluation code can automatically scale to additional reasoning samples without modification.

## Selective Prediction

The system evaluates the trade-off between:

~~~text
answering more questions
        vs
answering fewer questions more reliably
~~~

In the current pilot experiment, a higher confidence threshold reduced coverage but improved selective accuracy.

Example:

~~~text
threshold=0.50
coverage=0.600
selective_accuracy=0.500

threshold=0.80
coverage=0.400
selective_accuracy=0.750
~~~

This demonstrates the intended confidence-coverage trade-off:

~~~text
lower threshold
→ more answers
→ lower reliability

higher threshold
→ fewer answers
→ higher reliability
~~~

Because the pilot contains only a small number of reasoning samples, these values are intended to validate system behavior rather than establish benchmark performance.

## Error Analysis

Evaluation cases are divided into four categories:

- `correct_accepted`
- `correct_but_abstained`
- `error_caught_by_verifier`
- `overconfident_wrong`

This separates reasoning errors from verification errors.

For example:

`error_caught_by_verifier`

means:

~~~text
reasoning prediction is wrong
        +
verifier abstains
~~~

which is desirable behavior.

`overconfident_wrong`

means:

~~~text
reasoning prediction is wrong
        +
verifier still accepts it
~~~

which identifies cases requiring future improvement.

The current pilot included both correctly caught errors and overconfident failures, providing concrete examples for error analysis.

## Multimodal Failure Analysis

The modular architecture allows failures to be separated into different stages:

~~~text
video sampling error
        ↓
visual caption error
        ↓
retrieval error
        ↓
Theory-of-Mind reasoning error
        ↓
option-mapping error
        ↓
verification error
~~~

This is more informative than treating the entire system as a single black-box vision-language model.

## Project Structure

~~~text
multimodal-tom-agent/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── evaluate.py
│   ├── extract_evidence.py
│   ├── main.py
│   ├── preprocess.py
│   ├── reason.py
│   ├── retrieve.py
│   ├── schemas.py
│   └── verify.py
│
├── space/
│   ├── app.py
│   ├── demo_samples.json
│   ├── export_demo.py
│   ├── requirements.txt
│   └── README.md
│
├── tests/
│   └── test_pipeline.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt
├── .gitignore
└── README.md
~~~

## Installation

Install dependencies:

~~~bash
python3 -m pip install -r requirements.txt
~~~

## Preprocessing

Download the MuMA-ToM benchmark, normalize the question records, and sample video frames:

~~~bash
python3 src/preprocess.py
~~~

This generates the normalized benchmark and sampled frame representations.

## Extract Multimodal Evidence

Test the evidence extraction pipeline on one episode:

~~~bash
python3 src/extract_evidence.py --max-episodes 1
~~~

Resume processing the remaining episodes:

~~~bash
python3 src/extract_evidence.py --resume
~~~

The extractor generates:

- visual evidence from BLIP captions
- sentence-level textual evidence
- unified evidence records

## Retrieve Evidence

Run multimodal evidence retrieval:

~~~bash
python3 src/retrieve.py
~~~

The default configuration retrieves modality-balanced Top-K evidence for all benchmark questions.

To test one sample:

~~~bash
python3 src/retrieve.py --max-samples 1
~~~

## Run Theory-of-Mind Reasoning

Run one sample:

~~~bash
python3 src/reason.py --max-samples 1
~~~

Resume previously interrupted reasoning:

~~~bash
python3 src/reason.py --resume
~~~

Because local language-model inference can be computationally expensive, reasoning can also be run on a smaller evaluation subset.

Example:

~~~bash
python3 src/reason.py --resume --max-samples 50
~~~

## Verify Predictions

Run verification over all currently available reasoning outputs:

~~~bash
python3 src/verify.py
~~~

The verifier computes confidence, evidence support, support margin, and abstention decisions.

## Evaluate

Run the evaluation pipeline:

~~~bash
python3 src/evaluate.py
~~~

Outputs include:

~~~text
data/processed/evaluation_summary.json
data/processed/error_analysis.jsonl
~~~

These generated files are excluded from Git.

## Inspect an Individual Sample

The CLI can display the complete output pipeline for a benchmark sample.

Example:

~~~bash
python3 src/main.py --sample-id 4005_1
~~~

The command shows:

- question type
- question and options
- gold answer
- retrieved evidence
- model prediction
- reasoning
- confidence
- verification status
- abstention decision

## FastAPI

A lightweight FastAPI interface is included.

Start the server:

~~~bash
uvicorn src.api:app --reload
~~~

The server runs at:

~~~text
http://127.0.0.1:8000
~~~

Interactive Swagger documentation is available at:

~~~text
http://127.0.0.1:8000/docs
~~~

### Health Check

~~~text
GET /health
~~~

Example response:

~~~json
{
  "status": "ok"
}
~~~

### Inspect Sample

~~~text
GET /samples/{sample_id}
~~~

Example:

~~~text
GET /samples/4005_1
~~~

The endpoint returns the available:

- benchmark record
- retrieval result
- reasoning output
- verification result

for that sample.

## Hugging Face Spaces Demo

A lightweight interactive demo is deployed on Hugging Face Spaces:

https://huggingface.co/spaces/qzeng16/multimodal-tom-agent

The public Space uses precomputed pilot outputs rather than rerunning BLIP and Qwen inference for every interaction.

This keeps the demo responsive while still exposing the full reasoning pipeline:

~~~text
Benchmark Sample
      ↓
Question
      ↓
Retrieved Multimodal Evidence
      ↓
Theory-of-Mind Reasoning
      ↓
Prediction
      ↓
Confidence
      ↓
Verification / Abstention
~~~

The demo allows users to select benchmark samples and inspect:

- question type
- multiple-choice question
- ranked text and visual evidence
- retrieval scores
- Theory-of-Mind reasoning
- final prediction
- confidence score
- verification status
- abstention decision

The Space is implemented with Gradio.

## Tests

Run the lightweight test suite:

~~~bash
pytest -q
~~~

The current tests cover deterministic utilities such as:

- question choice parsing
- answer-label parsing
- confidence clamping
- sigmoid calculation

The repository also includes a GitHub Actions workflow that automatically runs the test suite on pushes and pull requests to `main`.

## Continuous Integration

GitHub Actions is configured in:

~~~text
.github/workflows/ci.yml
~~~

Each push to `main` automatically:

~~~text
Checkout repository
        ↓
Set up Python
        ↓
Install dependencies
        ↓
Run pytest
        ↓
Pass / Fail
~~~

The CI workflow intentionally avoids full BLIP or Qwen inference so that automated tests remain lightweight and reproducible.

## Generated Artifacts

Large or generated data files are intentionally excluded from Git.

Examples include:

~~~text
data/raw/
data/processed/frames/
data/processed/frame_manifest.json
data/processed/benchmark.jsonl
data/processed/evidence.jsonl
data/processed/retrieval.jsonl
data/processed/reasoning.jsonl
data/processed/verification.jsonl
data/processed/evaluation_summary.json
data/processed/error_analysis.jsonl
~~~

This keeps the repository focused on reproducible source code rather than generated model outputs.

## Design Goals

The project emphasizes:

- multimodal evidence grounding
- explicit intermediate representations
- Theory-of-Mind-specific reasoning
- interpretable retrieval
- separation of reasoning and option mapping
- confidence-aware prediction
- selective abstention
- structured error analysis
- modular AI engineering
- reproducible evaluation
- deployable interactive demonstration

## Key Engineering Decisions

### Episode-Level Visual Processing

Visual evidence is generated once per video episode and reused for multiple questions.

This avoids repeatedly processing the same video frames.

### Lightweight Visual Representation

The system uses sampled frames and BLIP captions instead of full-video inference.

This reduces compute requirements and creates interpretable visual evidence.

### Modality-Balanced Retrieval

Top-K retrieval preserves visual evidence when available rather than allowing text evidence to dominate every result.

### Separate Reasoning and Option Mapping

The language model reasons about answer content.

A deterministic semantic-mapping stage converts that content into A/B/C.

This prevents formatting mistakes from being treated as reasoning failures.

### Confidence-Aware Output

Predictions are not automatically trusted.

A separate verifier estimates support and can abstain when evidence is weak or contradictory.

### Lightweight Deployment

The Hugging Face Space presents precomputed outputs from the complete pipeline instead of downloading and executing the full BLIP and Qwen stack for every user interaction.

This makes the public demo fast, inexpensive, and reliable while preserving the interpretability of the full architecture.

## Limitations

The current system has several limitations.

First, visual reasoning is based on eight uniformly sampled video frames. Important short-duration actions may occur between sampled frames.

Second, BLIP converts images into short captions, which can discard details relevant to social reasoning.

Third, the system currently represents video primarily as independent frame evidence rather than explicitly modeling temporal actions.

Fourth, `Qwen2.5-1.5B-Instruct` is intentionally lightweight enough for local inference but can struggle with difficult nested Theory-of-Mind questions.

Fifth, semantic similarity is used for both retrieval and option mapping. Stronger learned multimodal retrieval or cross-encoder reranking could improve performance.

Finally, the currently reported reasoning experiment is a small pilot subset rather than a full 900-question reasoning evaluation.

## Future Work

Potential extensions include:

- stronger vision-language models
- temporal action extraction
- event-level video representations
- cross-modal reranking
- learned evidence fusion
- larger reasoning models
- stronger confidence calibration
- full-benchmark reasoning evaluation
- retrieval ablation without modality balancing
- visual-only and text-only ablations
- learned verification models
- live Hugging Face inference with GPU acceleration

## Summary

This project implements an end-to-end multimodal Theory-of-Mind reasoning system:

~~~text
Video + Text
     ↓
Evidence Extraction
     ↓
Semantic Retrieval
     ↓
Theory-of-Mind Reasoning
     ↓
Semantic Option Mapping
     ↓
Verification
     ↓
Confidence / Abstention
     ↓
Evaluation + Error Analysis
     ↓
API + CI + Interactive Demo
~~~

The project focuses not only on producing predictions, but also on making the intermediate evidence, reasoning process, confidence signals, and failure modes explicit.