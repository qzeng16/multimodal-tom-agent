from typing import List, Literal, Optional

from pydantic import BaseModel, Field


QuestionType = Literal[
    "belief",
    "social_goal",
    "belief_of_goal",
]


EvidenceSource = Literal[
    "visual",
    "text",
    "action",
]


VerificationStatus = Literal[
    "supported",
    "uncertain",
    "conflicted",
]


class EpisodeInput(BaseModel):
    sample_id: str
    episode_id: str
    question_id: str

    video_path: str
    context_text: str

    question: str
    question_type: QuestionType

    choices: List[str]

    gold_answer: str
    gold_answer_text: str


class EvidenceItem(BaseModel):
    evidence_id: str
    source: EvidenceSource
    content: str

    timestamp: Optional[float] = None
    frame_path: Optional[str] = None
    score: Optional[float] = None


class EpisodeEvidence(BaseModel):
    episode_id: str
    evidence: List[EvidenceItem]


class RetrievedEvidenceItem(BaseModel):
    evidence_id: str
    source: EvidenceSource
    content: str

    timestamp: Optional[float] = None
    frame_path: Optional[str] = None

    retrieval_score: float
    rank: int


class RetrievalResult(BaseModel):
    sample_id: str
    episode_id: str

    question_type: QuestionType
    question: str

    top_k: int
    modality_balanced: bool

    evidence: List[RetrievedEvidenceItem]


class ReasoningResult(BaseModel):
    sample_id: str
    episode_id: str

    question_type: QuestionType

    prediction: Optional[str] = None

    used_evidence_ids: List[str] = Field(
        default_factory=list
    )

    reasoning_summary: str
    parse_ok: bool
    raw_output: str

    # Added for later runs.
    # Existing reasoning.jsonl files remain compatible.
    answer_text: Optional[str] = None
    mapping_score: Optional[float] = None


class VerificationResult(BaseModel):
    sample_id: str
    episode_id: str

    prediction: Optional[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    mapping_score: float
    selected_support: float
    alternative_support: float
    support_margin: float
    evidence_score: float

    verification_status: VerificationStatus

    abstained: bool

    used_evidence_ids: List[str] = Field(
        default_factory=list
    )


class ReasoningOutput(BaseModel):
    sample_id: str

    prediction: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence: List[EvidenceItem]

    reasoning_summary: str

    abstained: bool = False