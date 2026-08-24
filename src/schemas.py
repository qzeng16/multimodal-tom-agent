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

    evidence: List[
        RetrievedEvidenceItem
    ]


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