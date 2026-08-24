import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

SRC = ROOT / "src"

sys.path.insert(
    0,
    str(SRC),
)


from preprocess import (
    parse_answer_letter,
    parse_choices,
)

from verify import (
    clamp,
    sigmoid,
)


def test_parse_choices():
    question = """
Question text

A) apple
B) banana
C) orange
"""

    choices = parse_choices(
        question
    )

    assert choices == [
        "A) apple",
        "B) banana",
        "C) orange",
    ]


def test_parse_answer_letter():

    assert (
        parse_answer_letter(
            "B) banana"
        )
        == "B"
    )


def test_clamp():

    assert clamp(-1.0) == 0.0
    assert clamp(2.0) == 1.0
    assert clamp(0.5) == 0.5


def test_sigmoid():

    value = sigmoid(
        0.0
    )

    assert abs(
        value - 0.5
    ) < 1e-6