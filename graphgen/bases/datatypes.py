from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict


@dataclass
class QAPair:
    """
    A pair of question and answer.
    """

    question: str
    answer: str
