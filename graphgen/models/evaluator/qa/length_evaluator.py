import os

from graphgen.bases import BaseQAEvaluator, QAPair
from graphgen.models.tokenizer import Tokenizer


class LengthEvaluator(BaseQAEvaluator):
    def __init__(self, tokenizer_name: str = None):
        tokenizer_model = tokenizer_name or os.environ.get(
            "TOKENIZER_MODEL", "cl100k_base"
        )
        self.tokenizer: Tokenizer = Tokenizer(tokenizer_model)

    async def evaluate(self, pair: QAPair) -> dict[str, float]:
        """
        Evaluate the length of the qa pair.
        """
        content = pair.question + pair.answer
        return {"length": self.tokenizer.count_tokens(content)}
