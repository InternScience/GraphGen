from graphgen.bases import BaseEvaluator, QAPair
from graphgen.models.tokenizer import Tokenizer


class LengthEvaluator(BaseEvaluator):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def evaluate(self, pair: QAPair) -> float:
        """
        Evaluate the length of the qa pair.
        """
        content = pair.question + pair.answer
        tokens = self.tokenizer.encode(content)
        return len(tokens)
