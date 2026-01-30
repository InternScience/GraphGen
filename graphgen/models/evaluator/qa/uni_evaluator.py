# https://github.com/maszhongming/UniEval/tree/main
from typing import List, Optional

from graphgen.bases import BaseQAEvaluator, QAPair


class UniEvaluator(BaseQAEvaluator):
    """
    UniEvaluator for single QAPair evaluation across quality dimensions.

    Dimensions: naturalness, coherence, understandability

    Usage:
        evaluator = UniEvaluator()
        pair = QAPair(question="...", answer="...")
        scores = evaluator.evaluate(pair)
        # {"naturalness": 0.85, "coherence": 0.92, "understandability": 0.88}
    """

    DEFAULT_MODEL: str = "MingZhong/unieval-sum"
    DEFAULT_DIMS: List[str] = ["naturalness", "coherence", "understandability"]
    DEFAULT_MAX_LENGTH: int = 2560

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model name/path
            max_length: Tokenizer max sequence length
            device: 'cuda', 'cpu', or None for auto-detect
        """
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch

        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length or self.DEFAULT_MAX_LENGTH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Pre-compute Yes/No token IDs
        self._yes_id = self.tokenizer("Yes")["input_ids"][0]
        self._no_id = self.tokenizer("No")["input_ids"][0]

    @staticmethod
    def _build_input_text(dimension: str, question: str, answer: str) -> str:
        """Construct input text for specified dimension."""
        if dimension == "naturalness":
            return f"question: Is this a natural response? </s> response: {answer}"
        if dimension == "coherence":
            return f"question: Is this a coherent response? </s> response: {answer} </s> history: {question}"
        if dimension == "understandability":
            return (
                f"question: Is this an understandable response? </s> response: {answer}"
            )
        raise NotImplementedError(f"Unsupported dimension '{dimension}'")

    async def evaluate(
        self,
        pair: QAPair,
        dimensions: Optional[List[str]] = None,
    ) -> dict[str, float]:
        """Evaluate a single QAPair across specified dimensions."""
        dimensions = dimensions or self.DEFAULT_DIMS

        # Validate dimensions
        invalid = set(dimensions) - set(self.DEFAULT_DIMS)
        if invalid:
            raise ValueError(
                f"Invalid dimensions: {invalid}. Available: {self.DEFAULT_DIMS}"
            )

        results = {}
        no_token = self.torch.tensor([[self._no_id]], device=self.device)

        for dim in dimensions:
            # Tokenize input
            src = self.tokenizer(
                self._build_input_text(dim, pair.question, pair.answer),
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            src_tokens = src["input_ids"].to(self.device)
            src_mask = src["attention_mask"].to(self.device)

            # Score
            with self.torch.no_grad():
                logits = self.model(
                    input_ids=src_tokens,
                    attention_mask=src_mask,
                    labels=no_token,
                    use_cache=False,
                ).logits[
                    :, 0, :
                ]  # [1, vocab_size]

                probs = self.torch.softmax(logits, dim=-1)[0]
                score = probs[self._yes_id] / (probs[self._yes_id] + probs[self._no_id])

            results[dim] = score.item()

        return results
