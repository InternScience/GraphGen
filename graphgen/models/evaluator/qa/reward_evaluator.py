from typing import Optional
from graphgen.bases import BaseEvaluator, QAPair


class RewardEvaluator(BaseEvaluator):
    """
    Reward Model Evaluator for single QAPair evaluation.
    """

    def __init__(
        self,
        reward_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
        max_length: int = 2560,
        device: Optional[str] = None,
    ):
        """
        Initialize the reward evaluator.
        
        Args:
            reward_name: Model name or path on HuggingFace Hub
            max_length: Maximum token length for the model
            device: Device to run the model on. If None, auto-detect CUDA/CPU.
        """
        self.reward_name = reward_name
        self.max_length = max_length

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch

        # Set device (auto-detect if not specified)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(reward_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(reward_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load reward model '{reward_name}': {e}") from e

    def evaluate(self, pair: QAPair) -> float:
        """
        Evaluate a single question-answer pair using the reward model.
        
        Args:
            pair: QAPair containing question and answer strings
            
        Returns:
            Score as a float
        """
        # Tokenize
        inputs = self.tokenizer(
            pair.question,
            pair.answer,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get score
        with self.torch.no_grad():
            score = self.model(**inputs).logits[0].item()

        return score
