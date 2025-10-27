from typing import Any, List, Optional

import numpy as np
from transformers import AutoTokenizer

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.datatypes import Token


class TensorRTBackend(BaseLLMWrapper):
    """
    Async inference backend based on TensorRT-LLM
    """

    def __init__(
        self,
        engine_dir: str,
        tokenizer_dir: str,
        topk: int = 5,
        temperature=0.0,
        top_p=1.0,
        **kwargs: Any
    ):
        super().__init__(temperature=temperature, top_p=top_p, **kwargs)
        try:
            from tensorrt_llm.runtime import ModelRunnerCpp
        except ImportError as exc:
            raise ImportError(
                "Please install tensorrt-llm to use TensorRTBackend: pip install tensorrt-llm"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.runner = ModelRunnerCpp.from_dir(engine_dir)
        self.topk = topk
        self.temperature = temperature
        self.top_p = top_p

    def _parse_generation(self, output_ids) -> str:
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        full = "\n".join(history or []) + "\n" + text
        ids = self.tokenizer.encode(full)
        output_ids = self.runner.generate(
            [ids],
            max_new_tokens=512,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self._parse_generation(output_ids)

    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        full = "\n".join(history or []) + "\n" + text
        ids = self.tokenizer.encode(full)
        *_, logits = self.runner.generate(
            [ids],
            max_new_tokens=1,
            temperature=0,
            output_logits=True,
        )
        logits = logits[0, -1, :]
        probs = np.softmax(logits)
        top_idx = np.argpartition(probs, -self.topk)[-self.topk :]
        top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
        return [
            Token(self.tokenizer.decode([idx]), float(probs[idx])) for idx in top_idx
        ]

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        full = "\n".join(history or []) + "\n" + text
        ids = self.tokenizer.encode(full)
        logprob_tokens = []
        for i in range(1, len(ids) + 1):
            trunc = ids[: i - 1] + ids[i:] if i < len(ids) else ids[:-1]
            *_, logits = self.runner.generate(
                [trunc],
                max_new_tokens=1,
                temperature=0,
                output_logits=True,
            )
            logits = logits[0, -1, :]
            probs = np.softmax(logits)
            true_id = ids[i - 1]
            logprob_tokens.append(
                Token(self.tokenizer.decode([true_id]), float(probs[true_id]))
            )
        return logprob_tokens
