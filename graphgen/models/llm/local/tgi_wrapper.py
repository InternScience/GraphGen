import math
from typing import Any, List, Optional

from huggingface_hub import InferenceClient

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.datatypes import Token


class TGIWrapper(BaseLLMWrapper):
    """
    Async inference backend based on TGI (Text-Generation-Inference)
    """

    def __init__(
        self,
        model_url: str,  # e.g. "http://localhost:8080"
        temperature: float = 0.0,
        top_p: float = 1.0,
        topk: int = 5,
        **kwargs: Any
    ):
        super().__init__(temperature=temperature, top_p=top_p, **kwargs)
        self.client = InferenceClient(model=model_url, token=False)
        self.topk = topk
        self.model_url = model_url

    @staticmethod
    def _messages_to_str(prompt: str, history: Optional[List[str]] = None) -> str:
        if not history:
            return prompt
        return "\n".join(history) + "\n" + prompt

    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        text = self._messages_to_str(text, history)
        out = await self.client.text_generation(
            text,
            max_new_tokens=extra.get("max_new_tokens", 512),
            temperature=self.temperature or None,
            top_p=self.top_p if self.top_p < 1.0 else None,
            stop_sequences=extra.get("stop", None),
            details=False,
        )
        return out

    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        text = self._messages_to_str(text, history)
        out = await self.client.text_generation(
            text,
            max_new_tokens=1,
            temperature=0,
            details=True,
            decoder_input_details=True,
        )
        # TGI 返回的 tokens[0].logprob.topk 字段
        topk = out.details.tokens[0].logprob.topk
        return [Token(t.token, math.exp(t.logprob)) for t in topk[: self.topk]]

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        """
        TGI does not provide a direct interface for "conditional probability of each input token",
        here we approximate it with "input prefix + next token".
        To implement it strictly, you can use /generate_stream and truncate it bit by bit.
        """
        text = self._messages_to_str(text, history)
        ids = self.client.tokenizer.encode(text)
        tokens: List[Token] = []
        for i in range(1, len(ids) + 1):
            prefix_ids = ids[:i]
            prefix = self.client.tokenizer.decode(prefix_ids)
            out = await self.client.text_generation(
                prefix,
                max_new_tokens=1,
                temperature=0,
                details=True,
                decoder_input_details=True,
            )
            t = out.details.tokens[0]
            tokens.append(Token(t.token, math.exp(t.logprob)))
        return tokens
