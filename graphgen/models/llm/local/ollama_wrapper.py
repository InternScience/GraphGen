from typing import Any, List, Optional

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.datatypes import Token


class OllamaBackend(BaseLLMWrapper):
    """
    Async inference backend based on Ollama local server
    """

    def __init__(
        self,
        model: str,  # e.g. "llama3.1:8b"
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        top_p: float = 1.0,
        topk: int = 5,
        **kwargs: Any
    ):
        try:
            import ollama
        except ImportError as exc:
            raise ImportError(
                "Please install ollama to use OllamaBackend: pip install ollama>=0.1.5"
            ) from exc
        super().__init__(temperature=temperature, top_p=top_p, **kwargs)
        self.client = ollama.AsyncClient(host=host)
        self.model = model
        self.topk = topk

    @staticmethod
    def _messages_to_str(prompt: str, history: Optional[List[str]] = None) -> str:
        if not history:
            return prompt
        return "\n".join(history) + "\n" + prompt

    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        text = self._messages_to_str(text, history)
        resp = await self.client.generate(
            model=self.model,
            prompt=text,
            options={
                "temperature": self.temperature or 0,
                "top_p": self.top_p if self.top_p < 1.0 else 1,
            },
            stream=False,
        )
        return resp["response"]

    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        raise NotImplementedError(
            "Ollama backend does not support per-token top-k yet."
        )

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        raise NotImplementedError(
            "Ollama backend does not support per-token input probabilities yet."
        )
