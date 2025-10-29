import math
from typing import Any, Dict, List, Optional

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token


# TODO: implement SGLangWrapper methods
class SGLangWrapper(BaseLLMWrapper):
    """
    Async inference backend based on SGLang offline engine.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        topk: int = 5,
        **kwargs: Any,
    ):
        super().__init__(temperature=temperature, top_p=top_p, **kwargs)
        try:
            import sglang as sgl
            from sglang.utils import async_stream_and_merge, stream_and_merge
        except ImportError as exc:
            raise ImportError(
                "SGLangWrapper requires sglang. Install it with: "
                "uv pip install sglang --prerelease=allow"
            ) from exc

        self.model_path: str = model
        self.temperature = temperature
        self.top_p = top_p
        self.topk = topk

        # Initialise the offline engine
        self.engine = sgl.Engine(model_path=self.model_path)

        # Keep helpers for streaming
        self.async_stream_and_merge = async_stream_and_merge
        self.stream_and_merge = stream_and_merge

    @staticmethod
    def _build_sampling_params(
        temperature: float,
        top_p: float,
        max_tokens: int,
        topk: int,
        logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Build SGLang-compatible sampling-params dict."""
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
        }
        if logprobs and topk > 0:
            params["logprobs"] = topk
        return params

    def _prep_prompt(self, text: str, history: Optional[List[str]] = None) -> str:
        """Convert raw text (+ optional history) into a single prompt string."""
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if history:
            assert len(history) % 2 == 0, "History must have even length (u/a turns)."
            parts.extend(history)
        parts.append(text)
        return "\n".join(parts)

    def _tokens_from_output(self, output: Dict[str, Any]) -> List[Token]:
        """
        Convert SGLang logprobs output into List[Token].
        SGLang returns:
            output['logprobs'][t] -> {
                "token": <str>,
                "logprob": <float>,
                "top_k_tokens": [...],
                "top_k_logprobs": [...],
            }
        """
        tokens: List[Token] = []
        if "logprobs" not in output or not output["logprobs"]:
            return tokens

        for entry in output["logprobs"]:
            token_str = entry["token"]
            logprob = entry["logprob"]
            prob = math.exp(logprob)

            top_candidates = []
            if self.topk > 0 and "top_k_tokens" in entry:
                for tok, lp in zip(entry["top_k_tokens"], entry["top_k_logprobs"]):
                    top_candidates.append(Token(tok, math.exp(lp)))

            tokens.append(Token(token_str, prob, top_candidates=top_candidates))
        return tokens

    async def generate_answer(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> str:
        prompt = self._prep_prompt(text, history)
        sampling_params = self._build_sampling_params(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            topk=0,  # no logprobs needed for simple generation
        )

        outputs = self.engine.generate([prompt], sampling_params)
        return self.filter_think_tags(outputs[0]["text"])

    async def generate_topk_per_token(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> List[Token]:
        prompt = self._prep_prompt(text, history)
        sampling_params = self._build_sampling_params(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=5,  # keep short for token-level analysis
            topk=self.topk,
            logprobs=True,
        )

        outputs = self.engine.generate([prompt], sampling_params)
        return self._tokens_from_output(outputs[0])

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        """
        Return per-token probabilities for the *input* sequence.
        SGLang offline engine does not expose this directly; we emulate by
        generating 0 new tokens with logprobs enabled (returns prompt logprobs).
        """
        prompt = self._prep_prompt(text, history)
        sampling_params = self._build_sampling_params(
            temperature=0.0,  # deterministic
            top_p=1.0,
            max_tokens=0,  # generate nothing
            topk=self.topk,
            logprobs=True,
        )

        outputs = self.engine.generate([prompt], sampling_params)
        # SGLang returns prompt logprobs under key 'prompt_logprobs' when max_new_tokens=0
        prompt_logprobs = outputs[0].get("prompt_logprobs", [])
        tokens: List[Token] = []
        for entry in prompt_logprobs:
            tokens.append(
                Token(
                    text=entry["token"],
                    prob=math.exp(entry["logprob"]),
                    top_candidates=[],  # SGLang does not give top-k for prompt tokens
                )
            )
        return tokens

    def shutdown(self) -> None:
        """Gracefully shutdown the SGLang engine."""
        if hasattr(self, "engine"):
            self.engine.shutdown()

    def restart(self) -> None:
        """Restart the SGLang engine."""
        self.shutdown()
        self.engine = self.engine.__class__(model_path=self.model_path)
