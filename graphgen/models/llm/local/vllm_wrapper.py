import math
import uuid
from typing import Any, List, Optional

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token


class VLLMWrapper(BaseLLMWrapper):
    """
    Async inference backend based on vLLM.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.6,
        top_p: float = 1.0,
        top_k: int = 5,
        **kwargs: Any,
    ):
        temperature = float(temperature)
        top_p = float(top_p)
        top_k = int(top_k)

        super().__init__(temperature=temperature, top_p=top_p, top_k=top_k, **kwargs)
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "VLLMWrapper requires vllm. Install it with: uv pip install vllm"
            ) from exc

        self.SamplingParams = SamplingParams

        engine_args = AsyncEngineArgs(
            model=model,
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            trust_remote_code=kwargs.get("trust_remote_code", True),
            disable_log_stats=False,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @staticmethod
    def _build_inputs(prompt: str, history: Optional[List[str]] = None) -> str:
        msgs = history or []
        lines = []
        for m in msgs:
            if isinstance(m, dict):
                role = m.get("role", "")
                content = m.get("content", "")
                lines.append(f"{role}: {content}")
            else:
                lines.append(str(m))
        lines.append(prompt)
        return "\n".join(lines)

    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        full_prompt = self._build_inputs(text, history)
        request_id = f"graphgen_req_{uuid.uuid4()}"

        sp = self.SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 1.0,
            top_p=self.top_p if self.temperature > 0 else 1.0,
            max_tokens=extra.get("max_new_tokens", 2048),
        )

        result_generator = self.engine.generate(full_prompt, sp, request_id=request_id)

        final_output = None
        async for request_output in result_generator:
            final_output = request_output

        if not final_output or not final_output.outputs:
            return ""

        return final_output.outputs[0].text

    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        full_prompt = self._build_inputs(text, history)
        request_id = f"graphgen_topk_{uuid.uuid4()}"

        sp = self.SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=self.top_k,
            prompt_logprobs=1,
        )

        result_generator = self.engine.generate(full_prompt, sp, request_id=request_id)

        final_output = None
        async for request_output in result_generator:
            final_output = request_output

        if (
            not final_output
            or not final_output.outputs
            or not final_output.outputs[0].logprobs
        ):
            return []

        top_logprobs = final_output.outputs[0].logprobs[0]

        candidate_tokens = []
        for _, logprob_obj in top_logprobs.items():
            tok_str = (
                logprob_obj.decoded_token.strip() if logprob_obj.decoded_token else ""
            )
            prob = float(math.exp(logprob_obj.logprob))
            candidate_tokens.append(Token(tok_str, prob))

        candidate_tokens.sort(key=lambda x: -x.prob)

        if candidate_tokens:
            main_token = Token(
                text=candidate_tokens[0].text,
                prob=candidate_tokens[0].prob,
                top_candidates=candidate_tokens,
            )
            return [main_token]
        return []

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        raise NotImplementedError(
            "VLLMWrapper does not support per-token logprobs yet."
        )
