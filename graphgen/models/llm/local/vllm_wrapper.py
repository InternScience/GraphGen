import math
import uuid
from typing import Any, List, Optional
import asyncio

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
        timeout: float = 300,
        **kwargs: Any,
    ):
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
        self.timeout = float(timeout)
        self.tokenizer = self.engine.engine.tokenizer.tokenizer

    def _build_inputs(self, prompt: str, history: Optional[List[dict]] = None) -> Any:
        messages = history or []
        messages.append({"role": "user", "content": prompt})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        full_prompt = self._build_inputs(text, history)
        request_id = f"graphgen_req_{uuid.uuid4()}"

        sp = self.SamplingParams(
            temperature=self.temperature if self.temperature >= 0 else 1.0,
            top_p=self.top_p if self.top_p >= 0 else 1.0,
            max_tokens=extra.get("max_new_tokens", 2048),
            repetition_penalty=extra.get("repetition_penalty", 1.05),
        )

        try:
            result_generator = self.engine.generate(full_prompt, sp, request_id=request_id)
            final_output = None
            async with asyncio.timeout(self.timeout):
                async for request_output in result_generator:
                    if request_output.finished:
                        final_output = request_output
                        break

            if not final_output or not final_output.outputs:
                return ""

            result_text = final_output.outputs[0].text
            return result_text

        except (Exception, asyncio.CancelledError, asyncio.TimeoutError):
            await self.engine.abort(request_id)
            raise

    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        full_prompt = self._build_inputs(text, history)
        request_id = f"graphgen_topk_{uuid.uuid4()}"

        sp = self.SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=self.top_k,
        )

        try:
            result_generator = self.engine.generate(full_prompt, sp, request_id=request_id)
            final_output = None
            async with asyncio.timeout(self.timeout):
                async for request_output in result_generator:
                    if request_output.finished:
                        final_output = request_output
                        break

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

        except (Exception, asyncio.CancelledError, asyncio.TimeoutError):
            await self.engine.abort(request_id)
            raise

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        raise NotImplementedError(
            "VLLMWrapper does not support per-token logprobs yet."
        )
