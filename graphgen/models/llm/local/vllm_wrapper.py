from typing import Any, List, Optional

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token


class VLLMWrapper(BaseLLMWrapper):
    """
    Async inference backend based on vLLM (https://github.com/vllm-project/vllm)
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.0,
        top_p: float = 1.0,
        topk: int = 5,
        **kwargs: Any,
    ):
        super().__init__(temperature=temperature, top_p=top_p, **kwargs)

        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "VLLMWrapper requires vllm. Install it with:  pip install vllm"
            ) from exc

        self.SamplingParams = SamplingParams

        engine_args = AsyncEngineArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.temperature = temperature
        self.top_p = top_p
        self.topk = topk

    # ------------------------------------------------------------------
    # helper：把 history 拼成多轮格式（与 HFWrapper 保持一致）
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 1. 常规生成
    # ------------------------------------------------------------------
    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        full_prompt = self._build_inputs(text, history)

        sp = self.SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 1.0,
            top_p=self.top_p if self.temperature > 0 else 1.0,
            max_tokens=extra.get("max_new_tokens", 512),
        )

        # vLLM 的异步接口
        results = []
        async for req_output in self.engine.generate(
            full_prompt, sp, request_id="graphgen_req"
        ):
            results = req_output.outputs
        # 取最后一次返回
        return results[-1].text

    # ------------------------------------------------------------------
    # 2. 只生成 1 个新 token，返回 top-k 概率
    # ------------------------------------------------------------------
    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        full_prompt = self._build_inputs(text, history)

        # 强制 greedy（temperature=0）并返回 logprobs
        sp = self.SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=self.topk,  # vLLM 会给出 top-k 的 logprob
        )

        results = []
        async for req_output in self.engine.generate(
            full_prompt, sp, request_id="graphgen_topk"
        ):
            results = req_output.outputs
        top_logprobs = results[-1].logprobs[0]  # 第 1 个新生成 token 的 top-k

        tokens = []
        for _, logprob_obj in top_logprobs.items():
            tok_str = logprob_obj.decoded_token
            prob = float(logprob_obj.logprob.exp())
            tokens.append(Token(tok_str, prob))
        # 按概率从高到低排序
        tokens.sort(key=lambda x: -x.prob)
        return tokens

    # ------------------------------------------------------------------
    # 3. 逐 token 计算“被模型预测到”的概率（与 HFWrapper 语义对齐）
    # ------------------------------------------------------------------
    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        full_prompt = self._build_inputs(text, history)

        # vLLM 没有现成的“mask 一个 token 再算 prob”接口，
        # 我们采用最直观的方式：把 prompt 一次性送进去，打开
        # prompt_logprobs=True，让 vLLM 返回 *输入部分* 每个位置的
        # logprob，然后挑出对应 token 的概率即可。
        sp = self.SamplingParams(
            temperature=0,
            max_tokens=0,  # 不生成新 token
            prompt_logprobs=1,  # 只要 top-1 就够了
        )

        results = []
        async for req_output in self.engine.generate(
            full_prompt, sp, request_id="graphgen_prob"
        ):
            results = req_output.outputs

        # prompt_logprobs 是一个 list，长度 = prompt token 数，
        # 每个元素是 dict{token_id: logprob_obj} 或 None（首个位置为 None）
        prompt_logprobs = results[-1].prompt_logprobs

        tokens = []
        for _, logprob_dict in enumerate(prompt_logprobs):
            if logprob_dict is None:
                continue
            # 这里每个 dict 只有 1 个 kv，因为 top-1
            _, logprob_obj = next(iter(logprob_dict.items()))
            tok_str = logprob_obj.decoded_token
            prob = float(logprob_obj.logprob.exp())
            tokens.append(Token(tok_str, prob))
        return tokens
