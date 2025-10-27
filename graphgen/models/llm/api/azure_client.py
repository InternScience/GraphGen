import math
from typing import Any, Dict, List, Optional

import openai
from openai import APIConnectionError, APITimeoutError, AsyncAzureOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token
from graphgen.models.llm.limitter import RPM, TPM


def get_top_response_tokens(response: openai.ChatCompletion) -> List[Token]:
    token_logprobs = response.choices[0].logprobs.content
    tokens = []
    for token_prob in token_logprobs:
        prob = math.exp(token_prob.logprob)
        candidate_tokens = [
            Token(t.token, math.exp(t.logprob)) for t in token_prob.top_logprobs
        ]
        token = Token(token_prob.token, prob, top_candidates=candidate_tokens)
        tokens.append(token)
    return tokens


class AzureClient(BaseLLMWrapper):
    """
    直接复用 openai.AsyncAzureOpenAI，参数与 OpenAIClient 几乎一致，
    仅把 endpoint、api_version、deployment_name 换掉即可。
    """

    def __init__(
        self,
        *,
        model_name: str,  # 对应 azure 的 deployment_name
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2024-02-15-preview",
        json_mode: bool = False,
        seed: Optional[int] = None,
        topk_per_token: int = 5,
        request_limit: bool = False,
        rpm: Optional[RPM] = None,
        tpm: Optional[TPM] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.json_mode = json_mode
        self.seed = seed
        self.topk_per_token = topk_per_token
        self.request_limit = request_limit
        self.rpm = rpm or RPM()
        self.tpm = tpm or TPM()

        self.token_usage: List[Dict[str, int]] = []
        self.__post_init__()

    def __post_init__(self):
        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    # _pre_generate 与 OpenAIClient 完全一致，直接抄
    def _pre_generate(self, text: str, history: List[str]) -> Dict[str, Any]:
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed:
            kwargs["seed"] = self.seed
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": text})

        if history:
            assert len(history) % 2 == 0
            messages = history + messages

        kwargs["messages"] = messages
        return kwargs

    # ---------------- generate_answer ----------------
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError)
        ),
    )
    async def generate_answer(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> str:
        kwargs = self._pre_generate(text, history or [])

        prompt_tokens = sum(
            len(self.tokenizer.encode(m["content"])) for m in kwargs["messages"]
        )
        est = prompt_tokens + kwargs["max_tokens"]

        if self.request_limit:
            await self.rpm.wait(silent=True)
            await self.tpm.wait(est, silent=True)

        completion = await self.client.chat.completions.create(
            model=self.model_name, **kwargs
        )
        if hasattr(completion, "usage"):
            self.token_usage.append(
                {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
            )
        return self.filter_think_tags(completion.choices[0].message.content)

    # ---------------- generate_topk_per_token ----------------
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError)
        ),
    )
    async def generate_topk_per_token(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> List[Token]:
        kwargs = self._pre_generate(text, history or [])
        if self.topk_per_token > 0:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.topk_per_token
        kwargs["max_tokens"] = 5

        completion = await self.client.chat.completions.create(
            model=self.model_name, **kwargs
        )
        return get_top_response_tokens(completion)

    # ---------------- generate_inputs_prob ----------------
    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        raise NotImplementedError
