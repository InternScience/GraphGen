import asyncio
import math
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token
from graphgen.models.llm.limitter import RPM, TPM


class OllamaClient(BaseLLMWrapper):
    """
    Requires a local or remote Ollama server to be running (default port 11434).
    The /api/chat endpoint in Ollama 0.1.24+ supports stream=False
    and raw=true to return logprobs, but the top_logprobs field is not yet implemented by the official API.
    """

    def __init__(
        self,
        *,
        model_name: str = "llama3.1",
        base_url: str = "http://localhost:11434",
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
        self.base_url = base_url.rstrip("/")
        self.json_mode = json_mode
        self.seed = seed
        self.topk_per_token = topk_per_token
        self.request_limit = request_limit
        self.rpm = rpm or RPM()
        self.tpm = tpm or TPM()

        self.token_usage: List[Dict[str, int]] = []
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_payload(self, text: str, history: List[str]) -> Dict[str, Any]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # chatml format: alternating user and assistant messages
        if history and isinstance(history[0], dict):
            messages.extend(history)

        messages.append({"role": "user", "content": text})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
        }
        if self.seed is not None:
            payload["options"]["seed"] = self.seed
        if self.json_mode:
            payload["format"] = "json"
        if self.topk_per_token > 0:
            payload["options"]["logprobs"] = True
        return payload

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def generate_answer(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> str:
        payload = self._build_payload(text, history or [])
        prompt_tokens = sum(
            len(self.tokenizer.encode(m["content"])) for m in payload["messages"]
        )
        est = prompt_tokens + self.max_tokens

        if self.request_limit:
            await self.rpm.wait(silent=True)
            await self.tpm.wait(est, silent=True)

        async with self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        # {"message":{"content":"..."}, "prompt_eval_count":xx, "eval_count":yy}
        content = data["message"]["content"]
        self.token_usage.append(
            {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0)
                + data.get("eval_count", 0),
            }
        )
        return self.filter_think_tags(content)

    async def generate_topk_per_token(
        self,
        text: str,
        history: Optional[List[str]] = None,
        **extra: Any,
    ) -> List[Token]:
        payload = self._build_payload(text, history or [])
        payload["options"]["num_predict"] = 5
        async with self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        tokens = []
        for item in data.get("message", {}).get("logprobs", {}).get("content", []):
            tokens.append(Token(item["token"], math.exp(item["logprob"])))
        return tokens

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        raise NotImplementedError("Ollama API does not support per-token logprobs yet.")
