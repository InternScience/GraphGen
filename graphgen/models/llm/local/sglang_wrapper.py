import math
from typing import Any, List, Optional

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.datatypes import Token


class SGLangBackend(BaseLLMWrapper):
    """
    Async inference backend based on SGLang
    """

    def __init__(
        self,
        model_path: str,
        tp_size: int = 1,
        max_context_len: int = 4096,
        server_url: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        topk: int = 5,
        **kwargs: Any
    ):
        super().__init__(temperature=temperature, top_p=top_p, **kwargs)
        try:
            import sglang as sgl
            from sglang.backend.runtime_endpoint import RuntimeEndpoint
        except ImportError as exc:
            raise ImportError(
                "Please install sglang to use SGLangBackend: pip install sglang[all]>=0.4.4"
            ) from exc
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.topk = topk

        # if server_url is given, connect to remote server; else launch local runtime
        if server_url:
            self.runtime = RuntimeEndpoint(server_url)
        else:
            sgl.set_default_backend(
                sgl.Runtime(
                    model_path, tp_size=tp_size, max_context_len=max_context_len
                )
            )
            self.runtime = sgl.get_default_backend()

        self.tokenizer = self.runtime.get_tokenizer()

    @staticmethod
    def _messages_to_str(prompt: str, history: Optional[List[str]] = None) -> str:
        if not history:
            return prompt
        return "\n".join(history) + "\n" + prompt

    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        text = self._messages_to_str(text, history)

        output = await self.runtime.generate(
            text,
            max_new_tokens=512,
            temperature=self.temperature if self.temperature > 0 else 0,
            top_p=self.top_p,
            stop=None,
        )
        return output

    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        text = self._messages_to_str(text, history)

        output_obj = await self.runtime.generate(
            text,
            max_new_tokens=1,
            temperature=0,
            return_logprob=True,
            top_logprobs=self.topk,
            logprob_start_len=0,
        )

        topk_list = output_obj["meta_info"]["top_logprobs"][
            0
        ]  # List[ (token_str, logprob), ... ]
        return [Token(tok, math.exp(logprob)) for tok, logprob in topk_list]

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        text = self._messages_to_str(text, history)
        ids = self.tokenizer.encode(text)
        if not ids:
            return []

        logprob_tokens: List[Token] = []

        for i in range(1, len(ids) + 1):
            trunc_ids = ids[: i - 1] + ids[i:] if i < len(ids) else ids[:-1]
            trunc_text = self.tokenizer.decode(trunc_ids)

            output_obj = await self.runtime.generate(
                trunc_text,
                max_new_tokens=1,
                temperature=0,
                return_logprob=True,
                top_logprobs=1,
                logprob_start_len=len(trunc_ids) - 1,
            )
            top1 = output_obj["meta_info"]["top_logprobs"][0][0]
            logprob_tokens.append(Token(top1[0], math.exp(top1[1])))

        return logprob_tokens
