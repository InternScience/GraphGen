from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token


class HuggingFaceWrapper(BaseLLMWrapper):
    """
    Async inference backend based on HuggingFace Transformers
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        temperature=0.0,
        top_p=1.0,
        topk=5,
        **kwargs: Any
    ):
        super().__init__(temperature=temperature, top_p=top_p, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.temperature = temperature
        self.top_p = top_p
        self.topk = topk

    @staticmethod
    def _build_inputs(prompt: str, history: Optional[List[str]] = None):
        msgs = history or []
        msgs.append(prompt)
        full = "\n".join(msgs)
        return full

    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        full = self._build_inputs(text, history)
        inputs = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        max_new = 512
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=self.temperature if self.temperature > 0 else 0.0,
                top_p=self.top_p if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0, inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        full = self._build_inputs(text, history)
        inputs = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=0,
                return_dict_in_generate=True,
                output_scores=True,
            )
        scores = out.scores[0][0]  # vocab
        probs = torch.softmax(scores, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=self.topk)
        tokens = []
        for p, idx in zip(top_probs.cpu().numpy(), top_idx.cpu().numpy()):
            tokens.append(Token(self.tokenizer.decode([idx]), float(p)))
        return tokens

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        full = self._build_inputs(text, history)
        ids = self.tokenizer.encode(full)
        logprobs = []
        for i in range(1, len(ids) + 1):
            trunc = ids[: i - 1] + ids[i:] if i < len(ids) else ids[:-1]
            inputs = torch.tensor([trunc]).to(self.model.device)
            with torch.no_grad():
                logits = self.model(inputs).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            true_id = ids[i - 1]
            logprobs.append(
                Token(
                    self.tokenizer.decode([true_id]),
                    float(probs[true_id].cpu()),
                )
            )
        return logprobs
