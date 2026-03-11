import os
from typing import Any, Dict, List, Optional

from ray import serve
from starlette.requests import Request

from graphgen.bases.datatypes import Token
from graphgen.models.tokenizer import Tokenizer


@serve.deployment
class LLMDeployment:
    def __init__(self, backend: str, config: Dict[str, Any]):
        self.backend = backend

        # Initialize tokenizer if needed
        tokenizer_model = os.environ.get("TOKENIZER_MODEL", "cl100k_base")
        if "tokenizer" not in config:
            tokenizer = Tokenizer(model_name=tokenizer_model)
            config["tokenizer"] = tokenizer

        if backend == "vllm":
            from graphgen.models.llm.local.vllm_wrapper import VLLMWrapper

            self.llm_instance = VLLMWrapper(**config)
        elif backend == "huggingface":
            from graphgen.models.llm.local.hf_wrapper import HuggingFaceWrapper

            self.llm_instance = HuggingFaceWrapper(**config)
        elif backend == "sglang":
            from graphgen.models.llm.local.sglang_wrapper import SGLangWrapper

            self.llm_instance = SGLangWrapper(**config)
        else:
            raise NotImplementedError(
                f"Backend {backend} is not implemented for Ray Serve yet."
            )

    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        return await self.llm_instance.generate_answer(text, history, **extra)

    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        return await self.llm_instance.generate_topk_per_token(text, history, **extra)

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        return await self.llm_instance.generate_inputs_prob(text, history, **extra)

    async def __call__(self, request: Request) -> Dict:
        try:
            data = await request.json()
            text = data.get("text")
            history = data.get("history")
            method = data.get("method", "generate_answer")
            kwargs = data.get("kwargs", {})

            if method == "generate_answer":
                result = await self.generate_answer(text, history, **kwargs)
            elif method == "generate_topk_per_token":
                result = await self.generate_topk_per_token(text, history, **kwargs)
            elif method == "generate_inputs_prob":
                result = await self.generate_inputs_prob(text, history, **kwargs)
            else:
                return {"error": f"Method {method} not supported"}

            return {"result": result}
        except Exception as e:
            return {"error": str(e)}


def app_builder(args: Dict[str, str]) -> Any:
    """
    Builder function for 'serve run'.
    Usage: serve run graphgen.models.llm.local.ray_serve_deployment:app_builder backend=vllm model=...
    """
    # args comes from the command line key=value pairs
    backend = args.pop("backend", "vllm")
    # remaining args are treated as config
    return LLMDeployment.bind(backend=backend, config=args)
