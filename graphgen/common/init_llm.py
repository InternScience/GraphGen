import os
from typing import Any, Dict, Optional

import ray

from graphgen.bases import BaseLLMWrapper
from graphgen.common.init_storage import get_actor_handle
from graphgen.models import Tokenizer


class LLMServiceActor:
    """
    A Ray actor class to wrap LLM wrapper instances for distributed usage.
    """

    def __init__(self, backend: str, config: Dict[str, Any]):
        self.backend = backend
        tokenizer_model = os.environ.get("TOKENIZER_MODEL", "cl100k_base")
        tokenizer = Tokenizer(model_name=tokenizer_model)
        config["tokenizer"] = tokenizer

        if backend == "http_api":
            from graphgen.models.llm.api.http_client import HTTPClient

            self.llm_instance = HTTPClient(**config)
        elif backend in ("openai_api", "azure_openai_api"):
            from graphgen.models.llm.api.openai_client import OpenAIClient

            # pass in concrete backend to the OpenAIClient so that internally we can distinguish
            # between OpenAI and Azure OpenAI
            self.llm_instance = OpenAIClient(**config, backend=backend)
        elif backend == "ollama_api":
            from graphgen.models.llm.api.ollama_client import OllamaClient

            self.llm_instance = OllamaClient(**config)
        elif backend == "huggingface":
            from graphgen.models.llm.local.hf_wrapper import HuggingFaceWrapper

            self.llm_instance = HuggingFaceWrapper(**config)
        elif backend == "sglang":
            from graphgen.models.llm.local.sglang_wrapper import SGLangWrapper

            self.llm_instance = SGLangWrapper(**config)

        elif backend == "vllm":
            from graphgen.models.llm.local.vllm_wrapper import VLLMWrapper

            self.llm_instance = VLLMWrapper(**config)
        else:
            raise NotImplementedError(f"Backend {backend} is not implemented yet.")

    async def generate_answer(
        self, text: str, history: Optional[list[str]] = None, **extra: Any
    ) -> str:
        return await self.llm_instance.generate_answer(text, history, **extra)

    async def generate_topk_per_token(
        self, text: str, history: Optional[list[str]] = None, **extra: Any
    ) -> list:
        return await self.llm_instance.generate_topk_per_token(text, history, **extra)

    async def generate_inputs_prob(
        self, text: str, history: Optional[list[str]] = None, **extra: Any
    ) -> list:
        return await self.llm_instance.generate_inputs_prob(text, history, **extra)

    def ready(self) -> bool:
        """A simple method to check if the actor is ready."""
        return True


class LLMServiceProxy(BaseLLMWrapper):
    """
    A proxy class to interact with the LLMServiceActor for distributed LLM operations.
    """

    def __init__(self, actor_name: str):
        super().__init__()
        self.actor_handle = get_actor_handle(actor_name)
        self._create_local_tokenizer()

    async def generate_answer(
        self, text: str, history: Optional[list[str]] = None, **extra: Any
    ) -> str:
        object_ref = self.actor_handle.generate_answer.remote(text, history, **extra)
        return await object_ref

    async def generate_topk_per_token(
        self, text: str, history: Optional[list[str]] = None, **extra: Any
    ) -> list:
        object_ref = self.actor_handle.generate_topk_per_token.remote(
            text, history, **extra
        )
        return await object_ref

    async def generate_inputs_prob(
        self, text: str, history: Optional[list[str]] = None, **extra: Any
    ) -> list:
        object_ref = self.actor_handle.generate_inputs_prob.remote(
            text, history, **extra
        )
        return await object_ref

    def _create_local_tokenizer(self):
        tokenizer_model = os.environ.get("TOKENIZER_MODEL", "cl100k_base")
        self.tokenizer = Tokenizer(model_name=tokenizer_model)


class LLMFactory:
    """
    A factory class to create LLM wrapper instances based on the specified backend.
    Supported backends include:
    - http_api: HTTPClient
    - openai_api: OpenAIClient
    - ollama_api: OllamaClient
    - huggingface: HuggingFaceWrapper
    - sglang: SGLangWrapper
    """

    @staticmethod
    def create_llm(
        model_type: str, backend: str, config: Dict[str, Any]
    ) -> BaseLLMWrapper:
        if not config:
            raise ValueError(
                f"No configuration provided for LLM {model_type} with backend {backend}."
            )

        actor_name = f"Actor_LLM_{model_type}"
        try:
            ray.get_actor(actor_name)
        except ValueError:
            print(f"Creating Ray actor for LLM {model_type} with backend {backend}.")
            num_gpus = int(config.pop("num_gpus", 0))
            actor = (
                ray.remote(LLMServiceActor)
                .options(
                    name=actor_name,
                    num_gpus=num_gpus,
                    lifetime="detached",
                    get_if_exists=True,
                )
                .remote(backend, config)
            )

            # wait for actor to be ready
            ray.get(actor.ready.remote())

        return LLMServiceProxy(actor_name)


def _load_env_group(prefix: str) -> Dict[str, Any]:
    """
    Collect environment variables with the given prefix into a dictionary,
    stripping the prefix from the keys.
    """
    return {
        k[len(prefix) :].lower(): v
        for k, v in os.environ.items()
        if k.startswith(prefix)
    }


def init_llm(model_type: str) -> Optional[BaseLLMWrapper]:
    if model_type == "synthesizer":
        prefix = "SYNTHESIZER_"
    elif model_type == "trainee":
        prefix = "TRAINEE_"
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented yet.")
    config = _load_env_group(prefix)
    # if config is empty, return None
    if not config:
        return None
    backend = config.pop("backend")
    llm_wrapper = LLMFactory.create_llm(model_type, backend, config)
    return llm_wrapper
