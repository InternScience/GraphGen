import os
from typing import Any, Dict

from graphgen.bases import BaseLLMWrapper
from graphgen.models import HTTPClient, OpenAIClient, Tokenizer


class LLMFactory:
    """
    A factory class to create LLM wrapper instances based on the specified backend.
    Supported backends include:
    - http_api: HTTPClient
    - openai_api: OpenAIClient
    - ollama_api: OllamaClient
    - ollama: OllamaWrapper
    - deepspeed: DeepSpeedWrapper
    - huggingface: HuggingFaceWrapper
    - tgi: TGIWrapper
    - sglang: SGLangWrapper
    - tensorrt: TensorRTWrapper
    """

    @staticmethod
    def create_llm_wrapper(backend: str, config: Dict[str, Any]) -> BaseLLMWrapper:
        # add tokenizer
        tokenizer: Tokenizer = Tokenizer(
            os.environ.get("TOKENIZER_MODEL", "cl100k_base"),
        )
        config["tokenizer"] = tokenizer
        if backend == "http_api":
            return HTTPClient(**config)
        if backend == "openai_api":
            return OpenAIClient(**config)
        raise NotImplementedError(f"Backend {backend} is not implemented yet.")


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


def init_llm(model_type: str) -> BaseLLMWrapper:
    if model_type == "synthesizer":
        prefix = "SYNTHESIZER_"
    elif model_type == "trainee":
        prefix = "TRAINEE_"
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented yet.")
    config = _load_env_group(prefix)
    backend = config.pop("backend")
    llm_wrapper = LLMFactory.create_llm_wrapper(backend, config)
    return llm_wrapper
