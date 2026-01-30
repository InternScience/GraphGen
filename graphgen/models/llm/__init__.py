from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api.http_client import HTTPClient
    from .api.ollama_client import OllamaClient
    from .api.openai_client import OpenAIClient
    from .local.hf_wrapper import HuggingFaceWrapper


_import_map = {
    "HTTPClient": ".api.http_client",
    "OllamaClient": ".api.ollama_client",
    "OpenAIClient": ".api.openai_client",
    "HuggingFaceWrapper": ".local.hf_wrapper",
}


def __getattr__(name):
    if name in _import_map:
        import importlib

        module = importlib.import_module(_import_map[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_import_map.keys())
