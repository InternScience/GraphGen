from typing import Any, List, Optional

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper
from graphgen.bases.datatypes import Token


class RayServeClient(BaseLLMWrapper):
    """
    A client to interact with a Ray Serve deployment.
    """

    def __init__(
        self,
        *,
        app_name: Optional[str] = None,
        deployment_name: Optional[str] = None,
        serve_backend: Optional[str] = None,
        **kwargs: Any,
    ):
        try:
            from ray import serve
        except ImportError as e:
            raise ImportError(
                "Ray is not installed. Please install it with `pip install ray[serve]`."
            ) from e

        super().__init__(**kwargs)

        # Try to get existing handle first
        self.handle = None
        if app_name:
            try:
                self.handle = serve.get_app_handle(app_name)
            except Exception:
                pass
        elif deployment_name:
            try:
                self.handle = serve.get_deployment(deployment_name).get_handle()
            except Exception:
                pass

        # If no handle found, try to deploy if serve_backend is provided
        if self.handle is None:
            if serve_backend:
                if not app_name:
                    import uuid

                    app_name = f"llm_app_{serve_backend}_{uuid.uuid4().hex[:8]}"

                print(
                    f"Deploying Ray Serve app '{app_name}' with backend '{serve_backend}'..."
                )
                from graphgen.models.llm.local.ray_serve_deployment import LLMDeployment

                # Filter kwargs to avoid passing unrelated args if necessary,
                # but LLMDeployment config accepts everything for now.
                # Note: We need to pass kwargs as the config dict.
                deployment = LLMDeployment.bind(backend=serve_backend, config=kwargs)
                serve.run(deployment, name=app_name, route_prefix=f"/{app_name}")
                self.handle = serve.get_app_handle(app_name)
            elif app_name or deployment_name:
                raise ValueError(
                    f"Ray Serve app/deployment '{app_name or deployment_name}' "
                    "not found and 'serve_backend' not provided to deploy it."
                )
            else:
                raise ValueError(
                    "Either 'app_name', 'deployment_name' or 'serve_backend' "
                    "must be provided for RayServeClient."
                )

    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        """Generate answer from the model."""
        return await self.handle.generate_answer.remote(text, history, **extra)

    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        """Generate top-k tokens for the next token prediction."""
        return await self.handle.generate_topk_per_token.remote(text, history, **extra)

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        """Generate probabilities for each token in the input."""
        return await self.handle.generate_inputs_prob.remote(text, history, **extra)
