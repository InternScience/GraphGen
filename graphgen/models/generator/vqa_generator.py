from dataclasses import dataclass
from typing import Any

from graphgen.bases import BaseGenerator


@dataclass
class VQAGenerator(BaseGenerator):
    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        pass

    @staticmethod
    def parse_response(response: str) -> Any:
        pass
