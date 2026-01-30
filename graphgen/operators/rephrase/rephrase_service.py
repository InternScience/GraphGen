from typing import Tuple

from graphgen.bases import BaseLLMWrapper, BaseOperator
from graphgen.common.init_llm import init_llm
from graphgen.utils import run_concurrent


class RephraseService(BaseOperator):
    """
    Generate question-answer pairs based on nodes and edges.
    """

    def __init__(
        self,
        working_dir: str = "cache",
        method: str = "aggregated",
        **rephrase_kwargs,
    ):
        super().__init__(working_dir=working_dir, op_name="rephrase_service")
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.method = method
        self.rephrase_kwargs = rephrase_kwargs

        if self.method == "style_controlled":
            from graphgen.models import StyleControlledRephraser

            self.rephraser = StyleControlledRephraser(
                self.llm_client,
                style=rephrase_kwargs.get("style", "critical_analysis"),
            )
        else:
            raise ValueError(f"Unsupported rephrase method: {self.method}")

    def process(self, batch: list) -> Tuple[list, dict]:
        """
        Rephrase the texts in the batch.
        :return: A tuple of (results, meta_updates)
            results: A list of dicts containing rephrased texts. Each dict has the structure:
                {"_trace_id": str, "content": str}
            meta_updates: A dict mapping source IDs to lists of trace IDs for the rephrased texts.
        """
        final_results = []
        meta_updates = {}

        results = run_concurrent(
            self.rephraser.rephrase,
            batch,
            desc="Rephrasing texts",
            unit="batch",
        )

        for input_trace_id, rephrased in zip(
            [item["_trace_id"] for item in batch], results
        ):
            if not rephrased:
                continue
            rephrased["_trace_id"] = self.get_trace_id(rephrased)
            results.append(rephrased)
            meta_updates.setdefault(input_trace_id, []).append(rephrased["_trace_id"])
            final_results.append(rephrased)

        return final_results, meta_updates
