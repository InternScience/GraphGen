from typing import Any

from graphgen.bases import QAPair
from graphgen.utils import run_concurrent


def transform_to_qa_format(
    items: list[dict], format_hint: str = "auto"
) -> list[dict[str, str]]:
    extractors = {
        "ChatML": lambda x: (
            next(
                (
                    m["content"]
                    for m in x.get("messages", [])
                    if m.get("role") == "user"
                ),
                "",
            ),
            next(
                (
                    m["content"]
                    for m in x.get("messages", [])
                    if m.get("role") == "assistant"
                ),
                "",
            ),
        ),
        "Alpaca": lambda x: (
            f"{x.get('instruction', '')}\n\n{x['input']}".strip()
            if x.get("input")
            else x.get("instruction", ""),
            x.get("output", ""),
        ),
        "Sharegpt": lambda x: (
            next(
                (
                    c["value"]
                    for c in x.get("conversations", [])
                    if c.get("from") == "human"
                ),
                "",
            ),
            next(
                (
                    c["value"]
                    for c in x.get("conversations", [])
                    if c.get("from") in ("gpt", "assistant")
                ),
                "",
            ),
        ),
    }

    auto_detect = {
        "messages": "ChatML",
        "conversations": "Sharegpt",
        "instruction": "Alpaca",
    }

    transformed = []
    for item in items:
        fmt = format_hint
        if fmt == "auto":
            fmt = next(
                (fmt_name for key, fmt_name in auto_detect.items() if key in item), None
            )
            if not fmt:
                raise ValueError(
                    "Could not auto-detect format. Please specify format_hint."
                )

        question, answer = extractors[fmt](item)
        options = None
        if "\nOptions:\n" in question:
            q_part, opt_part = question.split("\nOptions:\n", 1)
            question = q_part
            options = {
                k.strip(): v.strip()
                for line in opt_part.strip().split("\n")
                if "." in line
                for k, v in [line.split(".", 1)]
            }

        result = {"question": question.strip(), "answer": answer.strip()}
        if options:
            result["options"] = options
        transformed.append(result)

    return transformed


def evaluate_qa(
    qa_evaluators: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, Any]:
    items = transform_to_qa_format(items)
    items = [QAPair.from_dict(item) for item in items]

    results = {}
    for key, qa_evaluator in qa_evaluators.items():
        result = run_concurrent(
            qa_evaluator.evaluate,
            items,
            desc=f"Evaluating QA with {key}",
        )
        results[key] = result
    return results
