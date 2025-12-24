import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import OpenAIClient, Tokenizer
from graphgen.utils import compute_content_hash, create_event_loop

PROMPT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the questions.
USER: Convert the following paragraph into a conversational format with
multiple tags of "Question:" followed by "Answer:":{doc}.

Examples and format:
---
Question: What was the stock’s closing price on Friday? Answer: $21.51.
---
Question: How much did the stock rise on Friday? Answer: $2.11 or about 11 percent.
---
Question: What was the revenue drop in the first quarter compared to the same period last year? Answer: The revenue dropped 15 percent.
---
"""


def _post_process(content: str) -> list:
    raw_qas = content.split("---")
    qas = []
    for item in raw_qas:
        try:
            if "Question:" in item and "Answer:" in item:
                question = item.split("Question:")[1].split("Answer:")[0].strip()
                answer = item.split("Answer:")[1].strip()
                qas.append((question, answer))
        except Exception as e:
            print(f"Error: {e}")
            continue
    return qas


@dataclass
class Wrap:
    llm_client: OpenAIClient = None
    max_concurrent: int = 1000

    def generate(self, docs: List[List[dict]]) -> List[dict]:
        loop = create_event_loop()
        return loop.run_until_complete(self.async_generate(docs))

    async def async_generate(self, docs: List[List[dict]]) -> dict:
        final_results = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_chunk(content: str):
            async with semaphore:
                prompt = PROMPT_TEMPLATE.format(doc=content)
                return await self.llm_client.generate_answer(prompt)

        tasks = []
        for doc in docs:
            if isinstance(doc, list):
                for chunk in doc:
                    tasks.append(process_chunk(chunk["content"]))
            elif isinstance(doc, dict):
                tasks.append(process_chunk(doc["content"]))

        for result in tqdm_async(
            asyncio.as_completed(tasks), total=len(tasks), desc="Generating using Wrap"
        ):
            try:
                qas = _post_process(await result)
                for qa in qas:
                    final_results[compute_content_hash(qa[0])] = {
                        "question": qa[0],
                        "answer": qa[1],
                    }
            except Exception as e:
                print(f"Error: {e}")
        return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="Raw context jsonl path.",
        default="resources/input_examples/json_demo.json",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="Output file path.",
        default="cache/data/wrap.json",
        type=str,
    )

    args = parser.parse_args()

    load_dotenv()

    llm_client = OpenAIClient(
        model=os.getenv("SYNTHESIZER_MODEL"),
        api_key=os.getenv("SYNTHESIZER_API_KEY"),
        base_url=os.getenv("SYNTHESIZER_BASE_URL"),
        tokenizer=Tokenizer(model_name=os.getenv("TOKENIZER_MODEL")),
    )

    wrap = Wrap(llm_client=llm_client)

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = wrap.generate(data)

    # Save results
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
