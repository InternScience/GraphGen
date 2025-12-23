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

PROMPT_TEMPLATE = """Instruction: X
Output:{doc}

What kind of instruction could this be the answer to?
X:"""


@dataclass
class LongForm:
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
                question = await self.llm_client.generate_answer(prompt)
                if question is None:
                    return {}
                return {
                    compute_content_hash(question): {
                        "question": question,
                        "answer": content,
                    }
                }

        tasks = []
        for doc in docs:
            if isinstance(doc, list):
                for chunk in doc:
                    tasks.append(process_chunk(chunk["content"]))
            elif isinstance(doc, dict):
                tasks.append(process_chunk(doc["content"]))

        for result in tqdm_async(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Generating using LongForm",
        ):
            try:
                qa = await result
                if qa:
                    final_results.update(qa)
            except Exception as e:  # pylint: disable=broad-except
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
        default="cache/data/longform.json",
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

    longform = LongForm(llm_client=llm_client)

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = longform.generate(data)

    # Save results
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
