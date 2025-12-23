import argparse
import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import OpenAIClient, Tokenizer
from graphgen.utils import compute_content_hash, create_event_loop

# Prompts from entigraph_utils/prompt_utils.py
OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES = """
As a knowledge analyzer, your task is to dissect and understand an article provided by the user. You are required to perform the following steps:
1. Summarize the Article: Provide a concise summary of the entire article, capturing the main points and themes.
2. Extract Entities: Identify and list all significant "nouns" or entities mentioned within the article. These entities should include but not limited to:
    * People: Any individuals mentioned in the article, using the names or references provided.
    * Places: Both specific locations and abstract spaces relevant to the content.
    * Object: Any concrete object that is referenced by the provided content.
    * Concepts: Any significant abstract ideas or themes that are central to the article's discussion.

Try to exhaust as many entities as possible. Your response should be structured in a JSON format to organize the information effectively. Ensure that the summary is brief yet comprehensive, and the list of entities is detailed and accurate.

Here is the format you should use for your response:

{
  "summary":  "<A concise summary of the article>",
  "entities": ["entity1", "entity2", ...]
}
"""

OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS = """
You will act as a knowledge analyzer tasked with dissecting an article provided by the user. Your role involves two main objectives:
1. Rephrasing Content: The user will identify two specific entities mentioned in the article. You are required to rephrase the content of the article twice:
    * Once, emphasizing the first entity.
    * Again, emphasizing the second entity.
2. Analyzing Interactions: Discuss how the two specified entities interact within the context of the article.

Your responses should provide clear segregation between the rephrased content and the interaction analysis. Ensure each section of the output include sufficient context, ideally referencing the article's title to maintain clarity about the discussion's focus.
Here is the format you should follow for your response:

### Discussion of <title> in relation to <entity1>
<Rephrased content focusing on the first entity>

### Discussion of <title> in relation to <entity2>
<Rephrased content focusing on the second entity>

### Discussion of Interaction between <entity1> and <entity2> in context of <title>
<Discussion on how the two entities interact within the article>
"""

OPENAI_API_SYSTEM_QUALITY_QA_SFT = """You are an assistant to help read \
a article and then rephrase it in a question answering format. \
The user will provide you with an article with its content. \
You need to generate a paraphrase of the same article in question and answer format with \
multiple tags of "Question: ..." followed by "Answer: ...".
Remember to keep the meaning and every content of the article intact.

Here is the format you should follow for your response:
Question: <Question>
Answer: <Answer>

Here is the article you need to rephrase:
{doc}
"""


@dataclass
class EntiGraph:
    model_name: str
    api_key: str
    base_url: str
    max_concurrent: int = 1000

    def __post_init__(self):
        self.tokenizer = Tokenizer()

        # Initialize specialized clients for different tasks to handle different system prompts and modes
        self.client_entities = OpenAIClient(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            tokenizer=self.tokenizer,
            system_prompt=OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES,
            json_mode=True
        )

        self.client_relations = OpenAIClient(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            tokenizer=self.tokenizer,
            system_prompt=OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS
        )

        self.client_qa = OpenAIClient(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            tokenizer=self.tokenizer,
            system_prompt="You are an assistant to help read a article \
                and then rephrase it in a question answering format."
        )

    async def generate_entities(self, content: str) -> Dict:
        prompt = f"""
        ### Document Content:
        {content}
        """
        max_tries = 5
        while max_tries > 0:
            try:
                response_str = await self.client_entities.generate_answer(prompt)
                if not response_str:
                    return None
                # Clean up json string if needed (sometimes markdown code blocks)
                if "```json" in response_str:
                    response_str = response_str.split("```json")[1].split("```")[0].strip()
                elif "```" in response_str:
                    response_str = response_str.split("```")[1].split("```")[0].strip()

                # Find start and end of json
                start = response_str.find("{")
                end = response_str.rfind("}")
                if start != -1 and end != -1:
                    response_str = response_str[start : end + 1]

                response = json.loads(response_str)
                if "entities" in response and "summary" in response:
                    return response
            except Exception as e:
                print(f"Failed to generate entities: {e}")
            max_tries -= 1
        return None

    async def generate_two_entity_relations(self, document: str, entity1: str, entity2: str) -> str:
        prompt = f"""
        ### Document Content:
        {document}
        ### Entities:
        - {entity1}
        - {entity2}
        """
        return await self.client_relations.generate_answer(prompt)

    async def generate_qa_sft(self, content: str) -> str:
        # We format the prompt using the template logic
        prompt = OPENAI_API_SYSTEM_QUALITY_QA_SFT.format(doc=content)
        return await self.client_qa.generate_answer(prompt)

    def generate(self, input_docs: List[str]) -> List[dict]:
        loop = create_event_loop()
        return loop.run_until_complete(self.async_generate(input_docs))

    async def async_generate(self, input_docs: List[str]) -> dict:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # 1. Generate Entities
        async def process_entities(doc_text):
            async with semaphore:
                res = await self.generate_entities(doc_text)
                if res:
                    return {
                        "document": doc_text,
                        "entities": res["entities"],
                        "summary": res["summary"]
                    }
                return None

        entities_results = []
        for result in tqdm_async(
            asyncio.as_completed([process_entities(doc) for doc in input_docs]),
            total=len(input_docs),
            desc="Generating entities"
        ):
            res = await result
            if res:
                entities_results.append(res)

        # 2. Generate Relations (Pairs)
        pair_list = []
        random.seed(42)
        for item in entities_results:
            entities = item["entities"]
            doc_text = item["document"]
            temp_pairs = []
            for i, entity_i in enumerate(entities):
                for j in range(i + 1, len(entities)):
                    temp_pairs.append((doc_text, entity_i, entities[j]))

            # Sample max 10 pairs per document
            pair_list.extend(random.sample(temp_pairs, min(len(temp_pairs), 10)))

        async def process_relation(pair):
            async with semaphore:
                doc_text, e1, e2 = pair
                try:
                    return await self.generate_two_entity_relations(doc_text, e1, e2)
                except Exception as e:
                    print(f"Error generating relations: {e}")
                    return None

        corpus = []
        for result in tqdm_async(
            asyncio.as_completed([process_relation(pair) for pair in pair_list]),
            total=len(pair_list),
            desc="Generating relations"
        ):
            res = await result
            if res:
                corpus.append(res)

        # Combine summaries and relation discussions into the corpus for QA generation
        full_corpus = [item["summary"] for item in entities_results] + corpus

        # 3. Generate QA SFT
        final_results = {}

        async def process_qa(text):
            async with semaphore:
                try:
                    qa_text = await self.generate_qa_sft(text)
                    if qa_text:
                        return _post_process_synthetic_data(qa_text)
                except Exception as e:
                    print(f"Error generating QA: {e}")
                return {}

        for result in tqdm_async(
            asyncio.as_completed([process_qa(text) for text in full_corpus]),
            total=len(full_corpus),
            desc="Generating QA SFT"
        ):
            qas = await result
            if qas:
                final_results.update(qas)

        return final_results


def _post_process_synthetic_data(data: str) -> dict:
    # Logic from original code
    block = data.split("\n\n")
    qas = {}
    for line in block:
        if "Question: " in line and "Answer: " in line:
            try:
                question = line.split("Question: ")[1].split("Answer: ")[0].strip()
                answer = line.split("Answer: ")[1].strip()
                if question and answer:
                    qas[compute_content_hash(question)] = {
                        "question": question,
                        "answer": answer,
                    }
            except IndexError:
                continue
    return qas


def load_from_json(file_obj) -> List[str]:
    """Helper to load docs from a standard JSON list."""
    documents = []
    data = json.load(file_obj)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                for chunk in item:
                    if isinstance(chunk, dict) and "content" in chunk:
                        documents.append(chunk["content"])
            elif isinstance(item, dict) and "content" in item:
                documents.append(item["content"])
    return documents


def load_from_jsonl(file_obj) -> List[str]:
    """Helper to load docs from a JSONL file."""
    documents = []
    file_obj.seek(0)
    for line in file_obj:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict) and "content" in item:
                documents.append(item["content"])
        except json.JSONDecodeError:
            continue
    return documents


def load_and_dedup_data(input_file: str) -> List[str]:
    documents = []
    with open(input_file, "r", encoding="utf-8") as file_obj:
        try:
            documents = load_from_json(file_obj)
        except json.JSONDecodeError:
            # Try JSONL
            documents = load_from_jsonl(file_obj)

    # Dedup
    deduped = {}
    for text in documents:
        h = compute_content_hash(text)
        if h not in deduped:
            deduped[h] = text
    return list(deduped.values())


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
        default="cache/data/entigraph.json",
        type=str,
    )

    args = parser.parse_args()

    load_dotenv()

    # Load data
    docs = load_and_dedup_data(args.input_file)

    entigraph = EntiGraph(
        model_name=os.getenv("SYNTHESIZER_MODEL"),
        api_key=os.getenv("SYNTHESIZER_API_KEY"),
        base_url=os.getenv("SYNTHESIZER_BASE_URL"),
    )

    results = entigraph.generate(docs)

    # Save results
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
