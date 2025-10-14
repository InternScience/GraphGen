# import asyncio
# from typing import Dict
#
# import gradio as gr
# from tqdm.asyncio import tqdm as tqdm_async
#
# from graphgen.models import JsonKVStorage, NetworkXStorage, OpenAIClient, Tokenizer
# from graphgen.operators.partition.split_kg import get_batches_with_strategy
# from graphgen.templates import MULTI_HOP_GENERATION_PROMPT, QUESTION_GENERATION_PROMPT
# from graphgen.utils import (
#     compute_content_hash,
#     detect_main_language,
#     logger,
#     run_concurrent,
# )
#
#
# def get_average_loss(batch: tuple, loss_strategy: str) -> float:
#     try:
#         if loss_strategy == "only_edge":
#             return sum(edge[2]["loss"] for edge in batch[1]) / len(batch[1])
#         if loss_strategy == "both":
#             return sum(edge[2]["loss"] for edge in batch[1]) + sum(
#                 node["loss"] for node in batch[0]
#             ) / (len(batch[0]) + len(batch[1]))
#         raise ValueError("Invalid loss strategy")
#     except Exception as e:  # pylint: disable=broad-except
#         logger.warning(
#             "Loss not found in some nodes or edges, setting loss to -1.0: %s", e
#         )
#         return -1.0
#
#
# def _post_process_synthetic_data(data):
#     block = data.split("\n\n")
#     qas = []
#     for line in block:
#         if "Question:" in line and "Answer:" in line:
#             question = line.split("Question:")[1].split("Answer:")[0].strip()
#             answer = line.split("Answer:")[1].strip()
#             qas.append({"question": question, "answer": answer})
#         elif "问题：" in line and "答案：" in line:
#             question = line.split("问题：")[1].split("答案：")[0].strip()
#             answer = line.split("答案：")[1].strip()
#             qas.append({"question": question, "answer": answer})
#         elif "问题:" in line and "回答:" in line:
#             question = line.split("问题:")[1].split("回答:")[0].strip()
#             answer = line.split("回答:")[1].strip()
#             qas.append({"question": question, "answer": answer})
#     return qas
#
#
# async def traverse_graph_for_multi_hop(
#     llm_client: OpenAIClient,
#     tokenizer: Tokenizer,
#     graph_storage: NetworkXStorage,
#     traverse_strategy: Dict,
#     text_chunks_storage: JsonKVStorage,
#     progress_bar: gr.Progress = None,
#     max_concurrent: int = 1000,
# ) -> dict:
#     """
#     Traverse the graph for multi-hop
#
#     :param llm_client
#     :param tokenizer
#     :param graph_storage
#     :param traverse_strategy
#     :param text_chunks_storage
#     :param progress_bar
#     :param max_concurrent
#     :return: question and answer
#     """
#     semaphore = asyncio.Semaphore(max_concurrent)
#
#     edges = list(await graph_storage.get_all_edges())
#     nodes = list(await graph_storage.get_all_nodes())
#
#     results = {}
#     edges, nodes = await _pre_tokenize(graph_storage, tokenizer, edges, nodes)
#
#     processing_batches = await get_batches_with_strategy(
#         nodes, edges, graph_storage, traverse_strategy
#     )
#
#     async def _process_single_batch(_process_batch: tuple) -> dict:
#         async with semaphore:
#             try:
#                 language = (
#                     "Chinese"
#                     if detect_main_language(_process_batch[0][0]["description"]) == "zh"
#                     else "English"
#                 )
#
#                 _process_nodes = _process_batch[0]
#                 _process_edges = _process_batch[1]
#
#                 entities = [
#                     f"{_process_node['node_id']}: {_process_node['description']}"
#                     for _process_node in _process_nodes
#                 ]
#
#                 relations = [
#                     f"{_process_edge[0]} -- {_process_edge[1]}: {_process_edge[2]['description']}"
#                     for _process_edge in _process_edges
#                 ]
#
#                 entities_str = "\n".join(
#                     [f"{index + 1}. {entity}" for index, entity in enumerate(entities)]
#                 )
#                 relations_str = "\n".join(
#                     [
#                         f"{index + 1}. {relation}"
#                         for index, relation in enumerate(relations)
#                     ]
#                 )
#
#                 prompt = MULTI_HOP_GENERATION_PROMPT[language].format(
#                     entities=entities_str, relationships=relations_str
#                 )
#
#                 context = await llm_client.generate_answer(prompt)
#
#                 # post-process the context
#                 if "Question:" in context and "Answer:" in context:
#                     question = context.split("Question:")[1].split("Answer:")[0].strip()
#                     answer = context.split("Answer:")[1].strip()
#                 elif "问题：" in context and "答案：" in context:
#                     question = context.split("问题：")[1].split("答案：")[0].strip()
#                     answer = context.split("答案：")[1].strip()
#                 else:
#                     return {}
#
#                 question = question.strip('"')
#                 answer = answer.strip('"')
#
#                 logger.info("Question: %s", question)
#                 logger.info("Answer: %s", answer)
#
#                 return {
#                     compute_content_hash(question): {
#                         "question": question,
#                         "answer": answer,
#                         "loss": get_average_loss(
#                             _process_batch, traverse_strategy["loss_strategy"]
#                         ),
#                     }
#                 }
#
#             except Exception as e:  # pylint: disable=broad-except
#                 logger.error("Error occurred while processing batch: %s", e)
#                 return {}
#
#     results_list = await run_concurrent(
#         _process_single_batch,
#         processing_batches,
#         progress_bar=progress_bar,
#         desc="[4/4]Generating QAs",
#     )
#
#     for res in results_list:
#         results.update(res)
#
#     return results
