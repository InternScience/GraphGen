"""Event-entity KG builder (PMS fork, SAG-adapted).

来源与许可：核心设计改编自 Zleap-AI/SAG `zleap/sag/modules/extract/{schema,grounding}.py`
与 `prompts/extract_document.yaml` v3.1（MIT，Copyright (c) 2026 sag contributors，
arXiv:2606.15971）。改编点见 PMS fork docs/PMS_PATCHES.md #7 与 Phase 2 方案文档 §3.2。

与 LightRAGKGBuilder 的差异：
- 每 chunk 抽取"语义完整的事件"而非三元组：事件 = EVENT 节点 + "事件→参与者"
  星型边（超边的星型编码，n 元关系不拆三元组）；
- 严格响应合同（禁合同外字段、is_valid 自报、无锚 fail-closed 丢弃）；
- 数字接地（SAG grounding 移植）：事件中一切数字必须在原文逐字出现，否则弃；
- 事件节点 id 确定性（内容 SHA），chunk 重叠去重由官方 merge_nodes SEP 合并免费获得。

extract() 返回与官方 ({entity_name: [node]}, {(src, tgt): [edge]}) 相同的形状，
merge_nodes / merge_edges / 描述摘要全部复用官方实现。
"""

from __future__ import annotations

import json

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper, Chunk
from graphgen.templates import EVENT_ENTITY_EXTRACTION_PROMPT
from graphgen.templates.kg.event_entity_extraction import PMS_ENTITY_TYPES
from graphgen.utils import detect_main_language, logger

from graphgen.models.kg_builder.event_contract import (
    PROJECT_ID_RE,
    _ALLOWED_ENTITY_KEYS,
    _ALLOWED_EVENT_KEYS,
    _ALLOWED_TYPES,
    event_anchor,
    event_node_name,
    numbers_in,
    parse_extraction_response,
)


class EventEntityKGBuilder:
    """事件-实体抽取器：extract → 星型编码节点/边；合并复用官方 LightRAG merge。"""

    def __init__(self, llm_client: BaseLLMWrapper):
        from graphgen.models import LightRAGKGBuilder

        self.llm_client = llm_client
        self._merger = LightRAGKGBuilder(llm_client=llm_client)

    async def extract(
        self, chunk: Chunk
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        """单 chunk 事件抽取，返回与官方同形的 (nodes_data, edges_data)。"""
        content = chunk.content
        language = detect_main_language(content)
        entity_types_json = json.dumps(PMS_ENTITY_TYPES, ensure_ascii=False)
        request_json = json.dumps(
            {
                "type": "request",
                "data": {
                    "items": [{"id": 1, "content": content}],
                    "meta": {"entity_types": "__ENTITY_TYPES__"},
                },
            },
            ensure_ascii=False,
        ).replace("__ENTITY_TYPES__", entity_types_json)
        # 模板含 JSON 字面大括号，禁用 str.format；用占位符替换注入输入。
        prompt = EVENT_ENTITY_EXTRACTION_PROMPT[language].replace("{input_text}", request_json)

        response = await self.llm_client.generate_answer(prompt)
        events, dropped = parse_extraction_response(response, chunk_content=content)
        if sum(dropped.values()):
            logger.warning(
                "Event extraction dropped: %s (chunk=%s)", dropped, chunk.id
            )
        if not events:
            return {}, {}

        nodes: Dict[str, List[dict]] = defaultdict(list)
        edges: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for event in events:
            anchor = event["anchor"]
            event_name = event_node_name(anchor, event["title"], event["content"])
            description = f"{event['title']}：{event['content']}"
            # 事件节点（EVENT 实体，扁平属性；参与者由星型边表达）
            nodes[event_name].append(
                {
                    "entity_type": "EVENT",
                    "entity_name": event_name,
                    "description": description,
                    "source_id": chunk.id,
                }
            )
            # 参与者实体（去重保序；锚点项目实体若缺失则补齐）
            seen: set[str] = set()
            participants: list[dict] = []
            for entity in event["entities"]:
                if entity["name"] not in seen:
                    seen.add(entity["name"])
                    participants.append(entity)
            if anchor not in seen:
                participants.append(
                    {
                        "type": "PROJECT",
                        "name": anchor,
                        "description": f"项目 {anchor}，本事项的锚定项目",
                    }
                )
                seen.add(anchor)
            for entity in participants:
                nodes[entity["name"]].append(
                    {
                        "entity_type": entity["type"],
                        "entity_name": entity["name"],
                        "description": entity["description"],
                        "source_id": chunk.id,
                    }
                )
                edges[(event_name, entity["name"])].append(
                    {
                        "src_id": event_name,
                        "tgt_id": entity["name"],
                        "description": f"事件参与者：{entity['name']}（{entity['type']}）—— {entity['description']}",
                        "source_id": chunk.id,
                        "role": entity["type"],
                    }
                )
        return dict(nodes), dict(edges)

    async def merge_nodes(self, node_data: tuple, kg_instance: BaseGraphStorage) -> dict:
        return await self._merger.merge_nodes(node_data, kg_instance=kg_instance)

    async def merge_edges(self, edges_data: tuple, kg_instance: BaseGraphStorage) -> dict:
        return await self._merger.merge_edges(edges_data, kg_instance=kg_instance)
