ENTITY_TYPE_CONFLICT_PROMPT_ZH = """你是一个知识图谱一致性评估专家。你的任务是判断同一个实体在不同文本块中被提取为不同的类型，是否存在语义冲突。

实体名称：{entity_name}

在不同文本块中的类型提取结果：
{type_extractions}

预设的实体类型列表（供参考）：
concept, date, location, keyword, organization, person, event, work, nature, artificial, science, technology, mission, gene

请判断这些类型是否存在语义冲突（即它们是否描述的是同一类事物，还是存在矛盾）。
注意：如果类型只是同一概念的不同表述（如 concept 和 keyword），可能不算严重冲突。

请以 JSON 格式返回：
{{
    "has_conflict": <true/false>,
    "conflict_severity": <0-1之间的浮点数，0表示无冲突，1表示严重冲突>,
    "conflict_reasoning": "<冲突判断的理由>",
    "conflicting_types": ["<存在冲突的类型对>"],
    "recommended_type": "<如果存在冲突，推荐的正确类型（必须是预设类型之一）>"
}}
"""

ENTITY_TYPE_CONFLICT_PROMPT_EN = (
    """You are a Knowledge Graph Consistency Assessment Expert. """
    """Your task is to determine whether there are semantic conflicts """
    """when the same entity is extracted as different types in different text blocks.

Entity Name: {entity_name}

Type extraction results from different text blocks:
{type_extractions}

Preset entity type list (for reference):
concept, date, location, keyword, organization, person, event, work, nature, artificial, science, technology, mission, gene

Please determine whether these types have semantic conflicts (i.e., whether they describe the same category of things, or if there are contradictions).
Note: If types are just different expressions of the same concept (such as concept and keyword), it may not be considered a serious conflict.

Please return in JSON format:
{{
    "has_conflict": <true/false>,
    "conflict_severity": <float between 0-1, where 0 means no conflict, 1 means severe conflict>,
    "conflict_reasoning": "<reasoning for conflict judgment>",
    "conflicting_types": ["<pairs of conflicting types>"],
    "recommended_type": "<if there is a conflict, the recommended correct type (must be one of the preset types)>"
}}
"""
)

ENTITY_DESCRIPTION_CONFLICT_PROMPT_ZH = """你是一个知识图谱一致性评估专家。你的任务是判断同一个实体在不同文本块中的描述是否存在语义冲突。

实体名称：{entity_name}

在不同文本块中的描述：
{descriptions}

请判断这些描述是否存在语义冲突（即它们是否描述的是同一个实体，还是存在矛盾的信息）。

请以 JSON 格式返回：
{{
    "has_conflict": <true/false>,
    "conflict_severity": <0-1之间的浮点数>,
    "conflict_reasoning": "<冲突判断的理由>",
    "conflicting_descriptions": ["<存在冲突的描述对>"],
    "conflict_details": "<具体的冲突内容>"
}}
"""

ENTITY_DESCRIPTION_CONFLICT_PROMPT_EN = (
    """You are a Knowledge Graph Consistency Assessment Expert. """
    """Your task is to determine whether there are semantic conflicts """
    """in the descriptions of the same entity across different text blocks.

Entity Name: {entity_name}

Descriptions from different text blocks:
{descriptions}

Please determine whether these descriptions have semantic conflicts (i.e., whether they describe the same entity, or if there is contradictory information).

Please return in JSON format:
{{
    "has_conflict": <true/false>,
    "conflict_severity": <float between 0-1>,
    "conflict_reasoning": "<reasoning for conflict judgment>",
    "conflicting_descriptions": ["<pairs of conflicting descriptions>"],
    "conflict_details": "<specific conflict content>"
}}
"""
)

RELATION_CONFLICT_PROMPT_ZH = """你是一个知识图谱一致性评估专家。你的任务是判断同一对实体在不同文本块中的关系描述是否存在语义冲突。

实体对：{source_entity} -> {target_entity}

在不同文本块中的关系描述：
{relation_descriptions}

请判断这些关系描述是否存在语义冲突。

请以 JSON 格式返回：
{{
    "has_conflict": <true/false>,
    "conflict_severity": <0-1之间的浮点数>,
    "conflict_reasoning": "<冲突判断的理由>",
    "conflicting_relations": ["<存在冲突的关系描述对>"]
}}
"""

RELATION_CONFLICT_PROMPT_EN = (
    """You are a Knowledge Graph Consistency Assessment Expert. """
    """Your task is to determine whether there are semantic conflicts """
    """in the relation descriptions of the same entity pair across different text blocks.

Entity Pair: {source_entity} -> {target_entity}

Relation descriptions from different text blocks:
{relation_descriptions}

Please determine whether these relation descriptions have semantic conflicts.

Please return in JSON format:
{{
    "has_conflict": <true/false>,
    "conflict_severity": <float between 0-1>,
    "conflict_reasoning": "<reasoning for conflict judgment>",
    "conflicting_relations": ["<pairs of conflicting relation descriptions>"]
}}
"""
)

ENTITY_EXTRACTION_PROMPT_ZH = """从以下文本块中提取指定实体的类型和描述。

**重要**：你只需要提取指定的实体，不要提取其他实体。

实体名称：{entity_name}

文本块：
{chunk_content}

请从文本块中找到并提取**仅此实体**（实体名称：{entity_name}）的以下信息：

1. entity_type: 实体类型，必须是以下预设类型之一（小写）：
   - concept: 概念
   - date: 日期
   - location: 地点
   - keyword: 关键词
   - organization: 组织
   - person: 人物
   - event: 事件
   - work: 作品/工作
   - nature: 自然
   - artificial: 人工
   - science: 科学
   - technology: 技术
   - mission: 任务
   - gene: 基因

   如果无法确定类型，请使用 "concept" 作为默认值。

2. description: 实体描述（简要描述该实体在文本中的作用和特征）

请以 JSON 格式返回：
{{
    "entity_type": "<实体类型（必须是上述预设类型之一）>",
    "description": "<实体描述>"
}}
"""

ENTITY_EXTRACTION_PROMPT_EN = """Extract the type and description of the specified entity from the following text block.

**Important**: You should only extract the specified entity, do not extract other entities.

Entity Name: {entity_name}

Text Block:
{chunk_content}

Please find and extract the following information for **this entity only** (entity name: {entity_name}) from the text block:

1. entity_type: Entity type, must be one of the following preset types (lowercase):
   - concept: concept
   - date: date
   - location: location
   - keyword: keyword
   - organization: organization
   - person: person
   - event: event
   - work: work
   - nature: nature
   - artificial: artificial
   - science: science
   - technology: technology
   - mission: mission
   - gene: gene

   If the type cannot be determined, please use "concept" as the default value.

2. description: Entity description (briefly describe the role and characteristics of this entity in the text)

Please return in JSON format:
{{
    "entity_type": "<entity type (must be one of the preset types above)>",
    "description": "<entity description>"
}}
"""

CONSISTENCY_EVALUATION_PROMPT = {
    "zh": {
        "ENTITY_TYPE_CONFLICT": ENTITY_TYPE_CONFLICT_PROMPT_ZH,
        "ENTITY_DESCRIPTION_CONFLICT": ENTITY_DESCRIPTION_CONFLICT_PROMPT_ZH,
        "RELATION_CONFLICT": RELATION_CONFLICT_PROMPT_ZH,
        "ENTITY_EXTRACTION": ENTITY_EXTRACTION_PROMPT_ZH,
    },
    "en": {
        "ENTITY_TYPE_CONFLICT": ENTITY_TYPE_CONFLICT_PROMPT_EN,
        "ENTITY_DESCRIPTION_CONFLICT": ENTITY_DESCRIPTION_CONFLICT_PROMPT_EN,
        "RELATION_CONFLICT": RELATION_CONFLICT_PROMPT_EN,
        "ENTITY_EXTRACTION": ENTITY_EXTRACTION_PROMPT_EN,
    },
}
