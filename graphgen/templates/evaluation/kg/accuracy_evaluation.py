ENTITY_EVALUATION_PROMPT_ZH = """你是一个知识图谱质量评估专家。你的任务是从给定的文本块和提取的实体列表，评估实体提取的质量。

评估维度：
1. ACCURACY (准确性, 权重: 40%): 提取的实体是否正确，是否有误提取或错误识别
2. COMPLETENESS (完整性, 权重: 40%): 是否遗漏了文本中的重要实体
3. PRECISION (精确性, 权重: 20%): 提取的实体是否精确，命名是否准确

评分标准（每个维度 0-1 分）：
- EXCELLENT (0.8-1.0): 高质量提取
- GOOD (0.6-0.79): 良好质量，有少量问题
- ACCEPTABLE (0.4-0.59): 可接受，有明显问题
- POOR (0.0-0.39): 质量差，需要改进

综合评分 = 0.4 × Accuracy + 0.4 × Completeness + 0.2 × Precision

请评估以下内容：

原始文本块：
{chunk_content}

提取的实体列表：
{extracted_entities}

请以 JSON 格式返回评估结果：
{{
    "accuracy": <0-1之间的浮点数>,
    "completeness": <0-1之间的浮点数>,
    "precision": <0-1之间的浮点数>,
    "overall_score": <综合评分>,
    "accuracy_reasoning": "<准确性评估理由>",
    "completeness_reasoning": "<完整性评估理由，包括遗漏的重要实体>",
    "precision_reasoning": "<精确性评估理由>",
    "issues": ["<发现的问题列表>"]
}}
"""

ENTITY_EVALUATION_PROMPT_EN = """You are a Knowledge Graph Quality Assessment Expert. \
Your task is to evaluate the quality of entity extraction from a given text block and extracted entity list.

Evaluation Dimensions:
1. ACCURACY (Weight: 40%): Whether the extracted entities are correct, and if there are any false extractions or misidentifications
2. COMPLETENESS (Weight: 40%): Whether important entities from the text are missing
3. PRECISION (Weight: 20%): Whether the extracted entities are precise and accurately named

Scoring Criteria (0-1 scale for each dimension):
- EXCELLENT (0.8-1.0): High-quality extraction
- GOOD (0.6-0.79): Good quality with minor issues
- ACCEPTABLE (0.4-0.59): Acceptable with noticeable issues
- POOR (0.0-0.39): Poor quality, needs improvement

Overall Score = 0.4 × Accuracy + 0.4 × Completeness + 0.2 × Precision

Please evaluate the following:

Original Text Block:
{chunk_content}

Extracted Entity List:
{extracted_entities}

Please return the evaluation result in JSON format:
{{
    "accuracy": <float between 0-1>,
    "completeness": <float between 0-1>,
    "precision": <float between 0-1>,
    "overall_score": <overall score>,
    "accuracy_reasoning": "<reasoning for accuracy assessment>",
    "completeness_reasoning": "<reasoning for completeness assessment, including important missing entities>",
    "precision_reasoning": "<reasoning for precision assessment>",
    "issues": ["<list of identified issues>"]
}}
"""

RELATION_EVALUATION_PROMPT_ZH = """你是一个知识图谱质量评估专家。你的任务是从给定的文本块和提取的关系列表，评估关系抽取的质量。

评估维度：
1. ACCURACY (准确性, 权重: 40%): 提取的关系是否正确，关系描述是否准确
2. COMPLETENESS (完整性, 权重: 40%): 是否遗漏了文本中的重要关系
3. PRECISION (精确性, 权重: 20%): 关系描述是否精确，是否过于宽泛

评分标准（每个维度 0-1 分）：
- EXCELLENT (0.8-1.0): 高质量提取
- GOOD (0.6-0.79): 良好质量，有少量问题
- ACCEPTABLE (0.4-0.59): 可接受，有明显问题
- POOR (0.0-0.39): 质量差，需要改进

综合评分 = 0.4 × Accuracy + 0.4 × Completeness + 0.2 × Precision

请评估以下内容：

原始文本块：
{chunk_content}

提取的关系列表：
{extracted_relations}

请以 JSON 格式返回评估结果：
{{
    "accuracy": <0-1之间的浮点数>,
    "completeness": <0-1之间的浮点数>,
    "precision": <0-1之间的浮点数>,
    "overall_score": <综合评分>,
    "accuracy_reasoning": "<准确性评估理由>",
    "completeness_reasoning": "<完整性评估理由，包括遗漏的重要关系>",
    "precision_reasoning": "<精确性评估理由>",
    "issues": ["<发现的问题列表>"]
}}
"""

RELATION_EVALUATION_PROMPT_EN = """You are a Knowledge Graph Quality Assessment Expert. \
Your task is to evaluate the quality of relation extraction from a given text block and extracted relation list.

Evaluation Dimensions:
1. ACCURACY (Weight: 40%): Whether the extracted relations are correct and the relation descriptions are accurate
2. COMPLETENESS (Weight: 40%): Whether important relations from the text are missing
3. PRECISION (Weight: 20%): Whether the relation descriptions are precise and not overly broad

Scoring Criteria (0-1 scale for each dimension):
- EXCELLENT (0.8-1.0): High-quality extraction
- GOOD (0.6-0.79): Good quality with minor issues
- ACCEPTABLE (0.4-0.59): Acceptable with noticeable issues
- POOR (0.0-0.39): Poor quality, needs improvement

Overall Score = 0.4 × Accuracy + 0.4 × Completeness + 0.2 × Precision

Please evaluate the following:

Original Text Block:
{chunk_content}

Extracted Relation List:
{extracted_relations}

Please return the evaluation result in JSON format:
{{
    "accuracy": <float between 0-1>,
    "completeness": <float between 0-1>,
    "precision": <float between 0-1>,
    "overall_score": <overall score>,
    "accuracy_reasoning": "<reasoning for accuracy assessment>",
    "completeness_reasoning": "<reasoning for completeness assessment, including important missing relations>",
    "precision_reasoning": "<reasoning for precision assessment>",
    "issues": ["<list of identified issues>"]
}}
"""

ACCURACY_EVALUATION_PROMPT = {
    "zh": {
        "ENTITY": ENTITY_EVALUATION_PROMPT_ZH,
        "RELATION": RELATION_EVALUATION_PROMPT_ZH,
    },
    "en": {
        "ENTITY": ENTITY_EVALUATION_PROMPT_EN,
        "RELATION": RELATION_EVALUATION_PROMPT_EN,
    },
}
