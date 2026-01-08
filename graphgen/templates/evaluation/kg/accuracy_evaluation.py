ENTITY_EVALUATION_PROMPT_ZH = """你是一个知识图谱质量评估专家。你的任务是从给定的文本块和提取的实体列表，评估实体提取的质量。

评估维度：
1. ACCURACY (准确性, 权重: 40%): 提取的实体是否真实存在于文本中，是否存在误提取（False Positive）
   - 检查：实体是否在文本中实际出现，是否将非实体文本误识别为实体
   - 示例：文本提到"蛋白质A"，但提取了文本中不存在的"蛋白质B" → 准确性低
   - 示例：将"研究显示"这样的非实体短语提取为实体 → 准确性低

2. COMPLETENESS (完整性, 权重: 40%): 是否遗漏了文本中的重要实体（Recall）
   - 检查：文本中的重要实体是否都被提取，是否存在遗漏（False Negative）
   - 示例：文本提到5个重要蛋白质，但只提取了3个 → 完整性低
   - 示例：所有关键实体都被提取 → 完整性高

3. PRECISION (精确性, 权重: 20%): 提取的实体命名是否精确、边界是否准确、类型是否正确
   - 检查：实体名称是否完整准确，边界是否正确，实体类型分类是否正确
   - 示例：应提取"人类胰岛素受体蛋白"，但只提取了"胰岛素" → 精确性低（边界不准确）
   - 示例：应分类为"蛋白质"，但分类为"基因" → 精确性低（类型错误）
   - 示例：应提取"COVID-19"，但提取了"冠状病毒" → 精确性低（命名不够精确）

评分标准（每个维度 0-1 分）：
- EXCELLENT (0.8-1.0): 高质量提取，错误率 < 20%
- GOOD (0.6-0.79): 良好质量，有少量问题，错误率 20-40%
- ACCEPTABLE (0.4-0.59): 可接受，有明显问题，错误率 40-60%
- POOR (0.0-0.39): 质量差，需要改进，错误率 > 60%

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
1. ACCURACY (Weight: 40%): Whether the extracted entities actually exist in the text, and if there are any false extractions (False Positives)
   - Check: Do entities actually appear in the text? Are non-entity phrases incorrectly identified as entities?
   - Example: Text mentions "Protein A", but "Protein B" (not in text) is extracted → Low accuracy
   - Example: Phrases like "research shows" are extracted as entities → Low accuracy

2. COMPLETENESS (Weight: 40%): Whether important entities from the text are missing (Recall, False Negatives)
   - Check: Are all important entities from the text extracted? Are there any omissions?
   - Example: Text mentions 5 important proteins, but only 3 are extracted → Low completeness
   - Example: All key entities are extracted → High completeness

3. PRECISION (Weight: 20%): Whether extracted entities are precisely named, have correct boundaries, and correct types
   - Check: Are entity names complete and accurate? Are boundaries correct? Are entity types correctly classified?
   - Example: Should extract "Human Insulin Receptor Protein", but only "Insulin" is extracted → Low precision (incorrect boundary)
   - Example: Should be classified as "Protein", but classified as "Gene" → Low precision (incorrect type)
   - Example: Should extract "COVID-19", but "Coronavirus" is extracted → Low precision (naming not precise enough)

Scoring Criteria (0-1 scale for each dimension):
- EXCELLENT (0.8-1.0): High-quality extraction, error rate < 20%
- GOOD (0.6-0.79): Good quality with minor issues, error rate 20-40%
- ACCEPTABLE (0.4-0.59): Acceptable with noticeable issues, error rate 40-60%
- POOR (0.0-0.39): Poor quality, needs improvement, error rate > 60%

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
1. ACCURACY (准确性, 权重: 40%): 提取的关系是否真实存在于文本中，是否存在误提取（False Positive）
   - 检查：关系是否在文本中实际表达，是否将不存在的关系误识别为关系
   - 示例：文本中A和B没有关系，但提取了"A-作用于->B" → 准确性低
   - 示例：将文本中的并列关系误识别为因果关系 → 准确性低

2. COMPLETENESS (完整性, 权重: 40%): 是否遗漏了文本中的重要关系（Recall）
   - 检查：文本中表达的重要关系是否都被提取，是否存在遗漏（False Negative）
   - 示例：文本明确表达了5个关系，但只提取了3个 → 完整性低
   - 示例：所有关键关系都被提取 → 完整性高

3. PRECISION (精确性, 权重: 20%): 关系描述是否精确，关系类型是否正确，是否过于宽泛
   - 检查：关系类型是否准确，关系描述是否具体，是否使用了过于宽泛的关系类型
   - 示例：应提取"抑制"关系，但提取了"影响"关系 → 精确性低（类型不够精确）
   - 示例：应提取"直接结合"，但提取了"相关" → 精确性低（描述过于宽泛）
   - 示例：关系方向是否正确（如"A激活B" vs "B被A激活"）→ 精确性检查

评分标准（每个维度 0-1 分）：
- EXCELLENT (0.8-1.0): 高质量提取，错误率 < 20%
- GOOD (0.6-0.79): 良好质量，有少量问题，错误率 20-40%
- ACCEPTABLE (0.4-0.59): 可接受，有明显问题，错误率 40-60%
- POOR (0.0-0.39): 质量差，需要改进，错误率 > 60%

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
1. ACCURACY (Weight: 40%): Whether the extracted relations actually exist in the text, and if there are any false extractions (False Positives)
   - Check: Do relations actually appear in the text? Are non-existent relations incorrectly identified?
   - Example: Text shows no relation between A and B, but "A-acts_on->B" is extracted → Low accuracy
   - Example: A parallel relationship in text is misidentified as a causal relationship → Low accuracy

2. COMPLETENESS (Weight: 40%): Whether important relations from the text are missing (Recall, False Negatives)
   - Check: Are all important relations expressed in the text extracted? Are there any omissions?
   - Example: Text explicitly expresses 5 relations, but only 3 are extracted → Low completeness
   - Example: All key relations are extracted → High completeness

3. PRECISION (Weight: 20%): Whether relation descriptions are precise, relation types are correct, and not overly broad
   - Check: Are relation types accurate? Are relation descriptions specific? Are overly broad relation types used?
   - Example: Should extract "inhibits" relation, but "affects" is extracted → Low precision (type not precise enough)
   - Example: Should extract "directly binds", but "related" is extracted → Low precision (description too broad)
   - Example: Is relation direction correct (e.g., "A activates B" vs "B is activated by A") → Precision check

Scoring Criteria (0-1 scale for each dimension):
- EXCELLENT (0.8-1.0): High-quality extraction, error rate < 20%
- GOOD (0.6-0.79): Good quality with minor issues, error rate 20-40%
- ACCEPTABLE (0.4-0.59): Acceptable with noticeable issues, error rate 40-60%
- POOR (0.0-0.39): Poor quality, needs improvement, error rate > 60%

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
