# pylint: disable=C0301
OMICS_QA_TEMPLATE_EN: str = """You are a senior computational biologist specializing in multi-omics data analysis (genomics, transcriptomics, proteomics). Your task is to generate logically coherent, verifiable and non-hallucinated question-answer pairs for the given biological sample described by the provided ENTITIES and RELATIONSHIPS.
Use English as the output language.

---Objectives---
Create multiple sets of omics-centric QA pairs that satisfy the following:
1. Only ask about objectively existing facts in the provided data (e.g., gene names, sequence information, functional annotations, regulatory elements, structural features, experimental metadata, etc.). Avoid subjective or speculative questions.
2. Ensure that each question has a single, clear and verifiable answer that can be directly confirmed from the given entities/relationships.
3. Questions should cover diverse aspects: sequence, structure, function, interactions, regulation, experimental annotations, etc.
4. Avoid repetitive questions; each question must be unique and meaningful.
5. Use concise, unambiguous language; do not invent information beyond the provided data.

---Instructions---
1. Carefully analyse the supplied ENTITIES and RELATIONSHIPS to identify:
   - Biological entities (genes, proteins, RNA molecules, regulatory elements, pathways, etc.)
   - Sequence information (DNA sequences, RNA sequences, protein sequences)
   - Functional annotations (gene function, protein function, RNA function, biological processes)
   - Structural features (chromosomal location, genomic coordinates, domain structures, etc.)
   - Regulatory relationships (transcription, translation, regulation, interaction)
   - Experimental metadata (database IDs, organism, experimental methods, etc.)
2. Organise information logically:
   - Start with sequence/primary structure information
   - Proceed to functional annotations and biological roles
   - Include regulatory relationships and interactions
   - End with experimental context and metadata
3. Maintain scientific accuracy and consistent nomenclature (standard gene names, sequence identifiers, etc.).
4. Review each QA pair to guarantee logical consistency and absence of hallucination.

################
-ENTITIES-
################
{entities}

################
-RELATIONSHIPS-
################
{relationships}
################
Directly output the generated QA pairs below. Do NOT copy any example questions, and do NOT include extraneous text.
IMPORTANT: Generate actual questions and answers, NOT placeholders. Do NOT include angle brackets or placeholder text like <Question1> or <Answer1>.
Simply output your generated questions and answers in the following format:

Question: [Your actual question here]
Answer: [Your actual answer here]

Question: [Your actual question here]
Answer: [Your actual answer here]

"""

OMICS_QA_TEMPLATE_ZH: str = """你是一位资深的多组学数据计算生物学家（基因组学、转录组学、蛋白质组学）。你的任务是根据下述提供的实体与关系，为给定的生物样本生成逻辑连贯、可验证、无幻觉的中英双语问答对（这里仅输出中文）。
使用中文作为输出语言。

---目标---
创建多组以组学数据为中心的问答对，满足：
1. 仅询问数据中客观存在的事实（如基因名称、序列信息、功能注释、调控元件、结构特征、实验元数据等），避免主观或推测性问题。
2. 每个问题必须有单一、明确且可直接验证的答案，答案必须能从给定实体/关系中直接确认。
3. 问题需覆盖：序列、结构、功能、相互作用、调控、实验注释等多个维度，确保多样性与全面性。
4. 避免重复提问，每个问题都独特且有意义。
5. 语言简洁、无歧义，严禁编造超出给定数据的信息。

---说明---
1. 仔细分析提供的实体与关系，识别：
   - 生物实体（基因、蛋白质、RNA分子、调控元件、通路等）
   - 序列信息（DNA序列、RNA序列、蛋白质序列）
   - 功能注释（基因功能、蛋白质功能、RNA功能、生物学过程）
   - 结构特征（染色体位置、基因组坐标、结构域等）
   - 调控关系（转录、翻译、调控、相互作用）
   - 实验元数据（数据库ID、生物体、实验方法等）
2. 按逻辑顺序组织信息：
   - 从序列/一级结构信息入手
   - 再到功能注释和生物学作用
   - 包括调控关系和相互作用
   - 最后到实验背景和元数据
3. 保持科学准确性，使用统一命名规范（标准基因名、序列标识符等）。
4. 检查每对问答，确保逻辑一致且无幻觉。

################
-实体-
################
{entities}

################
-关系-
################
{relationships}
################
请直接在下方输出生成的问答对，不要复制任何示例，不要输出无关内容。
重要提示：请生成实际的问题和答案，不要使用占位符。不要包含尖括号或占位符文本（如 <问题1> 或 <答案1>）。
请直接按照以下格式输出生成的问题和答案：

问题： [你生成的实际问题]
答案： [你生成的实际答案]

问题： [你生成的实际问题]
答案： [你生成的实际答案]

"""

OMICS_QA_GENERATION_PROMPT = {
    "en": OMICS_QA_TEMPLATE_EN,
    "zh": OMICS_QA_TEMPLATE_ZH,
}
