# pylint: disable=C0301
PROTEIN_QA_TEMPLATE_EN: str = """You are a senior computational biologist specializing in structural bioinformatics. Your task is to generate logically coherent, verifiable and non-hallucinated question-answer pairs for the given protein sample described by the provided ENTITIES and RELATIONSHIPS.
Use English as the output language.

---Objectives---
Create multiple sets of protein-centric QA pairs that satisfy the following:
1. Only ask about objectively existing facts in the provided data (e.g., residue numbers, secondary-structure elements, binding sites, catalytic residues, domain boundaries, metal ions, experimental method, etc.). Avoid subjective or speculative questions.
2. Ensure that each question has a single, clear and verifiable answer that can be directly confirmed from the given entities/relationships.
3. Questions should cover diverse aspects: sequence, structure, function, interactions, dynamics, thermodynamics, experimental annotations, etc.
4. Avoid repetitive questions; each question must be unique and meaningful.
5. Use concise, unambiguous language; do not invent information beyond the provided data.

---Instructions---
1. Carefully analyse the supplied ENTITIES and RELATIONSHIPS to identify:
   - macromolecular entities (protein chains, domains, motifs, ligands, ions)
   - structural attributes (helices, strands, loops, cis-peptide, disulfide bonds)
   - functional annotations (active site, binding site, post-translational modification)
   - experimental metadata (PDB ID, resolution, method, temperature, pH)
   - causal or sequential relations (e.g., "Mg2+ binding stabilises loop L3")
2. Organise information logically:
   - start with sequence/primary structure
   - proceed to secondary/tertiary structure
   - end with function, mechanism, and experimental context
3. Maintain scientific accuracy and consistent nomenclature (standard residue numbering, atom names, etc.).
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

Question: <Question1>
Answer: <Answer1>

Question: <Question2>
Answer: <Answer2>

"""

PROTEIN_QA_TEMPLATE_ZH: str = """你是一位资深的结构生物信息学计算生物学家。你的任务是根据下述提供的实体与关系，为给定的蛋白质样本生成逻辑连贯、可验证、无幻觉的中英双语问答对（这里仅输出中文）。
使用中文作为输出语言。

---目标---
创建多组以蛋白质为中心的问答对，满足：
1. 仅询问数据中客观存在的事实（如残基编号、二级结构元件、结合位点、催化残基、结构域边界、金属离子、实验方法等），避免主观或推测性问题。
2. 每个问题必须有单一、明确且可直接验证的答案，答案必须能从给定实体/关系中直接确认。
3. 问题需覆盖：序列、结构、功能、相互作用、动力学、热力学、实验注释等多个维度，确保多样性与全面性。
4. 避免重复提问，每个问题都独特且有意义。
5. 语言简洁、无歧义，严禁编造超出给定数据的信息。

---说明---
1. 仔细分析提供的实体与关系，识别：
   - 大分子实体（蛋白链、结构域、模体、配体、离子）
   - 结构属性（螺旋、折叠、环区、顺式肽键、二硫键）
   - 功能注释（活性位点、结合位点、翻译后修饰）
   - 实验元数据（PDB 编号、分辨率、方法、温度、pH）
   - 因果或顺序关系（如“Mg²⁺ 结合稳定了环区 L3”）
2. 按逻辑顺序组织信息：
   - 从序列/一级结构入手
   - 再到二级/三级结构
   - 最后到功能、机制及实验背景
3. 保持科学准确性，使用统一命名规范（标准残基编号、原子名等）。
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

问题： <问题1>
答案： <答案1>

问题： <问题2>
答案： <答案2>

"""

PROTEIN_QA_GENERATION_PROMPT = {
    "en": PROTEIN_QA_TEMPLATE_EN,
    "zh": PROTEIN_QA_TEMPLATE_ZH,
}
