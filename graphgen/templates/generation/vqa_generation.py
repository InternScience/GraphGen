# pylint: disable=C0301
TEMPLATE_EN: str = """You are a senior VQA data engineer. Your task is to generate logically coherent, verifiable and non-hallucinated question-answer pairs for the given multi-modal samples.
Use English as the output language.

---Objectives---
Create multiple sets of VQA question-answer pairs that satisfy the following:
1. Only ask about objectively existing facts in the given data, avoiding subjective or ambiguous questions.
2. Ensure that each question has a clear and verifiable answer, avoiding questions with no answer or uncertainty.
3. Questions should cover various aspects of both image and text content, ensuring diversity and comprehensiveness.
4. Avoid repetitive questions, ensuring that each question is unique and meaningful.
5. Use clear and concise language, avoiding complex or ambiguous wording.
6. Prioritize high training value for VLM: include entity recognition, relation reasoning, numerical reading, and cross-modal grounding.
7. Treat every `Evidence:` line as hard grounding. Do not ask or answer anything unsupported by those evidence spans.

---DRAM-Centric Guidance---
If the sample is related to memory systems (e.g., DRAM, SRAM, HBM, LPDDR, GDDR, DIMM, timing parameters, channels, banks, ranks), prioritize these question styles:
- Structure and topology: module/channel/bank/rank relationships
- Timing and constraints: tRCD, tRP, tRAS, CAS latency, refresh, frequency
- Performance and capacity: bandwidth, latency, data rate, capacity, power
- Comparison and trade-off: differences between generations/standards/configurations
- Cross-modal evidence grounding: answers must be directly supported by image/text entities and relationships
- Evidence-aware generation: prefer questions whose answers can be justified by one or more explicit `Evidence:` spans

---Instructions---
1. Carefully analyze the provided entities and relationships to identify:
    - Key concepts and their hierarchical relationships
    - Temporal sequences and time order
    - Cause-and-effect relationships
    - Dependencies between different elements
2. Organize the information into a logical sequence by:
    - Starting with foundational concepts
    - Gradually building up to more complex relationships
    - Grouping related ideas together
    - Creating clear transitions between sections
3. Maintain the following when generating question-answer pairs:
    - Logical flow
    - Clear connections between concepts
    - Appropriate context and background
    - Coherent narrative structure
4. Review and refine the question-answer pairs to ensure:
    - Overall logical consistency
    - Clear cause-and-effect relationships
    - Every answer can be justified with the provided evidence snippets
5. Generate 6 to 10 QA pairs, and keep balanced difficulty:
    - 30% factual extraction (easy)
    - 50% relational/numerical reasoning (medium)
    - 20% multi-step inference (hard)

################
-Entities-
################
{entities}
################
-Relationships-
################
{relationships}
################

Please directly output the generated questions and answers, do not directly copy the example questions and answers, and do not provide irrelevant information.

Here is the response format you should follow:
<question>question1</question>
<answer>answer1</answer>
<question>question2</question>
<answer>answer2</answer>

Output:
"""

TEMPLATE_ZH: str = """---角色---
你是一位资深 VQA 数据工程师。你需要为给定的多模态样本生成逻辑连贯、可验证、无幻觉的问答对。
使用中文作为输出语言。

---目标---
创建多组 VQA 问答对，满足：
1. 仅询问给定数据中客观存在的事实，避免主观或模糊的问题。
2. 确保每个问题都有明确且可验证的答案，避免无答案或不确定的问题。
3. 问题应涵盖图像和文本内容的各个方面，确保多样性和全面性。
4. 避免重复问题，确保每个问题都是独特且有意义的。
5. 使用清晰简洁的语言，避免复杂或含糊的措辞。
6. 优先保证对 VLM 训练有效：覆盖实体识别、关系推理、数值读取和跨模态对齐。
7. 将每条 `Evidence:` 视为硬约束；凡是没有证据支撑的信息都不要提问或作答。

---DRAM 场景增强---
当样本与存储器系统相关（如 DRAM、SRAM、HBM、LPDDR、GDDR、DIMM、时序参数、channel、bank、rank）时，优先生成：
- 结构/拓扑类问题：模块、通道、bank、rank 之间的关系
- 时序/约束类问题：tRCD、tRP、tRAS、CAS latency、refresh、频率
- 性能/容量类问题：带宽、延迟、数据速率、容量、功耗
- 比较/权衡类问题：不同代际、标准或配置差异
- 跨模态证据问题：答案必须能在图像/文本实体与关系中直接定位依据
- 证据感知生成：优先生成能够被一条或多条 `Evidence:` 直接支撑的问题

---说明---
1. 仔细分析提供的实体和关系，以识别：
    - 关键概念及其层级关系
    - 时间序列和时间顺序
    - 因果关系
    - 不同元素之间的依赖关系
2. 通过以下方式将信息组织成逻辑顺序：
    - 从基础概念开始
    - 逐步建立更复杂的关系
    - 将相关的想法分组在一起
    - 在各部分之间创建清晰的过渡
3. 生成问答对时保持：
    - 逻辑流畅
    - 概念之间的清晰联系
    - 适当的上下文和背景
    - 连贯的叙述结构
4. 检查和完善问答对以确保：
    - 整体逻辑一致性
    - 清晰的因果关系
    - 每个答案都能被提供的证据片段直接支撑
5. 输出 6 到 10 组问答，并保持难度结构：
    - 30% 事实抽取（简单）
    - 50% 关系/数值推理（中等）
    - 20% 多步推断（困难）

################
-实体-
################
{entities}

################
-关系-
################
{relationships}
################

请直接输出生成的问题和答案，不要直接复制示例问题和答案，也不要提供无关信息。
以下是你应遵循的响应格式：
<question>question1</question>
<answer>answer1</answer>
<question>question2</question>
<answer>answer2</answer>

输出：
"""

VQA_GENERATION_PROMPT = {"en": TEMPLATE_EN, "zh": TEMPLATE_ZH}
