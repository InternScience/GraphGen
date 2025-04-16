TEMPLATE_CONTEXT_EN: str = """---Role---

You are an NLP expert responsible for generating a logically structured and coherent rephrased version of the TEXT based on ENTITIES and RELATIONSHIPS provided below. You may refer to the original text to assist in generating the rephrased version, but ensure that the final output text meets the requirements.
Use {language} as output language.

---Goal---
To generate a version of the text that is rephrased and conveys the same meaning as the original entity and relationship descriptions, while:
1. Following a clear logical flow and structure
2. Establishing proper cause-and-effect relationships
3. Ensuring temporal and sequential consistency
4. Creating smooth transitions between ideas using conjunctions and appropriate linking words like "firstly," "however," "therefore," etc.

---Instructions---
1. Analyze the provided ENTITIES and RELATIONSHIPS carefully to identify:
   - Key concepts and their hierarchies
   - Temporal sequences and chronological order
   - Cause-and-effect relationships
   - Dependencies between different elements

2. Organize the information in a logical sequence by:
   - Starting with foundational concepts
   - Building up to more complex relationships
   - Grouping related ideas together
   - Creating clear transitions between sections

3. Rephrase the text while maintaining:
   - Logical flow and progression
   - Clear connections between ideas
   - Proper context and background
   - Coherent narrative structure

4. Review and refine the text to ensure:
   - Logical consistency throughout
   - Clear cause-and-effect relationships

################
-ORIGINAL TEXT-
################
{original_text}

################
-ENTITIES-
################
{entities}

################
-RELATIONSHIPS-
################
{relationships}

"""

TEMPLATE_CONTEXT_ZH: str = """---角色---

你是一位NLP专家，负责根据下面提供的实体和关系生成逻辑结构清晰且连贯的文本重述版本。你可以参考原始文本辅助生成，但需要确保最终输出的文本符合要求。
使用{language}作为输出语言。

---目标---

生成文本的重述版本，使其传达与原始实体和关系描述相同的含义，同时：
1. 遵循清晰的逻辑流和结构
2. 建立适当的因果关系
3. 确保时间和顺序的一致性
4. 使用连词和适当的连接词(如"首先"、"然而"、"因此"等)创造流畅的过渡

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
3. 重述文本时保持：
    - 逻辑流畅
    - 概念之间的清晰联系
    - 适当的上下文和背景
    - 连贯的叙述结构
4. 检查和完善文本以确保：
    - 整体逻辑一致性
    - 清晰的因果关系

################
-原始文本-
################
{original_text}

################
-实体-
################
{entities}

################
-关系-
################
{relationships}

"""

TEMPLATE_EN: str = """---Role---

You are an NLP expert responsible for generating a logically structured and coherent rephrased version of the TEXT based on ENTITIES and RELATIONSHIPS provided below.
Use {language} as output language.

---Goal---
To generate a version of the text that is rephrased and conveys the same meaning as the original entity and relationship descriptions, while:
1. Following a clear logical flow and structure
2. Establishing proper cause-and-effect relationships
3. Ensuring temporal and sequential consistency
4. Creating smooth transitions between ideas using conjunctions and appropriate linking words like "firstly," "however," "therefore," etc.

---Instructions---
1. Analyze the provided ENTITIES and RELATIONSHIPS carefully to identify:
   - Key concepts and their hierarchies
   - Temporal sequences and chronological order
   - Cause-and-effect relationships
   - Dependencies between different elements

2. Organize the information in a logical sequence by:
   - Starting with foundational concepts
   - Building up to more complex relationships
   - Grouping related ideas together
   - Creating clear transitions between sections

3. Rephrase the text while maintaining:
   - Logical flow and progression
   - Clear connections between ideas
   - Proper context and background
   - Coherent narrative structure

4. Review and refine the text to ensure:
   - Logical consistency throughout
   - Clear cause-and-effect relationships

################
-ENTITIES-
################
{entities}

################
-RELATIONSHIPS-
################
{relationships}

"""

TEMPLATE_ZH: str = """---角色---

你是一位NLP专家，负责根据下面提供的实体和关系生成逻辑结构清晰且连贯的文本重述版本。
使用{language}作为输出语言。

---目标---

生成文本的重述版本，使其传达与原始实体和关系描述相同的含义，同时：
1. 遵循清晰的逻辑流和结构
2. 建立适当的因果关系
3. 确保时间和顺序的一致性
4. 使用连词和适当的连接词(如"首先"、"然而"、"因此"等)创造流畅的过渡

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
3. 重述文本时保持：
    - 逻辑流畅
    - 概念之间的清晰联系
    - 适当的上下文和背景
    - 连贯的叙述结构
4. 检查和完善文本以确保：
    - 整体逻辑一致性
    - 清晰的因果关系

################
-实体-
################
{entities}

################
-关系-
################
{relationships}

"""

EASY_REQUIREMENT_EN = """
---Requirements---
- Requires a concise and straightforward summary, focusing on core meaning.
- Uses simple language, avoiding complex sentence structures.
- Does not need excessive details or examples; just the basic concepts and relationships.

################
Please directly output the coherent rephrased text below, without any additional content.

Rephrased Text:
"""

EASY_REQUIREMENT_ZH = """
---要求---
- 要求简洁明了，主要传达核心意思。
- 使用简单的语言，避免复杂的句子结构。
- 不需要过多的细节或示例，只需基本概念和关系。

################
请在下方直接输出连贯的重述文本，不要输出任何额外的内容。

重述文本:
"""

MEDIUM_REQUIREMENT_ZH = """
################
请在下方直接输出连贯的重述文本，不要输出任何额外的内容。

重述文本:
"""

MEDIUM_REQUIREMENT_EN = """
################
Please directly output the coherent rephrased text below, without any additional content.

Rephrased Text:
"""

HARD_REQUIREMENT_EN = """
---Requirements---
- Requires an in-depth exploration of complex relationships and nuances.
- Includes detailed background information, emphasizing logical consistency and complexity.

################
Please directly output the coherent rephrased text below, without any additional content.

Rephrased Text:
"""

HARD_REQUIREMENT_ZH = """
---要求---
- 需要深入探讨复杂的关系和细微差别。
- 包括详细的背景信息，强调逻辑一致性和复杂性。

################
请在下方直接输出连贯的重述文本，不要输出任何额外的内容。

重述文本:
"""

ANSWER_REPHRASING_PROMPT= {
    "easy": {
        "English": {
            "TEMPLATE": TEMPLATE_EN + EASY_REQUIREMENT_EN,
            "CONTEXT_TEMPLATE": TEMPLATE_CONTEXT_EN + EASY_REQUIREMENT_EN
        },
        "Chinese": {
            "TEMPLATE": TEMPLATE_ZH + EASY_REQUIREMENT_ZH,
            "CONTEXT_TEMPLATE": TEMPLATE_CONTEXT_ZH + EASY_REQUIREMENT_ZH
        }
    },
    "medium": {
        "English": {
            "TEMPLATE": TEMPLATE_EN + MEDIUM_REQUIREMENT_EN,
            "CONTEXT_TEMPLATE": TEMPLATE_CONTEXT_EN + MEDIUM_REQUIREMENT_EN
        },
        "Chinese": {
            "TEMPLATE": TEMPLATE_ZH + MEDIUM_REQUIREMENT_ZH,
            "CONTEXT_TEMPLATE": TEMPLATE_CONTEXT_ZH + MEDIUM_REQUIREMENT_ZH
        }
    },
    "hard": {
        "English": {
            "TEMPLATE": TEMPLATE_EN + HARD_REQUIREMENT_EN,
            "CONTEXT_TEMPLATE": TEMPLATE_CONTEXT_EN + HARD_REQUIREMENT_EN
        },
        "Chinese": {
            "TEMPLATE": TEMPLATE_ZH + HARD_REQUIREMENT_ZH,
            "CONTEXT_TEMPLATE": TEMPLATE_CONTEXT_ZH + HARD_REQUIREMENT_ZH
        }
    }
}
