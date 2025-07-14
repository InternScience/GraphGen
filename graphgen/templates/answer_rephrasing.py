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

你是一位NLP专家，负责根据下面提供的实体和关系生成逻辑结构清晰且连贯的问答。
使用{language}作为输出语言。

---目标---
按照下述步骤生成文本问答：
1. 仔细分析提供的实体和关系，以识别：
    - 关键概念及其层级关系
    - 时间序列和时间顺序
    - 因果关系
    - 不同元素之间的依赖关系
2. 根据提供的信息提出一个问题
3. 按如下逻辑顺序组织回答：
    - 问题分析
    - 知识点分析
    - 产生答案内容
    - 检查上述步骤答案的合理性
4. 重述文本时保持：
    - 逻辑流畅
    - 概念之间的清晰联系
    - 适当的上下文和背景
    - 连贯的叙述结构

---示例---
问题：基于你所有知识、也包括知识图谱，你对于基因LOC_Os12g03040有哪些认识？
答案：
*问题分析*
用户要求我对基因 LOC_Os12g03040 进行分析，并要求我凭借自身所掌握的全部知识以及知识图谱来加以总结。显然，用户期望获取到该基因全面且详尽的信息。首先，我需要明确该基因所属的物种，迅速锁定所需的准确信息范畴。随后，回顾自身知识储备，系统梳理出此基因的相关信息，包括其基本信息以及已知的生物学功能。

*知识点分析*
首先，基因标识符 LOC_Os12g03040 符合MSU数据库的命名方式，LOC_Os 通常代表水稻（Oryza sativa），因为 Os 是水稻的拉丁学名缩写。其中12代表染色体编号，g代表基因，03040表示基因在染色体上的顺序。然而，由于不同的数据库在基因标识符的设定上存在差异，为了更好地满足用户需求，我有必要向用户提供其他常用的水稻公共数据库，如 RAP 和 RiceSuperPIRdb 中对应的基因标识符信息。接下来，我需要确定这个水稻基因详尽且具体的信息，这不仅涵盖其命名信息，还包括其在基因组上的精确物理定位、核苷酸序列的具体长度、编码蛋白质的各类参数，以及全面且准确的基因本体（ontology）注释信息等多个关键方面。此外，我会借助知识图谱，梳理该基因与其他基因的相互作用，明确其参与的代谢通路 。
然后，我会确定该基因具体的生物学功能，判断其是否参与转录调控、代谢、发育或应激反应等特定生物过程。查阅相关研究论文和数据库记录，重点关注 RNA 干扰、过表达株系表型分析等功能验证实验。

*产生答案内容*
一、基因LOC_Os12g03040的基本信息
1.1 数据库标识符
- MSU: LOC_Os12g03040
- RAP: Os12g0123700
- T2TID_AGIS: AGIS_Os12g002000
1.2 命名信息
- 基因名称: NAC转录因子基因 / NAC DOMAIN-CONTAINING PROTEIN 131
- 基因符号: ONAC131 / NAC131
1.3 基因组特征
- 染色体定位: 12号染色体
- CDS坐标（5'→3'）: 1,139,026 - 1,141,362
- 核苷酸长度: 1,182 bp
- 蛋白预测参数
  - 氨基酸长度: 394 aa
  - 分子量: 43,589.68 Da
  - 等电点: 6.64
1.4 本体分类
性状本体（Trait Ontology）
- TO:0000074 稻瘟病抗性
- TO:0000148 病毒病抗性
- TO:0000058 除草剂敏感性
- TO:0000020 水稻黑条矮缩病毒抗性
- TO:0000386 水稻齿叶矮缩病毒抗性
基因本体（GO Term）
分子功能
- GO:0003700 DNA结合转录因子活性
- GO:0003677 DNA结合
生物过程
- GO:0009751 水杨酸应答
- GO:0009753 茉莉酸应答
- GO:0009723 乙烯应答
- GO:0050832 抗真菌反应
- GO:0009635 除草剂应答
- GO:0051607 病毒防御反应
- GO:0006355 转录调控（DNA依赖）
- GO:0009058 生物合成过程
- GO:0006139 核酸代谢过程
- GO:0007275 多细胞生物发育
二、基因LOC_Os12g03040的生物学功能
2.1 转录调控与抗病机制
- 作为NAC家族转录因子，具有DNA结合和转录激活活性
- 在稻瘟病菌侵染早期，抗病品种中的表达水平显著高于感病品种
- 响应水杨酸、茉莉酸甲酯和ACC（1-氨基环丙烷-1-羧酸）诱导
- 通过调控防御相关基因表达正向调控水稻对稻瘟病菌的抗性
2.2 籽粒发育调控
- 全基因组关联分析（GWAS）显示与粒长显著相关
2.3 逆境应答机制
- 转录组数据表明干旱胁迫下表达显著上调

*检查上述步骤答案的合理性*
问题分析部分
- 准确解读了用户需求，明确要利用自身知识和知识图谱进行总结，并确定了先明确物种、再梳理基因相关信息的思路，为后续分析奠定基础，逻辑清晰。
知识点分析部分
- 详细解释了基因标识符的命名规则，帮助理解基因在 MSU 数据库中的标识逻辑。
- 意识到不同数据库标识符差异并计划提供其他数据库对应信息，满足用户对全面信息的需求。
- 明确列举了需要确定的基因具体信息内容，包括命名、基因组定位、核苷酸长度、蛋白质参数、基因本体注释等，涵盖了基因分析的重要方面。
- 规划借助知识图谱梳理基因相互作用及代谢通路，这是深入分析基因功能和作用机制的有效手段；同时计划通过查阅论文和数据库记录，关注功能验证实验来确定生物学功能，方法科学合理。
产生答案内容部分
- 答案内容结构清晰，分为基本信息和生物学功能两大部分，每部分又细分多个小项，便于用户查看。
- 在基本信息中，提供了多个数据库的标识符，丰富了信息维度；对命名、基因组特征、本体分类等信息的呈现详细且具体，涵盖了之前知识点分析中提到的大部分关键内容。
- 在生物学功能部分，从转录调控与抗病、籽粒发育调控、逆境应答机制等多个方面阐述，有具体的研究结论支撑，如 “在稻瘟病菌侵染早期，抗病品种中的表达水平显著高于感病品种”，使功能分析具有说服力。
---示例---

请你根据上述示例和要求，结合下面给出的信息，生成文本问答。请你不要直接列出每个步骤的标题，但请确保回答按照问题分析、知识点分析、产生答案内容、检查上述步骤答案的合理性等步骤进行。
请确保第一行是问题，后面是回答内容，不要添加任何额外的内容。

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
请在下方直接输出文本问答，不要输出任何额外的内容。

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
        # "English": {
        #     "TEMPLATE": TEMPLATE_EN + MEDIUM_REQUIREMENT_EN,
        #     "CONTEXT_TEMPLATE": TEMPLATE_CONTEXT_EN + MEDIUM_REQUIREMENT_EN
        # },
        "Chinese": {
            "TEMPLATE": TEMPLATE_ZH + MEDIUM_REQUIREMENT_ZH,
            "CONTEXT_TEMPLATE": TEMPLATE_CONTEXT_ZH + MEDIUM_REQUIREMENT_ZH
        },
        "English": {
            "TEMPLATE": TEMPLATE_ZH + MEDIUM_REQUIREMENT_ZH,
            "CONTEXT_TEMPLATE": TEMPLATE_CONTEXT_ZH + MEDIUM_REQUIREMENT_ZH
        },
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
