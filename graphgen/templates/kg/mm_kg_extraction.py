# pylint: disable=C0301
TEMPLATE_EN: str = """You are an expert in multi-modal data analysis and knowledge graph construction. Your task is to extract named entities and relationships from a given multi-modal data chunk and its accompanying text.

-Objective-
Given a multi-modal data chunk (e.g., image, table, formula, etc. + accompanying text), construct a knowledge graph centered around the "central multi-modal entity":
- The central entity must be the image/table/formula itself (e.g., image-c71ef797e99af81047fbc7509609c765).
- Related entities and relationships must be extracted from the accompanying text.
- Only retain edges directly connected to the central entity, forming a star-shaped graph.
Use English as the output language.

-Steps-
1. Identify the unique central multi-modal entity and recognize all text entities directly related to the central entity from the accompanying text.
    For the central entity, extract the following information:
    - entity_name: Use the unique identifier of the data chunk (e.g., image-c71ef797e99af81047fbc7509609c765).
    - entity_type: Label according to the type of data chunk (image, table, formula, etc.).
    - entity_summary: A brief description of the content of the data chunk and its role in the accompanying text.
    For each entity recognized from the accompanying text, extract the following information:
    - entity_name: The name of the entity, capitalized
    - entity_type: One of the following types: [{entity_types}]
    - entity_summary: A comprehensive summary of the entity's attributes and activities
    Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>)

2. From the entities identified in Step 1, recognize all (source_entity, target_entity) pairs that are *obviously related* to each other.
    For each pair of related entities, extract the following information:
    - source_entity: The name of the source entity identified in Step 1
    - target_entity: The name of the target entity identified in Step 1
    - relationship_summary: Explain why you think the source entity and target entity are related to each other
    Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_summary>)
    
3. Return the output list of all entities and relationships identified in Steps 1 and 2 in English. Use **{record_delimiter}** as the list separator.

4. Upon completion, output {completion_delimiter}

################
-Example-
################
Multi-modal data chunk type: image
Multi-modal data chunk unique identifier: image-c71ef797e99af81047fbc7509609c765
Accompanying text: The Eiffel Tower is an iconic structure in Paris, France, designed by Gustave Eiffel and completed in 1889. It stands 324 meters tall and is one of the tallest structures in the world. The Eiffel Tower is located on the banks of the Seine River and attracts millions of visitors each year. It is not only an engineering marvel but also an important symbol of French culture.
################
Output:
("entity"{tuple_delimiter}"image-c71ef797e99af81047fbc7509609c765"{tuple_delimiter}"image"{tuple_delimiter}"This is an image showcasing the iconic structure in Paris, France, the Eiffel Tower, highlighting its full height of 324 meters along with the riverside scenery, symbolizing both engineering and cultural significance"){record_delimiter}
("entity"{tuple_delimiter}"Eiffel Tower"{tuple_delimiter}"landmark"{tuple_delimiter}"The Eiffel Tower is an iconic structure in Paris, France, designed by Gustave Eiffel and completed in 1889, standing 324 meters tall, located on the banks of the Seine River, attracting millions of visitors each year"){record_delimiter}
("entity"{tuple_delimiter}"Paris, France"{tuple_delimiter}"location"{tuple_delimiter}"Paris, France is the capital of France, known for its rich historical and cultural heritage and as the location of the Eiffel Tower"){record_delimiter}
("entity"{tuple_delimiter}"Gustave Eiffel"{tuple_delimiter}"person"{tuple_delimiter}"Gustave Eiffel is a renowned French engineer who designed and built the Eiffel Tower"){record_delimiter}
("entity"{tuple_delimiter}"Seine River"{tuple_delimiter}"location"{tuple_delimiter}"The Seine River is a major river flowing through Paris, France, with the Eiffel Tower located on its banks"){completion_delimiter}
("relationship"{tuple_delimiter}"image-c71ef797e99af81047fbc7509609c765"{tuple_delimiter}"Eiffel Tower"{tuple_delimiter}"The image showcases the iconic structure, the Eiffel Tower"){record_delimiter}
("relationship"{tuple_delimiter}"image-c71ef797e99af81047fbc7509609c765"{tuple_delimiter}"Paris, France"{tuple_delimiter}"The image's background is Paris, France, highlighting the geographical location of the Eiffel Tower"){record_delimiter}
("relationship"{tuple_delimiter}"image-c71ef797e99af81047fbc7509609c765"{tuple_delimiter}"Gustave Eiffel"{tuple_delimiter}"The Eiffel Tower in the image was designed by Gustave Eiffel"){record_delimiter}
("relationship"{tuple_delimiter}"image-c71ef797e99af81047fbc7509609c765"{tuple_delimiter}"Seine River"{tuple_delimiter}"The image showcases the scenery of the Eiffel Tower located on the banks of the Seine River"){completion_delimiter}
################

-Real Data-
Multi-modal data chunk type: {chunk_type}
Multi-modal data chunk unique identifier: {chunk_id}
Accompanying text: {chunk_text}
################
Output:
"""

TEMPLATE_ZH: str = """你是一个多模态数据分析和知识图谱构建专家。你的任务是从给定的多模态数据块及其伴随文本中抽取命名实体与关系。

-目标-
给定一个多模态数据块（例如图像、表格、公式等 + 伴随文本），构建以「中心多模态实体」为核心的知识图：
- 中心实体必须是图像/表格/公式本身（如 image-c71ef797e99af81047fbc7509609c765）。
- 相关实体和关系必须从伴随文本中抽取。
- 只保留与中心实体直接相连的边，形成星型图。
使用中文作为输出语言。

-步骤-
1. 确定唯一的中心多模态实体，从伴随文本中识别所有与中心实体直接相关的文本实体。
   对于中心实体，提取以下信息：
    - entity_name：使用数据块的唯一标识符（如 image-c71ef797e99af81047fbc7509609c765）。
    - entity_type：根据数据块类型（图像、表格、公式等）进行标注。
    - entity_summary：简要描述数据块的内容和其在伴随文本中的作用。
   对于从伴随文本中识别的每个实体，提取以下信息：
    - entity_name：实体的名称，首字母大写
    - entity_type：以下类型之一：[{entity_types}]
    - entity_summary：实体的属性与活动的全面总结
    将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>)

2. 从步骤1中识别的实体中，识别所有（源实体，目标实体）对，这些实体彼此之间*明显相关*。
   对于每对相关的实体，提取以下信息：
   - source_entity：步骤1中识别的源实体名称
   - target_entity：步骤1中识别的目标实体名称
   - relationship_summary：解释为什么你认为源实体和目标实体彼此相关
   将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_summary>)

3. 以中文返回步骤1和2中识别出的所有实体和关系的输出列表。使用**{record_delimiter}**作为列表分隔符。

4. 完成后，输出{completion_delimiter}

################
-示例-
################
多模态数据块类型：image
多模态数据块唯一标识符：image-c71ef797e99af81047fbc7509609c765
伴随文本：埃菲尔铁塔是法国巴黎的标志性结构，由古斯塔夫·埃菲尔设计并于1889年建成。它高324米，是世界上最高的建筑之一。埃菲尔铁塔位于塞纳河畔，吸引了数百万游客前来参观。它不仅是工程学的奇迹，也是法国文化的重要象征。
################
输出：
("entity"{tuple_delimiter}"image-c71ef797e99af81047fbc7509609c765"{tuple_delimiter}"image"{tuple_delimiter}"这是一张展示法国巴黎标志性建筑的图像，主体为埃菲尔铁塔，呈现其324米高度的全貌与河畔景观，具有工程与文化双重象征意义"){record_delimiter}
("entity"{tuple_delimiter}"埃菲尔铁塔"{tuple_delimiter}"landmark"{tuple_delimiter}"埃菲尔铁塔是法国巴黎的标志性结构，由古斯塔夫·埃菲尔设计并于1889年建成，高324米，是世界上最高的建筑之一，位于塞纳河畔，吸引了数百万游客前来参观"){record_delimiter}
("entity"{tuple_delimiter}"法国巴黎"{tuple_delimiter}"location"{tuple_delimiter}"法国巴黎是法国的首都，以其丰富的历史文化遗产和作为埃菲尔铁塔所在地而闻名"){record_delimiter}
("entity"{tuple_delimiter}"古斯塔夫·埃菲尔"{tuple_delimiter}"person"{tuple_delimiter}"古斯塔夫·埃菲尔是法国著名的工程师，设计并建造了埃菲尔铁塔"){record_delimiter}
("entity"{tuple_delimiter}"塞纳河"{tuple_delimiter}"location"{tuple_delimiter}"塞纳河是流经法国巴黎的重要河流，埃菲尔铁塔位于其畔"){completion_delimiter}
("relationship"{tuple_delimiter}"image-c71ef797e99af81047fbc7509609c765"{tuple_delimiter}"埃菲尔铁塔"{tuple_delimiter}"图像展示了埃菲尔铁塔这一标志性建筑"){record_delimiter}
("relationship"{tuple_delimiter}"image-c71ef797e99af81047fbc7509609c765"{tuple_delimiter}"法国巴黎"{tuple_delimiter}"图像背景为法国巴黎，突显了埃菲尔铁塔的地理位置"){record_delimiter}
("relationship"{tuple_delimiter}"image-c71ef797e99af81047fbc7509609c765"{tuple_delimiter}"古斯塔夫·埃菲尔"{tuple_delimiter}"图像中的埃菲尔铁塔是由古斯塔夫·埃菲尔设计的"){record_delimiter}
("relationship"{tuple_delimiter}"image-c71ef797e99af81047fbc7509609c765"{tuple_delimiter}"塞纳河"{tuple_delimiter}"图像展示了埃菲尔铁塔位于塞纳河畔的景观"){completion_delimiter}
################

-真实数据-
多模态数据块类型： {chunk_type}
多模态数据块唯一标识符： {chunk_id}
伴随文本： {chunk_text}
################
输出：
"""


MMKG_EXTRACTION_PROMPT: dict = {
    "en": TEMPLATE_EN,
    "zh": TEMPLATE_ZH,
    "FORMAT": {
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>",
        "entity_types": "concept, date, location, keyword, organization, person, event, work, nature, artificial, \
science, technology, mission, gene",
    },
}
