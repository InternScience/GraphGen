# pylint: disable=C0301
TEMPLATE_EN: str = """You are an expert in multi-modal data analysis and knowledge graph construction. Your task is to extract named entities and relationships from a given multi-modal data chunk and its accompanying text.

-Objective-
Given a multi-modal data chunk (e.g., image, table, formula, etc. + accompanying text), construct a knowledge graph centered around the "central multi-modal entity":
- The central entity must be the image/table/formula itself (e.g., image-c71ef797e99af81047fbc7509609c765).
- Related entities and relationships must be extracted from the accompanying text.
- Only retain edges directly connected to the central entity, forming a star-shaped graph.
Use English as the output language.
This extraction should also work well for semiconductor memory figures/tables/formulas and should prefer concrete technical entities over generic prose terms.

-Steps-
1. Identify the unique central multi-modal entity and recognize all text entities directly related to the central entity from the accompanying text.
    For the central entity, extract the following information:
    - entity_name: Use the unique identifier of the data chunk (e.g., image-c71ef797e99af81047fbc7509609c765).
    - entity_type: Label according to the type of data chunk (image, table, formula, etc.).
    - entity_summary: A brief description of the content of the data chunk and its role in the accompanying text.
    - evidence_span: A short, verbatim quote from the accompanying text that grounds this entity.
    For each entity recognized from the accompanying text, extract the following information:
    - entity_name: The name of the entity, capitalized
    - entity_type: One of the following types: [{entity_types}]
    - entity_summary: A comprehensive summary of the entity's attributes, role, or measured/specification values
    - evidence_span: A short, verbatim quote from the accompanying text that grounds this entity
    - For semiconductor memory content, prefer concrete technical entities (memory product/family, interface standard, component, substructure, timing parameter, performance or power metric, operating condition, process technology, material, signal, test method, failure mode, organization).
    - Avoid generic nouns such as "system", "performance", "method", or units alone unless they are explicitly used as named technical entities.
    Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>{tuple_delimiter}<evidence_span>)

2. From the entities identified in Step 1, recognize all (source_entity, target_entity) pairs that are *obviously related* to each other.
    For each pair of related entities, extract the following information:
    - source_entity: The name of the source entity identified in Step 1
    - target_entity: The name of the target entity identified in Step 1
    - relation_type: one type from [{relation_types}]
    - relationship_summary: Explain why you think the source entity and target entity are related to each other
    - evidence_span: A short, verbatim quote from the accompanying text that grounds this relationship
    - confidence: confidence score between 0 and 1
    - Only keep explicit, technically meaningful relations that are directly grounded in the accompanying text.
    Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relation_type>{tuple_delimiter}<relationship_summary>{tuple_delimiter}<evidence_span>{tuple_delimiter}<confidence>)

3. Return the output list of all entities and relationships identified in Steps 1 and 2 in English. Use **{record_delimiter}** as the list separator.

4. Upon completion, output {completion_delimiter}

################
-Example-
################
Multi-modal data chunk type: table
Multi-modal data chunk unique identifier: table-hbm3-spec-01
Accompanying text: Table 2 compares HBM3 and HBM2E. HBM3 reaches 819 GB/s stack bandwidth at 6.4 Gb/s per pin, while HBM2E reaches 460 GB/s. The HBM3 stack uses 8 DRAM dies connected to a base die through TSVs. Higher temperature increases refresh overhead and standby power.
################
Output:
("entity"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"table"{tuple_delimiter}"This table summarizes comparative HBM memory specifications, including bandwidth, per-pin rate, stack structure, TSV interconnect, and temperature-related overhead/power trends."{tuple_delimiter}"Table 2 compares HBM3 and HBM2E."){record_delimiter}
("entity"{tuple_delimiter}"HBM3"{tuple_delimiter}"memory_product"{tuple_delimiter}"HBM3 is a high-bandwidth memory generation whose stack bandwidth, per-pin rate, stack composition, and thermal behavior are described in the accompanying text."{tuple_delimiter}"HBM3 reaches 819 GB/s stack bandwidth at 6.4 Gb/s per pin"){record_delimiter}
("entity"{tuple_delimiter}"HBM2E"{tuple_delimiter}"memory_product"{tuple_delimiter}"HBM2E is the comparison baseline memory generation in the table, with 460 GB/s bandwidth."{tuple_delimiter}"HBM2E reaches 460 GB/s"){record_delimiter}
("entity"{tuple_delimiter}"819 GB/s"{tuple_delimiter}"performance_metric"{tuple_delimiter}"819 GB/s is the stack bandwidth specified for HBM3."{tuple_delimiter}"819 GB/s stack bandwidth"){record_delimiter}
("entity"{tuple_delimiter}"6.4 Gb/s per pin"{tuple_delimiter}"performance_metric"{tuple_delimiter}"6.4 Gb/s per pin is the signaling/data-rate condition associated with the HBM3 bandwidth figure."{tuple_delimiter}"at 6.4 Gb/s per pin"){record_delimiter}
("entity"{tuple_delimiter}"8 DRAM dies"{tuple_delimiter}"substructure"{tuple_delimiter}"8 DRAM dies are structural elements of the HBM3 stack."{tuple_delimiter}"uses 8 DRAM dies"){record_delimiter}
("entity"{tuple_delimiter}"base die"{tuple_delimiter}"component"{tuple_delimiter}"The base die is a component in the HBM3 stack connected with DRAM dies through TSVs."{tuple_delimiter}"connected to a base die through TSVs"){record_delimiter}
("entity"{tuple_delimiter}"TSVs"{tuple_delimiter}"component"{tuple_delimiter}"TSVs are vertical interconnect structures used between the HBM3 DRAM dies and base die."{tuple_delimiter}"through TSVs"){record_delimiter}
("entity"{tuple_delimiter}"Higher temperature"{tuple_delimiter}"operating_condition"{tuple_delimiter}"Higher temperature is an operating condition that worsens refresh overhead and standby power according to the text."{tuple_delimiter}"Higher temperature increases refresh overhead and standby power"){record_delimiter}
("entity"{tuple_delimiter}"refresh overhead"{tuple_delimiter}"performance_metric"{tuple_delimiter}"Refresh overhead is the overhead metric that increases at higher temperature."{tuple_delimiter}"increases refresh overhead"){record_delimiter}
("entity"{tuple_delimiter}"standby power"{tuple_delimiter}"power_metric"{tuple_delimiter}"Standby power is the power metric that increases at higher temperature."{tuple_delimiter}"standby power"){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"HBM3"{tuple_delimiter}"specification_of"{tuple_delimiter}"The table includes specification information about HBM3."{tuple_delimiter}"Table 2 compares HBM3 and HBM2E"{tuple_delimiter}0.96){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"HBM2E"{tuple_delimiter}"specification_of"{tuple_delimiter}"The table includes specification information about HBM2E."{tuple_delimiter}"Table 2 compares HBM3 and HBM2E"{tuple_delimiter}0.96){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"819 GB/s"{tuple_delimiter}"has_bandwidth"{tuple_delimiter}"The table reports 819 GB/s as a bandwidth metric in the HBM3 comparison."{tuple_delimiter}"HBM3 reaches 819 GB/s stack bandwidth"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"6.4 Gb/s per pin"{tuple_delimiter}"measured_by"{tuple_delimiter}"The table reports the HBM3 bandwidth figure under a 6.4 Gb/s per pin condition."{tuple_delimiter}"at 6.4 Gb/s per pin"{tuple_delimiter}0.9){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"8 DRAM dies"{tuple_delimiter}"part_of"{tuple_delimiter}"The table text states that the HBM3 stack uses 8 DRAM dies."{tuple_delimiter}"uses 8 DRAM dies"{tuple_delimiter}0.95){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"TSVs"{tuple_delimiter}"connected_to"{tuple_delimiter}"The table text describes TSV-based interconnect as part of the stack structure."{tuple_delimiter}"through TSVs"{tuple_delimiter}0.92){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"Higher temperature"{tuple_delimiter}"impacts"{tuple_delimiter}"The table text states that higher temperature changes overhead and power behavior."{tuple_delimiter}"Higher temperature increases refresh overhead and standby power"{tuple_delimiter}0.91){completion_delimiter}
################################

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
该提示也应适用于半导体存储器相关图/表/公式，优先抽取具体技术实体，而不是泛化叙述词。

-步骤-
1. 确定唯一的中心多模态实体，从伴随文本中识别所有与中心实体直接相关的文本实体。
   对于中心实体，提取以下信息：
    - entity_name：使用数据块的唯一标识符（如 image-c71ef797e99af81047fbc7509609c765）。
    - entity_type：根据数据块类型（图像、表格、公式等）进行标注。
    - entity_summary：简要描述数据块的内容和其在伴随文本中的作用。
    - evidence_span：用于支撑该实体的伴随文本原文短句。
   对于从伴随文本中识别的每个实体，提取以下信息：
    - entity_name：实体的名称，首字母大写
    - entity_type：以下类型之一：[{entity_types}]
    - entity_summary：实体的属性、作用或规格/测量值的全面总结
    - evidence_span：用于支撑该实体的伴随文本原文短句
    - 对于存储器技术内容，优先抽取具体技术实体（存储器产品/家族、标准接口、组件、子结构、时序参数、性能或功耗指标、运行条件、工艺技术、材料、信号、测试方法、失效模式、组织机构）。
    - 避免抽取“系统”“性能”“方法”等泛化名词，或仅包含单位的片段，除非文本明确将其作为命名技术实体使用。
    将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>{tuple_delimiter}<evidence_span>)

2. 从步骤1中识别的实体中，识别所有（源实体，目标实体）对，这些实体彼此之间*明显相关*。
   对于每对相关的实体，提取以下信息：
   - source_entity：步骤1中识别的源实体名称
   - target_entity：步骤1中识别的目标实体名称
   - relation_type：从[{relation_types}]中选择一个关系类型
   - relationship_summary：解释为什么你认为源实体和目标实体彼此相关
   - evidence_span：用于支撑该关系的伴随文本原文短句
   - confidence：0到1之间的置信度
   - 只保留伴随文本中直接明确、且具有技术含义的关系。
   将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relation_type>{tuple_delimiter}<relationship_summary>{tuple_delimiter}<evidence_span>{tuple_delimiter}<confidence>)

3. 以中文返回步骤1和2中识别出的所有实体和关系的输出列表。使用**{record_delimiter}**作为列表分隔符。

4. 完成后，输出{completion_delimiter}

################
-示例-
################
多模态数据块类型：table
多模态数据块唯一标识符：table-hbm3-spec-01
伴随文本：表 2 对比了 HBM3 与 HBM2E。HBM3 在 6.4 Gb/s per pin 条件下可达到 819 GB/s 堆叠带宽，而 HBM2E 为 460 GB/s。HBM3 堆叠包含 8 个 DRAM die，并通过 TSV 与 base die 互连。更高温度会增加刷新开销和待机功耗。
################
输出：
("entity"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"table"{tuple_delimiter}"该表格汇总了 HBM 存储器规格对比信息，包括带宽、pin 速率、堆叠结构、TSV 互连以及温度相关的开销/功耗变化。"{tuple_delimiter}"表 2 对比了 HBM3 与 HBM2E。"){record_delimiter}
("entity"{tuple_delimiter}"HBM3"{tuple_delimiter}"memory_product"{tuple_delimiter}"HBM3 是文中描述的高带宽存储器代际对象，涉及堆叠带宽、pin 速率、结构组成与热影响。"{tuple_delimiter}"HBM3 在 6.4 Gb/s per pin 条件下可达到 819 GB/s 堆叠带宽"){record_delimiter}
("entity"{tuple_delimiter}"HBM2E"{tuple_delimiter}"memory_product"{tuple_delimiter}"HBM2E 是该表格中的对比基线存储器产品，带宽为 460 GB/s。"{tuple_delimiter}"HBM2E 为 460 GB/s"){record_delimiter}
("entity"{tuple_delimiter}"819 GB/s"{tuple_delimiter}"performance_metric"{tuple_delimiter}"819 GB/s 是 HBM3 的堆叠带宽指标。"{tuple_delimiter}"819 GB/s 堆叠带宽"){record_delimiter}
("entity"{tuple_delimiter}"6.4 Gb/s per pin"{tuple_delimiter}"performance_metric"{tuple_delimiter}"6.4 Gb/s per pin 是与 HBM3 带宽指标对应的传输速率条件。"{tuple_delimiter}"在 6.4 Gb/s per pin 条件下"){record_delimiter}
("entity"{tuple_delimiter}"8 个 DRAM die"{tuple_delimiter}"substructure"{tuple_delimiter}"8 个 DRAM die 是 HBM3 堆叠的结构组成部分。"{tuple_delimiter}"包含 8 个 DRAM die"){record_delimiter}
("entity"{tuple_delimiter}"TSV"{tuple_delimiter}"component"{tuple_delimiter}"TSV 是 HBM3 堆叠中用于 die 间垂直互连的结构。"{tuple_delimiter}"通过 TSV 与 base die 互连"){record_delimiter}
("entity"{tuple_delimiter}"base die"{tuple_delimiter}"component"{tuple_delimiter}"base die 是 HBM3 堆叠中的基础芯片层，并通过 TSV 与其他 die 互连。"{tuple_delimiter}"通过 TSV 与 base die 互连"){record_delimiter}
("entity"{tuple_delimiter}"更高温度"{tuple_delimiter}"operating_condition"{tuple_delimiter}"更高温度是影响刷新开销和待机功耗的运行条件。"{tuple_delimiter}"更高温度会增加刷新开销和待机功耗"){record_delimiter}
("entity"{tuple_delimiter}"刷新开销"{tuple_delimiter}"performance_metric"{tuple_delimiter}"刷新开销是在更高温度下增加的系统开销指标。"{tuple_delimiter}"增加刷新开销"){record_delimiter}
("entity"{tuple_delimiter}"待机功耗"{tuple_delimiter}"power_metric"{tuple_delimiter}"待机功耗是在更高温度下增加的功耗指标。"{tuple_delimiter}"待机功耗"){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"HBM3"{tuple_delimiter}"specification_of"{tuple_delimiter}"该表格包含 HBM3 的规格对比信息。"{tuple_delimiter}"表 2 对比了 HBM3 与 HBM2E"{tuple_delimiter}0.96){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"HBM2E"{tuple_delimiter}"specification_of"{tuple_delimiter}"该表格包含 HBM2E 的规格对比信息。"{tuple_delimiter}"表 2 对比了 HBM3 与 HBM2E"{tuple_delimiter}0.96){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"819 GB/s"{tuple_delimiter}"has_bandwidth"{tuple_delimiter}"表格给出了 HBM3 的带宽指标 819 GB/s。"{tuple_delimiter}"819 GB/s 堆叠带宽"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"6.4 Gb/s per pin"{tuple_delimiter}"measured_by"{tuple_delimiter}"表格说明该带宽指标是在 6.4 Gb/s per pin 条件下给出的。"{tuple_delimiter}"在 6.4 Gb/s per pin 条件下"{tuple_delimiter}0.9){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"8 个 DRAM die"{tuple_delimiter}"part_of"{tuple_delimiter}"表格文字说明 HBM3 堆叠包含 8 个 DRAM die。"{tuple_delimiter}"包含 8 个 DRAM die"{tuple_delimiter}0.95){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"TSV"{tuple_delimiter}"connected_to"{tuple_delimiter}"表格文字将 TSV 描述为堆叠互连结构的一部分。"{tuple_delimiter}"通过 TSV 与 base die 互连"{tuple_delimiter}0.92){record_delimiter}
("relationship"{tuple_delimiter}"table-hbm3-spec-01"{tuple_delimiter}"更高温度"{tuple_delimiter}"impacts"{tuple_delimiter}"表格文字说明更高温度会改变开销与功耗表现。"{tuple_delimiter}"更高温度会增加刷新开销和待机功耗"{tuple_delimiter}0.91){completion_delimiter}
################################

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
        "entity_types": "memory_product, memory_family, interface_standard, component, substructure, timing_parameter, \
performance_metric, power_metric, capacity_metric, operating_condition, process_technology, material, signal, test_method, failure_mode, organization",
        "relation_types": "related_to, part_of, contains, connected_to, interacts_with, affects, impacts, depends_on, causes, enables, measured_by, has_timing, has_bandwidth, has_latency, has_capacity, consumes_power, compatible_with, uses_protocol, specification_of, tradeoff_with",
    },
}
