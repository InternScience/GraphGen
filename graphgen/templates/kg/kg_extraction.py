# pylint: disable=C0301
TEMPLATE_EN: str = """You are an NLP expert, skilled at analyzing text to extract named entities and their relationships.

-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use English as output language.

This extraction is optimized for semiconductor memory technical documents (e.g., DRAM, SRAM, HBM, GDDR, LPDDR, NAND, memory controller, channel, bank, rank, timing parameters such as tRCD/tRP/tRAS/tWR, bandwidth, latency, density, process node, voltage, temperature, refresh behavior, ECC, and power).

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_summary: Comprehensive summary of the entity's attributes, roles, limits, or measured/specification values
- evidence_span: a short, verbatim quote from the input text that directly supports the entity (MUST come from the text)
Entity-type restrictions for semiconductor memory documents:
- Prefer concrete technical entities such as memory families/products, standards/interfaces, hardware blocks, substructures, timing parameters, performance/power/capacity metrics, operating conditions, process technologies, materials, signals, test methods, failure modes, and organizations.
- Do NOT label generic discourse words as entities unless they denote a specific technical item in the text. Avoid vague spans like "system", "performance", "method", "result", "device" by themselves.
- Do NOT create separate entities for units alone (e.g., "ns", "V", "MT/s") or for bare adjectives unless the text uses them as a named technical term.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>{tuple_delimiter}<evidence_span>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation_type: one type from [{relation_types}]
- relationship_summary: explanation as to why you think the source entity and the target entity are related to each other
- evidence_span: a short, verbatim quote from the input text that directly supports the relationship (MUST come from the text)
- confidence: confidence score between 0 and 1, where 1 means strongest evidence
Only extract technically meaningful relations that are explicit in the text, such as specification, composition, electrical/architectural connection, compatibility, measurement, impact, or trade-off.
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relation_type>{tuple_delimiter}<relationship_summary>{tuple_delimiter}<evidence_span>{tuple_delimiter}<confidence>)

Hard constraints for relationship extraction:
- Do not invent any relationship not grounded in the text.
- If no direct evidence exists, DO NOT output that relationship.
- For memory-domain content, prioritize concrete technical relations: specification_of, part_of, connected_to, measured_by, impacts, tradeoff_with, compatible_with, uses_protocol, has_timing, has_bandwidth, has_latency, has_capacity, consumes_power.

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

All examples below use the required 5-field entity format and 7-field relationship format.

################
-Examples-
################
-Example 1-
Text:
################
The LPDDR5X device supports a peak data rate of 8533 MT/s on each 16-bit channel. In the tested configuration, tRCD and tRP are both 18 ns at VDD2H = 1.05 V. Relative to LPDDR5, LPDDR5X increases bandwidth but raises PHY power. The die is fabricated on a 12 nm process.
################
Output:
("entity"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"memory_product"{tuple_delimiter}"LPDDR5X is a low-power DRAM product/generation whose specification in this text includes per-channel data rate, timing values, operating voltage, bandwidth behavior, and implementation details."{tuple_delimiter}"The LPDDR5X device supports a peak data rate of 8533 MT/s"){record_delimiter}
("entity"{tuple_delimiter}"16-bit channel"{tuple_delimiter}"substructure"{tuple_delimiter}"A 16-bit memory channel is the channel width associated with the LPDDR5X data-rate specification in the text."{tuple_delimiter}"on each 16-bit channel"){record_delimiter}
("entity"{tuple_delimiter}"8533 MT/s"{tuple_delimiter}"performance_metric"{tuple_delimiter}"8533 MT/s is the peak data-rate metric specified for LPDDR5X in this configuration."{tuple_delimiter}"peak data rate of 8533 MT/s"){record_delimiter}
("entity"{tuple_delimiter}"tRCD"{tuple_delimiter}"timing_parameter"{tuple_delimiter}"tRCD is a DRAM timing parameter specified as 18 ns in the tested LPDDR5X configuration."{tuple_delimiter}"tRCD and tRP are both 18 ns"){record_delimiter}
("entity"{tuple_delimiter}"tRP"{tuple_delimiter}"timing_parameter"{tuple_delimiter}"tRP is a DRAM timing parameter specified as 18 ns in the tested LPDDR5X configuration."{tuple_delimiter}"tRCD and tRP are both 18 ns"){record_delimiter}
("entity"{tuple_delimiter}"VDD2H = 1.05 V"{tuple_delimiter}"operating_condition"{tuple_delimiter}"VDD2H = 1.05 V is the operating voltage condition under which the timing values are reported."{tuple_delimiter}"at VDD2H = 1.05 V"){record_delimiter}
("entity"{tuple_delimiter}"LPDDR5"{tuple_delimiter}"memory_product"{tuple_delimiter}"LPDDR5 is the earlier memory generation used as the comparison baseline for bandwidth and PHY power."{tuple_delimiter}"Relative to LPDDR5, LPDDR5X increases bandwidth but raises PHY power"){record_delimiter}
("entity"{tuple_delimiter}"bandwidth"{tuple_delimiter}"performance_metric"{tuple_delimiter}"Bandwidth is the performance metric that LPDDR5X improves relative to LPDDR5 according to the text."{tuple_delimiter}"LPDDR5X increases bandwidth"){record_delimiter}
("entity"{tuple_delimiter}"PHY power"{tuple_delimiter}"power_metric"{tuple_delimiter}"PHY power is the power-related metric that increases for LPDDR5X relative to LPDDR5 in the text."{tuple_delimiter}"raises PHY power"){record_delimiter}
("entity"{tuple_delimiter}"12 nm process"{tuple_delimiter}"process_technology"{tuple_delimiter}"The 12 nm process is the fabrication technology used for the die described in the text."{tuple_delimiter}"fabricated on a 12 nm process"){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"8533 MT/s"{tuple_delimiter}"has_bandwidth"{tuple_delimiter}"The text specifies 8533 MT/s as the peak data-rate metric for LPDDR5X."{tuple_delimiter}"supports a peak data rate of 8533 MT/s"{tuple_delimiter}0.98){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"16-bit channel"{tuple_delimiter}"part_of"{tuple_delimiter}"The text ties the LPDDR5X specification to each 16-bit channel, making the channel a structural sub-part in this context."{tuple_delimiter}"on each 16-bit channel"{tuple_delimiter}0.84){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"tRCD"{tuple_delimiter}"has_timing"{tuple_delimiter}"The text provides a tRCD timing specification for LPDDR5X."{tuple_delimiter}"tRCD and tRP are both 18 ns"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"tRP"{tuple_delimiter}"has_timing"{tuple_delimiter}"The text provides a tRP timing specification for LPDDR5X."{tuple_delimiter}"tRCD and tRP are both 18 ns"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"VDD2H = 1.05 V"{tuple_delimiter}"tRCD"{tuple_delimiter}"measured_by"{tuple_delimiter}"The text states that the reported timing values, including tRCD, are given under the VDD2H = 1.05 V operating condition."{tuple_delimiter}"tRCD and tRP are both 18 ns at VDD2H = 1.05 V"{tuple_delimiter}0.9){record_delimiter}
("relationship"{tuple_delimiter}"VDD2H = 1.05 V"{tuple_delimiter}"tRP"{tuple_delimiter}"measured_by"{tuple_delimiter}"The text states that the reported timing values, including tRP, are given under the VDD2H = 1.05 V operating condition."{tuple_delimiter}"tRCD and tRP are both 18 ns at VDD2H = 1.05 V"{tuple_delimiter}0.9){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"bandwidth"{tuple_delimiter}"impacts"{tuple_delimiter}"The text explicitly says LPDDR5X increases bandwidth relative to LPDDR5."{tuple_delimiter}"LPDDR5X increases bandwidth"{tuple_delimiter}0.95){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"PHY power"{tuple_delimiter}"consumes_power"{tuple_delimiter}"The text explicitly states that LPDDR5X raises PHY power."{tuple_delimiter}"raises PHY power"{tuple_delimiter}0.93){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"LPDDR5"{tuple_delimiter}"tradeoff_with"{tuple_delimiter}"The text frames LPDDR5X as a trade-off against LPDDR5, improving bandwidth while increasing PHY power."{tuple_delimiter}"Relative to LPDDR5, LPDDR5X increases bandwidth but raises PHY power"{tuple_delimiter}0.88){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"12 nm process"{tuple_delimiter}"specification_of"{tuple_delimiter}"The text states that the LPDDR5X die is fabricated on a 12 nm process."{tuple_delimiter}"The die is fabricated on a 12 nm process"{tuple_delimiter}0.91){record_delimiter}
("content_keywords"{tuple_delimiter}"LPDDR5X specification, timing parameters, operating voltage, bandwidth-power tradeoff, process technology"){completion_delimiter}

-Example 2-
Text:
#############
该 HBM3 堆叠包含 8 个 DRAM die，并通过 TSV 与 base die 互连。实验结果表明，在 6.4 Gb/s pin 速率下，堆叠带宽达到 819 GB/s，但温度升高会增加刷新开销并拉高待机功耗。控制器采用 ECC 机制以提升可靠性。
#############
输出：
("entity"{tuple_delimiter}"HBM3 堆叠"{tuple_delimiter}"memory_product"{tuple_delimiter}"HBM3 堆叠是该段文本的核心存储器对象，其规格涉及堆叠结构、互连方式、速率、带宽、温度影响、待机功耗与可靠性机制。"{tuple_delimiter}"该 HBM3 堆叠包含 8 个 DRAM die"){record_delimiter}
("entity"{tuple_delimiter}"8 个 DRAM die"{tuple_delimiter}"substructure"{tuple_delimiter}"8 个 DRAM die 是 HBM3 堆叠的组成结构。"{tuple_delimiter}"包含 8 个 DRAM die"){record_delimiter}
("entity"{tuple_delimiter}"TSV"{tuple_delimiter}"component"{tuple_delimiter}"TSV 是用于连接 HBM3 堆叠中 die 的互连结构。"{tuple_delimiter}"通过 TSV 与 base die 互连"){record_delimiter}
("entity"{tuple_delimiter}"base die"{tuple_delimiter}"component"{tuple_delimiter}"base die 是 HBM3 堆叠中的基础芯片层，并与 TSV 互连。"{tuple_delimiter}"通过 TSV 与 base die 互连"){record_delimiter}
("entity"{tuple_delimiter}"6.4 Gb/s pin 速率"{tuple_delimiter}"performance_metric"{tuple_delimiter}"6.4 Gb/s pin 速率是实验中的链路速度条件。"{tuple_delimiter}"在 6.4 Gb/s pin 速率下"){record_delimiter}
("entity"{tuple_delimiter}"819 GB/s"{tuple_delimiter}"performance_metric"{tuple_delimiter}"819 GB/s 是该 HBM3 堆叠达到的带宽指标。"{tuple_delimiter}"堆叠带宽达到 819 GB/s"){record_delimiter}
("entity"{tuple_delimiter}"温度升高"{tuple_delimiter}"operating_condition"{tuple_delimiter}"温度升高是影响刷新开销和待机功耗的运行条件。"{tuple_delimiter}"温度升高会增加刷新开销并拉高待机功耗"){record_delimiter}
("entity"{tuple_delimiter}"刷新开销"{tuple_delimiter}"performance_metric"{tuple_delimiter}"刷新开销是随温度升高而增加的系统开销指标。"{tuple_delimiter}"增加刷新开销"){record_delimiter}
("entity"{tuple_delimiter}"待机功耗"{tuple_delimiter}"power_metric"{tuple_delimiter}"待机功耗是随温度升高而上升的功耗指标。"{tuple_delimiter}"拉高待机功耗"){record_delimiter}
("entity"{tuple_delimiter}"控制器"{tuple_delimiter}"component"{tuple_delimiter}"控制器是该系统中的控制模块，并采用 ECC 机制。"{tuple_delimiter}"控制器采用 ECC 机制"){record_delimiter}
("entity"{tuple_delimiter}"ECC 机制"{tuple_delimiter}"interface_standard"{tuple_delimiter}"ECC 机制是用于提升可靠性的纠错机制。"{tuple_delimiter}"采用 ECC 机制以提升可靠性"){record_delimiter}
("relationship"{tuple_delimiter}"HBM3 堆叠"{tuple_delimiter}"8 个 DRAM die"{tuple_delimiter}"part_of"{tuple_delimiter}"文本明确说明 8 个 DRAM die 是 HBM3 堆叠的组成部分。"{tuple_delimiter}"包含 8 个 DRAM die"{tuple_delimiter}0.98){record_delimiter}
("relationship"{tuple_delimiter}"TSV"{tuple_delimiter}"base die"{tuple_delimiter}"connected_to"{tuple_delimiter}"文本明确指出 TSV 与 base die 互连。"{tuple_delimiter}"通过 TSV 与 base die 互连"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"HBM3 堆叠"{tuple_delimiter}"819 GB/s"{tuple_delimiter}"has_bandwidth"{tuple_delimiter}"文本给出了 HBM3 堆叠的带宽指标 819 GB/s。"{tuple_delimiter}"堆叠带宽达到 819 GB/s"{tuple_delimiter}0.99){record_delimiter}
("relationship"{tuple_delimiter}"6.4 Gb/s pin 速率"{tuple_delimiter}"819 GB/s"{tuple_delimiter}"measured_by"{tuple_delimiter}"文本说明 819 GB/s 带宽是在 6.4 Gb/s pin 速率条件下得到的。"{tuple_delimiter}"在 6.4 Gb/s pin 速率下，堆叠带宽达到 819 GB/s"{tuple_delimiter}0.92){record_delimiter}
("relationship"{tuple_delimiter}"温度升高"{tuple_delimiter}"刷新开销"{tuple_delimiter}"impacts"{tuple_delimiter}"文本明确指出温度升高会增加刷新开销。"{tuple_delimiter}"温度升高会增加刷新开销"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"温度升高"{tuple_delimiter}"待机功耗"{tuple_delimiter}"impacts"{tuple_delimiter}"文本明确指出温度升高会拉高待机功耗。"{tuple_delimiter}"温度升高会增加刷新开销并拉高待机功耗"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"控制器"{tuple_delimiter}"ECC 机制"{tuple_delimiter}"uses_protocol"{tuple_delimiter}"文本说明控制器采用 ECC 机制以提升可靠性。"{tuple_delimiter}"控制器采用 ECC 机制以提升可靠性"{tuple_delimiter}0.95){record_delimiter}
("content_keywords"{tuple_delimiter}"HBM3 堆叠结构, TSV 互连, 带宽指标, 温度影响, 刷新开销, 待机功耗, ECC 可靠性"){completion_delimiter}

################
-Real Data-
################
Entity_types: {entity_types}
Text: {input_text}
################
Output:
"""


TEMPLATE_ZH: str = """你是一个NLP专家，擅长分析文本提取命名实体和关系。

-目标-
给定一个实体类型列表和可能与列表相关的文本，从文本中识别所有这些类型的实体，以及这些实体之间所有的关系。
使用中文作为输出语言。

本任务重点适配半导体存储器技术文档（如 DRAM、SRAM、HBM、GDDR、LPDDR、NAND、DIMM、内存控制器、通道、bank、rank、tRCD/tRP/tRAS/tWR、带宽、延迟、密度、电压、温度、刷新、ECC、工艺节点、功耗等）。

-步骤-
1. 识别所有实体。对于每个识别的实体，提取以下信息：
   - entity_name：实体的名称，首字母大写
   - entity_type：以下类型之一：[{entity_types}]
   - entity_summary：实体的属性、作用、限制条件或规格/测量值的全面总结
   - evidence_span：直接支持该实体的原文短句（必须是输入文本中的原文片段）
   存储器技术文档中的实体类型限制：
   - 优先抽取具体技术实体，例如存储器家族/产品、标准/接口、硬件模块、子结构、时序参数、性能/功耗/容量指标、运行条件、工艺技术、材料、信号、测试方法、失效模式、组织机构。
   - 不要把泛泛而谈的词当作实体，除非它在文中明确指向具体技术对象；避免抽取“系统”“性能”“方法”“结果”“器件”等空泛词语本身。
   - 不要仅把单位（如 ns、V、MT/s）或普通形容词单独作为实体，除非文本把它们作为完整命名技术项使用。
   将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>{tuple_delimiter}<evidence_span>)
   
2. 从步骤1中识别的实体中，识别所有（源实体，目标实体）对，这些实体彼此之间*明显相关*。
   对于每对相关的实体，提取以下信息：
   - source_entity：步骤1中识别的源实体名称
   - target_entity：步骤1中识别的目标实体名称
   - relation_type：从[{relation_types}]中选择一个关系类型
   - relationship_summary：解释为什么你认为源实体和目标实体彼此相关
   - evidence_span：直接支持该关系的原文短句（必须是输入文本中的原文片段）
   - confidence：0到1之间的置信度，1表示证据最强
   只抽取文本中明确表达、且具有技术含义的关系，例如规格归属、组成关系、电气/结构连接、兼容性、测量条件、影响关系或权衡关系。
   将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relation_type>{tuple_delimiter}<relationship_summary>{tuple_delimiter}<evidence_span>{tuple_delimiter}<confidence>)

关系抽取硬约束：
- 严禁编造文本中不存在的关系。
- 如果找不到直接证据，不要输出该关系。
- 在存储器领域优先输出技术性强的关系：specification_of、part_of、connected_to、measured_by、impacts、tradeoff_with、compatible_with、uses_protocol、has_timing、has_bandwidth、has_latency、has_capacity、consumes_power。

3. 识别总结整个文本的主要概念、主题或话题的高级关键词。这些应该捕捉文档中存在的总体思想。
   将内容级关键词格式化为("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以中文返回步骤1和2中识别出的所有实体和关系的输出列表。使用**{record_delimiter}**作为列表分隔符。

5. 完成后，输出{completion_delimiter}

下面的示例均使用要求的5字段实体格式和7字段关系格式。

################
-示例-
################
-示例 1-
文本：
################
LPDDR5X 器件在每个 16-bit channel 上支持 8533 MT/s 的峰值数据速率。在测试配置下，tRCD 和 tRP 在 VDD2H = 1.05 V 条件下均为 18 ns。相较于 LPDDR5，LPDDR5X 提升了带宽，但会增加 PHY 功耗。该 die 采用 12 nm 工艺制造。
################
输出：
("entity"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"memory_product"{tuple_delimiter}"LPDDR5X 是该段文本中的低功耗 DRAM 产品/代际对象，文本给出了其数据速率、时序参数、电压条件、带宽变化、功耗影响与制造工艺等规格信息。"{tuple_delimiter}"LPDDR5X 器件在每个 16-bit channel 上支持 8533 MT/s 的峰值数据速率"){record_delimiter}
("entity"{tuple_delimiter}"16-bit channel"{tuple_delimiter}"substructure"{tuple_delimiter}"16-bit channel 是 LPDDR5X 规格中涉及的通道结构单位。"{tuple_delimiter}"每个 16-bit channel 上"){record_delimiter}
("entity"{tuple_delimiter}"8533 MT/s"{tuple_delimiter}"performance_metric"{tuple_delimiter}"8533 MT/s 是 LPDDR5X 的峰值数据速率指标。"{tuple_delimiter}"8533 MT/s 的峰值数据速率"){record_delimiter}
("entity"{tuple_delimiter}"tRCD"{tuple_delimiter}"timing_parameter"{tuple_delimiter}"tRCD 是该 LPDDR5X 配置中的 DRAM 时序参数，数值为 18 ns。"{tuple_delimiter}"tRCD 和 tRP 在 VDD2H = 1.05 V 条件下均为 18 ns"){record_delimiter}
("entity"{tuple_delimiter}"tRP"{tuple_delimiter}"timing_parameter"{tuple_delimiter}"tRP 是该 LPDDR5X 配置中的 DRAM 时序参数，数值为 18 ns。"{tuple_delimiter}"tRCD 和 tRP 在 VDD2H = 1.05 V 条件下均为 18 ns"){record_delimiter}
("entity"{tuple_delimiter}"VDD2H = 1.05 V"{tuple_delimiter}"operating_condition"{tuple_delimiter}"VDD2H = 1.05 V 是报告 tRCD 与 tRP 数值时对应的工作电压条件。"{tuple_delimiter}"在 VDD2H = 1.05 V 条件下"){record_delimiter}
("entity"{tuple_delimiter}"LPDDR5"{tuple_delimiter}"memory_product"{tuple_delimiter}"LPDDR5 是文中用于对比带宽与 PHY 功耗的上一代存储器对象。"{tuple_delimiter}"相较于 LPDDR5，LPDDR5X 提升了带宽，但会增加 PHY 功耗"){record_delimiter}
("entity"{tuple_delimiter}"带宽"{tuple_delimiter}"performance_metric"{tuple_delimiter}"带宽是 LPDDR5X 相较于 LPDDR5 得到提升的性能指标。"{tuple_delimiter}"LPDDR5X 提升了带宽"){record_delimiter}
("entity"{tuple_delimiter}"PHY 功耗"{tuple_delimiter}"power_metric"{tuple_delimiter}"PHY 功耗是 LPDDR5X 相较于 LPDDR5 增加的功耗指标。"{tuple_delimiter}"会增加 PHY 功耗"){record_delimiter}
("entity"{tuple_delimiter}"12 nm 工艺"{tuple_delimiter}"process_technology"{tuple_delimiter}"12 nm 工艺是文中该 die 的制造工艺。"{tuple_delimiter}"采用 12 nm 工艺制造"){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"8533 MT/s"{tuple_delimiter}"has_bandwidth"{tuple_delimiter}"文本明确给出了 LPDDR5X 的峰值数据速率指标为 8533 MT/s。"{tuple_delimiter}"支持 8533 MT/s 的峰值数据速率"{tuple_delimiter}0.98){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"16-bit channel"{tuple_delimiter}"part_of"{tuple_delimiter}"文本将该速率规格限定在每个 16-bit channel 上，说明 channel 是该规格上下文中的结构单元。"{tuple_delimiter}"每个 16-bit channel 上"{tuple_delimiter}0.84){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"tRCD"{tuple_delimiter}"has_timing"{tuple_delimiter}"文本直接给出了 LPDDR5X 的 tRCD 时序参数。"{tuple_delimiter}"tRCD 和 tRP 在 VDD2H = 1.05 V 条件下均为 18 ns"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"tRP"{tuple_delimiter}"has_timing"{tuple_delimiter}"文本直接给出了 LPDDR5X 的 tRP 时序参数。"{tuple_delimiter}"tRCD 和 tRP 在 VDD2H = 1.05 V 条件下均为 18 ns"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"VDD2H = 1.05 V"{tuple_delimiter}"tRCD"{tuple_delimiter}"measured_by"{tuple_delimiter}"文本说明 tRCD 数值是在 VDD2H = 1.05 V 条件下报告的。"{tuple_delimiter}"tRCD 和 tRP 在 VDD2H = 1.05 V 条件下均为 18 ns"{tuple_delimiter}0.9){record_delimiter}
("relationship"{tuple_delimiter}"VDD2H = 1.05 V"{tuple_delimiter}"tRP"{tuple_delimiter}"measured_by"{tuple_delimiter}"文本说明 tRP 数值是在 VDD2H = 1.05 V 条件下报告的。"{tuple_delimiter}"tRCD 和 tRP 在 VDD2H = 1.05 V 条件下均为 18 ns"{tuple_delimiter}0.9){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"带宽"{tuple_delimiter}"impacts"{tuple_delimiter}"文本明确指出 LPDDR5X 会提升带宽。"{tuple_delimiter}"LPDDR5X 提升了带宽"{tuple_delimiter}0.95){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"PHY 功耗"{tuple_delimiter}"consumes_power"{tuple_delimiter}"文本明确指出 LPDDR5X 会增加 PHY 功耗。"{tuple_delimiter}"会增加 PHY 功耗"{tuple_delimiter}0.93){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"LPDDR5"{tuple_delimiter}"tradeoff_with"{tuple_delimiter}"文本把 LPDDR5X 与 LPDDR5 进行权衡比较：带宽更高，但 PHY 功耗也更高。"{tuple_delimiter}"相较于 LPDDR5，LPDDR5X 提升了带宽，但会增加 PHY 功耗"{tuple_delimiter}0.88){record_delimiter}
("relationship"{tuple_delimiter}"LPDDR5X"{tuple_delimiter}"12 nm 工艺"{tuple_delimiter}"specification_of"{tuple_delimiter}"文本说明该 LPDDR5X die 采用 12 nm 工艺制造。"{tuple_delimiter}"该 die 采用 12 nm 工艺制造"{tuple_delimiter}0.91){record_delimiter}
("content_keywords"{tuple_delimiter}"LPDDR5X 规格, 时序参数, 工作电压, 带宽-功耗权衡, 制造工艺"){completion_delimiter}

-示例 2-
文本：
################
该 HBM3 堆叠包含 8 个 DRAM die，并通过 TSV 与 base die 互连。实验结果表明，在 6.4 Gb/s pin 速率下，堆叠带宽达到 819 GB/s，但温度升高会增加刷新开销并拉高待机功耗。控制器采用 ECC 机制以提升可靠性。
################
输出：
("entity"{tuple_delimiter}"HBM3 堆叠"{tuple_delimiter}"memory_product"{tuple_delimiter}"HBM3 堆叠是该段文本的核心存储器对象，其规格涉及堆叠结构、互连方式、速率、带宽、温度影响、待机功耗与可靠性机制。"{tuple_delimiter}"该 HBM3 堆叠包含 8 个 DRAM die"){record_delimiter}
("entity"{tuple_delimiter}"8 个 DRAM die"{tuple_delimiter}"substructure"{tuple_delimiter}"8 个 DRAM die 是 HBM3 堆叠的组成结构。"{tuple_delimiter}"包含 8 个 DRAM die"){record_delimiter}
("entity"{tuple_delimiter}"TSV"{tuple_delimiter}"component"{tuple_delimiter}"TSV 是用于连接 HBM3 堆叠中 die 的互连结构。"{tuple_delimiter}"通过 TSV 与 base die 互连"){record_delimiter}
("entity"{tuple_delimiter}"base die"{tuple_delimiter}"component"{tuple_delimiter}"base die 是 HBM3 堆叠中的基础芯片层，并与 TSV 互连。"{tuple_delimiter}"通过 TSV 与 base die 互连"){record_delimiter}
("entity"{tuple_delimiter}"6.4 Gb/s pin 速率"{tuple_delimiter}"performance_metric"{tuple_delimiter}"6.4 Gb/s pin 速率是实验中的链路速度条件。"{tuple_delimiter}"在 6.4 Gb/s pin 速率下"){record_delimiter}
("entity"{tuple_delimiter}"819 GB/s"{tuple_delimiter}"performance_metric"{tuple_delimiter}"819 GB/s 是该 HBM3 堆叠达到的带宽指标。"{tuple_delimiter}"堆叠带宽达到 819 GB/s"){record_delimiter}
("entity"{tuple_delimiter}"温度升高"{tuple_delimiter}"operating_condition"{tuple_delimiter}"温度升高是影响刷新开销和待机功耗的运行条件。"{tuple_delimiter}"温度升高会增加刷新开销并拉高待机功耗"){record_delimiter}
("entity"{tuple_delimiter}"刷新开销"{tuple_delimiter}"performance_metric"{tuple_delimiter}"刷新开销是随温度升高而增加的系统开销指标。"{tuple_delimiter}"增加刷新开销"){record_delimiter}
("entity"{tuple_delimiter}"待机功耗"{tuple_delimiter}"power_metric"{tuple_delimiter}"待机功耗是随温度升高而上升的功耗指标。"{tuple_delimiter}"拉高待机功耗"){record_delimiter}
("entity"{tuple_delimiter}"控制器"{tuple_delimiter}"component"{tuple_delimiter}"控制器是该系统中的控制模块，并采用 ECC 机制。"{tuple_delimiter}"控制器采用 ECC 机制"){record_delimiter}
("entity"{tuple_delimiter}"ECC 机制"{tuple_delimiter}"interface_standard"{tuple_delimiter}"ECC 机制是用于提升可靠性的纠错机制。"{tuple_delimiter}"采用 ECC 机制以提升可靠性"){record_delimiter}
("relationship"{tuple_delimiter}"HBM3 堆叠"{tuple_delimiter}"8 个 DRAM die"{tuple_delimiter}"part_of"{tuple_delimiter}"文本明确说明 8 个 DRAM die 是 HBM3 堆叠的组成部分。"{tuple_delimiter}"包含 8 个 DRAM die"{tuple_delimiter}0.98){record_delimiter}
("relationship"{tuple_delimiter}"TSV"{tuple_delimiter}"base die"{tuple_delimiter}"connected_to"{tuple_delimiter}"文本明确指出 TSV 与 base die 互连。"{tuple_delimiter}"通过 TSV 与 base die 互连"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"HBM3 堆叠"{tuple_delimiter}"819 GB/s"{tuple_delimiter}"has_bandwidth"{tuple_delimiter}"文本给出了 HBM3 堆叠的带宽指标 819 GB/s。"{tuple_delimiter}"堆叠带宽达到 819 GB/s"{tuple_delimiter}0.99){record_delimiter}
("relationship"{tuple_delimiter}"6.4 Gb/s pin 速率"{tuple_delimiter}"819 GB/s"{tuple_delimiter}"measured_by"{tuple_delimiter}"文本说明 819 GB/s 带宽是在 6.4 Gb/s pin 速率条件下得到的。"{tuple_delimiter}"在 6.4 Gb/s pin 速率下，堆叠带宽达到 819 GB/s"{tuple_delimiter}0.92){record_delimiter}
("relationship"{tuple_delimiter}"温度升高"{tuple_delimiter}"刷新开销"{tuple_delimiter}"impacts"{tuple_delimiter}"文本明确指出温度升高会增加刷新开销。"{tuple_delimiter}"温度升高会增加刷新开销"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"温度升高"{tuple_delimiter}"待机功耗"{tuple_delimiter}"impacts"{tuple_delimiter}"文本明确指出温度升高会拉高待机功耗。"{tuple_delimiter}"温度升高会增加刷新开销并拉高待机功耗"{tuple_delimiter}0.97){record_delimiter}
("relationship"{tuple_delimiter}"控制器"{tuple_delimiter}"ECC 机制"{tuple_delimiter}"uses_protocol"{tuple_delimiter}"文本说明控制器采用 ECC 机制以提升可靠性。"{tuple_delimiter}"控制器采用 ECC 机制以提升可靠性"{tuple_delimiter}0.95){record_delimiter}
("content_keywords"{tuple_delimiter}"HBM3 堆叠结构, TSV 互连, 带宽指标, 温度影响, 刷新开销, 待机功耗, ECC 可靠性"){completion_delimiter}

-真实数据-
实体类型：{entity_types}
文本：{input_text}
################
输出：
"""

CONTINUE_EN: str = """MANY entities and relationships were missed in the last extraction.  \
Add them below using the same format:
"""

CONTINUE_ZH: str = """很多实体和关系在上一次的提取中可能被遗漏了。请在下面使用相同的格式添加它们："""

IF_LOOP_EN: str = """It appears some entities and relationships may have still been missed.  \
Answer YES | NO if there are still entities and relationships that need to be added.
"""

IF_LOOP_ZH: str = """看起来可能仍然遗漏了一些实体和关系。如果仍有实体和关系需要添加，请回答YES | NO。"""

KG_EXTRACTION_PROMPT: dict = {
    "en": {
        "TEMPLATE": TEMPLATE_EN,
        "CONTINUE": CONTINUE_EN,
        "IF_LOOP": IF_LOOP_EN,
    },
    "zh": {
        "TEMPLATE": TEMPLATE_ZH,
        "CONTINUE": CONTINUE_ZH,
        "IF_LOOP": IF_LOOP_ZH,
    },
    "FORMAT": {
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>",
        "entity_types": "memory_product, memory_family, interface_standard, component, substructure, timing_parameter, \
performance_metric, power_metric, capacity_metric, operating_condition, process_technology, material, signal, test_method, failure_mode, organization",
        "relation_types": "related_to, part_of, contains, connected_to, interacts_with, affects, impacts, depends_on, causes, enables, measured_by, has_timing, has_bandwidth, has_latency, has_capacity, consumes_power, compatible_with, uses_protocol, specification_of, tradeoff_with",
    },
}
