# pylint: disable=C0301
TEMPLATE_EN: str = """You are an expert in protein science and knowledge-graph construction.
Your task is to extract a star-shaped knowledge graph centered on **a single protein** mentioned in the given text.

-Goal-  
Given free-text that discusses one or more proteins, identify:  
1. The **central protein** (the first-mentioned protein or the protein explicitly indicated by the user).  
2. All entities that are **directly related** to this central protein.  
3. All relationships that **directly link** those entities to the central protein (star edges).  

Use English as the output language.

-Steps-  
1. Identify the **central protein entity** and all **directly-related entities** from the text.  
   For the **central protein**, extract:  
   - entity_name: use the full name or UniProt ID if given; capitalized.  
   - entity_type: always `protein`.  
   - entity_summary: concise description of its main biological role, location, or significance in the text.  

   For each **directly-related entity**, extract:  
   - entity_name: capitalized.  
   - entity_type: one of [{entity_types}].  
   - entity_summary: comprehensive summary of its attributes/activities **as stated in the text**.  

   Format each entity as  
   ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>)

2. From the entities found in Step 1, list every **(central protein → related entity)** pair that is **clearly related**.  
   For each pair extract:  
   - source_entity: the **central protein** name.  
   - target_entity: the related entity name.  
   - relationship_summary: short explanation of how the central protein is connected to this entity **according to the text**.  

   Format each relationship as  
   ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_summary>)

3. Output a single list of all entities and relationships from Steps 1–2, using **{record_delimiter}** as the delimiter.

4. Finish by printing {completion_delimiter}

################  
-Example-  
################  
Text:  
################  
The tumor-suppressor protein p53 is a transcription factor that responds to DNA damage.  
Phosphorylation of p53 by ATM kinase at serine-15 enhances its stability.  
MDM2, an E3 ubiquitin ligase, negatively regulates p53 via ubiquitination.  
################  
Output:  
("entity"{tuple_delimiter}"p53"{tuple_delimiter}"protein"{tuple_delimiter}"Tumor-suppressor transcription factor that responds to DNA damage and is regulated by post-translational modifications."){record_delimiter}  
("entity"{tuple_delimiter}"ATM kinase"{tuple_delimiter}"protein"{tuple_delimiter}"Protein kinase that phosphorylates p53 at serine-15, thereby enhancing p53 stability."){record_delimiter}  
("entity"{tuple_delimiter}"serine-15"{tuple_delimiter}"site"{tuple_delimiter}"Phosphorylation site on p53 that is targeted by ATM kinase."){record_delimiter}  
("entity"{tuple_delimiter}"MDM2"{tuple_delimiter}"protein"{tuple_delimiter}"E3 ubiquitin ligase that negatively regulates p53 through ubiquitination."){record_delimiter}  
("entity"{tuple_delimiter}"DNA damage"{tuple_delimiter}"concept"{tuple_delimiter}"Cellular stress signal that activates p53-mediated transcriptional response."){record_delimiter}  
("relationship"{tuple_delimiter}"p53"{tuple_delimiter}"ATM kinase"{tuple_delimiter}"ATM kinase phosphorylates p53, enhancing its stability."){record_delimiter}  
("relationship"{tuple_delimiter}"p53"{tuple_delimiter}"serine-15"{tuple_delimiter}"p53 is phosphorylated at serine-15 by ATM kinase."){record_delimiter}  
("relationship"{tuple_delimiter}"p53"{tuple_delimiter}"MDM2"{tuple_delimiter}"MDM2 ubiquitinates p53, negatively regulating its activity."){record_delimiter}  
("relationship"{tuple_delimiter}"p53"{tuple_delimiter}"DNA damage"{tuple_delimiter}"p53 acts as a sensor-transcription factor in response to DNA damage."){completion_delimiter}  

################  
-Real Data-  
Entity_types: {entity_types}  
Text: {input_text}  
################  
Output:  
"""


TEMPLATE_ZH: str = """您是蛋白质科学与知识图谱构建专家。
任务：从给定文本中抽取以**一个中心蛋白质**为核心的星型知识图谱。

-目标-  
文本可能提及一个或多个蛋白质，请：  
1. 确定**中心蛋白质**（文本首个提及或用户指定的蛋白）。  
2. 识别所有与中心蛋白**直接相关**的实体。  
3. 仅保留**中心蛋白→相关实体**的直接关系（星型边）。  

使用中文输出。

-步骤-  
1. 确定**中心蛋白质实体**及所有**直接相关实体**。  
   对于**中心蛋白质**：  
   - entity_name：全名或UniProt ID，首字母大写。  
   - entity_type：固定为`protein`。  
   - entity_summary：简述其在文中的生物学功能、定位或意义。  

   对于每个**直接相关实体**：  
   - entity_name：首字母大写。  
   - entity_type：可选类型[{entity_types}]。  
   - entity_summary：全面总结其在文中与中心蛋白相关的属性/活动。  

   格式：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>)

2. 在步骤1的实体中，列出所有**（中心蛋白→相关实体）**的明显关系对。  
   每对提取：  
   - source_entity：中心蛋白名称。  
   - target_entity：相关实体名称。  
   - relationship_summary：简要说明文中二者如何直接关联。  

   格式：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_summary>)

3. 将步骤1–2的所有实体与关系合并为单列表，用**{record_delimiter}**分隔。

4. 输出结束标记{completion_delimiter}

################  
-示例-  
################  
文本：  
################  
肿瘤抑制蛋白p53是一种转录因子，可响应DNA损伤。ATM激酶在第15位丝氨酸磷酸化p53，增强其稳定性。E3泛素连接酶MDM2通过泛素化负调控p53。  
################  
输出：  
("entity"{tuple_delimiter}"p53"{tuple_delimiter}"protein"{tuple_delimiter}"肿瘤抑制转录因子，能感知DNA损伤并通过翻译后修饰被调控。"){record_delimiter}  
("entity"{tuple_delimiter}"ATM激酶"{tuple_delimiter}"protein"{tuple_delimiter}"蛋白激酶，在丝氨酸-15位点磷酸化p53，从而提高其稳定性。"){record_delimiter}  
("entity"{tuple_delimiter}"丝氨酸-15"{tuple_delimiter}"site"{tuple_delimiter}"p53上被ATM激酶靶向的磷酸化位点。"){record_delimiter}  
("entity"{tuple_delimiter}"MDM2"{tuple_delimiter}"protein"{tuple_delimiter}"E3泛素连接酶，通过泛素化负调控p53。"){record_delimiter}  
("entity"{tuple_delimiter}"DNA损伤"{tuple_delimiter}"concept"{tuple_delimiter}"细胞内应激信号，可激活p53介导的转录应答。"){record_delimiter}  
("relationship"{tuple_delimiter}"p53"{tuple_delimiter}"ATM激酶"{tuple_delimiter}"ATM激酶磷酸化p53，增强其稳定性。"){record_delimiter}  
("relationship"{tuple_delimiter}"p53"{tuple_delimiter}"丝氨酸-15"{tuple_delimiter}"p53在该位点被ATM激酶磷酸化。"){record_delimiter}  
("relationship"{tuple_delimiter}"p53"{tuple_delimiter}"MDM2"{tuple_delimiter}"MDM2对p53进行泛素化，负向调控其活性。"){record_delimiter}  
("relationship"{tuple_delimiter}"p53"{tuple_delimiter}"DNA损伤"{tuple_delimiter}"p53作为感受器-转录因子响应DNA损伤。"){completion_delimiter}  

################  
-真实数据-  
实体类型：{entity_types}  
文本：{input_text}  
################  
输出：  
"""


PROTEIN_KG_EXTRACTION_PROMPT: dict = {
    "en": TEMPLATE_EN,
    "zh": TEMPLATE_ZH,
    "FORMAT": {
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>",
        "entity_types": "protein, gene, site, modification, pathway, disease, drug, organism, tissue, cell_line, "
        "experiment, technology, concept, location, organization, person, mission, science",
    },
}
