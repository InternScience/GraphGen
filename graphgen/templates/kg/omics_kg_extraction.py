# pylint: disable=C0301
TEMPLATE_EN: str = """You are a bioinformatics expert, skilled at analyzing biological sequences (DNA, RNA, protein) and their metadata to extract biological entities and their relationships.

-Goal-
Given a biological sequence chunk (DNA, RNA, or protein) along with its metadata, identify all relevant biological entities and their relationships.
Use English as output language.

-Steps-
1. Identify all biological entities. For each identified entity, extract the following information:
- entity_name: Name of the entity (gene name, protein name, RNA name, domain name, etc.), capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_summary: Comprehensive summary of the entity's biological function, structure, or properties
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *biologically related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_summary: explanation of the biological relationship (e.g., encodes, transcribes, translates, interacts, regulates, homologous_to, located_in, etc.)
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_summary>)

3. Identify high-level key words that summarize the main biological concepts, functions, or themes.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

################
-Examples-
################
-Example 1-
Sequence Metadata:
################
molecule_type: DNA
database: NCBI
id: NG_033923
gene_name: BRCA1
gene_description: BRCA1 DNA repair associated
organism: Homo sapiens
gene_type: protein-coding
chromosome: 17
genomic_location: 43044295-43125483
function: BRCA1 is a tumor suppressor gene involved in DNA repair
sequence_chunk: ATGCGATCGATCGATCG... (first 500bp of BRCA1 gene)
################
Output:
("entity"{tuple_delimiter}"BRCA1"{tuple_delimiter}"gene"{tuple_delimiter}"BRCA1 is a protein-coding tumor suppressor gene located on chromosome 17 in humans, involved in DNA repair mechanisms."){record_delimiter}
("entity"{tuple_delimiter}"Homo sapiens"{tuple_delimiter}"organism"{tuple_delimiter}"Human species, the organism in which BRCA1 gene is found."){record_delimiter}
("entity"{tuple_delimiter}"chromosome 17"{tuple_delimiter}"location"{tuple_delimiter}"Chromosome 17 is the chromosomal location of the BRCA1 gene in humans."){record_delimiter}
("entity"{tuple_delimiter}"DNA repair"{tuple_delimiter}"biological_process"{tuple_delimiter}"DNA repair is a biological process in which BRCA1 is involved as a tumor suppressor."){record_delimiter}
("relationship"{tuple_delimiter}"BRCA1"{tuple_delimiter}"Homo sapiens"{tuple_delimiter}"BRCA1 is a gene found in Homo sapiens."){record_delimiter}
("relationship"{tuple_delimiter}"BRCA1"{tuple_delimiter}"chromosome 17"{tuple_delimiter}"BRCA1 is located on chromosome 17 in the human genome."){record_delimiter}
("relationship"{tuple_delimiter}"BRCA1"{tuple_delimiter}"DNA repair"{tuple_delimiter}"BRCA1 is involved in DNA repair processes as a tumor suppressor gene."){record_delimiter}
("content_keywords"{tuple_delimiter}"tumor suppressor, DNA repair, genetic disease, cancer genetics"){completion_delimiter}

-Example 2-
Sequence Metadata:
################
molecule_type: RNA
database: RNAcentral
id: URS0000000001
rna_type: miRNA
description: hsa-let-7a-1 microRNA
organism: Homo sapiens
related_genes: ["LIN28", "HMGA2"]
sequence_chunk: CUCCUUUGACGUUAGCGGCGGACGGGUUAGUAACACGUGGGUAACCUACCUAUAAGACUGGGAUAACUUCGGGAAACCGGAGCUAAUACCGGAUAAUAUUUCGAACCGCAUGGUUCGAUAGUGAAAGAUGGUUUUGCUAUCACUUAUAGAUGGACCCGCGCCGUAUUAGCUAGUUGGUAAGGUAACGGCUUACCAAGGCGACGAUACGUAGCCGACCUGAGAGGGUGAUCGGCCACACUGGAACUGAGACACGGUCCAGACUCCUACGGGAGGCAGCAGGGG
################
Output:
("entity"{tuple_delimiter}"hsa-let-7a-1"{tuple_delimiter}"rna"{tuple_delimiter}"hsa-let-7a-1 is a microRNA (miRNA) found in Homo sapiens, involved in gene regulation."){record_delimiter}
("entity"{tuple_delimiter}"LIN28"{tuple_delimiter}"gene"{tuple_delimiter}"LIN28 is a gene related to hsa-let-7a-1 microRNA, involved in RNA processing and development."){record_delimiter}
("entity"{tuple_delimiter}"HMGA2"{tuple_delimiter}"gene"{tuple_delimiter}"HMGA2 is a gene related to hsa-let-7a-1 microRNA, involved in chromatin structure and gene expression."){record_delimiter}
("entity"{tuple_delimiter}"Homo sapiens"{tuple_delimiter}"organism"{tuple_delimiter}"Human species, the organism in which hsa-let-7a-1 is found."){record_delimiter}
("entity"{tuple_delimiter}"microRNA"{tuple_delimiter}"rna_type"{tuple_delimiter}"MicroRNA is a type of small non-coding RNA involved in post-transcriptional gene regulation."){record_delimiter}
("relationship"{tuple_delimiter}"hsa-let-7a-1"{tuple_delimiter}"Homo sapiens"{tuple_delimiter}"hsa-let-7a-1 is a microRNA found in Homo sapiens."){record_delimiter}
("relationship"{tuple_delimiter}"hsa-let-7a-1"{tuple_delimiter}"LIN28"{tuple_delimiter}"hsa-let-7a-1 microRNA is related to LIN28 gene, potentially regulating its expression."){record_delimiter}
("relationship"{tuple_delimiter}"hsa-let-7a-1"{tuple_delimiter}"HMGA2"{tuple_delimiter}"hsa-let-7a-1 microRNA is related to HMGA2 gene, potentially regulating its expression."){record_delimiter}
("relationship"{tuple_delimiter}"hsa-let-7a-1"{tuple_delimiter}"microRNA"{tuple_delimiter}"hsa-let-7a-1 belongs to the microRNA class of RNA molecules."){record_delimiter}
("content_keywords"{tuple_delimiter}"microRNA, gene regulation, post-transcriptional control, RNA processing"){completion_delimiter}

-Example 3-
Sequence Metadata:
################
molecule_type: protein
database: UniProt
id: P01308
protein_name: Insulin
organism: Homo sapiens
function: ["Regulates glucose metabolism", "Hormone signaling"]
sequence_chunk: MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN
################
Output:
("entity"{tuple_delimiter}"Insulin"{tuple_delimiter}"protein"{tuple_delimiter}"Insulin is a protein hormone in Homo sapiens that regulates glucose metabolism and hormone signaling."){record_delimiter}
("entity"{tuple_delimiter}"Homo sapiens"{tuple_delimiter}"organism"{tuple_delimiter}"Human species, the organism in which Insulin is produced."){record_delimiter}
("entity"{tuple_delimiter}"glucose metabolism"{tuple_delimiter}"biological_process"{tuple_delimiter}"Glucose metabolism is a biological process regulated by Insulin."){record_delimiter}
("entity"{tuple_delimiter}"hormone signaling"{tuple_delimiter}"biological_process"{tuple_delimiter}"Hormone signaling is a biological process in which Insulin participates as a signaling molecule."){record_delimiter}
("relationship"{tuple_delimiter}"Insulin"{tuple_delimiter}"Homo sapiens"{tuple_delimiter}"Insulin is a protein produced in Homo sapiens."){record_delimiter}
("relationship"{tuple_delimiter}"Insulin"{tuple_delimiter}"glucose metabolism"{tuple_delimiter}"Insulin regulates glucose metabolism in the body."){record_delimiter}
("relationship"{tuple_delimiter}"Insulin"{tuple_delimiter}"hormone signaling"{tuple_delimiter}"Insulin participates in hormone signaling pathways."){record_delimiter}
("content_keywords"{tuple_delimiter}"hormone, metabolism, glucose regulation, signaling pathway"){completion_delimiter}

################
-Real Data-
################
Entity_types: {entity_types}
Sequence Metadata: {metadata_text}
Sequence Chunk: {sequence_chunk}
################
Output:
"""


TEMPLATE_ZH: str = """你是一个生物信息学专家，擅长分析生物序列（DNA、RNA、蛋白质）及其元数据，提取生物实体及其关系。

-目标-
给定一个生物序列片段（DNA、RNA或蛋白质）及其元数据，识别所有相关的生物实体及其关系。
使用中文作为输出语言。

-步骤-
1. 识别所有生物实体。对于每个识别的实体，提取以下信息：
   - entity_name：实体的名称（基因名、蛋白质名、RNA名、功能域名等），首字母大写
   - entity_type：以下类型之一：[{entity_types}]
   - entity_summary：实体生物学功能、结构或属性的全面总结
   将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>)

2. 从步骤1中识别的实体中，识别所有（源实体，目标实体）对，这些实体彼此之间*在生物学上相关*。
   对于每对相关的实体，提取以下信息：
   - source_entity：步骤1中识别的源实体名称
   - target_entity：步骤1中识别的目标实体名称
   - relationship_summary：生物学关系的解释（例如：编码、转录、翻译、相互作用、调控、同源、位于等）
   将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_summary>)

3. 识别总结主要生物学概念、功能或主题的高级关键词。
   将内容级关键词格式化为("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以中文返回步骤1和2中识别出的所有实体和关系的输出列表。使用**{record_delimiter}**作为列表分隔符。

5. 完成后，输出{completion_delimiter}

################
-示例-
################
-示例 1-
序列元数据：
################
molecule_type: DNA
database: NCBI
id: NG_033923
gene_name: BRCA1
gene_description: BRCA1 DNA repair associated
organism: Homo sapiens
gene_type: protein-coding
chromosome: 17
genomic_location: 43044295-43125483
function: BRCA1 is a tumor suppressor gene involved in DNA repair
sequence_chunk: ATGCGATCGATCGATCG... (BRCA1基因的前500bp)
################
输出：
("entity"{tuple_delimiter}"BRCA1"{tuple_delimiter}"gene"{tuple_delimiter}"BRCA1是位于人类17号染色体上的蛋白质编码肿瘤抑制基因，参与DNA修复机制。"){record_delimiter}
("entity"{tuple_delimiter}"Homo sapiens"{tuple_delimiter}"organism"{tuple_delimiter}"人类，BRCA1基因所在的生物体。"){record_delimiter}
("entity"{tuple_delimiter}"17号染色体"{tuple_delimiter}"location"{tuple_delimiter}"17号染色体是BRCA1基因在人类基因组中的位置。"){record_delimiter}
("entity"{tuple_delimiter}"DNA修复"{tuple_delimiter}"biological_process"{tuple_delimiter}"DNA修复是BRCA1作为肿瘤抑制基因参与的生物学过程。"){record_delimiter}
("relationship"{tuple_delimiter}"BRCA1"{tuple_delimiter}"Homo sapiens"{tuple_delimiter}"BRCA1是在人类中发现的基因。"){record_delimiter}
("relationship"{tuple_delimiter}"BRCA1"{tuple_delimiter}"17号染色体"{tuple_delimiter}"BRCA1位于人类基因组的17号染色体上。"){record_delimiter}
("relationship"{tuple_delimiter}"BRCA1"{tuple_delimiter}"DNA修复"{tuple_delimiter}"BRCA1作为肿瘤抑制基因参与DNA修复过程。"){record_delimiter}
("content_keywords"{tuple_delimiter}"肿瘤抑制, DNA修复, 遗传疾病, 癌症遗传学"){completion_delimiter}

################
-真实数据-
################
实体类型：{entity_types}
序列元数据：{metadata_text}
序列片段：{sequence_chunk}
################
输出：
"""


CONTINUE_EN: str = """MANY entities and relationships were missed in the last extraction. \
Add them below using the same format:
"""

CONTINUE_ZH: str = """很多实体和关系在上一次的提取中可能被遗漏了。请在下面使用相同的格式添加它们："""

IF_LOOP_EN: str = """It appears some entities and relationships may have still been missed. \
Answer YES | NO if there are still entities and relationships that need to be added.
"""

IF_LOOP_ZH: str = """看起来可能仍然遗漏了一些实体和关系。如果仍有实体和关系需要添加，请回答YES | NO。"""

OMICS_KG_EXTRACTION_PROMPT: dict = {
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
        "entity_types": "gene, rna, protein, organism, location, biological_process, rna_type, protein_domain, \
mutation, pathway, disease, function, structure",
    },
}
