TEMPLATE_EN = """You are a biomedical entity extraction engine.
Given the text below, output **only** a JSON object that matches the following schema:

{{
  "Protein accession or ID": string | null,
  "Protein Name": string | null,
  "Gene Name from literature": string | null,
  "Source organism": string | null,
  "Primary biological role": string | null,
  "Subcellular localization (active site)": string | null,
  "Inducible by": string | null,
  "Post-translational modifications": string | null,
  "Literature_Name": string | null,
  "Experimental evidence": string | null,
  "Catalytic activity:": string | null,
  "Amino acids length:": int | null,
  "Protein family:": string | null,
  "Protein sequence:": string | null
}}

Rules:
1. If the field cannot be found, return `null` rather than guessing.
2. Copy the exact sentence or phrase from the text; do not rephrase.
3. For boolean/number fields, convert strictly (e.g., length → integer).
4. Output **only** the JSON object, no additional words.

Text:
===
{chunk}
===
"""

TEMPLATE_ZH = """你是一个生物医学实体抽取引擎。
根据以下文本，输出**仅**符合以下模式的JSON对象：

```json
{
  "蛋白质登录号或ID": string | null,
  "蛋白质名称": string | null,
  "文献中的基因名称": string | null,
  "来源生物体": string | null,
  "主要生物学功能": string | null,
  "亚细胞定位（活性位点）": string | null,
  "诱导物": string | null,
  "翻译后修饰": string | null,
  "文献名称": string | null,
  "实验依据": string | null,
  "催化活性": string | null,
  "氨基酸长度": int | null,
  "蛋白质家族": string | null,
  "蛋白质序列": string | null
}
```

规则：
1. 如果找不到该字段，返回`null`而不是猜测。
2. 直接复制文本中的句子或短语；不要改写。
3. 对于布尔值/数字字段，严格转换（例如，长度→整数）。
4. 输出**仅**为JSON对象，不要添加其他文字。

文本：
===
{chunk}
===
"""

PROTEIN_ANCHOR_PROMPT = {
    "en": TEMPLATE_EN,
    "zh": TEMPLATE_ZH,
}
