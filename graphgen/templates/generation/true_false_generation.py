TEMPLATE_TF_ZH: str = """请根据上下文资料生成独立的知识判断题，每个判断题包含一个陈述句，答案只能是正确(True)或错误(False)。

生成要求：
1. **语言一致性**：若上下文资料为中文，则生成中文问题；若为英文，则生成英文问题
2. **数量**：每个上下文资料生成{num_of_questions}个判断题
3. **独立性**：每个问题必须完整独立，不依赖其他问题
4. **准确性**：正确答案必须能从原文直接得出，陈述需有明确的判断依据

输出格式：
<qa_pairs>
<qa_pair>
<question>陈述句文本</question>
<answer>True或False</answer>
</qa_pair>
</qa_pairs>

示例（根据iPad Air 2生成2题）：
<qa_pairs>
<qa_pair>
<question>iPad Air 2于2014年发布。</question>
<answer>True</answer>
</qa_pair>
<qa_pair>
<question>iPad Air 2搭载的是A10处理器。</question>
<answer>False</answer>
</qa_pair>
</qa_pairs>


上下文资料：
{context}

请为以下资料生成{num_of_questions}个判断题：
"""


TEMPLATE_TF_EN: str = """Generate independent true/false questions based on the provided context. \
Each question should be a factual statement that can be clearly determined as true or false.

Requirements:
1. **Language Consistency**: Generate in the same language as the context (Chinese/English)
2. **Quantity**: Generate {num_of_questions} true/false questions per context
3. **Independence**: Each question must be self-contained
4. **Accuracy**: Correct answer must be directly derivable from the text with clear evidence

Output Format:
<qa_pairs>
<qa_pair>
<question>Statement text</question>
<answer>True or False</answer>
</qa_pair>
</qa_pairs>

Example (2 questions):
<qa_pairs>
<qa_pair>
<question>The iPad Air 2 was released in 2014.</question>
<answer>True</answer>
</qa_pair>
<qa_pair>
<question>The iPad Air 2 uses an A10 processor.</question>
<options>True
False</options>
<answer>False</answer>
</qa_pair>
</qa_pairs>

Context:
{context}

Please generate {num_of_questions} true/false questions for the following context:
"""


TF_GENERATION_PROMPT = {"zh": TEMPLATE_TF_ZH, "en": TEMPLATE_TF_EN}
