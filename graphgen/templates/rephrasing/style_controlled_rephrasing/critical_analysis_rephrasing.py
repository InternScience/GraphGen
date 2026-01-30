TEMPLATE_ZH = """
【任务】以学术批判视角改写以下内容，形成技术评论文章。

【核心要求】
1. 语气风格：客观理性，第三人称学术视角，使用规范学术用语
2. 内容结构：
   - 准确总结原文核心方法/发现（占比40%）
   - 分析技术优势与创新点（占比20%）
   - 指出潜在局限性与假设条件（占比20%）
   - 提出可能的改进方向或未来工作（占比20%）
3. 引用规范：保留原文所有关键引用，采用标准学术引用格式
4. 事实准确性：不得歪曲或误读原文技术细节

【输出格式】
- 标题：原标题 + "：一项批判性分析"
- 段落：标准学术论文章节结构
- 字数：与原文相当或略长

原文内容：
{text}

请输出批判性分析改写版本：
"""

TEMPLATE_EN = """
【Task】Rewrite the following content from an academic critical perspective as a technical commentary.

【Core Requirements】
1. Tone: Objective and rational, third-person academic perspective, using standard academic terminology
2. Structure:
   - Accurately summarize core methods/findings (40% of content)
   - Analyze technical advantages and innovations (20%)
   - Identify potential limitations and assumptions (20%)
   - Propose possible improvements or future work (20%)
3. Citations: Retain all key references from original, using standard academic citation format
4. Factual Accuracy: Do not distort or misinterpret technical details

【Output Format】
- Title: Original Title + ": A Critical Analysis"
- Paragraphs: Standard academic paper structure
- Length: Similar to or slightly longer than original

Original Content:
{text}

Please output the critically analyzed rewrite:
"""

CRITICAL_ANALYSIS_REPHRASING_PROMPTS = {
    "zh": TEMPLATE_ZH,
    "en": TEMPLATE_EN,
}
