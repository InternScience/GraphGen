TEMPLATE_ZH = """
【任务】以领域专家视角进行深度技术剖析。

【读者定位】
- 目标读者：同领域高级工程师/研究员
- 预期效果：揭示技术细节、设计权衡与实现原理

【核心要求】
1. 技术精确性：
   - 使用精确的专业术语和符号表示
   - 补充技术背景、相关工作和理论基础
   - 必要时用公式或代码片段说明
2. 深度维度：
   - 算法复杂度分析
   - 系统架构设计权衡
   - 性能瓶颈与优化空间
   - 边界条件和异常情况处理
3. 内容扩展：可在原文基础上增加30-50%的技术细节
4. 语气：权威、严谨、逻辑严密

【输出规范】
- 保持原文所有事实准确无误
- 新增细节需符合领域常识
- 使用标准技术文档格式

原文内容：
{text}

请输出技术深度剖析版本：
"""

TEMPLATE_EN = """
【Task】Conduct an in-depth technical analysis from a domain expert perspective.

【Audience】
- Target: Senior engineers/researchers in the same field
- Goal: Reveal technical details, design trade-offs, and implementation principles

【Core Requirements】
1. Technical Precision:
   - Use precise technical terminology and notation
   - Supplement with technical background, related work, and theoretical foundations
   - Include formulas or code snippets when necessary
2. Depth Dimensions:
   - Algorithmic complexity analysis
   - System architecture design trade-offs
   - Performance bottlenecks and optimization opportunities
   - Edge cases and exception handling
3. Content Expansion: May add 30-50% more technical details than original
4. Tone: Authoritative, rigorous, logically sound

【Output Specification】
- Maintain 100% factual accuracy from original
- Added details must align with domain common knowledge
- Use standard technical documentation format

Original Content:
{text}

Please output the technical deep-dive version:
"""

TECHNICAL_DEEP_DIVE_REPHRASING_PROMPTS = {
    "zh": TEMPLATE_ZH,
    "en": TEMPLATE_EN,
}
