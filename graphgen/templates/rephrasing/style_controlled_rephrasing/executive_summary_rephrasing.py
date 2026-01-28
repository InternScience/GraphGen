TEMPLATE_ZH = """
【任务】为高管层撰写决策摘要。

【读者假设】
- 职位：CTO/技术VP/产品总监
- 核心关切：技术价值、资源投入、竞争壁垒、商业影响

【核心要求】
1. 信息密度：每句话必须传达战略价值
2. 内容优先级：
   - 核心技术突破与创新价值（必须）
   - 与竞品的差异化优势（必须）
   - 实施成本与资源需求（必须）
   - 潜在商业应用场景（必须）
   - 技术风险评估（可选）
3. 语言风格：金字塔原理，结论先行，数据支撑
4. 简洁性：控制在原文长度的30-50%
5. 事实准确性：所有数据、性能指标必须与原文完全一致

【禁用表达】
- 避免"可能"、"也许"等不确定表述
- 禁用技术细节描述（除非直接影响决策）
- 避免行话和缩写

原文内容：
{text}

请直接输出高管决策摘要：
"""

TEMPLATE_EN = """
【Task】Write an executive summary for C-suite decision-making.

【Audience Assumption】
- Position: CTO/VP of Engineering/Product Director
- Core Concerns: Technical value, resource investment, competitive moats, business impact

【Core Requirements】
1. Information Density: Every sentence must convey strategic value
2. Content Priority:
   - Core technical breakthrough and innovation value (MUST)
   - Differentiated advantages over competitors (MUST)
   - Implementation cost and resource requirements (MUST)
   - Potential business application scenarios (MUST)
   - Technical risk assessment (OPTIONAL)
3. Language Style: Pyramid principle - lead with conclusions, support with data
4. Conciseness: 30-50% of original length
5. Factual Accuracy: All data and performance metrics must be identical to original

【Prohibited Expressions】
- Avoid uncertain terms like "maybe," "perhaps"
- No deep technical details (unless directly impacting decision)
- No jargon or unexplained acronyms

Original Content:
{text}

Please output the executive summary directly:
"""

EXECUTIVE_SUMMARY_REPHRASING_PROMPTS = {
    "zh": TEMPLATE_ZH,
    "en": TEMPLATE_EN,
}
