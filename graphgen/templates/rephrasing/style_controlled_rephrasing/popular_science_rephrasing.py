TEMPLATE_ZH = """
【任务】将以下技术文档改写为面向普通读者的科普文章。

【核心要求】
1. 语言风格：生动活泼，避免冷僻专业术语；必须使用术语时，需用生活化比喻或类比解释
2. 内容保真：所有核心事实、数据和技术结论必须准确无误，不得篡改或过度简化
3. 叙事结构：采用"问题-发现-应用"的故事线，增强可读性
4. 读者定位：假设读者具有高中文化水平，无专业背景
5. 篇幅控制：可适当扩展，但每段聚焦一个核心概念

【禁止行为】
- 不得删除关键技术细节
- 不得改变原意或事实
- 避免使用"这个东西"、"那个技术"等模糊指代

原文内容：
{text}

请直接输出改写后的科普文章：
"""

TEMPLATE_EN = """
【Task】Rewrite the following technical document as a popular science article for general readers.

【Core Requirements】
1. Language Style: Lively and engaging; avoid jargon; when technical terms are necessary, explain with everyday analogies or metaphors
2. Content Fidelity: All core facts, data, and technical conclusions must be accurate. Do not distort or oversimplify
3. Narrative Structure: Use a "problem-discovery-application" storyline to enhance readability
4. Audience: Assume high school education level, no technical background
5. Length: May expand moderately, but each paragraph should focus on one core concept

【Prohibited】
- Do not remove key technical details
- Do not change original meaning or facts
- Avoid vague references like "this thing" or "that technology"

Original Content:
{text}

Please output the rewritten popular science article directly:
"""

POPULAR_SCIENCE_REPHRASING_PROMPTS = {
    "zh": TEMPLATE_ZH,
    "en": TEMPLATE_EN,
}
