TEMPLATE_ZH = """
【任务】将技术文档改写为第一人称实践经验分享。

【角色设定】
- 身份：资深技术实践者/研究员
- 场景：技术博客/内部经验分享会
- 目标读者：同行从业者

【核心要求】
1. 视角：全程使用"我/我们"第一人称
2. 内容融合：
   - 保留原文所有技术事实（代码、数据、架构）
   - 添加个人实践中的观察、挑战与解决思路
   - 分享真实应用场景和效果数据
3. 语言风格：专业但亲和，避免过度口语化
4. 叙事元素：可包含"最初尝试-遇到问题-调整思路-最终效果"的故事线
5. 事实红线：技术细节必须与原文完全一致，不得虚构数据

【禁止】
- 不得编造不存在的个人经历
- 不得改变技术实现细节

原文内容：
{text}

请直接输出第一人称叙事版本：
"""

TEMPLATE_EN = """
【Task】Rewrite the technical document as a first-person practical experience sharing.

【Role Setting】
- Identity: Senior practitioner/researcher
- Scenario: Technical blog/internal sharing session
- Target Audience: Peer professionals

【Core Requirements】
1. Perspective: Use first-person "I/we" throughout
2. Content Integration:
   - Retain ALL technical facts (code, data, architecture) from original
   - Add personal observations, challenges, and solution approaches from practice
   - Share real application scenarios and performance data
3. Language Style: Professional yet approachable, avoid excessive colloquialism
4. Narrative: May include "initial attempt-encountered problem-adjusted approach-final result" storyline
5. Factual Baseline: Technical details must be identical to original, no fabricated data

【Prohibited】
- Do not invent non-existent personal experiences
- Do not alter technical implementation details

Original Content:
{text}

Please output the first-person narrative version directly:
"""

FIRST_PERSON_NARRATIVE_REPHRASING_PROMPTS = {
    "zh": TEMPLATE_ZH,
    "en": TEMPLATE_EN,
}
