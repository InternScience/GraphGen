TEMPLATE_ZH = """
【任务】将技术文档重构为自然问答对话。

【对话设计原则】
- 对话角色：提问者（好奇心驱动的学习者） vs 解答者（专家）
- 问题序列：从基础概念→技术细节→应用实践→深度追问，逻辑递进

【核心要求】
1. 问题设计：
   - 每个问题必须源于原文知识点
   - 问题要具体、明确，避免空泛
   - 体现真实学习过程中的疑惑点
2. 回答规范：
   - 回答必须准确、完整，引用原文事实
   - 保持专家解答的权威性
   - 可适当补充背景信息帮助理解
3. 对话流畅性：问题间有自然过渡，避免跳跃
4. 覆盖度：确保原文所有重要知识点都被至少一个问题覆盖
5. 事实核查：回答中的技术细节、数据必须与原文完全一致

【输出格式】
Q1: [问题1]
A1: [回答1]

Q2: [问题2]
A2: [回答2]
...

原文内容：
{text}

请输出问答对话版本：
"""

TEMPLATE_EN = """
【Task】Reconstruct the technical document as a natural Q&A dialogue.

【Dialogue Design Principles】
- Roles: Inquirer (curious learner) vs. Expert (domain specialist)
- Question Flow: From basic concepts → technical details → practical applications → deep follow-ups, logically progressive

【Core Requirements】
1. Question Design:
   - Each question must originate from original content knowledge points
   - Questions should be specific and clear, avoid vagueness
   - Reflect points of confusion in the real learning process
2. Answer Specification:
   - Answers must be accurate and complete, citing original facts
   - Maintain authoritative expert tone
   - May supplement background information when helpful
3. Dialogue Fluency: Natural transition between questions, avoid jumping
4. Coverage: Ensure ALL important knowledge points from original are covered by at least one question
5. Fact Check: Technical details and data in answers must be identical to original

【Output Format】
Q1: [Question 1]
A1: [Answer 1]

Q2: [Question 2]
A2: [Answer 2]
...

Original Content:
{text}

Please output the Q&A dialogue version:
"""


QA_DIALOGUE_FORMAT_REPHRASING_PROMPTS = {
    "zh": TEMPLATE_ZH,
    "en": TEMPLATE_EN,
}
