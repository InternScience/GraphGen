TEMPLATE_ZH = """
【任务】通过跨领域类比解释技术概念。

【类比原则】
- 类比源领域：生物学、物理学、建筑学、经济学、烹饪等领域
- 类比强度：类比关系需直观且深刻，避免牵强附会
- 目标：降低理解门槛，同时保持技术严谨性

【核心要求】
1. 双轨并行：每个技术概念配一个恰当类比
2. 类比结构：
   - 先介绍技术概念（准确、完整）
   - 再引入类比对象及其映射关系
   - 最后说明类比局限性和适用范围
3. 保真红线：技术部分必须与原文完全一致，不得因类比而简化
4. 创新性：鼓励使用新颖、出人意料但合理的类比
5. 篇幅：可比原文扩展20-40%

【评估标准】
- 类比恰当性（技术概念与类比对象的核心机制必须同构）
- 技术准确性（不得扭曲事实）
- 启发性（帮助读者建立深层理解）

原文内容：
{text}

请输出跨领域类比版本：
"""

TEMPLATE_EN = """
【Task】Explain technical concepts through cross-domain analogies.

【Analogy Principles】
- Source Domains: Biology, physics, architecture, economics, cooking, etc.
- Strength: Analogy should be intuitive yet profound, avoid forced comparisons
- Goal: Lower understanding barrier while maintaining technical rigor

【Core Requirements】
1. Dual Track: Pair each technical concept with an appropriate analogy
2. Analogy Structure:
   - First introduce technical concept (accurate and complete)
   - Then introduce analogy object and mapping relationship
   - Finally explain analogy limitations and applicable scope
3. Fidelity Baseline: Technical parts must be identical to original, no simplification for analogy's sake
4. Innovation: Encourage novel, surprising but reasonable analogies
5. Length: May expand 20-40% beyond original

【Evaluation Criteria】
- Analogy Appropriateness (core mechanisms must be isomorphic)
- Technical Accuracy (no factual distortion)
- Heuristic Value (helps build deep understanding)

Original Content:
{text}

Please output the cross-domain analogy version:
"""

CROSS_DOMAIN_ANALOGY_REPHRASING_PROMPTS = {
    "zh": TEMPLATE_ZH,
    "en": TEMPLATE_EN,
}
