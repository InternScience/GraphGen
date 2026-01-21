TEMPLATE_ZH = """
【任务】按技术发展史视角重构内容。

【叙事框架】
- 时间轴线：从起源→关键突破→当前状态→未来趋势
- 演进逻辑：揭示"技术瓶颈突破→新范式建立→新问题出现"的循环

【核心要求】
1. 时间准确性：所有时间点、版本号、发布顺序必须核实准确
2. 因果链：
   - 明确每个演进阶段的驱动力（理论突破/工程需求/硬件进步）
   - 指出技术演进的必然性与偶然性
3. 内容结构：
   - 背景与起源（技术诞生前的状态）
   - 关键里程碑（带具体时间）
   - 范式转移（革命性变化）
   - 当前成熟形态
   - 未来展望（基于原文技术路径）
4. 技术保真：所有技术描述必须与原文事实一致
5. 分析深度：不能仅罗列事实，必须揭示演进逻辑

【输出规范】
- 使用时间轴标记（如[2017]、[2020]）增强可读性
- 关键人物/团队需保留原名
- 禁止编造不存在的技术演进路径

原文内容：
{text}

请输出历史演进视角版本：
"""

TEMPLATE_EN = """
【Task】Reconstruct content from a technological history evolution perspective.

【Narrative Framework】
- Timeline: Origin → Key Breakthroughs → Current State → Future Trends
- Evolution Logic: Reveal the cycle of "technical bottleneck breakthrough → new paradigm establishment → new problems emerge"

【Core Requirements】
1. Temporal Accuracy: ALL dates, version numbers, and release sequences must be verified and accurate
2. Causality Chain:
   - Identify drivers of each evolution stage (theoretical breakthrough/engineering needs/hardware advances)
   - Point out inevitability and contingency of technical evolution
3. Content Structure:
   - Background & Origin (state before technology birth)
   - Key Milestones (with specific dates)
   - Paradigm Shifts (revolutionary changes)
   - Current Mature Form
   - Future Outlook (based on original's technical trajectory)
4. Technical Fidelity: ALL technical descriptions must be factually consistent with original
5. Analytical Depth: Must reveal evolution logic, not just list facts

【Output Specification】
- Use timeline markers ([2017], [2020]) for readability
- Keep original names of key people/teams
- DO NOT invent non-existent evolution paths

Original Content:
{text}

Please output the historical evolution version:
"""

HISTORICAL_EVOLUTION_PERSPECTIVE_REPHRASING_PROMPTS = {
    "zh": TEMPLATE_ZH,
    "en": TEMPLATE_EN,
}
