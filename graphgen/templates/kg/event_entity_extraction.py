# pylint: disable=C0301
"""事件-实体抽取提示词（SAG extract_document v3.1 改编，MIT）。

来源：Zleap-AI/SAG `zleap/sag/prompts/extract_document.yaml`（arXiv 2606.15971）。
改编点（PMS project-query 场景）：
- 事件必须含完整项目编号锚点（P-/I- + ≥6 位数字），无锚事件直接不产出；
- 字段锁定 / 反查禁止 / 越界语义禁止（与 pms_policy 出题铁律同源）;
- 数字接地：事件中一切数字必须在原文出现；
- 只输出扁平事项（不用 children 层级），便于星型编码进图。
响应是严格 JSON 合同：禁合同外字段，无效字段/无锚/数字不接地一律丢弃并计数。
"""

TEMPLATE_ZH: str = """## Role

你是专业的 PMS 项目管理文档事项与实体提取器。输入是当前 Chunk 片段。你的输出用于
构建可追溯的事项—实体图谱，供后续生成"项目查询"问答题使用。

## 事项提取原则

- 按主题组织信息，用完整的主谓宾句陈述事实，禁止机械拼接原文。
- 一个事项应围绕一个清晰主题；content 应概括"发生了什么、涉及谁、结果如何"，
  保留必要的数字和结论。
- 可以调整语序、消解有充分依据的指代，但不得改变主语、动作性质、因果或完成状态。
- "计划、预计、建议、可能、正在、已经完成"必须严格区分。
- 不得把外部常识、文件名或模型推测写成原文事实。
- 每个事项必须锚定在一个明确项目上：content 中必须出现该项目的完整项目编号
  （形如 P-24084947、I-26015101，字母+横线+至少 6 位数字）。无法锚定到完整项目
  编号的事项不得输出。
- 项目名称在同系统可能不唯一：提及时必须连同完整项目编号一起出现。
- 字段锁定：只允许陈述项目编号、项目名称、项目阶段、最终用户、销售工程师、
  项目成员及其角色。合同/金额/交付/付款/物料/变更/风险/会议/附件信息不提取。
- 禁止反查表述：人员只作为"某项目的成员/某项目某角色由谁担任"陈述，
  不得写成"某人属于哪个部门"等组织反查事实。

## 实体提取原则

- 只提取理解事项所必需的实体；实体类型只能从输入 meta.entity_types 中选择，
  没有合适类型时不要自造类型。
- 并列实体必须拆分，例如"甲公司和乙公司"应为两个 organization 实体。
- description 必须描述实体在本事项中的角色、动作或关系，不能只重复实体名称。
- 代词只有在当前片段能唯一消歧时才替换；无法确定时保留原文表达。
- 同名实体保持一致；不同实体不得因名称相近而合并。

## 事实边界（fail-closed）

- 事项的每个事实、数字、结论都必须在当前 Chunk 中有直接依据。
- 事项 content 中出现的所有数字（含编号中的数字）必须能在原文中逐字找到；
  找不到就删掉该表述或放弃该事项。
- 当前片段信息不足以支撑独立事项时必须返回空 items，不能靠推测拼出事项。

## Task

按以下顺序执行，不能跳步：

1. **全局扫描**：识别主体、动作、对象、时间、地点、数据、结果与不确定性；
   页眉页脚、目录编号、广告、版权声明、乱码视为噪音。
2. **聚合或拆分判断**：优先级固定为 总结性内容聚合 > 总结后的详细展开分离 >
   普通内容按主题拆分；仅有名称或修辞差异不强行拆分。
3. **事项构建**：title 精炼表达主题与关键结论；content 独立可读、含完整项目
   编号锚点、禁止推理说明。
4. **实体提取**：只从该事项陈述中提取实体，保留原文正式名称、数值单位和时间说法。
5. **最终自检**：确认无原文之外的事实；确认每个事项含完整项目编号锚点；
   确认所有数字可在原文找到；确认输出只有统一合同允许的字段。
   全部片段为噪音或信息不足时返回 items: []。

## Input

输入 JSON：{"type": "request", "data": {"items": [{"id": 1, "content": "..."}],
"meta": {"entity_types": [{"type": "...", "description": "..."}]}}}

## Output requirements

- 只返回统一 JSON 合同，不要 Markdown、解释文字或合同外字段：
  {"type": "response", "data": {"items": [
    {"title": "...", "content": "...",
     "entities": [{"type": "...", "name": "...", "description": "..."}],
     "is_valid": true}
  ]}}
- 无有效事项时返回 {"type": "response", "data": {"items": []}}，不得返回 null
  或制造占位事项。
- 相对时间（如"下月"）保留原文，不自行换算。

给定输入：
{input_text}

输出：
"""

#: 紧凑 JSON 请求包装（与 SAG request 合同同形；PMS 单 chunk 单 items）
INPUT_FORMAT: dict = {
    "items_template": '{{"type": "request", "data": {{"items": [{{"id": 1, "content": {content!r}}}], '
    '"meta": {{"entity_types": {entity_types}}}}}}}',
}

#: PMS 实体类型集（SAG 5W1H 类型集收敛到 project-query 可查询维度）
PMS_ENTITY_TYPES: list[dict] = [
    {"type": "PROJECT", "description": "项目本体，含完整项目编号"},
    {"type": "PERSON", "description": "人员，项目成员或干系人"},
    {"type": "ROLE", "description": "项目内职务或角色，如二级项目经理、销售工程师"},
    {"type": "ORGANIZATION", "description": "组织、部门、公司或团队"},
    {"type": "PHASE", "description": "项目阶段或状态，如售前、执行"},
    {"type": "LOCATION", "description": "地点或区域"},
    {"type": "DATE", "description": "日期或时间"},
    {"type": "METRIC", "description": "数量、比例等指标"},
]

EVENT_ENTITY_EXTRACTION_PROMPT = {
    "en": TEMPLATE_ZH,
    "zh": TEMPLATE_ZH,
    "FORMAT": {
        "entity_types": None,  # 运行时注入 PMS_ENTITY_TYPES
    },
}
