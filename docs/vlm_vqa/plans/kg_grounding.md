# KG Grounding 与 Hallucination 控制计划

## 导读

本文是 GraphGen 后续 roadmap 中优先级最高的专项计划，目标是提升 KG 构建质量、完善 evidence grounding，并系统抑制 hallucination。  
对当前项目而言，这不是一个局部优化项，而是后续跨模态对齐、深度问题生成和 VLM 训练收益提升的底座。

## 1. 当前问题

### 1.1 抽取结果仍偏“单轮生成 + merge”

当前 `LightRAGKGBuilder` 与 `MMKGBuilder` 已经可以完成：

- 实体抽取
- 关系抽取
- evidence 过滤
- merge

但整体流程仍主要建立在“一轮抽取结果足够可信”的假设上。对 DRAM 等专业文档来说，这会遇到几个问题：

- 专业实体命名容易碎片化
- 参数、标准、指标等类型边界不稳定
- relation type 受 prompt 输出波动影响较大
- 同一事实跨 chunk、多来源出现时缺少标准化仲裁

### 1.2 evidence 还没有成为统一的一等结构

当前 evidence 的主要载体仍是：

- `evidence_span`

它已经能被用于 grounded tree KG 的过滤与 VQA prompt 注入，但仍存在不足：

- 不能区分证据来自正文、caption、note 还是 table body
- 无法记录证据强度
- 无法显式表示多证据联合支持
- 无法为冲突关系提供证据级审计

### 1.3 幻觉控制缺乏显式 verifier

当前 hallucination 控制主要依赖：

- 抽取 prompt
- confidence threshold
- evidence 是否存在
- evidence 是否回指源文本

但还缺少专门的 verifier 层去回答：

- 这个 relation 是否被证据真正支持？
- 这个 relation description 是否夸大了证据含义？
- 两个 chunk 给出的 relation 是否互相冲突？

## 2. 目标状态

P0 的目标状态不是“完全无幻觉”，而是建立一套可工程化控制 hallucination 的质量框架。

目标包括：

1. 实体和关系抽取分层化，而不是一步到位。
2. evidence 从单字符串升级为可扩展的证据包结构。
3. 对专业术语做领域归一化，减少实体碎片化。
4. 在 merge 前后加入 verifier 和 conflict handling。
5. 把 tree 结构信息显式转化为结构先验边，增强后续 partition 与 generator 的可控性。

## 3. 建议改动

### 3.1 分层 KG 抽取策略

建议把当前“抽取 -> merge”的流程规划成更清晰的两段式或三段式。

#### 第一阶段：候选抽取

保持现有 LLM 抽取能力，但输出定位为 candidate，而不是最终 truth。

输出重点包括：

- candidate entity
- candidate relation
- 原始 evidence
- 初始 confidence

这一层重召回，不追求最终 precision。

#### 第二阶段：证据校验与标准化重写

新增 verifier 层，对 candidate 做二次判断：

- evidence 是否存在
- evidence 是否真的支持该 relation
- 描述是否超出了证据内容
- 命名是否符合领域规范

必要时让 verifier 只负责：

- 保留
- 重写
- 降权
- 丢弃

而不是再次自由生成完整图。

#### 第三阶段：merge 与冲突管理

在标准化后的实体和关系进入存储前，再做：

- canonical merge
- relation type 对齐
- 冲突计数
- 审计队列记录

### 3.2 轻量 schema 约束

对 DRAM 等领域，建议尽早建立轻量 schema，而不是完全让模型自由命名。

#### 建议的实体类型

- `COMPONENT`
- `PARAMETER`
- `METRIC`
- `STANDARD`
- `SIGNAL`
- `STATE`
- `FORMULA`
- `TABLE`
- `IMAGE`

这些类型不一定要求第一阶段全量实现，但至少应在计划中明确为目标 schema。

#### 建议的关系类型

- `part_of`
- `measured_by`
- `constrained_by`
- `compared_with`
- `depicted_in`
- `supported_by`
- `derived_from`

做 schema 的目的不是限制所有表达，而是降低：

- 同义 relation 反复改名
- 模糊 relation type 泛滥
- 下游 multihop 时路径语义不稳定

### 3.3 evidence 升级路线

建议把 evidence 的演进分成两个阶段。

#### 阶段一：兼容式扩展

短期不破坏当前 `evidence_span` 兼容性，继续保留：

- `evidence_span`

同时在 `metadata` 中新增兼容式结构：

- `metadata.evidence_items`

每条 item 计划包含：

- `evidence_text`
- `evidence_type`
- `source_chunk_id`
- `support_score`
- `modality`

其中 `evidence_type` 建议支持：

- `text`
- `table_caption`
- `table_cell`
- `image_caption`
- `note`
- `path_context`

#### 阶段二：证据结构上升为一等字段

当验证链路稳定后，再考虑把 evidence 从 `metadata` 提升为 KG 节点/边的正式结构字段，而不只是一段 span。

### 3.4 领域术语标准化

建议新增术语规范层，不再只依赖 merge 时的字符串聚合。

目标是处理这些问题：

- 缩写与全称割裂
- 参数名大小写不一致
- 同义词重复建点
- 专业术语在不同文档中的表达变体

建议规划配套资产：

- 术语规则表
- canonical 名称映射
- alias 列表
- 领域正则与词典

这部分可先以配置文件或文档规范形式存在，后续再变成正式组件。

### 3.5 relation verifier

建议增加一个显式 relation verifier 设计，作为 hallucination 控制核心。

它至少要处理：

- 证据缺失则降权或丢弃
- 证据存在但与 relation description 不一致则丢弃
- relation confidence 不只看抽取模型返回值，还结合证据可回指性
- 同一 relation 被多个来源支持时，提高 support_score

建议 verifier 输出这些信息：

- `support_score`
- `evidence_count`
- `verification_status`
- `conflict_flags`

### 3.6 多来源 merge 与冲突审计

对实体与关系都应规划冲突处理，而不是简单做并集。

#### 实体冲突处理

相同实体不同描述时：

- 优先保留证据更强的版本
- 优先保留来源更多的版本
- 必要时保留 canonical description + aliases，而不是粗暴拼接

#### 关系冲突处理

相同边如果出现不同 relation_type：

- 记录冲突次数
- 记录冲突来源
- 必要时送入审计队列

冲突不一定立刻阻断入库，但必须可追踪。

### 3.7 tree + graph 融合增强

当前 tree 信息主要通过 `path` 注入文本 chunk。后续建议继续规划“结构先验边”。

建议类型包括：

- sibling adjacency
- section ownership
- figure/table reference
- caption-of
- note-for
- appears-under-section

这些边不是事实边，而是结构边。它们的价值在于：

- 帮助 partition 更合理限定局部上下文
- 帮助 generator 理解 section 内聚性
- 帮助 multimodal alignment 建立可靠近邻关系

## 4. 公共接口与类型目标

本文建议在后续实现中逐步明确这些目标字段。

### 4.1 KG 节点

- `entity_type`
- `canonical_name`
- `aliases`
- `modality`
- `normalization_confidence`

### 4.2 KG 边

- `relation_type`
- `support_score`
- `evidence_count`
- `is_structural_edge`

### 4.3 Evidence

- `evidence_text`
- `evidence_type`
- `source_chunk_id`
- `support_score`
- `modality`

## 5. 分阶段实施建议

### 阶段 A：低侵入升级

先不大改 schema，只做兼容式增强：

- 保留 `evidence_span`
- 新增 `metadata.evidence_items`
- 增加 verifier 设计
- 设计术语规则表
- 增加冲突计数和审计日志

### 阶段 B：抽取链路升级

在验证阶段 A 有收益后：

- 引入候选抽取 + verifier 双阶段
- 引入更稳定的 canonical merge
- 引入结构先验边

### 阶段 C：字段正式升级

当 schema 和数据质量稳定后：

- 把 evidence 正式提升为结构字段
- 把 normalization 与 support 信息上升为标准节点/边属性

## 6. 阶段验收

每个阶段都应至少看这几类指标：

- 实体重复率是否下降
- 无证据关系比例是否下降
- evidence 回指成功率是否提升
- 冲突 relation 比例是否可观测
- grounded tree VQA 的 answerability 是否提升

## 7. 结论

P0 的本质不是“多做一些规则”，而是把 GraphGen 从“能抽图”提升到“能构建可审计、可验证、可支撑专业 VQA 的局部知识图”。只有底层 KG 和 evidence 更稳定，后续跨模态对齐、问题深度控制和 VLM 学习收益才有可靠基础。
