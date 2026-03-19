# KG-到-QA-到-VLM 训练价值评估闭环计划

## 导读

本文聚焦评估闭环。  
GraphGen 如果要长期演进，就不能只靠少量样例观察“看起来更好了”，而必须建立一套从 KG 质量到 QA 质量，再到 VLM 训练价值的 benchmark 与审计体系。

## 1. 当前问题

### 1.1 评估入口已经有，但覆盖面不够

当前仓库已有：

- KG 结构评估
- QA 评估
- triple 评估

QA 侧已有：

- `length`
- `mtld`
- `reward_score`
- `uni_score`

这些指标很有价值，但仍不足以回答以下关键问题：

- 这个 QA 是否 grounded？
- 这个问题是否真的可答？
- 这个 multihop 是否真的需要多跳？
- 这个 VQA 是否真的依赖 image/table？
- 这批数据是否更有利于领域知识学习？

### 1.2 当前缺少 gold set 与持续对比基线

如果没有固定 benchmark：

- 每次改 pipeline 只能靠抽样感受
- 很难分辨“质量变高”还是“风格变了”
- 很难比较不同阶段 roadmap 的真实收益

## 2. 目标状态

评估闭环的目标状态应当是：

1. KG 质量有一组稳定指标。
2. QA/VQA 质量有一组与 grounding、深度和跨模态直接相关的指标。
3. 有小规模高质量 gold set 作为回归基线。
4. 每次 pipeline 改动都能固定跑 benchmark 和样本审计。

## 3. 建议改动

### 3.1 KG 评估扩展

当前 `structure` 指标应继续保留，但还需要增加更贴近训练型数据质量的指标。

建议新增规划指标：

- `evidence_coverage`
- `evidence_validity`
- `entity_normalization_consistency`
- `cross_modal_alignment_precision`
- `relation_conflict_rate`

这些指标分别关注：

- 图中实体/关系被证据支持的覆盖程度
- 证据是否能回指和自洽
- 实体规范化是否稳定
- 跨模态边是否准确
- merge 后冲突是否仍然严重

### 3.2 QA/VQA 评估扩展

当前 QA 评估已有基础语言和 reward 指标，但还需要补齐面向训练价值的质量指标。

建议规划以下指标：

- `groundedness_score`
- `answerability_score`
- `hop_validity_score`
- `multimodal_dependency_score`
- `domain_learning_value_score`

它们的重点分别是：

- 是否真正被证据支持
- 是否能在给定上下文中闭合回答
- 是否真的满足多跳条件
- 是否真的依赖多模态证据
- 是否对专业知识学习有高价值

### 3.3 小规模 gold set 与审计集

建议在 `tests/fixtures` 或 `examples/input_examples` 邻近位置规划一组小规模 benchmark 数据。

建议覆盖的样本类型：

- DRAM 图文页面
- 带表格参数页面
- 多章节定义与约束页面

每类至少应有：

- gold entities / relations
- gold evidence span
- gold QA / VQA 样例

目的不是一开始就覆盖全部场景，而是先构建一组可回归、可手工审计的基准集。

### 3.4 数据生成闭环

建议把每次 pipeline 变更后的固定流程写进文档：

1. 跑 KG benchmark
2. 跑 QA benchmark
3. 跑样本抽检报告
4. 对比历史基线

这样“提高质量”才会变成可量化过程，而不是只看几条漂亮样例。

## 4. 公共接口与指标命名目标

后续建议把这些 metric 名称作为正式预留：

- `groundedness`
- `answerability`
- `hop_validity`
- `alignment_precision`
- `evidence_coverage`

评估接口仍建议沿用当前：

- `evaluate.target: qa | kg | triple`

后续只是在 `metrics` 维度逐渐扩展，不必推翻现有入口。

## 5. 分阶段实施建议

### 阶段 A：指标定义

先把每个指标的含义、理想方向和人工审计标准写清楚。

### 阶段 B：小规模 benchmark 建立

优先建立少量但高质量的 gold set，而不是一开始追求大而全。

### 阶段 C：自动评估接入

把指标逐步接入 `evaluate` 流程和实验报告流程。

## 6. 阶段验收

建议至少做到以下几点：

- 每次改动后都能和固定 benchmark 比较
- 能判断 groundedness 是否提升
- 能判断跨模态对齐是否更准
- 能判断 multihop 样本是否更真实
- 能判断样本是否更聚焦专业学习价值

## 7. 结论

评估闭环的价值，不只是给项目增加更多分数，而是把“感觉更好”变成“可以证明更好”。对 GraphGen 这种数据工程系统来说，benchmark 不是附属品，而是后续所有 roadmap 阶段持续迭代的共同基础。
