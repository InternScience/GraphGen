# P0 实施清单：KG Grounding 与 Hallucination 控制

## 导读

本文把前面的 roadmap 和 `plans/kg_grounding.md` 继续收敛成一份可直接执行的 P0 实施清单。  
目标不是再补一层抽象计划，而是明确：

- 先改哪些模块
- 每项改动要解决什么问题
- 预期会影响哪些链路
- 如何验证这些改动真的提升了质量

本文默认优先支持这些路径：

- `tree_vqa_config.yaml`
- `tree_atomic_config.yaml`
- 后续树结构驱动的 `multihop / aggregated / vqa`

## 1. P0 的完成标准

P0 完成时，至少应满足以下结果：

1. KG 节点和边不再只依赖单轮抽取结果直接 merge。
2. evidence 不再只是单一 `evidence_span`，而具备兼容式证据包结构。
3. DRAM 等专业实体的命名更稳定，实体碎片化明显下降。
4. 关系在入库前经过一层显式 verifier，而不是只靠 confidence threshold。
5. 结构信息不只停留在 `path` 注入，而开始进入图结构本身。

## 2. 优先级排序

建议 P0 按以下顺序推进：

### P0-A：证据包兼容扩展

优先级最高，因为它对现有 schema 侵入最小，但能为后续 verifier、对齐和评估提供统一基础。

### P0-B：relation verifier

在 evidence 有统一载体后，尽快加入关系校验层，先压 hallucination。

### P0-C：实体归一化

在 relation verifier 基本稳定后，减少实体碎片化和路径不稳定。

### P0-D：结构先验边

最后把 tree 的结构信息从“文本上下文提示”升级为“可被图消费的结构边”。

## 3. 模块级改动清单

### 3.1 `graphgen/models/kg_builder/light_rag_kg_builder.py`

这是 P0 的主战场。

#### 改动 1：把 `extract()` 拆成 candidate 与 verified 两层语义

短期不一定要真正拆文件，但建议在实现上形成两段逻辑：

- candidate parse
- verification / filtering

建议最小改法：

- 保留当前 LLM 抽取逻辑
- 在 `extract()` 解析完 entity/relation 后，不立刻直接进入 `nodes/edges`
- 先构造 candidate dict，再交给内部 verifier 函数

建议新增内部函数：

- `_build_evidence_items(...)`
- `_verify_entity_candidate(...)`
- `_verify_relation_candidate(...)`

这样可以在不推翻当前 builder 结构的前提下，引入更清晰的校验层。

#### 改动 2：兼容式 evidence 扩展

当前保留：

- `evidence_span`

新增兼容式字段到 entity / relation 的 `metadata`：

- `metadata.evidence_items`

每条 evidence item 的第一版建议结构：

```python
{
    "evidence_text": str,
    "evidence_type": str,
    "source_chunk_id": str,
    "support_score": float,
    "modality": str,
}
```

第一版默认规则：

- 文本抽取统一记为 `evidence_type = "text"`
- `source_chunk_id = chunk.id`
- `support_score` 先从 relation confidence 映射，或先给固定值 `1.0`
- `modality` 默认为 `text`

#### 改动 3：关系 verifier

新增关系 verifier，最小目标是：

- 无 evidence 且 `require_relation_evidence=True` 时直接丢弃
- evidence 存在但 `validate_evidence_in_source=True` 且无法回指时丢弃
- relation description 与 evidence 明显不一致时，标记 `verification_status = "weak"` 或直接丢弃

建议新增 verifier 输出字段：

- `support_score`
- `evidence_count`
- `verification_status`
- `conflict_flags`

第一版不必全写入图存储，但至少先挂到 relation dict 中，供 merge 使用。

#### 改动 4：merge 阶段支持 canonical / aliases

当前 `merge_nodes()` 基本按字符串聚合 description。后续第一阶段建议兼容扩展：

- 新增 `canonical_name`
- 新增 `aliases`
- 新增 `normalization_confidence`

第一版可采用轻量规则：

- `canonical_name` 默认就是当前 `entity_name`
- `aliases` 收集大小写变体、缩写/全称变体
- `normalization_confidence` 初始为规则命中率或固定默认值

短期不要求实现复杂实体链接，但必须把字段先留出来。

### 3.2 `graphgen/models/kg_builder/mm_kg_builder.py`

这是 image/table grounding 的关键入口。

#### 改动 1：为 MM entity/relation 注入 evidence_items

image/table 抽取后也应统一生成 `metadata.evidence_items`，不要只让文本 KG 有扩展证据结构。

建议第一版映射：

- image caption -> `image_caption`
- image note -> `note`
- table caption -> `table_caption`
- table body -> `table_body`

#### 改动 2：增强 payload 解析结果的 modality 标记

建议在 IMAGE/TABLE 节点 metadata 中进一步统一：

- `modality`
- `source_chunk_id`
- `section_path`

这样后续跨模态对齐和评估会更容易接入。

### 3.3 `graphgen/operators/tree_pipeline/build_tree_kg_service.py`

这是 tree 与 KG 融合的枢纽。

#### 改动 1：保留 path 注入，同时准备结构边生成入口

当前 `_inject_tree_context()` 已经把 path 作为文本上下文注入。下一步建议在 `process()` 结束前，增加一个结构边生成钩子。

建议新增私有函数：

- `_build_structural_edges(chunks)`

第一版结构边可只做三类：

- `appears_under_section`
- `caption_of`
- `note_for`

这些边先按低侵入方式加入 edge list，并标记：

- `is_structural_edge = True`

#### 改动 2：来源字段统一

确保 tree KG 产出的节点和边，都能稳定追踪：

- `source_id`
- `source_trace_id`
- `path`

这对后续 evidence 回指和 benchmark 很关键。

### 3.4 `graphgen/operators/tree_pipeline/build_grounded_tree_kg_service.py`

这部分逻辑目前主要是默认打开 evidence 开关。P0 中建议把它继续作为“严格模式”的入口。

建议第一版不改 public interface，只在文档和实现约定中明确：

- 以后凡是需要高可信训练集构建，优先走 grounded tree KG
- 新增 verifier / evidence_items / structural edges 时，优先在 grounded tree 路径上先稳定

### 3.5 新增术语规则资产

建议新增一个轻量资产目录，例如：

- `docs/domain_schema_dram.md`
- 或后续转成 `graphgen/resources/domain/dram_terms.yaml`

第一阶段先不要求程序完全消费该配置，但文档中要明确第一批规范对象：

- DRAM 标准名
- 参数名
- 常见缩写与全称
- 常见模块部件名

## 4. 第一批 public interface 目标

P0 第一批建议落地这些兼容字段。

### 4.1 节点

- `entity_type`
- `entity_name`
- `canonical_name`
- `aliases`
- `normalization_confidence`
- `evidence_span`
- `metadata.evidence_items`

### 4.2 边

- `relation_type`
- `support_score`
- `evidence_count`
- `verification_status`
- `conflict_flags`
- `is_structural_edge`
- `evidence_span`
- `metadata.evidence_items`

## 5. 验证实验清单

P0 不建议一边改一边只看主观样例，建议至少固定以下验证。

### 5.1 KG 质量对比

给定一批 DRAM 文档，比较改动前后：

- 实体重复率
- 无 evidence relation 比例
- evidence 回指成功率
- relation conflict 比例

### 5.2 Tree grounded 路径优先验证

先固定在：

- `tree_vqa_config.yaml`
- `tree_atomic_config.yaml`

这两条链路上验证，因为它们最依赖 tree + evidence，最能体现收益。

### 5.3 QA 样本抽检

每次 P0 子阶段结束后，固定抽检：

- 20 条 atomic
- 20 条 vqa
- 20 条 multihop 或 aggregated

重点看：

- 问题是否仍然可答
- 答案是否更 grounded
- hallucination 是否减少
- 图表相关问题是否更可信

## 6. 推荐迭代顺序

建议按 4 个小迭代推进：

### Iteration 1

- 为 text/mm KG 都补 `metadata.evidence_items`
- 不改外部配置
- 先确保旧链路兼容

### Iteration 2

- 引入 relation verifier
- 给 relation 增加 `support_score / evidence_count / verification_status`

### Iteration 3

- 加入 `canonical_name / aliases / normalization_confidence`
- 先用规则式归一化，别急着上复杂实体链接

### Iteration 4

- 增加第一批结构边
- 在 grounded tree 路径上优先试运行

## 7. 与后续阶段的衔接

P0 完成后，P1 和 P2 就有了更可靠的基础：

- P1 可以直接复用 `evidence_items` 和 `is_structural_edge` 做跨模态对齐
- P2 可以利用 `support_score`、`required_evidence_count`、`canonical_name` 控制问题深度和题型质量

因此，P0 的设计要尽量兼容后续阶段，而不是做成一次性的临时修补。

## 8. 结论

P0 的关键不是“改多少文件”，而是把当前 builder 体系从“抽取后直接 merge”推进到“候选抽取 -> evidence 校验 -> 规范化 merge -> 可审计入图”的方向。  
如果只能先做一件事，建议先把 `evidence_items + relation verifier` 做出来，因为这是后续所有质量提升最直接、最可复用的基座。
