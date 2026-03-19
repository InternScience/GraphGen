# 跨模态对齐增强计划

## 导读

本文聚焦 image、table 与 text 的“有效联系”。  
当前 GraphGen 已经能把 image/table 纳入图中，也能在 partition 时把它们当作 anchor 使用，但要真正服务高质量 VQA 和 VLM 训练，还需要把这些模态之间的联系从“局部共现”升级为“可解释、可验证、可打分的对齐关系”。

## 1. 当前问题

### 1.1 image/table 进入了图，但还没有被充分结构化

当前系统已经能：

- 从 markdown 中识别 image/table component
- 保留 `img_path`、caption、note、table body 等信息
- 让 image/table 在 KG 中成为特殊节点
- 在 partition 时把 image/table 作为 anchor

但这还不等于“有效跨模态对齐”。

现阶段的主要不足是：

- 图片和表格与正文的联系仍偏近邻共现
- 缺少显式对齐边类型
- 表格内部结构仍较粗
- 图片局部说明与正文引用关系仍未充分建模

### 1.2 当前 partition 仍容易引入无关跨章节上下文

虽然 tree pipeline 已经提供 `path`，但当前主流分区仍以图扩张为主。这样会有风险：

- 图扩张可能跨越无关 section
- 某些 image/table 可能被拉入语义上不相关的 text 节点
- 最终 VQA 看到的是“形式上邻近但并不构成有效证据”的上下文

## 2. 目标状态

跨模态对齐增强后的目标状态应当是：

1. image/table 不只是 anchor，更是可解释证据节点。
2. text、image、table 之间存在显式语义关系，而不只是被动共现。
3. 表格内部能支持更细粒度的数值、参数、比较与引用问题。
4. 分区时既利用 graph 连通性，也利用 tree 限域，减少错误扩张。
5. 后续 generator 能显式要求样本覆盖多模态证据，而不是碰运气。

## 3. 建议改动

### 3.1 新增跨模态对齐边类型

建议在规划中把以下关系类型视为重点候选：

- `illustrates`
- `mentions`
- `caption_for`
- `summarizes`
- `cell_supports`
- `parameter_shown_in`

这些边的意义分别是：

- `illustrates`：图像说明某个结构、流程或概念
- `mentions`：正文显式提到图或表中的对象
- `caption_for`：caption 与 image/table 的绑定
- `summarizes`：表格对正文结论的汇总
- `cell_supports`：具体单元格支持某一结论或参数
- `parameter_shown_in`：某参数或指标在某图表中被展示

这些边一旦被引入，image/table 在图中的角色就不再只是“一个模态节点”，而是可参与 reasoning 的证据节点。

### 3.2 表格细粒度建图

当前 table 基本上还是以：

- caption
- body

整体作为抽取输入。后续建议规划细粒度表格结构化。

#### 目标单元

- header
- row entity
- column metric
- cell value

#### 这样做的价值

- 能生成更高价值的数值比较题
- 能做“某参数在不同标准/配置下的差异”这类问题
- 能构造 text -> table -> parameter 的真实 multihop

短期不一定需要把每个单元格都入库，但应在计划里明确“表格不能长期只作为整块文本处理”。

### 3.3 图片细粒度 grounding

对 image component，建议把可稳定获取的信息明确纳入设计目标：

- caption
- nearby paragraph
- note
- referenced section title

其中：

- caption 和 note 是最直接的 image-local textual grounding
- nearby paragraph 和 section title 是更稳定的上下文限制信号

短期建议优先利用这些文档侧弱监督信号，而不是一开始就依赖重型视觉模型或 OCR region pipeline。

### 3.4 tree + graph 联合分区

建议在现有 `anchor_bfs` 之外规划一种联合分区模式。

核心思路：

1. 先按 tree section 限定候选范围。
2. 再在该局部范围内做 graph expansion。

这样做的好处是：

- 避免图扩张跨越无关章节
- 避免相邻但主题不同的图表被错误拉进同一个问题上下文
- 保留 tree 的局部一致性和 graph 的关系扩张能力

建议后续预留配置项：

- `partition.params.section_scoped`
- `partition.params.required_modalities`

### 3.5 “有效联系”的判定标准

后续需要明确：不是所有邻近 text 都应该连到 image/table 上。

建议引入两个评分概念：

- 对齐边置信度
- 跨模态支持分

这些分数的来源可综合：

- 文本显式引用
- caption/正文术语重合
- section 共属
- note 支持
- parameter 名称匹配

这些分数后续可以用于：

- filtering
- partition 约束
- generator 采样条件

## 4. 公共接口与类型目标

### 4.1 节点层目标

image/table 节点后续建议至少具备：

- `modality`
- `canonical_name`
- `metadata`
- `support_score`

### 4.2 边层目标

跨模态边后续建议具备：

- `relation_type`
- `support_score`
- `evidence_count`
- `is_structural_edge`

### 4.3 生成控制参数

后续建议预留：

- `generate.params.required_modalities`
- `generate.params.required_evidence_count`

让 generator 显式约束问题必须覆盖某些模态，而不只是默认接受任意社区。

## 5. 分阶段实施建议

### 阶段 A：弱监督对齐增强

优先利用现有稳定信号：

- caption
- note
- nearby paragraph
- section title
- path

建立初版跨模态边。

### 阶段 B：表格细粒度结构化

优先在 table 上做细粒度建图，因为对 DRAM 文档来说，表格往往比图像更容易直接提供高价值参数关系。

### 阶段 C：联合分区落地

在对齐边质量达到一定水平后，引入 tree 限域 + graph 扩张的新分区逻辑。

## 6. 阶段验收

每阶段建议至少验证：

- image/table 是否能与相关 text 建立更可解释的边
- 错误跨章节连接是否下降
- 生成样本中真正依赖多模态证据的问题比例是否上升
- 表格类样本中数值/参数/比较问题是否明显增加

## 7. 结论

跨模态对齐增强的目标不是让图里“多几个 image/table 节点”，而是让 image、table 与 text 形成真正可推理、可验证、可采样的证据网络。只有这样，后续的 multihop、aggregated 和高质量 VQA 才能从“看起来是多模态”变成“本质上依赖多模态”。
