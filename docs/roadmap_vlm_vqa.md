# GraphGen VLM VQA Roadmap

## 导读

本文是 GraphGen 面向专业文档 VLM 训练数据生成的顶层路线图。目标不是泛泛提升 QA 数量，而是围绕 DRAM、芯片、存储体系等专业材料，稳定生成高质量、低幻觉、强 grounding、跨模态强对齐、具备多层深度的训练型 VQA 数据。

这里的“训练型 VQA 数据”包括但不限于：

- `atomic`
- `multihop`
- `aggregated`
- `vqa`

整体判断是：GraphGen 已经具备从配置驱动 DAG、树结构解析、KG 构建、分区和 QA 生成的一整套骨架，但要真正支撑专业领域 VLM 学习，还需要从“树结构保真 + 证据约束 + 跨模态对齐 + 难度控制 + 评估闭环”五条线继续深化。

## 1. 总体目标

项目的终极目标可以概括为一句话：

从专业技术文档中自动构建高可信局部知识上下文，并据此生成适合 VLM 学习领域知识的高价值 VQA 样本。

这里至少包含五个子目标：

1. 让输入文档的结构被尽量保留，而不是被粗糙 chunk 打散。
2. 让图中的实体、关系和证据具备更高可信度，降低 hallucination。
3. 让 image、table 与 text 的联系从“共现”升级为“有效且可解释的对齐”。
4. 让问题深度不只靠 prompt，而是由图结构和证据结构共同控制。
5. 让整个数据构建流程变得可度量、可比较、可迭代优化。

## 2. 当前基础能力

目前仓库已经具备一些很重要的基础设施，这也是后续 roadmap 能成立的前提。

### 2.1 输入组织与结构保真

当前系统已经有 tree pipeline：

```text
read -> structure_analyze -> hierarchy_generate -> tree_construct -> tree_chunk
```

它可以把 markdown 或 MoDora 风格文档转成：

- text component
- table component
- image component
- 带 `path`、`level`、`parent_id` 的树节点和 chunk

这意味着系统已经从“纯字符分块”迈向“结构感知分块”。

### 2.2 建图与 evidence 基础能力

当前系统已经有：

- `build_kg`
- `build_tree_kg`
- `build_grounded_tree_kg`

其中 grounded tree KG 已经支持：

- `require_entity_evidence`
- `require_relation_evidence`
- `validate_evidence_in_source`

说明 evidence 不再只是文档说明性元数据，而已经开始进入实际过滤逻辑。

### 2.3 分区与生成能力

当前已有的生成模式包括：

- `atomic`
- `multihop`
- `aggregated`
- `vqa`
- 以及多选、填空、判断等其他 QA 模式

当前已有的分区能力包括：

- `dfs`
- `bfs`
- `anchor_bfs`

这意味着系统已经能通过不同社区划分策略，为不同题型提供不同局部上下文。

### 2.4 评估基础

当前仓库已有：

- KG 结构评估
- QA 质量评估
- triple 准确性评估

QA 侧已有的基础指标包括：

- `length`
- `mtld`
- `reward_score`
- `uni_score`

虽然这还不够，但至少说明系统已经有评估入口，而不是完全没有闭环。

## 3. 当前核心短板

尽管基础设施已经具备，但要面向 DRAM 等高专业度材料构建训练型 VQA 数据，现阶段还有几个明显短板。

### 3.1 KG 质量仍受单轮抽取约束

当前文本 KG 的核心仍然依赖单轮抽取加 merge。虽然有循环式 gleaning 和 evidence 过滤，但总体上仍缺少：

- 领域 schema 约束
- relation cross-check
- 术语标准化
- 冲突审计与重写

这会导致实体碎片化、关系命名不稳定、局部 hallucination 难以及时拦截。

### 3.2 evidence 还没有形成统一的数据层设计

当前 `evidence_span` 已经可用于 grounded tree KG 和 VQA prompt，但它仍然偏单字符串：

- 没有 evidence 类型层次
- 没有 evidence 质量分
- 没有多模态证据融合结构
- 没有显式冲突管理

这意味着 evidence 已经进入流程，但还没有成为第一等数据对象。

### 3.3 image/table 与 text 的联系还不够“有效”

当前 image/table 更多还是：

- 被识别成特殊模态节点
- 在 partition 时作为 anchor
- 或通过 caption/body 提供抽取上下文

但这种联系很多时候仍然偏“邻近共现”，还没有形成强约束的语义对齐边。因此图表虽然被纳入图中，但与正文的联系还不够细，不足以支撑更高质量的跨模态推理。

### 3.4 问题深度更多依赖 prompt，而不是结构约束

`multihop` 和 `aggregated` 当前已经存在，但其深度主要还是来自生成 prompt 的指令，而不是来自：

- 显式的 hop 数约束
- 多证据源约束
- 多模态覆盖约束
- 局部图路径可验证性约束

因此生成结果可能“看起来很深”，但推理链并不总是稳定。

### 3.5 缺少面向 VLM 学习收益的评估闭环

当前评估已经有基础指标，但还缺少更关键的问题：

- 这个 QA 是否真的 grounded？
- 这个多跳问题是否真的需要多跳？
- 这个样本是否真的依赖 image/table？
- 这批样本是否更有利于领域知识学习？

如果没有这类指标，后续优化就容易陷入“样例看起来不错，但整体收益不可控”的问题。

## 4. 三阶段路线图

后续 roadmap 建议拆成三个主阶段，外加一条贯穿始终的评估线。

### P0：KG Grounding 与 Hallucination 控制

这是最高优先级阶段。

目标：

- 让实体、关系和证据更加稳定
- 让幻觉在抽取阶段就被尽量压低
- 让后续 QA/VQA 的 answerability 和 groundedness 更可靠

重点包括：

- 分层 KG 抽取
- evidence 升级
- 术语归一化
- relation verifier
- tree + graph 结构边增强

没有这一阶段，后面的跨模态对齐和问题深度提升都会被底层噪声限制。

### P1：跨模态对齐增强

在 P0 基础上，把 image/table 与 text 的联系从“局部共存”升级为“可解释、可打分、可约束的对齐关系”。

目标：

- 让图像和表格真正参与推理
- 让 VQA 具备更真实的跨模态依赖
- 让 table/image 不只是 anchor，而是证据节点

重点包括：

- 跨模态关系类型扩展
- 表格细粒度建图
- image 细粒度 grounding
- tree 限域 + graph 扩张的联合分区

P1 做好以后，P2 的问题深度控制才会更真实、更可解释。

### P2：问题深度与训练价值优化

在前两阶段的基础上，进一步把 `atomic / multihop / aggregated / vqa` 变成一套分层、稳定、对专业知识学习高效的样本体系。

目标：

- 控制不同题型的深度和结构
- 减少“伪多跳”“伪跨模态”“伪高难题”
- 提高样本对专业概念学习的密度和效率

重点包括：

- 题型层次化设计
- 难度从 prompt 驱动升级到图驱动
- DRAM 专业高价值问法模板
- 数据采样策略与比例控制

## 5. 评估作为贯穿全流程的支撑线

虽然主阶段分为 P0、P1、P2，但评估不应等到最后再做。更准确的定位是：

- 评估设计贯穿所有阶段
- 但在 P0 之后应优先补齐为一套可持续 benchmark

这样每个阶段都能用统一基线验证收益，而不是只靠少量示例比对。

## 6. 阶段依赖关系

各阶段之间的依赖关系如下：

### 6.1 P0 是基础层

没有 P0：

- P1 的跨模态对齐容易把错误关系连得更密
- P2 的深题会更容易建立在错误图之上

因此 P0 不是优化项，而是质量底座。

### 6.2 P1 是 P2 的放大器

P1 做好以后：

- multihop 可以更自然地跨 text + table / text + image
- aggregated 可以更稳定地聚合跨模态证据
- VQA 的“图文联合理解”才更真实

因此 P1 强化后，P2 的收益会更明显。

### 6.3 评估线在 P0 后应立即形成闭环

虽然评估设计应从现在就开始，但建议在 P0 之后立即形成：

- KG benchmark
- QA benchmark
- 抽样审计流程

这样 P1/P2 的每次迭代才有统一判断标准。

## 7. 关联专项文档

本总纲与以下四份专项计划配套使用：

- `docs/plan_kg_grounding.md`
- `docs/plan_multimodal_alignment.md`
- `docs/plan_question_depth.md`
- `docs/plan_eval_benchmark.md`

阅读顺序建议为：

1. 本文，明确整体路线和阶段顺序
2. `plan_kg_grounding`
3. `plan_multimodal_alignment`
4. `plan_question_depth`
5. `plan_eval_benchmark`

## 8. 结论

GraphGen 当前已经具备相当扎实的框架基础，尤其是 tree pipeline 和 grounded tree KG，为专业文档 VQA 数据生成打下了很好的底座。后续真正决定上限的，不再是“能不能生成 QA”，而是：

- KG 是否可信
- evidence 是否可验证
- image/table 是否被有效接入 text 推理链
- 题型深度是否可控
- 整个数据集收益是否可度量

因此，后续 roadmap 的核心方向不是继续堆更多 prompt，而是把 tree、graph、evidence、multimodal alignment 与 evaluation 组合成一条真正稳定的知识数据工程链路。
