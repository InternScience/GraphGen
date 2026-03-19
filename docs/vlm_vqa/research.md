# GraphGen VQA / Atomic QA 研究文档

## 导读

本文聚焦四个层面：

1. `graphgen` 主框架如何从 YAML 配置驱动整个生成流程。
2. `examples/generate` 目录如何组织不同 QA 任务的示例配置。
3. `examples/generate/generate_vqa` 中 VQA 流程的执行原理、grounding 机制与输出行为。
4. 新增的 `examples/generate/generate_atomic_qa/tree_atomic_config.yaml` 到底在做什么，它与 tree VQA 和传统 atomic QA 的关系是什么。

核心结论是：GraphGen 本质上是一套“YAML 配置驱动的 Ray DAG + operator/service + generator/template + storage”的 QA 数据生成框架。配置文件定义执行图，`Engine` 负责调度，`operators` 负责数据变换与建图，`models/generator` 负责把局部图上下文转成具体 QA，KV/图存储负责缓存、血缘追踪与跨阶段复用。

## 1. 项目整体架构

### 1.1 入口链路

整个系统的统一入口是 `graphgen/run.py`。

它的职责很明确：

- 读取 `--config_file` 指向的 YAML。
- 从 `global_params.working_dir` 取工作目录，生成一个基于时间戳的唯一输出目录：`working_dir/output/<timestamp>/`。
- 初始化日志。
- 创建 `Engine(config, operators)`。
- 以一个空的 Ray dataset 作为初始输入，调用 `engine.execute(...)`。
- 在流程执行结束后，把本次实际使用的配置保存到输出目录中的 `config.yaml`。

因此，GraphGen 不是“写死流程”的脚本，而是“读取配置后执行一张计算图”的运行器。

### 1.2 `Engine` 的职责

`graphgen/engine.py` 是配置到执行的桥梁，主要负责以下事情：

- 用 `Config` / `Node` 模型校验配置合法性。
- 对 `nodes` 做拓扑排序，确保依赖关系正确。
- 根据每个节点的 `dependencies` 取上游 dataset；单依赖直接传递，多依赖则做 `union`。
- 根据节点的 `op_name`，从 `graphgen/operators/__init__.py` 中查到具体实现。
- 把 `global_params` 与当前节点自己的 `params` 做签名过滤后注入 operator，避免把无关参数乱传给实现类。
- 根据节点的 `type` 和 `execution_params` 决定用 `map_batches` 还是聚合式执行。
- 如果某个节点设置了 `save_output: true`，就把该节点输出写到 `working_dir/output/<timestamp>/<node_id>/`，随后再从 JSON 读回 Ray dataset，供后续阶段继续使用。

这意味着 YAML 中的每个节点都不是“描述性注释”，而是会被真正实例化和调度的 DAG 节点。

### 1.3 `operators` 注册与配置映射

`graphgen/operators/__init__.py` 把字符串形式的 `op_name` 映射到真实实现，例如：

- `read -> read`
- `chunk -> ChunkService`
- `build_kg -> BuildKGService`
- `generate -> GenerateService`
- `structure_analyze -> StructureAnalyzeService`
- `build_tree_kg -> BuildTreeKGService`
- `build_grounded_tree_kg -> BuildGroundedTreeKGService`

因此，YAML 里的：

```yaml
- id: generate
  op_name: generate
```

并不是泛泛地“调用某种生成逻辑”，而是会精确绑定到 `GenerateService`。

### 1.4 `BaseOperator` 的通用行为

几乎所有 service 类都继承 `graphgen/bases/base_operator.py` 中的 `BaseOperator`。这一层提供了非常关键的统一机制：

- 为每个 operator 初始化单独的 KV 存储 namespace。
- 通过 `_trace_id` 对输入与输出做内容级标识。
- 用 `_meta_forward` 和 `_meta_inverse` 维护“上游 trace_id 到下游 trace_id”的映射。
- 在 `split()` 中根据 KV 元数据判断哪些输入已经处理过，支持恢复与跳过重复计算。
- 在 `store()` 中统一保存本批结果和血缘映射。

这带来两个直接效果：

1. GraphGen 的每个阶段不是简单流式传值，而是带缓存和可追踪血缘的。
2. 中间节点不只是算完即丢，而是可以被恢复、索引和回查。

### 1.5 `global_params` 的实际意义

`global_params` 并不是装饰性配置，而是会影响多数节点的初始化行为：

- `working_dir`：决定日志、缓存、输出文件的根目录。
- `graph_backend`：决定图数据库后端，如 `kuzu` 或 `networkx`。
- `kv_backend`：决定 KV 存储后端，如 `rocksdb` 或 `json_kv`。

`Engine` 还会根据 operator 是否在签名里声明了 `kv_backend` / `graph_backend`，预先初始化对应的存储 actor。因此这些字段属于全局运行时基础设施配置。

## 2. `examples/generate` 配置目录的组织规律

`examples/generate` 目录的组织非常规律。每个子目录基本都对应一种 QA 数据生成模式，例如：

- `generate_vqa`
- `generate_atomic_qa`
- `generate_multi_choice_qa`
- `generate_true_false_qa`
- `generate_multi_hop_qa`

每个子目录通常包含三类内容：

- `README.md`：说明该模式的用途和基本用法。
- `*.sh`：一条简短的启动脚本，通常就是 `python3 -m graphgen.run --config_file ...`。
- `*.yaml`：一个或多个配置文件，定义完整 DAG。

这些 YAML 配置的共同结构是：

- `global_params`
- `nodes`

每个节点通常包含这些字段：

- `id`
- `op_name`
- `type`
- `dependencies`
- `params`
- `execution_params.replicas`
- `execution_params.batch_size`
- `save_output`

其中：

- `id` 是 DAG 内部节点名。
- `op_name` 决定绑定哪个 operator。
- `type` 决定是 source、map_batch 还是 aggregate 路径。
- `dependencies` 决定数据从哪几个上游节点过来。
- `params` 是节点自己的业务参数。
- `execution_params` 控制并发、副本数、batch 大小等执行层细节。
- `save_output` 决定该节点输出是否真正落盘。

### 2.1 两类典型链路

GraphGen 在 `examples/generate` 中大致体现出两种主流链路。

#### 传统链路

```text
read -> chunk -> build_kg -> partition -> generate
```

这条链路的思路是：

- 先读原始文档或样本。
- 对文本切块。
- 从 chunk 建 KG。
- 把整个图切成若干适合生成的数据社区。
- 再用 generator 根据每个社区生成 QA。

#### 树链路

```text
read -> structure_analyze -> hierarchy_generate -> tree_construct -> tree_chunk -> build_tree_kg/build_grounded_tree_kg -> partition -> generate
```

这条链路的思路是：

- 先把结构化 markdown 或 MoDora 风格内容解析成组件。
- 为组件补标题层级。
- 构造文档树。
- 以树节点为单位生成 path-aware chunk。
- 再对这些 chunk 建树感知 KG。
- 最后进行分区和 QA 生成。

树链路更适合“文档结构本身有意义”的输入，例如章节、图表、注释、标题层级都需要保留的资料。

## 3. VQA 流程原理

VQA 这里实际上存在两条相关但不完全相同的配置路径：

- `examples/generate/generate_vqa/vqa_config.yaml`
- `examples/generate/generate_vqa/tree_vqa_config.yaml`

前者是普通多模态链路，后者是树结构增强版。

### 3.1 普通 `vqa_config.yaml`

`vqa_config.yaml` 的节点链路是：

```text
read -> chunk -> build_kg -> partition -> generate
```

具体含义如下。

#### `read`

该配置的输入是：

```yaml
input_path:
  - examples/input_examples/vqa_demo.json
modalities:
  - text
  - image
```

这说明它面向的是 JSON 形式的多模态样本，而不是 markdown 文档。`read` 本身会根据文件后缀选择 reader，并在读入后为每条记录生成 `read-...` 前缀的 `_trace_id`。

#### `chunk`

`chunk` 对文本进行二次切分，参数为：

- `chunk_size: 1024`
- `chunk_overlap: 100`

也就是说，普通 VQA 流程默认仍以文本 chunk 为基本建图单位，而不是直接保留原文中的自然结构组件。

#### `build_kg`

`BuildKGService` 会把输入分成两类：

- `text_chunks`
- `mm_chunks`，即 `image` / `video` / `table` / `formula`

然后分别调用：

- `build_text_kg()`
- `build_mm_kg()`

其中：

- 文本 KG 由 `LightRAGKGBuilder` 抽取实体与关系。
- 多模态 KG 由 `MMKGBuilder` 抽取图像/表格中心实体及其关联关系。

最终图中节点会带有类似以下字段：

- `entity_type`
- `entity_name`
- `description`
- `evidence_span`
- `source_id`
- 对 IMAGE/TABLE 节点还可能有序列化后的 `metadata`

边则会带：

- `src_id`
- `tgt_id`
- `relation_type`
- `description`
- `evidence_span`
- `confidence`
- `source_id`

#### `partition`

普通 VQA 用的是：

```yaml
method: anchor_bfs
method_params:
  anchor_type: image
  max_units_per_community: 10
```

`AnchorBFSPartitioner` 的逻辑是：

- 先在图里找到 `entity_type` 中包含 `image` 的节点作为锚点。
- 以这些锚点为 seed，用 BFS 向周围扩张。
- 每个 community 最多包含 `max_units_per_community` 个“单位”，单位既可能是节点，也可能是边。

所以普通 VQA 的 community 不是任意随机切出来的，而是围绕图像节点展开的局部子图。这正适合“图像 + 周边文本事实”一起形成 VQA 上下文。

#### `generate`

最终的生成节点是：

```yaml
params:
  method: vqa
  data_format: ChatML
  min_question_length: 8
  min_answer_length: 2
  max_answer_length: 220
```

`GenerateService` 看到 `method: vqa` 后，会实例化 `VQAGenerator`，并把这些长度约束传入。

### 3.2 树版 `tree_vqa_config.yaml`

`tree_vqa_config.yaml` 的链路是：

```text
read -> structure_analyze -> hierarchy_generate -> tree_construct -> tree_chunk -> build_grounded_tree_kg -> partition -> generate
```

与普通 VQA 相比，最大的变化不在最后的 `generate`，而在前半段“输入如何被组织成建图上下文”。

#### `read`

输入不再是 JSON 多模态样本，而是 markdown：

```yaml
input_path:
  - tests/fixtures/tree_vqa_demo.md
```

这意味着系统并不依赖“上游已经帮你包装好图像字段”，而是试图从 markdown 文档中主动拆出 text、table、image 组件。

#### `structure_analyze`

`StructureAnalyzeService` 会把文档转成：

```text
type = component_pack
```

输出记录里最重要的是 `components` 数组。每个 component 已经是解析后的结构单元，例如：

- 文本段
- 表格块
- 图片块

它还会保留 `source_trace_id`，用来指向最初 `read` 阶段的原始文档。

#### `hierarchy_generate`

`HierarchyGenerateService` 并不生成新内容，它的核心工作是为每个 component 填充 `title_level`。如果原组件没有显式层级，就通过标题文本推断。

这一步的意义是：后续树构建时不需要再猜当前组件属于第几层标题。

#### `tree_construct`

`TreeConstructService` 会把扁平组件数组转成真正的树形结构：

- 根节点为 `root`
- 每个组件变成一个树节点
- 节点包含 `node_id`、`title`、`level`、`content`、`node_type`、`metadata`
- 同时计算 `path` 和 `parent_id`

因此，这一步之后，文档已经不再只是“一串 chunk”，而是一棵具有层级和路径的文档树。

#### `tree_chunk`

`TreeChunkService` 把树节点重新展开成下游可消费的 chunk 记录，但保留了树上下文。

它给每条 chunk 的 `metadata` 注入了：

- `path`
- `level`
- `node_id`
- `parent_id`
- `source_trace_id`

在 `tree_vqa_config.yaml` 中还配置了：

```yaml
split_text_nodes: false
```

这很关键。它表示对于 `node_type == text` 的树节点，不再像传统链路那样做二次分块，而是直接把原树节点内容作为一个 chunk 输出。对结构化 markdown 来说，这能保留预先分好的段落、图表邻近说明和组件边界。

#### `build_grounded_tree_kg`

这一步是 tree VQA 的核心差异点。

`BuildGroundedTreeKGService` 继承自 `BuildTreeKGService`，但默认强制开启了：

- `require_entity_evidence = True`
- `require_relation_evidence = True`
- `validate_evidence_in_source = True`

也就是说：

- 没有 `evidence_span` 的实体会被丢掉。
- 没有 `evidence_span` 的关系会被丢掉。
- 即使有 `evidence_span`，如果该证据文本不真的出现在源 chunk 中，也会被丢掉。

此外，`BuildTreeKGService` 对文本 chunk 还有一个非常重要的增强：如果 metadata 里有树路径，它会把文本 chunk 变成类似下面的上下文再交给抽取器：

```text
[Document Path]
root/...

[Chunk]
原始内容
```

这相当于把“文档位置”也纳入了信息抽取上下文，有助于减少关系抽取时的语义漂移。

#### `partition`

树版 VQA 用的是：

```yaml
anchor_type:
  - image
  - table
```

这比普通 `vqa_config.yaml` 更激进，因为它不只围绕 image，还围绕 table 做 community 划分。对于图表密集的技术材料，这更符合实际使用场景。

#### `generate`

这里仍然是 `method: vqa`，所以最终还是由 `VQAGenerator` 生成 VQA，只是它拿到的图上下文已经比普通 VQA 更强地带有证据和文档结构信息。

### 3.3 `VQAGenerator` 的工作方式

`graphgen/models/generator/vqa_generator.py` 可以分成四层理解。

#### 1. 上下文构造

`VQAGenerator.build_prompt()` 调用 `context_utils.build_grounded_context()`，把 partition 产生的 community 渲染成两段文本：

- 实体列表
- 关系列表

每个实体和关系都可能带上：

```text
Evidence: ...
```

也就是说，`evidence_span` 不只是建图时的内部字段，而会被直接送进 VQA prompt，成为 LLM 生成问答时必须参考的硬证据。

#### 2. Prompt 约束

`graphgen/templates/generation/vqa_generation.py` 中的 prompt 明确要求：

- 生成 6 到 10 组 QA。
- 问题必须客观、可验证、避免主观臆测。
- 每个回答都要能被给定 evidence 支撑。
- 尽量覆盖实体识别、关系推理、数值读取、跨模态对齐。
- 对 DRAM / memory system 场景做了额外强化，优先关注结构、时序、性能、比较与 grounded evidence。

换句话说，这个 VQA prompt 不是“任意生成几个图文问答”，而是很明确地朝训练数据工程方向做了约束。

#### 3. 响应解析

响应必须满足：

```text
<question>...</question>
<answer>...</answer>
```

`parse_response()` 会提取出多组 QA。

#### 4. 后置质量过滤

`VQAGenerator` 与多数其他 generator 的最大区别之一，是它自己做了比较严格的后置过滤。它会剔除：

- 问题或答案为空。
- 问题长度太短。
- 答案长度太短或太长。
- 包含 `todo`、`placeholder`、`n/a` 等低质量标记。
- 包含 `unknown`、`不确定`、`无法确定` 等不可靠回答。
- 与已有 QA 在归一化后重复。
- 问答文本与上下文关键词完全无交集。

这使得 `VQAGenerator` 不只是“提示词模板 + 解析器”，而是带数据清洗能力的生成器。

### 3.4 图片路径如何进入最终样本

`VQAGenerator._extract_img_path()` 会遍历 community 中的节点，尝试从节点 `metadata` 中取出：

- `img_path`
- 或 `path`

这些 metadata 是从 IMAGE 节点保存下来的 JSON 中解析出来的。拿到图片路径后，`format_generation_results()` 会按输出格式分别处理：

- `Alpaca`：写入 `image` 字段
- `Sharegpt`：把图片挂到 human 消息 value 中
- `ChatML`：把图片挂到 user content 中

因此，tree/grounded 流程中保留模态 metadata 并不是多余操作，它直接影响最终训练数据是否还能关联到原图像。

### 3.5 为什么树版 VQA 更 grounded

从实现上看，tree VQA 比普通 VQA 更 grounded，原因不是单一的，而是多层叠加：

1. markdown 先被拆成 text/table/image 组件，而不是把整篇文档当普通文本。
2. 文本 chunk 会带树路径上下文进入 KG 抽取。
3. `build_grounded_tree_kg` 强制证据存在，并验证证据确实出现在源 chunk 中。
4. 节点与边的 `evidence_span` 会继续进入 VQA prompt。
5. `partition` 围绕 image/table 等锚点组织局部社区，使图文证据更局部、更可控。

所以 tree VQA 的 grounded 性并不是只靠 prompt 实现的，而是从输入解析、建图、分区到生成全链路强化出来的。

## 4. `tree_utils` 与树管线细节

`graphgen/operators/tree_pipeline/tree_utils.py` 是树链路最值得深入看的文件之一，因为它决定了 markdown 会被拆成什么样的结构。

### 4.1 `normalize_components()` 的作用

`normalize_components(doc)` 是 `structure_analyze` 的核心解析入口。它首先把原始内容标准化成字符串，然后调用 `_parse_markdown_components()` 做规则解析。

如果解析后没有任何结构组件，但文档本身有内容，它会退化为单一 `text` 组件，保持兼容性。

### 4.2 标题识别规则

标题判断由 `is_title_line()` 和 `infer_title_level()` 完成，支持三大类形式：

- Markdown 标题：`#` 到 `######`
- 数字编号标题：如 `1 Introduction`、`2.1 Memory Model`
- 中文章节标题：如“第一章”“第2节”

`infer_title_level()` 会把这些标题映射到树层级。这样，树构建不要求输入一定已经是严格规范的 markdown heading，也能兼容带编号或中文章节名的技术文档。

### 4.3 表格组件解析

`_parse_markdown_components()` 遇到以 `<table` 开头的块时，会把整段 `<table>...</table>` 收集为一个 table 组件。

它还会做一件很重要的事：尝试从表格前面最近的一段连续文本中提取 caption。如果这段文本形如：

```text
Table 1. ...
```

就会作为 `table_caption` 附着到该表格组件上。

表格组件的典型结构包括：

- `type: table`
- `title`
- `content`
- `title_level`
- `metadata.table_body`
- `metadata.table_caption`

其中 `content` 不是简单复制原 HTML，而是由：

- `[Table Caption]`
- `[Table Body]`

两部分拼接而成，便于后续给 LLM 作为抽取上下文。

### 4.4 图片组件解析

图片识别支持两类形式：

- markdown 图片：`![...](...)`
- HTML 图片：`<img src="...">`

解析后会提取出 `img_path`。此外，图片块之后的连续文字也会被进一步分析：

- 普通说明行会进入 `image_caption`
- 以 `Note:` / `Notes:` 开头的行会进入 `note_text`

因此 image 组件的典型结构包括：

- `type: image`
- `title`
- `content`
- `title_level`
- `metadata.img_path`
- `metadata.image_caption`
- `metadata.note_text`

`content` 本身通常由 caption 和 note 拼起来，这样即使后续不直接读取 metadata，也能从内容中看到图像说明。

### 4.5 `tree_construct` 与 `tree_chunk`

`TreeConstructService` 会把这些组件放进一棵真正的树中。每个树节点至少包含：

- `node_id`
- `title`
- `level`
- `content`
- `node_type`
- `metadata`
- `path`
- `parent_id`

`TreeChunkService` 则负责把树节点重新展开成下游 KG builder 可消费的记录。输出 chunk 的 `metadata` 中会统一注入：

- `language`
- `length`
- `path`
- `level`
- `node_id`
- `parent_id`
- `source_trace_id`

然后再把原始组件自己的 metadata 合并进去。

`split_text_nodes=false` 的意义尤其重要：它避免对已经是“良好结构单元”的 paragraph/image/table 组件再做二次切碎。对 MoDora 风格输入来说，这直接影响图文邻接关系是否会被破坏。

## 5. `tree_atomic_config.yaml` 的原理、作用与和 VQA 的差异

这是本文最关键的部分。

`examples/generate/generate_atomic_qa/tree_atomic_config.yaml` 的完整链路是：

```text
read -> structure_analyze -> hierarchy_generate -> tree_construct -> tree_chunk -> build_tree_kg -> partition -> generate
```

从形状上看，它明显复用了 tree VQA 的前处理链路，但它不是简单的“把 VQA 改个名字”，而是一个定位很特殊的混合配置。

### 5.1 它复用了树前处理链路

和 tree VQA 一样，`tree_atomic_config.yaml` 也会：

- 从 markdown 读入结构化文档。
- 通过 `structure_analyze` 拆出 text/table/image 组件。
- 通过 `hierarchy_generate` 填标题层级。
- 通过 `tree_construct` 组织成文档树。
- 通过 `tree_chunk` 保留树路径和组件边界。

而且它同样设置了：

```yaml
split_text_nodes: false
```

这说明它非常强调“保留原文结构”，不希望再把已经整理好的组件切散。

### 5.2 它与 `tree_vqa_config.yaml` 的关键差异

这两个配置最核心的区别有五个。

#### 差异 1：KG builder 不同

`tree_vqa_config.yaml` 使用的是：

```yaml
op_name: build_grounded_tree_kg
```

而 `tree_atomic_config.yaml` 使用的是：

```yaml
op_name: build_tree_kg
```

这意味着 `tree_atomic_config.yaml` 不会默认强制：

- 实体必须带 evidence
- 关系必须带 evidence
- evidence_span 必须真的出现在源文本里

所以它的 grounding 严格度天然弱于 tree VQA。

#### 差异 2：生成方法不同

tree VQA：

```yaml
method: vqa
```

tree atomic：

```yaml
method: atomic
```

前者调用 `VQAGenerator`，后者调用 `AtomicGenerator`。

#### 差异 3：输出格式不同

tree VQA 用的是：

```yaml
data_format: ChatML
```

tree atomic 用的是：

```yaml
data_format: Alpaca
```

也就是说，tree atomic 的最终样本默认会更像单轮文本监督数据，而不是多模态聊天格式。

#### 差异 4：KG 分区方式表面相同，但生成语义不同

tree atomic 仍然使用：

```yaml
method: anchor_bfs
anchor_type:
  - image
  - table
max_units_per_community: 10
```

这说明它并没有切换回“最小 atomic 社区”的典型做法，而是继续围绕 image/table 锚点做局部图扩张。

#### 差异 5：最终任务目标不同

tree VQA 追求的是一组多样化、带图像路径、经过质量过滤的多模态问答。

tree atomic 追求的是单个 QA、Alpaca 风格输出、保留树结构上下文，但并不自动附带 VQA 专有的多题、多模态格式和后过滤机制。

### 5.3 为什么说它不是传统意义上的“最小 atomic QA”

传统 atomic QA 的参考配置是 `examples/generate/generate_atomic_qa/atomic_config.yaml`，其链路是：

```text
read -> chunk -> build_kg -> partition -> generate
```

关键参数是：

```yaml
method: dfs
method_params:
  max_units_per_community: 1
```

这个配置非常接近“每个最小社区只包含一个单位”，因此更像原教旨的 atomic QA：对非常局部的事实单元生成一个 QA。

而 `tree_atomic_config.yaml` 并没有这么做。它仍然使用：

- `anchor_bfs`
- `anchor_type = [image, table]`
- `max_units_per_community = 10`

所以它生成时看到的上下文很可能是一个围绕图/表展开的局部子图，包含多个节点和边，而不是单一三元组或单一最小事实点。

因此，更准确的描述是：

它本质上是“树结构保真 + 图表锚定分区 + atomic 单问单答生成”的混合方案。

它输出的是单条 QA，但输入上下文并不一定是“原子级别”的。

### 5.4 它的真实用途

从配置设计上看，`tree_atomic_config.yaml` 更适合以下目标：

- 输入是结构化 markdown 文档，而不是现成 JSON 样本。
- 希望保留章节、段落、图表、说明文字等树结构边界。
- 希望围绕图片和表格附近的局部内容生成单条 QA。
- 最终输出想要 `Alpaca` 风格样本，方便做单轮监督微调。

换句话说，它很像“把 tree VQA 那套结构保真和图表局部建图能力，迁移到单问单答监督数据生成上”。

### 5.5 它的风险与限制

这个配置也有几个需要明确指出的限制。

#### 1. grounding 严格度弱于 tree VQA

因为用的是 `build_tree_kg`，而不是 `build_grounded_tree_kg`，所以它不会默认做最严格的 evidence 过滤。

#### 2. `AtomicGenerator` 功能更朴素

它没有：

- VQA 那样的多组 QA 输出能力
- 图片路径挂载逻辑
- 长度/重复/关键词 grounding 的后置质量过滤
- DRAM/VQA 场景增强 prompt

#### 3. community 仍依赖 image/table 锚点

如果文档中没有被抽取成 IMAGE 或 TABLE 的节点，那么 `anchor_bfs` 可能根本找不到锚点，进而产不出 community，最终生成阶段就没有输入。

所以它并不是“对任何树文档都稳妥通用”的 atomic 配置，而是偏向图表驱动的树文档。

## 6. `AtomicGenerator` 与普通 `atomic_config.yaml`

### 6.1 普通 `atomic_config.yaml`

普通 atomic 配置的链路是：

```text
read -> chunk -> build_kg -> partition(dfs, max_units_per_community=1) -> generate(method=atomic)
```

这条链路的设计意图很明确：

- 不强调文档树结构。
- 不强调图像或表格锚点。
- 更强调把 KG 切成尽量小的社区。

因此它更接近“一个最小事实片段生成一个 QA”。

### 6.2 `AtomicGenerator` 做了什么

`graphgen/models/generator/atomic_generator.py` 本身非常克制，主要做三件事：

1. 调用 `build_grounded_context()`，把当前 community 的实体与关系拼成文本上下文。
2. 用 `ATOMIC_GENERATION_PROMPT` 要求 LLM 只输出一个 `<question>/<answer>`。
3. 用正则解析这一个 QA。

它没有复杂的后处理逻辑，也没有 VQA 专用的数据增强。

### 6.3 与 `VQAGenerator` 的直接对比

`AtomicGenerator` 和 `VQAGenerator` 的差别非常实质：

- `AtomicGenerator` 只产一条 QA，`VQAGenerator` 产多条 QA。
- `AtomicGenerator` 没有图像字段处理，`VQAGenerator` 会尝试提取 `img_path`。
- `AtomicGenerator` 没有 VQA 专用 prompt 约束，`VQAGenerator` 有难度配比、grounding 要求和 DRAM 场景强化。
- `AtomicGenerator` 没有后置质量过滤，`VQAGenerator` 有比较严格的过滤逻辑。

输出格式上，`AtomicGenerator` 复用了 `BaseGenerator.format_generation_results()`，因此默认只能得到纯文本结构的：

- `Alpaca`
- `Sharegpt`
- `ChatML`

不像 `VQAGenerator` 那样会把图片字段注入最终结构。

## 7. 关键字段与数据流说明

要真正理解 GraphGen 的原理，必须看懂几个贯穿流程的字段。

### `_trace_id`

每个阶段的记录都会有 `_trace_id`。这是基于内容哈希生成的指纹，用于：

- 去重
- 缓存恢复
- 血缘追踪
- KV 存储索引

不同阶段的 `_trace_id` 前缀和内容都可能变化，因此它代表的是“该阶段产物”的身份，而不一定等于原始文档身份。

### `source_trace_id`

在树链路中，`structure_analyze`、`tree_construct`、`tree_chunk` 等阶段会用 `source_trace_id` 把当前记录挂回最初读入的文档。这是树链路保持“从树节点回到源文档”能力的关键。

### `source_id`

KG 节点和边在 merge 后会记录 `source_id`，它指向源 chunk，可以是一对多关系，并用 `<SEP>` 连接多个来源。

这使得：

- 一个实体可以追溯到多个 chunk
- 一个关系也可以追溯到多个 chunk

`BuildKGService` / `BuildTreeKGService` 还会把这些 `source_id` 继续转成 meta 映射，方便从 chunk 追踪到节点和边。

### `entity_type`

`entity_type` 决定一个节点在图中的语义类别，例如 IMAGE、TABLE 或普通文本实体。`AnchorBFSPartitioner` 正是通过检查 `entity_type` 是否匹配 `anchor_type` 来选锚点的。

所以 `entity_type` 不只是展示字段，它直接影响 partition 结果。

### `metadata`

对 IMAGE/TABLE 节点来说，`metadata` 非常关键。多模态 KG builder 会把原始组件 metadata 持久化进去，例如：

- 图片路径
- 图片 caption
- 表格 body
- 表格 caption

后续 VQA 再从这里取 `img_path` 回填到输出数据格式中。

### `evidence_span`

这是 grounded tree VQA 最重要的证据字段之一。

它贯穿三个阶段：

1. KG 抽取时生成或过滤 evidence。
2. KG merge 时保存在节点/边上。
3. VQA prompt 构造时通过 `Evidence:` 行再次送给生成模型。

如果没有这个字段，tree VQA 的 grounded 性就会明显下降。

### `save_output`

在当前这些示例配置里，`save_output: true` 只出现在 `generate` 节点，因此最终真正持久化到 `working_dir/output/<timestamp>/generate/` 的，是生成好的训练样本，而不是中间 KG 或树结构。

这也解释了为什么这些示例目录更像“数据生成流水线”，而不是“中间分析产物导出流水线”。

## 8. 结论与建议

### 8.1 VQA 与 tree atomic 的定位差异

两者虽然都可以建立在树结构和图表邻域之上，但定位明显不同。

`tree_vqa_config.yaml` 更适合：

- 追求严格 grounded 的 VQA 数据
- 需要 evidence 强校验
- 需要多条 QA
- 需要保留图像路径并输出多模态聊天格式

`tree_atomic_config.yaml` 更适合：

- 追求单轮 `Alpaca` 风格 QA 样本
- 想保留结构化 markdown 的树边界
- 想围绕图表局部上下文生成单条 QA
- 接受 grounding 严格度弱于 tree VQA

### 8.2 使用建议

如果目标是高质量、证据更严格、适合多模态训练的 VQA 数据，优先使用 `examples/generate/generate_vqa/tree_vqa_config.yaml`。

如果目标是把结构化 markdown 文档转成单轮监督样本，同时又不想破坏树结构和图表边界，可以使用 `examples/generate/generate_atomic_qa/tree_atomic_config.yaml`。

如果目标是尽可能接近“最小事实粒度”的 atomic QA，不应把 `tree_atomic_config.yaml` 误解为严格原子化配置，而应优先参考 `examples/generate/generate_atomic_qa/atomic_config.yaml` 中的：

```yaml
method: dfs
max_units_per_community: 1
```

因为真正决定“是否原子化”的，不只是 generator 名字是否叫 `atomic`，还包括 partition 策略是否把上下文压缩到了最小事实单元。
