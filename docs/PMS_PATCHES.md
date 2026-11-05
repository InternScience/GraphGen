# PMS Fork 补丁台账

本仓库是 `open-sciencelab/GraphGen`（upstream，Apache-2.0）的自维护 fork，分支
`pms/main`。上游修复可通过 rebase 同步，冲突以本 fork 语义为准。PMS 主仓库只钉
住本 fork 的 commit SHA（receipt `_graphgen_source_identity` 自动记录）。

## #1 版本化 Prompt Profile（2026-09-18）

- **动机**：官方所有 prompt 是 `graphgen/templates/` 下的硬编码常量，无任何
  config 覆盖机制；此前 PMS 侧只能 monkeypatch `generator.build_prompt`。
- **实现**：
  - `graphgen/templates/prompt_profiles.py`：profile 加载器。profile = 内置 id
    （`graphgen/templates/profiles/<id>/`）或文件路径；每 method 一个
    `<template_key>.json`；结构与官方模板逐键校验（fail-closed）。
  - `graphgen/bases/base_generator.py`：`apply_prompt_profile()` +
    `template()`；generators 声明 `TEMPLATE_KEY` 并经 `self.template(...)`
    渲染；未应用 profile 时与官方常量逐字节一致。
  - 10 个 generate method（atomic/aggregated/multi_hop/cot/multi_choice/
    multi_answer/fill_in_blank/masked_fill_in_blank/true_false/vqa）全部接入。
  - `graphgen/operators/generate/generate_service.py`：generate 节点新增
    `prompt_profile` / `output_gate` / `llm_concurrency` 三个可选参数。
- **内置 profile**：`pms_policy.v2`（atomic + multi_hop 的 PMS project-query
  出题策略模板，正本迁移自 PMS 受控侧 v2 模板）。

## #2 确定性输出门（2026-09-18）

- `graphgen/operators/generate/gates.py`：注册表 + `pms_project_anchor` 门
  （题面必须含完整项目编号 P-/I-+≥6 位且答案非空，否则丢弃该 QA，不改写）。
- generate 节点经 `output_gate` 参数启用。

## #3 生成 LLM 并发封顶（2026-09-18）

- `graphgen/operators/generate/llm_concurrency.py`：事件循环级信号量，
  把同 loop 在途 generate 压到上限（35B-A3B 高并发实测会过载挂死）。
- generate 节点经 `llm_concurrency` 参数启用（PMS 默认 4）。

## #4 空响应有界重试（复位自丢失补丁，2026-09-18）

- `graphgen/models/llm/api/openai_client.py::generate_answer`：思考模型偶发
  仅返回 `<think>` 内容、过滤后为空串；`GRAPHGEN_EMPTY_RETRY=N`（默认 0，
  行为同官方）时对空响应做最多 N 次重试。历史版本此补丁仅存于 env 注入、
  源码实现已丢失，本条为正确复位。

## #5 空抽取 assistant prefill 重试（复位自丢失补丁，2026-09-18）

- `openai_client.generate_answer` 新增 `assistant_prefill` 参数；
  `schema_guided_extractor.extract` 在响应为空且
  `GRAPHGEN_EMPTY_EXTRACTION_PREFILL=1` 时，以确定性 prefill `("entity"<|>`
  重试一次。非空响应永不重试/改写。

## #6 logprob 调用关思考（复位自丢失补丁，2026-09-18）

- `generate_topk_per_token`：思考模型前若干 token 消耗在 `<think>` 内，导致
  judge/loss 的 next-token yes/no logprob 退化。`GRAPHGEN_DISABLE_THINKING=1`
  时对 logprob 调用注入 `extra_body.chat_template_kwargs.enable_thinking=false`
  （Qwen 兼容后端）。普通生成调用不受影响。

## #7 事件-实体星型图管线（Phase 2，2026-10-09）

SAG 思想移植（MIT，arXiv:2606.15971；参考源码 `workspace/sag_reference/`，方案见 PMS 主仓
`docs/2026-09-18-graphgen-phase2-sag-event-entity-plan.md` §3.2 复用地图）：

- `templates/kg/event_entity_extraction.py` 保留旧 Python 常量的兼容导出；实际策略由版本化 event profile 资源提供。
- `models/kg_builder/event_contract.py`：纯 stdlib 响应合同（禁合同外字段、is_valid 自报、
  无锚 fail-closed 丢弃并计数）+ 数字接地校验（SAG grounding 移植）+ 确定性事件 ID。
- `models/kg_builder/event_entity_kg_builder.py`：事件→星型编码（EVENT 节点 + 成员边，
  边属性含角色）。源实现对事件节点、参与者节点及成员边执行确定性合并；仅普通未标记边
  委托官方 LightRAG merge。**事件节点与事件成员边不走 LLM 摘要**（SAG 事件是事实文本，
  不能让 merge summarizer 把 event_id 扩写成“项目身份/分工总结”污染出题上下文）。事件
  描述限长 4,000 字符，且原文链以 `source_id` SEP 保持闭合。
- `operators/build_kg/`：`kg_method: event_entity` 分发（默认 light_rag 与官方逐字节一致）。
- `models/partitioner/event_join_partitioner.py`：共享实体 join + 同锚硬约束的事件簇分区
  （SAG join 语义离线物化），产出官方 `Community`，metadata 携带 event_ids。
- `models/generator/support.py` + atomic/multi_hop 解析 + GenerateService 校验：
  `<support>{"cited": [...]}` 引用必须存在于本分区节点集合（EVENT 节点名即事件审计
  ID，不依赖自定义 Community metadata 以保证 JSONL cache 往返一致），失败剥离保留题面。
- `templates/profiles/pms_policy.v3/`：v2 模板 + support 输出要求。

## #8 通用事件抽取 Profile 与多项目 SAG 能力

- 将事件抽取策略从 builder/契约代码中抽离为 `templates/event_profiles/<id>/` 下版本化 manifest + prompt。内置 `pms_event.v1` 保留原始 PMS prompt、8 类实体、P-/I- 锚点和旧事件 ID；`generic_event.v1` 与 `example_event.v1` 是机制示例/测试 fixture，不代表外部领域已验证策略。
- `templates/event_profile_loader.py` 校验 schema、实体类型、锚点 regex、prompt 占位符及图设置，暴露 manifest/prompt SHA-256；显式无效 profile fail-closed。profile 从包资源读取，不依赖当前工作目录。
- `build_kg.params.event_profile` 传入事件 builder。老 PMS 配置缺省解析为 `pms_event.v1`；新项目应显式传入自己的 profile。`light_rag` 默认保持不变。
- `event_contract.py` 仍保证严格响应 JSON、实体 allowlist 和可选数字接地；实体类型、锚点是否必需及锚点模式由 profile 驱动。无锚模式不自动合成 PROJECT/其他锚点实体。
- 事件节点写入显式 `anchor` 属性。`event_join` 按 `event_entity_type` 和 `anchor_attribute` 配置工作，`require_same_anchor` 控制同锚硬隔离；不再从 ID 推断锚点。PMS profile 使用旧 ID 以兼容既有图缓存，新 profile 使用不编码 domain anchor 的 ID。
- `setup.py` 与 `MANIFEST.in` 将生成 profile JSON、事件 manifest/prompt 作为 `graphgen` wheel/sdist 包资源。
- 多项目安装、profile authoring 与升级见 `docs/MULTI_PROJECT_SDK.md`；fork 不可变 tag 与发布门禁见 `docs/RELEASING.md`。

## 验收

- 默认路径（无 profile/gate/并发参数）与官方行为逐字节一致；
- PMS 主仓库 `tests/test_pms_graphgen_policy_generation.py`、
  `tests/test_run_graphgen_defaults.py` 为契约回归（rft 环境按路径加载
  纯 stdlib 模块校验）；
- 事件 profile 合同和真实生成由 GraphGen venv smoke 验收。

## #9 Ray LLM actor concurrency and logger context fallback (1.1.0)

- `LLMFactory.create_llm` reads `GRAPHGEN_LLM_ACTOR_MAX_CONCURRENCY`; default/`1` preserve Ray's serial async actor behavior, values greater than one pass an explicit bounded `max_concurrency` actor option.
- `get_current_logger` falls back to the standard `GraphGen` logger only when the ContextVar is absent. This does not catch or replace LLM/model errors and does not fabricate generation output.
- Covered by `tests/unit_tests/utils/test_init_llm_actor_concurrency.py` and `tests/unit_tests/utils/test_log.py`.

## #10 Event extraction prompt: no example-id copying (1.1.1)

- `templates/event_profiles/pms_event.v1/prompt.zh.txt` requires the anchor to be taken verbatim from the fragment and forbids copying the illustrative id, matching the generation-policy hardening in PMS `pms_policy.v2_2`.

## #11 畸形响应按空响应处理（2026-10-30）

- `openai_client.generate_answer`：思考模型把 `max_tokens` 全花在 `<think>` 内时，
  部分 OpenAI 兼容后端返回 `choices[0].message.content = None`；网关内容过滤偶发
  返回空 `choices`。官方实现直接 `filter_think_tags(completion.choices[0].message.content)`，
  两种情况分别抛 `TypeError` / `IndexError`。
- 这两类异常既非传输错误也非格式错误：池的可重试分类器判为不可重试，在**第一次
  尝试**就 fail-closed（跨源 failover 完全失效），上层 `run_concurrent` 再把该 chunk
  静默丢弃，造成数据缺口。
- 修复：读取 `content` 前先判 `choices` 是否为空，`None` 归一为 `""`，交给既有的
  `GRAPHGEN_EMPTY_RETRY` 有界重试语义处理。非空响应路径不变。
- 覆盖：`tests/unit_tests/models/test_openai_client_empty_response.py`（4 项：None
  content、空 choices、重试后成功、正常 `<think>` 剥离）。

## #12 graph_units source：从既有图存储枚举描述单元（2026-11-04）

- **动机**：loss 分区（quiz→judge→max_loss）的合法上游是 build_kg 的
  node/edge 条目流（`quiz_service` 只认 `item["node"]/item["edge"]`）；
  partition/generate 无上游时拿到空数据集根本不执行。对已建好的 KG 重做
  loss 分区只能全量重建 build_kg（含 LLM 描述改写，KG 非 byte-stable）。
- **实现**：`operators/graph_units/graph_units.py`：source 算子
  `graph_units`，从 graph storage 枚举全部带描述单元，输出
  `{"index": node_id|edge_tuple, "description": str, "_trace_id"}`，
  直接喂 judge（fork 中 quiz 产出无消费者，跳过 quiz 省一半调用）。
  图是唯一持久态（`<working_dir>/graph.graphml`），换新 working_dir 只拷
  graphml 即可复用整张图，且天然避开 BaseOperator `_meta_forward` 把旧
  QA recover 回数据流的坑。
- 空单元枚举 fail-closed（`graph_units_empty`）。纯枚举逻辑拆在
  `_collect_units`（无 ray，可单测）。
- 注册：`operators/__init__.py` `"graph_units"`。

## #13 loss 链路 fail-closed（2026-11-04）

- **judge index 归一**：跨 operator 的 Arrow/pandas 边界会把二元组
  `(src_id, tgt_id)` 变成 list；`JudgeService.process` 按
  `isinstance(index, tuple)` 分派节点/边写回，不归一会让全部边单元的
  loss **静默不落图**。`_process_single_judge` 入口把 2 元 list 归一成
  tuple。
- **分区 loss 覆盖率门**：`ECEPartitioner.partition` 新增
  `min_loss_coverage`（默认 0.95）。`unit_sampling != random` 时，真实
  loss（非 `default_loss=-log(0.1)` 假值）覆盖率低于阈值直接
  raise `partition_loss_coverage_too_low`。动机：没跑 judge / TRAINEE
  漏配（`init_llm` 返回 None）/ judge 逐条失败的单元全部落 default_loss
  假高值，恰聚在 max_loss 排序最前——旧行为下 run 正常出题但零信号且
  无任何报错。

## #14 合同 challenge 集合题输出门（2026-11-04）

- `operators/generate/gates.py` 新增 `pms_contract_challenge`：在
  `pms_contract_anchor`（合同号 token / “该合同”指代）基础上放行第三类
  锚点——完整项目编号（P-/I- + ≥6 位）。batch1003 腿B 挑战 profile 的
  集合/聚合题以项目编号为锚点，现役门会把它们整条静默丢弃（表现为
  “模型出不了集合题”的假象）。锚点真实性仍由 PMS 侧 mapper/审计把关，
  gate 只做形态存在性判定。

## #15 partition max_communities 截断（2026-11-04）

- **动机**：batch1003 试产只需几百个社区出题；全图 ECE 分区在 45,942 节点 +
  177,052 边上产出 ~2.7 万社区，generate 全跑是数千次教师调用。loss 采样下
  `ECEPartitioner` 按种子单元 loss 降序产出社区——**取前 N 即"盲区选题"**，
  截断本身就是选题机制而非降级。
- **实现**：`PartitionService` 新增 `max_communities`（默认 None = 官方全量
  行为不变）；`generator()` 产出计数达 N 即 break。配合 `unit_sampling:
  max_loss` 组合使用；random 采样下等价任意抽样。

## #16 graphml 原子写与跨 operator 传输可靠性（2026-11-04）

- **graphml 原子写**：`networkx_storage.write_nx_graph` 改唯一临时名
  （pid+uuid）+ `os.replace` 原子替换——整文件非原子写会让并发 reload 读到
  截断文件（ParseError: no element found）；固定 `.tmp` 名在 judge 多 actor
  并发写回时互踩（A replace 后 B 再 replace 报 FileNotFoundError）。写失败
  （不可序列化属性/磁盘满）清理 tmp 不残留。actor 内存图是唯一真源、每次
  全量快照，最后写入者胜出语义安全。
- **有界重试 reload**：judge 缓存 loss 写回（`_write_back_losses`）与
  partition `kg_instance.reload` 对刚被整写的 graphml 做 3 次有界重试；judge
  侧仍失败则跳过本次写回（图属性已有 loss 时写回是冗余兜底），不阻断主流程。
- **index 传输口径**：Arrow/pandas 边界把边二元组 index 变 list，
  graph_units 再把边 index 编码为 JSON 字符串（混合 str/list 列炸 Arrow
  tensor 分支）。judge 两条入口（`_process_single_judge`、缓存恢复行）统一经
  `_normalize_unit_index` 还原为 tuple 供图写回分派；对外产出（process 结果
  与 split 恢复行）统一把边 index 编码回 JSON 字符串、节点保持 str——列保持
  纯 str，Arrow 安全。`process` 对 run_concurrent 的 None 占位（输入行缺
  description 等异常逃出内层 try）与图写回同款守卫跳过，不崩整批。
- **已知约束**：缓存 loss 写回假设 judge `replicas=1`（当前生产配置）——
  `replicas>1` 时 `reload()` 会以文件内容覆盖 actor 内存图，丢其他副本未
  持久化更新，放开需 actor 侧批量写回 API。每个含缓存命中 batch 触发一次
  整图 reload+持久化，45,942 节点级图上为已知成本；如需优化可在 `__call__`
  级汇总一次写回。
