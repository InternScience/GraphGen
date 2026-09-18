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

## 验收

- 默认路径（无 profile/gate/并发参数）与官方行为逐字节一致；
- PMS 主仓库 `tests/test_pms_graphgen_policy_generation.py`、
  `tests/test_run_graphgen_defaults.py` 为契约回归（rft 环境按路径加载
  纯 stdlib 模块校验）；
- 真实生成由 GraphGen venv smoke 验收。
