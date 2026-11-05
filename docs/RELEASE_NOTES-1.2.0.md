# GraphGen 1.2.0 release notes

- Distribution: `graphg` 1.2.0
- Tag: `graphgen-fork-v1.2.0` → 代码提交 `75daf5c64e374cc28f2db9867ca390830d6019bc`
- Base: fork `pms/main` `d45b17be82b52da21daa9a8ffe813a303604c698`（1.1.2 发布线；upstream base 仍为 `origin/main` `3a3eb09`）
- 发布线：`pms/main`；upstream `origin` 不修改，上游拆分 PR #193/#194/#195 不受影响

## 行为变更（PMS_PATCHES.md #11–#16）

| # | 变更 | 类型 |
|---|---|---|
| #11 | OpenAI client：思考模型 `content=None` / 空 `choices` 畸形响应按空响应进入重试，不再 IndexError/TypeError 触发不可重试 fail-closed | bug 修复 |
| #12 | 新增 `graph_units` source 算子：从既有图存储枚举带描述单元（node str index / edge list index + trace id），支持对已建 KG 直接做 loss 分区 | 新能力 |
| #13 | judge：index 传输归一 + 缓存命中记录 loss 写回图存储；ECE 分区 loss 采样新增 `min_loss_coverage` fail-closed 门（默认 0.95） | 新能力 + 修复 |
| #14 | 新增 `pms_contract_anchor` / `pms_contract_challenge` 输出门（合同号/指代/项目锚点三出口） | 新能力 |
| #15 | `PartitionService` 新增 `max_communities` 截断（默认 None = 官方全量行为不变） | 新能力 |
| #16 | graphml 唯一临时名原子替换写（失败清理 tmp）；judge/partition 有界重试 reload；judge None 占位守卫；缓存恢复行与 process 产出统一 index=JSON 字符串口径；`_normalize_unit_index` 单一实现 | bug 修复 + 加固 |

## 独立功能审查与修复

发布前由并行 code-review agent 对 `d45b17b..75daf5c~`（a93f774）做 high 级独立审查，返回 8 项发现；其中 4 项在本版修复并附 4 个回归测试（`tests/unit_tests/operators/test_judge_transport_regressions.py`）：

1. `process()` 对 run_concurrent None 占位缺守卫 → AttributeError 崩整批（已修：同款 `if not isinstance(result, dict)` 守卫）；
2. `split()` 缓存恢复行绕过 index 归一 → 历史 list index 与新产出混型炸 Arrow（已修：恢复行统一 JSON 字符串口径）；
3. index 解码逻辑内联重复两份（已修：统一走 `_normalize_unit_index`）；
4. 代码注释引用的 PMS_PATCHES.md #16 条目缺失（已补文档）。

记录为已知约束（不构成本版缺陷，见 PMS_PATCHES.md #16）：缓存 loss 写回假设 judge `replicas=1`（当前生产配置）；每个含缓存命中 batch 一次整图 reload+持久化为已知成本；contract challenge 门与 anchor 门谓词目前两处表达。

## 兼容性

- 默认 LightRAG + ECE + random 采样行为不变；`max_communities`/`min_loss_coverage` 默认值保持官方语义。
- 合同门为代码注册表成员，需生成配置显式引用 `output_gate` 才生效。
- **profile 政策**：fork 内置 `pms_policy.v2/v3` 模板保持 1.1.2 字节不变（无生产消费方）；生产 prompt 真源在 PMS 仓库 `configs/graphgen_profiles/`（外部路径 profile，经 fork loader 支持）。4 个模板的工作树本地改动不进本版（profile 不可变政策，属 PMS 侧资产）。
- 本版修复 10-29/10-30 通用性审计 B6/J6 P0：新环境从锁定 git rev 安装即可获得合同出题门（此前锁定 rev 实测 `output_gate_not_registered:pms_contract_anchor`）。

## 测试与验收

见 `docs/RELEASE_MANIFEST-1.2.0.md`。
