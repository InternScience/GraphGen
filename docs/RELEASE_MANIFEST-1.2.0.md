# GraphGen 1.2.0 release manifest

Status: released（tag 已推送 fork）；PMS 生产 pin 已切换，收口记录如下。

## Source identity

- Distribution: `graphg` 1.2.0
- Tag: `graphgen-fork-v1.2.0` → `75daf5c64e374cc28f2db9867ca390830d6019bc`（pms/main 发布线）
- Base: `d45b17be82b52da21daa9a8ffe813a303604c698`（1.1.2 线）；upstream base `origin/main` `3a3eb097318d34b07f6f31cb60de5b20712d65ce`（无变动）
- Publication remote: `fork`（https://github.com/Leon-Algo/GraphGen.git）；upstream `origin` 与 PR #192/#193/#194/#195 未修改
- 发布方式：隔离 git worktree（live 工作树因合同 pilot 在跑保持只读）；提交内容与 live 工作树逐字节 `cmp` 一致（除 4 个 pms_policy.v2/v3 模板按政策排除与审查修复文件）
- PMS 生产 prompt 真源在 PMS 仓库 `configs/graphgen_profiles/`；fork 内置 `pms_policy.v2/v3` 保持 1.1.2 字节（wheel 清单实测）

## Patch list

`docs/PMS_PATCHES.md` #11–#16（#1–#10 见 1.1.2 及更早 release）：畸形响应按空响应重试、graph_units source、loss 链路 fail-closed（judge index 归一 + 缓存写回 + min_loss_coverage 门）、pms_contract_anchor/challenge 双合同门、partition max_communities 截断、graphml 原子写 + 有界重试 reload + index JSON 口径 + None 占位守卫。

## Test and smoke evidence

环境：隔离 release worktree（Python 3.12 依赖由 `workspace/graphgen_env/.venv` site-packages 提供；`PYTHONPATH` 将 worktree 排在最前，实测 `graphgen.__file__` 指向 worktree）。

- `pytest -c /dev/null -q tests/unit_tests tests/test_event_profile_loader_regressions.py tests/test_support_edge_citations.py tests/test_event_profiles.py tests/test_package_profiles.py tests/integration_tests/models/partitioner/` — **36 passed**（含新增 `tests/unit_tests/operators/test_judge_transport_regressions.py` 4 项：None 占位守卫、缓存恢复行 index 归一、graphml 失败清理 tmp、原子替换 happy path；TDD 红→绿验证）
- `pytest -c /dev/null -q <PMS>/tests/test_graphgen_production_source.py <PMS>/tests/test_pms_graphgen_policy_generation.py <PMS>/tests/test_run_graphgen_defaults.py <PMS>/tests/test_repair_project_query_graphgen_chunk_cache.py` — **42 passed, 1 skipped**（PMSAgent 工作树，修复前基线同口径）
- 独立功能审查：并行 code-review agent，high 级，范围 `d45b17b..a93f774`，8 项发现 → 4 项修复 + 4 项记录（详见 release notes）；修复后全套测试重跑通过
- `git diff --check` — passed（发布 commit 与 docs commit 各自复验）

## Build and clean-install evidence

- `uv build --out-dir /tmp/gg-release-v120/dist`（隔离 worktree）：
  - `graphg-1.2.0-py3-none-any.whl` SHA256 `493375dcf32d97434839044b2a3499e4c95737ea7232cc4f39e99e861bb6658b`
  - `graphg-1.2.0.tar.gz` SHA256 `79387a3acfcc215447e6e4d65bb00dea77cac1142d904befb0d8a2630335fc3f`
- wheel 清单实测（246 files）：`graphgen/operators/graph_units/*` 在包；`generate/gates.py` 在包；`templates/profiles/pms_policy.v2|v3` 为 1.1.2 字节；`event_profiles` 三套齐
- 干净 venv（graphgen_env 同款 Python 3.12）`pip install --no-deps` wheel 后实测：`_version.__version__ == '1.2.0'`、`graphgen.__file__` 指向干净 venv site-packages、`GATES ⊇ {pms_project_anchor, pms_contract_anchor, pms_contract_challenge}`、`graph_units` 注册、5 套内置 profile（2 prompt + 3 event）全部加载成功

## PMS production pin

- `workspace/graphgen_env/pyproject.toml` `[tool.uv.sources]` rev → `75daf5c64e374cc28f2db9867ca390830d6019bc`；`uv.lock` 同步再生成（两处 `?rev=75daf5c…#75daf5c…`，旧 rev 引用清零）
- 契约测试 `test_graphgen_production_source.py` `GRAPHGEN_REV` 同步换新 SHA
- PMS 兼容四套件在新 pin 口径 **42 passed, 1 skipped**
- **干净环境重建（审计 B6/J6/G3 解封）**：`UV_PROJECT_ENVIRONMENT=<clean> uv sync --locked` 全量重建成功；`direct_url.json` 实测 `{"url": "https://github.com/Leon-Algo/GraphGen.git", "vcs_info": {"commit_id": "75daf5c…"}}`——git rev 安装、非 editable；干净 env 内实测 `_version==1.2.0`、三门注册、`graph_units` 在册；launcher import 链（`pms_graphgen_policy_generation` + `graphgen_worker_pool`）在干净 env 冒烟通过，PMS 侧 profile 目录解析正常
- PMS pin commit：`987d529f`（fix(pms): 生产 GraphGen pin 升级 1.2.0，仅含 pyproject/uv.lock/测试契约三文件；落在本地 `main`——该线合并自 PMSAgent 且长期未推送远程，push 与否由用户决定）
- live venv（graphgen_env/.venv）当前仍为 editable 指向工作树（合同 pilot 在跑不可动）；pilot 结束后执行 `uv sync --locked` 收敛到 git rev 安装（已预查启动链不依赖 venv 内 pymongo/agentscope 历史残留，收敛安全）

## Known limitations

1. 缓存 loss 写回假设 judge `replicas=1`；多副本需 actor 侧批量写回 API（PMS_PATCHES #16）。
2. 含缓存命中 batch 逐批整图 reload+持久化，大图为已知 I/O 成本。
3. `pms_contract_challenge` 与 `pms_contract_anchor` 谓词目前两处表达，anchor 契约演进需同步两处。
4. fork 内置 `pms_policy.v2/v3` 与 PMS 侧正本 profile 内容并行，正本在 `configs/graphgen_profiles/`；fork 内置副本仅作包资产完整性演示，不作为生产真源。
