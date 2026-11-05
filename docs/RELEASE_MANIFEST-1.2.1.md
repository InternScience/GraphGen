# GraphGen 1.2.1 release manifest

Status: released（tag 已推送 fork）；PMS 生产 pin 切换记录见文末。

## Source identity

- Distribution: `graphg` 1.2.1
- Tag: `graphgen-fork-v1.2.1` → `ffa16b850c4fb958d4833b5fd6fb5ef0340edb05`（pms/main 顶端）
- Base: tag `graphgen-fork-v1.2.0` → `75daf5c64e374cc28f2db9867ca390830d6019bc`；直接父提交 `3553b4424ea2f6e35e74bd672ea6c6348b108e25`（1.2.0 docs 收口）；upstream base `origin/main` `3a3eb097318d34b07f6f31cb60de5b20712d65ce`（无变动）
- Publication remote: `fork`（https://github.com/Leon-Algo/GraphGen.git）；upstream `origin` 与 PR #192/#193/#194/#195 未修改
- 发布方式：隔离 git worktree（从 75daf5c 检出 → 应用冻结 patch + 5 个 untracked 文件 → bump `_version.py` 1.2.1 → rebase 到 pms/main 顶端）；发行树与冻结包 `/tmp/gg-loss-fix/freeze3/wt` 逐字节 diff 一致（仅 `_version.py` 版本行为差异）
- PMS 生产 prompt 真源在 PMS 仓库 `configs/graphgen_profiles/`；fork 内置 `pms_policy.v2/v3` 保持 1.1.2 字节（wheel 清单实测）

## 冻结审计链（本版全部发行内容的证据来源）

| 阶段 | 快照 SHA256 | patch SHA256 |
|---|---|---|
| freeze1（初版候选） | `69f22d5131c8c43c91fd3d08eb2233421088b4e31dc68cb41b4142e2e9c89619` | `c70861fdafef006a873be1d9a9bad260a51a88952aba4655cc7f3243de82d0c5` |
| freeze2（kuzu fail-closed 门控） | `2624a35484c0d1abf441f5d7305b689929b0df6235137aa7fedaa549966eb5ab` | `1640548c2dff03eb33c238d267981b0f9c6096a5bfcd0259a515c9ba5966ee22` |
| freeze3（发行版：doc-only 引用换稳定路径） | `5d1213a512041bab69b596ca83d0c7530fe7eda5ed52322db64b3542be9213c5` | `14ee8bfe713d9c06d43f92adfe61642ff444667bc2f062edd5de0424ad270eac` |

freeze1→freeze2→freeze3 每轮增量均经独立审查 agent 全树 diff 复核无夹带；代码自 freeze2 起零改动（freeze3 仅 PMS_PATCHES.md 一处证据路径换 PMS 仓库稳定引用）。

## Patch list

`docs/PMS_PATCHES.md` #17（单条新增）：judge loss 写回并发竞态修复——执行方事务写回（flock 内 reload→全目标校验→loss-only merge→flush 单 RPC）、actor 身份 SHA256(backend+realpath+namespace)、非 NetworkX 后端基类 NotImplementedError 门控、judge 写回失败整批 fail-closed（有意行为变更，替代 1.2.0 重试×3+静默跳过）。行为语义详见 `docs/RELEASE_NOTES-1.2.1.md`。

## Test and smoke evidence

环境：隔离 release worktree（Python 3.12 依赖由 `workspace/graphgen_env/.venv` site-packages 提供）。

- 单元 **29 passed**（原 25 + 后端门控 4）：release worktree 8.23s、冻结包解包副本 9.0s/9.1s、审查 agent 独立复跑 9.46s——三处独立通过
- 真实 Ray integration **8/8 PASS**（专用 63800 集群、两方独立复验：session 审查 agent + 飞轮SDK 会话；日志 `/tmp/gg-loss-fix/ray-freeze2.log`、`ray-final.log`；freeze2→freeze3 代码逐字节相同，Ray 证据直接适用）
- baseline 对照：同套件在 75daf5c **8/8 FAIL**（`ray-baseline.log`；ActorID 相同 4 项 + loss=None 4 项）——RED/GREEN 对照成立
- 确定性最小复现（场景 A/A'/B/C）：75daf5c 全部 REPRODUCED；候选串行控制场景 C 保持全部保留
- 飞轮SDK 独立复验收据：`/tmp/flywheel-fix-verify-1105/verification-receipt.json`（29 项自有探针 + 合并收集 29 项测试 + Kuzu 误用 NotImplementedError 负例 + 探针/日志证据 SHA 落档）
- `git diff --check` — passed（code/docs 各自复验）

## Build and clean-install evidence

- `uv build --out-dir /tmp/gg-release-v121/dist`（隔离 worktree）：
  - `graphg-1.2.1-py3-none-any.whl` SHA256 `c09cae6c0c1927254193c3d775a02e69e92e9d766643dbefde0c54e852837ae3`
  - `graphg-1.2.1.tar.gz` SHA256 `ea066d81b098bb44648d8b702048812f8f7c6ba59dfcc7ae854507a534b13ed0`
- wheel 清单实测（248 files）：`concurrency`/`graph_units`/`gates.py`/`base_storage`/`networkx_storage`/`init_storage`/`judge_service` 在包；`pms_policy.v2|v3` 与 `event_profiles` 三套（example/generic/pms v1）资产齐
- 干净 venv（uv venv + `--no-deps` 装 wheel，第三方依赖由 graphgen_env site-packages 提供）实测：`_version==1.2.1`、`graphgen.__file__` 指向干净 site-packages、prompt profiles pms_policy.v2/v3 解析、event profiles 三套加载、GATES ⊇ {pms_project_anchor, pms_contract_anchor, pms_contract_challenge}、`graph_units` 导入、`BaseGraphStorage.write_losses` 为 NotImplementedError 门控且 `NetworkXStorage` override 含 graph_write_lock、judge_service 导入正常——ALL CLEAN-INSTALL CHECKS PASSED

## PMS production pin

- `workspace/graphgen_env/pyproject.toml` rev → `ffa16b850c4fb958d4833b5fd6fb5ef0340edb05`；`uv.lock` 同步再生成；契约测试 `test_graphgen_production_source.py` `GRAPHGEN_REV` 同步换新 SHA
- PMS 兼容四套件在新 pin 口径重跑（结果见 pin commit）
- PMS pin commit：落本地 `main`（该线长期未推送远程，push 与否由用户决定）
- live venv（graphgen_env/.venv）收敛到 pin 安装需与 new skill data 会话（合同 pilot 所有者）确认无在跑依赖后执行

## 发布操作记录（透明化）

1. 首次 push 因 commit 基于 75daf5c 而非 pms/main 顶端 3553b44 被拒（non-fast-forward）；rebase 到 3553b44 后重推，rebase 内容零变化（diff 仅继承 1.2.0 docs 两文件）。
2. 首次被拒的 push 同时把 `graphgen-fork-v1.2.1` tag 推向了被 rebase 孤立的旧 commit `6972bd8`；发现后立即 force 替换为正确提交 `ffa16b8`。时间窗秒级、无消费者（pin 未切、无公告、产物未流通），最终远端状态已验证：branch 与 tag `^{}` 均指向 `ffa16b8`。
3. tag 一经发布不再移动；本条为事故记录，v1.2.1 的最终身份以 `ffa16b850c4fb958d4833b5fd6fb5ef0340edb05` 为准。

## Known limitations

见 `docs/RELEASE_NOTES-1.2.1.md`「已知边界」：协议外裸写者/跨 namespace KV 不在保护范围、同 unit last-loss-wins、multi-node/NFS flock 语义未实测、非 POSIX fail-closed 无降级。
