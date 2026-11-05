# GraphGen 1.2.1 release notes

- Distribution: `graphg` 1.2.1
- Base: tag `graphgen-fork-v1.2.0` → `75daf5c64e374cc28f2db9867ca390830d6019bc`（代码补丁基线）
- 发布线：`pms/main`；upstream `origin` 不修改，上游拆分 PR #193/#194/#195 不受影响
- 性质：PATCH——多副本 judge loss 写回并发竞态修复（PMS_PATCHES #17），行为变更均已落账

## 背景

1.2.0 发布后独立审查与真实 Ray 复现确认：judge 缓存/新判 loss 写回在 `replicas>1`（生产 run10 实际 `replicas: 8`）时存在丢失更新竞态——写回是 reload→update→flush 的多操作周期，共享 actor 内存被并发 reload 打断或独立内存 stale 整图快照覆盖时，已写 loss 静默丢失。确定性最小复现（场景 A/A'/B/C）在 75daf5c 上全部可复现；run10 全量审计（400,046/400,047 KV loss、100% 图边覆盖）证明该 run 未受损，但覆盖不构成逐值正确性证明，机制修复仍必要。

## 行为变更（PMS_PATCHES.md #17）

| 变更 | 说明 | 类型 |
|---|---|---|
| 执行方事务写回 `write_losses` | reload→全目标校验（缺失 KeyError fail-closed）→loss-only merge→flush 在持 POSIX flock 的执行进程内一次 RPC 完成；调用方不再拆多 RPC、不再持有锁；调用方死亡不提前释放锁 | 修复（竞态） |
| 具名 actor 身份改为 SHA256(backend+realpath(working_dir)+namespace) | 消除跨目录 actor 误复用与错误目录文件锁 | 修复（竞态） |
| 非全功能后端 fail-closed 门控 | `BaseGraphStorage.write_losses` 改为 `NotImplementedError`（消息含 backend 类名与原因）；事务实现下沉 `NetworkXStorage` override。默认 `graph_backend` 为 kuzu 的配置若启用 loss 写回将得到清晰报错而非静默部分提交（生产实证 `graph_backend: networkx`） | 行为变更（有意，fail-closed） |
| judge 写回失败整批传播 | 缓存写回重试×3+静默跳过（1.2.0）移除；写回失败向上抛出、由下游 `min_loss_coverage` 门响亮失败。替代 1.2.0 的兜底行为（项目不变式禁止静默兜底/零填充） | 行为变更（有意，fail-closed） |

## 兼容性

- 默认 LightRAG + ECE + random 采样、`max_communities`/`min_loss_coverage` 默认值不变；partition/graph_units/quiz 代码路径与 1.2.0 逐字节一致（裸 reload，靠同 actor 串行）。
- 仅 `graph_backend: networkx` 的 loss 写回受事务保护；kuzu 等其他后端被门控显式拒绝（不能声称原子写回）。
- KV 后端（json_kv/rocksdb）actor 名变更属版本切换；未发现双 actor 写同 KV 路径风险（actor 身份含目录 realpath）。

## 已知边界（随发行保留）

1. 仅保护参与 `write_losses` 协议的 loss 写者；独立 build_kg/裸快照写者、跨 Ray namespace/cluster 的 KV 并发不在保护范围。
2. 同 unit 多配置并发写回为 last-loss-wins。
3. multi-node Ray 需共享文件系统的 flock 语义；NFS 语义未实测。
4. 非 POSIX 平台（无 fcntl）import 即 fail-closed，不提供降级路径；Windows/macOS 未实测。

## 测试与验收证据

- 单元 **29 passed**（原 25 + 后端门控 4）：候选树与冻结包解包副本独立复跑均过（`unit-freeze2.log`、`unit-freeze3.log`）。
- 真实 Ray integration **8/8 PASS**（专用 63800 集群，两方独立复验）；同套件在 75daf5c baseline **8/8 FAIL**（ActorID 相同 4 项 + loss=None 4 项）——RED/GREEN 对照成立。
- 独立复验：①session 内独立审查 agent（freeze1/freeze2/freeze3 全树 diff 无夹带、NetworkX 事务语义逐字核对、KuzuStorage 真实实例门控负例、目录隔离/别名复用/双 submitter 探针）；②飞轮SDK 会话独立复验（自有私有 Ray 探针 29 项 + 定向重跑 + Kuzu 负例，收据与证据 SHA 落档 `/tmp/flywheel-fix-verify-1105/`）。
- 基线对照：确定性最小复现（`/tmp/gg-loss-race-test/test_loss_race.py`）在候选上场景 C（串行控制）保持全部保留。

发布门禁与构建证据见 `docs/RELEASE_MANIFEST-1.2.1.md`。
