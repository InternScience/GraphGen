# Docs 导航

当前 `docs/` 按主题拆成了两个主目录，避免所有文档平铺在一层。

## 1. `docs/vlm_vqa/`

这组文档聚焦 GraphGen 面向专业文档的 VLM / VQA 数据生成路线。

推荐阅读顺序：

1. `docs/vlm_vqa/research.md`
2. `docs/vlm_vqa/roadmap.md`
3. `docs/vlm_vqa/plans/kg_grounding.md`
4. `docs/vlm_vqa/plans/multimodal_alignment.md`
5. `docs/vlm_vqa/plans/question_depth.md`
6. `docs/vlm_vqa/plans/eval_benchmark.md`
7. `docs/vlm_vqa/execution/p0_checklist.md`

目录含义：

- `research.md`
  - 当前 GraphGen VQA / Atomic QA 原理研究
- `roadmap.md`
  - 顶层路线图
- `plans/`
  - 各专项规划文档
- `execution/`
  - 更贴近工程执行的阶段清单

## 2. `docs/tree_pipeline/`

这组文档聚焦 tree pipeline 的局部专题说明。

- `docs/tree_pipeline/structure_analyze_vqa_changes.md`
  - markdown 结构分析、image/table 组件拆分与 tree VQA 相关变更说明

## 3. 后续建议

如果后面继续补文档，建议沿用这个结构：

- VLM / VQA 总体路线与计划，继续放 `docs/vlm_vqa/`
- tree pipeline、chunk、parser、fixture 之类的专题说明，放 `docs/tree_pipeline/`
- 如果后续有 benchmark 结果、实验记录，可以再单独增加 `docs/experiments/`
