# Generate VQAs

## DRAM-oriented high-quality VQA pipeline

This workflow is suitable for generating VQA training data from memory-system materials (e.g., DRAM timing diagrams, architecture figures, specs).

### 1) Prepare input
- Put your multimodal samples in JSON format (text + image path).
- Ensure each sample has enough textual context and image metadata so the graph builder can connect entities and relations.

### 2) Run generation
```bash
bash examples/generate/generate_vqa/generate_vqa.sh
```

### 2.1) Tree-pipeline VQA
If your source is structured markdown / MoDora-style content, use `tree_vqa_config.yaml`.
This variant runs `structure_analyze -> hierarchy_generate -> tree_construct -> tree_chunk -> build_grounded_tree_kg`
before partitioning, so image/table VQA samples are grounded by tree-local evidence spans.

### 3) Quality controls already enabled
- Prompt-level constraints for DRAM/VQA reasoning (structure, timing, performance, comparison, grounding).
- Post-generation filtering in `VQAGenerator`:
  - drop empty QA pairs
  - drop uncertain answers (e.g., unknown)
  - deduplicate near-identical QA pairs
  - enforce context keyword grounding
- Evidence-aware context injection:
  - entities and relations can carry `evidence_span`
  - VQA prompts now include those evidence snippets explicitly
  - `build_grounded_tree_kg` can reject unsupported entity/relation evidence

### 4) Recommended config tuning
In `vqa_config.yaml` under `generate.params`, tune the general generation settings such as `data_format`.
