# Multi-omics Knowledge Graph QA Generation

This example demonstrates how to build knowledge graphs from multi-omics data (DNA, RNA, protein) and generate question-answer pairs using different QA generation methods.

## Pipeline Overview

The pipeline includes the following steps:

1. **read**: Read input files (JSONL format with sequence queries)
2. **search**: Search biological databases (NCBI for DNA, RNAcentral for RNA, UniProt for protein)
3. **chunk**: Chunk sequences and metadata
4. **build_kg**: Extract entities and relationships to build knowledge graph
5. **quiz** (optional): Generate quiz questions for KG nodes and edges
6. **judge** (optional): Judge the correctness of KG descriptions
7. **partition**: Partition the knowledge graph into communities
8. **generate**: Generate QA pairs from partitioned communities

## Available QA Generation Methods

This example provides configurations for different QA generation methods:

### 1. Atomic QA (`omics_atomic_config.yaml`)
- **Method**: `atomic`
- **Format**: Alpaca
- **Partition**: DFS with max_units_per_community=1
- **Use case**: Simple, single-fact questions
- **Run**: `./generate_omics_atomic.sh`

### 2. Aggregated QA (`omics_aggregated_config.yaml`)
- **Method**: `aggregated`
- **Format**: ChatML
- **Partition**: ECE with comprehension loss
- **Includes**: quiz and judge steps
- **Use case**: Comprehensive questions covering multiple facts
- **Run**: `./generate_omics_aggregated.sh`

### 3. Chain of Thought (CoT) QA (`omics_cot_config.yaml`)
- **Method**: `cot`
- **Format**: ShareGPT
- **Partition**: Leiden algorithm
- **Use case**: Questions requiring step-by-step reasoning
- **Run**: `./generate_omics_cot.sh`

### 4. Multi-hop QA (`omics_multi_hop_config.yaml`)
- **Method**: `multi_hop`
- **Format**: ChatML
- **Partition**: ECE with random sampling
- **Use case**: Questions requiring reasoning across multiple KG relationships
- **Run**: `./generate_omics_multi_hop.sh`

## Quick Start

### 1. Configure Input Data

Edit the config file to set:
- **Input file**: Change `input_path` in the `read_files` node
  - DNA: `examples/input_examples/search_dna_demo.jsonl`
  - RNA: `examples/input_examples/search_rna_demo.jsonl`
  - Protein: `examples/input_examples/search_protein_demo.jsonl`

### 2. Configure Data Source

Set the appropriate data source and parameters:

**For DNA (NCBI):**
```yaml
data_sources: [ncbi]
ncbi_params:
  email: your_email@example.com  # Required!
  tool: GraphGen
  use_local_blast: false
  max_concurrent: 5
```

**For RNA (RNAcentral):**
```yaml
data_sources: [rnacentral]
rnacentral_params:
  use_local_blast: false
  max_concurrent: 5
```

**For Protein (UniProt):**
```yaml
data_sources: [uniprot]
uniprot_params:
  use_local_blast: false
  max_concurrent: 5
```

### 3. Run the Pipeline

Use individual scripts for each QA method:

```bash
# Atomic QA
./generate_omics_atomic.sh

# Aggregated QA (includes quiz & judge)
./generate_omics_aggregated.sh

# Chain of Thought QA
./generate_omics_cot.sh

# Multi-hop QA
./generate_omics_multi_hop.sh
```

#### Direct Python Command

Or run directly with Python:

```bash
python3 -m graphgen.run \
  --config_file examples/generate/generate_omics_qa/omics_atomic_config.yaml \
  --output_dir cache/
```

## Input Format

Input files should be JSONL format with one query per line:

```jsonl
{"type": "text", "content": "BRCA1"}
{"type": "text", "content": ">query\nATGCGATCG..."}
{"type": "text", "content": "ATGCGATCG..."}
```

## Configuration Options

### Chunking Parameters
- `chunk_size`: Size for text metadata chunks (default: 1024)
- `chunk_overlap`: Overlap for text chunks (default: 100)
- `sequence_chunk_size`: Size for sequence chunks (default: 1000)
- `sequence_chunk_overlap`: Overlap for sequence chunks (default: 100)

### Partition Methods
- `dfs`: Depth-first search partitioning
- `bfs`: Breadth-first search partitioning
- `ece`: Error Comprehension Estimation (requires quiz & judge)
- `leiden`: Leiden community detection algorithm

### QA Generation Methods
- `atomic`: Single-fact questions
- `aggregated`: Multi-fact comprehensive questions
- `cot`: Chain of thought reasoning questions
- `multi_hop`: Multi-hop reasoning questions
- `vqa`: Visual question answering (not applicable for sequences)

### Output Formats
- `Alpaca`: Alpaca instruction format
- `ChatML`: ChatML conversation format
- `Sharegpt`: ShareGPT format

## Output

The pipeline generates:
- Knowledge graph with biological entities (genes, RNAs, proteins, organisms, etc.) and relationships
- QA pairs in the specified format (ChatML, Alpaca, or ShareGPT)
- Output location: `cache/` directory (configurable via `working_dir`)

## Notes

- **NCBI requires an email address** - Make sure to set `email` in `ncbi_params`
- **Quiz & Judge steps** are only included in the aggregated config (required for ECE partition with loss-based sampling)
- **Local BLAST** can be enabled if you have local databases set up (see `examples/search/build_db/`)
- Adjust `max_concurrent` based on your system resources and API rate limits
