# Multi-omics Knowledge Graph QA Generation

This example demonstrates how to build knowledge graphs from multi-omics data (DNA, RNA, protein) and generate question-answer pairs using the unified `omics_qa` method.

## Pipeline Overview

The pipeline includes the following steps:

1. **read**: Read input files (JSON/JSONL format with sequence queries or protein data)
2. **search**: Search biological databases (NCBI for DNA, RNAcentral for RNA, UniProt for protein) - *optional if input already contains search results*
3. **chunk**: Chunk sequences and metadata
4. **build_kg**: Extract entities and relationships to build knowledge graph
5. **partition**: Partition the knowledge graph into communities using anchor-based BFS
6. **generate**: Generate QA pairs from partitioned communities with automatic molecule caption extraction

## Key Features

- **Unified QA Generation**: Single `omics_qa` method supports DNA, RNA, and Protein
- **Automatic Caption Extraction**: Automatically extracts and attaches molecule-specific information (dna/rna/protein captions) to each QA pair
- **Flexible Configuration**: Easy to switch between DNA, RNA, and Protein by changing input file and data source
- **Anchor-based Partitioning**: Uses molecule type as anchor for BFS partitioning (dna/rna/protein)

## Quick Start

### 1. Configure Input Data

Edit `omics_qa_config.yaml` to set the input file path:

**For DNA:**
```yaml
input_path:
  - examples/input_examples/search_dna_demo.jsonl
```

**For RNA:**
```yaml
input_path:
  - examples/input_examples/search_rna_demo.jsonl
```

**For Protein:**
```yaml
input_path:
  - examples/input_examples/search_protein_demo.jsonl
```

### 2. Configure Data Source

Set the appropriate data source and parameters in the `search_data` node:

**For DNA (NCBI):**
```yaml
data_sources: [ncbi]
ncbi_params:
  email: your_email@example.com  # Required!
  tool: GraphGen
  use_local_blast: true
  local_blast_db: refseq_release/refseq_release
  blast_num_threads: 2
  max_concurrent: 5
```

**For RNA (RNAcentral):**
```yaml
data_sources: [rnacentral]
rnacentral_params:
  use_local_blast: true
  local_blast_db: rnacentral_ensembl_gencode_YYYYMMDD/ensembl_gencode_YYYYMMDD
  blast_num_threads: 2
  max_concurrent: 5
```

**For Protein (UniProt):**
```yaml
data_sources: [uniprot]
uniprot_params:
  use_local_blast: true
  local_blast_db: ${RELEASE}/uniprot_sprot
  blast_num_threads: 2
  max_concurrent: 5
```

### 3. Configure Anchor Type

Set the `anchor_type` in the `partition` node to match your molecule type:

```yaml
partition:
  params:
    method: anchor_bfs
    method_params:
      anchor_type: protein  # Change to "dna" or "rna" as needed
      max_units_per_community: 10
```

### 4. Run the Pipeline

```bash
./generate_omics_qa.sh
```

Or run directly with Python:

```bash
python3 -m graphgen.run \
  --config_file examples/generate/generate_omics_qa/omics_qa_config.yaml \
  --output_dir cache/
```

## Input Format

### For DNA/RNA (JSONL format):
```jsonl
{"type": "text", "content": "BRCA1"}
{"type": "text", "content": ">query\nATGCGATCG..."}
{"type": "text", "content": "ATGCGATCG..."}
```

### For Protein (JSONL format):
```jsonl
{"type": "text", "content": "P01308"}
{"type": "text", "content": "insulin"}
{"type": "text", "content": "MHHHHHHSSGVDLGTENLYFQSNAMDFPQQLEACVKQANQALSRFIAPLPFQNTPVVETMQYGALLGGKRLRPFLVYATGHMFGVSTNTLDAPAAAVECIHAYSLIHDDLPAMDDDDLRRGLPTCHVKFGEANAILAGDALQTLAFSILSDANMPEVSDRDRISMISELASASGIAGMCGGQALDLDAEGKHVPLDALERIHRHKTGALIRAAVRLGALSAGDKGRRALPVLDKYAESIGLAFQVQDDILDVVGDTATLGKRQGADQQLGKSTYPALLGLEQARKKARDLIDDARQALKQLAEQSLDTSALEALADYIIQRNK"}
```

## Output Format

The `omics_qa` method automatically extracts and attaches molecule-specific captions to QA pairs:

### Alpaca Format:
```json
{
  "instruction": "What is the function of this protein?",
  "input": "",
  "output": "The protein functions as...",
  "dna": {...},      # DNA caption (if molecule_type is DNA)
  "rna": {...},      # RNA caption (if molecule_type is RNA)
  "protein": {...}   # Protein caption (if molecule_type is protein)
}
```

### ChatML Format:
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "text": "What is the function of this protein?",
          "dna": {...},
          "rna": {...},
          "protein": {...}
        }
      ]
    },
    {
      "role": "assistant",
      "content": "The protein functions as..."
    }
  ]
}
```

## Caption Information

The generator automatically extracts relevant caption information based on molecule type:

- **DNA**: gene_name, gene_description, organism, chromosome, genomic_location, function, gene_type, etc.
- **RNA**: rna_type, description, organism, related_genes, gene_name, so_term, modifications, etc.
- **Protein**: protein_name, gene_names, organism, function, sequence, entry_name, etc.

## Configuration Options

### Chunking Parameters
- `chunk_size`: Size for text metadata chunks (default: 1024)
- `chunk_overlap`: Overlap for text chunks (default: 100)
- `sequence_chunk_size`: Size for sequence chunks (default: 1000)
- `sequence_chunk_overlap`: Overlap for sequence chunks (default: 100)

### Partition Parameters
- `method`: `anchor_bfs` (recommended for omics data)
- `anchor_type`: `dna`, `rna`, or `protein` (must match your data type)
- `max_units_per_community`: Maximum nodes and edges per community (default: 10)

### Generation Parameters
- `method`: `omics_qa` (unified method for DNA/RNA/Protein)
- `data_format`: `Alpaca`, `ChatML`, or `Sharegpt`

## Notes

- **NCBI requires an email address** - Make sure to set `email` in `ncbi_params`
- **Anchor type must match molecule type** - Set `anchor_type` to match your data (dna/rna/protein)
- **Local BLAST** can be enabled if you have local databases set up (see `examples/search/build_db/`)
- **Caption extraction** is automatic - The generator detects molecule type and extracts relevant caption information
- Adjust `max_concurrent` based on your system resources and API rate limits

## Examples

### Generate QA for Protein Data
1. Set `input_path` to `examples/input_examples/search_protein_demo.jsonl`
2. Set `data_sources: [uniprot]`
3. Set `anchor_type: protein`
4. Run `./generate_omics_qa.sh`

### Generate QA for DNA Data
1. Set `input_path` to `examples/input_examples/search_dna_demo.jsonl`
2. Set `data_sources: [ncbi]`
3. Set `anchor_type: dna`
4. Run `./generate_omics_qa.sh`

### Generate QA for RNA Data
1. Set `input_path` to `examples/input_examples/search_rna_demo.jsonl`
2. Set `data_sources: [rnacentral]`
3. Set `anchor_type: rna`
4. Run `./generate_omics_qa.sh`
