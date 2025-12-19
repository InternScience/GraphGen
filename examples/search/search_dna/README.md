# Search DNA Sequences

This example demonstrates how to search DNA sequences from NCBI RefSeq database using BLAST.

## Overview

The DNA search pipeline reads DNA sequence queries and searches against NCBI RefSeq database to find similar sequences and retrieve associated metadata.

## Quick Start

### 1. Build Local BLAST Database (Optional)

If you want to use local BLAST for faster searches, first build the database:

```bash
./build_db.sh [human_mouse_drosophila_yeast|representative|complete|all]
```

Options:
- `human_mouse_drosophila_yeast`: Download only Homo sapiens, Mus musculus, Drosophila melanogaster, and Saccharomyces cerevisiae sequences (minimal, smallest)
- `representative`: Download genomic sequences from major categories (recommended, smaller)
- `complete`: Download all complete genomic sequences from complete/ directory (very large)
- `all`: Download all genomic sequences from all categories (very large)

The script will create a BLAST database in `refseq_${RELEASE}/` directory.

### 2. Configure Search Parameters

Edit `search_dna_config.yaml` to set:

- **Input file path**: Set the path to your DNA sequence queries
- **NCBI parameters**:
  - `email`: Your email address (required by NCBI)
  - `tool`: Tool name for NCBI API
  - `use_local_blast`: Set to `true` if you have a local BLAST database
  - `local_blast_db`: Path to your local BLAST database (without .nhr extension)

Example configuration:
```yaml
input_path:
  - examples/input_examples/search_dna_demo.jsonl

data_sources: [ncbi]
ncbi_params:
  email: your_email@example.com  # Required!
  tool: GraphGen
  use_local_blast: true
  local_blast_db: refseq_release/refseq_release
```

### 3. Run the Search

```bash
./search_dna.sh
```

Or run directly with Python:

```bash
python3 -m graphgen.run \
  --config_file examples/search/search_dna/search_dna_config.yaml \
  --output_dir cache/
```

## Input Format

The input file should be in JSONL format with DNA sequence queries:

```jsonl
{"type": "text", "content": "BRCA1"}
{"type": "text", "content": ">query\nATGCGATCG..."}
{"type": "text", "content": "ATGCGATCG..."}
```

## Output

The search results will be saved in the output directory with matched sequences and metadata from NCBI RefSeq.

## Notes

- **NCBI requires an email address** - Make sure to set `email` in `ncbi_params`
- **Local BLAST** provides faster searches and doesn't require internet connection during search
- The local BLAST database can be very large (several GB to TB depending on the download type)
- Adjust `max_concurrent` based on your system resources and API rate limits
