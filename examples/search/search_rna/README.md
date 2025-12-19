# Search RNA Sequences

This example demonstrates how to search RNA sequences from RNAcentral database using BLAST.

## Overview

The RNA search pipeline reads RNA sequence queries and searches against RNAcentral database to find similar sequences and retrieve associated metadata.

## Quick Start

### 1. Build Local BLAST Database (Optional)

If you want to use local BLAST for faster searches, first build the database:

```bash
./build_db.sh [all|list|selected|database_name...]
```

Options:
- `all`: Download complete active database (~8.4G compressed)
- `list`: List all available database subsets
- `selected`: Download predefined database subsets (ensembl_gencode, mirbase, gtrnadb, refseq, lncbase, rfam)
- `database_name`: Download specific database subset (e.g., refseq, rfam, mirbase)

The script will create a BLAST database in `rnacentral_${RELEASE}/` or `rnacentral_${DB_NAME}_${RELEASE}/` directory.

### 2. Configure Search Parameters

Edit `search_rna_config.yaml` to set:

- **Input file path**: Set the path to your RNA sequence queries
- **RNAcentral parameters**:
  - `use_local_blast`: Set to `true` if you have a local BLAST database
  - `local_blast_db`: Path to your local BLAST database (without .nhr extension)

Example configuration:
```yaml
input_path:
  - examples/input_examples/search_rna_demo.jsonl

data_sources: [rnacentral]
rnacentral_params:
  use_local_blast: true
  local_blast_db: rnacentral_ensembl_gencode_YYYYMMDD/ensembl_gencode_YYYYMMDD
```

### 3. Run the Search

```bash
./search_rna.sh
```

Or run directly with Python:

```bash
python3 -m graphgen.run \
  --config_file examples/search/search_rna/search_rna_config.yaml \
  --output_dir cache/
```

## Input Format

The input file should be in JSONL format with RNA sequence queries:

```jsonl
{"type": "text", "content": "miR-21"}
{"type": "text", "content": ">query\nAUGCAUGC..."}
{"type": "text", "content": "AUGCAUGC..."}
```

## Output

The search results will be saved in the output directory with matched sequences and metadata from RNAcentral.

## Notes

- **Local BLAST** provides faster searches and doesn't require internet connection during search
- The complete RNAcentral database is large (~8.4G compressed), consider using specific database subsets for smaller downloads
- RNAcentral uses URS IDs (e.g., URS000149A9AF) which match the online RNAcentral API database
- Adjust `max_concurrent` based on your system resources and API rate limits
