# Search Protein Sequences

This example demonstrates how to search protein sequences from UniProt database using BLAST.

## Overview

The protein search pipeline reads protein sequence queries and searches against UniProt database to find similar sequences and retrieve associated metadata.

## Quick Start

### 1. Build Local BLAST Database (Optional)

If you want to use local BLAST for faster searches, first build the database:

```bash
./build_db.sh
```

The script will download UniProt Swiss-Prot database and create a BLAST database. You can configure the download mode:
- `sprot` (default): Download only Swiss-Prot (high quality, curated)
- `full`: Download both Swiss-Prot and TrEMBL (complete database)

The script will create a BLAST database in `${RELEASE}/` directory.

### 2. Configure Search Parameters

Edit `search_protein_config.yaml` to set:

- **Input file path**: Set the path to your protein sequence queries
- **UniProt parameters**:
  - `use_local_blast`: Set to `true` if you have a local BLAST database
  - `local_blast_db`: Path to your local BLAST database (format: `/path/to/${RELEASE}/uniprot_sprot`)

Example configuration:
```yaml
input_path:
  - examples/input_examples/search_protein_demo.jsonl

data_sources: [uniprot]
uniprot_params:
  use_local_blast: true
  local_blast_db: /your_path/2024_01/uniprot_sprot
  # options: uniprot_sprot (recommended, high quality), uniprot_trembl, or uniprot_${RELEASE} (merged database)
```

### 3. Run the Search

```bash
./search_uniprot.sh
```

Or run directly with Python:

```bash
python3 -m graphgen.run \
  --config_file examples/search/search_protein/search_protein_config.yaml \
  --output_dir cache/
```

## Input Format

The input file should be in JSONL format with protein sequence queries:

```jsonl
{"type": "protein", "content": "P01308"}
{"type": "protein", "content": "insulin"}
{"type": "protein", "content": "MHHHHHHSSGVDLGTENLYFQSNAMDFPQQLEACVKQANQALSRFIAPLPFQNTPVVETMQYGALLGGKRLRPFLVYATGHMFGVSTNTLDAPAAAVECIHAYSLIHDDLPAMDDDDLRRGLPTCHVKFGEANAILAGDALQTLAFSILSDANMPEVSDRDRISMISELASASGIAGMCGGQALDLDAEGKHVPLDALERIHRHKTGALIRAAVRLGALSAGDKGRRALPVLDKYAESIGLAFQVQDDILDVVGDTATLGKRQGADQQLGKSTYPALLGLEQARKKARDLIDDARQALKQLAEQSLDTSALEALADYIIQRNK"}
```

## Output

The search results will be saved in the output directory with matched sequences and metadata from UniProt.

## Notes

- **Local BLAST** provides faster searches and doesn't require internet connection during search
- **Swiss-Prot** is recommended for high-quality, curated protein sequences
- **TrEMBL** contains automatically annotated sequences (larger database)
- The merged database (`uniprot_${RELEASE}`) contains both Swiss-Prot and TrEMBL
- Adjust `max_concurrent` based on your system resources and API rate limits
