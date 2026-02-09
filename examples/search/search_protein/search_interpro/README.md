# Search Protein Domains with InterPro

This example demonstrates how to search for protein domain information and functional annotations using the InterPro database.

## Overview

The InterPro search pipeline reads protein queries (UniProt accession numbers or protein sequences) and searches the InterPro database to find domain matches, functional annotations, GO terms, and pathways.

InterPro supports two search modes:
1. **UniProt Accession Lookup**: Fast lookup of pre-computed domain information for known UniProt entries
2. **Protein Sequence Analysis**: Submit protein sequences for InterProScan analysis to discover domains

## Quick Start

### 1. Configure Search Parameters

Edit `search_interpro_config.yaml` to set:

- **Input file path**: Set the path to your protein sequence or UniProt ID queries
- **InterPro parameters**:
  - `email`: Your email address for EBI API requests (required)
  - `api_timeout`: Request timeout in seconds (default: 30)

Example configuration:
```yaml
input_path:
  - examples/input_examples/search_interpro_demo.jsonl

data_source: interpro
interpro_params:
  email: your_email@example.com
  api_timeout: 30
```

### 2. Run the Search

```bash
./search_interpro.sh
```

Or run directly with Python:

```bash
python3 -m graphgen.run \
  --config_file examples/search/search_interpro/search_interpro_config.yaml \
  --output_dir cache/
```

## Input Format

The input file should be in JSONL format with protein queries:

```jsonl
{"type": "protein", "content": "P01308"}
{"type": "protein", "content": "Q96KN2"}
{"type": "protein", "content": "MHHHHHHSSGVDLGTENLYFQSNAMDFPQQLEACVKQANQALSRFIAPLPFQNTPVVETMQYGALLGGKRLRPFLVYATGHMFGVSTNTLDAPAAAVECIHAYSLIHDDLPAMDDDDLRRGLPTCHVKFGEANAILAGDALQTLAFSILSDANMPEVSDRDRISMISELASASGIAGMCGGQALDLDAEGKHVPLDALERIHRHKTGALIRAAVRLGALSAGDKGRRALPVLDKYAESIGLAFQVQDDILDVVGDTATLGKRQGADQQLGKSTYPALLGLEQARKKARDLIDDARQALKQLAEQSLDTSALEALADYIIQRNK"}
```

Or in FASTA format:
```
>P01308
MHHHHHHSSGVDLGTENLYFQSNAMDFPQQLEACVKQANQALSRFIAPLPFQNTPVVETMQYGALLGGKRLRPFLVYATGHMFGVSTNTLDAPAAAVECIHAYSLIHDDLPAMDDDDLRRGLPTCHVKFGEANAILAGDALQTLAFSILSDANMPEVSDRDRISMISELASASGIAGMCGGQALDLDAEGKHVPLDALERIHRHKTGALIRAAVRLGALSAGDKGRRALPVLDKYAESIGLAFQVQDDILDVVGDTATLGKRQGADQQLGKSTYPALLGLEQARKKARDLIDDARQALKQLAEQSLDTSALEALADYIIQRNK

>insulin_sequence
MHHHHHHSSGVDLGTENLYFQS...
```


## Output

The search results will be saved in the output directory with:

```json
{
  "molecule_type": "protein",
  "database": "InterPro",
  "id": "P01308",
  "job_id": "iprscan5-R20240123-123456-xxxx-p1m",
  "content": {
    "results": [
      {
        "xref": [
          {
            "ref": "INTERPRO",
            "id": "IPR000001",
            "name": "Domain Name"
          }
        ],
        "signature_acc": "PF00001",
        "go_annotations": [
          {
            "id": "GO:0001234",
            "description": "biological process"
          }
        ]
      }
    ]
  },
  "url": "https://www.ebi.ac.uk/interpro/protein/uniprot/P01308/",
  "_search_query": "P01308"
}
```

## References

- **InterPro Database**: https://www.ebi.ac.uk/interpro/
- **EBI InterProScan API**: https://www.ebi.ac.uk/Tools/services/rest/iprscan5
- **UniProt Database**: https://www.uniprot.org/
