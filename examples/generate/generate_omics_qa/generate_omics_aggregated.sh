#!/bin/bash
# Generate aggregated QA pairs from multi-omics data

python3 -m graphgen.run \
  --config_file examples/generate/generate_omics_qa/omics_aggregated_config.yaml \
  --output_dir cache/
