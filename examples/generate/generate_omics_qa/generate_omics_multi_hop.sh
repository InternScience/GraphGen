#!/bin/bash
# Generate multi-hop QA pairs from multi-omics data

python3 -m graphgen.run \
  --config_file examples/generate/generate_omics_qa/omics_multi_hop_config.yaml \
  --output_dir cache/
