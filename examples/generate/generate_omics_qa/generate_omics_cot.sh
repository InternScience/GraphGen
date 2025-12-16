#!/bin/bash
# Generate CoT (Chain of Thought) QA pairs from multi-omics data

python3 -m graphgen.run \
  --config_file examples/generate/generate_omics_qa/omics_cot_config.yaml \
  --output_dir cache/
