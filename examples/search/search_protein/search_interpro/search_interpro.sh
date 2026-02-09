#!/bin/bash
# Search InterPro for protein domain annotations

python3 -m graphgen.run \
  --config_file examples/search/search_protein/search_interpro/search_interpro_config.yaml
