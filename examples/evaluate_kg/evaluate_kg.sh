python3 -m graphgen.operators.evaluate_kg.evaluate_kg \
    --working_dir cache \
    --graph_backend kuzu \
    --kv_backend rocksdb \
    --max_concurrent 10
