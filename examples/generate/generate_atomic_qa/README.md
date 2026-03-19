# Generate Atomic QAs

Atomic mode generates question-answer pairs that test basic, isolated knowledge from individual facts or relationships in the knowledge graph. 

`tree_atomic_config.yaml` uses the tree pipeline (`structure_analyze -> hierarchy_generate -> tree_construct -> tree_chunk`) to preserve MoDora-style document structure. It disables secondary paragraph chunking in `tree_chunk`, so pre-segmented paragraph, image, and table components stay intact before KG construction.
