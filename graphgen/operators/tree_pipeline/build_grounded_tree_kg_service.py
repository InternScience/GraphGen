from .build_tree_kg_service import BuildTreeKGService


class BuildGroundedTreeKGService(BuildTreeKGService):
    """Tree KG builder with evidence grounding enabled by default."""

    def __init__(
        self,
        working_dir: str = "cache",
        kv_backend: str = "rocksdb",
        graph_backend: str = "kuzu",
        **build_kwargs,
    ):
        grounded_kwargs = {
            "require_entity_evidence": True,
            "require_relation_evidence": True,
            "validate_evidence_in_source": True,
        }
        grounded_kwargs.update(build_kwargs)
        super().__init__(
            working_dir=working_dir,
            kv_backend=kv_backend,
            graph_backend=graph_backend,
            **grounded_kwargs,
        )
