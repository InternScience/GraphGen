from pathlib import Path

from graphgen.operators.tree_pipeline import (
    HierarchyGenerateService,
    StructureAnalyzeService,
    TreeChunkService,
    TreeConstructService,
)


def test_tree_pipeline_services_basic(tmp_path: Path):
    working_dir = str(tmp_path / "cache")

    structure_service = StructureAnalyzeService(
        working_dir=working_dir,
        kv_backend="json_kv",
    )
    hierarchy_service = HierarchyGenerateService(
        working_dir=working_dir,
        kv_backend="json_kv",
    )
    tree_service = TreeConstructService(
        working_dir=working_dir,
        kv_backend="json_kv",
    )
    chunk_service = TreeChunkService(
        working_dir=working_dir,
        kv_backend="json_kv",
        chunk_size=64,
        chunk_overlap=8,
    )

    input_docs = [
        {
            "_trace_id": "read-1",
            "type": "text",
            "content": "# Intro\nGraphGen is great.\n## Details\nSupports tree pipeline.",
            "metadata": {"source": "unit-test"},
        }
    ]

    structure_rows, _ = structure_service.process(input_docs)
    assert len(structure_rows) == 1
    assert structure_rows[0]["type"] == "component_pack"
    assert structure_rows[0]["components"]

    hierarchy_rows, _ = hierarchy_service.process(structure_rows)
    levels = [it["title_level"] for it in hierarchy_rows[0]["components"]]
    assert all(level >= 1 for level in levels)

    tree_rows, _ = tree_service.process(hierarchy_rows)
    assert len(tree_rows[0]["tree_nodes"]) >= 1
    assert tree_rows[0]["tree"]["node_id"] == "root"

    chunk_rows, _ = chunk_service.process(tree_rows)
    assert chunk_rows
    assert all("path" in row["metadata"] for row in chunk_rows)
    assert all(row["type"] == "text" for row in chunk_rows)
