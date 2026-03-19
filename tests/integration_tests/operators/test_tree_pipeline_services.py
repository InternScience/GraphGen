from pathlib import Path
from unittest.mock import patch

from graphgen.models.generator.vqa_generator import VQAGenerator
from graphgen.models.partitioner.anchor_bfs_partitioner import AnchorBFSPartitioner
from graphgen.operators.tree_pipeline import (
    BuildGroundedTreeKGService,
    HierarchyGenerateService,
    StructureAnalyzeService,
    TreeChunkService,
    TreeConstructService,
)
from graphgen.operators.tree_pipeline.tree_utils import normalize_components
from graphgen.storage import NetworkXStorage


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


def test_structure_analyze_markdown_vqa_components(tmp_path: Path):
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "tree_vqa_demo.md"
    content = fixture_path.read_text(encoding="utf-8")

    structure_service = StructureAnalyzeService(
        working_dir=str(tmp_path / "cache"),
        kv_backend="json_kv",
    )
    hierarchy_service = HierarchyGenerateService(
        working_dir=str(tmp_path / "cache"),
        kv_backend="json_kv",
    )
    tree_service = TreeConstructService(
        working_dir=str(tmp_path / "cache"),
        kv_backend="json_kv",
    )
    chunk_service = TreeChunkService(
        working_dir=str(tmp_path / "cache"),
        kv_backend="json_kv",
        chunk_size=256,
        chunk_overlap=32,
    )

    input_docs = [{"_trace_id": "read-md", "type": "text", "content": content}]

    structure_rows, _ = structure_service.process(input_docs)
    components = structure_rows[0]["components"]
    component_types = [component["type"] for component in components]
    assert component_types == ["text", "text", "table", "text", "table", "image", "image", "text"]

    first_table = components[2]
    assert first_table["metadata"]["table_caption"] == ["Table 1. Accuracy across baselines."]
    assert "<table>" in first_table["metadata"]["table_body"]
    assert "[Table Caption]" in first_table["content"]

    first_image = components[5]
    assert first_image["metadata"]["img_path"].endswith(".jpg")
    assert first_image["metadata"]["image_caption"] == [
        "Figure 1. The microscope image highlights the reactive region after treatment."
    ]
    assert "arrows mark the highlighted tissue" in first_image["metadata"]["note_text"]

    second_image = components[6]
    assert second_image["metadata"]["image_caption"] == []
    assert second_image["content"] == ""

    hierarchy_rows, _ = hierarchy_service.process(structure_rows)
    tree_rows, _ = tree_service.process(hierarchy_rows)
    chunk_rows, _ = chunk_service.process(tree_rows)

    table_chunks = [row for row in chunk_rows if row["type"] == "table"]
    image_chunks = [row for row in chunk_rows if row["type"] == "image"]
    assert len(table_chunks) == 2
    assert len(image_chunks) == 2
    assert table_chunks[0]["metadata"]["table_caption"] == ["Table 1. Accuracy across baselines."]
    assert "table_body" in table_chunks[0]["metadata"]
    assert image_chunks[0]["metadata"]["image_caption"] == [
        "Figure 1. The microscope image highlights the reactive region after treatment."
    ]
    assert "note_text" in image_chunks[0]["metadata"]
    assert image_chunks[1]["metadata"]["image_caption"] == []


def test_normalize_components_keeps_captionless_modalities():
    components = normalize_components(
        {
            "type": "text",
            "content": (
                "## Section\n"
                "<table><tr><td>A</td></tr></table>\n\n"
                "![Img](demo.png)\n"
            ),
        }
    )

    assert [component["type"] for component in components] == ["table", "image"]
    assert components[0]["metadata"]["table_caption"] == []
    assert components[1]["metadata"]["image_caption"] == []


def test_vqa_generator_omits_empty_image_payload():
    result = VQAGenerator.format_generation_results(
        {"question": "What does the table report?", "answer": "Latency is 12ms."},
        output_data_format="ChatML",
    )

    assert result["messages"][0]["content"] == [{"text": "What does the table report?"}]


def test_anchor_bfs_accepts_multiple_anchor_types(tmp_path: Path):
    storage = NetworkXStorage(working_dir=str(tmp_path), namespace="anchor_multi")
    storage.upsert_node("img-1", {"entity_type": "IMAGE"})
    storage.upsert_node("table-1", {"entity_type": "TABLE"})
    storage.upsert_node("text-1", {"entity_type": "TEXT"})

    partitioner = AnchorBFSPartitioner(anchor_type=["image", "table"])
    anchors = partitioner._pick_anchor_ids(storage.get_all_nodes())

    assert anchors == {"img-1", "table-1"}


def test_build_grounded_tree_kg_service_enables_evidence_checks(tmp_path: Path):
    class _DummyTokenizer:
        @staticmethod
        def count_tokens(text: str) -> int:
            return len(text.split())

    class _DummyLLM:
        tokenizer = _DummyTokenizer()

    with patch(
        "graphgen.operators.tree_pipeline.build_tree_kg_service.init_llm",
        return_value=_DummyLLM(),
    ):
        service = BuildGroundedTreeKGService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            graph_backend="networkx",
        )

    assert service.require_entity_evidence is True
    assert service.require_relation_evidence is True
    assert service.validate_evidence_in_source is True
