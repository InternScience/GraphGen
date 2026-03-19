from pathlib import Path

from graphgen.operators.tree_pipeline import (
    HierarchyGenerateService,
    StructureAnalyzeService,
    TreeChunkService,
    TreeConstructService,
)


def main() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "tree_vqa_demo.md"
    )
    content = fixture_path.read_text(encoding="utf-8")

    structure_service = StructureAnalyzeService(working_dir="cache", kv_backend="json_kv")
    hierarchy_service = HierarchyGenerateService(working_dir="cache", kv_backend="json_kv")
    tree_service = TreeConstructService(working_dir="cache", kv_backend="json_kv")
    chunk_service = TreeChunkService(
        working_dir="cache",
        kv_backend="json_kv",
        chunk_size=128,
        chunk_overlap=16,
        split_text_nodes=False,
    )

    input_docs = [{"_trace_id": "fixture-1", "type": "text", "content": content}]
    structure_rows, _ = structure_service.process(input_docs)
    hierarchy_rows, _ = hierarchy_service.process(structure_rows)
    tree_rows, _ = tree_service.process(hierarchy_rows)
    chunk_rows, _ = chunk_service.process(tree_rows)

    print("components:")
    for component in structure_rows[0]["components"]:
        print(component["type"], component.get("title"), component.get("metadata", {}))

    print("\nchunks:")
    for row in chunk_rows:
        print(row["type"], row["metadata"])


if __name__ == "__main__":
    main()
