from pathlib import Path

from tests.e2e_tests.conftest import run_generate_test


def test_extract_schema_guided(tmp_path: Path):
    run_generate_test(
        tmp_path,
        "examples/extract/extract_schema_guided/schema_guided_extraction_config.yaml",
    )
