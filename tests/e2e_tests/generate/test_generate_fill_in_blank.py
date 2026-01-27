from pathlib import Path

from tests.e2e_tests.conftest import run_generate_test


def test_generate_fill_in_blank(tmp_path: Path):
    run_generate_test(
        tmp_path, "examples/generate/generate_fill_in_blank_qa/fill_in_blank_config.yaml"
    )
