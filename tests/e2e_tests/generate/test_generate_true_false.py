from pathlib import Path

from tests.e2e_tests.conftest import run_generate_test


def test_generate_true_false(tmp_path: Path):
    run_generate_test(
        tmp_path, "examples/generate/generate_true_false_qa/true_false_config.yaml"
    )
