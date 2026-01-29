from pathlib import Path

from tests.e2e_tests.conftest import run_generate_test


def test_evaluate_kg(tmp_path: Path):
    run_generate_test(
        tmp_path, "examples/evaluate/evaluate_triple/triple_evaluation_config.yaml"
    )
