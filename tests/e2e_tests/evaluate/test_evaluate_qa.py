from pathlib import Path

from tests.e2e_tests.conftest import run_generate_test


def test_evaluate_qa(tmp_path: Path):
    run_generate_test(
        tmp_path, "examples/evaluate/evaluate_qa/qa_evaluation_config.yaml"
    )
