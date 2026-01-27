from pathlib import Path

from tests.e2e_tests.conftest import run_generate_test


def test_generate_multi_answer(tmp_path: Path):
    run_generate_test(
        tmp_path, "examples/generate/generate_multi_answer_qa/multi_answer_config.yaml"
    )
