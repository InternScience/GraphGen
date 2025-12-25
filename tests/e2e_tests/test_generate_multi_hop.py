from pathlib import Path

from .conftest import run_generate_test


def test_generate_multi_hop(tmp_path: Path):
    run_generate_test(
        tmp_path, "examples/generate/generate_multi_hop_qa/multi_hop_config.yaml"
    )
