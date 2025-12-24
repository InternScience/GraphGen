from pathlib import Path

from .conftest import run_generate_test


def test_generate_vqa(tmp_path: Path):
    run_generate_test(tmp_path, "examples/generate/generate_vqa/vqa_config.yaml")
