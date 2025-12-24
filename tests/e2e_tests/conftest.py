import json
import os
import subprocess
from pathlib import Path


def run_generate_test(tmp_path: Path, config_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    config_path = repo_root / config_name

    result = subprocess.run(
        [
            "python",
            "-m",
            "graphgen.run",
            "--config_file",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    run_root = repo_root / "cache" / "output"
    assert run_root.exists(), f"{run_root} does not exist"
    run_folders = sorted(
        [p for p in run_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True
    )
    assert run_folders, f"No run folders found in {run_root}"
    run_folder = run_folders[0]

    node_dirs = [p for p in run_folder.iterdir() if p.is_dir()]
    assert node_dirs, f"No node outputs found in {run_folder}"

    json_files = []
    for nd in node_dirs:
        json_files.extend(nd.glob("*.jsonl"))
    assert json_files, f"No JSONL output found under nodes in {run_folder}"

    log_file = repo_root / "cache" / "logs" / "Driver.log"
    assert log_file.exists(), "No log file generated"

    with open(json_files[0], "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        assert first_line, "JSONL output is empty"
        data = json.loads(first_line)
    assert isinstance(data, dict), "First JSONL record is not a dict"

    return run_folder, json_files[0]
