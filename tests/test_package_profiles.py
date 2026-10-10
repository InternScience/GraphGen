from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_event_profile_assets_are_in_source_manifest_and_wheel(tmp_path):
    import subprocess
    import tarfile
    import zipfile

    output = tmp_path / "dist"
    subprocess.run(["uv", "build", str(REPO), "--out-dir", str(output)], check=True, capture_output=True, text=True)
    wheel = next(output.glob("*.whl"))
    sdist = next(output.glob("*.tar.gz"))
    expected = {
        "pms_event.v1/profile.json",
        "pms_event.v1/prompt.zh.txt",
        "generic_event.v1/profile.json",
        "generic_event.v1/prompt.zh.txt",
        "example_event.v1/profile.json",
        "example_event.v1/prompt.zh.txt",
        "pms_policy.v2/atomic.json",
        "pms_policy.v3/multi_hop.json",
    }
    with zipfile.ZipFile(wheel) as archive:
        wheel_files = archive.namelist()
    with tarfile.open(sdist) as archive:
        sdist_files = archive.getnames()
    assert all(any(path.endswith(item) for path in wheel_files) for item in expected)
    assert all(any(path.endswith(item) for path in sdist_files) for item in expected)
