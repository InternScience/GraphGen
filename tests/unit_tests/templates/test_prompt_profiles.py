"""Tests for the versioned prompt-profile mechanism."""

import json
from pathlib import Path

import pytest

from graphgen.models.generator.atomic_generator import AtomicGenerator
from graphgen.templates import ATOMIC_GENERATION_PROMPT
from graphgen.templates.prompt_profiles import (
    BUILTIN_PROFILE_ROOT,
    load_profile_template,
    profile_template_sha256s,
    resolve_profile_dir,
)


class _FakeLLM:
    tokenizer = None


def test_builtin_example_profile_matches_official_structure():
    override = load_profile_template("example.v1", "atomic")
    assert override == ATOMIC_GENERATION_PROMPT


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError, match="prompt_profile_not_found"):
        resolve_profile_dir("no_such_profile")


def test_missing_template_key_fails_closed(tmp_path: Path):
    profile_dir = tmp_path / "my.profile"
    profile_dir.mkdir()
    with pytest.raises(ValueError, match="prompt_profile_template_missing"):
        load_profile_template(str(profile_dir), "atomic")


def test_invalid_json_fails_closed(tmp_path: Path):
    profile_dir = tmp_path / "my.profile"
    profile_dir.mkdir()
    (profile_dir / "atomic.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt_profile_json_invalid"):
        load_profile_template(str(profile_dir), "atomic")


def test_structure_mismatch_fails_closed(tmp_path: Path):
    profile_dir = tmp_path / "my.profile"
    profile_dir.mkdir()
    (profile_dir / "atomic.json").write_text(
        json.dumps({"xx": {}, "zh": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="prompt_profile_structure_mismatch"):
        load_profile_template(str(profile_dir), "atomic")


def test_unknown_template_key_fails_closed(tmp_path: Path):
    with pytest.raises(ValueError, match="prompt_profile_unknown_template_key"):
        load_profile_template(str(tmp_path), "no_such_key")


def test_profile_template_sha256s(tmp_path: Path):
    profile_dir = tmp_path / "my.profile"
    profile_dir.mkdir()
    (profile_dir / "atomic.json").write_text("{}", encoding="utf-8")
    digests = profile_template_sha256s(str(profile_dir), ["atomic", "cot"])
    assert set(digests) == {"atomic"}


def test_generator_profile_overrides_are_instance_local():
    generator_a = AtomicGenerator(_FakeLLM())
    generator_b = AtomicGenerator(_FakeLLM())
    generator_a.apply_prompt_profile("example.v1")
    batch = (
        [("n1", {"description": "alpha"})],
        [("n1", "n2", {"description": "beta"})],
    )
    prompt_a = generator_a.build_prompt(batch)
    prompt_b = generator_b.build_prompt(batch)
    assert prompt_a == prompt_b
    language = "en"
    assert prompt_a.startswith(ATOMIC_GENERATION_PROMPT[language].split("{")[0])


def test_builtin_profile_root_exists():
    assert (BUILTIN_PROFILE_ROOT / "example.v1" / "atomic.json").is_file()
