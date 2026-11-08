"""Tests for the event profile loader and the opt-in service wiring."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from graphgen.models.kg_builder.event_entity_kg_builder import EventEntityKGBuilder
from graphgen.templates.event_profile_loader import load_event_profile
from graphgen.utils import CURRENT_LOGGER_VAR


class FakeLLM:
    tokenizer = SimpleNamespace(count_tokens=lambda text: len(text))


def _write_profile(root, profile_id: str = "custom_event.v1", prompt: str = "{input_text}") -> str:
    manifest = {
        "schema_version": "graphgen_event_profile.v1",
        "profile_id": profile_id,
        "prompt_file": "prompt.txt",
        "entity_types": [
            {"type": "ACTION", "description": "An action or occurrence"},
            {"type": "ACTOR", "description": "A participant in the event"},
        ],
        "anchor": {"required": False, "pattern": None, "entity_type": None},
        "grounding": {"number_grounding": False},
        "graph": {"event_entity_type": "EVENT", "legacy_event_id": False},
    }
    root.mkdir(parents=True)
    (root / "profile.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "prompt.txt").write_text(prompt, encoding="utf-8")
    return str((root / "profile.json").resolve())


def test_generic_builtin_profile_loads():
    profile = load_event_profile("generic_event.v1")
    assert profile.profile_id == "generic_event.v1"
    assert profile.event_entity_type == "EVENT"
    assert profile.anchor_required is False
    assert profile.anchor_pattern is None
    assert "{input_text}" in profile.prompt
    assert [item["type"] for item in profile.entity_types] == ["ACTION", "ACTOR"]


def test_event_profile_is_always_required():
    with pytest.raises(ValueError, match="event_profile_must_be_non_empty_string"):
        load_event_profile(None)
    with pytest.raises(ValueError, match="event_profile_not_found"):
        load_event_profile("pms_event.v1")


def test_disk_profile_preserved_through_service_and_builder(tmp_path, monkeypatch):
    manifest = _write_profile(tmp_path / "profiles" / "custom_event.v1")

    import graphgen.common.init_storage as init_storage_module
    import graphgen.operators.build_kg.build_kg_service as service_module

    captured = {}

    def fake_build_event_entity_kg(**kwargs):
        captured.update(kwargs)
        builder = EventEntityKGBuilder(llm_client=FakeLLM(), profile=kwargs["event_profile"])
        assert builder.profile.prompt == "{input_text}"
        return [], []

    fake_storage = lambda **kwargs: SimpleNamespace(index_done_callback=lambda: None)  # noqa: E731
    monkeypatch.setattr(init_storage_module, "init_storage", fake_storage)
    monkeypatch.setattr(service_module, "init_storage", fake_storage)
    monkeypatch.setattr(service_module, "init_llm", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(service_module, "build_event_entity_kg", fake_build_event_entity_kg)

    service = service_module.BuildKGService(
        working_dir=str(tmp_path / "cache"),
        kv_backend="json_kv",
        kg_method="event_entity",
        event_profile=manifest,
    )
    assert service.event_profile == manifest

    # BaseOperator.__call__ installs the context logger before process()
    token = CURRENT_LOGGER_VAR.set(service.get_logger())
    try:
        service.process(
            [{"_trace_id": "chunk-1", "type": "text", "content": "The team completed the migration."}]
        )
    finally:
        CURRENT_LOGGER_VAR.reset(token)

    assert captured["event_profile"] == manifest
    builder = EventEntityKGBuilder(llm_client=FakeLLM(), profile=captured["event_profile"])
    assert builder.profile.profile_id == "custom_event.v1"
    assert builder.profile.prompt == "{input_text}"


def test_event_entity_mode_without_profile_fails_closed(tmp_path, monkeypatch):
    import graphgen.common.init_storage as init_storage_module
    import graphgen.operators.build_kg.build_kg_service as service_module

    fake_storage = lambda **kwargs: SimpleNamespace(index_done_callback=lambda: None)  # noqa: E731
    monkeypatch.setattr(init_storage_module, "init_storage", fake_storage)
    monkeypatch.setattr(service_module, "init_storage", fake_storage)
    monkeypatch.setattr(service_module, "init_llm", lambda *args, **kwargs: FakeLLM())

    with pytest.raises(ValueError, match="event_profile_must_be_non_empty_string"):
        service_module.BuildKGService(
            working_dir=str(tmp_path / "cache"),
            kv_backend="json_kv",
            kg_method="event_entity",
        )
