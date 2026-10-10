"""Load and validate packaged event extraction profiles."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

PROFILE_SCHEMA = "graphgen_event_profile.v1"
PROFILE_ROOT = files("graphgen.templates").joinpath("event_profiles")


@dataclass(frozen=True)
class EventExtractionProfile:
    profile_id: str
    entity_types: tuple[dict[str, str], ...]
    prompt: str
    anchor_pattern: str | None
    anchor_required: bool
    number_grounding: bool
    event_entity_type: str
    anchor_entity_type: str | None
    legacy_event_id: bool
    manifest_sha256: str
    prompt_sha256: str


def _load_prompt_resource(profile_id: str, prompt_file: str) -> bytes:
    if not prompt_file or Path(prompt_file).name != prompt_file:
        raise ValueError("event_profile_prompt_file_invalid")
    return PROFILE_ROOT.joinpath(profile_id, prompt_file).read_bytes()


def load_event_profile(profile: str | None = None) -> EventExtractionProfile:
    """Load a built-in profile ID or explicit manifest path; fail closed."""
    profile_id = "pms_event.v1" if profile is None else profile
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("event_profile_must_be_non_empty_string")
    if not "/" in profile_id and not "\\" in profile_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", profile_id):
        raise ValueError("event_profile_id_invalid")
    if "/" in profile_id or "\\" in profile_id:
        manifest_path = Path(profile_id).resolve()
        if not manifest_path.is_file():
            raise ValueError(f"event_profile_not_found:{profile_id}")
        manifest_bytes = manifest_path.read_bytes()
        data = json.loads(manifest_bytes.decode("utf-8"))
        prompt_file = str(data.get("prompt_file", ""))
        if not prompt_file or Path(prompt_file).name != prompt_file:
            raise ValueError("event_profile_prompt_file_invalid")
        prompt_path = (manifest_path.parent / prompt_file).resolve()
        if prompt_path.parent != manifest_path.parent:
            raise ValueError("event_profile_prompt_file_invalid")
        prompt_bytes = prompt_path.read_bytes()
    else:
        manifest_resource = PROFILE_ROOT.joinpath(profile_id, "profile.json")
        if not manifest_resource.is_file():
            raise ValueError(f"event_profile_not_found:{profile_id}")
        manifest_bytes = manifest_resource.read_bytes()
        data = json.loads(manifest_bytes.decode("utf-8"))
        prompt_bytes = _load_prompt_resource(profile_id, str(data.get("prompt_file", "")))

    if not isinstance(data, dict) or data.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("event_profile_schema_unsupported")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", str(data.get("profile_id", ""))):
        raise ValueError("event_profile_id_invalid")
    if data.get("profile_id") != profile_id and "/" not in profile_id and "\\" not in profile_id:
        raise ValueError("event_profile_id_mismatch")
    entity_types = data.get("entity_types")
    if not isinstance(entity_types, list) or not entity_types:
        raise ValueError("event_profile_entity_types_required")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in entity_types:
        if not isinstance(item, dict) or set(item) != {"type", "description"}:
            raise ValueError("event_profile_entity_type_invalid")
        name = str(item["type"]).strip().upper()
        description = str(item["description"]).strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", name) or not description or name in seen:
            raise ValueError("event_profile_entity_type_invalid")
        seen.add(name)
        normalized.append({"type": name, "description": description})

    anchor = data.get("anchor")
    if not isinstance(anchor, dict) or set(anchor) != {"required", "pattern", "entity_type"}:
        raise ValueError("event_profile_anchor_invalid")
    required, pattern, anchor_entity_type = anchor["required"], anchor["pattern"], anchor["entity_type"]
    if not isinstance(required, bool) or (pattern is not None and not isinstance(pattern, str)):
        raise ValueError("event_profile_anchor_invalid")
    if required and not pattern:
        raise ValueError("event_profile_required_anchor_pattern_missing")
    if pattern:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError("event_profile_anchor_pattern_invalid") from error
    if anchor_entity_type is not None and (not isinstance(anchor_entity_type, str) or anchor_entity_type.upper() not in seen):
        raise ValueError("event_profile_anchor_entity_type_unknown")
    if isinstance(anchor_entity_type, str):
        anchor_entity_type = anchor_entity_type.upper()

    grounding = data.get("grounding")
    if not isinstance(grounding, dict) or set(grounding) != {"number_grounding"} or not isinstance(grounding["number_grounding"], bool):
        raise ValueError("event_profile_grounding_invalid")
    graph = data.get("graph")
    if not isinstance(graph, dict) or set(graph) != {"event_entity_type", "legacy_event_id"}:
        raise ValueError("event_profile_graph_invalid")
    event_entity_type = str(graph["event_entity_type"]).strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", event_entity_type) or not isinstance(graph["legacy_event_id"], bool):
        raise ValueError("event_profile_event_entity_type_invalid")
    prompt = prompt_bytes.decode("utf-8")
    if not prompt.strip() or prompt.count("{input_text}") != 1:
        raise ValueError("event_profile_prompt_invalid")

    return EventExtractionProfile(
        profile_id=str(data["profile_id"]),
        entity_types=tuple(normalized),
        prompt=prompt,
        anchor_pattern=pattern,
        anchor_required=required,
        number_grounding=grounding["number_grounding"],
        event_entity_type=event_entity_type,
        anchor_entity_type=anchor_entity_type,
        legacy_event_id=graph["legacy_event_id"],
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
    )
