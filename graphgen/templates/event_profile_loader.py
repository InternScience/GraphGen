"""Load and validate packaged event extraction profiles."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

PROFILE_SCHEMA = "graphgen_event_profile.v1"
PROFILE_ROOT = files("graphgen.templates").joinpath("event_profiles")
_PROFILE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ENTITY_TYPE_RE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")


@dataclass(frozen=True)
class EventExtractionProfile:
    profile_id: str
    entity_types: tuple[dict[str, str], ...]
    prompt: str
    anchor_pattern: Optional[str]
    anchor_required: bool
    number_grounding: bool
    event_entity_type: str
    anchor_entity_type: Optional[str]
    legacy_event_id: bool
    manifest_sha256: str
    prompt_sha256: str


def _load_prompt_resource(profile_id: str, prompt_file: str) -> bytes:
    if not prompt_file or Path(prompt_file).name != prompt_file:
        raise ValueError("event_profile_prompt_file_invalid")
    return PROFILE_ROOT.joinpath(profile_id, prompt_file).read_bytes()


def _read_profile_bytes(profile_id: str) -> tuple[bytes, bytes]:
    """Read manifest and prompt bytes for a packaged ID or a filesystem path."""
    if "/" in profile_id or "\\" in profile_id:
        # an explicit manifest path is preserved as given; its prompt file is
        # resolved next to the manifest, never inside the package
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
        return manifest_bytes, prompt_path.read_bytes()

    if not _PROFILE_ID_RE.fullmatch(profile_id):
        raise ValueError("event_profile_id_invalid")
    manifest_resource = PROFILE_ROOT.joinpath(profile_id, "profile.json")
    if not manifest_resource.is_file():
        raise ValueError(f"event_profile_not_found:{profile_id}")
    manifest_bytes = manifest_resource.read_bytes()
    data = json.loads(manifest_bytes.decode("utf-8"))
    return manifest_bytes, _load_prompt_resource(profile_id, str(data.get("prompt_file", "")))


def _parse_entity_types(raw: Any) -> tuple[list[dict[str, str]], set[str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("event_profile_entity_types_required")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"type", "description"}:
            raise ValueError("event_profile_entity_type_invalid")
        name = str(item["type"]).strip().upper()
        description = str(item["description"]).strip()
        if not _ENTITY_TYPE_RE.fullmatch(name) or not description or name in seen:
            raise ValueError("event_profile_entity_type_invalid")
        seen.add(name)
        normalized.append({"type": name, "description": description})
    return normalized, seen


def _parse_anchor(raw: Any, known_types: set[str]) -> tuple[bool, Optional[str], Optional[str]]:
    if not isinstance(raw, dict) or set(raw) != {"required", "pattern", "entity_type"}:
        raise ValueError("event_profile_anchor_invalid")
    required, pattern, anchor_entity_type = raw["required"], raw["pattern"], raw["entity_type"]
    if not isinstance(required, bool) or (pattern is not None and not isinstance(pattern, str)):
        raise ValueError("event_profile_anchor_invalid")
    if required and not pattern:
        raise ValueError("event_profile_required_anchor_pattern_missing")
    if pattern:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError("event_profile_anchor_pattern_invalid") from error
    if anchor_entity_type is None:
        return required, pattern, None
    if not isinstance(anchor_entity_type, str) or anchor_entity_type.upper() not in known_types:
        raise ValueError("event_profile_anchor_entity_type_unknown")
    return required, pattern, anchor_entity_type.upper()


def _parse_grounding(raw: Any) -> bool:
    if not isinstance(raw, dict) or set(raw) != {"number_grounding"}:
        raise ValueError("event_profile_grounding_invalid")
    if not isinstance(raw["number_grounding"], bool):
        raise ValueError("event_profile_grounding_invalid")
    return raw["number_grounding"]


def _parse_graph(raw: Any) -> tuple[str, bool]:
    if not isinstance(raw, dict) or set(raw) != {"event_entity_type", "legacy_event_id"}:
        raise ValueError("event_profile_graph_invalid")
    event_entity_type = str(raw["event_entity_type"]).strip().upper()
    legacy_event_id = raw["legacy_event_id"]
    if not _ENTITY_TYPE_RE.fullmatch(event_entity_type) or not isinstance(legacy_event_id, bool):
        raise ValueError("event_profile_event_entity_type_invalid")
    return event_entity_type, legacy_event_id


def load_event_profile(profile: str) -> EventExtractionProfile:
    """Load a built-in profile ID or an explicit manifest path; fail closed.

    An event profile is always required: there is no implicit default.
    """
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("event_profile_must_be_non_empty_string")
    profile_id = profile
    manifest_bytes, prompt_bytes = _read_profile_bytes(profile_id)
    data = json.loads(manifest_bytes.decode("utf-8"))

    if not isinstance(data, dict) or data.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("event_profile_schema_unsupported")
    if not _PROFILE_ID_RE.fullmatch(str(data.get("profile_id", ""))):
        raise ValueError("event_profile_id_invalid")
    if data.get("profile_id") != profile_id and "/" not in profile_id and "\\" not in profile_id:
        raise ValueError("event_profile_id_mismatch")

    normalized, known_types = _parse_entity_types(data.get("entity_types"))
    anchor_required, anchor_pattern, anchor_entity_type = _parse_anchor(data.get("anchor"), known_types)
    number_grounding = _parse_grounding(data.get("grounding"))
    event_entity_type, legacy_event_id = _parse_graph(data.get("graph"))

    prompt = prompt_bytes.decode("utf-8")
    if not prompt.strip() or prompt.count("{input_text}") != 1:
        raise ValueError("event_profile_prompt_invalid")

    return EventExtractionProfile(
        profile_id=str(data["profile_id"]),
        entity_types=tuple(normalized),
        prompt=prompt,
        anchor_pattern=anchor_pattern,
        anchor_required=anchor_required,
        number_grounding=number_grounding,
        event_entity_type=event_entity_type,
        anchor_entity_type=anchor_entity_type,
        legacy_event_id=legacy_event_id,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
    )
