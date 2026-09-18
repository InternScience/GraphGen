"""Versioned prompt profiles: config-driven prompt customization for generators.

PMS fork capability (see docs/PMS_PATCHES.md #1). Official GraphGen hard-codes all
prompts as Python constants under ``graphgen.templates``; this module adds a
generic, fail-closed override channel:

- a *profile* is a directory containing one JSON file per generator
  ``TEMPLATE_KEY`` (e.g. ``atomic.json``);
- each JSON file must mirror the exact key structure of the corresponding
  official template (same languages, same nested keys, leaf values are str);
- ``profile`` values are either built-in profile ids resolved against
  ``graphgen/templates/profiles/`` or explicit filesystem paths (containing
  a path separator).

Default behaviour is untouched: when no profile is applied every generator
falls back to the official template constant byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from graphgen.templates import (
    AGGREGATED_GENERATION_PROMPT,
    ATOMIC_GENERATION_PROMPT,
    COT_GENERATION_PROMPT,
    FILL_IN_BLANK_GENERATION_PROMPT,
    MAQ_GENERATION_PROMPT,
    MCQ_GENERATION_PROMPT,
    MULTI_HOP_GENERATION_PROMPT,
    TF_GENERATION_PROMPT,
    VQA_GENERATION_PROMPT,
)

BUILTIN_PROFILE_ROOT = Path(__file__).resolve().parent / "profiles"

#: template_key -> official template constant (structure validation source).
OFFICIAL_TEMPLATES: dict[str, Any] = {
    "atomic": ATOMIC_GENERATION_PROMPT,
    "multi_hop": MULTI_HOP_GENERATION_PROMPT,
    "aggregated": AGGREGATED_GENERATION_PROMPT,
    "cot": COT_GENERATION_PROMPT,
    "multi_choice": MCQ_GENERATION_PROMPT,
    "multi_answer": MAQ_GENERATION_PROMPT,
    "fill_in_blank": FILL_IN_BLANK_GENERATION_PROMPT,
    "masked_fill_in_blank": AGGREGATED_GENERATION_PROMPT,
    "true_false": TF_GENERATION_PROMPT,
    "vqa": VQA_GENERATION_PROMPT,
}


def resolve_profile_dir(profile: str) -> Path:
    """Resolve a profile id / path to its directory; fail-closed if missing."""
    if not profile or not isinstance(profile, str):
        raise ValueError("prompt_profile_must_be_non_empty_string")
    if "/" in profile or "\\" in profile:
        profile_dir = Path(profile)
    else:
        profile_dir = BUILTIN_PROFILE_ROOT / profile
    if not profile_dir.is_dir():
        raise ValueError(f"prompt_profile_not_found:{profile}")
    return profile_dir


def _validate_structure(official: Any, override: Any, path: str) -> None:
    """Override must mirror the official template's key structure exactly."""
    if isinstance(official, dict):
        if not isinstance(override, dict):
            raise ValueError(f"prompt_profile_structure_mismatch:{path}")
        if set(override.keys()) != set(official.keys()):
            raise ValueError(
                f"prompt_profile_structure_mismatch:{path}:"
                f"keys={sorted(map(str, override.keys()))}"
            )
        for key in official:
            _validate_structure(official[key], override[key], f"{path}.{key}")
        return
    if not isinstance(override, str) or not override.strip():
        raise ValueError(f"prompt_profile_structure_mismatch:{path}:non_string_leaf")


def load_profile_template(profile: str, template_key: str) -> dict[str, Any]:
    """Load one generator's template override from a profile.

    :param profile: profile id (builtin) or filesystem path.
    :param template_key: generator TEMPLATE_KEY, must exist in OFFICIAL_TEMPLATES.
    :return: parsed override dict (same structure as the official template).
    """
    if template_key not in OFFICIAL_TEMPLATES:
        raise ValueError(f"prompt_profile_unknown_template_key:{template_key}")
    profile_file = resolve_profile_dir(profile) / f"{template_key}.json"
    if not profile_file.is_file():
        raise ValueError(f"prompt_profile_template_missing:{profile}/{template_key}")
    try:
        override = json.loads(profile_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"prompt_profile_json_invalid:{profile_file}:{error}") from error
    _validate_structure(OFFICIAL_TEMPLATES[template_key], override, template_key)
    return override


def profile_template_sha256s(profile: str, template_keys: list[str]) -> dict[str, str]:
    """Lineage fingerprint: SHA-256 of each override file actually in use."""
    profile_dir = resolve_profile_dir(profile)
    digests: dict[str, str] = {}
    for template_key in template_keys:
        profile_file = profile_dir / f"{template_key}.json"
        if not profile_file.is_file():
            continue
        digests[template_key] = hashlib.sha256(profile_file.read_bytes()).hexdigest()
    return digests


def apply_prompt_profile(generator: Any, profile: str) -> None:
    """Attach profile templates to a generator instance (BaseGenerator API)."""
    generator.apply_prompt_profile(profile)
