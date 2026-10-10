# GraphGen Fork: reusable Graph + SAG event capabilities

This repository is the independently maintained `Leon-Algo/GraphGen` fork of GraphGen. It remains compatible with the upstream project where documented, while shipping opt-in fork capabilities. The upstream PR is not a release dependency: projects consume reviewed fork commits/tags directly.

## What is reusable

- **GraphGen foundation:** pipeline/operators, storage integration, generators, and the default LightRAG + ECE route.
- **Fork capabilities:** versioned generation prompt profiles, output gates, generation concurrency controls, and the opt-in event/entity star-graph builder plus event-join partitioner.
- **SAG-derived event mechanics:** event extraction response contract, numeric grounding, star encoding (one event node linked to participants), deterministic merging, and shared-participant joins. Adapted assets and code retain their upstream license/source attribution in `docs/PMS_PATCHES.md`.

These mechanisms do not define a universal business ontology. A consuming project selects a versioned event extraction profile and its own generation prompt profile. The `pms_event.v1` and `pms_policy.v2/v3` profiles are PMS-specific examples and are never implicitly applied to unrelated projects. `generic_event.v1` and `example_event.v1` are synthetic examples, not validated domain policies.

## Install a reviewed fork version

For a reproducible project, pin an immutable commit in the consuming project's `pyproject.toml` and commit its lock file:

```toml
[project]
dependencies = [
  "graphg @ git+https://github.com/Leon-Algo/GraphGen.git@<full-commit-sha>",
]
```

Resolve and lock with `uv lock` (or add it with `uv add`). Avoid moving branch refs in production. For short-lived development only, install `git+https://github.com/Leon-Algo/GraphGen.git@pms/main`. The distribution name is `graphg`; Python imports use `graphgen`.

## Select event extraction explicitly

The default `kg_method` remains `light_rag`; the normal ECE partitioner remains unchanged. To use the event route, configure the build node explicitly:

```yaml
- id: build_kg
  op_name: build_kg
  type: map
  params:
    kg_method: event_entity
    event_profile: generic_event.v1

- id: partition
  op_name: partition
  type: map
  params:
    method: event_join
    method_params:
      event_entity_type: EVENT
      anchor_attribute: anchor
      require_same_anchor: false
      max_events_per_community: 6
      max_tokens_per_community: 4096
```

For a project-specific profile, copy a profile directory into a project-controlled location and point `event_profile` to the absolute or explicitly configured `profile.json` path; do not rely on the process working directory. Never edit a profile already used by a released run. The loader validates the manifest and prompt, fails closed on missing/invalid inputs, and records SHA-256 fingerprints on the loaded profile object.

An event profile contains:

- `schema_version` and immutable `profile_id`;
- `prompt_file` with the required `{input_text}` placeholder;
- `entity_types` allowlist used both in prompt input and response validation;
- `anchor` with required/optional status, optional regex, and optional anchor entity type;
- `grounding.number_grounding`;
- `graph.event_entity_type` and `graph.legacy_event_id`.

Regex capture group 1 is the anchor when present; otherwise the complete match is used. Event nodes carry the extracted anchor as an explicit string attribute; the partitioner never parses it from the node ID. `require_same_anchor: true` rejects missing anchors and prevents joins across anchor values. Set it to false only when cross-event joining without anchor isolation is acceptable. Limits remain explicit partition parameters.

The packaged `generic_event.v1` profile demonstrates an anchor-free schema; `example_event.v1` demonstrates a custom anchor and domain entity set. They are fixtures, not a claim that a particular project's extraction policy has been validated. PMS workflows should explicitly select `pms_event.v1` and the corresponding PMS generation policy; legacy configurations with `kg_method: event_entity` and no profile retain the PMS-compatible profile.

## Generation profiles and gates

Generation profiles use the independent `prompt_profile` parameter (built-in profile ID or explicit profile directory); do not assume it selects event extraction policy. PMS output gate `pms_project_anchor` is opt-in and is not a generic default. Each project should own its prompt policy and only enable domain gates it has validated.

## Profile authoring checklist

1. Define a versioned profile ID and the consuming project's allowed entity types.
2. Write prompt instructions for that domain; include `{input_text}` exactly once or as required by the application.
3. Set an explicit anchor regex or `required: false`; set `anchor_entity_type` only if it is in the entity allowlist.
4. Decide numeric grounding and whether event_join must enforce same-anchor isolation.
5. Test supported, missing, malformed, ungrounded, and unknown-entity responses with a fake LLM before real model smoke.
6. Test extraction → star graph → partition on representative project data, then pin the profile bytes and GraphGen commit in the consuming project.

## Upgrade and compatibility policy

- A released tag and profile directory are immutable. Fixes and policy changes get a new commit and a new tag/profile ID; do not retarget tags or edit released profile contents in place.
- Consumers pin a full commit SHA or immutable release tag and commit their lock file. Upgrade by changing the pin deliberately, rerunning tests/smokes, and reviewing GraphGen's release notes and `docs/PMS_PATCHES.md`.
- `light_rag` remains the default. `event_entity` is opt-in. PMS profile defaults exist only for backward compatibility of the PMS fork integration; new projects should always specify their profile explicitly.
- Fork release/tag format, validation gate, attribution, upstream base, and exact release evidence are maintained in `docs/RELEASING.md`.
