# Generate QAs from an Event-Entity Graph

An opt-in alternative to the default LightRAG entity/relation graph. Instead of
entities linked by relations, `build_kg` extracts **events** and their
**participants** into a star graph (one event node, one membership edge per
participant), and `partition` groups events that share participants into
communities. Any generator can then run on those communities; this example
uses `multi_hop`.

```bash
bash examples/generate/generate_event_qa/generate_event.sh
```

## Configuration

`build_kg` node:

| key             | value                                                                                                                                                  |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `kg_method`     | `event_entity` (default `light_rag`, which ignores the keys below)                                                                                      |
| `event_profile` | **required**. Built-in profile id (`generic_event.v1`) or a filesystem path to a `profile.json`; the prompt file named in the manifest must sit next to it |

`partition` node, `method: event_join`, `method_params`:

| key                        | default | meaning                                                                                                   |
|----------------------------|---------|-----------------------------------------------------------------------------------------------------------|
| `max_events_per_community` | 6       | stop growing a community after this many events                                                           |
| `min_events_per_community` | 1       | drop communities with fewer events                                                                        |
| `max_tokens_per_community` | 4096    | token budget per community (event `length` attributes)                                                    |
| `require_same_anchor`      | false   | only join events whose `anchor` attribute matches; set `true` only with a profile whose `anchor.required` is true |
| `event_entity_type`        | `EVENT` | node `entity_type` that marks event nodes; must match `graph.event_entity_type` of the profile            |
| `anchor_attribute`         | `anchor`| node attribute holding the anchor                                                                         |

## Event profiles

A profile is a directory with `profile.json` and the prompt file it names
(see `graphgen/templates/event_profiles/generic_event.v1/`). The manifest
declares the allowed participant `entity_types`, an optional `anchor` regex
(e.g. a ticket or project id that every event must mention), whether numbers
in extracted events must appear in the source chunk (`grounding`), and the
`entity_type` used for event nodes. Validation is fail-closed: a malformed
manifest or prompt raises at service construction time, and there is no
default profile.

To use your own profile, copy the built-in directory, edit it, and point
`event_profile` at the new `profile.json` path.
