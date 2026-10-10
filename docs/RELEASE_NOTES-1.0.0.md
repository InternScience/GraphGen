# GraphGen Fork 1.0.0 — initial multi-project SDK release

Tag: `graphgen-fork-v1.0.0` (to be created only after passing release validation).

Distribution version: `1.0.0` (`graphg`). Upstream base: `InternScience/GraphGen` commit `3a3eb097318d34b07f6f31cb60de5b20712d65ce`.

## Highlights

- Profile-driven event/entity extraction: projects supply entity types, prompt, anchor policy, numeric grounding and event graph identity without patching GraphGen source.
- Built-in `pms_event.v1` preserves PMS extraction behavior and legacy event IDs; `generic_event.v1` and `example_event.v1` are synthetic examples only.
- `event_join` reads explicit anchor node attributes and supports opt-in same-anchor isolation independently of event ID encoding.
- Default LightRAG + ECE remains unchanged; event extraction is opt-in.
- Event and generation prompt profile assets are included in wheel and sdist and load through package resources.
- See [multi-project SDK guide](MULTI_PROJECT_SDK.md), [release policy](RELEASING.md), and [patch ledger](PMS_PATCHES.md).

## Compatibility and validation record

GraphGen partition integration: 12 passed after correcting tests to the synchronous storage/partitioner APIs. Profile, fake-LLM builder/partition, and package tests: 12 passed. PMS contract regression: `tests/test_pms_graphgen_policy_generation.py` — 23 passed, 1 skipped.
A real one-chunk GraphGen run using the configured PMS launcher/model pool completed extraction → event_join → COT generation. The synthetic non-PMS event anchored on `CASE-A7`; the graph contained the event and two participants; the final QA identified the operator and completed mission, included no internal event hash, and the process exited 0. The run used a synthetic fixture and validates integration mechanics, not any production project's domain accuracy.

The existing GraphGen partitioner integration suite could not collect because its tests import an unexported `NetworkXStorage` from `graphgen.models`. An unrelated GraphGen extraction e2e test failed at Ray GCS startup (`Could not read 'temp_dir' from GCS`). These environment/baseline failures are not represented as passing tests.

SAG-derived code/prompt attribution and MIT licensing are tracked in `docs/PMS_PATCHES.md`.
