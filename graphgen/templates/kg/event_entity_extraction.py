"""Backward-compatible exports for the original PMS event extraction prompt."""
from __future__ import annotations

from graphgen.templates.event_profile_loader import load_event_profile

_pms_profile = load_event_profile("pms_event.v1")
PMS_ENTITY_TYPES: list[dict] = list(_pms_profile.entity_types)
TEMPLATE_ZH = _pms_profile.prompt
EVENT_ENTITY_EXTRACTION_PROMPT = {
    "en": TEMPLATE_ZH,
    "zh": TEMPLATE_ZH,
    "FORMAT": {"entity_types": None},
}
INPUT_FORMAT = {
    "items_template": '{{"type": "request", "data": {{"items": [{{"id": 1, "content": {content!r}}}], '
    '"meta": {{"entity_types": {entity_types}}}}}}}',
}
