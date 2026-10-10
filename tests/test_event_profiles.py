from graphgen.models.generator.support import build_generation_view, validate_support


def test_generation_view_hides_event_ids_and_maps_support_back_to_canonical_id():
    event_id = "event:opaque-hash"
    item = {
        "nodes": [
            (event_id, {"entity_type": "EVENT", "graph_role": "event", "description": "Migration completed：CASE-A7 event fact"}),
            ("Operator", {"entity_type": "ACTOR", "description": "performed the action"}),
        ],
        "edges": [
            (event_id, "Operator", {"description": "participated"}),
        ],
    }
    view = build_generation_view(item)
    assert view["nodes"][0][0] == "Migration completed"
    assert event_id not in str(view["nodes"])
    mapped = view["resolve_support"]({"cited": ["Migration completed"]})
    assert mapped == {"cited": [event_id]}
    assert validate_support(mapped, set(view["canonical_nodes"])) == mapped
