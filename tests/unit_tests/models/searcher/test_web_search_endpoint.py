"""Tests for the configurable endpoint of the web searchers."""

from __future__ import annotations

from graphgen.models.searcher.web import bing_search, google_search
from graphgen.models.searcher.web.bing_search import BING_SEARCH_V7_ENDPOINT, BingSearch
from graphgen.models.searcher.web.google_search import (
    GOOGLE_SEARCH_ENDPOINT,
    GoogleSearch,
)


class FakeResponse:
    ok = True
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _record_get(monkeypatch, module, payload):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(payload)

    monkeypatch.setattr(module.requests, "get", fake_get)
    return calls


def test_bing_search_uses_default_endpoint(monkeypatch):
    calls = _record_get(monkeypatch, bing_search, {"webPages": {"value": [{"a": 1}]}})

    assert BingSearch("key").search("query") == [{"a": 1}]
    assert calls[0][0] == BING_SEARCH_V7_ENDPOINT


def test_bing_search_uses_custom_endpoint(monkeypatch):
    calls = _record_get(monkeypatch, bing_search, {"webPages": {"value": [{"a": 1}]}})
    endpoint = "https://search.example.com/v7.0/search"

    assert BingSearch("key", endpoint=endpoint).search("query") == [{"a": 1}]
    assert calls[0][0] == endpoint
    assert calls[0][1]["headers"] == {"Ocp-Apim-Subscription-Key": "key"}


def test_google_search_uses_default_endpoint(monkeypatch):
    calls = _record_get(monkeypatch, google_search, {"items": [{"a": 1}]})

    assert GoogleSearch("key", "cx").search("query") == [{"a": 1}]
    assert calls[0][0] == GOOGLE_SEARCH_ENDPOINT


def test_google_search_uses_custom_endpoint(monkeypatch):
    calls = _record_get(monkeypatch, google_search, {"items": [{"a": 1}]})
    endpoint = "https://search.example.com/customsearch/v1"

    assert GoogleSearch("key", "cx", endpoint=endpoint).search("query") == [{"a": 1}]
    assert calls[0][0] == endpoint
    assert calls[0][1]["params"]["cx"] == "cx"
