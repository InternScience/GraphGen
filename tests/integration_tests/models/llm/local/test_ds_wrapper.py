import sys

import pytest

from graphgen.models.llm.local.ds_wrapper import DeepSpeedBackend


def test_deepspeed_backend_init(monkeypatch):
    class DummyModel:
        def eval(self):
            pass

    class DummyModule:
        def __init__(self):
            self.module = DummyModel()

    def dummy_initialize(model, config):
        return DummyModule(), None, None, None

    monkeypatch.setitem(
        sys.modules,
        "deepspeed",
        type("ds", (), {"initialize": staticmethod(dummy_initialize)})(),
    )
    backend = DeepSpeedBackend(model=DummyModel())
    assert hasattr(backend.model, "module")
    assert hasattr(backend.model.module, "eval")


def test_deepspeed_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "deepspeed", None)
    with pytest.raises(ImportError):
        DeepSpeedBackend(model=object())
