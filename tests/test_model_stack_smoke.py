from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_smoke_module():
    module_path = PROJECT_ROOT / "scripts" / "smoke_model_stack.py"
    spec = importlib.util.spec_from_file_location("smoke_model_stack", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_smoke_config_reads_generation_and_dense_sections():
    mod = _load_smoke_module()
    cfg = {
        "generation": {
            "model_name": "t5-small",
            "revision": "abc123",
            "max_input_length": 256,
            "max_new_tokens": 64,
            "top_k_passages": 2,
        },
        "dense": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": "def456",
        },
    }

    out = mod.build_smoke_config(cfg, max_new_tokens=12)

    assert out.generation_model_name == "t5-small"
    assert out.generation_revision == "abc123"
    assert out.dense_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert out.dense_revision == "def456"
    assert out.max_input_length == 256
    assert out.max_new_tokens == 12
    assert out.top_k_passages == 2


def test_build_smoke_config_uses_config_generation_budget_when_not_overridden():
    mod = _load_smoke_module()
    cfg = {
        "generation": {"model_name": "t5-small", "max_new_tokens": 32},
        "dense": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
    }

    out = mod.build_smoke_config(cfg, max_new_tokens=None)

    assert out.max_new_tokens == 32
    assert out.generation_revision is None
    assert out.dense_revision is None


def test_build_smoke_config_rejects_missing_model_names():
    mod = _load_smoke_module()

    with pytest.raises(ValueError, match="generation.model_name"):
        mod.build_smoke_config({"generation": {}, "dense": {"model_name": "encoder"}})

    with pytest.raises(ValueError, match="dense.model_name"):
        mod.build_smoke_config({"generation": {"model_name": "generator"}, "dense": {}})
