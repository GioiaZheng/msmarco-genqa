"""Edge-case tests for ``msmarco_genqa.evaluation.nli_grounding``.

The forward-pass path is intentionally NOT exercised here — that would
download a ~140 MB model on every test run. The integration test for
the model output happens in the W7-A audit run itself. These tests
cover the deterministic input-validation / edge-case paths that don't
touch the model.
"""

from __future__ import annotations

import pytest

from msmarco_genqa.evaluation.nli_grounding import per_query_nli_entailment


class TestInputValidation:
    def test_empty_inputs_returns_empty_list(self):
        assert per_query_nli_entailment([], []) == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            per_query_nli_entailment(["a"], [])
        with pytest.raises(ValueError, match="same length"):
            per_query_nli_entailment([], [["x"]])
