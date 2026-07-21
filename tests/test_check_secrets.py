from __future__ import annotations

from scripts.check_secrets import scan_file


def _scan(tmp_path, text: str):
    path = tmp_path / "candidate.txt"
    path.write_text(text, encoding="utf-8")
    return scan_file(path)


def test_line_level_hint_does_not_hide_a_credential(tmp_path):
    token = "hf_" + "A1b2" * 9
    assert _scan(tmp_path, f"sample output accidentally included {token}\n") == [
        (1, "Hugging Face token")
    ]


def test_credential_shaped_placeholder_is_allowed(tmp_path):
    placeholder = "hf_" + "x" * 36
    assert _scan(tmp_path, f"HF_TOKEN={placeholder}\n") == []


def test_modern_openai_token_shape_is_detected(tmp_path):
    token = "sk-" + "proj-" + "A1b2" * 10
    assert _scan(tmp_path, f"OPENAI_API_KEY={token}\n") == [(1, "sk-prefixed API key")]
