from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "labs"
    / "vllm-serving-benchmark"
    / "request_client.py"
)
SPEC = importlib.util.spec_from_file_location("week2_request_client", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CLIENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLIENT
SPEC.loader.exec_module(CLIENT)


def test_percentile_uses_linear_interpolation() -> None:
    assert CLIENT.percentile([], 0.5) is None
    assert CLIENT.percentile([1], 0.95) == 1
    assert CLIENT.percentile([1, 2, 3, 4], 0.5) == 2.5


def test_summarize_returns_distribution() -> None:
    summary = CLIENT.summarize([10, 20, 30, 40])
    assert summary["count"] == 4
    assert summary["mean"] == 25.0
    assert summary["p50"] == 25.0
    assert summary["max"] == 40.0


def test_extract_piece_handles_completion_and_chat() -> None:
    completion = {"choices": [{"text": "hello"}]}
    chat = {"choices": [{"delta": {"content": "world"}}]}
    assert CLIENT.extract_piece(completion, "completions") == "hello"
    assert CLIENT.extract_piece(chat, "chat") == "world"
    assert CLIENT.extract_piece({"choices": []}, "chat") == ""


def test_load_prompts_accepts_strings_and_objects(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '"first"\n{"prompt": "second"}\n',
        encoding="utf-8",
    )
    assert CLIENT.load_prompts(path, "fallback") == ["first", "second"]
    assert CLIENT.load_prompts(None, "fallback") == ["fallback"]
