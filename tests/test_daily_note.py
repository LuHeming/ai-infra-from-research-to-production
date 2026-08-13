from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path


def load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_daily_note.py"
    spec = importlib.util.spec_from_file_location("create_daily_note", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load create_daily_note.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_note_contains_date() -> None:
    module = load_module()
    content = module.render_note(date(2026, 8, 5))
    assert "# 2026-08-05 学习记录" in content
    assert "## 今日目标" in content
