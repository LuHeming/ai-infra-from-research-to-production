from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a daily learning note.")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format. Defaults to today.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing note.")
    return parser.parse_args()


def resolve_date(raw: str | None) -> date:
    if raw is None:
        return date.today()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit("--date must use YYYY-MM-DD format") from exc


def render_note(note_date: date) -> str:
    template_path = Path(__file__).resolve().parents[1] / "templates" / "daily-note-template.md"
    template = template_path.read_text(encoding="utf-8")
    return template.replace("{{ date }}", note_date.isoformat())


def main() -> None:
    args = parse_args()
    note_date = resolve_date(args.date)
    target = (
        args.root
        / "daily"
        / f"{note_date.year:04d}"
        / f"{note_date.month:02d}"
        / f"{note_date.isoformat()}.md"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not args.force:
        raise SystemExit(f"{target} already exists. Use --force to overwrite.")

    target.write_text(render_note(note_date), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
