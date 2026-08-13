from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def nested_get(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSV summary from benchmark JSON files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    columns = {
        "file": None,
        "date": "date",
        "model": "model",
        "method": "method",
        "gpu": "environment.gpu",
        "torch": "environment.torch",
        "cuda": "environment.cuda",
        "latency_ms_mean": "metrics.latency_ms_mean",
        "latency_ms_p50": "metrics.latency_ms_p50",
        "latency_ms_p95": "metrics.latency_ms_p95",
        "tokens_per_second": "metrics.tokens_per_second",
        "peak_memory_mb": "metrics.peak_memory_mb",
        "perplexity": "metrics.perplexity",
        "accuracy": "metrics.accuracy",
    }

    rows: list[dict[str, Any]] = []
    for path in sorted(args.input_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Skip {path}: {exc}")
            continue

        row = {"file": path.name}
        for column, data_path in columns.items():
            if data_path is not None:
                row[column] = nested_get(data, data_path)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
