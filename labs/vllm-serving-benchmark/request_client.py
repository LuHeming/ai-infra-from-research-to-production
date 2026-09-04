#!/usr/bin/env python3
"""Closed-loop streaming benchmark client for a vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RequestResult:
    request_id: int
    success: bool
    input_chars: int
    output_chars: int
    output_chunks: int
    ttft_ms: float | None
    e2e_ms: float
    stream_chunk_gap_ms: list[float]
    usage: dict[str, Any] | None
    error: str | None


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile for 0 <= quantile <= 1."""
    items = sorted(float(value) for value in values)
    if not items:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    position = (len(items) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    weight = position - lower
    return items[lower] * (1.0 - weight) + items[upper] * weight


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    items = [float(value) for value in values]
    if not items:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(items),
        "mean": round(statistics.fmean(items), 3),
        "p50": round(percentile(items, 0.50) or 0.0, 3),
        "p95": round(percentile(items, 0.95) or 0.0, 3),
        "p99": round(percentile(items, 0.99) or 0.0, 3),
        "max": round(max(items), 3),
    }


def load_prompts(path: Path | None, fallback: str) -> list[str]:
    if path is None:
        return [fallback]
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSONL: {exc}"
                ) from exc
            if isinstance(item, str):
                prompt = item
            elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
                prompt = item["prompt"]
            else:
                raise ValueError(
                    f"{path}:{line_number} must be a JSON string or object "
                    'with a string "prompt" field'
                )
            if prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"{path} does not contain any non-empty prompts")
    return prompts


def make_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    if args.include_usage:
        common["stream_options"] = {"include_usage": True}
    if args.endpoint == "chat":
        common["messages"] = [{"role": "user", "content": prompt}]
    else:
        common["prompt"] = prompt
    return common


def extract_piece(event: dict[str, Any], endpoint: str) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    if endpoint == "chat":
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return ""
        content = delta.get("content")
    else:
        content = choice.get("text")
    return content if isinstance(content, str) else ""


def compact_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(512).decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    suffix = f": {body}" if body else ""
    return f"HTTP {exc.code} {exc.reason}{suffix}"


def run_request(
    request_id: int,
    prompt: str,
    args: argparse.Namespace,
) -> RequestResult:
    endpoint_path = (
        "/v1/chat/completions"
        if args.endpoint == "chat"
        else "/v1/completions"
    )
    url = args.base_url.rstrip("/") + endpoint_path
    data = json.dumps(make_payload(args, prompt)).encode("utf-8")
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "User-Agent": "ai-infra-week2-benchmark/1.0",
    }
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    start = time.perf_counter()
    first_content_at: float | None = None
    previous_content_at: float | None = None
    chunk_gaps: list[float] = []
    output_chars = 0
    output_chunks = 0
    usage: dict[str, Any] | None = None

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                event_text = line[5:].strip()
                if not event_text or event_text == "[DONE]":
                    continue
                try:
                    event = json.loads(event_text)
                except json.JSONDecodeError:
                    continue

                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = event_usage

                piece = extract_piece(event, args.endpoint)
                if not piece:
                    continue

                now = time.perf_counter()
                if first_content_at is None:
                    first_content_at = now
                elif previous_content_at is not None:
                    chunk_gaps.append((now - previous_content_at) * 1000.0)
                previous_content_at = now
                output_chars += len(piece)
                output_chunks += 1

        end = time.perf_counter()
        return RequestResult(
            request_id=request_id,
            success=True,
            input_chars=len(prompt),
            output_chars=output_chars,
            output_chunks=output_chunks,
            ttft_ms=(
                (first_content_at - start) * 1000.0
                if first_content_at is not None
                else None
            ),
            e2e_ms=(end - start) * 1000.0,
            stream_chunk_gap_ms=chunk_gaps,
            usage=usage,
            error=None,
        )
    except urllib.error.HTTPError as exc:
        error = compact_http_error(exc)
    except urllib.error.URLError as exc:
        error = f"URL error: {exc.reason}"
    except TimeoutError:
        error = f"timeout after {args.timeout} seconds"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    end = time.perf_counter()
    return RequestResult(
        request_id=request_id,
        success=False,
        input_chars=len(prompt),
        output_chars=output_chars,
        output_chunks=output_chunks,
        ttft_ms=(
            (first_content_at - start) * 1000.0
            if first_content_at is not None
            else None
        ),
        e2e_ms=(end - start) * 1000.0,
        stream_chunk_gap_ms=chunk_gaps,
        usage=usage,
        error=error,
    )


def execute_closed_loop(
    args: argparse.Namespace,
    prompts: list[str],
    num_requests: int,
    concurrency: int,
) -> tuple[list[RequestResult], float]:
    started = time.perf_counter()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(run_request, index, prompts[index % len(prompts)], args)
            for index in range(num_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    duration = time.perf_counter() - started
    results.sort(key=lambda item: item.request_id)
    return results, duration


def build_report(
    args: argparse.Namespace,
    prompts: list[str],
    results: list[RequestResult],
    duration: float,
) -> dict[str, Any]:
    successful = [item for item in results if item.success]
    ttft = [item.ttft_ms for item in successful if item.ttft_ms is not None]
    e2e = [item.e2e_ms for item in successful]
    chunk_gaps = [
        gap
        for item in successful
        for gap in item.stream_chunk_gap_ms
    ]
    completion_tokens = sum(
        int(item.usage.get("completion_tokens", 0))
        for item in successful
        if item.usage
    )
    prompt_tokens = sum(
        int(item.usage.get("prompt_tokens", 0))
        for item in successful
        if item.usage
    )

    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "model": args.model,
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "warmup_requests": args.warmup_requests,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timeout_seconds": args.timeout,
            "include_usage": args.include_usage,
            "prompt_source": (
                str(args.prompts_file) if args.prompts_file else "inline"
            ),
            "unique_prompts": len(prompts),
        },
        "summary": {
            "duration_seconds": round(duration, 6),
            "completed": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "success_rate": round(len(successful) / len(results), 6),
            "requests_per_second": round(
                len(successful) / duration if duration else 0.0, 6
            ),
            "prompt_tokens_from_usage": prompt_tokens or None,
            "completion_tokens_from_usage": completion_tokens or None,
            "output_tokens_per_second_from_usage": (
                round(completion_tokens / duration, 6)
                if completion_tokens and duration
                else None
            ),
            "ttft_ms": summarize(ttft),
            "e2e_ms": summarize(e2e),
            "stream_chunk_gap_ms": summarize(chunk_gaps),
        },
        "requests": [asdict(item) for item in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop streaming benchmark for a vLLM server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--endpoint",
        choices=("completions", "chat"),
        default="completions",
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY"),
        help="Defaults to VLLM_API_KEY; never written to results.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain prefill and decode in two concise sentences.",
    )
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--include-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request token usage in the final streaming event.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/raw-results/week-02-client.json"),
    )
    args = parser.parse_args()

    for name in ("num_requests", "concurrency", "max_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests cannot be negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        prompts = load_prompts(args.prompts_file, args.prompt)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.warmup_requests:
        warmups, _ = execute_closed_loop(
            args,
            prompts,
            args.warmup_requests,
            min(args.concurrency, args.warmup_requests),
        )
        warmup_failures = [item for item in warmups if not item.success]
        if warmup_failures:
            raise SystemExit(
                "warmup failed: " + (warmup_failures[0].error or "unknown error")
            )

    results, duration = execute_closed_loop(
        args,
        prompts,
        args.num_requests,
        args.concurrency,
    )
    report = build_report(args, prompts, results, duration)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
