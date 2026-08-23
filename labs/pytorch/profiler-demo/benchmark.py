from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one Hugging Face causal language model.")
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--prompt", default="AI infrastructure is")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-profile", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def synchronize(torch_module: Any, device: str) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be non-negative and --repeat must be positive")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install torch transformers") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False.")

    dtype = getattr(torch, args.dtype)
    if args.device == "cpu" and dtype == torch.float16:
        raise SystemExit("float16 on CPU is not recommended; use --dtype float32 or bfloat16.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=dtype,
    )
    model.eval().to(args.device)

    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    input_tokens = int(encoded["input_ids"].numel())

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(**encoded)
        synchronize(torch, args.device)

        latencies_ms: list[float] = []
        for _ in range(args.repeat):
            synchronize(torch, args.device)
            start = time.perf_counter()
            model(**encoded)
            synchronize(torch, args.device)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        if not args.no_profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as profiler:
                with torch.profiler.record_function("model_forward"):
                    model(**encoded)
                synchronize(torch, args.device)

            trace_path = args.output_dir / "profile-trace.json"
            profiler.export_chrome_trace(str(trace_path))
            sort_key = "cuda_time_total" if args.device == "cuda" else "cpu_time_total"
            print(
                profiler.key_averages(group_by_input_shape=True).table(
                    sort_by=sort_key,
                    row_limit=15,
                )
            )

    mean_ms = statistics.fmean(latencies_ms)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None
    )
    peak_reserved_mb = (
        torch.cuda.max_memory_reserved() / 1024**2 if args.device == "cuda" else None
    )
    result = {
        "model": args.model,
        "model_revision": args.model_revision,
        "device": args.device,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "input_tokens": input_tokens,
        "warmup_runs": args.warmup,
        "repeat_runs": args.repeat,
        "seed": args.seed,
        "latency_ms": {
            "mean": mean_ms,
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "samples": latencies_ms,
        },
        "input_tokens_per_second": input_tokens / (mean_ms / 1000),
        "peak_allocated_memory_mb": peak_memory_mb,
        "peak_reserved_memory_mb": peak_reserved_mb,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    result_path = args.output_dir / "benchmark.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved benchmark to {result_path}")


if __name__ == "__main__":
    main()
