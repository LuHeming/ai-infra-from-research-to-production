#!/usr/bin/env python3
"""Small, reproducible model-compression lab for CPU or CUDA."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def percentile(values: Iterable[float], quantile: float) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    position = (len(items) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    weight = position - lower
    return items[lower] * (1.0 - weight) + items[upper] * weight


def pack_signed_int4(values: Tensor) -> tuple[Tensor, int]:
    """Pack signed values in [-8, 7] into two nibbles per byte."""
    flattened = values.to(torch.int16).flatten()
    if flattened.numel() and (
        flattened.min().item() < -8 or flattened.max().item() > 7
    ):
        raise ValueError("INT4 values must be in [-8, 7]")
    original_numel = flattened.numel()
    if original_numel % 2:
        flattened = F.pad(flattened, (0, 1))
    unsigned = torch.bitwise_and(flattened, 0x0F).to(torch.uint8)
    low = unsigned[0::2]
    high = torch.bitwise_left_shift(unsigned[1::2], 4)
    return torch.bitwise_or(low, high), original_numel


def unpack_signed_int4(packed: Tensor, original_numel: int) -> Tensor:
    """Unpack two's-complement nibbles into int8 values."""
    packed = packed.to(torch.uint8).flatten()
    low = torch.bitwise_and(packed, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(packed, 4), 0x0F)
    unsigned = torch.stack((low, high), dim=1).flatten()[:original_numel]
    signed = unsigned.to(torch.int16)
    signed = torch.where(signed >= 8, signed - 16, signed)
    return signed.to(torch.int8)


def groupwise_symmetric_quantize(
    weight: Tensor,
    bits: int,
    group_size: int,
) -> dict[str, Any]:
    """Quantize each row in groups along the input-feature dimension."""
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor")
    if bits not in (4, 8):
        raise ValueError("this lab implements only INT4 and INT8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    rows, columns = weight.shape
    padded_columns = math.ceil(columns / group_size) * group_size
    padding = padded_columns - columns
    working = weight.float()
    if padding:
        working = F.pad(working, (0, padding))

    groups = working.reshape(rows, padded_columns // group_size, group_size)
    qmax = (1 << (bits - 1)) - 1
    scales = groups.abs().amax(dim=-1, keepdim=True) / qmax
    scales = scales.clamp_min(torch.finfo(torch.float32).eps)
    quantized = torch.round(groups / scales).clamp(-qmax, qmax).to(torch.int8)

    packed: Tensor | None = None
    if bits == 4:
        packed, original_numel = pack_signed_int4(quantized)
        quantized = unpack_signed_int4(packed, original_numel).reshape_as(quantized)

    restored = (quantized.float() * scales).reshape(rows, padded_columns)
    restored = restored[:, :columns].to(weight.dtype)

    scale_tensor = scales.squeeze(-1)
    ideal_weight_bytes = math.ceil(rows * padded_columns * bits / 8)
    packed_weight_bytes = tensor_bytes(packed) if packed is not None else ideal_weight_bytes
    artifact_bytes = packed_weight_bytes + tensor_bytes(scale_tensor)

    return {
        "weight": restored,
        "quantized": quantized,
        "scales": scale_tensor,
        "packed": packed,
        "padding": padding,
        "qmin": -qmax,
        "qmax": qmax,
        "artifact_bytes": artifact_bytes,
        "packed_weight_bytes": packed_weight_bytes,
        "scale_bytes": tensor_bytes(scale_tensor),
        "int8_container_bytes": tensor_bytes(quantized),
        "execution_path": "dequantized dense weight; educational, not a low-bit kernel",
    }


def magnitude_prune(weight: Tensor, sparsity: float) -> Tensor:
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be between 0 and 1")
    result = weight.clone()
    prune_count = round(result.numel() * sparsity)
    if prune_count <= 0:
        return result
    if prune_count >= result.numel():
        return torch.zeros_like(result)
    indices = torch.topk(
        result.abs().flatten(),
        k=prune_count,
        largest=False,
        sorted=False,
    ).indices
    result.flatten()[indices] = 0
    return result


def prune_two_of_four(weight: Tensor) -> Tensor:
    """Keep the two largest magnitudes in every consecutive group of four."""
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor")
    if weight.shape[1] % 4:
        raise ValueError("the last dimension must be divisible by 4 for 2:4 pruning")
    groups = weight.reshape(weight.shape[0], -1, 4)
    keep_indices = torch.topk(groups.abs(), k=2, dim=-1, sorted=False).indices
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(-1, keep_indices, True)
    return (groups * mask).reshape_as(weight)


def low_rank_factors(weight: Tensor, rank: int) -> tuple[Tensor, Tensor, float]:
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor")
    maximum_rank = min(weight.shape)
    if not 1 <= rank <= maximum_rank:
        raise ValueError(f"rank must be between 1 and {maximum_rank}")
    u, singular_values, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    left = u[:, :rank] * singular_values[:rank]
    right = vh[:rank, :]
    total_energy = singular_values.square().sum()
    retained = singular_values[:rank].square().sum() / total_energy.clamp_min(1e-30)
    return left.to(weight.dtype), right.to(weight.dtype), float(retained.item())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    operation: Callable[[], Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float | int]:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        synchronize(device)
        samples: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            operation()
            synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "count": len(samples),
        "mean_ms": round(statistics.fmean(samples), 6),
        "median_ms": round(statistics.median(samples), 6),
        "p95_ms": round(percentile(samples, 0.95), 6),
        "min_ms": round(min(samples), 6),
        "max_ms": round(max(samples), 6),
    }


def relative_mse(reference: Tensor, candidate: Tensor) -> float:
    numerator = torch.mean((reference.float() - candidate.float()).square())
    denominator = torch.mean(reference.float().square()).clamp_min(1e-30)
    return float((numerator / denominator).item())


def cosine_similarity(reference: Tensor, candidate: Tensor) -> float:
    value = F.cosine_similarity(
        reference.float().flatten(),
        candidate.float().flatten(),
        dim=0,
        eps=1e-12,
    )
    return float(value.item())


def zero_fraction(tensor: Tensor) -> float:
    return float((tensor == 0).float().mean().item())


def add_outliers(weight: Tensor, scale: float) -> Tensor:
    if scale <= 0:
        raise ValueError("outlier_scale must be positive")
    result = weight.clone()
    if scale == 1.0:
        return result
    flattened = result.flatten()
    outlier_count = max(1, flattened.numel() // 1000)
    step = max(1, flattened.numel() // outlier_count)
    flattened[::step] *= scale
    return result


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def build_experiment(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = DTYPES[args.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU Linear is not a portable baseline; use float32")

    weight = torch.randn(args.rows, args.cols, device=device, dtype=dtype)
    weight = add_outliers(weight, args.outlier_scale)
    inputs = torch.randn(args.batch_size, args.cols, device=device, dtype=dtype)

    def baseline_operation():
        return F.linear(inputs, weight)
    baseline_output = baseline_operation()
    dense_bytes = tensor_bytes(weight)

    method_details: dict[str, Any]
    artifact_bytes = dense_bytes
    candidate_weight = weight
    left: Tensor | None = None
    right: Tensor | None = None

    if args.method == "baseline":
        method_details = {
            "execution_path": "dense baseline",
            "artifact_bytes": dense_bytes,
        }
    elif args.method == "quantize":
        quantized = groupwise_symmetric_quantize(weight, args.bits, args.group_size)
        candidate_weight = quantized.pop("weight")
        quantized.pop("quantized")
        quantized.pop("scales")
        quantized.pop("packed")
        artifact_bytes = int(quantized["artifact_bytes"])
        method_details = quantized
        method_details.update({"bits": args.bits, "group_size": args.group_size})
    elif args.method == "unstructured":
        candidate_weight = magnitude_prune(weight, args.sparsity)
        method_details = {
            "requested_sparsity": args.sparsity,
            "artifact_bytes": dense_bytes,
            "execution_path": "dense tensor containing zeros",
        }
    elif args.method == "2to4":
        candidate_weight = prune_two_of_four(weight)
        method_details = {
            "pattern": "2:4",
            "artifact_bytes": dense_bytes,
            "execution_path": "dense tensor containing a 2:4 pattern",
        }
    elif args.method == "low-rank":
        left, right, retained_energy = low_rank_factors(weight, args.rank)
        candidate_weight = left @ right
        artifact_bytes = tensor_bytes(left) + tensor_bytes(right)
        method_details = {
            "rank": args.rank,
            "retained_spectral_energy": retained_energy,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "artifact_bytes": artifact_bytes,
            "execution_path": "two dense linear operations",
        }
    else:
        raise ValueError(f"unsupported method: {args.method}")

    if left is not None and right is not None:
        def candidate_operation():
            return F.linear(F.linear(inputs, right), left)
    else:
        def candidate_operation():
            return F.linear(inputs, candidate_weight)

    candidate_output = candidate_operation()
    baseline_latency = benchmark(
        baseline_operation,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    candidate_latency = benchmark(
        candidate_operation,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    speedup = (
        float(baseline_latency["median_ms"]) / float(candidate_latency["median_ms"])
    )

    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or platform.machine()
            ),
            "dtype": args.dtype,
        },
        "config": {
            "method": args.method,
            "rows": args.rows,
            "cols": args.cols,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "outlier_scale": args.outlier_scale,
        },
        "representation": {
            "dense_weight_bytes": dense_bytes,
            "compressed_artifact_bytes": artifact_bytes,
            "idealized_compression_ratio": round(dense_bytes / artifact_bytes, 6),
            "materialized_execution_weight_bytes": tensor_bytes(candidate_weight),
            "actual_zero_fraction": round(zero_fraction(candidate_weight), 8),
            **method_details,
        },
        "correctness": {
            "relative_weight_mse": relative_mse(weight, candidate_weight),
            "relative_output_mse": relative_mse(baseline_output, candidate_output),
            "output_cosine_similarity": cosine_similarity(
                baseline_output,
                candidate_output,
            ),
            "output_max_abs_error": float(
                (baseline_output.float() - candidate_output.float()).abs().max().item()
            ),
            "all_finite": bool(torch.isfinite(candidate_output).all().item()),
        },
        "performance": {
            "baseline": baseline_latency,
            "candidate": candidate_latency,
            "median_speedup": round(speedup, 6),
            "warning": (
                "Quantize and pruning modes execute a dense materialized weight. "
                "They demonstrate numerics and the deployment gap, not production "
                "low-bit or sparse-kernel speed."
            ),
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible matrix-level model-compression lab.",
    )
    parser.add_argument(
        "--method",
        choices=("baseline", "quantize", "unstructured", "2to4", "low-rank"),
        default="baseline",
    )
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--outlier-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/raw-results/week-03/compression-lab.json"),
    )
    args = parser.parse_args()

    positive = ("rows", "cols", "batch_size", "group_size", "rank", "iterations")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not 0.0 <= args.sparsity <= 1.0:
        parser.error("--sparsity must be between 0 and 1")
    if args.outlier_scale <= 0:
        parser.error("--outlier-scale must be positive")
    if args.method == "2to4" and args.cols % 4:
        parser.error("--cols must be divisible by 4 for 2:4 pruning")
    if args.method == "low-rank" and args.rank > min(args.rows, args.cols):
        parser.error("--rank cannot exceed min(rows, cols)")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = build_experiment(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)

    summary = {
        "method": report["config"]["method"],
        "compression_ratio": report["representation"]["idealized_compression_ratio"],
        "relative_output_mse": report["correctness"]["relative_output_mse"],
        "output_cosine": report["correctness"]["output_cosine_similarity"],
        "zero_fraction": report["representation"]["actual_zero_fraction"],
        "baseline_median_ms": report["performance"]["baseline"]["median_ms"],
        "candidate_median_ms": report["performance"]["candidate"]["median_ms"],
        "median_speedup": report["performance"]["median_speedup"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
