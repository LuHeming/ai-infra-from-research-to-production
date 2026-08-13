from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def safe_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def collect_torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}

    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]

    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a reproducible environment snapshot.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "torch": collect_torch_info(),
        "git_commit": safe_command(["git", "rev-parse", "HEAD"]),
        "nvidia_smi": safe_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "pip_freeze": safe_command([sys.executable, "-m", "pip", "freeze"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
