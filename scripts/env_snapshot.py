"""Write environment snapshot to a file for reproducibility."""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def write_environment_snapshot(outdir: Path) -> None:
    """Write environment_snapshot.txt to outdir."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
    ]

    try:
        import torch
        lines.append(f"torch: {torch.__version__}")
        lines.append(f"cuda: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_version: {torch.version.cuda}")
            lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    except ImportError:
        lines.append("torch: not installed")
        lines.append("cuda: N/A")

    lines.append("")
    lines.append("--- pip freeze ---")

    try:
        freeze_output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
        ).decode()
        lines.append(freeze_output)
    except Exception as e:
        lines.append(f"(pip freeze failed: {e})")

    (outdir / "environment_snapshot.txt").write_text("\n".join(lines), encoding="utf-8")
