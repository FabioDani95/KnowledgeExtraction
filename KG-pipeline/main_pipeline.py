#!/usr/bin/env python3
"""Convenience orchestrator: runs symbolic stage, neural extraction, then KG merge."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], cwd: Path) -> None:
    print(f"\n==> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run symbolic parsing, neural extraction, and KG merge in sequence",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose to the symbolic orchestrator",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config

    # 1) Symbolic stage
    symbolic_cmd = [
        sys.executable,
        "src/Symbolic_orchestrator.py",
        "--stage",
        "symbolic",
        "--pipeline",
        "all",
        "--config",
        str(config_path),
    ]
    if args.verbose:
        symbolic_cmd.append("--verbose")
    run_step(symbolic_cmd, project_root)

    # 2) Neural extraction
    neural_cmd = [
        sys.executable,
        "src/Neural_extraction.py",
        "--config",
        str(config_path),
        "--profile",
        "all",
    ]
    run_step(neural_cmd, project_root)

    # 3) Merge KGs
    merge_cmd = [
        sys.executable,
        "src/merge_kgs.py",
        "--config",
        str(config_path),
    ]
    run_step(merge_cmd, project_root)

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
