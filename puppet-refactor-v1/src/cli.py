#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.refactor_engine import refactor_module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="Path to module root")
    ap.add_argument("--task", required=True, help="Refactor task")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument(
        "--mode",
        choices=["diff", "plan", "report"],
        default="diff",
        help="Output mode: diff (default), plan, or full report",
    )
    args = ap.parse_args()

    res = refactor_module(args.module, args.task, cfg_path=args.config)

    if args.mode == "diff":
        print(res.diff)
    elif args.mode == "plan":
        print(res.plan)
    else:
        print(res.report_md)


if __name__ == "__main__":
    main()
