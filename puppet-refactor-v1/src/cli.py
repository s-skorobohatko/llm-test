#!/usr/bin/env python3
import argparse
from src.refactor_engine import refactor_module

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    res = refactor_module(args.module, args.task, cfg_path=args.config)
    print(res.report_md)

if __name__ == "__main__":
    main()
