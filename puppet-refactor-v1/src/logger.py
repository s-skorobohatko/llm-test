from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Logger:
    enabled: bool = True
    prefix: str = "[refactor]"

    def info(self, msg: str) -> None:
        if not self.enabled:
            return
        print(f"{_ts()} {self.prefix} {msg}", file=sys.stderr, flush=True)

    def phase(self, name: str, extra: Optional[str] = None) -> None:
        if extra:
            self.info(f"== {name} :: {extra}")
        else:
            self.info(f"== {name}")

    def metric(self, key: str, value: object) -> None:
        self.info(f"{key}={value}")
