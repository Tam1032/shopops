"""Run traces as JSON lines: one event per model call or tool call."""

import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, path: str | Path | None = None, run_id: str = ""):
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.events: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **data) -> None:
        event = {"ts": round(time.time(), 3), "run_id": self.run_id, "kind": kind, **data}
        self.events.append(event)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
