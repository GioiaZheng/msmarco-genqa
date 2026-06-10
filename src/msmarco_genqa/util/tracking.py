"""Small experiment-tracking facade.

The default backend writes JSONL events, which keeps local runs reproducible
without requiring a hosted service. Optional MLflow / W&B backends are enabled
only when those packages are installed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


@dataclass
class ExperimentTracker:
    backend: str = "jsonl"
    output_dir: Path = Path("outputs/tracking")
    run_name: str = "run"
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend = self.backend.lower()
        self.output_dir = Path(self.output_dir)
        if self.backend == "jsonl":
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.output_dir / "events.jsonl"
        elif self.backend == "mlflow":
            import mlflow  # type: ignore

            self._mlflow = mlflow
            mlflow.start_run(run_name=self.run_name)
            for key, value in self.tags.items():
                mlflow.set_tag(key, _jsonable(value))
        elif self.backend in {"wandb", "weights-and-biases"}:
            import wandb  # type: ignore

            self._wandb = wandb
            wandb.init(project="msmarco-genqa", name=self.run_name, tags=list(self.tags))
        elif self.backend in {"none", "off"}:
            pass
        else:
            raise ValueError(f"unknown tracking backend: {self.backend}")

    def log_params(self, params: dict[str, Any]) -> None:
        self._log("params", params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        payload: dict[str, Any] = {"metrics": metrics}
        if step is not None:
            payload["step"] = step
        self._log("metrics", payload)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        self._log("artifact", {"path": str(path), "name": name})

    def close(self) -> None:
        if self.backend == "mlflow":
            self._mlflow.end_run()
        elif self.backend in {"wandb", "weights-and-biases"}:
            self._wandb.finish()

    def _log(self, kind: str, payload: dict[str, Any]) -> None:
        payload = {k: _jsonable(v) for k, v in payload.items()}
        if self.backend == "jsonl":
            record = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_name": self.run_name,
                "kind": kind,
                "payload": payload,
            }
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        elif self.backend == "mlflow":
            if kind == "params":
                self._mlflow.log_params(payload)
            elif kind == "metrics":
                self._mlflow.log_metrics(payload["metrics"], step=payload.get("step"))
            elif kind == "artifact":
                self._mlflow.log_artifact(payload["path"], artifact_path=payload.get("name"))
        elif self.backend in {"wandb", "weights-and-biases"}:
            if kind == "params":
                self._wandb.config.update(payload, allow_val_change=True)
            elif kind == "metrics":
                self._wandb.log(payload["metrics"], step=payload.get("step"))
            elif kind == "artifact":
                artifact = self._wandb.Artifact(payload.get("name") or "artifact", type="file")
                artifact.add_file(payload["path"])
                self._wandb.log_artifact(artifact)

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
