"""Small experiment-tracking facade.

The default backend writes JSONL events, which keeps local runs reproducible
without requiring a hosted service. Optional MLflow / W&B backends are enabled
only when those packages are installed.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(v) for v in value]
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
        requested_backend = self.backend.lower()
        self.backend = requested_backend
        self.remote_backend: str | None = None
        self.output_dir = Path(self.output_dir)
        if requested_backend not in {"none", "off"}:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.output_dir / "events.jsonl"

        if requested_backend == "jsonl":
            pass
        elif requested_backend == "mlflow":
            self._start_mlflow()
        elif requested_backend in {"wandb", "weights-and-biases"}:
            self._start_wandb()
        elif requested_backend not in {"none", "off"}:
            raise ValueError(f"unknown tracking backend: {self.backend}")
        if requested_backend not in {"none", "off"}:
            self._write_jsonl(
                "run",
                {
                    "requested_backend": requested_backend,
                    "remote_backend": self.remote_backend,
                    "tags": self.tags,
                },
            )

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
        if self.remote_backend == "mlflow":
            self._mlflow.end_run()
        elif self.remote_backend == "wandb":
            self._wandb.finish()

    def _log(self, kind: str, payload: dict[str, Any]) -> None:
        payload = {k: _jsonable(v) for k, v in payload.items()}
        if self.backend not in {"none", "off"}:
            self._write_jsonl(kind, payload)
        if self.remote_backend == "mlflow":
            if kind == "params":
                self._mlflow.log_params(payload)
            elif kind == "metrics":
                self._mlflow.log_metrics(payload["metrics"], step=payload.get("step"))
            elif kind == "artifact":
                self._mlflow.log_artifact(payload["path"], artifact_path=payload.get("name"))
        elif self.remote_backend == "wandb":
            if kind == "params":
                self._wandb.config.update(payload, allow_val_change=True)
            elif kind == "metrics":
                self._wandb.log(payload["metrics"], step=payload.get("step"))
            elif kind == "artifact":
                artifact = self._wandb.Artifact(payload.get("name") or "artifact", type="file")
                artifact.add_file(payload["path"])
                self._wandb.log_artifact(artifact)

    def _write_jsonl(self, kind: str, payload: dict[str, Any]) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_name": self.run_name,
            "kind": kind,
            "payload": {k: _jsonable(v) for k, v in payload.items()},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _start_mlflow(self) -> None:
        try:
            import mlflow  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package state
            self._warn_remote_unavailable("mlflow", exc)
            return

        try:
            self._mlflow = mlflow
            mlflow.start_run(run_name=self.run_name)
            for key, value in self.tags.items():
                mlflow.set_tag(key, _jsonable(value))
            self.remote_backend = "mlflow"
        except Exception as exc:  # pragma: no cover - depends on local tracking config
            self._warn_remote_unavailable("mlflow", exc)

    def _start_wandb(self) -> None:
        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package state
            self._warn_remote_unavailable("wandb", exc)
            return

        try:
            self._wandb = wandb
            wandb.init(project="msmarco-genqa", name=self.run_name, tags=list(self.tags))
            self.remote_backend = "wandb"
        except Exception as exc:  # pragma: no cover - depends on local tracking config
            self._warn_remote_unavailable("wandb", exc)

    def _warn_remote_unavailable(self, backend: str, exc: Exception) -> None:
        warnings.warn(
            f"{backend} tracking is unavailable; continuing with local JSONL events ({exc})",
            RuntimeWarning,
            stacklevel=2,
        )

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
