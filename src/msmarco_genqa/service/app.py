"""FastAPI service wrapper for RAG generation.

The service is intentionally optional: importing this module does not require
FastAPI. Install the `serve` extra and run `mgq-serve` to expose `/health` and
`/generate`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from msmarco_genqa.generation.rag_generator import RAGGenerationConfig, RAGGenerator


@dataclass
class GenerationService:
    generator: object

    def answer(self, query: str, passages: Sequence[str]) -> dict[str, object]:
        prediction = self.generator.generate(query, list(passages))
        return {
            "query": query,
            "answer": prediction,
            "n_passages": len(passages),
        }


def load_passages_jsonl(path: str | Path) -> dict[str, str]:
    passages: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            passages[str(row["id"])] = str(row["text"])
    return passages


def build_generator_from_env() -> RAGGenerator:
    return RAGGenerator(
        RAGGenerationConfig(
            model_name=os.getenv("MGQ_MODEL_NAME", "t5-small"),
            revision=os.getenv("MGQ_MODEL_REVISION") or None,
            max_input_length=int(os.getenv("MGQ_MAX_INPUT_LENGTH", "512")),
            max_new_tokens=int(os.getenv("MGQ_MAX_NEW_TOKENS", "64")),
            top_k_passages=int(os.getenv("MGQ_TOP_K_PASSAGES", "3")),
            device=os.getenv("MGQ_DEVICE") or None,
            batch_size=int(os.getenv("MGQ_BATCH_SIZE", "4")),
        )
    )


def create_app(generator: object | None = None):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install the serving extra first: pip install -e '.[serve]'") from exc

    class GenerateRequest(BaseModel):
        query: str = Field(min_length=1)
        passages: list[str] = Field(default_factory=list)

    app = FastAPI(title="MS MARCO GenQA", version="0.1.0")
    service = GenerationService(generator or build_generator_from_env())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict[str, object]:
        return service.answer(req.query, req.passages)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the RAG generator over FastAPI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install the serving extra first: pip install -e '.[serve]'") from exc
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
