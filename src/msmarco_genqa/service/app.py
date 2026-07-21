"""FastAPI service wrapper for RAG generation.

The service is intentionally optional: importing this module does not require
FastAPI. Install the `serve` extra and run `mgq-serve` to expose `/health` and
`/generate`.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from msmarco_genqa.generation.rag_generator import RAGGenerationConfig, RAGGenerator
from msmarco_genqa.util.input_validation import (
    InputValidationError,
    iter_jsonl_objects,
    normalize_passage_list,
    normalize_text,
)


@dataclass
class GenerationService:
    generator: object

    def answer(self, query: str, passages: Sequence[str]) -> dict[str, object]:
        clean_query = normalize_text(query, field="query", max_chars=2048)
        clean_passages = normalize_passage_list(
            passages,
            max_passage_chars=4096,
            allow_empty_list=True,
        )
        prediction = self.generator.generate(clean_query, clean_passages)
        return {
            "query": clean_query,
            "answer": prediction,
            "n_passages": len(clean_passages),
        }


def load_passages_jsonl(path: str | Path) -> dict[str, str]:
    passages: dict[str, str] = {}
    for line_number, row in iter_jsonl_objects(path):
        if "id" not in row:
            raise InputValidationError("missing id", field="id", line_number=line_number, path=Path(path))
        if "text" not in row:
            raise InputValidationError(
                "missing text",
                field="text",
                line_number=line_number,
                path=Path(path),
            )
        doc_id = normalize_text(
            row["id"],
            field="id",
            line_number=line_number,
            path=Path(path),
        )
        if doc_id in passages:
            raise InputValidationError(
                f"duplicate passage id {doc_id!r}",
                field="id",
                line_number=line_number,
                path=Path(path),
            )
        passages[doc_id] = normalize_text(
            row["text"],
            field="text",
            allow_empty=False,
            max_chars=4096,
            line_number=line_number,
            path=Path(path),
        )
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
            max_query_chars=int(os.getenv("MGQ_MAX_QUERY_CHARS", "2048")),
            max_passage_chars=int(os.getenv("MGQ_MAX_PASSAGE_CHARS", "4096")),
        )
    )


def is_loopback_host(host: str) -> bool:
    """Return whether *host* is an explicit local-only bind target."""

    normalized = host.strip()
    if normalized.casefold().rstrip(".") == "localhost":
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_bind_host(host: str, *, allow_remote: bool) -> None:
    """Reject accidental network exposure unless the operator opts in."""

    if not allow_remote and not is_loopback_host(host):
        raise ValueError(
            "refusing to bind to a non-loopback host without --allow-remote; "
            "the demo service has no authentication, TLS, or rate limiting"
        )


def create_app(generator: object | None = None):
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install the serving extra first: pip install -e '.[serve]'") from exc

    class GenerateRequest(BaseModel):
        query: str = Field(min_length=1)
        passages: list[str] = Field(default_factory=list)

    app = FastAPI(title="MS MARCO GenQA", version="0.1.0")
    service = GenerationService(generator or build_generator_from_env())

    @app.exception_handler(InputValidationError)
    def input_validation_error(_request, exc: InputValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "input_validation",
                    "message": exc.message,
                    "field": exc.field,
                }
            },
        )

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
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help=(
            "allow a non-loopback bind; the demo service has no authentication, "
            "TLS, or rate limiting"
        ),
    )
    args = parser.parse_args()
    try:
        validate_bind_host(args.host, allow_remote=args.allow_remote)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install the serving extra first: pip install -e '.[serve]'") from exc
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
