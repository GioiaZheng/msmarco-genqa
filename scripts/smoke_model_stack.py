"""Opt-in smoke test for the HuggingFace model stack.

This script intentionally loads real model weights. It is not part of the
default unit-test or CI path; run it before accepting torch / transformers /
sentence-transformers upgrades that could change generation, embedding, or
reranking behaviour.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelStackSmokeConfig:
    generation_model_name: str
    generation_revision: str | None
    dense_model_name: str
    dense_revision: str | None
    max_input_length: int
    max_new_tokens: int
    top_k_passages: int


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return cfg


def build_smoke_config(
    cfg: dict[str, Any],
    *,
    max_new_tokens: int | None = 16,
) -> ModelStackSmokeConfig:
    generation = cfg.get("generation") or {}
    dense = cfg.get("dense") or {}
    if not isinstance(generation, dict):
        raise ValueError("generation must be a mapping")
    if not isinstance(dense, dict):
        raise ValueError("dense must be a mapping")

    generation_model = str(generation.get("model_name") or "").strip()
    dense_model = str(dense.get("model_name") or "").strip()
    if not generation_model:
        raise ValueError("generation.model_name is required")
    if not dense_model:
        raise ValueError("dense.model_name is required")

    return ModelStackSmokeConfig(
        generation_model_name=generation_model,
        generation_revision=generation.get("revision") or None,
        dense_model_name=dense_model,
        dense_revision=dense.get("revision") or None,
        max_input_length=int(generation.get("max_input_length", 512)),
        max_new_tokens=int(max_new_tokens or generation.get("max_new_tokens", 64)),
        top_k_passages=int(generation.get("top_k_passages", 3)),
    )


def package_versions() -> dict[str, str]:
    import sentence_transformers
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
    }


def run_generation_smoke(config: ModelStackSmokeConfig, *, device: str) -> dict[str, Any]:
    from msmarco_genqa.generation.rag_generator import RAGGenerationConfig, RAGGenerator

    query = "Where is Rome located?"
    passages = [
        "Rome is the capital city of Italy and is located in the Lazio region.",
        "The Tiber River runs through Rome.",
        "Italy is a country in Southern Europe.",
    ]
    generator = RAGGenerator(
        RAGGenerationConfig(
            model_name=config.generation_model_name,
            revision=config.generation_revision,
            max_input_length=config.max_input_length,
            max_new_tokens=config.max_new_tokens,
            top_k_passages=config.top_k_passages,
            device=device,
            batch_size=1,
        )
    )
    answer = generator.generate(query, passages).strip()
    if not answer:
        raise RuntimeError("generation smoke produced an empty answer")
    return {
        "model": config.generation_model_name,
        "revision": config.generation_revision,
        "answer": answer,
    }


def run_dense_smoke(config: ModelStackSmokeConfig, *, device: str) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        config.dense_model_name,
        revision=config.dense_revision,
        device=device,
    )
    embeddings = encoder.encode(
        [
            "Where is Rome located?",
            "Rome is the capital city of Italy and is located in the Lazio region.",
        ],
        normalize_embeddings=True,
    )
    shape = list(embeddings.shape)
    if len(shape) != 2 or shape[0] != 2 or shape[1] <= 0:
        raise RuntimeError(f"dense smoke produced invalid embedding shape: {shape}")
    norm = float((embeddings[0] ** 2).sum() ** 0.5)
    if not 0.99 <= norm <= 1.01:
        raise RuntimeError(f"dense smoke expected normalized embeddings, got norm={norm:.4f}")
    return {
        "model": config.dense_model_name,
        "revision": config.dense_revision,
        "embedding_shape": shape,
        "embedding_norm": norm,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
        help="Config file whose model names and revisions should be smoke-tested.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the smoke run. Keep CPU for reproducible local checks.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Short generation budget for the smoke prompt.",
    )
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_smoke_config(
        load_config(args.config),
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps({"config": asdict(cfg)}, sort_keys=True))
    print(json.dumps({"packages": package_versions()}, sort_keys=True))
    if not args.skip_generation:
        print(json.dumps({"generation": run_generation_smoke(cfg, device=args.device)}))
    if not args.skip_dense:
        print(json.dumps({"dense": run_dense_smoke(cfg, device=args.device)}, sort_keys=True))
    print("model_stack_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
