"""RAG generator: a Seq2Seq model conditioned on a question and the top
retrieved passages.

The format mirrors the prototype in week 3 of the notebooks::

    question: <query> context: <passage_1> <passage_2> ...

so that the only thing that changes between the prototype and the official
baseline is the *quality* of the retrieved passages (toy 3-passage corpus
vs. real top-k from the BM25 index built in week 2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass
class RAGGenerationConfig:
    model_name: str = "t5-small"
    # HF revision pin (40-hex SHA). ``None`` means "use whatever main is
    # pointing at right now" — the historical behaviour. Configs produced
    # after infra/reproducibility-round1 always set this.
    revision: str | None = None
    max_input_length: int = 512
    max_new_tokens: int = 64
    top_k_passages: int = 3
    device: str | None = None
    batch_size: int = 4


class RAGGenerator:
    """Concatenation-style RAG generator for short-form answers."""

    def __init__(self, config: RAGGenerationConfig | None = None) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self.config = config or RAGGenerationConfig()
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        logger.info(
            "Loading %s (revision=%s) on %s",
            self.config.model_name,
            self.config.revision or "<unpinned>",
            device,
        )
        hf_kwargs: dict = {}
        if self.config.revision is not None:
            hf_kwargs["revision"] = self.config.revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, **hf_kwargs
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name, **hf_kwargs
        ).to(device)
        self.model.eval()
        self._torch = torch

    def build_prompt(self, query: str, passages: Sequence[str]) -> str:
        passages = list(passages)[: self.config.top_k_passages]
        context = " ".join(p.strip() for p in passages if p and p.strip())
        return f"question: {query} context: {context}"

    def generate(self, query: str, passages: Sequence[str]) -> str:
        return self.generate_batch([query], [list(passages)])[0]

    def generate_batch(
        self,
        queries: Sequence[str],
        passages_per_query: Sequence[Sequence[str]],
    ) -> list[str]:
        if len(queries) != len(passages_per_query):
            raise ValueError("queries and passages_per_query must have same length")

        prompts = [self.build_prompt(q, p) for q, p in zip(queries, passages_per_query)]
        outputs: list[str] = []
        bs = max(1, self.config.batch_size)
        for start in range(0, len(prompts), bs):
            batch = prompts[start : start + bs]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_input_length,
                padding=True,
            ).to(self.device)
            with self._torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.config.max_new_tokens,
                )
            outputs.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return outputs
