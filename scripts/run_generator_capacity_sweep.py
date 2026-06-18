"""Generator capacity sweep — T5-base vs T5-small on the same RAG-stage prompts.

The headline question (per the NLI-grounding analysis finding that
T5-small's NLI grounding Δ is the *only* metric whose Δ reverses sign vs
the RAG-stage surface-form story): **does T5-base — same prompt format,
more capacity — flip the NLI Δ back to positive?**

To answer it we need T5-base scored on *both* arms (BM25 and
reranked) so that ΔNLI = NLI(T5-base, reranked) − NLI(T5-base, BM25)
can be compared head-to-head with the existing T5-small Δ.

This driver chains:

1. ``experiments/run_generation_baseline.py --model-name t5-base`` on
   BM25 (first-stage run.tsv) and on reranked (cross-encoder run.tsv);
   identical mutual
   ``--restrict-to-run`` so the eval qid sets match.
2. ``scripts/bootstrap_generation_comparison.py`` over the T5-base
   predictions for Token-F1 / ROUGE-L / BLEU / EM Δ + paired bootstrap.
3. ``scripts/bertscore_paired_eval.py --n-pairs 0`` (full dev) for the
   semantic BERTScore Δ.
4. ``scripts/grounding_audit.py --nli-n-pairs 3000`` for lex / 3-gram /
   NLI grounding Δ.

Output: ``outputs/generator_capacity_generator_comparison/`` with a side-by-side
table of every (generator, arm) cell and the cross-generator
Δ-of-Δ on the NLI metric (the load-bearing comparison).

CPU cost: ~1 h per generation run × 2 = ~2 h, plus ~30 min re-scoring
(NLI on 3 000-pair subsample + BERTScore on full 6 980). Default
total ~2.5 h on a 6-core MacBook.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("run_generator_capacity_sweep")


# Existing T5-small numbers — paths to the on-disk outputs whose
# headline metrics we pull from to populate the comparison table.
T5_SMALL_CELLS = {
    "bm25": {
        "predictions": "outputs/generation_bm25_full",
        "label": "T5-small × BM25",
    },
    "rerank": {
        "predictions": "outputs/generation_reranked_full",
        "label": "T5-small × Reranked",
    },
}
T5_SMALL_BOOTSTRAP = "outputs/generation_bootstrap_full/bootstrap_ci.json"
T5_SMALL_BERTSCORE = "outputs/bertscore_proxy/bertscore_proxy_ci.json"
T5_SMALL_GROUNDING = "outputs/grounding/summary.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", type=str, default="t5-base")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--num-eval-queries",
        type=int,
        default=9999,
        help=(
            "Number of queries per generation arm. Default 9999 covers "
            "full dev/small (mutually restricted on the BM25 and rerank "
            "qid sets). Pass a small value (e.g. 50) for a smoke test "
            "that exercises the full pipeline without paying full cost."
        ),
    )
    p.add_argument(
        "--nli-n-pairs", type=int, default=3000,
        help="Paired-qid subsample for NLI grounding (matches the T5-small NLI-grounding run).",
    )
    p.add_argument(
        "--bertscore-n-pairs", type=int, default=0,
        help="0 = full dev; matches the BERTScore convention for proxy CI.",
    )
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--skip-grounding", action="store_true")
    p.add_argument("--skip-bertscore", action="store_true")
    p.add_argument("--skip-bootstrap", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/generator_capacity_generator_comparison",
    )
    return p.parse_args()


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def run_subproc(cmd: list[str], label: str) -> None:
    logger.info("[%s] %s", label, " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    logger.info("[%s] done in %.1f min", label, (time.time() - t0) / 60)


def step_generate_t5_base(args: argparse.Namespace) -> dict[str, str]:
    """Generate T5-base predictions on BM25 and reranked top-3.

    Returns mapping arm → output dir path (relative to PROJECT_ROOT).
    Mutual ``--restrict-to-run`` so the two arms cover the same qids.
    """
    safe = model_safe_name(args.model_name)
    out_bm25 = f"outputs/generator_capacity_generation_{safe}_bm25"
    out_rerank = f"outputs/generator_capacity_generation_{safe}_reranked"
    if args.skip_generation:
        return {"bm25": out_bm25, "rerank": out_rerank}

    common = [
        sys.executable,
        str(PROJECT_ROOT / "experiments/run_generation_baseline.py"),
        "--model-name", args.model_name,
        "--max-new-tokens", str(args.max_new_tokens),
        "--num-eval-queries", str(args.num_eval_queries),
    ]
    run_subproc(
        common + [
            "--input-run", "outputs/bm25_baseline/run.tsv",
            "--output-dir", out_bm25,
            "--retrieval-source", "bm25",
            "--restrict-to-run", "outputs/cross_encoder_rerank_full/run.tsv",
        ],
        f"generate {args.model_name} × BM25",
    )
    run_subproc(
        common + [
            "--input-run", "outputs/cross_encoder_rerank_full/run.tsv",
            "--output-dir", out_rerank,
            "--retrieval-source", "reranked",
            "--restrict-to-run", "outputs/bm25_baseline/run.tsv",
        ],
        f"generate {args.model_name} × Reranked",
    )
    return {"bm25": out_bm25, "rerank": out_rerank}


def step_bootstrap(args: argparse.Namespace, gen_dirs: dict[str, str]) -> str:
    out_dir = f"outputs/generator_capacity_bootstrap_{model_safe_name(args.model_name)}"
    if args.skip_bootstrap:
        return out_dir
    run_subproc(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/bootstrap_generation_comparison.py"),
            "--bm25-dir", gen_dirs["bm25"],
            "--reranked-dir", gen_dirs["rerank"],
            "--output-dir", out_dir,
        ],
        f"bootstrap surface metrics ({args.model_name})",
    )
    return out_dir


def step_bertscore(args: argparse.Namespace, gen_dirs: dict[str, str]) -> str:
    out_dir = f"outputs/generator_capacity_bertscore_{model_safe_name(args.model_name)}"
    if args.skip_bertscore:
        return out_dir
    run_subproc(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/bertscore_paired_eval.py"),
            "--bm25-dir", gen_dirs["bm25"],
            "--reranked-dir", gen_dirs["rerank"],
            "--output-dir", out_dir,
            "--n-pairs", str(args.bertscore_n_pairs),
        ],
        f"BERTScore ({args.model_name})",
    )
    return out_dir


def step_grounding(args: argparse.Namespace, gen_dirs: dict[str, str]) -> str:
    out_dir = f"outputs/generator_capacity_grounding_{model_safe_name(args.model_name)}"
    if args.skip_grounding:
        return out_dir
    run_subproc(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/grounding_audit.py"),
            "--bm25-dir", gen_dirs["bm25"],
            "--reranked-dir", gen_dirs["rerank"],
            "--output-dir", out_dir,
            "--nli-n-pairs", str(args.nli_n_pairs),
        ],
        f"grounding audit ({args.model_name})",
    )
    return out_dir


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def aggregate(
    args: argparse.Namespace,
    gen_dirs: dict[str, str],
    bootstrap_dir: str,
    bertscore_dir: str,
    grounding_dir: str,
) -> dict[str, Any]:
    """Build the side-by-side comparison row dict."""
    safe = model_safe_name(args.model_name)

    # T5-base headline numbers
    t5b_bootstrap = load_json(PROJECT_ROOT / bootstrap_dir / "bootstrap_ci.json")
    t5b_bertscore = load_json(PROJECT_ROOT / bertscore_dir / "bertscore_proxy_ci.json")
    t5b_grounding = load_json(PROJECT_ROOT / grounding_dir / "summary.json")

    # T5-small headline numbers (already on disk)
    t5s_bootstrap = load_json(PROJECT_ROOT / T5_SMALL_BOOTSTRAP)
    t5s_bertscore = load_json(PROJECT_ROOT / T5_SMALL_BERTSCORE)
    t5s_grounding = load_json(PROJECT_ROOT / T5_SMALL_GROUNDING)

    def cell_from_bootstrap(boot: dict[str, Any]) -> dict[str, Any]:
        # bootstrap_generation_comparison.py emits {token_f1, rouge_l, bleu,
        # exact_match}: each with mean_a/mean_b/delta + CI.
        out: dict[str, Any] = {}
        for k in ("token_f1", "rouge_l", "bleu", "exact_match"):
            block = boot.get(k) or {}
            if not block:
                continue
            out[k] = {
                "bm25": block.get("mean_a"),
                "rerank": block.get("mean_b"),
                "delta": block.get("mean_delta"),
                "ci_low": block.get("ci_low"),
                "ci_high": block.get("ci_high"),
                "p_two_sided": block.get("p_two_sided"),
            }
        return out

    def cell_from_bertscore(bs: dict[str, Any]) -> dict[str, Any]:
        # bertscore_paired_eval.py emits a top-level {bertscore:
        # {mean_bm25, mean_rerank}, bootstrap: {delta_mean, ci_low, ci_high,
        # p_two_sided}}.
        bs_block = bs.get("bertscore") or {}
        boot = bs.get("bootstrap") or {}
        return {
            "bertscore_f1": {
                "bm25": bs_block.get("mean_bm25"),
                "rerank": bs_block.get("mean_rerank"),
                "delta": boot.get("delta_mean") or bs_block.get("delta_mean"),
                "ci_low": boot.get("ci_low") or bs_block.get("ci_low"),
                "ci_high": boot.get("ci_high") or bs_block.get("ci_high"),
                "p_two_sided": boot.get("p_two_sided") or bs_block.get("p_two_sided"),
            }
        }

    def cell_from_grounding(g: dict[str, Any]) -> dict[str, Any]:
        # grounding_audit.py emits metrics: {lexical_..., ngram_grounding,
        # nli_entailment_grounding} each with mean_bm25/mean_rerank/delta_mean/CI.
        metrics = g.get("metrics") or {}
        out: dict[str, Any] = {}
        for key, label in (
            ("lexical_content_token_grounding", "lex"),
            ("ngram_grounding", "ngram"),
            ("nli_entailment_grounding", "nli"),
        ):
            block = metrics.get(key) or {}
            if not block:
                continue
            out[label] = {
                "bm25": block.get("mean_bm25"),
                "rerank": block.get("mean_rerank"),
                "delta": block.get("delta_mean"),
                "ci_low": block.get("ci_low"),
                "ci_high": block.get("ci_high"),
                "p_two_sided": block.get("p_two_sided"),
            }
        return out

    payload: dict[str, Any] = {
        "task": "generator_capacity_sweep",
        "t5_small": {
            **cell_from_bootstrap(t5s_bootstrap),
            **cell_from_bertscore(t5s_bertscore),
            **cell_from_grounding(t5s_grounding),
        },
        "t5_base": {
            **cell_from_bootstrap(t5b_bootstrap),
            **cell_from_bertscore(t5b_bertscore),
            **cell_from_grounding(t5b_grounding),
        },
        "gen_dirs": gen_dirs,
        "model_safe": safe,
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    METRIC_ORDER = [
        ("token_f1", "Token-F1"),
        ("rouge_l", "ROUGE-L"),
        ("bleu", "BLEU"),
        ("exact_match", "Exact Match"),
        ("bertscore_f1", "BERTScore-F1 (DistilBERT)"),
        ("lex", "Lexical grounding"),
        ("ngram", "3-gram grounding"),
        ("nli", "NLI-entailment grounding"),
    ]
    lines: list[str] = []
    lines.append("# Generator-capacity sweep — T5-base vs T5-small generator comparison")
    lines.append("")
    lines.append(
        "Same prompt format, same retrieval inputs, only the generator "
        "changes. Each cell reports `mean_bm25 / mean_rerank` for that "
        "(generator, metric) pair; the Δ column is rerank − BM25 with "
        "the existing paired-bootstrap CI."
    )
    lines.append("")
    lines.append("## 1. Per-metric BM25-vs-rerank Δ by generator")
    lines.append("")
    lines.append("| metric | T5-small BM25 / rerank | T5-small Δ (95 % CI) | T5-base BM25 / rerank | T5-base Δ (95 % CI) | sign-flip? |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in METRIC_ORDER:
        t5s = payload["t5_small"].get(key, {})
        t5b = payload["t5_base"].get(key, {})
        if not t5s and not t5b:
            continue

        def fmt(cell: dict[str, Any]) -> tuple[str, str]:
            if not cell:
                return ("—", "—")
            bm = cell.get("bm25")
            rr = cell.get("rerank")
            dl = cell.get("delta")
            ci_l = cell.get("ci_low")
            ci_h = cell.get("ci_high")
            bm_rr = (
                f"{bm:.4f} / {rr:.4f}" if bm is not None and rr is not None else "—"
            )
            if dl is None:
                return (bm_rr, "—")
            ci_str = (
                f"[{ci_l:+.4f}, {ci_h:+.4f}]"
                if ci_l is not None and ci_h is not None
                else ""
            )
            return (bm_rr, f"**{dl:+.4f}** {ci_str}")

        t5s_lvl, t5s_delta = fmt(t5s)
        t5b_lvl, t5b_delta = fmt(t5b)
        sign_t5s = (t5s.get("delta") or 0.0)
        sign_t5b = (t5b.get("delta") or 0.0)
        flipped = "**yes**" if (sign_t5s * sign_t5b < 0) else "no"
        lines.append(
            f"| {label} | {t5s_lvl} | {t5s_delta} | {t5b_lvl} | {t5b_delta} | {flipped} |"
        )
    lines.append("")
    lines.append("## 2. Headline read")
    lines.append("")
    nli_s = payload["t5_small"].get("nli", {}) or {}
    nli_b = payload["t5_base"].get("nli", {}) or {}
    if nli_s.get("delta") is not None and nli_b.get("delta") is not None:
        s_delta = nli_s["delta"]
        b_delta = nli_b["delta"]
        if s_delta * b_delta < 0:
            verdict = (
                f"T5-base **flips the NLI grounding Δ** from "
                f"{s_delta:+.4f} (T5-small) to {b_delta:+.4f} (T5-base). "
                "Capacity reverses the NLI-grounding reverse-sign finding: the "
                "earlier negative Δ is a T5-small-specific artefact of "
                "fragmentary / mid-word-cut outputs the sentence-level "
                "NLI cross-encoder cannot entail."
            )
        else:
            verdict = (
                f"T5-base does **not** flip the NLI grounding Δ "
                f"(T5-small {s_delta:+.4f}, T5-base {b_delta:+.4f}, "
                "same sign). The reverse-sign behaviour from the NLI-grounding analysis is "
                "not generator-capacity-driven; the prompt format itself "
                "(`question: ... context: ...`) is the likely culprit."
            )
        lines.append(verdict)
        lines.append("")
    lines.append("")
    lines.append("## 3. Caveats")
    lines.append("")
    lines.append(
        "- BERTScore here is the generation-analysis proxy DistilBERT setup "
        "(`distilbert-base-uncased`, `rescale_with_baseline=True`), not "
        "the canonical `roberta-large` BERTScore."
    )
    lines.append(
        "- NLI grounding uses `cross-encoder/nli-deberta-v3-small` on a "
        "3 000-paired-qid subsample (seed = 42), the same convention as "
        "the T5-small NLI-grounding audit."
    )
    lines.append(
        "- T5-base is **not fine-tuned** for MS MARCO QA either; the "
        "delta vs T5-small is therefore a pure capacity test on the "
        "same `question: ... context: ...` prompt."
    )
    lines.append(
        "- flan-t5-base and Llama-2-7b-chat are deferred (project ddl)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    gen_dirs = step_generate_t5_base(args)
    bootstrap_dir = step_bootstrap(args, gen_dirs)
    bertscore_dir = step_bertscore(args, gen_dirs)
    grounding_dir = step_grounding(args, gen_dirs)

    payload = aggregate(args, gen_dirs, bootstrap_dir, bertscore_dir, grounding_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    (args.output_dir / "summary.md").write_text(render_markdown(payload))
    logger.info("Wrote %s/summary.{json,md}",
                args.output_dir.relative_to(PROJECT_ROOT))

    # ---- console summary ----
    print()
    print("=== Generator-capacity sweep — T5-small vs T5-base ===")
    print(f"  {'metric':28s}  {'T5-small Δ':>11s}  {'T5-base Δ':>11s}  {'flip?':>6s}")
    METRIC_ORDER = [
        ("token_f1", "Token-F1"),
        ("rouge_l", "ROUGE-L"),
        ("bertscore_f1", "BERTScore-F1"),
        ("lex", "lex grounding"),
        ("ngram", "ngram grounding"),
        ("nli", "NLI grounding"),
    ]
    for key, label in METRIC_ORDER:
        t5s = payload["t5_small"].get(key, {})
        t5b = payload["t5_base"].get(key, {})
        ds = t5s.get("delta")
        db = t5b.get("delta")
        if ds is None or db is None:
            continue
        flipped = "yes" if (ds * db < 0) else "no"
        print(
            f"  {label:28s}  {ds:>+11.4f}  {db:>+11.4f}  {flipped:>6s}"
        )


if __name__ == "__main__":
    main()
