"""R5 metric-robustness: cross-backbone verdict for the NLI factorial.

Reads every ``<input-dir>/*/summary.json`` written by
``scripts/run_nli_factorial.py`` and collapses them into one verdict: for
each ``(score formula, threshold)`` cell, does the rerank effect reverse
sign (grounding drops, CI upper bound below zero) across *all* backbones?

The headline is the baseline cell (``entailment`` formula, threshold-free
mean — the W7-A regime). ``baseline_robust_reversal`` true across >=3
backbones is the Axis A paper's go condition: the "rerank lowers NLI
grounding" finding survives the metric-choice attack surface.

Writes ``verdict.json`` (machine-readable) and ``verdict.md`` (a table) into
the input dir, and prints the table. The numbers depend on the run, so both
files are gitignored artefacts; only the finalised table enters git later as
prose in the experiments doc.

Usage::

    python scripts/aggregate_nli_factorial.py --input-dir outputs/r5_nli_factorial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from msmarco_genqa.evaluation.nli_factorial import aggregate_backbones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/r5_nli_factorial",
        help="Dir holding one subdir per backbone x direction run.",
    )
    return parser.parse_args()


def load_summaries(input_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(input_dir.glob("*/summary.json")):
        with open(summary_path) as f:
            summaries.append(json.load(f))
    return summaries


def _fmt_threshold(t: Any) -> str:
    return "mean" if t is None else f"@{t}"


def render_markdown(agg: dict[str, Any]) -> str:
    backbones = agg["backbones"]
    lines = [
        "# R5 NLI grounding factorial — cross-backbone verdict",
        "",
        f"Backbones ({agg['n_backbones']}): {', '.join(backbones)}",
        "",
        f"**Baseline cell (entailment / mean) robust reversal across all "
        f"backbones: {agg['headline']['baseline_robust_reversal']}** "
        f"({agg['headline']['baseline_n_reverse']}/"
        f"{agg['headline']['baseline_n_backbones']} reverse)",
        "",
        f"Robust-reversal cells: {agg['n_robust_reversal_cells']}/{agg['n_cells']}",
        "",
    ]
    header = ["formula", "thr"] + backbones + ["robust↺"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for cell in agg["cells"]:
        per = {p["backbone"]: p for p in cell["per_backbone"]}
        row = [cell["formula"], _fmt_threshold(cell["threshold"])]
        for b in backbones:
            p = per.get(b)
            if p is None:
                row.append("—")
            else:
                row.append(
                    f"{p['delta']:+.3f} [{p['ci_low']:+.3f},{p['ci_high']:+.3f}]"
                )
        row.append("yes" if cell["robust_reversal"] else "no")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summaries = load_summaries(args.input_dir)
    if not summaries:
        raise SystemExit(
            f"No */summary.json under {args.input_dir}. Run "
            "scripts/run_nli_factorial.py first."
        )

    agg = aggregate_backbones(summaries)

    verdict_json = args.input_dir / "verdict.json"
    with open(verdict_json, "w") as f:
        json.dump(agg, f, indent=2)
    md = render_markdown(agg)
    verdict_md = args.input_dir / "verdict.md"
    with open(verdict_md, "w") as f:
        f.write(md)

    print(md)
    print(f"\nWrote {verdict_json}\nWrote {verdict_md}")


if __name__ == "__main__":
    main()
