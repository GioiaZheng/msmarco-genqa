"""Regenerate the three prototype notebooks in a tighter, more engineering style.

Run from the project root::

    python scripts/rewrite_notebooks.py

The notebooks are *prototype* artifacts, not the official deliverables. They
all use a sampled subset of the HuggingFace ``ms_marco`` v2.1 *validation*
split so the data download is bounded and the notebook can finish on a CPU
laptop. The official numbers come from the script-based pipeline under
``experiments/``; the notebooks point at those scripts for honest results.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4()),
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": str(uuid.uuid4()),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {path}")


# Shared setup cell: PROJECT_ROOT, FIGURES_DIR, src/ on sys.path. Matches the
# format the previous notebooks used so paths and figures land in the same
# place.
SETUP_CODE = '''import sys
from pathlib import Path


def _find_project_root(markers=(".git", "pyproject.toml", "requirements.txt")):
    p = Path.cwd().resolve()
    for parent in (p, *p.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    return p


PROJECT_ROOT = _find_project_root()
FIGURES_DIR = PROJECT_ROOT / "figures"
SRC_DIR = PROJECT_ROOT / "src"
FIGURES_DIR.mkdir(exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("PROJECT_ROOT:", PROJECT_ROOT)
'''


# --------------------------------------------------------------------------- #
# Week 1 — EDA on MS MARCO QA v2.1 (validation split, sampled).
# --------------------------------------------------------------------------- #

WEEK01_CELLS = [
    md(
        "# Week 1 — EDA on MS MARCO\n\n"
        "Quick look at the MS MARCO QA v2.1 dataset: shape, query/passage "
        "lengths, query types, answer types. Used to size sampling decisions "
        "for later weeks.\n\n"
        "We use the **validation** split (~101k queries, smaller download "
        "than train) and sample within it to keep this notebook fast on a "
        "laptop."
    ),
    md("## Setup"),
    code(SETUP_CODE),
    code(
        "import re\n"
        "from collections import Counter\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from datasets import load_dataset\n"
    ),
    md(
        "## 1. Load data\n\n"
        "Validation split is enough for EDA and avoids the multi-GB train "
        "download."
    ),
    code('ds = load_dataset("ms_marco", "v2.1", split="validation")\nds'),
    code(
        "SAMPLE_SIZE = 5000\n"
        "sample = ds.select(range(min(SAMPLE_SIZE, len(ds))))\n"
        "print(f\"Working with {len(sample)} sampled rows\")\n"
        "sample[0]"
    ),
    md("Each row = one query + up to 10 candidate passages + a human answer + a `query_type` label."),
    md("## 2. Query length"),
    code(
        "queries = [x[\"query\"] for x in sample]\n"
        "query_lens = [len(q.split()) for q in queries]\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.hist(query_lens, bins=range(0, 25))\n"
        "ax.set(xlabel=\"# words\", ylabel=\"# queries\", title=\"Query length\")\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / \"query_length_distribution.png\", dpi=150, bbox_inches=\"tight\")\n"
        "plt.show()\n"
        "print(f\"median={int(np.median(query_lens))}  p95={int(np.percentile(query_lens, 95))}  max={max(query_lens)}\")"
    ),
    md("Most queries are short (5–10 words). Long-tail queries beyond 15 words are rare."),
    md("## 3. Passage length"),
    code(
        "passage_lens = []\n"
        "for x in sample:\n"
        "    for p in x[\"passages\"][\"passage_text\"]:\n"
        "        passage_lens.append(len(p.split()))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.hist(passage_lens, bins=50, range=(0, 200))\n"
        "ax.set(xlabel=\"# words\", ylabel=\"# passages\", title=\"Passage length (clipped at 200)\")\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / \"passage_length_distribution.png\", dpi=150)\n"
        "plt.show()\n"
        "print(f\"median={int(np.median(passage_lens))}  p95={int(np.percentile(passage_lens, 95))}  max={max(passage_lens)}\")"
    ),
    md("Passages are ~10× longer than queries (median ~50 words). Some are truncated to a single sentence."),
    md("## 4. Frequent query terms"),
    code(
        "tokens = [t for q in queries for t in q.lower().split()]\n"
        "Counter(tokens).most_common(15)"
    ),
    md('Wh-words ("what", "how", "is") dominate. Reasonable signal for a QA task.'),
    md("## 5. Query type distribution"),
    code(
        "qt_counts = pd.Series([x[\"query_type\"] for x in sample]).value_counts()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "qt_counts.plot(kind=\"bar\", ax=ax)\n"
        "ax.set(xlabel=\"query_type\", ylabel=\"# queries\", title=\"Query type distribution\")\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / \"query_type_distribution.png\", dpi=150, bbox_inches=\"tight\")\n"
        "plt.show()\n"
        "qt_counts"
    ),
    md("`DESCRIPTION` and `NUMERIC` are most common. `LOCATION` and `PERSON` are smaller buckets."),
    md("## 6. Answer types"),
    code(
        "def classify_answer(a: str) -> str:\n"
        "    a = (a or \"\").strip()\n"
        "    if not a:\n"
        "        return \"empty\"\n"
        "    if a.lower().startswith(\"no answer\"):\n"
        "        return \"no_answer\"\n"
        "    n = len(a.split())\n"
        "    if n == 1:\n"
        "        return \"single_word\"\n"
        "    if n <= 5:\n"
        "        return \"short\"\n"
        "    return \"long\"\n"
        "\n"
        "df = pd.DataFrame({\n"
        "    \"query_type\": [x[\"query_type\"] for x in sample],\n"
        "    \"answer_type\": [classify_answer(x[\"answers\"][0] if x[\"answers\"] else \"\") for x in sample],\n"
        "})\n"
        "ct = pd.crosstab(df[\"query_type\"], df[\"answer_type\"])\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "ct.plot(kind=\"bar\", stacked=True, ax=ax)\n"
        "ax.set(xlabel=\"query_type\", ylabel=\"# queries\", title=\"Answer types per query type\")\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / \"answer_type_by_query_type.png\", dpi=150, bbox_inches=\"tight\")\n"
        "plt.show()\n"
        "ct"
    ),
    md(
        "Many `DESCRIPTION` queries get long answers; most `NUMERIC` queries get short or single-word "
        "answers. A non-trivial fraction is `no_answer`, which we filter out for SFT-style supervision."
    ),
    md("## 7. Relevant passages per query"),
    code(
        "rel_counts = pd.Series([sum(x[\"passages\"][\"is_selected\"]) for x in sample])\n"
        "rel_counts.value_counts().sort_index()"
    ),
    md(
        "Most queries have **0 or 1** marked-relevant passage; very few have ≥2. This matches the "
        "MS MARCO Passage Ranking dev/small qrels structure (binary, sparse)."
    ),
    md(
        "## 8. Limitations\n\n"
        "- We sampled 5k queries from the validation split for speed; full-split numbers may differ slightly.\n"
        "- The HuggingFace `ms_marco` v2.1 dataset is the QA flavour; the *official* passage ranking corpus "
        "(8.8M passages) lives elsewhere and is loaded via `ir_datasets` in the script pipeline.\n"
        "- `query_type` is a noisy heuristic label, not a clean taxonomy.\n\n"
        "## Next\n\n"
        "- Week 2 prototype: BM25 on a sampled closed-set corpus (this notebook’s data).\n"
        "- Official Week 2 baseline (MS MARCO Passage dev/small): see `experiments/run_retrieval.py`."
    ),
]


# --------------------------------------------------------------------------- #
# Week 2 — Prototype BM25 on a sampled corpus.
# --------------------------------------------------------------------------- #

WEEK02_CELLS = [
    md(
        "# Week 2 — Prototype BM25 on sampled MS MARCO\n\n"
        "Lightweight BM25 retrieval on a closed-set corpus flattened from a "
        "small slice of MS MARCO. Eval is MRR@10 over 30 queries drawn from "
        "the same slice, so the number is **optimistic by design** "
        "(corpus and queries share provenance).\n\n"
        "**The honest baseline lives in `experiments/run_retrieval.py`** "
        "(full official corpus, dev/small queries, `bm25s` backend). Most "
        "recent run: MRR@10 = 0.1703, Recall@100 = 0.6212, Recall@1000 = 0.8154 "
        "on the 6,980-query dev/small set."
    ),
    md("## Setup"),
    code(SETUP_CODE),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from datasets import load_dataset\n"
        "from rank_bm25 import BM25Okapi\n"
    ),
    md("## 1. Build a sampled corpus\n\nFlatten passages from 5k validation rows, deduplicate."),
    code(
        'ds = load_dataset("ms_marco", "v2.1", split="validation")\n'
        "subset = ds.select(range(5000))\n"
        "\n"
        "corpus, seen = [], set()\n"
        "for q in subset:\n"
        "    for p in q[\"passages\"][\"passage_text\"]:\n"
        "        p = (p or \"\").strip()\n"
        "        if p and p not in seen:\n"
        "            corpus.append(p)\n"
        "            seen.add(p)\n"
        "print(f\"corpus size: {len(corpus)} unique passages\")"
    ),
    md("## 2. Tokenize and index"),
    code(
        "tokenized = [doc.lower().split() for doc in corpus]\n"
        "bm25 = BM25Okapi(tokenized)"
    ),
    md("## 3. One-query retrieval check"),
    code(
        'def retrieve(q: str, k: int = 5):\n'
        "    scores = bm25.get_scores(q.lower().split())\n"
        "    top = np.argsort(scores)[::-1][:k]\n"
        "    return [{\"score\": float(scores[i]), \"passage\": corpus[i]} for i in top]\n"
        "\n"
        'q = "what was the immediate impact of the success of the manhattan project"\n'
        "for r in retrieve(q, k=3):\n"
        "    print(f'{r[\"score\"]:.2f}  {r[\"passage\"][:120]}...')"
    ),
    md("## 4. MRR@10 on 30 queries"),
    code(
        "def evaluate_mrr(eval_queries, k=10):\n"
        "    rrs, ranks = [], []\n"
        "    for q in eval_queries:\n"
        "        results = retrieve(q[\"query\"], k=k)\n"
        "        relevant = {p.strip() for p, lab in zip(q[\"passages\"][\"passage_text\"], q[\"passages\"][\"is_selected\"]) if lab == 1 and p.strip()}\n"
        "        rank = next((i + 1 for i, r in enumerate(results) if r[\"passage\"] in relevant), 0)\n"
        "        rrs.append(1.0 / rank if rank else 0.0)\n"
        "        ranks.append(rank)\n"
        "    return float(np.mean(rrs)), rrs, ranks\n"
        "\n"
        "eval_queries = subset.select(range(30))\n"
        "mrr10, rrs, ranks = evaluate_mrr(eval_queries, k=10)\n"
        "print(f\"MRR@10 (sampled closed-set, n=30): {mrr10:.4f}\")"
    ),
    md(
        "Closed-set ⇒ optimistic number. The full open-set result on dev/small is in "
        "`outputs/week02_bm25/metrics.json`."
    ),
    md("## 5. Distribution plots"),
    code(
        "fig, ax = plt.subplots(figsize=(6.5, 4))\n"
        "ax.hist(rrs, bins=15)\n"
        "ax.set(xlabel=\"Reciprocal Rank\", ylabel=\"# queries\", title=\"RR distribution (n=30)\")\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / \"rr_distribution.png\", dpi=150)\n"
        "plt.show()"
    ),
    code(
        "hits = [r for r in ranks if r > 0]\n"
        "fig, ax = plt.subplots(figsize=(6.5, 4))\n"
        "ax.hist(hits, bins=range(1, 12), align=\"left\")\n"
        "ax.set(xlabel=\"rank of first relevant passage\", ylabel=\"# queries\", title=\"Hit rank (top-10)\", xticks=range(1, 11))\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / \"hit_rank_distribution.png\", dpi=150)\n"
        "plt.show()"
    ),
    md(
        "Successful retrievals concentrate at ranks 1–3; many queries have no hit at all in the top 10, which is "
        "consistent with BM25’s reliance on lexical overlap.\n\n"
        "## Limitations\n\n"
        "- Closed-set: corpus passages are drawn from the *same* queries we evaluate on, so the relevant passage "
        "is always present. Inflates MRR.\n"
        "- 30-query eval is too small for stable numbers.\n"
        "- `rank_bm25` defaults; no `k1`/`b` tuning.\n\n"
        "## Next\n\n"
        "- Use the official baseline numbers from `experiments/run_retrieval.py` for any comparison.\n"
        "- Add a dense or hybrid retriever in a future week."
    ),
]


# --------------------------------------------------------------------------- #
# Week 3 — Prototype RAG on a toy corpus + 3 hand-written queries.
# --------------------------------------------------------------------------- #

WEEK03_CELLS = [
    md(
        "# Week 3 — Prototype RAG (T5-small + BM25)\n\n"
        "Smallest possible end-to-end check: a 3-passage corpus, T5-small as "
        "the generator, three hand-written queries. ROUGE/BLEU here are "
        "diagnostic only — they verify the pipeline runs, they are **not** "
        "benchmark numbers.\n\n"
        "The honest end-to-end baseline runs in "
        "`experiments/run_generation_baseline.py` against the real Week 2 "
        "BM25 retrieval results."
    ),
    md("## Setup"),
    code(SETUP_CODE),
    code(
        "import torch\n"
        "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\n"
        "import evaluate\n"
        "\n"
        "from bm25_retriever import BM25Retriever\n"
        "\n"
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
        "print(\"device:\", device)"
    ),
    md("## 1. BM25 sanity check on a 3-passage toy corpus"),
    code(
        "corpus = [\n"
        "    \"Canberra is the capital city of Australia.\",\n"
        "    \"Sydney is the largest city in Australia.\",\n"
        "    \"Australia is a country in Oceania.\",\n"
        "]\n"
        "retriever = BM25Retriever(corpus)\n"
        "for r in retriever.retrieve(\"what is the capital of australia\", k=3):\n"
        "    print(f'{r[\"score\"]:.3f}  {r[\"passage\"]}')"
    ),
    md("Canberra ranked first — retriever is wired correctly."),
    md("## 2. Load T5-small"),
    code(
        'MODEL_NAME = "t5-small"\n'
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
        "model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)\n"
        "model.eval()\n"
        "f\"loaded {MODEL_NAME} on {device}\""
    ),
    md("## 3. RAG generate function"),
    code(
        "def build_prompt(query: str, results) -> str:\n"
        "    context = \" \".join(r[\"passage\"] for r in results)\n"
        "    return f\"question: {query} context: {context}\"\n"
        "\n"
        "def generate_answer(query: str, k: int = 3, max_new_tokens: int = 64) -> str:\n"
        "    results = retriever.retrieve(query, k=k)\n"
        "    inp = tokenizer(build_prompt(query, results), return_tensors=\"pt\", truncation=True, max_length=512).to(device)\n"
        "    with torch.no_grad():\n"
        "        out = model.generate(**inp, max_new_tokens=max_new_tokens)\n"
        "    return tokenizer.decode(out[0], skip_special_tokens=True)\n"
        "\n"
        "generate_answer(\"what is the capital of australia\")"
    ),
    md(
        "Output contains the right fact (Canberra) but is verbose — pretrained T5-small without "
        "fine-tuning tends to copy/paraphrase context rather than emit a short answer."
    ),
    md("## 4. Eval on 3 hand-written queries"),
    code(
        "eval_queries = [\n"
        "    \"what is the capital of australia\",\n"
        "    \"where is the eiffel tower located\",\n"
        "    \"who invented the telephone\",\n"
        "]\n"
        "references = [\"Canberra\", \"Paris\", \"Alexander Graham Bell\"]\n"
        "predictions = [generate_answer(q) for q in eval_queries]\n"
        "for q, r, p in zip(eval_queries, references, predictions):\n"
        "    print(f\"Q: {q}\\nREF: {r}\\nPRED: {p}\\n\")"
    ),
    md("## 5. ROUGE-L / BLEU"),
    code(
        "rouge = evaluate.load(\"rouge\")\n"
        "bleu = evaluate.load(\"bleu\")\n"
        "rouge_l = rouge.compute(predictions=predictions, references=references)[\"rougeL\"]\n"
        "bleu_s = bleu.compute(predictions=predictions, references=[[r] for r in references])[\"bleu\"]\n"
        "print(f\"ROUGE-L: {rouge_l:.4f}\")\n"
        "print(f\"BLEU:    {bleu_s:.4f}\")"
    ),
    md(
        "Tiny 3-query eval, no fine-tuning, retrievals from a 3-passage corpus that doesn’t cover query 2 "
        "or 3 — these scores are diagnostic only.\n\n"
        "## Limitations\n\n"
        "- 3-passage corpus + 3 hand-written queries is a smoke test, not an evaluation.\n"
        "- T5-small is not fine-tuned on MS MARCO QA.\n"
        "- Surface-form metrics (ROUGE / BLEU) heavily penalise paraphrases.\n\n"
        "## Next — official RAG baseline\n\n"
        "Run the script-based pipeline against the real Week 2 BM25 retrievals:\n\n"
        "```bash\n"
        "python experiments/run_generation_baseline.py\n"
        "python -m src.reporting.build_report --week week03\n"
        "```\n"
    ),
]


# --------------------------------------------------------------------------- #
# Week 4 — Prototype dense retrieval on a 6-passage toy corpus.
# --------------------------------------------------------------------------- #

# Distinct setup cell for W4: the env vars must be set BEFORE faiss/torch are
# imported, otherwise the macOS libomp duplicate-library issue makes the
# kernel hang or crash. See ``experiments/run_dense_retrieval.py`` for the
# same workaround at script level.
W4_SETUP_CODE = '''import sys
from pathlib import Path


def _find_project_root(markers=(".git", "pyproject.toml", "requirements.txt")):
    p = Path.cwd().resolve()
    for parent in (p, *p.parents):
        if any((parent / m).exists() for m in markers):
            return parent
    return p


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# macOS libomp workaround: faiss-cpu and torch each ship libomp.dylib.
# Loading both in the same process aborts/segfaults unless we tell the
# runtime it's fine. Must be set before the first faiss/torch import.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

print("PROJECT_ROOT:", PROJECT_ROOT)
'''


WEEK04_CELLS = [
    md(
        "# Week 4 — Prototype dense retrieval (SBERT + FAISS)\n\n"
        "Smallest possible end-to-end check: a 6-passage toy corpus, "
        "`all-MiniLM-L6-v2` as the encoder, three hand-written queries, and "
        "a side-by-side BM25 comparison on the *same* tiny corpus. Numbers "
        "here are diagnostic only — they verify the pipeline runs, they are "
        "**not** benchmark numbers.\n\n"
        "The honest baseline lives in `experiments/run_dense_retrieval.py` "
        "(50k qrels-anchored sample of MS MARCO Passage); see "
        "`outputs/week04_dense/metrics.json` and "
        "`reports/generated/week04_dense.md` for the real numbers."
    ),
    md("## Setup"),
    code(W4_SETUP_CODE),
    code(
        "from src.retrieval.dense import DenseRetriever\n"
        "from src.retrieval.bm25 import BM25Retriever"
    ),
    md(
        "## 1. Toy corpus\n\n"
        "Six passages, deliberately chosen so a sparse retriever can win on "
        "lexical overlap and a dense retriever can win on paraphrase / "
        "synonym overlap."
    ),
    code(
        "corpus = [\n"
        "    \"Canberra is the capital city of Australia.\",                              # d0\n"
        "    \"The Eiffel Tower is a wrought-iron lattice tower located in Paris.\",      # d1\n"
        "    \"William Shakespeare wrote the tragedy Hamlet in the early 17th century.\", # d2\n"
        "    \"Sydney is the largest city in Australia but it is not the capital.\",      # d3\n"
        "    \"The Pacific Ocean is the largest ocean on Earth.\",                        # d4\n"
        "    \"Photosynthesis converts sunlight into chemical energy in green plants.\",  # d5\n"
        "]\n"
        "doc_ids = [f\"d{i}\" for i in range(len(corpus))]"
    ),
    md("## 2. Build the dense (SBERT + FAISS) and BM25 indexes on the same corpus"),
    code(
        "dense = DenseRetriever(model_name=\"sentence-transformers/all-MiniLM-L6-v2\", device=\"cpu\")\n"
        "dense.build(corpus, doc_ids)\n"
        "\n"
        "bm25 = BM25Retriever(corpus_texts=corpus, doc_ids=doc_ids, k1=1.5, b=0.75)\n"
        "bm25.build()\n"
        "\"both retrievers ready\""
    ),
    md(
        "## 3. Three queries — lexical-easy, paraphrase, semantic\n\n"
        "We expect both retrievers to pick the obvious answer on lexical "
        "queries. The interesting case is Q3, where the query word "
        "*\"playwright\"* doesn't appear in any passage but is semantically "
        "close to *\"wrote a tragedy\"* in d2."
    ),
    code(
        "queries = [\n"
        "    \"what is the capital of australia\",\n"
        "    \"where is the eiffel tower\",\n"
        "    \"which playwright wrote hamlet\",\n"
        "]\n"
        "\n"
        "def show(label, scores, ids):\n"
        "    print(f\"  {label:6s} top-3: \" + \", \".join(f\"{d}({s:.3f})\" for d, s in zip(ids[:3], scores[:3])))\n"
        "\n"
        "for q in queries:\n"
        "    print(f\"Q: {q}\")\n"
        "    d_scores, d_ids = dense.retrieve(q, k=3)\n"
        "    b_scores, b_ids = bm25.retrieve(q, k=3)\n"
        "    show(\"dense\",  d_scores, d_ids)\n"
        "    show(\"bm25\",   b_scores, b_ids)\n"
        "    print()"
    ),
    md(
        "## 4. Where dense vs BM25 actually disagree\n\n"
        "On the tiny corpus both retrievers get the right top-1 because "
        "Q3 still contains the literal word *\"hamlet\"*. To see the gap, "
        "drop the keyword and force a purely semantic query:"
    ),
    code(
        "q = \"a famous english tragedy from the renaissance\"\n"
        "scores_d, ids_d = dense.retrieve(q, k=3)\n"
        "scores_b, ids_b = bm25.retrieve(q, k=3)\n"
        "print(\"dense:\", list(zip(ids_d, [f'{s:.3f}' for s in scores_d])))\n"
        "print(\"bm25: \", list(zip(ids_b, [f'{s:.3f}' for s in scores_b])))"
    ),
    md(
        "BM25 has no signal (no content words overlap with the corpus), so "
        "it returns near-zero scores in arbitrary order. The dense retriever "
        "still ranks d2 (Shakespeare/Hamlet) at the top because the SBERT "
        "embedding maps \"english tragedy\" close to \"Shakespeare's tragedy "
        "Hamlet\". This is the qualitative reason to add dense retrieval in "
        "the first place.\n\n"
        "## Limitations\n\n"
        "- 6-passage corpus is a smoke test, not an evaluation.\n"
        "- `all-MiniLM-L6-v2` is generic-purpose, not MS-MARCO-tuned.\n"
        "- FAISS `IndexFlatIP` on 6 vectors is microseconds; latency is "
        "only meaningful at the 50k+ scale.\n\n"
        "## Next — official dense baseline\n\n"
        "Run the script-based pipeline on the 50k qrels-anchored sample:\n\n"
        "```bash\n"
        "python experiments/run_dense_retrieval.py --sample-size 50000 --rebuild-index\n"
        "python -m src.reporting.build_report --week week04\n"
        "```\n"
    ),
]


def main() -> None:
    write_notebook(NOTEBOOKS_DIR / "week01_eda.ipynb", WEEK01_CELLS)
    write_notebook(NOTEBOOKS_DIR / "week02_retrieval.ipynb", WEEK02_CELLS)
    write_notebook(NOTEBOOKS_DIR / "week03_generation.ipynb", WEEK03_CELLS)
    write_notebook(NOTEBOOKS_DIR / "week04_dense.ipynb", WEEK04_CELLS)


if __name__ == "__main__":
    main()
