#!/usr/bin/env python3
"""Adjudication-stage pooling audit of gold completeness (predeclared 23 July 2026).

TREC-style pooling: after annotation and adjudication are complete, compare the
compiled gold against the union of locator-CONFIRMED candidates from every
detection system. A confirmed candidate that matches no gold span is either
out of scope for a reason (the adjudicator excludes it and moves on) or a
quotation both annotators missed — in which case the adjudicator adds it via the
annotation app as a manual span whose notes contain the machine-readable token
``pooling_recovered``. `eval_compare.py` then reports recall both with and
without such spans via the ``provenance:`` strata.

Run AFTER `compile_gold_v2.py`, never before annotation is finished:

    python3 pooling_audit.py                 # audit every article with gold
    python3 pooling_audit.py --threshold 0.6 # span-match threshold (default 0.6)

Output: a per-article review list on stdout and
``evaluation/pooling_review.csv`` for the adjudicator.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from eval_compare import maximum_cardinality_alignment
from project_paths import CANDIDATES_DIR, GOLD_DIR


def load_pool(article_stem):
    """Union of locator-confirmed spans for one article across all systems."""
    # Collect every confirmed candidate from all detection systems for the given article.
    pool = []
    for path in sorted(CANDIDATES_DIR.glob(f"{article_stem}__*.candidates.results.json")):
        tag = path.name.split("__", 1)[1].replace(".candidates.results.json", "")
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("confirmed_pynchon_fiction"):
                pool.append({
                    "quote": row.get("quote", ""),
                    "system": tag,
                    "pid": row.get("pid", ""),
                    "novel": row.get("novel", ""),
                    "section": row.get("section", ""),
                })
    return pool


def dedupe_pool(pool, threshold):
    """Collapse near-identical confirmed spans found by multiple systems."""
    # Merge spans that match above the threshold, combining system tags.
    kept = []
    for row in pool:
        match = maximum_cardinality_alignment([row], kept, threshold)
        if match:
            kept[match[0][1]]["system"] += "+" + row["system"]
        else:
            kept.append(dict(row))
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="span-match token-similarity threshold (same default "
                             "as eval_compare alignment)")
    parser.add_argument("--output", default=str(CANDIDATES_DIR.parent / "pooling_review.csv"))
    args = parser.parse_args()

    gold_files = sorted(GOLD_DIR.glob("*.gold.json"))
    if not gold_files:
        raise SystemExit("no compiled gold found — run compile_gold_v2.py first; "
                         "the pooling audit is an adjudication-stage step")

    review_rows = []
    for gold_path in gold_files:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        stem = gold["article"]
        pool = dedupe_pool(load_pool(stem), args.threshold)
        matches = maximum_cardinality_alignment(pool, gold["quotes"], args.threshold)
        matched_pool = {pi for pi, _gi, _s in matches}
        unmatched = [row for i, row in enumerate(pool) if i not in matched_pool]
        print(f"{stem}: pool {len(pool)} confirmed spans; "
              f"{len(unmatched)} unmatched by gold")
        for row in unmatched:
            print(f"  [{row['system']}] {row['novel']} {row['section']}: "
                  f"{row['quote'][:80]}")
            review_rows.append({"article": stem, **row})

    out = Path(args.output)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "article", "quote", "system", "pid", "novel", "section"])
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"\n{len(review_rows)} spans to review -> {out}")
    print("For each: exclude with a reason, or add via the annotation app with "
          "'pooling_recovered' in the span's notes, then re-run compile_gold_v2.py.")


if __name__ == "__main__":
    main()
