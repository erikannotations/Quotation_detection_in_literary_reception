#!/usr/bin/env python3
"""Render the CHR 2027 paper's Table 3: detection precision by span length.


Two views exist, and the paper must say which one it prints:

* ``raw`` (the paper's Table 3): every span the detector emitted is scored,
  including spans below the five-word floor (``quotation_spans.MIN_QUOTE_WORDS``)
  that the headline metrics in Section 4 exclude as out of scope. This is the
  honest view for a table about detector failure modes: a system that emits
  sub-floor junk (Quid: 47 of 1,059; the open-weight LLM: 45 of 297) should
  show that in its short-span column. Sub-floor spans land in the shortest
  band, so that band is labelled "<= 7 words", not "5-7 words".
* ``scored``: only spans meeting the floor, matching Section 4's accounting
  (Quid 1,012 of 1,059). Shown so the sensitivity is visible, per the
  reasoning documented in ``eval_compare.discover``.

Band edges (<=7 / 8-12 / 13+) follow the paper; lengths use the frozen
``quotation_spans.word_count`` for candidates and gold alike.

Usage:
    python3 precision_by_length.py [--view raw|scored|both] [--format md|tex]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import eval_compare as ec
from quotation_spans import under_minimum, word_count

CANDIDATES_DIR = SCRIPT_DIR / "evaluation" / "candidates"
SAMPLE_MANIFEST = SCRIPT_DIR / "evaluation" / "sample_manifest.csv"
MATCH_THRESHOLD = 0.6

# Paper label -> candidate-file tag. Same systems, same order, as Table 3.
SYSTEMS = [
    ("Quotation-mark heuristic", "heuristic_marked"),
    ("Quid 2.6.7", "quid_2.6.7_hp128_s0p85_m5_c100000"),
    ("text-matcher, evaluation setting", "textmatcher_t2c3n2m3_nostops_0.1.7"),
    ("Frontier LLM", "anthropic_claude-fable-5"),
    ("Open-weight LLM", "ollama_gemma4-26b-mlx"),
]

BAND_LABELS = ["≤7 words", "8–12 words", "13+ words"]


def band_of(words):
    return 0 if words <= 7 else (1 if words <= 12 else 2)


def held_out_stems():
    # Load held-out (test-split) article stems from the sample manifest.
    stems = []
    with open(SAMPLE_MANIFEST, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "development":
                stems.append(Path(row["text_path"]).stem)
    return stems


def score_bands(tag, stems, view):
    """Return ([n, tp] per band, raw candidate count, sub-floor count)."""
    bands = [[0, 0], [0, 0], [0, 0]]
    raw_total = sub_floor = 0
    for stem in stems:
        path = CANDIDATES_DIR / f"{stem}__{tag}.candidates.json"
        candidates = json.loads(path.read_text(encoding="utf-8"))
        raw_total += len(candidates)
        sub_floor += sum(
            1 for row in candidates if under_minimum(row.get("quote", ""))
        )
        if view == "scored":
            candidates = [
                row for row in candidates if not under_minimum(row.get("quote", ""))
            ]
        gold = ec.load_gold(stem)
        pairs = ec.maximum_cardinality_alignment(candidates, gold, MATCH_THRESHOLD)
        matched = {pi for pi, _gi, _score in pairs}
        for pi, row in enumerate(candidates):
            index = band_of(word_count(row.get("quote", "")))
            bands[index][0] += 1
            bands[index][1] += int(pi in matched)
    return bands, raw_total, sub_floor


def gold_band_counts(stems):
    counts = [0, 0, 0]
    for stem in stems:
        for item in ec.load_gold(stem):
            counts[band_of(word_count(item.get("quote", "")))] += 1
    return counts


def render(view, stems, fmt):
    print(f"# view={view} (match_threshold={MATCH_THRESHOLD})")
    print(f"# gold spans per band: {gold_band_counts(stems)}")
    for label, tag in SYSTEMS:
        bands, raw_total, sub_floor = score_bands(tag, stems, view)
        cells = [
            f"{tp / n:.3f}" if n else "—" for n, tp in bands
        ]
        if fmt == "tex":
            print(f"    {label} & " + " & ".join(cells) + r" \\")
        else:
            print(f"| {label} | " + " | ".join(cells) + " |")
        ns = ", ".join(str(n) for n, _tp in bands)
        print(
            f"#   n per band: {ns}; raw candidates: {raw_total}; "
            f"below five-word floor: {sub_floor}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", choices=["raw", "scored", "both"], default="both")
    parser.add_argument("--format", choices=["md", "tex"], default="md")
    args = parser.parse_args()

    stems = held_out_stems()
    print(f"# held-out articles: {len(stems)}")
    views = ["raw", "scored"] if args.view == "both" else [args.view]
    for view in views:
        print()
        render(view, stems, args.format)


if __name__ == "__main__":
    main()
