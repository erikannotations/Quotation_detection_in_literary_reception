#!/usr/bin/env python3
"""Build every declared hybrid variant from the single-system candidate files.

This script declares every variant as data and rebuilds all of them, so the family can
be regenerated, extended, and checked. Run `--verify` to confirm it reproduces the
stored candidates for the variants that already exist.

Operators, matching the definitions recorded in `models_frozen.json`:

  union         merge the members, collapsing a span only when another CONTAINS it,
                without transitive closure (`combine_candidates.merge_candidate_rows`)
  gate          keep only this member's spans that overlap a MARKED span of the
                article — deterministic, uses no gold, and follows from the frozen
                target being marked quotation
  intersect     keep each of this member's spans that overlaps some span of the named
                other system, retaining THIS member's boundary

`gate` and `intersect` are span filters applied to one member before the union; a
variant is then the union of its filtered members. A one-member variant is that filter
alone.
"""
import argparse
import json
from pathlib import Path

from combine_candidates import merge_candidate_rows, valid_span
from project_paths import CANDIDATES_DIR
from quotation_spans import article_body, extract_marked_spans

# Single-system tags. Kept here so a variant declaration reads as algebra rather than
# as filename fragments.
M = "heuristic_marked"
Q = "quid_2.6.7_hp128_s0p85_m5_c100000"
GC = "anthropic_claude-fable-5"
GO = "ollama_gemma4-26b-mlx"
T = "textmatcher_t2c3n2m3_nostops_0.1.7"
T_STRICT = "textmatcher_t3c5n3m8_nostops_0.1.7"

# member := (system tag, operator, operand)
VARIANTS = {
    "hybrid":        [(Q, None, None), (GC, None, None)],
    "hybrid_qgated": [(Q, "gate", None), (GC, None, None)],
    "hybrid_mgc":    [(M, None, None), (GC, None, None)],
    "hybrid_mgo":    [(M, None, None), (GO, None, None)],
    "hybrid_gcgo":   [(GC, None, None), (GO, None, None)],
    "hybrid_xqgc":   [(GC, "intersect", Q)],
    "hybrid_all":    [(M, None, None), (Q, "gate", None),
                      (GC, None, None), (GO, None, None)],
    # text-matcher is the high-precision, low-recall lexical system, so it poses the
    # mirror image of the Quid variants: Quid needed something to remove its false
    # positives, text-matcher needs something to supply what it never proposes.
    "hybrid_tgc":    [(T, None, None), (GC, None, None)],
    "hybrid_tgo":    [(T, None, None), (GO, None, None)],
    # No AI anywhere: do the two dedicated text-reuse systems complete each other? This
    # is the control for reading any hybrid gain as an LLM contribution.
    "hybrid_qt":     [(Q, None, None), (T, None, None)],
    # Agreement-only, the text-matcher analogue of hybrid_xqgc.
    "hybrid_xtgc":   [(GC, "intersect", T)],
}

FROZEN_VARIANTS = ("hybrid", "hybrid_qgated", "hybrid_mgc", "hybrid_mgo",
                   "hybrid_gcgo", "hybrid_xqgc", "hybrid_all")


def load(stem, tag, candidates_dir):
    path = candidates_dir / f"{stem}__{tag}.candidates.json"
    if not path.exists():
        raise SystemExit(f"missing input: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def span_overlaps(row, other):
    """General overlap, unlike the containment rule the union merge uses.

    The filters ask whether a span COINCIDES with evidence from elsewhere, which a
    partial overlap satisfies; the merge asks which of two spans to keep as the
    representative, where partial overlap is precisely the case that must not collapse.
    """
    if not (valid_span(row) and valid_span(other)):
        return False
    return (row["article_start_char"] < other["article_end_char"]
            and other["article_start_char"] < row["article_end_char"])


def apply_filter(rows, operator, operand_rows, body):
    # Filter spans by operator: gate keeps overlaps with marked spans, intersect with operand rows.
    if operator is None:
        return rows
    if operator == "gate":
        marked = [{"article_start_char": span["start_char"],
                   "article_end_char": span["end_char"]}
                  for span in extract_marked_spans(body)]
        return [row for row in rows
                if any(span_overlaps(row, span) for span in marked)]
    if operator == "intersect":
        return [row for row in rows
                if any(span_overlaps(row, other) for other in operand_rows)]
    raise SystemExit(f"unknown operator: {operator}")


def build(stem, body, members, tag, candidates_dir):
    rows = []
    for system_tag, operator, operand in members:
        member_rows = load(stem, system_tag, candidates_dir)
        operand_rows = load(stem, operand, candidates_dir) if operand else []
        rows.extend(apply_filter(member_rows, operator, operand_rows, body))
    return merge_candidate_rows(rows, body, tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("articles", nargs="+", help="CORPUS article plaintexts")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS),
                        default=sorted(VARIANTS))
    parser.add_argument("--candidates-dir", default=str(CANDIDATES_DIR))
    parser.add_argument("--out-dir", default=None,
                        help="defaults to --candidates-dir")
    parser.add_argument(
        "--verify", action="store_true",
        help="rebuild and compare against the stored candidates instead of writing; "
             "exits non-zero on any difference",
    )
    args = parser.parse_args()
    candidates_dir = Path(args.candidates_dir)
    out_dir = Path(args.out_dir) if args.out_dir else candidates_dir

    differences = 0
    for filename in args.articles:
        path = Path(filename)
        body = article_body(path.read_text(encoding="utf-8"))
        for tag in args.variants:
            output = build(path.stem, body, VARIANTS[tag], tag, candidates_dir)
            out = out_dir / f"{path.stem}__{tag}.candidates.json"
            payload = json.dumps(output, ensure_ascii=False, indent=2)
            if args.verify:
                if not out.exists():
                    print(f"ABSENT   {out.name}")
                    continue
                stored = out.read_text(encoding="utf-8")
                same = json.loads(stored) == output
                print(f"{'MATCH   ' if same else 'DIFFERS '} {out.name} "
                      f"({len(output)} candidates)")
                differences += not same
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                out.write_text(payload, encoding="utf-8")
                print(f"{tag}: {len(output)} candidates -> {out.name}")
    if args.verify and differences:
        raise SystemExit(f"{differences} variant(s) do not reproduce")


if __name__ == "__main__":
    main()
