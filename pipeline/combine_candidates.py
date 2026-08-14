#!/usr/bin/env python3
"""Form a provenance-preserving hybrid union of detector candidate files."""
import argparse
import json
import re
from pathlib import Path

from locate_quotes import normalize_query
from project_paths import CANDIDATES_DIR
from quotation_spans import article_body, find_quote_offsets, word_count

# A contained span is absorbed into its container only when the two record the SAME
# quotation. Below this coverage the inner span is a fragment of the outer, not a noisy
# copy of it — the re-quotation case, where a scholar block-quotes a passage and then
# quotes a phrase from inside it again. See the 27 July 2026 audit entry.
SAME_SPAN_COVERAGE = 0.6
# The frozen scope minimum. A shorter span cannot be an in-scope quotation on its own,
# so keeping it separate could only ever add a false positive.
MIN_QUOTE_WORDS = 5


def valid_span(row):
    start, end = row.get("article_start_char"), row.get("article_end_char")
    return isinstance(start, int) and isinstance(end, int) and 0 <= start < end


def overlaps(a, b):
    """True when one span contains the other, or the two record the same quotation.

    Deliberately *not* general overlap: the rule is to retain the longer span
    "when one contains another"; collapsing partial overlaps too was an
    over-generalization that lost true positives.
    """
    if valid_span(a) and valid_span(b):
        return (a["article_start_char"] <= b["article_start_char"]
                and b["article_end_char"] <= a["article_end_char"]) or (
                b["article_start_char"] <= a["article_start_char"]
                and a["article_end_char"] <= b["article_end_char"])
    return normalize_query(a.get("quote", "")) == normalize_query(b.get("quote", ""))


def _extent(row):
    """Span length, falling back to quote length when offsets are unavailable."""
    if valid_span(row):
        return row["article_end_char"] - row["article_start_char"]
    return len(row.get("quote", ""))


def merge_candidate_rows(rows, body, tag="hybrid"):
    """Collapse contained spans, retaining the longest member and detector provenance."""
    prepared = []
    for input_i, original in enumerate(rows):
        row = dict(original)
        if not valid_span(row):
            start, end, occurrences = find_quote_offsets(body, row.get("quote", ""))
            row["article_start_char"] = start
            row["article_end_char"] = end
            row["article_offset_occurrences"] = occurrences
        row["_input_i"] = input_i
        prepared.append(row)

    # Group by CONTAINMENT ONLY, and deliberately WITHOUT transitive closure.
    # Taking the transitive closure reintroduces the very bug this rule replaced:
    # a short span contained in two larger spans that merely overlap each other
    # bridges them into one component, and the "longest member" rule then discards
    # the tighter of the two. Concretely, A=[5,32] and C=[13,50] partially overlap
    # and neither contains the other, but B=[13,23] sits inside both — under
    # closure all three collapse to C and the precise span A is lost. That is a
    # realistic three-detector configuration (a short heuristic span inside both a
    # tight generative span and an over-long Quid span). So: every MAXIMAL span
    # survives as its own row, and a contained span contributes only its
    # provenance, to each container that swallows it.
    ordered = sorted(
        prepared,
        key=lambda r: (
            r.get("article_start_char") if valid_span(r) else 10**18,
            -(len(r.get("quote", ""))),
            r["_input_i"],
        ),
    )

    def same_quotation(inner, outer):
        """Whether a contained span is a noisy copy of its container, not a fragment.

        """
        inner_words = word_count(inner.get("quote", ""))
        if inner_words < MIN_QUOTE_WORDS:
            return True
        outer_words = word_count(outer.get("quote", ""))
        if not outer_words:
            return True
        return inner_words / outer_words >= SAME_SPAN_COVERAGE

    def contained_in(inner, outer):
        return inner is not outer and overlaps(inner, outer) and (
            _extent(outer) > _extent(inner)
            or (_extent(outer) == _extent(inner)
                and outer["_input_i"] < inner["_input_i"])) and same_quotation(
            inner, outer)

    # Assemble maximal contained-span groups, preserving one representative per group.
    components = []
    for row in ordered:
        if any(contained_in(row, other) for other in ordered):
            continue
        components.append(
            [row] + [other for other in ordered if contained_in(other, row)])

    # Build output, selecting the longest member from each group and collecting detector provenance.
    output = []
    for component in sorted(
        components,
        key=lambda c: min((r.get("article_start_char", 10**18)
                           if valid_span(r) else 10**18) for r in c),
    ):
        representative = max(
            component,
            key=lambda r: (
                (r["article_end_char"] - r["article_start_char"]) if valid_span(r)
                else len(r.get("quote", "")),
                len(r.get("quote", "")),
                -r["_input_i"],
            ),
        )
        model_tags = sorted({r.get("model_tag", "unknown") for r in component})
        member_ids = [r.get("id", "") for r in component]
        guesses = list(dict.fromkeys(filter(None, (r.get("detector_guess", "")
                                                    for r in component))))
        works = list(dict.fromkeys(filter(None, (r.get("attributed_work", "")
                                                  for r in component))))
        output.append({
            "id": "",  # assigned after stable article-order sorting
            "article": representative.get("article", ""),
            "model_tag": tag,
            "quote": representative.get("quote", ""),
            "detector_guess": " | ".join(guesses),
            "attributed_work": " | ".join(works),
            "attributed_page": representative.get("attributed_page", ""),
            "confidence": "hybrid",
            "article_start_char": representative.get("article_start_char"),
            "article_end_char": representative.get("article_end_char"),
            "proposed_by": model_tags,
            "hybrid_member_ids": member_ids,
        })

    # Keyed on the variant tag, not the literal "hybrid": the frozen candidate files
    # carry tag-based ids, and with eleven variants a shared prefix would give the same
    # id to different spans in different files. Callers that pass no tag are unaffected.
    for i, row in enumerate(output, 1):
        row["id"] = f"{row['article']}-{tag}-{i:04d}"
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("article", help="CORPUS article plaintext used for offset recovery")
    parser.add_argument("candidate_files", nargs="+", help="two or more detector outputs")
    parser.add_argument("--tag", default="hybrid")
    parser.add_argument("--out-dir", default=str(CANDIDATES_DIR))
    args = parser.parse_args()
    if len(args.candidate_files) < 2:
        raise SystemExit("hybrid union requires at least two candidate files")

    article_path = Path(args.article)
    body = article_body(article_path.read_text(encoding="utf-8"))
    rows = []
    for filename in args.candidate_files:
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        rows.extend(data)
    articles = {row.get("article", "") for row in rows if row.get("article")}
    if len(articles) > 1:
        raise SystemExit(f"candidate files contain multiple articles: {sorted(articles)}")

    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.tag).strip("-") or "hybrid"
    output = merge_candidate_rows(rows, body, tag)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{article_path.stem}__{tag}.candidates.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)} inputs -> {len(output)} hybrid candidates -> {out}")


if __name__ == "__main__":
    main()
