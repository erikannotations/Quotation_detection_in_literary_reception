#!/usr/bin/env python3
"""Validate completed annotation templates and compile evaluator-ready gold files."""
import argparse
import json
import sys
from pathlib import Path
from project_paths import ANNOTATIONS_DIR, GOLD_DIR
from quotation_spans import MIN_QUOTE_WORDS, under_minimum, word_count
from review_annotations import metadata_normalized_body

# Article plaintexts, for compile-time re-validation of span offsets.
ARTICLE_DIRS = [
    Path(__file__).resolve().parent.parent / "annotation_workspace" / name
    for name in ("01_orbit_articles", "03_non_orbit_cc_articles")
]
ARTICLE_DIRS_BY_STEM = {
    path.stem: path
    for directory in ARTICLE_DIRS if directory.is_dir()
    for path in directory.glob("*.txt")
}

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=str(ANNOTATIONS_DIR))
    parser.add_argument("--output", default=str(GOLD_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    compiled = 0
    incomplete = []
    failures = []
    all_excluded = []
    for path in sorted(Path(args.annotations).glob("*.annotation.json")):
        # One malformed or half-finished file must not abort compilation of every
        # other completed article: record the problem, skip the file, keep going,
        # and fail the run at the end (hardened 22 July 2026, pre-annotation audit).
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            failures.append(f"{path.name}: unreadable ({error})")
            continue
        if data.get("review_status") != "complete":
            incomplete.append(path.name)
            continue
        spans = data.get("candidates", []) + data.get("manually_added_spans", [])
        # Identity guards. Manual span ids are per-annotator serials (`KEY-mNNNN`),
        # so two annotators adding spans in different orders mint the SAME id for
        # DIFFERENT passages; an adjudication or pooling merge keyed on id therefore
        # silently skips or overwrites.
        # Offsets are the safe key, so both are checked here: gold
        # ids must be unique because `gold_id` is what the evaluator keys results on,
        # and duplicate offsets mean the same passage would enter gold twice and be
        # scored twice.
        from collections import Counter
        id_counts = Counter(row.get("annotation_id") for row in spans)
        repeated_ids = sorted(i for i, n in id_counts.items() if n > 1)
        if repeated_ids:
            failures.append(f"{path.name}: duplicate annotation ids {repeated_ids[:5]} — "
                            f"gold is keyed on these downstream")
            continue
        in_scope_offsets = Counter(
            (row.get("start_char"), row.get("end_char")) for row in spans
            if row.get("is_in_scope"))
        repeated_spans = sorted(k for k, n in in_scope_offsets.items() if n > 1)
        if repeated_spans:
            failures.append(f"{path.name}: the same in-scope span appears more than once "
                            f"at {repeated_spans[:3]} — it would enter gold twice")
            continue

        unresolved = [row.get("annotation_id", "<missing id>") for row in spans
                      if row.get("is_in_scope") is None]
        if unresolved:
            failures.append(f"{path.name}: unresolved labels: {unresolved[:8]}")
            continue
        # Re-verify offsets at COMPILE time, not only at annotation time. The
        # annotation app validates on save, but gold has been hand-edited outside
        # the app (the 26 July adjudication and alterations sweep), and until now
        # nothing re-checked that a span still sliced back to its recorded text.
        # A stale offset would enter gold silently and every system would then be
        # scored against a quotation the article does not contain.
        source = ARTICLE_DIRS_BY_STEM.get(data.get("article"))
        if source is None:
            # Synthetic fixtures have no article on disk. Say so loudly rather than
            # failing: a real article is always present, so real gold is always
            # re-validated, and a silent skip is what this check exists to prevent.
            print(f"NOTE: {path.name}: no article text found; span offsets were NOT "
                  f"re-validated for this file", file=sys.stderr)
        else:
            try:
                body = metadata_normalized_body(source.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                failures.append(f"{path.name}: article text unreadable ({error})")
                continue
            drifted = [row.get("annotation_id", "<missing id>") for row in spans
                       if body[row.get("start_char", 0):row.get("end_char", 0)]
                       != row.get("quote")]
            if drifted:
                failures.append(f"{path.name}: offsets no longer match the article "
                                f"text for {drifted[:8]}")
                continue

        problems = []
        excluded = []
        for row in spans:
            if not row.get("is_in_scope"):
                continue
            # The minimum is part of the frozen target definition, so a shorter span
            # cannot be gold whatever else is true of it. Enforced here because
            # nothing else enforces it on a SEEDED candidate: the interface checks the
            # length of manually added and corrected spans, but `set_positive` does
            # not.
            # This is the only gate that sees both annotators'
            # returned work regardless of which version of the app produced it.
            #
            # EXCLUDED, not failed (amended 27 July 2026): a returning annotation must
            # not be able to block the whole pipeline over a span that is out of scope
            # by arithmetic. The exclusion is recorded in the gold file itself and
            # printed, so it is auditable rather than silent — which is the property
            # that matters, since the span never had a legitimate route into gold.
            if under_minimum(row.get("quote", "")):
                excluded.append({
                    "gold_id": row.get("annotation_id"),
                    "quote": row.get("quote", ""),
                    "words": word_count(row.get("quote", "")),
                    "reason": f"under the {MIN_QUOTE_WORDS}-word minimum",
                })
                continue
            if row.get("in_primary_corpus") is False:
                # In scope but not locatable in the ten primary texts: valid gold
                # (eval_compare scores it via the locatable/in_primary_corpus
                # split), provided the reason is on record.
                if not (row.get("notes") or "").strip():
                    problems.append(f"in-scope {row.get('annotation_id')} marked "
                                    "not-locatable but has no explanatory note")
            elif not row.get("true_pid") or not row.get("true_work"):
                problems.append(f"in-scope {row.get('annotation_id')} lacks source work")
            elif not isinstance(row.get("true_section_n"), int) or not row.get("true_section_label"):
                problems.append(f"in-scope {row.get('annotation_id')} lacks a confirmed chapter/episode")
        if problems:
            failures.append(f"{path.name}: " + "; ".join(problems[:8]))
            continue
        excluded_ids = {item["gold_id"] for item in excluded}
        quotes = []
        for row in spans:
            if not row.get("is_in_scope"):
                continue
            if row.get("annotation_id") in excluded_ids:
                continue
            quotes.append({
                "gold_id": row["annotation_id"],
                "quote": row["quote"],
                "article_start_char": row.get("start_char"),
                "article_end_char": row.get("end_char"),
                "true_pid": row["true_pid"],
                "true_work": row["true_work"],
                "true_section_n": row["true_section_n"],
                "true_section_label": row["true_section_label"],
                "in_primary_corpus": bool(row.get("in_primary_corpus", True)),
                "quotation_form": row.get("quotation_form", ""),
                "alterations": row.get("alterations", []),
                "notes": row.get("notes", ""),
            })
        output = {
            "schema_version": "2.0",
            "article": data["article"],
            "corpus_id": data["corpus_id"],
            "split": data["split"],
            "body_sha256": data["body_sha256"],
            "quotes": quotes,
            # Always present, empty in the normal case, so a reader can tell "nothing
            # was excluded" from "this field predates the check".
            "excluded_out_of_scope": excluded,
        }
        if excluded:
            all_excluded.append((path.name, excluded))
        destination = out_dir / f"{data['article']}.gold.json"
        destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        compiled += 1
    print(f"compiled {compiled}; not yet complete {len(incomplete)}; "
          f"failed {len(failures)}; spans excluded as out of scope "
          f"{sum(len(items) for _name, items in all_excluded)}")
    for name in incomplete:
        print(f"  not complete: {name}")
    for name, items in all_excluded:
        # Printed per span, not summarised: an annotator confirmed each of these in
        # scope, so each is a scope disagreement worth an adjudicator's eye even though
        # the arithmetic has already settled it.
        for item in items:
            print(f"  EXCLUDED from gold: {name}: {item['gold_id']} is "
                  f"{item['words']} words ({item['reason']}): {item['quote'][:48]!r}")
    for message in failures:
        print(f"  FAILED: {message}")
    if failures:
        raise SystemExit("gold compilation had failures (listed above)")


if __name__ == "__main__":
    main()
