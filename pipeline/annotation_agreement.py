#!/usr/bin/env python3
"""Measure independent span, scope-label, source-work, and locus annotation agreement."""
import argparse
import json
from pathlib import Path

from locate_quotes import normalize_query
from quotation_spans import under_minimum


def spans(data):
    """Comparable spans for one annotator's file.

    Spans below the frozen minimum are dropped before anything is compared. They are
    out of scope by arithmetic, so no annotator judgement about them is meaningful —
    but six of them are sitting in templates already in both annotators' hands
    (seeded before the 23 July `word_count()` bracket fix), three in the second
    annotator's set. Left in, a differing label on one would register as a genuine
    scope disagreement and depress the reported inter-annotator agreement over a span
    neither annotator was ever being asked to judge. Excluded here so the agreement
    figures describe the task as defined.
    """
    rows = data.get("candidates", []) + data.get("manually_added_spans", [])
    return [row for row in rows if not under_minimum(row.get("quote", ""))]


def span_score(a, b):
    a0, a1 = a.get("start_char"), a.get("end_char")
    b0, b1 = b.get("start_char"), b.get("end_char")
    if all(isinstance(x, int) for x in (a0, a1, b0, b1)):
        overlap = max(0, min(a1, b1) - max(a0, b0))
        if overlap:
            return overlap / max(1, min(a1 - a0, b1 - b0))
    aq, bq = normalize_query(a.get("quote", "")), normalize_query(b.get("quote", ""))
    return 1.0 if aq and aq == bq else 0.0


def align(a_rows, b_rows, threshold=0.5):
    # Maximum-cardinality bipartite matching via augmenting paths, strongest edges tried first.
    edges = {
        ai: sorted(
            ((bi, span_score(a, b)) for bi, b in enumerate(b_rows)
             if span_score(a, b) >= threshold),
            key=lambda item: (-item[1], item[0]),
        )
        for ai, a in enumerate(a_rows)
    }
    b_to_a = {}

    def augment(ai, seen):
        for bi, _score in edges[ai]:
            if bi in seen:
                continue
            seen.add(bi)
            if bi not in b_to_a or augment(b_to_a[bi], seen):
                b_to_a[bi] = ai
                return True
        return False

    order = sorted(range(len(a_rows)), key=lambda ai: (-(edges[ai][0][1]
                                                        if edges[ai] else -1), ai))
    for ai in order:
        augment(ai, set())
    return [(ai, bi, span_score(a_rows[ai], b_rows[bi])) for bi, ai in b_to_a.items()]


def kappa_binary(pairs):
    if not pairs:
        return float("nan")
    observed = sum(a == b for a, b in pairs) / len(pairs)
    pa = sum(bool(a) for a, _b in pairs) / len(pairs)
    pb = sum(bool(b) for _a, b in pairs) / len(pairs)
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def compare_documents(a_data, b_data, threshold=0.5):
    a_rows, b_rows = spans(a_data), spans(b_data)
    matches = align(a_rows, b_rows, threshold)
    matched = len(matches)
    precision = matched / len(a_rows) if a_rows else float("nan")
    recall = matched / len(b_rows) if b_rows else float("nan")
    span_f1 = (2 * precision * recall / (precision + recall)
               if precision == precision and recall == recall and precision + recall else float("nan"))

    # Extract agreement pairs for scope label, source work, and locus (chapter/section).
    label_pairs = []
    source_pairs = []
    locus_pairs = []
    for ai, bi, _score in matches:
        a, b = a_rows[ai], b_rows[bi]
        if a.get("is_in_scope") is not None and b.get("is_in_scope") is not None:
            label_pairs.append((bool(a["is_in_scope"]), bool(b["is_in_scope"])))
            if a["is_in_scope"] and b["is_in_scope"]:
                source_pairs.append((a.get("true_pid", ""), b.get("true_pid", "")))
                # Locus agreement is conditioned on agreeing which work it is --
                # comparing chapter numbers across a source disagreement isn't
                # meaningful (Chapter 5 of one novel vs. Chapter 5 of another).
                if a.get("true_pid") and a["true_pid"] == b.get("true_pid"):
                    locus_pairs.append((a.get("true_section_n"), b.get("true_section_n")))
    return {
        "article": a_data.get("article", b_data.get("article", "")),
        "a_spans": len(a_rows), "b_spans": len(b_rows), "matched_spans": matched,
        "span_precision_a_to_b": precision, "span_recall_a_to_b": recall,
        "span_f1": span_f1,
        "scope_n": len(label_pairs),
        "scope_agreement": (sum(a == b for a, b in label_pairs) / len(label_pairs)
                            if label_pairs else float("nan")),
        "scope_kappa": kappa_binary(label_pairs),
        "source_n": len(source_pairs),
        "source_agreement": (sum(a == b for a, b in source_pairs) / len(source_pairs)
                             if source_pairs else float("nan")),
        "locus_n": len(locus_pairs),
        "locus_agreement": (sum(a == b for a, b in locus_pairs) / len(locus_pairs)
                            if locus_pairs else float("nan")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reviewer_a")
    parser.add_argument("reviewer_b")
    parser.add_argument("--span-threshold", type=float, default=0.5)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    a_dir, b_dir = Path(args.reviewer_a), Path(args.reviewer_b)
    common = sorted({p.name for p in a_dir.glob("*.annotation.json")}
                    & {p.name for p in b_dir.glob("*.annotation.json")})
    if not common:
        raise SystemExit("no same-named annotation files found in the two reviewer directories")
    rows = []
    for name in common:
        a = json.loads((a_dir / name).read_text(encoding="utf-8"))
        b = json.loads((b_dir / name).read_text(encoding="utf-8"))
        rows.append(compare_documents(a, b, args.span_threshold))

    def mean(key):
        values = [row[key] for row in rows if row[key] == row[key]]
        return sum(values) / len(values) if values else float("nan")

    output = {
        "documents": len(rows), "span_threshold": args.span_threshold,
        "macro_span_f1": mean("span_f1"),
        "macro_scope_agreement": mean("scope_agreement"),
        "macro_scope_kappa": mean("scope_kappa"),
        "macro_source_agreement": mean("source_agreement"),
        "macro_locus_agreement": mean("locus_agreement"),
        "document_metrics": rows,
    }
    print(
        f"documents={output['documents']} span_F1={output['macro_span_f1']:.3f} "
        f"scope_agreement={output['macro_scope_agreement']:.3f} "
        f"scope_kappa={output['macro_scope_kappa']:.3f} "
        f"source_agreement={output['macro_source_agreement']:.3f} "
        f"locus_agreement={output['macro_locus_agreement']:.3f}"
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
