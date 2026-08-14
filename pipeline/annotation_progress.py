#!/usr/bin/env python3
"""Report annotation workload and completion by development/test split."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from project_paths import ANNOTATIONS_DIR, SAMPLE_MANIFEST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=str(ANNOTATIONS_DIR))
    parser.add_argument("--manifest", default=str(SAMPLE_MANIFEST))
    parser.add_argument("--json-out")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    with Path(args.manifest).open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    split_by_stem = {Path(row["text_path"]).stem: row["split"] for row in manifest}
    documents = []
    for path in sorted(Path(args.annotations).glob("*.annotation.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("candidates", []) + data.get("manually_added_spans", [])
        labelled = sum(row.get("is_in_scope") is not None for row in rows)
        positive = sum(row.get("is_in_scope") is True for row in rows)
        documents.append({
            "article": data["article"],
            "split": data.get("split") or split_by_stem.get(data["article"], "unknown"),
            "status": data.get("review_status", "unreviewed"),
            "spans": len(rows), "labelled": labelled,
            "unresolved": len(rows) - labelled, "positive": positive,
        })

    totals = defaultdict(lambda: defaultdict(int))
    for row in documents:
        split = row["split"]
        totals[split]["documents"] += 1
        totals[split]["complete_documents"] += int(row["status"] == "complete")
        for key in ("spans", "labelled", "unresolved", "positive"):
            totals[split][key] += row[key]
    output = {"splits": {key: dict(value) for key, value in totals.items()},
              "documents": documents}

    print(f"{'split':12} {'docs':>4} {'done':>4} {'spans':>6} {'labelled':>8} {'open':>6} {'positive':>8}")
    print("-" * 62)
    for split in sorted(totals):
        row = totals[split]
        print(f"{split:12} {row['documents']:>4} {row['complete_documents']:>4} "
              f"{row['spans']:>6} {row['labelled']:>8} {row['unresolved']:>6} "
              f"{row['positive']:>8}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.require_complete and any(row["status"] != "complete" for row in documents):
        raise SystemExit("annotation set is not complete")


if __name__ == "__main__":
    main()
