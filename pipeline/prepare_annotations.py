#!/usr/bin/env python3
"""Seed human-review templates from the deterministic sample manifest.

The automatic spans are only annotation aids. Reviewers must inspect the complete
article and add missed block quotations or other marked spans before setting the
document status to `complete`.
"""
import argparse
import csv
import json
from pathlib import Path

from quotation_spans import article_body, extract_marked_spans
from project_paths import ANNOTATIONS_DIR, SAMPLE_MANIFEST

BASE = Path(__file__).resolve().parent
DEFAULT_CORPUS = BASE.parent.parent / "Reception_Corpus" / "corpus_articles"


def context(text, start, end, width=180):
    return {
        "before": text[max(0, start - width):start].replace("\n", " "),
        "after": text[end:min(len(text), end + width)].replace("\n", " "),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(SAMPLE_MANIFEST))
    parser.add_argument("--corpus-root", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output-dir", default=str(ANNOTATIONS_DIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with Path(args.manifest).open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Generate annotation templates with marked-span candidates for each sampled article.
    written = skipped = 0
    for item in manifest:
        source = Path(args.corpus_root) / item["text_path"]
        body = article_body(source.read_text(encoding="utf-8"))
        candidates = []
        for i, span in enumerate(extract_marked_spans(body), 1):
            row = {
                "annotation_id": f"{item['zotero_key']}-q{i:04d}",
                **span,
                **context(body, span["start_char"], span["end_char"]),
                "is_in_scope": None,
                "exclusion_category": "",
                "true_pid": "",
                "true_work": "",
                "true_section_n": None,
                "true_section_label": "",
                "in_primary_corpus": None,
                "alterations": [],
                "notes": "",
            }
            candidates.append(row)
        template = {
            "schema_version": "2.0-working",
            "article": source.stem,
            "corpus_id": item["corpus_id"],
            "zotero_key": item["zotero_key"],
            "split": item["split"],
            "body_sha256": item["body_sha256"],
            "reviewer": "",
            "review_status": "unreviewed",
            "full_document_reviewed": False,
            "completed_at": "",
            "automatic_candidate_warning": (
                "Not exhaustive: inspect the full article and add missed block/single-quote spans."
            ),
            "candidates": candidates,
            "manually_added_spans": [],
        }
        destination = out_dir / f"{source.stem}.annotation.json"
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        destination.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1
    print(f"annotation templates: wrote {written}, skipped {skipped} existing")


if __name__ == "__main__":
    main()
