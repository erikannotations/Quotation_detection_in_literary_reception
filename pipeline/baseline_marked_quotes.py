#!/usr/bin/env python3
"""High-recall non-LLM baseline: quotation marks -> deterministic locator.

This baseline extracts every marked span of at least five words without attempting
authorship classification. `run_pipeline.py` then tests each span against the primary
corpus. Block quotations lost during plaintext conversion remain a known limitation.
"""
import argparse
import json
import re
from pathlib import Path

from quotation_spans import article_body, extract_marked_spans
from project_paths import CANDIDATES_DIR


def article_key(text, path):
    match = re.search(r"^Key:\s*(\S+)", text, re.M)
    return match.group(1) if match else path.stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("articles", nargs="+")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--out-dir", default=str(CANDIDATES_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each article: extract marked spans and format as candidate records.
    for filename in args.articles:
        path = Path(filename)
        raw = path.read_text(encoding="utf-8")
        body = article_body(raw)
        key = article_key(raw, path)
        spans = extract_marked_spans(body, min_words=args.min_words)
        rows = []
        for i, span in enumerate(spans, 1):
            rows.append({
                "id": f"{key}-mq-{i:04d}",
                "article": path.stem,
                "model_tag": "heuristic_marked",
                "quote": span["quote"],
                "detector_guess": "unclassified marked span",
                "attributed_work": "",
                "attributed_page": "",
                "confidence": "baseline",
                "article_start_char": span["start_char"],
                "article_end_char": span["end_char"],
                "quotation_form": span["quotation_form"],
            })
        out = out_dir / f"{path.stem}__heuristic_marked.candidates.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name}: {len(rows)} marked spans -> {out}")


if __name__ == "__main__":
    main()
