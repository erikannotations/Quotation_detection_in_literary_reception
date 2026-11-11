#!/usr/bin/env python3
"""Re-score the released evaluation from the released materials alone.

The released gold and candidate files carry no quoted text (the primary texts
are in copyright). Each record instead keeps character offsets
(``article_start_char``/``article_end_char``) into its article's body text,
and the sixteen CC BY articles ship in ``articles/``. This script makes the
re-scoring claim executable:

1. Copies the release into a sibling working directory (``rescore_work/``);
   nothing in the release itself is modified.
2. Rebuilds every record's ``quote`` text by slicing the article body at the
   record's offsets — exactly reversing the redaction.
3. Restricts the evaluation to the twelve held-out test articles (the paper's
   headline numbers; the split for each article is in
   ``evaluation/sample_manifest.csv``).
4. Runs the shipped scorer (``eval_compare.py``) and table generator
   unmodified, writing ``rescore_work/pipeline/evaluation/rescore_metrics.json``
   and printing the results table.

Known, deliberate limit: a small number of candidate rows (LLM proposals whose
wording could not be aligned to the article, so they have no offsets) cannot
be rebuilt — their only content was the redacted text. These rows are counted
and reported below (62 rows on the test set). They come from the two LLM
systems, so those systems and the hybrids containing them re-score within
0.01 F1 of the published values, with individual precision and recall figures
differing by up to 0.016, in either direction for hybrids; the published
values scored these rows against their original, unredacted text. All other
systems reproduce the published numbers exactly.

Run from the release root or from ``pipeline/``:  python3 rescore_release.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def find_release_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent, here):
        if (cand / "articles").is_dir() and (cand / "pipeline").is_dir():
            return cand
    sys.exit("This script re-scores the released artifact: it must be run "
             "inside the release, next to the articles/ folder.")


def main() -> None:
    release = find_release_root()
    work = release.parent / "rescore_work"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(release, work)

    pipeline = work / "pipeline"
    articles = work / "articles"
    sys.path.insert(0, str(pipeline))
    from quotation_spans import article_body  # shipped module

    manifest = pipeline / "evaluation" / "sample_manifest.csv"
    split_by_stem = {}
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = Path(row["text_path"]).stem
            split_by_stem[stem] = row["split"]
    test_stems = {s for s, sp in split_by_stem.items() if sp == "test"}

    bodies = {}

    def body(stem: str) -> str:
        if stem not in bodies:
            text = (articles / f"{stem}.txt").read_text(encoding="utf-8")
            bodies[stem] = article_body(text)
        return bodies[stem]

    def rehydrate(rows, stem, counts):
        for r in rows:
            sc, ec = r.get("article_start_char"), r.get("article_end_char")
            if sc is not None and ec is not None:
                r["quote"] = body(stem)[sc:ec]
                counts["rebuilt"] += 1
            else:
                counts["unrecoverable"] += 1
        return rows

    counts = {"rebuilt": 0, "unrecoverable": 0}
    unrecoverable_by_file = {}

    gold_dir = pipeline / "evaluation" / "gold"
    for p in sorted(gold_dir.glob("*.gold.json")):
        stem = p.name[: -len(".gold.json")]
        if stem not in test_stems:
            p.unlink()
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        before = counts["unrecoverable"]
        rehydrate(d["quotes"], stem, counts)
        if counts["unrecoverable"] > before:
            unrecoverable_by_file[p.name] = counts["unrecoverable"] - before
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1),
                     encoding="utf-8")

    cand_dir = pipeline / "evaluation" / "candidates"
    for p in sorted(cand_dir.glob("*__*")):
        stem = p.name.split("__", 1)[0]
        if stem not in test_stems:
            p.unlink()
            continue
        if not p.name.endswith(".candidates.json"):
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))
        before = counts["unrecoverable"]
        rehydrate(rows, stem, counts)
        if counts["unrecoverable"] > before:
            unrecoverable_by_file[p.name] = counts["unrecoverable"] - before
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                     encoding="utf-8")

    print(f"quote text rebuilt from offsets: {counts['rebuilt']} rows")
    print(f"unrecoverable (no offsets; text redacted): "
          f"{counts['unrecoverable']} rows")
    for name, n in sorted(unrecoverable_by_file.items()):
        print(f"  {name}: {n}")
    print()

    metrics = pipeline / "evaluation" / "rescore_metrics.json"
    r = subprocess.run(
        [sys.executable, "eval_compare.py", "--no-legacy-gold",
         "--json-out", str(metrics)],
        cwd=pipeline)
    if r.returncode != 0:
        sys.exit("eval_compare.py failed")
    subprocess.run(
        [sys.executable, "make_results_table.py",
         "--metrics-path", str(metrics), "--format", "md", "--table", "all"],
        cwd=pipeline)

    print()
    print("Note: systems with unrecoverable rows above (the LLM detectors and "
          "hybrids that include them) differ from the published values by up "
          "to about 0.01 F1, because the published values included those "
          "rows' original text; every other system reproduces the published "
          "numbers exactly.")


if __name__ == "__main__":
    main()
