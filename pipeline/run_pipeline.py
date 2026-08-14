#!/usr/bin/env python3
"""Batch-run the locate step over a candidates file and report matches.

candidates JSON: list of {id, quote, detector_guess?}. A candidate is *confirmed*
as a Pynchon-fiction quotation iff it locates in the primary corpus at/above the
score threshold. Writes results CSV + JSON next to the candidates file.
"""
import csv
import json
import sys
from pathlib import Path
from locate_quotes import Corpus
from project_paths import LEGACY_CANDIDATES_DIR

THRESHOLD = 0.75  # min difflib ratio of the anchor fragment to confirm

FIELDS = [
    "id", "detector_guess", "confirmed_pynchon_fiction", "novel", "pid",
    "section", "section_n", "part", "char_offset", "score", "source_ambiguous",
    "candidate_pids", "n_near_locations", "quote", "quote_preview",
]


def main(cand_path):
    cand_path = Path(cand_path)
    cands = json.loads(cand_path.read_text(encoding="utf-8"))
    # eval_compare.discover() keys results by candidate id, so a duplicate id would
    # silently drop every row but the last. None collide today; fail loudly if one
    # ever does rather than losing candidates inside a frozen run.
    ids = [cand.get("id") for cand in cands]
    duplicates = sorted({i for i in ids if i is not None and ids.count(i) > 1})
    if duplicates:
        raise SystemExit(
            f"{cand_path.name}: duplicate candidate ids {duplicates[:5]} — results are "
            f"keyed by id downstream and rows would be silently lost")
    # Score each candidate by attempting to locate it in the primary corpus.
    c = Corpus()
    rows = []
    for cand in cands:
        r = c.locate(cand["quote"])
        confirmed = bool(r and r["score"] >= THRESHOLD)
        rows.append({
            "id": cand["id"],
            "detector_guess": cand.get("detector_guess", ""),
            "confirmed_pynchon_fiction": confirmed,
            "novel": r["novel"] if confirmed else "",
            "pid": r["pid"] if confirmed else "",
            "section": (r["section"] or "") if confirmed else "",
            "section_n": r["section_n"] if confirmed else "",
            "part": (r["part"] or "") if confirmed and r["part"] else "",
            "char_offset": r["char_offset"] if confirmed else "",
            "score": r["score"] if r else "",
            "source_ambiguous": r.get("source_ambiguous", False) if confirmed else False,
            "candidate_pids": "|".join(r.get("candidate_pids", [])) if confirmed else "",
            "n_near_locations": r.get("n_near_locations", 0) if confirmed else 0,
            "quote": cand["quote"],
            "quote_preview": cand["quote"][:60].replace("\n", " "),
        })

    out_csv = cand_path.with_suffix(".results.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    cand_path.with_suffix(".results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'id':8} {'conf':5} {'novel':20} {'section':12} {'score':>5}  guess")
    print("-" * 92)
    for r in rows:
        mark = "YES" if r["confirmed_pynchon_fiction"] else "no"
        loc = f"{r['novel'][:20]:20} {str(r['section'])[:12]:12}" if r["confirmed_pynchon_fiction"] else f"{'—':20} {'—':12}"
        print(f"{r['id']:8} {mark:5} {loc} {str(r['score']):>5}  {r['detector_guess'][:34]}")
    n_conf = sum(r["confirmed_pynchon_fiction"] for r in rows)
    print(f"\nconfirmed {n_conf}/{len(rows)}  -> {out_csv.name}")


if __name__ == "__main__":
    default = LEGACY_CANDIDATES_DIR / "candidates_eve.json"
    main(sys.argv[1] if len(sys.argv) > 1 else default)
