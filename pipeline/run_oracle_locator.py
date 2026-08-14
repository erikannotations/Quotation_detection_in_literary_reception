#!/usr/bin/env python3
"""Run gold quotation spans through the locator to measure its independent ceiling."""
import argparse
import json
from pathlib import Path

from project_paths import CANDIDATES_DIR, GOLD_DIR
from run_pipeline import main as run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", default=str(GOLD_DIR))
    parser.add_argument("--out-dir", default=str(CANDIDATES_DIR))
    args = parser.parse_args()

    gold_paths = sorted(Path(args.gold_dir).glob("*.gold.json"))
    if not gold_paths:
        raise SystemExit("no compiled gold files found; complete annotations first")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tally locator performance on gold spans to measure its independent ceiling.
    total_locatable = top_ok = alt_ok = confirmed = 0
    section_denom = section_ok = 0
    for gold_path in gold_paths:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        # Convert gold spans to candidate format for pipeline processing.
        rows = [{
            "id": item["gold_id"],
            "article": gold["article"],
            "model_tag": "oracle_gold",
            "quote": item["quote"],
            "detector_guess": "gold span",
            "attributed_work": item.get("true_work", ""),
            "attributed_page": "",
            "confidence": "oracle",
            "article_start_char": item.get("article_start_char"),
            "article_end_char": item.get("article_end_char"),
        } for item in gold.get("quotes", [])]
        candidate_path = out_dir / f"{gold['article']}__oracle_gold.candidates.json"
        candidate_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        run_pipeline(candidate_path)

        results = {
            row["id"]: row for row in json.loads(
                candidate_path.with_suffix(".results.json").read_text(encoding="utf-8")
            )
        }
        for item in gold.get("quotes", []):
            if not item.get("in_primary_corpus", True):
                continue
            total_locatable += 1
            result = results.get(item["gold_id"], {})
            if not result.get("confirmed_pynchon_fiction"):
                continue
            confirmed += 1
            true_pid = item.get("true_pid", "")
            top_correct = result.get("pid", "") == true_pid
            top_ok += int(top_correct)
            alternatives = set(filter(None, (result.get("candidate_pids") or "").split("|")))
            alternatives.add(result.get("pid", ""))
            alt_ok += int(true_pid in alternatives)
            # Score chapter/section accuracy when the locator recovered the correct work.
            true_section_n = item.get("true_section_n")
            if true_section_n is not None:
                section_denom += 1
                section_ok += int(top_correct and result.get("section_n") == true_section_n)

    def ratio(n, d):
        return n / d if d else float("nan")

    print(
        f"oracle locator: locatable={total_locatable} confirmed={confirmed} "
        f"top={top_ok}/{total_locatable} ({ratio(top_ok, total_locatable):.3f}) "
        f"alternative={alt_ok}/{total_locatable} ({ratio(alt_ok, total_locatable):.3f}) "
        f"section={section_ok}/{section_denom} ({ratio(section_ok, section_denom):.3f})"
    )


if __name__ == "__main__":
    main()
