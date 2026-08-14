#!/usr/bin/env python3
"""Build Against the Day (P07) chapter boundaries from exact chapter-opening strings.


The author transcribed the verbatim first sentence of all 70 of the novel's actual
chapters (auxiliary/against_the_day/atd_chapter_first_strings.csv), each already
labelled with its Part. Each sentence is located directly in the plaintext: exact
character offset, no interpolation, no wiki dependency at all. Chapter-to-Part
membership is cross-checked two ways — against the CSV's own Part column, and
independently against the five exact Part-marker offsets already used elsewhere in
the pipeline — before anything is written.
"""
import csv
import json
from locate_quotes import Corpus, normalize_query
from project_paths import ATD_AUX, CHAPTER_INDEX, PRIMARY_FILES, SEGMENTATION_DIR

CSV_PATH = ATD_AUX / "atd_chapter_first_strings.csv"
PART_TITLES = ["The Light Over the Ranges", "Iceland Spar", "Bilocations",
               "Against the Day", "Rue du Départ"]
PARTS_WORDS = ["One", "Two", "Three", "Four", "Five"]


def load_rows():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [r["Chapter"] for r in rows] != [str(i) for i in range(1, len(rows) + 1)]:
        raise SystemExit("CSV chapter numbers are not a contiguous 1..N sequence")
    return rows


def part_boundaries(text):
    """Exact offsets of the five Part markers, plus a sentinel end-of-book offset."""
    offs, bounds = [], {}
    off = 0
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if s in PARTS_WORDS and PARTS_WORDS.index(s) == len(bounds):
            bounds[PARTS_WORDS.index(s)] = off
        off += len(line)
    if len(bounds) != 5:
        raise SystemExit(f"expected 5 exact Part markers, found {len(bounds)}")
    offs = [bounds[i] for i in range(5)]
    offs.append(len(text))
    return offs


def locate_chapters(c, rows):
    """Return {chapter: exact_offset}, raising on any missing/ambiguous sentence."""
    norm, imap = c.norm["P07"], c.imap["P07"]
    offsets = {}
    for row in rows:
        ch = int(row["Chapter"])
        sentence = row["First string"]
        qn = normalize_query(sentence)
        first = norm.find(qn)
        if first < 0:
            raise SystemExit(f"chapter {ch} opening not found verbatim in P07.txt: {sentence[:70]!r}...")
        if norm.find(qn, first + 1) >= 0:
            raise SystemExit(f"chapter {ch} opening matches more than once in P07.txt: {sentence[:70]!r}...")
        offsets[ch] = imap[first]
    ordered = [offsets[ch] for ch in sorted(offsets)]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise SystemExit(f"chapter openings are not strictly increasing through the text: {ordered}")
    return offsets


def part_of_offset(offset, poffs):
    for i in range(5):
        if poffs[i] <= offset < poffs[i + 1]:
            return i + 1
    return 5


def main():
    c = Corpus()
    text = c.texts["P07"]
    rows = load_rows()
    offsets = locate_chapters(c, rows)
    poffs = part_boundaries(text)

    # Cross-check CSV part labels against exact text offsets to ensure consistency.
    label_to_part = {f"Part {['One', 'Two', 'Three', 'Four', 'Five'][i]}: {t}": i + 1
                      for i, t in enumerate(PART_TITLES)}
    mismatches = []
    for row in rows:
        ch = int(row["Chapter"])
        expected_part = label_to_part.get(row["Section"])
        if expected_part is None:
            raise SystemExit(f"chapter {ch}: unrecognised Section label {row['Section']!r}")
        actual_part = part_of_offset(offsets[ch], poffs)
        if expected_part != actual_part:
            mismatches.append((ch, expected_part, actual_part))
    if mismatches:
        for ch, exp, act in mismatches:
            print(f"  MISMATCH ch{ch}: CSV says Part {exp}, offset falls in Part {act}")
        raise SystemExit(f"{len(mismatches)} chapter(s) disagree with their CSV Part label; aborting")

    secs = []
    for ch in sorted(offsets):
        part_n = part_of_offset(offsets[ch], poffs)
        secs.append({
            "n": ch,
            "label": f"Chapter {ch}",
            "part": f"Part {PARTS_WORDS[part_n - 1]}: {PART_TITLES[part_n - 1]}",
            "start": offsets[ch],
            "end": None,  # filled below
            "approx": False,
        })
    secs[0]["start"] = 0  # include any leading whitespace/title material
    for i in range(len(secs) - 1):
        secs[i]["end"] = secs[i + 1]["start"]
    secs[-1]["end"] = len(text)

    # Persist chapter boundaries in both JSON and XML formats.
    SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    (SEGMENTATION_DIR / "P07.sections.json").write_text(
        json.dumps(secs, ensure_ascii=False, indent=2), encoding="utf-8")

    def xa(s): return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    def xt(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<work id="P07" title="Against the Day">']
    for s in secs:
        out.append(f'<section n="{s["n"]}" label="{xa(s["label"])}" part="{xa(s["part"])}">')
        out.append(xt(text[s["start"]:s["end"]]))
        out.append("</section>")
    out.append("</work>")
    (SEGMENTATION_DIR / "P07.xml").write_text("\n".join(out), encoding="utf-8")

    per_part = {p: sum(1 for s in secs if s["part"].startswith(f"Part {PARTS_WORDS[p-1]}:")) for p in range(1, 6)}
    print(f"chapter openings located: {len(offsets)}/{len(rows)}, all unique and strictly increasing")
    print(f"chapters per Part (cross-checked against CSV labels and exact Part markers): {per_part}")
    print("chapter start offsets:", [s["start"] for s in secs])

    idx_path = CHAPTER_INDEX
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    idx["P07"] = {"title": "Against the Day", "n_sections": len(secs),
                  "offset_only": False, "chapters_approximate": False}
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    return secs


if __name__ == "__main__":
    main()
