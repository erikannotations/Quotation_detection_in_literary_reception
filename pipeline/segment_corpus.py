#!/usr/bin/env python3
"""Segment the Pynchon primary corpus into chapters/episodes.

Originals are left untouched so character offsets stay stable for the
quotation-location step. For each text we emit:
  - derived/segmentation/<id>.sections.json : ordered character offsets
  - derived/segmentation/<id>.xml : text wrapped in <section> elements (local artifact)
and a combined chapter index plus a console report of detected section counts.

Segmentation markers differ per work. Vineland's exact chapter markers are absent
from its plaintext, so this script emits an offset-only placeholder which
`vine_build_chapters.py` subsequently replaces with anchored approximate chapters.
"""
import json
import re
from project_paths import CHAPTER_INDEX, PRIMARY_FILES, SEGMENTATION_DIR

# title shown in output; matches the (ignored) metadata roughly but set here explicitly
TITLES = {
    "P01": "Slow Learner (stories)", "P02": "V.", "P03": "The Crying of Lot 49",
    "P04": "Gravity's Rainbow", "P05": "Vineland", "P06": "Mason & Dixon",
    "P07": "Against the Day", "P08": "Inherent Vice", "P09": "Bleeding Edge",
    "P10": "Shadow Ticket",
}

def line_offsets(text):
    """Yield (line_text_without_newline, start_char_offset) for each line."""
    off = 0
    for line in text.splitlines(keepends=True):
        yield line.rstrip("\n"), off
        off += len(line)


def sections_from_boundaries(text, bounds):
    """bounds: sorted list of (start_offset, label, part). Returns section dicts."""
    if bounds and bounds[0][0] > 0:
        # Title pages and epigraphs belong to the first addressable section. Leaving
        # a gap here also makes section lookup before the first heading fall through
        # to the final section, which is actively misleading.
        _start, label, part = bounds[0]
        bounds = [(0, label, part)] + list(bounds[1:])
    secs = []
    for i, (start, label, part) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        secs.append({"n": i + 1, "label": label, "part": part,
                     "start": start, "end": end})
    return secs


def seg_numeric(text, lo, hi):
    """Indented/standalone arabic numerals lo..hi as chapter heads (in order)."""
    bounds = []
    expect = lo
    for line, off in line_offsets(text):
        s = line.strip()
        if s.isdigit() and int(s) == expect and expect <= hi:
            bounds.append((off, f"Chapter {expect}", None))
            expect += 1
    return bounds


def seg_caps_words(text, words):
    """Standalone spelled-out chapter words (e.g. ONE, TWO) — case-insensitive."""
    wantset = {w.upper() for w in words}
    seen = []
    bounds = []
    for line, off in line_offsets(text):
        s = line.strip().upper().rstrip(".")
        if s in wantset and s not in seen:
            seen.append(s)
            bounds.append((off, f"Chapter {words[len(seen)-1]}", None))
    return bounds


def seg_gr(text):
    """Gravity's Rainbow: episodes start at the book opening and after each spark.

    The text has 73 spark dividers (• • • • • • •). Canonically GR has 73
    episodes; the divider after the final episode yields an empty trailing span
    which we drop. Each episode is tagged with the Part it falls in, by offset.
    """
    PART_TITLES = ["Beyond the Zero", "Un Perm' au Casino Hermann Goering",
                   "In the Zone", "The Counterforce"]
    spark = re.compile(r"^[•·\*\s]{5,}$")
    lines = list(line_offsets(text))
    part_offsets = []  # (offset, label) for the 4 part headers
    spark_offsets = []
    for idx, (line, off) in enumerate(lines):
        s = line.strip()
        if s.isdigit() and int(s) in (1, 2, 3, 4):
            nxt = lines[idx + 2][0].strip() if idx + 2 < len(lines) else ""
            if PART_TITLES[int(s) - 1].split()[0] in nxt:
                part_offsets.append((off, f"Part {int(s)}: {PART_TITLES[int(s)-1]}"))
        elif spark.match(line) and "•" in line:
            spark_offsets.append(off)

    def part_for(offset):
        cur = part_offsets[0][1] if part_offsets else None  # offset 0 -> Part 1
        for poff, plabel in part_offsets:
            if poff <= offset:
                cur = plabel
        return cur

    # Episode 1 = Part-1 epigraph + "A screaming comes across the sky..." span, so
    # drop the first spark as a boundary (it only ends the epigraph front-matter).
    starts = [0] + spark_offsets[1:]
    bounds = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        if not text[start:end].strip():      # drop empty trailing span
            continue
        bounds.append((start, f"Episode {len(bounds)+1}", part_for(start)))
    return bounds


def seg_atd(text):
    """Against the Day: 5 spelled parts (part-level locus; finer position by offset)."""
    PARTS = ["One", "Two", "Three", "Four", "Five"]
    PART_TITLES = ["The Light Over the Ranges", "Iceland Spar", "Bilocations",
                   "Against the Day", "Rue du Depart"]
    bounds = []
    for line, off in line_offsets(text):
        s = line.strip()
        if s in PARTS and len(bounds) == PARTS.index(s):
            p = PARTS.index(s)
            bounds.append((off, f"Part {p+1}: {PART_TITLES[p]}", None))
    return bounds


def seg_short_fiction(text):
    """Six short stories: one uncollected story followed by the Slow Learner five."""
    titles = ["The Small Rain", "Low-Lands", "Low-lands", "Entropy",
              "Under the Rose", "The Secret Integration"]
    bounds = [(0, "Mortality and Mercy in Vienna", None)]
    seen = set()
    for line, off in line_offsets(text):
        s = line.strip()
        if s in titles and s.lower() not in seen:
            seen.add(s.lower())
            bounds.append((off, s, None))
    return bounds


def seg_v(text):
    """V.: 'Chapter One'..'Chapter Sixteen' headers, then 'Epilogue' (17 sections)."""
    words = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
             "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen"]
    targets = [f"Chapter {w}" for w in words] + ["Epilogue"]
    bounds, expect = [], 0
    for line, off in line_offsets(text):
        if expect < len(targets) and line.strip() == targets[expect]:
            bounds.append((off, targets[expect], None))
            expect += 1
    return bounds


def seg_md(text):
    """Mason & Dixon: 78 standalone-numbered chapters, grouped into 3 named parts.
    (The table of contents at the top uses 'n - n - n' ranges, not standalone digits,
    so it does not interfere with chapter detection.)"""
    def part_of(n):
        if n <= 25:
            return "Part 1: Latitudes and Departures"
        if n <= 73:
            return "Part 2: America"
        return "Part 3: Last Transit"
    bounds, expect = [], 1
    for line, off in line_offsets(text):
        s = line.strip()
        if s.isdigit() and int(s) == expect and expect <= 78:
            bounds.append((off, f"Chapter {expect}", part_of(expect)))
            expect += 1
    return bounds


# per-novel configuration; None -> offset-only (flagged)
def segment(pid, text):
    if pid == "P03":
        return seg_numeric(text, 1, 6), False
    if pid == "P06":
        return seg_md(text), False
    if pid == "P02":
        return seg_v(text), False
    if pid == "P09":
        return seg_numeric(text, 1, 41), False
    if pid == "P10":
        return seg_numeric(text, 1, 39), False
    if pid == "P08":
        return seg_caps_words(text, ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX",
                                     "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN",
                                     "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
                                     "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN",
                                     "TWENTY", "TWENTY-ONE"]), False
    if pid == "P04":
        return seg_gr(text), False
    if pid == "P07":
        return seg_atd(text), False
    if pid == "P01":
        return seg_short_fiction(text), False
    # P05 (Vineland): no reliable in-text markers (chapters built separately by
    # page-interpolation in vine_build_chapters.py)
    return [(0, "(whole text — no chapter markers)", None)], True


def main():
    SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    index = {}
    print(f"{'id':4} {'title':24} {'sections':>8}  flag")
    print("-" * 55)
    for pid, txt in PRIMARY_FILES.items():
        if not txt.exists():
            raise FileNotFoundError(f"missing primary text for {pid}: {txt}")
        text = txt.read_text(encoding="utf-8")
        bounds, offset_only = segment(pid, text)
        bounds = sorted(bounds)
        secs = sections_from_boundaries(text, bounds)
        (SEGMENTATION_DIR / f"{pid}.sections.json").write_text(
            json.dumps(secs, ensure_ascii=False, indent=2), encoding="utf-8")
        # XML artifact
        parts = ['<?xml version="1.0" encoding="UTF-8"?>',
                 f'<work id="{pid}" title="{TITLES.get(pid, pid)}">']
        for s in secs:
            seg_text = text[s["start"]:s["end"]]
            attrs = f'n="{s["n"]}" label="{xml_attr(s["label"])}"'
            if s["part"]:
                attrs += f' part="{xml_attr(s["part"])}"'
            parts.append(f'<section {attrs}>')
            parts.append(xml_text(seg_text))
            parts.append('</section>')
        parts.append('</work>')
        (SEGMENTATION_DIR / f"{pid}.xml").write_text("\n".join(parts), encoding="utf-8")
        index[pid] = {"title": TITLES.get(pid, pid), "n_sections": len(secs),
                      "offset_only": offset_only}
        flag = "OFFSET-ONLY" if offset_only else ""
        print(f"{pid:4} {TITLES.get(pid, pid)[:24]:24} {len(secs):>8}  {flag}")
    CHAPTER_INDEX.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def xml_attr(s):
    return (s.replace("&", "&amp;").replace('"', "&quot;")
             .replace("<", "&lt;").replace(">", "&gt;"))


def xml_text(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
