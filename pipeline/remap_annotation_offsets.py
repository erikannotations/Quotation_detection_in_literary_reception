#!/usr/bin/env python3
"""Remap an annotation file from one revision of an article text to another.

Written 30 July 2026 for a specific, foreseeable problem: the second annotator's package
carries an article text that has since been corrected here, so the file she returns is in
the OLD coordinate system. Comparing it against the author's file without remapping would
manufacture span disagreements out of arithmetic and depress the reported inter-annotator
agreement — the one number the independent subset exists to produce.


Usage:

    python3 remap_annotation_offsets.py RETURNED.annotation.json \\
        --from-text legacy/sezer_ellipsis_correction/pre_fix_article.txt \\
        --to-text ../annotation_workspace/03_non_orbit_cc_articles/2021_sezer....txt

Add --dry-run first. It reports what would move and changes nothing.
"""
import argparse
import difflib
import hashlib
import json
import re
import shutil
from pathlib import Path

from review_annotations import metadata_normalized_body, update_context


def offset_map(old_body, new_body):
    """Map every old body offset (0..len inclusive) to its offset in the new body."""
    # Build cumulative offset mapping by tracking insertions and deletions through difflib opcodes.
    mapping = list(range(len(old_body) + 1))
    delta = 0
    for _tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, old_body, new_body, autojunk=False).get_opcodes():
        for offset in range(i1, i2):
            mapping[offset] = offset + delta
        delta += (j2 - j1) - (i2 - i1)
        for offset in range(i2, len(old_body) + 1):
            mapping[offset] = offset + delta
    return mapping


def content_words(text):
    """Alphanumeric content only.

    This is the invariant a remap must preserve. Whitespace and punctuation are exactly
    what a text correction changes — rejoining ". . ." alters both — so comparing them
    would reject a correct remap. The words must not change: if they do, the span has
    been mapped somewhere else, which is the failure worth stopping for.
    """
    return re.sub(r"[^0-9a-zA-Z]+", " ", text).lower().split()


def remap(data, old_body, new_body, mapping):
    """Remap every span in place. Returns (moved, requoted, repunctuated)."""
    # Remap all candidate and manual spans using the offset mapping, preserving word content.
    moved = requoted = repunctuated = 0
    rows = data.get("candidates", []) + data.get("manually_added_spans", [])
    for row in rows:
        old_start, old_end = row["start_char"], row["end_char"]
        if not (0 <= old_start < old_end <= len(old_body)):
            raise SystemExit(
                f"{row.get('annotation_id')}: offsets {old_start}-{old_end} do not fit "
                f"the --from-text ({len(old_body)} chars). Wrong source revision?")
        start, end = mapping[old_start], mapping[old_end]
        quote = new_body[start:end]
        if content_words(quote) != content_words(row["quote"]):
            raise SystemExit(
                f"{row.get('annotation_id')}: remapped span covers different words.\n"
                f"  stored:   {row['quote'][:90]!r}\n"
                f"  remapped: {quote[:90]!r}")
        moved += (start, end) != (old_start, old_end)
        if quote != row["quote"]:
            requoted += 1
            if re.sub(r"\s+", " ", quote) != re.sub(r"\s+", " ", row["quote"]):
                repunctuated += 1
        row["start_char"], row["end_char"] = start, end
        row["quote"] = quote
        update_context(row, new_body)
    return moved, requoted, repunctuated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", help="the returned .annotation.json to remap")
    parser.add_argument("--from-text", required=True,
                        help="article text the annotation was made against")
    parser.add_argument("--to-text", required=True,
                        help="article text now in use")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and write nothing")
    args = parser.parse_args()

    old_body = metadata_normalized_body(
        Path(args.from_text).read_text(encoding="utf-8"))
    new_body = metadata_normalized_body(Path(args.to_text).read_text(encoding="utf-8"))
    path = Path(args.annotation)
    data = json.loads(path.read_text(encoding="utf-8"))

    stored = data.get("body_sha256")
    old_hash = hashlib.sha256(old_body.encode("utf-8")).hexdigest()
    new_hash = hashlib.sha256(new_body.encode("utf-8")).hexdigest()
    if stored == new_hash:
        print("Already on the current text: nothing to remap.")
        return
    if stored != old_hash:
        raise SystemExit(
            "The annotation's body_sha256 matches neither text.\n"
            f"  stored:    {stored}\n  --from-text: {old_hash}\n  --to-text:   {new_hash}\n"
            "Identify the revision it was made against before remapping.")

    moved, requoted, repunctuated = remap(
        data, old_body, new_body, offset_map(old_body, new_body))
    data["body_sha256"] = new_hash

    print(f"{path.name}: {moved} spans moved, {requoted} quotes rewritten "
          f"({repunctuated} beyond whitespace)")
    if args.dry_run:
        print("dry run: nothing written")
        return
    backup = path.with_suffix(path.suffix + ".pre_remap")
    shutil.copy2(path, backup)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written; previous version kept at {backup.name}")


if __name__ == "__main__":
    main()
