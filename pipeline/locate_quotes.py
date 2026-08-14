#!/usr/bin/env python3
"""Locate candidate Pynchon quotations within the primary corpus.

Given a quotation string (as a scholar reproduced it), find where — if anywhere —
it occurs in the 10 primary texts, and report novel + chapter/episode + char offset
+ a similarity score. Built to tolerate the things scholars do to quotes:
  - ellipses (… / ... / [...]) marking omitted text  -> match the fragments
  - changed case, curly vs straight quotes, dashes, [sic]/[bracketed] insertions
  - whitespace/line-break differences

Matching strategy: normalise text and query to a canonical form with a char-offset
map back to the original. For each ellipsis-delimited fragment, search shingles from
throughout the fragment rather than only its prefix. Every hit estimates a possible
fragment start, which is scored against the surrounding source window with difflib.
All near-best source works are retained so repeated passages and cross-work reuse are
reported as ambiguity rather than silently resolved by corpus order.
"""
import json
import re
import sys
import difflib
from project_paths import CHAPTER_INDEX, PRIMARY_FILES, SEGMENTATION_DIR
PIDS = [f"P0{i}" for i in range(1, 10)] + ["P10"]
NEAR_MARGIN = 0.02

ELLIPSIS = re.compile(r"\s*(?:\.\s*\.\s*\.+|…|\[\s*\.\.\.\s*\]|\[\s*…\s*\])\s*")
# Square brackets are dropped *without* a gap, so an editorial case-change like
# "[w]here" rejoins to "where". Quotes and apostrophes instead become separators
# (a space) so an elision a scholar reproduced without a following space —
# "tho'others" vs the source's "tho' others" — still aligns. Both query and corpus
# are normalised the same way, so contractions ("don't") stay symmetric.
DROP_CHARS = "[]"

# A bracketed run standing as a WHOLE word is an editorial *insertion* — a word the
# scholar added that the source does not contain ("enormous eyes, lambent [and]
# extravagantly lashed", where Lot 49 has a comma). Dropping only the brackets, as
# DROP_CHARS does, leaves "and" in the query as a token the source lacks, and the
# fragment then fails to align at all — not a low score, no match. That is the right
# treatment for an *alteration* inside a word ("[w]here", "improvise[s]"), where the
# bracketed characters replace the original and must be kept, so the two cases are
# distinguished by whether the brackets stand alone. Insertions are retried removed
# rather than always removed, because a bracketed whole word is occasionally a
# clarification the source does contain.
# Both sides matter. Requiring whitespace (or a string edge) BEFORE the opening bracket
# excludes "improvise[s]"; requiring it AFTER the closing bracket excludes "[w]here",
# which would otherwise be stripped to "here" whenever it opened a quotation.
BRACKET_INSERTION = re.compile(r"(?:(?<=\s)|(?<=^))\[[^\[\]]+\](?=\s|$)\s*")


def drop_bracket_insertions(quote):
    """Remove space-delimited bracketed runs: the editorial-insertion reading.

    Ellipsis markers written as "[...]" are left to ELLIPSIS, which splits on them.
    """
    without = BRACKET_INSERTION.sub("", quote)
    return re.sub(r"\s+", " ", without).strip() or quote


def normalize_with_map(text):
    """Return (norm_str, idx_map) where idx_map[i] = original offset of norm[i]."""
    norm, idx = [], []
    prev_space = False
    for i, ch in enumerate(text):
        c = ch.lower()
        if c in DROP_CHARS:
            continue
        if c.isalnum():
            norm.append(c)
            idx.append(i)
            prev_space = False
        else:
            # any punctuation / whitespace / dash collapses to a single space
            if not prev_space:
                norm.append(" ")
                idx.append(i)
                prev_space = True
    return "".join(norm), idx


def normalize_query(q):
    n, _ = normalize_with_map(q)
    return re.sub(r"\s+", " ", n).strip()


class Corpus:
    def __init__(self):
        self.texts, self.norm, self.imap, self.secs = {}, {}, {}, {}
        for pid in PIDS:
            t = PRIMARY_FILES[pid].read_text(encoding="utf-8")
            self.texts[pid] = t
            n, m = normalize_with_map(t)
            self.norm[pid] = n
            self.imap[pid] = m
            self.secs[pid] = json.loads(
                (SEGMENTATION_DIR / f"{pid}.sections.json").read_text(encoding="utf-8")
            )
        self.titles = {p: v["title"] for p, v in
                       json.loads(CHAPTER_INDEX.read_text(encoding="utf-8")).items()}

    def work_title(self, pid, section):
        """Return the bibliographic work represented by a located section."""
        if pid == "P01":
            if section["label"] == "Mortality and Mercy in Vienna":
                return section["label"]
            return "Slow Learner"
        return self.titles[pid]

    def section_at(self, pid, offset):
        for s in self.secs[pid]:
            if s["start"] <= offset < s["end"]:
                return s
        return self.secs[pid][-1]

    @staticmethod
    def _word_spans(text):
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    @staticmethod
    def _all_positions(text, needle, limit=200):
        """Return non-overlapping occurrences, bounded for pathological common text."""
        out, start = [], 0
        while len(out) < limit:
            pos = text.find(needle, start)
            if pos < 0:
                break
            out.append(pos)
            start = pos + max(1, len(needle))
        return out

    def _fragment_candidates(self, frag_norm, min_anchor_words=5):
        """Return scored possible starts for one normalized quotation fragment."""
        spans = self._word_spans(frag_norm)
        n_words = len(spans)
        if n_words < min_anchor_words:
            return []

        # Longer anchors are more discriminating. Sliding across the fragment makes
        # the matcher robust to an alteration near either edge.
        sizes = []
        for size in (n_words, 16, 12, 10, 8, 6, 5):
            size = min(size, n_words)
            if size >= min_anchor_words and size not in sizes:
                sizes.append(size)

        seen_starts, candidates = set(), []
        for size in sizes:
            for wi in range(0, n_words - size + 1):
                q0 = spans[wi][0]
                q1 = spans[wi + size - 1][1]
                shingle = frag_norm[q0:q1]
                for pid in PIDS:
                    for pos in self._all_positions(self.norm[pid], shingle):
                        est = max(0, pos - q0)
                        key = (pid, est)
                        if key in seen_starts:
                            continue
                        seen_starts.add(key)
                        win = self.norm[pid][est:est + len(frag_norm)]
                        # autojunk must be off: with windows >=200 chars difflib
                        # treats every common letter as junk and scores near-identical
                        # block quotes at ~0.0, so long quotations never located.
                        score = difflib.SequenceMatcher(
                            None, frag_norm, win, autojunk=False
                        ).ratio()
                        candidates.append({
                            "pid": pid,
                            "norm_pos": est,
                            "score": score,
                            "anchor_words": size,
                        })
            # A full-length or long near-unique anchor is enough; avoid generating a
            # large tail of weaker versions of the same starts.
            if candidates and size >= min(10, n_words):
                break
        return candidates

    def locate_all(self, quote, min_anchor_words=5, score_floor=0.50, limit=20):
        """Return ranked source locations instead of silently selecting the first hit.

        The editorial-insertion reading is scored here rather than in ``locate``
        so that every caller gets it.
        """
        results = self._locate_reading(quote, min_anchor_words, score_floor, limit)
        without_insertions = drop_bracket_insertions(quote)
        if without_insertions != quote:
            # Score both readings and keep the better. Trying rather than assuming
            # means a bracketed word the source genuinely contains is unaffected: that
            # reading simply wins on score.
            alternative = self._locate_reading(
                without_insertions, min_anchor_words, score_floor, limit)
            best_new = alternative[0]["score"] if alternative else -1
            best_old = results[0]["score"] if results else -1
            if best_new > best_old:
                results = alternative
        return results

    def _locate_reading(self, quote, min_anchor_words=5, score_floor=0.50, limit=20):
        """Rank source locations for one exact reading of a quotation."""
        frags = [normalize_query(f) for f in ELLIPSIS.split(quote)]
        frags = [f for f in frags if len(f.split()) >= min_anchor_words]
        if not frags:
            return []

        candidates = []
        for frag in frags:
            for row in self._fragment_candidates(frag, min_anchor_words):
                if row["score"] >= score_floor:
                    row = dict(row)
                    row["fragment"] = frag
                    candidates.append(row)
        if not candidates:
            return []

        # Keep the strongest evidence for a given estimated source start.
        dedup = {}
        for row in candidates:
            key = (row["pid"], row["norm_pos"])
            old = dedup.get(key)
            if old is None or (row["score"], row["anchor_words"]) > (old["score"], old["anchor_words"]):
                dedup[key] = row

        ranked = sorted(
            dedup.values(),
            key=lambda r: (-r["score"], -r["anchor_words"], r["pid"], r["norm_pos"]),
        )[:limit]
        out = []
        for row in ranked:
            pid, pos = row["pid"], row["norm_pos"]
            # `norm_pos` is an ESTIMATE: the anchor's source position minus the
            # anchor's offset within the query. When the quotation's opening words
            # are altered and normalize longer than the source's, that extrapolation
            # lands BEFORE the quotation's true start — and for a quotation that
            # opens a chapter it lands before the chapter boundary, so section_at
            # returned the previous chapter at a fully confirming score.
            # Fixed because the error is silent and confidently scored, not
            # because it fires often. Recover a VERIFIED position instead: the
            # first substantial run difflib actually matched.
            frag = row["fragment"]
            win = self.norm[pid][pos:pos + len(frag)]
            blocks = [b for b in difflib.SequenceMatcher(
                None, frag, win, autojunk=False).get_matching_blocks() if b.size]
            # Anchor on the first SUBSTANTIAL matched run, not merely the first
            # matched character: a single incidental letter can match at window
            # offset 0 and defeat the correction entirely.
            if blocks:
                longest = max(b.size for b in blocks)
                anchor = next(b for b in blocks if b.size >= min(20, longest))
                # Section comes from a position that provably matched, so it cannot
                # fall outside the quotation. The reported offset is still the
                # quotation's start, back-mapped from that verified anchor.
                verified = pos + anchor.b
                pos = max(0, verified - anchor.a)
                # normalize_with_map collapses a whole run of punctuation and
                # whitespace to ONE space whose recorded offset is the run's FIRST
                # character. A chapter separator ("• • • • • • •") is such a run, so
                # a verified position landing on that space maps back to before the
                # separator — i.e. into the previous chapter. Advance to the first
                # real character so the mapped offset is inside the quotation.
                while (verified < len(self.norm[pid])
                       and self.norm[pid][verified] == " "):
                    verified += 1
            else:
                verified = pos
            orig_off = self.imap[pid][min(pos, len(self.imap[pid]) - 1)]
            sec = self.section_at(
                pid, self.imap[pid][min(verified, len(self.imap[pid]) - 1)])
            snip = self.texts[pid][orig_off:orig_off + 90].replace("\n", " ").strip()
            out.append({
                "novel": self.work_title(pid, sec),
                "pid": pid,
                "section": sec["label"],
                "section_n": sec["n"],
                "part": sec["part"],
                "char_offset": orig_off,
                "score": round(row["score"], 3),
                "anchor_words": row["anchor_words"],
                "n_fragments": len(frags),
                "snippet": snip,
            })
        return out

    def locate(self, quote, min_anchor_words=5):
        """Backward-compatible best location, with ambiguity metadata attached."""
        locations = self.locate_all(quote, min_anchor_words=min_anchor_words)
        if not locations:
            return None
        best = dict(locations[0])
        near = [r for r in locations if r["score"] >= best["score"] - NEAR_MARGIN]
        best["n_near_locations"] = len(near)
        best["candidate_pids"] = sorted({r["pid"] for r in near})
        best["source_ambiguous"] = len(best["candidate_pids"]) > 1
        return best


# ---- self-test: pull real snippets, perturb them, confirm relocation ----
def _selftest():
    c = Corpus()
    cases = []
    # P03 CoL — a sentence from chapter 1 region
    for pid, approx_off, label in [("P03", 200, "CoL"), ("P04", 300, "GR ep1"),
                                   ("P06", 5000, "M&D"), ("P08", 400, "IV")]:
        t = c.texts[pid]
        # grab a ~25-word verbatim slice starting at a word boundary
        start = t.find(" ", approx_off) + 1
        slice_words = t[start:start + 220].split()
        verb = " ".join(slice_words[:24])
        cases.append((label + " verbatim", verb))
        # ellipsis variant: drop the middle
        w = verb.split()
        ell = " ".join(w[:6]) + " . . . " + " ".join(w[-6:])
        cases.append((label + " +ellipsis", ell))
        # case + [sic] variant
        cases.append((label + " +CASE/[sic]", verb.upper().replace(" ", " ", 1) + " [sic]"))

    print(f"{'case':22} {'->':2} {'novel':22} {'section':16} {'score':>5}")
    print("-" * 78)
    for name, q in cases:
        r = c.locate(q)
        if r:
            print(f"{name:22} -> {r['novel'][:22]:22} {str(r['section'])[:16]:16} {r['score']:>5}")
        else:
            print(f"{name:22} -> NO MATCH")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    else:
        c = Corpus()
        q = sys.stdin.read()
        print(json.dumps(c.locate(q), ensure_ascii=False, indent=2))
