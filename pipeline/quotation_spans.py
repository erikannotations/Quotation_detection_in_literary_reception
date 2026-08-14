"""Shared marked-quotation extraction helpers for baselines and annotation setup."""
import re

SEP_RE = re.compile(r"^={10,}\s*$", re.M)

# Straight single quotes were originally excluded as indistinguishable from
# apostrophes. On 23 July 2026 a CONSERVATIVE straight-single tier was added
# (author decision; see the audit log): opener must follow whitespace/start of
# line, closer must precede whitespace or punctuation, interior may not contain
# another straight single quote (so possessives and contractions inside a
# quotation end the match — conservative by design), plus the usual five-word
# minimum and paragraph-boundary guard. Empirically this yields 8 extra
# candidates across the whole 16-article sample and no apostrophe noise. Block
# quotations remain invisible to all tiers; the human full-article read is
# still required for those.
def marked_pattern(opening, closing):
    """Match paired marks without wandering into another paragraph.

    OCR/plaintext frequently contains an unmatched mark. A bare ``.+?`` then
    swallows pages until it finds a compatible closer, creating a spurious giant
    quotation. Real inline marked quotations may wrap across one line, but not a
    blank-line paragraph boundary in the corpus representation.
    """
    return re.compile(
        re.escape(opening)
        + r"(?P<quote>(?:(?!" + re.escape(closing) + r"|\n\s*\n).)+)"
        + re.escape(closing),
        re.S,
    )


MARK_PATTERNS = [
    ("curly_double", marked_pattern("“", "”")),
    ("straight_double", marked_pattern('"', '"')),
    ("curly_single", marked_pattern("‘", "’")),
    ("straight_single", re.compile(
        r"(?:(?<=\s)|^)'"
        r"(?P<quote>(?:(?!\n\s*\n)[^']){15,400}?)"
        r"'(?=[\s,.;:!?)\]])",
        re.S | re.M,
    )),
]


def article_body(text):
    parts = SEP_RE.split(text, maxsplit=1)
    body = parts[1] if len(parts) == 2 else text
    return body.strip()


#: The frozen target's length floor. Kept here, beside ``word_count``, so that every
#: stage applying the rule — gold compilation, inter-annotator agreement, evaluation —
#: derives it from one definition rather than repeating the literal 5. The two must
#: travel together: the rule is only meaningful given this tokenisation, and the
#: 23 July 2026 bracket fix changed which spans satisfy it.
MIN_QUOTE_WORDS = 5


def under_minimum(quote):
    """True when a span is too short to be in scope for the frozen task.

    A span shorter than the minimum cannot be gold, cannot match gold, and is not an
    answer to the task, so it is excluded wherever it appears rather than scored.
    """
    return word_count(quote or "") < MIN_QUOTE_WORDS


def word_count(text):
    # Bracketed editorial alterations ("[t]he", "improvise[s]") belong to the
    # word they modify — annotators count naturally, so "[t]he death of magic"
    # is four words, not five. Strip the brackets before tokenizing; "[…]" and
    # multi-word insertions are untouched (\w only) and count as before.
    text = re.sub(r"\[(\w+)\]", r"\1", text)
    return len(re.findall(r"\b\w+(?:[’'-]\w+)*\b", text, re.UNICODE))


def find_quote_offsets(text, quote, start_hint=None, occurrence_index=0):
    """Locate a detector-returned quotation in an article body.

    Exact matching is attempted first, then a whitespace-flexible, case-insensitive
    match. Returns ``(start, end, occurrence_count)`` or ``(None, None, 0)``. When a
    repeated quotation has no hint, the first occurrence is retained but the count
    makes the ambiguity explicit for downstream review.
    """
    quote = (quote or "").strip()
    if not quote:
        return None, None, 0

    def on_word_boundaries(begin, finish):
        """Reject a hit that starts or ends inside a word.

        A bare find() matches "he said five words here" inside "t*he said five
        words here*", returning offsets that begin mid-word. Nothing downstream
        re-checks them, and the hybrid combiner then makes containment decisions
        from the wrong span.
        """
        before = text[begin - 1] if begin > 0 else " "
        after = text[finish] if finish < len(text) else " "
        return not ((text[begin].isalnum() and before.isalnum())
                    or (text[finish - 1].isalnum() and after.isalnum()))

    exact = []
    start = 0
    while True:
        pos = text.find(quote, start)
        if pos < 0:
            break
        if on_word_boundaries(pos, pos + len(quote)):
            exact.append((pos, pos + len(quote)))
        start = pos + max(1, len(quote))
    matches = exact

    if not matches:
        pieces = re.split(r"\s+", quote)
        pattern = re.compile(r"\s+".join(re.escape(piece) for piece in pieces), re.I)
        matches = [(m.start(), m.end()) for m in pattern.finditer(text)
                   if on_word_boundaries(m.start(), m.end())]
    if not matches:
        # Punctuation-normalised fallback, sharing the locator's normalisation so a
        # detector-returned quote whose apostrophes, dashes, ellipses, or brackets
        # differ from the article's typography still aligns to exact offsets.
        from locate_quotes import normalize_query, normalize_with_map

        norm_quote = normalize_query(quote)
        if norm_quote:
            norm_text, imap = normalize_with_map(text)
            pos = norm_text.find(norm_quote)
            while pos >= 0:
                begin, finish = imap[pos], imap[pos + len(norm_quote) - 1] + 1
                if on_word_boundaries(begin, finish):
                    matches.append((begin, finish))
                pos = norm_text.find(norm_quote, pos + 1)
    if not matches:
        return None, None, 0

    if start_hint is None:
        chosen = matches[min(max(0, occurrence_index), len(matches) - 1)]
    else:
        chosen = min(matches, key=lambda span: abs(span[0] - start_hint))
    return chosen[0], chosen[1], len(matches)


def extract_marked_spans(text, min_words=5):
    """Return non-overlapping quote-mark spans, with body-relative offsets."""
    spans = []
    for form, pattern in MARK_PATTERNS:
        for match in pattern.finditer(text):
            raw_quote = match.group("quote")
            # Keep offsets synchronized when whitespace occurs just inside the
            # quotation marks. Annotation completion validates that the stored quote
            # is exactly ``text[start_char:end_char]``.
            left_trim = len(raw_quote) - len(raw_quote.lstrip())
            right_trim = len(raw_quote) - len(raw_quote.rstrip())
            start = match.start("quote") + left_trim
            end = match.end("quote") - right_trim
            quote = text[start:end]
            if word_count(quote) < min_words:
                continue
            spans.append({
                "start_char": start,
                "end_char": end,
                "quote": quote,
                "quotation_form": form,
            })

    # Nested quotation marks can produce overlapping candidates. Prefer the longer
    # enclosing span and retain disjoint spans in article order.
    chosen = []
    for row in sorted(spans, key=lambda r: (r["start_char"], -(r["end_char"] - r["start_char"]))):
        if any(row["start_char"] < old["end_char"] and old["start_char"] < row["end_char"]
               for old in chosen):
            continue
        chosen.append(row)
    return sorted(chosen, key=lambda r: r["start_char"])
