# Annotation interface guide

For day-to-day annotation, use the project-root `annotation_workspace/Annotation
Manual.html` and its two `.command` launchers. The visible workspace contains nothing
else annotators need. The present file remains the pipeline-level reference for the
terminal interface.

The interface reviews automatically suggested spans without treating them as gold. It
autosaves every confirmed change, retains the immediately previous file as
`*.annotation.json.bak`, and checks that each template still belongs to the exact CORPUS
article version recorded in the sample manifest.

## Start with development articles

From the project root:

```sh
python3 "Pynchon corpus/review_annotations.py"
```

The menu lists only the four development articles. To open the shortest development
article directly:

```sh
python3 "Pynchon corpus/review_annotations.py" --article mentz --reviewer author
```

List progress without entering the interface:

```sh
python3 "Pynchon corpus/review_annotations.py" --list
```

`--split` (default `development`) only filters which of the files already in
`--annotations` are listed; there is no lock. Which documents an annotator can reach at
all is decided by which annotation files sit in that folder. The held-out test
templates are kept out of both annotators' working folders (in
`annotation_workspace/.staged_until_freeze/`) until models, prompts, thresholds, and
hybrid rules have been frozen; see `paper/annotation_administration_2026-07-18.md` for
the restore procedure.

## Annotation scope

The main task includes marked direct quotations of at least five words from Pynchon's
fiction in the ten local primary inputs. It excludes titles, scare quotations,
quotations of critics or other authors, Pynchon nonfiction, allusion, paraphrase, and
summary.

**Nested quotation (rule added 2 August 2026).** A Pynchon quotation appearing *inside* a
quotation of another scholar — the article block-quotes a critic, and the critic's text
contains a passage of Pynchon — is **out of scope**, category `other_author_or_critic`.
The span the article marks is the critic's; the selection of the Pynchon passage was
theirs, not this article's. The rule is a consistency decision rather than a principled
one: nested quotations were never systematically sought across the corpus, so admitting
the ones that happen to surface would bias the gold toward wherever we looked. For a
study of circulation rather than of selection the opposite ruling would be defensible.
Encountered once in the sixteen-document sample (Wallhead quoting Olster quoting
*Mason & Dixon*).

**Quotations in reference lists and cited titles.** Scholars often title their work with a
phrase from the novel, so bibliographies contain genuine Pynchon text that is not
quotation *by this article*. These are out of scope, category `title_or_scare_quote`.
The same applies to a cited book's title appearing in running prose.

Marking is judged against the published article; consult the published PDF where
anything is unclear. The corpus plaintext preserves block-quotation formatting, so what
you see in the app matches the published marking. Set each in-scope span's quotation
form (e.g. `block`) — it is reported descriptively.

For every suggested annotation:

- `y` marks an in-scope quotation. The locator supplies ranked source suggestions, but
  the annotator must confirm one rather than treating it as a label. Accepting a
  suggestion (Enter, or its list number) confirms its work *and* its chapter/episode
  together; typing a PID directly instead additionally prompts for the chapter/episode
  number, since that path is the rare unconfirmed-source fallback. Leave the source or
  its chapter/episode unresolved if uncertain — never force a choice or downgrade to an
  exclusion to satisfy the completion check. Unresolved spans block completion by
  design; they are settled at adjudication by a manual search of the primary texts (see
  the adjudication rules in `paper/annotation_administration_2026-07-18.md`).
- If a quotation is genuinely Pynchon's fiction but you are confident it cannot be
  located in the ten primary texts (a variant edition, or text outside the corpus),
  choose "In scope — NOT locatable in the ten primary texts" (web dropdown, or `x` at
  the source prompt in the terminal) and explain in the notes — the note is required
  before the document can be completed. This is deliberately different from leaving
  the source unresolved (which means "findable, but I haven't confirmed it yet").
- `n` marks an exclusion and requires a reason. When excluding, ignore the source
  and chapter fields entirely — they are cleared automatically on save; only the
  exclusion reason is recorded. A genuine Pynchon-fiction quotation of fewer than
  five words gets its own reason ("Pynchon's fiction but fewer than five words");
  count words naturally — hyphenated compounds and contractions are one word.
- Quotations interrupted by attribution ("...," Olson wrote, "...") appear as
  separate marked spans; decide each on its own. Pick the exclusion reason by whose
  words they are first — non-Pynchon spans are excluded as another author/critic
  regardless of length; the fewer-than-five-words reason is only for genuine Pynchon
  fiction, applied per span. Parenthetical page numbers are never part of a span.
- `e` corrects boundaries, source, alterations, notes, quotation form, or exclusion.
- `u` clears a decision; `z` undoes the most recent change in the current session.

The interface advances to the next unresolved annotation after `y` or `n`. Press `h` at
any time for the complete command list.

**Alterations by scholar** (in-scope spans only): record how the quotation as printed differs
from Pynchon's canonical text, comparing against the matched source snippet in the
locator suggestion. Use these comma-separated tags, consistently:

- `ellipsis` — text omitted (… or […]);
- `brackets` — editorial insertion or change in square brackets ("[w]here");
- `wording` — the words differ (misquotation, changed tense, swapped word).

Leave the field **empty when the quote is verbatim** — empty means "unaltered" and
is analysed as such. Do not tag pure typography (curly vs straight quotes, dashes);
the pipeline normalises those. For anything not covered, use the nearest tag and
explain in Notes.

## Find omissions

The automatic suggestions are not exhaustive. After reviewing them:

1. Use `v` to page through the complete article.
2. Check especially for block quotations and passages marked only with single quotes.
3. Use `/` to search the article.
4. Use `m` to add a missed quotation by pasting its article wording on one line.
   Whitespace differences are tolerated. Character offsets in the form `@START:END`
   are also accepted.

If a missed block overlaps an automatically suggested nested quotation, the interface
warns about the overlap. Usually the right action is to correct the existing annotation's
boundary with `e`, then `b`, rather than create duplicate gold spans.

## Complete an article

Use `c` only after the full-document pass. Completion is blocked while any span lacks:

- an in/out-of-scope decision;
- an exclusion category when out of scope;
- a confirmed PID, work, and chapter/episode when in scope; or
- valid offsets matching the article body exactly.

The final confirmation requires typing `COMPLETE`. Completed files can then be compiled
with `compile_gold_v2.py`. Editing a completed annotation automatically returns it to
`in_progress` until the full-document check is confirmed again.

## Independent annotation

The interface accepts `--annotations PATH` and `--reviewer ID`, so a second reviewer can
work from a separate directory of untouched templates. Do not give the second reviewer
the first reviewer's labelled files: independent review must begin from the original
unreviewed templates.
