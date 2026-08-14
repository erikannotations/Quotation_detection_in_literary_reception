# Detecting quotations of Pynchon's fiction in scholarly articles — evaluation artifact

This artifact accompanies a CHR 2027 submission on scholarly quotation detection.
It contains everything needed to inspect the frozen experimental protocol and to
audit the reported numbers, with the redactions described below.

## Contents

- `pipeline/` — the complete detection-and-evaluation codebase.
  - The frozen LLM prompt and output schema are in `pipeline/detect_quotes.py`
    (the `SYSTEM` constant and the structured-output field definitions).
  - Frozen model identifiers, inference settings and freeze records:
    `pipeline/models_frozen.json`, `pipeline/evaluation/run_manifest.frozen.json`.
  - Per-run inference records (model version, dates, token counts, decoding and
    context-window settings, wall time): `pipeline/evaluation/candidates/*.runinfo.json`.
  - Evaluation protocol: `pipeline/eval_compare.py` (bipartite matching at
    token-similarity 0.60; the five-word scope floor in `pipeline/quotation_spans.py`).
  - Reported-table generators: `pipeline/make_results_table.py` (Table 1),
    `pipeline/precision_by_length.py` (Table 3).
  - All metrics reported in the paper: `pipeline/evaluation/*.json`.
- `articles/` — the sixteen annotated CC BY articles (four development, twelve
  held-out test; each file's split is recorded in
  `pipeline/evaluation/sample_manifest.csv`). See the README and
  `ATTRIBUTION.csv` there for licensing and attribution.
- `docs/` — the annotation manual shown to both annotators.

## Redactions

Thomas Pynchon's fiction is in copyright, and this artifact contains no text
from it. In `pipeline/evaluation/gold/*.gold.json` and
`pipeline/evaluation/candidates/*` the `quote`/`quote_preview` fields and
free-text annotation notes have been removed. Every remaining record keeps its
character offsets (`article_start_char`, `article_end_char`) into the article
body, the article's SHA-256, and its source attribution (work and
chapter/episode), so any span can be recovered from the openly licensed
articles in `articles/`. Source-linking (locus) verification requires the
primary texts.

**To re-score the reported detection results**, run
`python3 pipeline/rescore_release.py` from the release root. It copies the
release to a sibling working directory, rebuilds every record's quote text
from its offsets and the shipped articles (exactly reversing the redaction),
restricts to the twelve held-out test articles, and runs the shipped scorer
unmodified. The lexical baselines, the heuristic and the gold-to-locator
ceiling reproduce the published numbers exactly. One caveat is inherent to
the redaction: a small number of LLM candidate rows are proposals whose
wording could not be aligned to article offsets, so the redacted text was
their only content and cannot be rebuilt; the script counts and reports
them, and the two LLM systems (and hybrids containing them) re-score
slightly below the published values (third decimal), which scored those rows
against their original text.

This copy is anonymized for double-blind review: the reception corpus's name
and archive location, withheld here, will be cited in the de-anonymized
version of this artifact.

Consequences for the code and tests:

- `vine_build_chapters.py` and its output are not included, as the script
  embeds text from the novel; `atd_build_chapters.py` is included but reads an
  anchor CSV that is not shipped, for the same reason.
- In `tests/test_pipeline.py`, a handful of fixture strings that quoted the
  source texts have been replaced with synthetic sentences of identical
  typographic shape; the affected tests pass unchanged. Tests that require the
  primary texts or the annotation workspace (the locator tests, the
  corpus-structure test, and the browser annotation tests) will error in this
  copy; the remaining 70 tests pass.

## Layout note

`pipeline/` resolves all paths relative to its own location
(`pipeline/project_paths.py`); no absolute paths are assumed. Python
dependencies: `pipeline/requirements-baselines.txt` and
`pipeline/requirements-textmatcher.txt`.
