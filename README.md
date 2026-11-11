# Detecting quotations of Pynchon's fiction in scholarly articles — evaluation artifact

This repository accompanies the paper:

> Ketzan, Erik, and Irina Torde. 2027. "Not Every Match Is a Quotation: Evaluating
> LLMs for Quotation Detection in Literary Reception." In *Proceedings of the
> Computational Humanities Research Conference 2027 (CHR 2027)*. Anthology of
> Computers and the Humanities. (Forthcoming.)

It contains everything needed to inspect the frozen experimental protocol and to
audit the reported numbers, with the redactions described below.

The annotated articles are drawn from the **Open Pynchon Reception Corpus (OPRC)**:

- Corpus: Ketzan, Erik. 2026. *Open Pynchon Reception Corpus*. King's College
  London Research Data Repository (KORDS). https://doi.org/10.18742/33188055
- Data paper: Ketzan, Erik. 2026. "The Open Pynchon Reception Corpus: From
  Bibliographic Metadata to Reusable Scholarly Full Text." *Journal of Open
  Humanities Data* 12: 142, pp. 1–6. https://doi.org/10.5334/johd.632

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
them (62 rows on the test set). The published values scored those rows
against their original text, so the two LLM systems and the hybrids containing
them re-score within 0.01 F1 of the published values, with individual
precision and recall figures differing by up to 0.016 (for example, frontier
LLM detection F1 0.967 against the published 0.976). The difference can go in
either direction for hybrids. Restoring the 62 rows' original text reproduces
every published value exactly.

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

## License

- Code in `pipeline/`: MIT License (see `LICENSE`).
- Annotations, gold standard, candidate lists, metrics and documentation:
  Creative Commons Attribution 4.0 International (CC BY 4.0).
- Articles in `articles/`: each remains under its original CC BY license, as
  recorded in `articles/ATTRIBUTION.csv`; please cite the original articles.
