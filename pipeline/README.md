# Pynchon quotation-pipeline workspace

The primary texts remain one canonical plaintext file per work. Chapters and episodes
are represented by generated character-offset indexes rather than duplicated physical
chapter files.

## Layout

```text
sources/
  primary/                 canonical P01–P10 plaintext inputs
  metadata.csv             local primary-work metadata
derived/
  segmentation/            generated sections JSON, XML, and chapter index
auxiliary/
  vineland/                harvested annotation pages used as P05 anchors
  against_the_day/         harvested P07 episode/page table
evaluation/
  sample_manifest.csv      fixed 4-development / 12-test sample (full test pool
                           restored 22 July 2026)
  annotations/             human-review working files
  gold/                    completed gold-v2 files and instructions
  candidates/              active system outputs and locator results
  legacy/                  historical POC labels, censuses, and candidate sets
legacy/
  source_backups/          superseded P02/P06 inputs
tests/                     regression suite
```

Pipeline scripts remain at this directory's top level so existing imports and commands
stay simple. `project_paths.py` is the single source of truth for directory locations.
Superseded evaluation scripts are retained under `evaluation/legacy/scripts/`.

## Rebuild segmentation

The order matters because the first command creates a P05 placeholder and part-level P07
index; the next two replace them with the exact 15-chapter and 70-chapter indexes, each
located directly in the plaintext from author-transcribed chapter-opening sentences (no
page interpolation, no scraped wiki data).

```sh
python3 segment_corpus.py
python3 atd_build_chapters.py
python3 vine_build_chapters.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

## Evaluation workflow

```sh
python3 sample_corpus.py
python3 prepare_annotations.py
python3 review_annotations.py
python3 compile_gold_v2.py
python3 run_oracle_locator.py
python3 eval_compare.py --bootstrap 5000 --paired \
  --json-out evaluation/evaluation_metrics.json
python3 make_run_manifest.py --status development \
  --output evaluation/run_manifest.development.json
```

Additional experiment tools:

- `review_annotations.py`: autosaving terminal annotation interface; see
  `ANNOTATION_GUIDE.md`;
- `combine_candidates.py`: provenance-preserving hybrid union;
- `annotation_agreement.py`: span, scope-label, and source agreement;
- `annotation_progress.py`: workload and completion summary by split;
- `run_oracle_locator.py`: source-linking ceiling from gold spans;
- `make_run_manifest.py`: hashes and freezes corpus, code, prompt, thresholds, models,
  and environment.

The contents of `sources/primary/` and generated XML are local research inputs, not
release artifacts.




