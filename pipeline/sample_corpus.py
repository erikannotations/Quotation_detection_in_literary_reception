#!/usr/bin/env python3
"""Build the deterministic 4-development / 12-test CHR sample.

The test split is the original frozen 12-document, five-stratum pool in full.
"""
import argparse
import csv
import hashlib
from pathlib import Path
from project_paths import SAMPLE_MANIFEST
from quotation_spans import word_count
from review_annotations import metadata_normalized_body

BASE = Path(__file__).resolve().parent
#: This project's own copies of the sampled article texts, which take precedence over
#: the corpus working directory when the two diverge (see the note in main()).
LOCAL_ARTICLES = {
    path.stem: path
    for name in ("01_orbit_articles", "03_non_orbit_cc_articles")
    for path in (BASE.parent / "annotation_workspace" / name).glob("*.txt")
}
DEFAULT_METADATA = BASE.parent.parent / "Reception_Corpus" / "corpus_articles" / "metadata.csv"
DEV_KEYS = {"YT54NC7G", "7L5Z97IZ", "5E3W868P", "YUZMXRKY"}
DEV_STRATA = {"7L5Z97IZ": "random-replacement"}  # all others: "pre-inspected"
#
TEST_POOL = [
    ("non_orbit", ["US8FE2Q5"]),
    ("orbit_2012-2014", ["C6VQPDG4", "E2TDDQA5", "KF8LYIPQ", "9VU7EIQJ"]),
    ("orbit_2015-2017", ["7IIYT69Y", "8Q6F6MLD", "9RSYBDHU"]),
    ("orbit_2018-2020", ["LS5YAQLE", "W6SQ6GJ8"]),
    ("orbit_2021-2025", ["89JQFT3W", "YKVFQH5I"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--output", default=str(SAMPLE_MANIFEST))
    args = parser.parse_args()

    with Path(args.metadata).open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["item_type"] == "journalArticle"]
    by_key = {row["zotero_key"]: row for row in rows}
    missing = DEV_KEYS - set(by_key)
    if missing:
        raise SystemExit(f"development keys missing from CORPUS metadata: {sorted(missing)}")

    # Gather development articles from the metadata, assigning strata.
    selected = []
    for key in sorted(DEV_KEYS):
        row = dict(by_key[key])
        row.update({"split": "development",
                    "stratum": DEV_STRATA.get(key, "pre-inspected")})
        selected.append(row)

    # Gather test articles, validating that all pool keys exist in metadata.
    for stratum, keys in TEST_POOL:
        missing = set(keys) - set(by_key)
        if missing:
            raise SystemExit(f"test-pool keys missing from CORPUS metadata: {sorted(missing)}")
        for key in keys:
            row = dict(by_key[key])
            row.update({"split": "test", "stratum": stratum})
            selected.append(row)

    # Describe the text this project actually reads, not what the corpus metadata
    # records about it: where a local copy corrects an upstream extraction fault,
    # taking the hash from upstream metadata would record the uncorrected text's
    # hash against the corrected file. `annotation_web` already prefers the local
    # copy for exactly this reason; the manifest now agrees with it.
    for row in selected:
        local = LOCAL_ARTICLES.get(Path(row["text_path"]).stem)
        if local is None:
            continue
        body = metadata_normalized_body(local.read_text(encoding="utf-8"))
        row["body_word_count"] = word_count(body)
        row["body_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()

    fields = ["split", "stratum", "corpus_id", "zotero_key", "year", "author", "title",
              "corpus_section", "text_path", "body_word_count", "body_sha256"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in selected)
    counts = {split: sum(row["split"] == split for row in selected)
              for split in ("development", "test")}
    print(f"wrote {len(selected)} articles to {args.output}: {counts}")


if __name__ == "__main__":
    main()
