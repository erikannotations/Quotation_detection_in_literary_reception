#!/usr/bin/env python3
"""Interactive, autosaving review interface for CHR quotation annotations.

The interface deliberately treats automatic quote-mark spans as proposals, not gold.
It supports source and chapter/episode confirmation, boundary correction, manual
missed-span addition, and a guarded full-document completion step. It uses only the
Python standard library.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pydoc
import re
import shutil
import stat
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from project_paths import ANNOTATIONS_DIR, SAMPLE_MANIFEST, SEGMENTATION_DIR
from quotation_spans import article_body, find_quote_offsets, word_count


BASE = Path(__file__).resolve().parent
DEFAULT_CORPUS = BASE.parent.parent / "Reception_Corpus" / "corpus_articles"
PIDS = [f"P0{i}" for i in range(1, 10)] + ["P10"]
WORK_TITLES = {
    "P01": "Slow Learner",
    "P02": "V.",
    "P03": "The Crying of Lot 49",
    "P04": "Gravity's Rainbow",
    "P05": "Vineland",
    "P06": "Mason & Dixon",
    "P07": "Against the Day",
    "P08": "Inherent Vice",
    "P09": "Bleeding Edge",
    "P10": "Shadow Ticket",
}
EXCLUSIONS = [
    ("other_author_or_critic", "quotation from another author or critic"),
    ("title_or_scare_quote", "title, term, or scare quotation"),
    ("pynchon_nonfiction", "Pynchon nonfiction"),
    ("pynchon_under_five_words", "Pynchon's fiction but fewer than five words"),
    ("not_direct_quotation", "allusion, paraphrase, summary, or otherwise indirect"),
    ("duplicate_or_boundary_artifact", "duplicate/nested span or extraction artifact"),
    ("other", "other reason (explain in notes)"),
]


_SECTION_LABEL_CACHE = {}


def load_section_labels(pid):
    """Return {n: label} for one work's exact chapter/episode index.

    Reads the small per-work ``derived/segmentation/<pid>.sections.json`` directly
    rather than instantiating ``locate_quotes.Corpus`` (which loads and normalizes
    all ten primary texts) -- this is called from request-handling and validation
    paths where that cost matters. Cached per pid for the life of the process; the
    segmentation files are build artifacts that do not change during a review
    session.
    """
    if pid not in _SECTION_LABEL_CACHE:
        path = SEGMENTATION_DIR / f"{pid}.sections.json"
        if not path.exists():
            _SECTION_LABEL_CACHE[pid] = {}
        else:
            secs = json.loads(path.read_text(encoding="utf-8"))
            _SECTION_LABEL_CACHE[pid] = {s["n"]: s["label"] for s in secs}
    return _SECTION_LABEL_CACHE[pid]


def annotation_rows(data):
    """Return automatic and manually added rows in stable interface order."""
    return data.get("candidates", []) + data.get("manually_added_spans", [])


def annotation_progress(data):
    rows = annotation_rows(data)
    labelled = sum(row.get("is_in_scope") is not None for row in rows)
    positives = sum(row.get("is_in_scope") is True for row in rows)
    return {
        "total": len(rows),
        "labelled": labelled,
        "open": len(rows) - labelled,
        "positives": positives,
    }


def completion_issues(data, body=None):
    """List conditions that make an annotation unsafe to compile as gold."""
    issues = []
    for row in annotation_rows(data):
        aid = row.get("annotation_id", "<missing id>")
        label = row.get("is_in_scope")
        if label is None:
            issues.append(f"{aid}: scope label unresolved")
        elif label is True:
            # A span shorter than the frozen five-word minimum cannot be in scope,
            # whatever else is true of it. `set_positive` does not check length —
            # only manual creation and correction do — so a SEEDED candidate can be
            # confirmed in scope without ever passing a length test. Six seeded
            # candidates are affected, all bracketed alterations that counted as five
            # words before the 23 July `word_count()` fix and count as four after it.
            # Named with its exclusion category so the fix is one click, not a puzzle.
            if word_count(row.get("quote", "")) < 5:
                issues.append(
                    f"{aid}: {word_count(row.get('quote', ''))} words, under the "
                    f"five-word minimum — exclude as 'pynchon_under_five_words'")
            elif row.get("in_primary_corpus") is False:
                # In scope but not locatable in the ten primary texts: no
                # source/section required, but the reason must be on record.
                if not (row.get("notes") or "").strip():
                    issues.append(f"{aid}: not-locatable span needs an explanatory note")
            else:
                true_pid = row.get("true_pid")
                if true_pid not in PIDS or not row.get("true_work"):
                    issues.append(f"{aid}: source work unresolved")
                else:
                    section_n = row.get("true_section_n")
                    if not isinstance(section_n, int) or section_n not in load_section_labels(true_pid):
                        issues.append(f"{aid}: chapter/episode unresolved")
        elif not row.get("exclusion_category"):
            issues.append(f"{aid}: exclusion category missing")

        start, end = row.get("start_char"), row.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            issues.append(f"{aid}: invalid article offsets")
        elif body is not None:
            if end > len(body):
                issues.append(f"{aid}: article offsets exceed body length")
            elif body[start:end] != row.get("quote"):
                issues.append(f"{aid}: stored quote does not equal body slice")

    # Two live spans over the same article text would count one quotation twice
    # in the gold data. Excluded rows are ignored: a span absorbed into a larger
    # one legitimately stays behind as a duplicate_or_boundary_artifact record.
    live = sorted(
        (row["start_char"], row["end_char"], row.get("annotation_id", "<missing id>"))
        for row in annotation_rows(data)
        if row.get("is_in_scope") is not False
        and isinstance(row.get("start_char"), int)
        and isinstance(row.get("end_char"), int)
        and 0 <= row["start_char"] < row["end_char"]
    )
    reach_end, reach_aid = -1, ""
    for start, end, aid in live:
        if start < reach_end:
            issues.append(
                f"{reach_aid} and {aid}: spans overlap — merge them, or exclude "
                "one as duplicate_or_boundary_artifact")
        if end > reach_end:
            reach_end, reach_aid = end, aid
    return issues


def metadata_normalized_body(raw_text):
    """Reproduce the normalization used for CORPUS ``body_sha256`` metadata."""
    lines = raw_text.splitlines()
    separators = [
        index for index, line in enumerate(lines)
        if len(line.strip()) >= 10 and set(line.strip()) == {"="}
    ]
    if len(separators) != 1:
        raise ValueError(f"expected one CORPUS metadata separator, found {len(separators)}")
    return "\n".join(lines[separators[0] + 1:]).lstrip("\n").rstrip() + "\n"


def verify_body_hash(raw_text, expected):
    actual = hashlib.sha256(metadata_normalized_body(raw_text).encode("utf-8")).hexdigest()
    return actual == expected, actual


def update_context(row, body, width=180):
    start, end = row["start_char"], row["end_char"]
    row["before"] = body[max(0, start - width):start].replace("\n", " ")
    row["after"] = body[end:min(len(body), end + width)].replace("\n", " ")


def make_manual_span(data, body, start, end, quotation_form="manual_block"):
    """Create and append an unresolved manual span, returning the new row."""
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(body)):
        raise ValueError("manual span offsets are outside the article body")
    quote = body[start:end]
    if word_count(quote) < 5:
        raise ValueError("the main task requires quotations of at least five words")
    used = {row.get("annotation_id") for row in annotation_rows(data)}
    serial = 1
    while f"{data['zotero_key']}-m{serial:04d}" in used:
        serial += 1
    row = {
        "annotation_id": f"{data['zotero_key']}-m{serial:04d}",
        "start_char": start,
        "end_char": end,
        "quote": quote,
        "quotation_form": quotation_form,
        "before": "",
        "after": "",
        "is_in_scope": None,
        "exclusion_category": "",
        "true_pid": "",
        "true_work": "",
        "true_section_n": None,
        "true_section_label": "",
        "in_primary_corpus": None,
        "alterations": [],
        "notes": "",
    }
    update_context(row, body)
    data.setdefault("manually_added_spans", []).append(row)
    return row


def set_positive(row, pid="", work="", alterations=None, section_n=None):
    """Confirm an in-scope span. ``section_n`` is the exact chapter/episode index.

    The stored label is always re-derived here from (pid, section_n) against the
    segmentation index -- never accepted as free text from a caller -- so the
    canonical number and its display label cannot drift apart. An invalid or
    missing section_n leaves the chapter/episode unresolved (None), which
    ``completion_issues`` treats the same as an unresolved source work.
    """
    row["is_in_scope"] = True
    row["exclusion_category"] = ""
    row["true_pid"] = pid
    row["true_work"] = work
    row["in_primary_corpus"] = True
    row["alterations"] = list(alterations or [])
    labels = load_section_labels(pid) if pid else {}
    if isinstance(section_n, int) and section_n in labels:
        row["true_section_n"] = section_n
        row["true_section_label"] = labels[section_n]
    else:
        row["true_section_n"] = None
        row["true_section_label"] = ""


UNLOCATED = "__unlocated__"


def set_positive_unlocated(row, alterations=None):
    """Confirm an in-scope span that genuinely cannot be located in the ten
    primary texts (variant edition, uncollected text, a scholar's misattribution
    of fiction). Requires an explanatory note before the document can be marked
    complete (enforced by ``completion_issues``). Added 22 July 2026 — this
    state was previously unrepresentable, and a single such span blocked its
    whole document from compiling into gold.
    """
    row["is_in_scope"] = True
    row["exclusion_category"] = ""
    row["true_pid"] = ""
    row["true_work"] = ""
    row["true_section_n"] = None
    row["true_section_label"] = ""
    row["in_primary_corpus"] = False
    row["alterations"] = list(alterations or [])


def set_negative(row, category):
    if category not in {key for key, _ in EXCLUSIONS}:
        raise ValueError(f"unknown exclusion category: {category}")
    row["is_in_scope"] = False
    row["exclusion_category"] = category
    row["true_pid"] = ""
    row["true_work"] = ""
    row["true_section_n"] = None
    row["true_section_label"] = ""
    row["in_primary_corpus"] = False
    row["alterations"] = []


def make_writable(path):
    """Clear a read-only bit the app set on its own working file.

    Annotation files seeded from ``.pristine_templates`` inherit that folder's
    read-only mode through ``shutil.copy2``, and the mode is copied onward to the
    ``.bak``. Windows enforces this harder than macOS: both ``copy2`` over a
    read-only backup and ``os.replace`` onto a read-only target raise
    PermissionError. The archival copies in ``.pristine_templates`` are the thing
    meant to stay read-only, so a working copy the app owns should not.
    """
    path = Path(path)
    if path.exists():
        try:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass  # Best effort; the write below reports any real failure.


def save_annotation(path, data, backup=True):
    """Atomically autosave JSON and retain the immediately previous version."""
    path = Path(path)
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        make_writable(backup_path)
        shutil.copy2(path, backup_path)
        make_writable(backup_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_writable(path)
    os.replace(temporary, path)


def load_manifest(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clip(text, limit=1400):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + " … [display clipped] … " + text[-half:]


def wrapped(text, indent="", width=None):
    width = width or max(68, min(110, shutil.get_terminal_size((90, 24)).columns))
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


class ReviewSession:
    def __init__(self, annotation_path, manifest_row, corpus_root, reviewer=""):
        self.path = Path(annotation_path)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.manifest_row = manifest_row
        self.source_path = Path(corpus_root) / manifest_row["text_path"]
        raw = self.source_path.read_text(encoding="utf-8")
        valid, actual = verify_body_hash(raw, self.data["body_sha256"])
        if not valid:
            raise ValueError(
                f"body checksum mismatch for {self.source_path.name}:\n"
                f" expected {self.data['body_sha256']}\n actual   {actual}\n"
                "Do not annotate until the corpus/template version mismatch is resolved."
            )
        self.body = article_body(raw)
        self.index = 0
        self.history = []
        self.locator = None
        self.locator_cache = {}
        if reviewer and not self.data.get("reviewer"):
            self.data["reviewer"] = reviewer
            save_annotation(self.path, self.data)

    def rows(self):
        return annotation_rows(self.data)

    def remember(self):
        self.history.append(copy.deepcopy(self.data))
        self.history = self.history[-50:]

    def commit(self):
        if self.data.get("review_status") == "complete":
            self.data["full_document_reviewed"] = False
            self.data["completed_at"] = ""
        self.data["review_status"] = "in_progress"
        self.data["last_saved_at"] = datetime.now(timezone.utc).isoformat()
        save_annotation(self.path, self.data)

    def undo(self):
        if not self.history:
            print("Nothing to undo in this session.")
            return
        self.data = self.history.pop()
        save_annotation(self.path, self.data)
        self.index = min(self.index, max(0, len(self.rows()) - 1))
        print("Restored the previous annotation state.")

    def display_row(self):
        rows = self.rows()
        if not rows:
            print("This document has no spans. Use 'm' to add one after full-document review.")
            return
        row = rows[self.index]
        progress = annotation_progress(self.data)
        label = "UNREVIEWED"
        if row.get("is_in_scope") is True:
            source = f"{row.get('true_pid') or '?'} / {row.get('true_work') or '?'} / {row.get('true_section_label') or 'chapter/episode?'}"
            label = f"IN SCOPE — {source}"
        elif row.get("is_in_scope") is False:
            label = f"OUT OF SCOPE — {row.get('exclusion_category') or '?'}"
        print("\n" + "=" * 78)
        print(f"{self.data['article']}  [{self.data['split']}]  status={self.data.get('review_status')}")
        print(
            f"Span {self.index + 1}/{len(rows)}: {row.get('annotation_id')}  "
            f"reviewed={progress['labelled']}/{progress['total']}  positives={progress['positives']}"
        )
        print(f"Label: {label}; form={row.get('quotation_form', '?')}; offsets={row.get('start_char')}:{row.get('end_char')}")
        print("-" * 78)
        start, end = row["start_char"], row["end_char"]
        before = clip(self.body[max(0, start - 260):start], 500)
        quote = clip(self.body[start:end])
        after = clip(self.body[end:min(len(self.body), end + 260)], 500)
        print(wrapped(before, "  "))
        print(wrapped(f">>> {quote} <<<", "  "))
        print(wrapped(after, "  "))
        if row.get("alterations"):
            print(f"Alterations: {', '.join(row['alterations'])}")
        if row.get("notes"):
            print(wrapped(f"Notes: {row['notes']}"))

    def _locator_results(self, row):
        aid = row["annotation_id"]
        if aid not in self.locator_cache:
            if self.locator is None:
                print("Loading the primary-text locator …")
                from locate_quotes import Corpus
                self.locator = Corpus()
            self.locator_cache[aid] = self.locator.locate_all(row["quote"], limit=5)
        results = self.locator_cache[aid]
        if not results:
            print("No locator suggestion. Leave the source unresolved rather than guessing.")
        else:
            print("Locator suggestions (confirm against the quotation; these are not labels):")
            for number, result in enumerate(results, 1):
                print(
                    f"  {number}. {result['pid']}  {result['novel']} — {result['section']}  "
                    f"score={result['score']:.3f}"
                )
        return results

    def choose_source(self, row):
        """Return (pid, work, section_n). A suggestion supplies all three at once;
        a manually typed PID additionally prompts for the chapter/episode number,
        since that path is the rare unconfirmed-source fallback (settled by
        primary-text search) rather than confirming a suggestion already shown.
        """
        results = self._locator_results(row)
        default = results[0] if results else None
        prompt = "Source PID or suggestion number"
        if default:
            prompt += f" [Enter accepts 1: {default['pid']} {default['section']}]"
        else:
            prompt += " [u leaves unresolved]"
        prompt += " [x = in scope but NOT locatable in the ten primary texts]"
        answer = input(prompt + ": ").strip()
        if not answer and default:
            chosen = default
            return chosen["pid"], chosen["novel"], chosen["section_n"]
        if answer.isdigit() and 1 <= int(answer) <= len(results):
            chosen = results[int(answer) - 1]
            return chosen["pid"], chosen["novel"], chosen["section_n"]
        pid = answer.upper()
        if pid == "X":
            return UNLOCATED, "", None
        if pid in {"", "U", "?"}:
            return "", "", None
        if pid not in PIDS:
            print("Unknown PID; source left unresolved.")
            return "", "", None
        if pid == "P01":
            answer = input("P01 work [1 Slow Learner / 2 Mortality and Mercy in Vienna]: ").strip()
            work = "Mortality and Mercy in Vienna" if answer == "2" else "Slow Learner"
        else:
            work = WORK_TITLES[pid]
        return pid, work, self.choose_section(pid)

    @staticmethod
    def choose_section(pid):
        labels = load_section_labels(pid)
        if not labels:
            print(f"No chapter/episode index available for {pid}; left unresolved.")
            return None
        answer = input(f"Chapter/episode number for {pid} [1-{max(labels)}; blank leaves unresolved]: ").strip()
        if not answer.isdigit() or int(answer) not in labels:
            if answer:
                print("Not a valid chapter/episode number; left unresolved.")
            return None
        n = int(answer)
        print(f"  -> {labels[n]}")
        return n

    @staticmethod
    def choose_alterations(current=None):
        current_text = ",".join(current or [])
        prompt = (
            "Alterations (comma-separated: ellipsis, brackets, case, punctuation, "
            "wording, other; Enter means exact)"
        )
        if current_text:
            prompt += f" [{current_text}]"
        answer = input(prompt + ": ").strip()
        if not answer:
            return list(current or []) if current_text else []
        if answer.lower() == "exact":
            return []
        return [item.strip().lower() for item in answer.split(",") if item.strip()]

    @staticmethod
    def choose_exclusion(current=""):
        print("Exclusion category:")
        for number, (key, description) in enumerate(EXCLUSIONS, 1):
            suffix = " (current)" if key == current else ""
            print(f"  {number}. {description} [{key}]{suffix}")
        answer = input("Category number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(EXCLUSIONS):
            return EXCLUSIONS[int(answer) - 1][0]
        print("No valid category selected; annotation unchanged.")
        return None

    def label_positive(self):
        row = self.rows()[self.index]
        pid, work, section_n = self.choose_source(row)
        alterations = self.choose_alterations(row.get("alterations"))
        self.remember()
        if pid == UNLOCATED:
            set_positive_unlocated(row, alterations)
            if not (row.get("notes") or "").strip():
                print("Add an explanatory note (edit > n) — required before completion.")
        else:
            set_positive(row, pid, work, alterations, section_n)
        self.commit()
        self.advance_to_open()

    def label_negative(self):
        row = self.rows()[self.index]
        category = self.choose_exclusion(row.get("exclusion_category", ""))
        if not category:
            return
        self.remember()
        set_negative(row, category)
        self.commit()
        self.advance_to_open()

    def clear_label(self):
        row = self.rows()[self.index]
        self.remember()
        row.update({
            "is_in_scope": None,
            "exclusion_category": "",
            "true_pid": "",
            "true_work": "",
            "true_section_n": None,
            "true_section_label": "",
            "in_primary_corpus": None,
            "alterations": [],
        })
        self.commit()

    def advance_to_open(self):
        rows = self.rows()
        for offset in range(1, len(rows) + 1):
            candidate = (self.index + offset) % len(rows)
            if rows[candidate].get("is_in_scope") is None:
                self.index = candidate
                return
        self.index = min(self.index + 1, len(rows) - 1)

    def choose_occurrence(self, query):
        start, end, count = find_quote_offsets(self.body, query)
        if start is None:
            print("That text was not found in the normalized article body.")
            return None
        if count == 1:
            return start, end
        print(f"Found {count} occurrences:")
        spans = []
        for occurrence in range(count):
            s, e, _ = find_quote_offsets(self.body, query, occurrence_index=occurrence)
            spans.append((s, e))
            context = clip(self.body[max(0, s - 70):min(len(self.body), e + 70)], 250)
            print(wrapped(f"  {occurrence + 1}. [{s}:{e}] {context}"))
        answer = input("Occurrence number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(spans):
            return spans[int(answer) - 1]
        print("No occurrence selected.")
        return None

    def edit_boundary(self):
        row = self.rows()[self.index]
        print("Paste the complete quotation as it appears in the article on one line.")
        print("Whitespace may differ. Alternatively enter @START:END character offsets.")
        answer = input("Corrected span: ").strip()
        if not answer:
            return
        if answer.startswith("@"):
            match = re.fullmatch(r"@(\d+):(\d+)", answer)
            if not match:
                print("Offset syntax must be @START:END.")
                return
            start, end = map(int, match.groups())
            if not (0 <= start < end <= len(self.body)):
                print("Offsets are outside the article body.")
                return
        else:
            span = self.choose_occurrence(answer)
            if not span:
                return
            start, end = span
        if word_count(self.body[start:end]) < 5:
            print("The corrected span has fewer than five words and is outside the main task.")
            return
        self.remember()
        row["start_char"], row["end_char"] = start, end
        row["quote"] = self.body[start:end]
        update_context(row, self.body)
        self.locator_cache.pop(row["annotation_id"], None)
        self.commit()

    def edit_row(self):
        while True:
            row = self.rows()[self.index]
            print("Edit: [b]oundary [s]ource [a]lterations [n]otes [x]exclusion [f]orm [q]back")
            command = input("edit> ").strip().lower()
            if command in {"q", ""}:
                return
            if command == "b":
                self.edit_boundary()
            elif command == "s":
                pid, work, section_n = self.choose_source(row)
                self.remember()
                if pid == UNLOCATED:
                    set_positive_unlocated(row, row.get("alterations"))
                    if not (row.get("notes") or "").strip():
                        print("Add an explanatory note (n) — required before completion.")
                else:
                    row["true_pid"], row["true_work"] = pid, work
                    if pid:
                        row["is_in_scope"] = True
                        row["in_primary_corpus"] = True
                        row["exclusion_category"] = ""
                        labels = load_section_labels(pid)
                        row["true_section_n"] = section_n if isinstance(section_n, int) and section_n in labels else None
                        row["true_section_label"] = labels.get(row["true_section_n"], "")
                    else:
                        row["true_section_n"], row["true_section_label"] = None, ""
                self.commit()
            elif command == "a":
                changes = self.choose_alterations(row.get("alterations"))
                self.remember()
                row["alterations"] = changes
                self.commit()
            elif command == "n":
                note = input(f"Notes [{row.get('notes', '')}]: ").strip()
                if note:
                    self.remember()
                    row["notes"] = note
                    self.commit()
            elif command == "x":
                category = self.choose_exclusion(row.get("exclusion_category", ""))
                if category:
                    self.remember()
                    set_negative(row, category)
                    self.commit()
            elif command == "f":
                form = input(f"Quotation form [{row.get('quotation_form', '')}]: ").strip()
                if form:
                    self.remember()
                    row["quotation_form"] = form
                    self.commit()
            else:
                print("Unknown edit command.")

    def add_manual(self):
        print("Add a quotation missed by the automatic suggestions.")
        print("Paste its article text on one line, or enter @START:END character offsets.")
        answer = input("Manual span: ").strip()
        if not answer:
            return
        if answer.startswith("@"):
            match = re.fullmatch(r"@(\d+):(\d+)", answer)
            if not match:
                print("Offset syntax must be @START:END.")
                return
            span = tuple(map(int, match.groups()))
        else:
            span = self.choose_occurrence(answer)
            if not span:
                return
        start, end = span
        overlaps = [
            row["annotation_id"] for row in self.rows()
            if start < row["end_char"] and row["start_char"] < end
        ]
        if overlaps:
            print(f"Warning: this overlaps existing span(s): {', '.join(overlaps)}")
            if input("Add it anyway? [y/N]: ").strip().lower() != "y":
                print("Use boundary editing on the existing span if it is incomplete.")
                return
        form = input("Form [block/single_quote/other; default block]: ").strip() or "block"
        self.remember()
        try:
            row = make_manual_span(self.data, self.body, start, end, f"manual_{form}")
        except ValueError as error:
            self.history.pop()
            print(error)
            return
        self.commit()
        self.index = self.rows().index(row)
        print(f"Added {row['annotation_id']}; label it now with y or n.")

    def search_article(self):
        query = input("Literal article search: ").strip()
        if not query:
            return
        pattern = re.compile(r"\s+".join(re.escape(piece) for piece in re.split(r"\s+", query)), re.I)
        matches = list(pattern.finditer(self.body))
        print(f"Found {len(matches)} occurrence(s).")
        for number, match in enumerate(matches[:20], 1):
            context = clip(
                self.body[max(0, match.start() - 100):min(len(self.body), match.end() + 100)],
                320,
            )
            print(wrapped(f"  {number}. [{match.start()}:{match.end()}] {context}"))
        if len(matches) > 20:
            print("Only the first 20 are displayed; refine the search phrase.")

    def complete_document(self):
        issues = completion_issues(self.data, self.body)
        if issues:
            print(f"Cannot complete: {len(issues)} issue(s) remain.")
            for issue in issues[:15]:
                print(f"  - {issue}")
            if len(issues) > 15:
                print(f"  … and {len(issues) - 15} more")
            return
        print("Completion requires a detector-independent pass through the complete article.")
        print(f"Source file: {self.source_path}")
        confirmation = input("Type COMPLETE only if block and single-quoted spans were checked: ").strip()
        if confirmation != "COMPLETE":
            print("Document remains in progress.")
            return
        self.remember()
        self.data["review_status"] = "complete"
        self.data["full_document_reviewed"] = True
        self.data["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.data["last_saved_at"] = self.data["completed_at"]
        save_annotation(self.path, self.data)
        print("Document marked complete and saved.")

    @staticmethod
    def help():
        print(
            "Commands:\n"
            "  y  in-scope Pynchon-fiction quotation; confirm source and alterations\n"
            "  n  out of scope; choose an exclusion category\n"
            "  Enter/]/[  next / next / previous span     g  go to span number\n"
            "  l  show source-locator suggestions         e  edit current annotation\n"
            "  u  clear current label                     z  undo last saved change\n"
            "  m  add a missed block/single-quoted span   /  search complete article\n"
            "  v  page through the complete article       c  validate/mark document complete\n"
            "  d  return to document list                 q  quit interface\n"
            "  h  show this help"
        )

    # Interactive review loop: display spans, accept decisions (in-scope/out-scope/edit), and advance through the document.
    def run(self):
        # Begin at the first unresolved row; ``advance_to_open`` intentionally starts
        # after the current row and is used only after recording a decision.
        for index, row in enumerate(self.rows()):
            if row.get("is_in_scope") is None:
                self.index = index
                break
        while True:
            self.display_row()
            command = input("\nreview> ").strip().lower()
            if command == "y" and self.rows():
                self.label_positive()
            elif command == "n" and self.rows():
                self.label_negative()
            elif command in {"", "]"} and self.rows():
                self.index = (self.index + 1) % len(self.rows())
            elif command == "[" and self.rows():
                self.index = (self.index - 1) % len(self.rows())
            elif command == "g" and self.rows():
                answer = input(f"Span number [1-{len(self.rows())}]: ").strip()
                if answer.isdigit() and 1 <= int(answer) <= len(self.rows()):
                    self.index = int(answer) - 1
            elif command == "l" and self.rows():
                self._locator_results(self.rows()[self.index])
                input("Press Enter to continue …")
            elif command == "e" and self.rows():
                self.edit_row()
            elif command == "u" and self.rows():
                self.clear_label()
            elif command == "z":
                self.undo()
            elif command == "m":
                self.add_manual()
            elif command == "/":
                self.search_article()
                input("Press Enter to continue …")
            elif command == "v":
                pydoc.pager(
                    f"ARTICLE: {self.data['article']}\n"
                    "Inspect the entire body for missed block and single-quoted passages.\n\n"
                    + self.body
                )
            elif command == "c":
                self.complete_document()
            elif command == "d":
                return "documents"
            elif command == "q":
                return "quit"
            elif command == "h":
                self.help()
                input("Press Enter to continue …")
            else:
                print("Unknown command; enter h for help.")


def discover_documents(annotations, manifest, split):
    manifest_by_stem = {Path(row["text_path"]).stem: row for row in manifest}
    documents = []
    for path in sorted(Path(annotations).glob("*.annotation.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if split != "all" and data.get("split") != split:
            continue
        row = manifest_by_stem.get(data.get("article", ""))
        if row is None:
            raise ValueError(f"{path.name}: article is absent from the sample manifest")
        documents.append((path, data, row))
    return documents


def print_document_list(documents):
    print("\nAnnotation documents:")
    for number, (path, data, _) in enumerate(documents, 1):
        progress = annotation_progress(data)
        print(
            f"  {number:2d}. [{data.get('split', '?')[:3]}] {path.stem:<62} "
            f"{progress['labelled']:>3}/{progress['total']:<3}  {data.get('review_status', '?')}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(ANNOTATIONS_DIR))
    parser.add_argument("--manifest", default=str(SAMPLE_MANIFEST))
    parser.add_argument("--corpus-root", default=str(DEFAULT_CORPUS))
    parser.add_argument("--split", choices=("development", "test", "all"), default="development")
    parser.add_argument("--article", help="open the unique filename/article substring")
    parser.add_argument("--reviewer", default="", help="optional reviewer name or identifier")
    parser.add_argument("--list", action="store_true", help="list workload without entering review")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    documents = discover_documents(args.annotations, manifest, args.split)
    if args.article:
        needle = args.article.lower()
        documents = [item for item in documents if needle in item[0].name.lower()]
        if len(documents) != 1:
            raise SystemExit(f"--article matched {len(documents)} documents; provide a unique substring")
    if not documents:
        raise SystemExit(f"no {args.split} annotation documents found in {args.annotations}")
    if args.list:
        print_document_list(documents)
        return

    while True:
        if len(documents) == 1:
            chosen = documents[0]
        else:
            print_document_list(documents)
            answer = input("Document number, r to refresh, or q to quit: ").strip().lower()
            if answer == "q":
                return
            if answer == "r":
                documents = discover_documents(args.annotations, manifest, args.split)
                continue
            if not answer.isdigit() or not 1 <= int(answer) <= len(documents):
                print("Choose a listed document number.")
                continue
            chosen = documents[int(answer) - 1]
        try:
            session = ReviewSession(chosen[0], chosen[2], args.corpus_root, args.reviewer)
            result = session.run()
        except (EOFError, KeyboardInterrupt):
            print("\nExited; all confirmed changes were already autosaved.")
            return
        except ValueError as error:
            print(f"Cannot open document: {error}", file=sys.stderr)
            if len(documents) == 1:
                raise SystemExit(1)
            continue
        if result == "quit" or len(documents) == 1:
            return
        documents = discover_documents(args.annotations, manifest, args.split)


if __name__ == "__main__":
    main()
