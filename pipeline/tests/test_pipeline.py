import json
import csv
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from annotation_agreement import compare_documents  # noqa: E402
from baseline_text_matcher import (  # noqa: E402
    HYPHEN_HEAL, heal_shifts, merge_matches, to_article_offset,
)
from combine_candidates import merge_candidate_rows  # noqa: E402
from eval_compare import (  # noqa: E402
    aggregate_articles, bootstrap_intervals, maximum_cardinality_alignment,
    paired_bootstrap_difference, score_article, section_match,
)
from locate_quotes import Corpus, PRIMARY_FILES  # noqa: E402
from project_paths import CHAPTER_INDEX, SAMPLE_MANIFEST, SEGMENTATION_DIR  # noqa: E402
from quotation_spans import (  # noqa: E402
    extract_marked_spans, find_quote_offsets, under_minimum, word_count,
)
from review_annotations import (  # noqa: E402
    completion_issues, make_manual_span, metadata_normalized_body, save_annotation,
    set_negative, set_positive, verify_body_hash,
)


class CorpusStructureTests(unittest.TestCase):
    def test_expected_section_counts_and_labels(self):
        index = json.loads(CHAPTER_INDEX.read_text(encoding="utf-8"))
        self.assertEqual(index["P10"]["n_sections"], 39)
        self.assertEqual(index["P07"]["n_sections"], 70)
        p01 = json.loads((SEGMENTATION_DIR / "P01.sections.json").read_text(encoding="utf-8"))
        self.assertEqual(p01[0]["label"], "Mortality and Mercy in Vienna")

    def test_sections_cover_each_source_contiguously(self):
        for pid, source_path in PRIMARY_FILES.items():
            text = source_path.read_text(encoding="utf-8")
            sections = json.loads((SEGMENTATION_DIR / f"{pid}.sections.json").read_text())
            self.assertTrue(sections, pid)
            self.assertEqual(sections[0]["start"], 0, pid)
            self.assertEqual(sections[-1]["end"], len(text), pid)
            for left, right in zip(sections, sections[1:]):
                self.assertEqual(left["end"], right["start"], pid)
                self.assertLess(left["start"], left["end"], pid)

    def test_annotation_app_copies_are_in_sync_with_the_corpus(self):
        """The annotation app runs from its own copies, so the corpus can drift ahead.

        The app is launched from `annotation_workspace/annotation_app/`, and its
        `project_paths` resolves the sample manifest inside that directory — so
        changing the manifest here does nothing for the annotator until the copy
        is updated.

        Nothing warns about this at run time: the app works perfectly, on stale data.
        """
        app = BASE.parent / "annotation_workspace" / "annotation_app"
        if not app.is_dir():
            self.skipTest("annotation app directory not present")
        for name in ("annotation_web.py", "review_annotations.py", "quotation_spans.py",
                     "locate_quotes.py", "project_paths.py"):
            shipped = app / name
            if not shipped.exists():
                continue
            self.assertEqual(
                (BASE / name).read_text(encoding="utf-8"),
                shipped.read_text(encoding="utf-8"),
                f"annotation_app/{name} has drifted from the corpus copy — "
                f"the annotator is running different code")
        shipped_manifest = app / "evaluation" / "sample_manifest.csv"
        if shipped_manifest.exists():
            self.assertEqual(
                SAMPLE_MANIFEST.read_text(encoding="utf-8"),
                shipped_manifest.read_text(encoding="utf-8"),
                "annotation_app's sample manifest has drifted from the corpus copy — "
                "the annotator cannot open articles added since it was last synced")

    def test_no_sampled_article_repeats_a_large_block_of_itself(self):
        """Guard against the extraction artifact found in Harvey on 27 July 2026.

        Nothing in the pipeline noticed: a duplicate is valid text at valid offsets,
        so every integrity check passes. A repeated paragraph is normal (a scholar may
        quote a phrase twice); a repeated *block* of thousands of characters is not.
        """
        import re
        from collections import Counter
        with SAMPLE_MANIFEST.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        local = {
            path.stem: path
            for name in ("01_orbit_articles", "03_non_orbit_cc_articles")
            for path in (BASE.parent / "annotation_workspace" / name).glob("*.txt")
        }
        for row in rows:
            path = local.get(Path(row["text_path"]).stem)
            if path is None:
                continue
            body = path.read_text(encoding="utf-8").split("=" * 40, 1)[-1]
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body)
                          if len(p.strip()) > 200]
            counts = Counter(paragraphs)
            repeated = sum((n - 1) * len(p) for p, n in counts.items() if n > 1)
            self.assertLess(
                repeated, 2000,
                f"{path.stem}: {repeated} characters of duplicated block text — "
                f"check for an extraction artifact, as in Harvey C6VQPDG4")

    def test_sample_manifest_has_fixed_disjoint_splits(self):
        with SAMPLE_MANIFEST.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        development = {row["zotero_key"] for row in rows if row["split"] == "development"}
        test = {row["zotero_key"] for row in rows if row["split"] == "test"}
        self.assertEqual(development, {"YT54NC7G", "7L5Z97IZ", "5E3W868P", "YUZMXRKY"})
        # This assertion failing is the intended behaviour for any *unintended*
        # change to the split — update it only alongside a recorded decision.
        self.assertEqual(
            test,
            {"US8FE2Q5", "9RSYBDHU", "C6VQPDG4", "E2TDDQA5", "KF8LYIPQ",
             "7IIYT69Y", "8Q6F6MLD", "9VU7EIQJ", "LS5YAQLE", "W6SQ6GJ8",
             "89JQFT3W", "YKVFQH5I"},
        )
        test_strata = [row["stratum"] for row in rows if row["split"] == "test"]
        self.assertEqual(len(test_strata), 12)
        self.assertEqual(len(set(test_strata)), 5)
        self.assertTrue(development.isdisjoint(test))
        # Pinned so neither imbalance fades from
        # view, since both must be reported as sampling limitations.
        self.assertEqual(
            dict(Counter(test_strata)),
            {"non_orbit": 1, "orbit_2012-2014": 4, "orbit_2015-2017": 3,
             "orbit_2018-2020": 2, "orbit_2021-2025": 2},
        )


class LocatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus()

    def shadow_ticket_slice(self):
        section = self.corpus.secs["P10"][0]
        text = self.corpus.texts["P10"][section["start"]:section["end"]]
        # Skip the numeric heading and construct the fixture dynamically so no
        # passage from the primary text is embedded in the test suite.
        words = text.split()[1:25]
        self.assertGreaterEqual(len(words), 20)
        return words

    def test_shadow_ticket_locates_to_chapter_one(self):
        result = self.corpus.locate(" ".join(self.shadow_ticket_slice()))
        self.assertIsNotNone(result)
        self.assertEqual(result["pid"], "P10")
        self.assertEqual(result["section"], "Chapter 1")
        self.assertEqual(result["section_n"], 1)

    def test_located_section_n_matches_the_segmentation_index(self):
        # section_n is the exact, matchable id gold/results compare on -- the
        # label string alone ("Chapter One" vs "Chapter 1" vs "Episode 1") is
        # not consistent enough across works to serve that role.
        result = self.corpus.locate(" ".join(self.shadow_ticket_slice()))
        section = self.corpus.section_at("P10", result["char_offset"])
        self.assertEqual(result["section_n"], section["n"])
        self.assertIsInstance(result["section_n"], int)

    def test_altered_prefix_still_locates(self):
        words = self.shadow_ticket_slice()
        words[0] = "[altered]"
        result = self.corpus.locate(" ".join(words))
        self.assertIsNotNone(result)
        self.assertEqual(result["pid"], "P10")

    def test_bracketed_insertion_does_not_defeat_the_match(self):
        """A bracketed WHOLE word is an editorial insertion the source does not have.

        Found during Harvey annotation: "enormous eyes, lambent [and] extravagantly
        lashed", where Lot 49 has a comma in place of the inserted "and". Dropping
        only the brackets — the rule that correctly handles "[w]here" — left "and" in
        the query as a token the source lacks, and the fragment then failed to align
        at all: no match, not a low score. The insertion reading is now tried as well
        and the better of the two kept.
        """
        source = self.corpus.texts["P03"]
        start = source.index("enormous eyes")
        original = source[start:start + 44]
        self.assertIn("lambent", original)
        # Reconstruct the scholar's reading: the comma after "lambent" replaced by an
        # inserted word in brackets.
        quoted = original.replace("lambent,", "lambent [and]", 1)
        self.assertNotEqual(quoted, original)
        result = self.corpus.locate(quoted)
        self.assertIsNotNone(result, "bracketed insertion defeated the match")
        self.assertEqual(result["pid"], "P03")
        self.assertGreaterEqual(result["score"], 0.75)

    def test_bracketed_alteration_inside_a_word_is_not_stripped(self):
        """The insertion rule must not swallow "[w]here" or "improvise[s]".

        Those brackets replace characters of a real word, so their content has to
        survive; only a run delimited by whitespace on both sides is an insertion.
        """
        from locate_quotes import drop_bracket_insertions
        self.assertEqual(drop_bracket_insertions("[w]here it began"), "[w]here it began")
        self.assertEqual(drop_bracket_insertions("improvise[s] a routine"),
                         "improvise[s] a routine")
        self.assertEqual(drop_bracket_insertions("lambent [and] extravagantly"),
                         "lambent extravagantly")
        # An altered opening word still locates, which is what would break if the
        # leading-bracket case were treated as an insertion.
        source = self.corpus.texts["P03"]
        start = source.index("enormous eyes")
        altered = "[h]is " + source[start:start + 44]
        result = self.corpus.locate(altered)
        self.assertIsNotNone(result)
        self.assertEqual(result["pid"], "P03")

    def test_altered_opening_word_does_not_shift_to_the_previous_section(self):
        """A quotation that opens a chapter, with its first word altered, must not
        be attributed to chapter n-1. The estimated start (anchor position minus
        the anchor's offset within the query) can precede the chapter boundary, and
        the index map compounds it by recording a collapsed separator run at the
        run's first character. Before the fix this mislabelled 35 of 36 sampled
        chapter openings; it is invisible on development, where locus accuracy is
        1.000, so only a direct test protects it."""
        for pid in ("P04", "P06", "P07", "P10"):
            for section in self.corpus.secs[pid][1:6]:
                words = self.corpus.texts[pid][
                    section["start"]:section["start"] + 500].split()
                if len(words) < 24:
                    continue
                result = self.corpus.locate("Whereupon " + " ".join(words[1:24]))
                self.assertIsNotNone(result, f"{pid} {section['label']} did not locate")
                self.assertEqual(
                    (result["pid"], result["section_n"]), (pid, section["n"]),
                    f"{pid} {section['label']} located to "
                    f"{result['pid']} section {result['section_n']}")


class EvaluationTests(unittest.TestCase):
    def test_duplicate_predictions_only_match_one_gold_span(self):
        predictions = [{"quote": "one two three four five six"}] * 2
        gold = [{"quote": "one two three four five six"}]
        self.assertEqual(len(maximum_cardinality_alignment(predictions, gold, 0.6)), 1)

    def test_detection_recall_is_stratified_by_quotation_form(self):
        gold = [
            {"quote": "found span with five plus words", "quotation_form": "curly_double",
             "notes": ""},
            {"quote": "missed block quotation set off without marks",
             "quotation_form": "manual_block", "notes": "pooling_recovered; from audit"},
        ]
        candidates = [{"id": "a-01", "quote": "found span with five plus words"}]
        summed = aggregate_articles([score_article(("a", gold, candidates, {}), 0.6)])
        strata = summed["recall_by_stratum"]
        self.assertEqual(set(strata),
                         {"form:curly_double", "form:manual_block",
                          "provenance:annotator", "provenance:system_recovered"})
        self.assertEqual(strata["form:manual_block"]["fn"], 1)
        self.assertEqual(strata["form:curly_double"]["tp"], 1)
        self.assertEqual(strata["provenance:annotator"]["recall"], 1.0)
        self.assertEqual(strata["provenance:system_recovered"]["fn"], 1)

    def test_section_match_requires_both_work_and_section(self):
        gold = {"true_pid": "P04", "true_section_n": 9}
        self.assertTrue(section_match({"pid": "P04", "section_n": 9}, gold))
        self.assertFalse(section_match({"pid": "P04", "section_n": 10}, gold),
                          "same work, wrong chapter/episode must not match")
        self.assertFalse(section_match({"pid": "P02", "section_n": 9}, gold),
                          "a matching section number under the wrong work is not a locus match")
        self.assertIsNone(section_match({"pid": "P04", "section_n": 9}, {"true_pid": "P04"}),
                           "gold with no confirmed section is undefined, not False")

    def test_eval_compare_reports_section_accuracy(self):
        gold = [
            {"quote": "one two three four five", "true_pid": "P04",
             "true_section_n": 9, "quotation_form": "curly_double", "notes": ""},
            {"quote": "six seven eight nine ten", "true_pid": "P04",
             "true_section_n": 12, "quotation_form": "curly_double", "notes": ""},
        ]
        candidates = [
            {"id": "a-01", "quote": "one two three four five"},
            {"id": "a-02", "quote": "six seven eight nine ten"},
        ]
        results = {
            # Correct work and section.
            "a-01": {"id": "a-01", "confirmed_pynchon_fiction": True, "pid": "P04",
                     "section_n": 9, "candidate_pids": "P04"},
            # Correct work, wrong section.
            "a-02": {"id": "a-02", "confirmed_pynchon_fiction": True, "pid": "P04",
                     "section_n": 11, "candidate_pids": "P04"},
        }
        row = score_article(("a", gold, candidates, results), 0.6)
        self.assertEqual(row["section_ok"], 1)
        self.assertEqual(row["section_denom"], 2)
        summed = aggregate_articles([row])
        self.assertAlmostEqual(summed["section_acc"], 0.5)
        # Top-source accuracy is easier to satisfy than the locus accuracy it
        # composes with -- both candidates land the correct work.
        self.assertAlmostEqual(summed["top_acc"], 1.0)

    def test_unmatched_quote_mark_does_not_swallow_paragraphs(self):
        text = '“valid quoted span has six whole words”\n\n“orphan opening mark\n\nother text”'
        spans = extract_marked_spans(text, min_words=5)
        self.assertEqual([row["quote"] for row in spans], ["valid quoted span has six whole words"])

    def test_word_count_treats_bracketed_alterations_as_one_word(self):
        self.assertEqual(word_count("[t]he death of magic"), 4)
        self.assertEqual(word_count("[k]eep[s] it bouncing."), 3)
        self.assertEqual(word_count("improvise[s] a MILF-night routine"), 4)
        # Ellipsis brackets and whole-word insertions are unaffected.
        self.assertEqual(word_count("growth of capabilities [capacités] here"), 5)
        self.assertEqual(word_count("one […] two three four five"), 5)

    def test_marked_span_trimming_keeps_offsets_exact(self):
        text = "before “  valid quoted span has six whole words  ” after"
        span = extract_marked_spans(text)[0]
        self.assertEqual(span["quote"], "valid quoted span has six whole words")
        self.assertEqual(text[span["start_char"]:span["end_char"]], span["quote"])

    def test_quote_offset_recovery_tolerates_whitespace(self):
        text = "before A quoted\npassage with six words here after"
        start, end, occurrences = find_quote_offsets(
            text, "A quoted passage with six words here"
        )
        self.assertEqual(text[start:end], "A quoted\npassage with six words here")
        self.assertEqual(occurrences, 1)

    def test_quote_offset_recovery_tolerates_typography_differences(self):
        # A generative detector may return straight apostrophes and three dots
        # where the article prints curly apostrophes and an ellipsis character.
        text = "before “tho’ sailors sang on … beside Marlow’s own chart” after"
        start, end, occurrences = find_quote_offsets(
            text, "tho' sailors sang on ... beside Marlow's own chart"
        )
        self.assertEqual(
            text[start:end], "tho’ sailors sang on … beside Marlow’s own chart"
        )
        self.assertEqual(occurrences, 1)

    def test_quote_offset_recovery_can_select_repeated_occurrence(self):
        text = "repeat these five whole words now; repeat these five whole words now"
        first = find_quote_offsets(text, "repeat these five whole words now", occurrence_index=0)
        second = find_quote_offsets(text, "repeat these five whole words now", occurrence_index=1)
        self.assertLess(first[0], second[0])
        self.assertEqual(first[2], 2)

    def test_hybrid_merges_overlap_and_preserves_provenance(self):
        body = "zero one two three four five six seven eight nine"
        rows = [
            {"id": "q1", "article": "a", "model_tag": "quid", "quote": "two three four five",
             "article_start_char": 9, "article_end_char": 28},
            {"id": "g1", "article": "a", "model_tag": "generative",
             "quote": "one two three four five six", "article_start_char": 5,
             "article_end_char": 32},
            {"id": "g2", "article": "a", "model_tag": "generative", "quote": "eight nine",
             "article_start_char": 40, "article_end_char": 50},
        ]
        merged = merge_candidate_rows(rows, body)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["quote"], "one two three four five six")
        self.assertEqual(merged[0]["proposed_by"], ["generative", "quid"])

    def test_hybrid_collapses_exact_duplicates_into_one_span(self):
        body = "zero one two three four five six seven eight nine"
        rows = [
            {"id": "q1", "article": "a", "model_tag": "quid", "quote": "two three four five",
             "article_start_char": 9, "article_end_char": 28},
            {"id": "g1", "article": "a", "model_tag": "generative",
             "quote": "two three four five", "article_start_char": 9,
             "article_end_char": 28},
        ]
        merged = merge_candidate_rows(rows, body)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["quote"], "two three four five")
        self.assertEqual(merged[0]["proposed_by"], ["generative", "quid"])

    def test_hybrid_keeps_the_containing_span_for_a_nested_quotation(self):
        body = "zero one two three four five six seven eight nine"
        rows = [
            {"id": "g1", "article": "a", "model_tag": "generative",
             "quote": "one two three four five six", "article_start_char": 5,
             "article_end_char": 32},
            {"id": "q1", "article": "a", "model_tag": "quid", "quote": "three four",
             "article_start_char": 13, "article_end_char": 23},
        ]
        merged = merge_candidate_rows(rows, body)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["quote"], "one two three four five six")
        self.assertEqual(merged[0]["proposed_by"], ["generative", "quid"])

    def test_hybrid_keeps_a_requoted_phrase_inside_a_block_quotation(self):
        """A re-quotation is two gold spans, so the merge must not collapse it.

        Containment alone deleted the inner span, costing a true positive whenever no
        other member of the union happened to re-propose it. Two development gold spans
        are affected — e.g. Smith's "gods hiding in the landscape", which the closed
        detector proposes both on its own and inside a 27-word span. See the 27 July
        2026 audit entry. The inner span survives only when it could be an in-scope
        quotation itself: at least five words, and a small enough fraction of its
        container to be a fragment rather than a noisy copy.
        """
        body = ("Discussing the passage, she quotes the whole of it: the gods hiding "
                "in the landscape were there before anyone thought to look for them, "
                "and returns later to the gods hiding in the landscape alone.")
        outer = ("the gods hiding in the landscape were there before anyone thought "
                 "to look for them")
        inner = "gods hiding in the landscape"
        rows = [
            {"id": "g1", "article": "a", "model_tag": "generative", "quote": outer,
             "article_start_char": body.index(outer),
             "article_end_char": body.index(outer) + len(outer)},
            {"id": "g2", "article": "a", "model_tag": "generative", "quote": inner,
             "article_start_char": body.index(inner),
             "article_end_char": body.index(inner) + len(inner)},
        ]
        merged = merge_candidate_rows(rows, body)
        self.assertEqual([row["quote"] for row in merged], [outer, inner])

    def test_hybrid_still_collapses_a_boundary_variant_of_one_quotation(self):
        """The dedup case the containment rule exists for must keep working.

        Two detectors proposing the same quotation with slightly different boundaries
        are one prediction, not two; left uncollapsed the second becomes a false
        positive under one-to-one alignment.
        """
        body = "She writes that the gods hiding in the landscape were there before."
        outer = "the gods hiding in the landscape were there"
        inner = "gods hiding in the landscape were there"
        rows = [
            {"id": "q1", "article": "a", "model_tag": "quid", "quote": outer,
             "article_start_char": body.index(outer),
             "article_end_char": body.index(outer) + len(outer)},
            {"id": "g1", "article": "a", "model_tag": "generative", "quote": inner,
             "article_start_char": body.index(inner),
             "article_end_char": body.index(inner) + len(inner)},
        ]
        merged = merge_candidate_rows(rows, body)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["quote"], outer)
        self.assertEqual(merged[0]["proposed_by"], ["generative", "quid"])

    def test_hybrid_keeps_both_spans_of_a_partial_overlap(self):
        """Plan §4.4 collapses a span only when another *contains* it. A partial
        overlap is two competing boundaries, not a nesting: keeping only the
        longer one discarded a correct generative span on Mentz 5E3W868P-q0012
        in favour of a Quid span that ran past the quotation's end."""
        body = "zero one two three four five six seven eight nine"
        rows = [
            {"id": "g1", "article": "a", "model_tag": "generative",
             "quote": "one two three four", "article_start_char": 5,
             "article_end_char": 23},
            {"id": "q1", "article": "a", "model_tag": "quid",
             "quote": "two three four five six seven", "article_start_char": 9,
             "article_end_char": 38},
        ]
        merged = merge_candidate_rows(rows, body)
        self.assertEqual(len(merged), 2)
        self.assertEqual([row["quote"] for row in merged],
                         ["one two three four", "two three four five six seven"])

    def test_hybrid_does_not_bridge_partial_overlaps_through_a_contained_span(self):
        """B sits inside both A and C, but A and C only partially overlap. Taking
        the transitive closure would collapse all three onto the longest (C) and
        discard the precise span A — the Mentz 5E3W868P-q0012 failure, reachable
        again with three detectors. A and C must both survive."""
        body = "zero one two three four five six seven eight nine ten eleven twelve"
        rows = [
            {"id": "A", "article": "a", "model_tag": "generative", "quote": body[5:32],
             "article_start_char": 5, "article_end_char": 32},
            {"id": "B", "article": "a", "model_tag": "heuristic", "quote": body[13:23],
             "article_start_char": 13, "article_end_char": 23},
            {"id": "C", "article": "a", "model_tag": "quid", "quote": body[13:50],
             "article_start_char": 13, "article_end_char": 50},
        ]
        merged = merge_candidate_rows(rows, body)
        self.assertEqual([(r["article_start_char"], r["article_end_char"])
                          for r in merged], [(5, 32), (13, 50)])
        for row in merged:
            self.assertIn("heuristic", row["proposed_by"])

    def test_hybrid_keeps_disjoint_spans_separate(self):
        body = "zero one two three four five six seven eight nine"
        rows = [
            {"id": "g1", "article": "a", "model_tag": "generative", "quote": "one two",
             "article_start_char": 5, "article_end_char": 12},
            {"id": "q1", "article": "a", "model_tag": "quid", "quote": "eight nine",
             "article_start_char": 38, "article_end_char": 48},
        ]
        merged = merge_candidate_rows(rows, body)
        self.assertEqual(len(merged), 2)
        self.assertEqual([row["proposed_by"] for row in merged],
                         [["generative"], ["quid"]])

    def test_annotation_agreement_reports_scope_source_and_locus(self):
        base = {
            "article": "a", "manually_added_spans": [],
            "candidates": [
                {"start_char": 10, "end_char": 20, "quote": "one two three four five",
                 "is_in_scope": True, "true_pid": "P03", "true_section_n": 1},
                {"start_char": 30, "end_char": 40, "quote": "six seven eight nine ten",
                 "is_in_scope": False, "true_pid": ""},
            ],
        }
        metrics = compare_documents(base, json.loads(json.dumps(base)))
        self.assertEqual(metrics["span_f1"], 1.0)
        self.assertEqual(metrics["scope_agreement"], 1.0)
        self.assertEqual(metrics["source_agreement"], 1.0)
        self.assertEqual(metrics["locus_n"], 1)
        self.assertEqual(metrics["locus_agreement"], 1.0)

    def test_locus_agreement_is_not_counted_without_source_agreement(self):
        a = {
            "article": "a", "manually_added_spans": [],
            "candidates": [{"start_char": 10, "end_char": 20, "quote": "one two three four five",
                            "is_in_scope": True, "true_pid": "P03", "true_section_n": 1}],
        }
        b = json.loads(json.dumps(a))
        b["candidates"][0]["true_pid"] = "P04"
        b["candidates"][0]["true_section_n"] = 1  # same number, different (disagreed) work
        metrics = compare_documents(a, b)
        self.assertEqual(metrics["source_agreement"], 0.0)
        self.assertEqual(metrics["locus_n"], 0, "a locus comparison under a source disagreement is not meaningful")

    def test_article_bootstrap_is_deterministic(self):
        articles = [
            {"articles": 1, "candidates": 2, "gold": 1, "det_tp": 1, "det_fp": 1,
             "det_fn": 0, "end_tp": 1, "end_fp": 0, "end_fn": 0,
             "top_ok": 1, "alt_ok": 1, "top_denom": 1,
             "section_ok": 1, "section_denom": 1, "det_f1": 2 / 3},
            {"articles": 1, "candidates": 1, "gold": 1, "det_tp": 1, "det_fp": 0,
             "det_fn": 0, "end_tp": 1, "end_fp": 0, "end_fn": 0,
             "top_ok": 1, "alt_ok": 1, "top_denom": 1,
             "section_ok": 1, "section_denom": 1, "det_f1": 1.0},
        ]
        first = bootstrap_intervals(articles, iterations=50, seed=7)
        second = bootstrap_intervals(articles, iterations=50, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first["det_f1"]["low"], first["det_f1"]["high"])

    def test_paired_bootstrap_uses_shared_articles(self):
        base = {
            "articles": 1, "candidates": 1, "gold": 1, "det_tp": 1, "det_fp": 0,
            "det_fn": 0, "end_tp": 1, "end_fp": 0, "end_fn": 0,
            "top_ok": 1, "alt_ok": 1, "top_denom": 1,
            "section_ok": 1, "section_denom": 1, "det_f1": 1.0,
        }
        worse = dict(base, candidates=2, det_fp=1, det_f1=2 / 3)
        a = [dict(base, stem="shared"), dict(base, stem="a_only")]
        b = [dict(worse, stem="shared"), dict(worse, stem="b_only")]
        result = paired_bootstrap_difference(a, b, iterations=20, seed=3)
        self.assertEqual(result["shared_articles"], 1)
        self.assertGreater(result["differences"]["det_f1"]["point"], 0)

    def test_annotation_interface_manual_span_and_completion_validation(self):
        body = "prefix these six words form a manual quoted span suffix"
        data = {
            "zotero_key": "ABC123", "candidates": [], "manually_added_spans": [],
        }
        start = body.index("these")
        end = body.index(" suffix")
        row = make_manual_span(data, body, start, end)
        self.assertEqual(row["annotation_id"], "ABC123-m0001")
        self.assertTrue(any("scope label unresolved" in issue
                            for issue in completion_issues(data, body)))
        set_positive(row, "P03", "The Crying of Lot 49", ["ellipsis"])
        self.assertTrue(any("chapter/episode unresolved" in issue
                            for issue in completion_issues(data, body)))
        set_positive(row, "P03", "The Crying of Lot 49", ["ellipsis"], section_n=1)
        self.assertEqual(row["true_section_label"], "Chapter 1")
        self.assertEqual(completion_issues(data, body), [])

    def test_unlocated_in_scope_span_needs_note_then_completes(self):
        from review_annotations import set_positive_unlocated
        body = "prefix these six words form a manual quoted span suffix"
        data = {"zotero_key": "ABC123", "candidates": [], "manually_added_spans": []}
        row = make_manual_span(data, body, body.index("these"), body.index(" suffix"))
        set_positive_unlocated(row, ["wording"])
        self.assertTrue(row["is_in_scope"])
        self.assertIs(row["in_primary_corpus"], False)
        self.assertEqual(row["true_pid"], "")
        self.assertIsNone(row["true_section_n"])
        self.assertTrue(any("explanatory note" in issue
                            for issue in completion_issues(data, body)))
        row["notes"] = "quotes a variant edition; not in the canonical plaintext"
        self.assertEqual(completion_issues(data, body), [])

    def test_annotation_interface_requires_exclusion_reason(self):
        body = "these six words form a proposed quoted span"
        row = {
            "annotation_id": "q1", "start_char": 0, "end_char": len(body),
            "quote": body, "is_in_scope": False, "exclusion_category": "",
        }
        data = {"candidates": [row], "manually_added_spans": []}
        self.assertTrue(any("exclusion category missing" in issue
                            for issue in completion_issues(data, body)))
        set_negative(row, "other_author_or_critic")
        self.assertEqual(completion_issues(data, body), [])

    def test_corpus_body_hash_normalization_and_autosave_backup(self):
        raw = "Title: Synthetic\n\n==========\n\nbody has several words here   \n"
        normalized = metadata_normalized_body(raw)
        import hashlib
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        self.assertTrue(verify_body_hash(raw, digest)[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.annotation.json"
            save_annotation(path, {"value": 1})
            save_annotation(path, {"value": 2})
            backup = path.with_suffix(path.suffix + ".bak")
            self.assertEqual(json.loads(path.read_text())["value"], 2)
            self.assertEqual(json.loads(backup.read_text())["value"], 1)


class StraightSingleExtractionTests(unittest.TestCase):
    """The conservative straight-single tier added 23 July 2026."""

    def test_space_bounded_straight_single_quotation_is_extracted(self):
        text = "He wrote that 'the profane church of America rises here' in 2012."
        spans = extract_marked_spans(text)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["quotation_form"], "straight_single")
        self.assertEqual(spans[0]["quote"], "the profane church of America rises here")

    def test_apostrophes_and_possessives_produce_no_spans(self):
        text = ("Pynchon's readers' favorite scenes aren't quoted in Miller's "
                "article, and the scholars' consensus isn't challenged at all.")
        self.assertEqual(extract_marked_spans(text), [])

    def test_single_nested_inside_double_defers_to_enclosing_span(self):
        text = ('She notes that "he called it \'a whistling drifts over the harbor\' '
                'in the famous opening line of the novel" without comment.')
        spans = extract_marked_spans(text)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["quotation_form"], "straight_double")


class TextMatcherOffsetTests(unittest.TestCase):
    """text-matcher reports offsets into its own hyphen-healed copy of the article.

    Its `Text.preprocess()` rejoins words broken across a line before tokenising, so
    every span it returns after the first heal is short by two characters per heal.
    Slicing the article with those offsets returns quotations shifted off their word
    boundaries — silently, since the result is still plausible prose. These tests pin
    the correction, and run without text-matcher installed: only the offset arithmetic
    is exercised, not the matcher.
    """

    def heal(self, text):
        """Reproduce text_matcher.matcher.Text.preprocess()."""
        return HYPHEN_HEAL.sub(r"\1\2", text)

    def test_no_hyphenation_leaves_offsets_untouched(self):
        body = "A whistling drifts over the harbor. It has occurred once more."
        shifts = heal_shifts(body)
        self.assertEqual(shifts, [])
        self.assertEqual(to_article_offset(11, shifts), 11)

    def test_span_after_a_heal_maps_back_to_the_word_boundary(self):
        body = ("Scholars ac- commodate the text, and later note that "
                "a whistling drifts over the harbor.")
        healed = self.heal(body)
        quote = "a whistling drifts over the harbor"
        start = healed.index(quote)
        shifts = heal_shifts(body)
        mapped = to_article_offset(start, shifts)
        self.assertEqual(healed[start:start + len(quote)], quote)
        self.assertEqual(body[mapped:mapped + len(quote)], quote)
        self.assertEqual(mapped - start, 2)

    def test_every_healed_offset_round_trips_across_multiple_heals(self):
        body = ("The dis- placed persons at the mar- gins of the war are re- membered "
                "only as a whistling drifts over the harbor, and the roc- ket falls.")
        healed = self.heal(body)
        shifts = heal_shifts(body)
        self.assertEqual(len(shifts), 4)
        for position, char in enumerate(healed):
            # Healed text drops only the "- " pairs, so every surviving character must
            # map to the identical character in the article.
            self.assertEqual(body[to_article_offset(position, shifts)], char, position)

    def test_merge_keeps_every_proposing_work_for_an_overlapping_span(self):
        merged = merge_matches([
            {"start": 100, "end": 200, "pid": "P04"},
            {"start": 150, "end": 260, "pid": "P07"},
            {"start": 900, "end": 950, "pid": "P04"},
        ])
        self.assertEqual(len(merged), 2)
        self.assertEqual((merged[0]["start"], merged[0]["end"]), (100, 260))
        self.assertEqual(merged[0]["candidate_pids"], ["P04", "P07"])
        self.assertEqual(merged[1]["candidate_pids"], ["P04"])


class CompileGoldTests(unittest.TestCase):
    """compile_gold_v2.py must compile good files and report—not abort on—bad ones."""

    def _template(self, article, status="complete", quotes=None):
        return {
            "schema_version": "2.0-working", "article": article,
            "corpus_id": f"CORPUS-{article}", "zotero_key": article,
            "split": "development", "body_sha256": "0" * 64,
            "review_status": status, "candidates": quotes or [],
            "manually_added_spans": [],
        }

    def _span(self, i, **overrides):
        row = {
            "annotation_id": f"K-q{i:04d}", "quote": "a marked quotation of five words",
            "start_char": 10, "end_char": 45, "is_in_scope": True,
            "exclusion_category": "", "true_pid": "P02", "true_work": "The Crying of Lot 49",
            "true_section_n": 3, "true_section_label": "Chapter 3",
            "in_primary_corpus": True, "quotation_form": "curly_double",
            "alterations": [], "notes": "",
        }
        row.update(overrides)
        return row

    def test_duplicate_annotation_ids_fail_compilation(self):
        """Manual span ids are per-annotator serials and collide across annotators.

        second annotator's identically numbered ones. `gold_id` is `annotation_id`, and
        the evaluator keys results on it, so a merge that introduced a collision would
        lose a span silently. This fires before that can reach gold.
        """
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            annotations = Path(directory) / "annotations"
            gold = Path(directory) / "gold"
            annotations.mkdir()
            clashing = self._template("clash", quotes=[
                self._span(1), self._span(1, quote="a second marked quotation of words")])
            (annotations / "clash.annotation.json").write_text(
                json.dumps(clashing), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(BASE / "compile_gold_v2.py"),
                 "--annotations", str(annotations), "--output", str(gold)],
                capture_output=True, text=True, cwd=str(BASE))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate annotation ids", result.stdout)
            self.assertFalse((gold / "clash.gold.json").exists())

    def test_the_same_in_scope_span_twice_fails_compilation(self):
        """Seeding once produced the same span two and three times over.

        Those were all out of scope so nothing reached gold, but an in-scope repeat
        would be scored twice — once as a match and once as a false positive for every
        system that found it only once.
        """
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            annotations = Path(directory) / "annotations"
            gold = Path(directory) / "gold"
            annotations.mkdir()
            twice = self._template("twice", quotes=[
                self._span(1), self._span(2)])   # distinct ids, identical offsets
            (annotations / "twice.annotation.json").write_text(
                json.dumps(twice), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(BASE / "compile_gold_v2.py"),
                 "--annotations", str(annotations), "--output", str(gold)],
                capture_output=True, text=True, cwd=str(BASE))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("appears more than once", result.stdout)

    def test_in_scope_span_under_five_words_never_reaches_gold(self):
        """The five-word minimum is part of the frozen target, so a shorter span
        cannot be gold. Nothing else enforces it on a *seeded* candidate:
        `set_positive` does not check length, and the templates were seeded before
        the 23 July `word_count()` bracket fix, so six candidates that cleared the
        bar when seeded no longer do — three of them in the second annotator's set.
        The compiler is the only gate that sees both annotators' returned work.
        """
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            annotations = Path(directory) / "annotations"
            gold = Path(directory) / "gold"
            annotations.mkdir()
            # "[s]omeone in a hurry" is the real Leupold case: four words once the
            # bracketed alteration counts as one word with the word it modifies.
            short = self._template("short_span", quotes=[
                self._span(1, quote="[s]omeone in a hurry")])
            (annotations / "short_span.annotation.json").write_text(
                json.dumps(short), encoding="utf-8")
            fine = self._template("fine_article", quotes=[self._span(1)])
            (annotations / "fine_article.annotation.json").write_text(
                json.dumps(fine), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(BASE / "compile_gold_v2.py"),
                 "--annotations", str(annotations), "--output", str(gold)],
                capture_output=True, text=True, cwd=str(BASE))
            # Excluded, not fatal: a returning annotation must not be able to block
            # the pipeline over a span that is out of scope by arithmetic.
            self.assertEqual(result.returncode, 0)
            compiled = json.loads((gold / "short_span.gold.json").read_text())
            self.assertEqual(compiled["quotes"], [])
            # ...and the exclusion is on the record in the gold file itself, so it is
            # auditable rather than silent.
            self.assertEqual(len(compiled["excluded_out_of_scope"]), 1)
            excluded = compiled["excluded_out_of_scope"][0]
            self.assertEqual(excluded["gold_id"], "K-q0001")
            self.assertEqual(excluded["words"], 4)
            self.assertIn("EXCLUDED from gold", result.stdout)
            self.assertIn("K-q0001", result.stdout)
            # A clean file still compiles, and records an empty exclusion list rather
            # than omitting the field, so "nothing excluded" is distinguishable from
            # "this gold predates the check".
            fine_gold = json.loads((gold / "fine_article.gold.json").read_text())
            self.assertEqual(len(fine_gold["quotes"]), 1)
            self.assertEqual(fine_gold["excluded_out_of_scope"], [])

    def test_out_of_scope_predictions_are_excluded_from_scoring_and_counted(self):
        """Gold cannot contain a sub-minimum span, so a sub-minimum prediction can
        only ever be a guaranteed false positive. It is excluded from scoring and
        counted, the count being the compliance statistic — and the exclusion is
        reversible with --score-out-of-scope, because it moves several systems.
        """
        predictions = [
            {"quote": "a marked quotation of five words"},
            {"quote": "too short"},
        ]
        gold = [{"quote": "a marked quotation of five words"}]
        kept = [p for p in predictions if not under_minimum(p["quote"])]
        self.assertEqual(len(kept), 1)
        # Scored with the short span present it is an unavoidable false positive;
        # excluded, precision measures detection.
        self.assertEqual(len(maximum_cardinality_alignment(predictions, gold, 0.6)), 1)
        self.assertEqual(len(maximum_cardinality_alignment(kept, gold, 0.6)), 1)

    def test_completion_blocks_an_in_scope_span_under_five_words(self):
        data = self._template("short_span", status="in_progress", quotes=[
            self._span(1, quote="[s]omeone in a hurry")])
        issues = completion_issues(data)
        self.assertTrue(any("five-word minimum" in issue for issue in issues), issues)
        self.assertTrue(any("pynchon_under_five_words" in issue for issue in issues),
                        issues)
        # The same span excluded is fine, and a five-word span in scope is fine.
        excluded = self._template("ok", quotes=[
            self._span(1, quote="[s]omeone in a hurry", is_in_scope=False,
                       exclusion_category="pynchon_under_five_words")])
        self.assertEqual(
            [i for i in completion_issues(excluded) if "five-word" in i], [])

    def test_bad_file_is_reported_and_skipped_not_fatal_to_others(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            annotations = Path(directory) / "annotations"
            gold = Path(directory) / "gold"
            annotations.mkdir()
            good = self._template("good_article", quotes=[self._span(1)])
            (annotations / "good_article.annotation.json").write_text(
                json.dumps(good), encoding="utf-8")
            (annotations / "broken.annotation.json").write_text(
                "{not valid json", encoding="utf-8")
            missing = self._template("missing_section", quotes=[
                self._span(1, true_section_n=None, true_section_label="")])
            (annotations / "missing_section.annotation.json").write_text(
                json.dumps(missing), encoding="utf-8")
            unlocated = self._template("unlocated_ok", quotes=[
                self._span(1, true_pid="", true_work="", true_section_n=None,
                           true_section_label="", in_primary_corpus=False,
                           notes="variant edition; confirmed not in canonical plaintext")])
            (annotations / "unlocated_ok.annotation.json").write_text(
                json.dumps(unlocated), encoding="utf-8")
            unlocated_bad = self._template("unlocated_no_note", quotes=[
                self._span(1, true_pid="", true_work="", true_section_n=None,
                           true_section_label="", in_primary_corpus=False, notes="")])
            (annotations / "unlocated_no_note.annotation.json").write_text(
                json.dumps(unlocated_bad), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(BASE / "compile_gold_v2.py"),
                 "--annotations", str(annotations), "--output", str(gold)],
                capture_output=True, text=True, cwd=str(BASE))
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((gold / "good_article.gold.json").exists())
            compiled = json.loads((gold / "good_article.gold.json").read_text())
            self.assertEqual(len(compiled["quotes"]), 1)
            self.assertIn("broken.annotation.json", result.stdout)
            self.assertIn("missing_section", result.stdout)
            self.assertIn("unlocated_no_note", result.stdout)
            unlocated_gold = json.loads((gold / "unlocated_ok.gold.json").read_text())
            self.assertEqual(len(unlocated_gold["quotes"]), 1)
            self.assertIs(unlocated_gold["quotes"][0]["in_primary_corpus"], False)
            self.assertIsNone(unlocated_gold["quotes"][0]["true_section_n"])


class OllamaSalvageTests(unittest.TestCase):
    """The 22 July content-salvage acceptance path for the Ollama backend."""

    def setUp(self):
        from detect_quotes import build_schema, salvage_detection
        self.Detection = build_schema()
        self.salvage = salvage_detection

    def test_fenced_json_block_is_accepted_and_validated(self):
        content = ('Here are the quotations.\n```json\n{"quotes": [{"quote": '
                   '"a marked quotation of five words"}]}\n```\n')
        det, label = self.salvage(content, self.Detection)
        self.assertEqual(label, "content_json")
        self.assertEqual(len(det.quotes), 1)
        self.assertEqual(det.quotes[0].attributed_work, "")

    def test_unterminated_string_ending_in_typographic_quote_is_repaired(self):
        content = ('```json\n{\n  "quotes": [\n    {\n'
                   '      "quote": "heroes remain “still small boys inside”\n'
                   '    },\n    {\n'
                   '      "quote": "another quotation of five words",\n'
                   '      "attributed_work": "Vineland"\n    }\n  ]\n}\n```')
        det, label = self.salvage(content, self.Detection)
        self.assertEqual(label, "content_json_repaired")
        self.assertEqual(len(det.quotes), 2)
        self.assertTrue(det.quotes[0].quote.endswith("inside”"))

    def test_bare_array_shape_is_accepted(self):
        """The model sometimes answers with a bare array instead of the
        {"quotes": [...]} envelope. Scanning only for {...} spans from the first
        entry's brace to the last entry's, which is never valid JSON — observed
        26 July 2026 discarding 29 sound quotations from one windowed response."""
        content = ('```json\n[{"quote": "a marked quotation of five words",\n'
                   '  "attributed_work": "Inherent Vice"}]\n```')
        det, label = self.salvage(content, self.Detection)
        self.assertEqual(label, "content_json")
        self.assertEqual(len(det.quotes), 1)
        self.assertEqual(det.quotes[0].attributed_work, "Inherent Vice")

    def test_one_corrupt_entry_does_not_discard_the_others(self):
        """Whole-document parsing is all-or-nothing: a single glitched line cost
        the 28 good entries around it. Per-entry recovery runs only after every
        whole-document path fails, so articles those paths already accept are
        untouched."""
        content = ('[\n {"quote": "first marked quotation five words"},\n'
                   ' de: "Everybody. Nobody. Does it matter?",\n'
                   ' {"quote": "third marked quotation five words"}\n]')
        stats = {}
        det, label = self.salvage(content, self.Detection, stats=stats)
        self.assertEqual(label, "content_json_per_entry")
        self.assertEqual(len(det.quotes), 2)
        self.assertIn("recovered", stats)

    def test_per_entry_recovery_never_fires_when_whole_document_parses(self):
        """Guards the safety property that development results cannot move."""
        content = '{"quotes": [{"quote": "a marked quotation of five words"}]}'
        det, label = self.salvage(content, self.Detection)
        self.assertEqual(label, "content_json")

    def test_content_with_no_json_still_raises(self):
        with self.assertRaises(ValueError):
            self.salvage("No quotations were found in this article.",
                         self.Detection)

    def test_bare_json_without_fence_is_accepted(self):
        content = 'Sure: {"quotes": []} — nothing found.'
        det, label = self.salvage(content, self.Detection)
        self.assertEqual(label, "content_json")
        self.assertEqual(det.quotes, [])

    def test_null_and_numeric_optional_fields_coerce_instead_of_failing(self):
        det = self.Detection.model_validate({"quotes": [
            {"quote": "a marked quotation of five words",
             "attributed_page": None, "attributed_work": None,
             "detector_guess": None, "confidence": None},
            {"quote": "another marked quotation five words", "attributed_page": 42},
        ]})
        self.assertEqual(det.quotes[0].attributed_page, "")
        self.assertEqual(det.quotes[1].attributed_page, "42")

    def test_second_fenced_block_used_when_first_is_not_the_answer(self):
        content = ('An example of the format:\n```json\n{"example": true}\n```\n'
                   'And here is the real extraction:\n```json\n{"quotes": '
                   '[{"quote": "a marked quotation of five words"}]}\n```\n')
        det, label = self.salvage(content, self.Detection)
        self.assertEqual(label, "content_json")
        self.assertEqual(len(det.quotes), 1)

    def test_repair_does_not_truncate_multiline_quote_followed_by_prose(self):
        from detect_quotes import _close_unterminated_strings
        raw = ('{\n  "quotes": [\n    {\n'
               '      "quote": "a block quotation that continues\n'
               'onto a second printed line of prose",\n'
               '      "attributed_work": "V."\n    }\n  ]\n}')
        self.assertEqual(_close_unterminated_strings(raw), raw)

    def test_prose_without_json_raises(self):
        with self.assertRaises(ValueError):
            self.salvage("I found several quotations from Gravity's Rainbow.",
                         self.Detection)

    def test_literal_newlines_inside_block_quotation_strings_parse(self):
        content = ('```json\n{"quotes": [{"quote": "a block quotation\nspanning '
                   'two printed lines"}]}\n```')
        det, label = self.salvage(content, self.Detection)
        self.assertEqual(len(det.quotes), 1)
        self.assertIn("\n", det.quotes[0].quote)


class TestJsonLineComments(unittest.TestCase):
    """gemma4 annotates uncertain fields with `// …`, which is not legal JSON. One
    comment invalidated the whole document AND defeated per-entry recovery for that
    entry, silently dropping exactly the quotations the model was least sure of."""

    def setUp(self):
        from detect_quotes import build_schema, salvage_detection
        self.salvage = salvage_detection
        self.Detection = build_schema()

    def test_commented_document_is_recovered_whole(self):
        content = ('```json\n[\n  {"quote": "a marked quotation of five words",\n'
                   '   "attributed_page": "753"}, // Note: page uncertain\n'
                   '  {"quote": "another marked quotation of six words here"}\n]\n```')
        det, label = self.salvage(content, self.Detection)
        self.assertEqual(len(det.quotes), 2)
        self.assertEqual(label, "content_json_repaired")

    def test_comment_inside_an_entry_does_not_lose_that_entry(self):
        content = ('```json\n[\n  {"quote": "first marked quotation of five words",\n'
                   '   "attributed_page": "753" // contextually likely\n  },\n'
                   '  {"quote": "second marked quotation of five words"}\n]\n```')
        det, _label = self.salvage(content, self.Detection)
        self.assertEqual(len(det.quotes), 2)

    def test_double_slash_inside_a_quoted_string_is_preserved(self):
        from detect_quotes import _strip_json_line_comments
        raw = '{"quote": "see http://example.com/a for the passage cited"}'
        self.assertEqual(_strip_json_line_comments(raw), raw)

    def test_escaped_quote_does_not_desynchronise_the_string_scan(self):
        from detect_quotes import _strip_json_line_comments
        raw = '{"quote": "he said \\"no\\" firmly", "attributed_page": "7"} // note'
        self.assertEqual(
            _strip_json_line_comments(raw),
            '{"quote": "he said \\"no\\" firmly", "attributed_page": "7"}')

    def test_uncommented_json_still_accepted_as_content_json(self):
        # The comment strip must not change the label of a clean response.
        content = ('```json\n{"quotes": [{"quote": "a marked quotation of five '
                   'words"}]}\n```')
        _det, label = self.salvage(content, self.Detection)
        self.assertEqual(label, "content_json")


class TestToolCallSwallowSignature(unittest.TestCase):
    """Ollama 0.32.1 discards a gemma4 native tool call whose JSON repair fails,
    returning an empty message for a generation of thousands of tokens (diagnosed
    26 July 2026). The signature must be distinguished from the other empty-ish
    outcomes so the harness retries in the way that actually recovers."""

    def signature(self, **overrides):
        from detect_quotes import tool_call_swallowed
        record = {"done_reason": "stop", "eval_count": 2640,
                  "content_chars": 0, "n_tool_calls": 0}
        record.update(overrides)
        return tool_call_swallowed(record)

    def test_swallow_signature_detected(self):
        # The captured failing run: 2,640 tokens generated, nothing returned.
        self.assertTrue(self.signature())

    def test_output_truncation_is_not_a_swallow(self):
        # done_reason "length" means the cap was hit; the reply is cut mid-output
        # and a tools-free retry would not help.
        self.assertFalse(self.signature(done_reason="length"))

    def test_content_bearing_response_is_not_a_swallow(self):
        # Format drift into plain content is the salvageable path, not a swallow.
        self.assertFalse(self.signature(content_chars=12000))

    def test_successful_tool_call_is_not_a_swallow(self):
        self.assertFalse(self.signature(n_tool_calls=1))

    def test_trivial_generation_is_not_a_swallow(self):
        # A near-empty generation is a different failure; requiring substantial
        # eval_count keeps the recovery retry off unrelated cases.
        self.assertFalse(self.signature(eval_count=1))

    def test_missing_eval_count_is_not_a_swallow(self):
        self.assertFalse(self.signature(eval_count=None))


class TestSwallowRecoveryRetry(unittest.TestCase):
    """The recovery retry must re-ask WITHOUT the tools declaration: nothing
    client-side can recover bytes the server discarded, so the only route back is
    the content path that salvage already handles."""

    GOOD_CONTENT = ('```json\n{"quotes": [{"quote": "a marked quotation of five '
                    'words"}]}\n```')

    def run_detect(self, first_response):
        import detect_quotes
        from detect_quotes import build_schema
        sent = []

        def fake_chat(host, payload):
            sent.append(payload)
            if len(sent) == 1:
                return first_response
            return {"done_reason": "stop", "eval_count": 3300,
                    "message": {"role": "assistant", "content": self.GOOD_CONTENT}}

        original = detect_quotes._ollama_chat
        detect_quotes._ollama_chat = fake_chat
        try:
            quotes, runinfo = detect_quotes.detect_ollama(
                "gemma4:26b-mlx", "article body", build_schema())
        finally:
            detect_quotes._ollama_chat = original
        return quotes, runinfo, sent

    def test_swallow_triggers_a_tools_free_retry(self):
        swallowed = {"done_reason": "stop", "eval_count": 2640,
                     "message": {"role": "assistant", "content": ""}}
        quotes, runinfo, sent = self.run_detect(swallowed)
        self.assertEqual(len(sent), 2)
        self.assertIn("tools", sent[0])
        self.assertNotIn("tools", sent[1])
        self.assertNotIn("tool_choice", sent[1])
        self.assertTrue(runinfo["tool_call_swallow_recovery"])
        self.assertTrue(runinfo["attempts"][0]["tool_call_swallow_suspected"])
        self.assertFalse(runinfo["attempts"][1]["tools_enabled"])
        self.assertEqual(len(quotes), 1)

    def test_ordinary_format_drift_still_gets_the_reminder_retry(self):
        # Content arrived but held no usable JSON: the tools declaration is not
        # implicated, so the predeclared reminder retry stands.
        drifted = {"done_reason": "stop", "eval_count": 900,
                   "message": {"role": "assistant",
                               "content": "I found several quotations."}}
        quotes, runinfo, sent = self.run_detect(drifted)
        self.assertEqual(len(sent), 2)
        self.assertIn("tools", sent[1])
        self.assertNotIn("tool_call_swallow_recovery", runinfo)
        # The assistant turn was non-empty, so it is carried into the retry.
        self.assertEqual(sent[1]["messages"][-2]["role"], "assistant")
        self.assertEqual(len(quotes), 1)

    def test_truncated_generation_is_rejected_and_retried_fresh(self):
        # Salvage can parse entries out of a severed generation, but that is a
        # PREFIX of the answer, not the answer. It must not be scored.
        truncated = {"done_reason": "length", "eval_count": 24576,
                     "message": {"role": "assistant",
                                 "content": self.GOOD_CONTENT + '{"quote": "cut off'}}
        quotes, runinfo, sent = self.run_detect(truncated)
        first = runinfo["attempts"][0]
        self.assertTrue(first["rejected_truncated"])
        self.assertIsNone(first["accepted_via"])
        self.assertTrue(runinfo["truncation_retry"])
        # A fresh retry: tools still declared, and no assistant turn carried over.
        self.assertIn("tools", sent[1])
        self.assertEqual([m["role"] for m in sent[1]["messages"]],
                         [m["role"] for m in sent[0]["messages"]])
        self.assertEqual(len(quotes), 1)

    def test_truncation_retry_does_not_carry_the_runaway_into_the_prompt(self):
        # The reminder path would append the assistant turn — on a runaway that is
        # the whole truncated generation, inflating the prompt for no benefit.
        runaway = {"done_reason": "length", "eval_count": 24576,
                   "message": {"role": "assistant", "content": "x" * 119032}}
        _quotes, _runinfo, sent = self.run_detect(runaway)
        self.assertLess(sum(len(m["content"]) for m in sent[1]["messages"]),
                        sum(len(m["content"]) for m in sent[0]["messages"]) + 1000)

    def test_empty_assistant_turn_is_not_carried_into_a_reminder_retry(self):
        # Below the swallow threshold, so this takes the reminder path — but the
        # model must not be scolded while unable to see its own (empty) answer.
        trivial = {"done_reason": "stop", "eval_count": 1,
                   "message": {"role": "assistant", "content": ""}}
        quotes, runinfo, sent = self.run_detect(trivial)
        roles = [m["role"] for m in sent[1]["messages"]]
        self.assertNotIn("assistant", roles)
        self.assertEqual(roles[-1], "user")


if __name__ == "__main__":
    unittest.main()
