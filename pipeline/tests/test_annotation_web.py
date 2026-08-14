import json
import shutil
import tempfile
import threading
import unittest
import urllib.parse
import urllib.request
from pathlib import Path

from annotation_web import AnnotationServer, AnnotationStore, Handler
from locate_quotes import Corpus
from project_paths import ANNOTATIONS_DIR, SAMPLE_MANIFEST


class BrowserAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.annotations = Path(self.temporary.name)
        self.dev_name = "2025_mentz_blue_american_5E3W868P.annotation.json"
        self.test_name = "2025_leupold_not_who_YKVFQH5I.annotation.json"
        shutil.copy2(ANNOTATIONS_DIR / self.dev_name, self.annotations / self.dev_name)
        shutil.copy2(ANNOTATIONS_DIR / self.test_name, self.annotations / self.test_name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_store_lists_the_folder_and_preserves_source_work_choice(self):
        """The store offers exactly what is in the annotator's folder.

        The application has no notion of splits: which documents an annotator can
        reach is decided by which files have been placed in their directory.
        """
        store = AnnotationStore(
            self.annotations, SAMPLE_MANIFEST,
            Path(__file__).resolve().parents[3] / "Reception_Corpus" / "corpus_articles",
            "reviewer_b",
        )
        names = [name for name, _data, _progress in store.documents()]
        self.assertEqual(names, sorted([self.dev_name, self.test_name]))
        path, data, _body, _manifest = store.load(self.dev_name)
        self.assertEqual(data["reviewer"], "reviewer_b")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["reviewer"], "reviewer_b")

    def test_dashboard_article_and_autosave_over_http(self):
        corpus = Path(__file__).resolve().parents[3] / "Reception_Corpus" / "corpus_articles"
        store = AnnotationStore(
            self.annotations, SAMPLE_MANIFEST, corpus, "reviewer_b"
        )
        try:
            server = AnnotationServer(("127.0.0.1", 0), Handler, store)
        except PermissionError:
            self.skipTest("managed sandbox prohibits binding a localhost test port")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            dashboard = urllib.request.urlopen(base + "/").read().decode()
            self.assertIn("Assigned documents", dashboard)
            self.assertIn("mentz", dashboard.lower())
            self.assertNotIn("letzler", dashboard.lower())

            query = urllib.parse.urlencode({"name": self.dev_name, "i": 0})
            article = urllib.request.urlopen(base + "/article?" + query).read().decode()
            self.assertIn("Save decision and continue", article)
            self.assertIn("Open published PDF", article)

            form = urllib.parse.urlencode({
                "token": server.token,
                "name": self.dev_name,
                "i": "0",
                "decision": "in",
                "pid": "P01::Mortality and Mercy in Vienna",
                "exclusion": "",
                "quotation_form": "quotation_marks",
                "alterations": "ellipsis, brackets",
                "notes": "browser test",
            }).encode()
            request = urllib.request.Request(base + "/save", data=form)
            urllib.request.urlopen(request).read()
            saved = json.loads((self.annotations / self.dev_name).read_text())
            row = saved["candidates"][0]
            self.assertTrue(row["is_in_scope"])
            self.assertEqual(row["true_pid"], "P01")
            self.assertEqual(row["true_work"], "Mortality and Mercy in Vienna")
            self.assertEqual(row["alterations"], ["ellipsis", "brackets"])
            self.assertIn("browser test", row["notes"])
            self.assertEqual(saved["review_status"], "in_progress")

            unlocated_form = urllib.parse.urlencode({
                "token": server.token,
                "name": self.dev_name,
                "i": "0",
                "decision": "in",
                "pid": "__unlocated__",
                "exclusion": "",
                "quotation_form": "quotation_marks",
                "alterations": "",
                "notes": "variant edition; not in the canonical plaintext",
            }).encode()
            urllib.request.urlopen(
                urllib.request.Request(base + "/save", data=unlocated_form)).read()
            saved = json.loads((self.annotations / self.dev_name).read_text())
            row = saved["candidates"][0]
            self.assertTrue(row["is_in_scope"])
            self.assertIs(row["in_primary_corpus"], False)
            self.assertEqual(row["true_pid"], "")
            self.assertIsNone(row["true_section_n"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_save_decision_confirms_chapter_and_rejects_out_of_range_section(self):
        """The section_n field is validated server-side, not trusted from the client.

        A valid number derives the label from the segmentation index and confirms
        the chapter/episode; an out-of-range number for the chosen work is rejected
        (left unresolved) rather than silently stored.
        """
        corpus = Path(__file__).resolve().parents[3] / "Reception_Corpus" / "corpus_articles"
        store = AnnotationStore(self.annotations, SAMPLE_MANIFEST, corpus, "reviewer_b")
        try:
            server = AnnotationServer(("127.0.0.1", 0), Handler, store)
        except PermissionError:
            self.skipTest("managed sandbox prohibits binding a localhost test port")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def save(section_n):
            form = urllib.parse.urlencode({
                "token": server.token,
                "name": self.dev_name,
                "i": "0",
                "decision": "in",
                "pid": "P01::Mortality and Mercy in Vienna",
                "section_n": section_n,
                "exclusion": "",
                "quotation_form": "",
                "alterations": "",
                "notes": "",
            }).encode()
            urllib.request.urlopen(urllib.request.Request(base + "/save", data=form)).read()
            return json.loads((self.annotations / self.dev_name).read_text())["candidates"][0]

        try:
            row = save("1")  # "Mortality and Mercy in Vienna" is P01 section 1.
            self.assertEqual(row["true_section_n"], 1)
            self.assertEqual(row["true_section_label"], "Mortality and Mercy in Vienna")

            row = save("99")  # P01 has 6 sections; 99 is out of range
            self.assertIsNone(row["true_section_n"])
            self.assertEqual(row["true_section_label"], "")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_locator_suggestion_renders_a_confirm_button(self):
        """A findable candidate's suggestion is a one-click confirm, not a lookup.

        The candidate quote is swapped for a verbatim slice of a primary text
        (dynamically pulled, not hardcoded, so no Pynchon prose sits in the test
        suite) purely so the locator is guaranteed to find it; this does not
        affect the article body checksum, which is validated independently.
        """
        corpus = Corpus()
        section = corpus.secs["P02"][0]
        words = corpus.texts["P02"][section["start"]:section["end"]].split()[:8]
        findable_quote = " ".join(words)

        path = self.annotations / self.dev_name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["candidates"][0]["quote"] = findable_quote
        path.write_text(json.dumps(data), encoding="utf-8")

        corpus = Path(__file__).resolve().parents[3] / "Reception_Corpus" / "corpus_articles"
        store = AnnotationStore(self.annotations, SAMPLE_MANIFEST, corpus, "reviewer_b")
        try:
            server = AnnotationServer(("127.0.0.1", 0), Handler, store)
        except PermissionError:
            self.skipTest("managed sandbox prohibits binding a localhost test port")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            article = urllib.request.urlopen(
                base + "/article?" + urllib.parse.urlencode({"name": self.dev_name, "i": 0, "locate": 1})
            ).read().decode()
            self.assertIn('data-pid-work="P02::V.', article)
            self.assertIn("data-section-n=", article)
            self.assertIn('onclick="confirmSuggestion(this)"', article)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
