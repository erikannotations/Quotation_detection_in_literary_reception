#!/usr/bin/env python3
"""Local browser interface for human quotation annotation.

This is a GUI wrapper around the same schema and validation functions used by
``review_annotations.py``. It binds only to 127.0.0.1, opens the default browser,
autosaves every decision, and requires no third-party Python packages.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import secrets
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from locate_quotes import Corpus
from project_paths import SAMPLE_MANIFEST
from quotation_spans import article_body, find_quote_offsets, word_count
from review_annotations import (
    EXCLUSIONS,
    PIDS,
    WORK_TITLES,
    annotation_progress,
    annotation_rows,
    completion_issues,
    load_manifest,
    make_manual_span,
    save_annotation,
    set_negative,
    set_positive,
    set_positive_unlocated,
    update_context,
    verify_body_hash,
)

BASE = Path(__file__).resolve().parent
DEFAULT_CORPUS = BASE.parent.parent / "Reception_Corpus" / "corpus_articles"

CSS = """
:root { color-scheme: light; --ink:#222; --muted:#62666b; --line:#d7d9dc;
  --paper:#fff; --wash:#f4f5f6; --blue:#245ea8; --green:#246b45; --red:#9b3030; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:var(--wash); font:17px/1.5 Georgia,serif; }
header { background:#20252b; color:white; padding:14px 24px; }
header a { color:white; }
nav { max-width:1100px; margin:auto; display:flex; gap:18px; align-items:center; }
nav .grow { flex:1; }
main { max-width:1100px; margin:24px auto; padding:0 18px 60px; }
.panel { background:var(--paper); border:1px solid var(--line); border-radius:10px;
  padding:20px; margin:16px 0; box-shadow:0 1px 2px #0000000c; }
.docs { display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:14px; }
.doc { background:white; border:1px solid var(--line); border-radius:9px; padding:16px; }
h1,h2,h3 { line-height:1.2; }
h1 { font-size:1.75rem; margin:0 0 8px; } h2 { font-size:1.3rem; }
a { color:var(--blue); }
button,.button,select,input,textarea { font:16px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
button,.button { display:inline-block; border:0; border-radius:7px; background:var(--blue);
  color:white; padding:10px 15px; cursor:pointer; text-decoration:none; }
.secondary { background:#59616a; } .danger { background:var(--red); }
.success { background:var(--green); }
label { display:block; font:bold 15px/1.3 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  margin:13px 0 5px; }
select,input[type=text],textarea { width:100%; border:1px solid #aeb2b7; border-radius:6px;
  background:white; padding:9px; }
textarea { min-height:75px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.context { white-space:pre-wrap; overflow-wrap:anywhere; color:#3d4145; }
.quote { white-space:pre-wrap; overflow-wrap:anywhere; border-left:5px solid var(--blue);
  background:#eef4fb; padding:14px 17px; margin:12px 0; font-size:1.08rem; }
.status { display:inline-block; border-radius:999px; padding:3px 9px; background:#e4e7ea;
  font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.in { background:#dff2e7; color:#174c30; } .out { background:#f6e2e2; color:#722; }
.message { border-left:5px solid var(--blue); background:#edf4fc; padding:12px 15px; }
.warning { border-left-color:#c07617; background:#fff3df; }
.error { border-left-color:var(--red); background:#fae8e8; }
.progress { height:11px; background:#e1e3e5; border-radius:8px; overflow:hidden; }
.progress span { display:block; height:100%; background:var(--green); }
.row-nav { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.row-nav .grow { flex:1; }
.muted { color:var(--muted); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.suggestion { border:1px solid var(--line); border-radius:7px; padding:9px; margin:7px 0; }
::highlight(recorded) { background-color:#e3efe0; }
::highlight(editing) { background-color:#ffe3a3; }
select:disabled, input:disabled { background:var(--wash); color:var(--muted); cursor:not-allowed; }
pre.article { white-space:pre-wrap; overflow-wrap:anywhere; background:white; border:1px solid var(--line);
  padding:22px; border-radius:8px; user-select:text; }
table { border-collapse:collapse; width:100%; } td,th { border-bottom:1px solid var(--line); padding:8px; text-align:left; }
@media(max-width:700px){ .grid{grid-template-columns:1fr;} body{font-size:16px;} }
"""


def esc(value):
    return html.escape(str(value or ""), quote=True)


def now_utc():
    return datetime.now(timezone.utc).isoformat()


class AnnotationStore:
    def __init__(self, annotations, manifest, corpus_root, reviewer):
        self.annotations = Path(annotations).resolve()
        self.annotations.mkdir(parents=True, exist_ok=True)
        self.corpus_root = Path(corpus_root).resolve()
        self.reviewer = reviewer
        self.manifest = load_manifest(manifest)
        self.by_stem = {Path(row["text_path"]).stem: row for row in self.manifest}
        self.locator = None
        self.locator_cache = {}

    def documents(self):
        docs = []
        for path in sorted(self.annotations.glob("*.annotation.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            progress = annotation_progress(data)
            docs.append((path.name, data, progress))
        return docs

    def load(self, name):
        safe = Path(name).name
        if safe != name or not safe.endswith(".annotation.json"):
            raise ValueError("Invalid annotation filename")
        path = self.annotations / safe
        if not path.exists():
            raise FileNotFoundError(safe)
        data = json.loads(path.read_text(encoding="utf-8"))
        manifest_row = self.by_stem.get(data.get("article", ""))
        if not manifest_row:
            raise ValueError("Article is absent from the frozen sample manifest")
        # Prefer a local copy alongside the annotation workspace so a reviewer
        # package doesn't need access to the full corpus repo.
        local_source = self.annotations.parent / manifest_row["text_path"]
        source_path = local_source if local_source.exists() else self.corpus_root / manifest_row["text_path"]
        if not source_path.exists():
            # The corpus reorganised its directories on 30 July 2026 —
            # `03_non_orbit_cc_articles` became `03_non_orbit_articles` — while its
            # metadata kept recording the old path, so a manifest path can name a
            # directory that no longer exists. Annotators are insulated by their local
            # copies, which take priority above; this fallback keeps the corpus route
            # working by filename, since the stem is the stable identifier and the
            # checksum below still has to match whatever is found.
            matches = sorted(self.corpus_root.rglob(Path(manifest_row["text_path"]).name))
            if matches:
                source_path = matches[0]
        raw = source_path.read_text(encoding="utf-8")
        valid, actual = verify_body_hash(raw, data["body_sha256"])
        if not valid:
            raise ValueError(
                f"Article checksum mismatch; expected {data['body_sha256']}, found {actual}"
            )
        if self.reviewer and not data.get("reviewer"):
            data["reviewer"] = self.reviewer
            save_annotation(path, data)
        return path, data, article_body(raw), manifest_row

    @staticmethod
    def commit(path, data):
        if data.get("review_status") == "complete":
            data["full_document_reviewed"] = False
            data["completed_at"] = ""
        data["review_status"] = "in_progress"
        data["last_saved_at"] = now_utc()
        save_annotation(path, data)

    def suggestions(self, row):
        key = (row.get("annotation_id", ""), row.get("quote", ""))
        if key not in self.locator_cache:
            if self.locator is None:
                self.locator = Corpus()
            self.locator_cache[key] = self.locator.locate_all(row.get("quote", ""), limit=5)
        return self.locator_cache[key]

    def pdf_path(self, data):
        key = data.get("zotero_key", "")
        workspace = self.annotations.parent
        roots = [
            # Local copies placed alongside the reviewer's annotation workspace
            # take priority so annotators don't need access to the full corpus repo.
            workspace / "orbit_articles" / "PDFs",
            workspace / "non_orbit_cc_articles" / "PDFs",
            workspace / "non_orbit_cc_articles" / "PDFs CC BY",
            workspace / "non_orbit_cc_articles" / "PDFs CC BY ND",
            self.corpus_root / "01_orbit_articles" / "PDFs",
            # The corpus reorganised its non-Orbit PDFs from "PDFs CC BY" to "PDFs"
            # during its 30 July 2026 extraction audit. Both names are searched: the
            # annotators' workspace copies still use the old one, and a lookup that
            # tracked only the current upstream layout would silently drop the
            # "Open published PDF" link for every non-Orbit article.
            self.corpus_root / "03_non_orbit_cc_articles" / "PDFs",
            self.corpus_root / "03_non_orbit_cc_articles" / "PDFs CC BY",
            self.corpus_root / "03_non_orbit_cc_articles" / "PDFs CC BY ND",
        ]
        for root in roots:
            matches = list(root.glob(f"*_{key}.pdf")) if root.exists() else []
            if matches:
                return matches[0]
        # Last resort, for the same reason as the article-text fallback: the corpus
        # reorganises its directories from time to time, and the zotero key in the
        # filename is the stable identifier. Searching by name survives a rename that
        # an enumerated list of directories cannot.
        for root in (self.annotations.parent, self.corpus_root):
            if root.exists():
                matches = sorted(root.rglob(f"*_{key}.pdf"))
                if matches:
                    return matches[0]
        return None


class AnnotationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, store):
        super().__init__(address, handler)
        self.store = store
        self.token = secrets.token_urlsafe(24)


class Handler(BaseHTTPRequestHandler):
    server_version = "CHRAnnotation/1.0"

    def log_message(self, fmt, *args):
        # Keep the launcher window quiet except for startup/shutdown information.
        return

    @property
    def store(self):
        return self.server.store

    def page(self, title, content):
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} — CHR annotation</title><style>{CSS}</style></head><body>
<header><nav><strong>CHR quotation annotation</strong><span class="grow"></span>
<a href="/">Documents</a><a href="/help">Help</a>
<form action="/shutdown" method="post" style="margin:0"><input type="hidden" name="token" value="{self.server.token}">
<button class="secondary" type="submit">Finish for now</button></form></nav></header>
<main>{content}</main></body></html>"""

    def send_html(self, title, content, status=200):
        data = self.page(title, content).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, path, message="", level="message"):
        if message:
            separator = "&" if "?" in path else "?"
            path += separator + urllib.parse.urlencode({"msg": message, "level": level})
        self.send_response(303)
        self.send_header("Location", path)
        self.end_headers()

    def query(self):
        parsed = urllib.parse.urlparse(self.path)
        return parsed.path, {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()}

    def form(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise ValueError("Submitted form is too large")
        raw = self.rfile.read(length).decode("utf-8")
        return {k: v[-1] for k, v in urllib.parse.parse_qs(raw, keep_blank_values=True).items()}

    def verify_token(self, form):
        if not secrets.compare_digest(form.get("token", ""), self.server.token):
            raise PermissionError("Invalid form token; reload the page and try again")

    @staticmethod
    def message(query):
        if not query.get("msg"):
            return ""
        level = query.get("level", "message")
        return f'<div class="panel {esc(level)}">{esc(query["msg"])}</div>'

    def do_GET(self):
        try:
            path, query = self.query()
            if path == "/":
                self.render_dashboard(query)
            elif path == "/article":
                self.render_article(query)
            elif path == "/full":
                self.render_full_article(query)
            elif path == "/pdf":
                self.serve_pdf(query)
            elif path == "/help":
                self.render_help()
            else:
                self.send_html("Not found", "<h1>Page not found</h1>", 404)
        except Exception as error:
            self.send_html("Error", f'<div class="panel error"><h1>Unable to continue</h1><p>{esc(error)}</p><p><a href="/">Return to documents</a></p></div>', 400)

    def do_POST(self):
        try:
            path, _query = self.query()
            form = self.form()
            self.verify_token(form)
            if path == "/save":
                self.save_decision(form)
            elif path == "/boundary":
                self.save_boundary(form)
            elif path == "/manual":
                self.add_manual(form)
            elif path == "/complete":
                self.complete(form)
            elif path == "/shutdown":
                self.send_html("Stopped", "<div class='panel'><h1>Everything is saved</h1><p>You may close this browser tab.</p></div>")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.send_html("Not found", "<h1>Page not found</h1>", 404)
        except Exception as error:
            name = locals().get("form", {}).get("name", "")
            index = locals().get("form", {}).get("i", "0")
            target = "/"
            if name:
                target = "/article?" + urllib.parse.urlencode({"name": name, "i": index})
            self.redirect(target, str(error), "error")

    def render_dashboard(self, query):
        cards = []
        for name, data, progress in self.store.documents():
            total, labelled = progress["total"], progress["labelled"]
            percent = int(100 * labelled / total) if total else int(data.get("review_status") == "complete") * 100
            cards.append(f"""<div class="doc"><span class="status">{esc(data.get('split'))}</span>
<h2>{esc(data.get('article', name).replace('_', ' '))}</h2>
<p class="muted">{labelled}/{total} annotations reviewed · {esc(data.get('review_status','unreviewed'))}</p>
<div class="progress"><span style="width:{percent}%"></span></div><p>
<a class="button" href="/article?{urllib.parse.urlencode({'name':name,'i':0})}">{'Review' if labelled == 0 else 'Continue'}</a></p></div>""")
        if not cards:
            cards.append("<div class='panel warning'><h2>No documents assigned</h2><p>The project owner has not yet placed an independent-review document in this workspace. Nothing is wrong; ask the owner which article has been assigned.</p></div>")
        content = self.message(query) + f"<h1>Assigned documents</h1><p>Select a document to begin or continue. Every change is saved automatically.</p><div class='docs'>{''.join(cards)}</div>"
        self.send_html("Documents", content)

    def render_article(self, query):
        name = query.get("name", "")
        path, data, body, _manifest = self.store.load(name)
        rows = annotation_rows(data)
        if not rows:
            index = 0
            row = None
        else:
            index = max(0, min(int(query.get("i", "0")), len(rows) - 1))
            row = rows[index]
        progress = annotation_progress(data)
        percent = int(100 * progress["labelled"] / progress["total"]) if progress["total"] else 0
        previous_i = max(0, index - 1)
        next_i = min(max(0, len(rows) - 1), index + 1)
        pdf = f"<a class='button secondary' target='_blank' href='/pdf?{urllib.parse.urlencode({'name':name})}'>Open published PDF</a>" if self.store.pdf_path(data) else ""
        head = f"""{self.message(query)}<div class="row-nav"><div><h1>{esc(data['article'].replace('_',' '))}</h1>
<div class="muted">{progress['labelled']}/{progress['total']} reviewed · {progress['positives']} included · status {esc(data.get('review_status'))}</div></div>
<span class="grow"></span>{pdf}<a class="button secondary" href="/full?{urllib.parse.urlencode({'name':name})}">Read full article</a></div>
<div class="progress"><span style="width:{percent}%"></span></div>"""
        if row is None:
            detail = "<div class='panel'><p>No automatically suggested annotations. Read the full article, add any marked Pynchon quotations, then complete the document.</p></div>"
        else:
            state = "Unreviewed"
            state_class = ""
            if row.get("is_in_scope") is True:
                source = row.get("true_pid") or "source unresolved"
                if row.get("true_section_label"):
                    source += f" — {row['true_section_label']}"
                elif row.get("true_pid"):
                    source += " — chapter/episode unresolved"
                state, state_class = f"Included — {source}", "in"
            elif row.get("is_in_scope") is False:
                state, state_class = f"Excluded — {row.get('exclusion_category')}", "out"
            suggestions = ""
            if query.get("locate") == "1":
                try:
                    found = self.store.suggestions(row)
                except (FileNotFoundError, OSError):
                    suggestions = "<p class='message warning'>Locator data is unavailable in this workspace. Record uncertainty rather than guessing.</p>"
                else:
                    if found:
                        items = []
                        for result in found:
                            pid_work_value = f"{result['pid']}::{result['novel']}"
                            items.append(
                                f"<div class='suggestion'><strong>{esc(result['pid'])}: {esc(result['novel'])}</strong> — {esc(result['section'])} "
                                f"<span class='muted'>score {result['score']:.3f}</span> "
                                f"<button type=\"button\" class=\"secondary\" data-pid-work=\"{esc(pid_work_value)}\" "
                                f"data-section-n=\"{esc(result['section_n'])}\" data-section-label=\"{esc(result['section'])}\" "
                                f"onclick=\"confirmSuggestion(this)\">Confirm this source and chapter/episode</button></div>"
                            )
                        suggestions = "<h3>Possible sources in Pynchon's texts</h3><p class='muted'>These are search results, not labels. Confirm from the quotation and article context. \"Confirm\" fills in the source and chapter/episode fields below; you can still adjust them or the alterations before saving.</p>" + "".join(items)
                    else:
                        suggestions = "<p class='message warning'>The locator found no source suggestion. Record uncertainty rather than guessing.</p>"
            pid_options = ['<option value="">Unresolved / decide later</option>']
            unlocated_selected = " selected" if (
                row.get("is_in_scope") and row.get("in_primary_corpus") is False
            ) else ""
            pid_options.append(
                f'<option value="__unlocated__"{unlocated_selected}>'
                "In scope — NOT locatable in the ten primary texts (explain in notes)</option>"
            )
            source_choices = [
                ("P01", "Slow Learner"),
                ("P01", "Mortality and Mercy in Vienna"),
            ] + [(pid, WORK_TITLES[pid]) for pid in PIDS if pid != "P01"]
            for pid, work in source_choices:
                selected = " selected" if (
                    row.get("true_pid") == pid and row.get("true_work") == work
                ) else ""
                value = f"{pid}::{work}"
                pid_options.append(
                    f'<option value="{esc(value)}"{selected}>{pid} — {esc(work)}</option>'
                )
            ex_options = ['<option value="">Choose a reason</option>']
            for key, description in EXCLUSIONS:
                selected = " selected" if row.get("exclusion_category") == key else ""
                ex_options.append(f'<option value="{esc(key)}"{selected}>{esc(description)}</option>')
            decision = "unreviewed" if row.get("is_in_scope") is None else ("in" if row["is_in_scope"] else "out")
            decision_options = "".join(
                f'<option value="{value}"{" selected" if decision == value else ""}>{label}</option>'
                for value, label in (("unreviewed","Not yet decided"),("in","Include: Pynchon fiction quotation"),("out","Exclude"))
            )
            section_n_value = row.get("true_section_n")
            section_confirmed_text = f"Confirmed: {row['true_section_label']}" if row.get("true_section_label") else ""
            start, end = row["start_char"], row["end_char"]
            before = body[max(0, start-350):start]
            after = body[end:min(len(body), end+350)]
            detail = f"""<div class="panel"><div class="row-nav"><span class="status {state_class}">{esc(state)}</span><span class="grow"></span><span class="muted">Annotation {index+1} of {len(rows)} · {esc(row.get('annotation_id'))}</span></div>
<div class="context">{esc(before)}</div><div class="quote">{esc(row.get('quote'))}</div><div class="context">{esc(after)}</div>
<p><a href="/article?{urllib.parse.urlencode({'name':name,'i':index,'locate':1})}">Search Pynchon's texts for this quotation</a> ·
<a href="/full?{urllib.parse.urlencode({'name':name,'edit':index})}">Correct boundaries using the full article</a></p>{suggestions}
<form id="decisionForm" action="/save" method="post"><input type="hidden" name="token" value="{self.server.token}"><input type="hidden" name="name" value="{esc(name)}"><input type="hidden" name="i" value="{index}">
<div class="grid"><div><label>Decision</label><select name="decision">{decision_options}</select></div>
<div><label>Source work (for included quotations)</label><select name="pid">{''.join(pid_options)}</select></div></div>
<div class="grid"><div><label>Chapter/episode number (for included quotations)</label>
<input type="number" min="1" name="section_n" value="{esc(section_n_value) if section_n_value is not None else ''}" placeholder="Use a locator suggestion above, or the section number">
<span id="sectionConfirmed" class="muted">{esc(section_confirmed_text)}</span></div>
<div><label>Exclusion reason (for excluded spans)</label><select name="exclusion">{''.join(ex_options)}</select></div></div>
<div class="grid"><div><label>Quotation form</label><input type="text" name="quotation_form" list="formVocab" value="{esc(row.get('quotation_form'))}" placeholder="e.g. curly_double or manual_block">
<datalist id="formVocab"><option value="curly_double"><option value="straight_double"><option value="curly_single"><option value="manual_block"><option value="manual_single_quote"><option value="manual_other"></datalist></div>
<div><label>Alterations by scholar, comma separated</label><input type="text" name="alterations" value="{esc(', '.join(row.get('alterations',[])))}" placeholder="ellipsis, brackets, wording"></div></div>
<label>Notes</label><textarea name="notes" placeholder="Optional: explain uncertainty or an unusual decision">{esc(row.get('notes'))}</textarea>
<p style="text-align:right"><button class="success" type="submit">Save decision and continue</button></p></form>
<script>
function syncFieldStates() {{
  const form = document.getElementById('decisionForm');
  const decision = form.querySelector('select[name="decision"]').value;
  const included = decision === 'in';
  for (const name of ['pid', 'section_n', 'alterations']) {{
    form.querySelector('[name="' + name + '"]').disabled = !included;
  }}
  form.querySelector('select[name="exclusion"]').disabled = decision !== 'out';
}}
function confirmSuggestion(btn) {{
  document.querySelector('#decisionForm select[name="decision"]').value = 'in';
  syncFieldStates();
  document.querySelector('#decisionForm select[name="pid"]').value = btn.dataset.pidWork;
  document.querySelector('#decisionForm input[name="section_n"]').value = btn.dataset.sectionN;
  document.getElementById('sectionConfirmed').textContent = 'Confirmed: ' + btn.dataset.sectionLabel;
}}
document.querySelector('#decisionForm select[name="decision"]').addEventListener('change', syncFieldStates);
syncFieldStates();
</script></div>"""
        jump_options = "".join(
            f'<option value="{i}"{" selected" if i == index else ""}>'
            f'{i + 1} · {"✓" if r.get("is_in_scope") is not None else "—"} '
            f'{esc(r.get("annotation_id"))}</option>'
            for i, r in enumerate(rows)
        )
        jump = (
            f'<span class="muted">Jump to:</span> '
            f'<select style="width:auto" onchange="location.href=\'/article?'
            f'{urllib.parse.urlencode({"name": name})}&i=\' + this.value">'
            f"{jump_options}</select>"
        ) if rows else ""
        navigation = f"""<div class="row-nav"><a class="button secondary" href="/article?{urllib.parse.urlencode({'name':name,'i':previous_i})}">← Previous</a><span class="grow"></span>{jump}<span class="grow"></span><a class="button secondary" href="/article?{urllib.parse.urlencode({'name':name,'i':next_i})}">Next →</a></div>"""
        complete = f"""<div class="panel"><h2>Finish this document</h2><p>Use this only after every annotation is decided and you have read the complete published article for missed block or single-quoted passages.</p>
<form action="/complete" method="post"><input type="hidden" name="token" value="{self.server.token}"><input type="hidden" name="name" value="{esc(name)}"><input type="hidden" name="i" value="{index}">
<label><input type="checkbox" name="confirmed" value="yes" required> I checked the full article for omissions.</label><p><button type="submit">Validate and mark complete</button></p></form></div>"""
        self.send_html("Review", head + detail + navigation + complete)

    def render_full_article(self, query):
        name = query.get("name", "")
        _path, data, body, _manifest = self.store.load(name)
        edit = query.get("edit")
        edit_i = int(edit) if edit is not None and edit.isdigit() else None
        action = "/boundary" if edit_i is not None else "/manual"
        action_label = "Replace the current annotation with this selection" if edit_i is not None else "Add selected text as a missed quotation"
        hidden_i = f'<input type="hidden" name="i" value="{edit_i}">' if edit_i is not None else ""
        rows_all = annotation_rows(data)
        editing_span = None
        editing_aid = ""
        if edit_i is not None and 0 <= edit_i < len(rows_all):
            editing_row = rows_all[edit_i]
            editing_aid = editing_row.get("annotation_id", "")
            if (isinstance(editing_row.get("start_char"), int)
                    and isinstance(editing_row.get("end_char"), int)
                    and 0 <= editing_row["start_char"] < editing_row["end_char"] <= len(body)):
                editing_span = [editing_row["start_char"], editing_row["end_char"]]
        explanation = (f"You are correcting annotation {edit_i + 1} ({editing_aid}). "
                       "Its current text is shown in amber in the article below. "
                       "Highlight the corrected quotation, click 'Use my highlighted "
                       "text', then submit."
                       if edit_i is not None else
                       "Select a missed quotation in the article below. The selection must "
                       "contain at least five words. Everything already recorded is "
                       "highlighted in green — you are looking for quotation-like text "
                       "with NO highlight, especially indented block quotations and "
                       "quotations in single quotation marks.")
        recorded_spans = sorted(
            (row["start_char"], row["end_char"])
            for row in rows_all
            if isinstance(row.get("start_char"), int)
            and isinstance(row.get("end_char"), int)
            and 0 <= row["start_char"] < row["end_char"] <= len(body)
        )
        form_extra = "" if edit_i is not None else """<label>Quotation form</label><select name="quotation_form"><option value="block">Block quotation</option><option value="single_quote">Single quotation marks</option><option value="other">Other marked form</option></select>"""
        content = f"""<div class="row-nav"><h1>Full article</h1><span class="grow"></span><a class="button secondary" href="/article?{urllib.parse.urlencode({'name':name,'i':edit_i or 0})}">Back to annotations</a></div>
<div class="panel"><p>{esc(explanation)}</p><form id="selectionForm" action="{action}" method="post"><input type="hidden" name="token" value="{self.server.token}"><input type="hidden" name="name" value="{esc(name)}">{hidden_i}
<input id="selectionStart" type="hidden" name="start"><input id="selectionEnd" type="hidden" name="end"><textarea id="selectionText" name="quote" readonly placeholder="Selected article text will appear here"></textarea>{form_extra}
<p><button type="button" onclick="captureSelection()">Use my highlighted text</button> <button class="success" type="submit">{esc(action_label)}</button></p></form></div>
<pre id="articleBody" class="article">{esc(body)}</pre>
<script>
function captureSelection(quiet) {{
  const root=document.getElementById('articleBody'), sel=window.getSelection();
  if(!sel.rangeCount || sel.isCollapsed || !root.contains(sel.anchorNode) || !root.contains(sel.focusNode)) {{ if(!quiet) alert('Highlight text inside the article first.'); return; }}
  const range=sel.getRangeAt(0), before=document.createRange(); before.selectNodeContents(root); before.setEnd(range.startContainer,range.startOffset);
  const start=before.toString().length, text=range.toString();
  document.getElementById('selectionStart').value=start; document.getElementById('selectionEnd').value=start+text.length;
  document.getElementById('selectionText').value=text; if(!quiet) window.scrollTo({{top:0,behavior:'smooth'}});
}}
// Submitting without the capture step is the most common mistake: try to
// capture the live selection automatically, and only block if there is
// genuinely nothing to use.
document.getElementById('selectionForm').addEventListener('submit', function (ev) {{
  if (!document.getElementById('selectionStart').value) {{
    captureSelection(true);
  }}
  if (!document.getElementById('selectionStart').value) {{
    ev.preventDefault();
    alert("First highlight the passage in the article below, then click " +
          "'Use my highlighted text' — your selection will appear in the box.");
  }}
}});
// Highlight already-recorded spans (proposals + manual additions) so the reader
// hunts for quotation-like text WITHOUT a highlight. Uses the CSS Custom
// Highlight API: no DOM changes, so captureSelection offsets are unaffected;
// silently does nothing on browsers without support.
const recordedSpans = {json.dumps(recorded_spans)};
const editingSpan = {json.dumps(editing_span)};
const articleRoot = document.getElementById('articleBody');
const textNodes = [];
{{
  const walker = document.createTreeWalker(articleRoot, NodeFilter.SHOW_TEXT);
  let tn;
  while ((tn = walker.nextNode())) textNodes.push(tn);
}}
function locate(offset) {{
  let acc = 0;
  for (const node of textNodes) {{
    if (offset <= acc + node.data.length) return [node, offset - acc];
    acc += node.data.length;
  }}
  const last = textNodes[textNodes.length - 1];
  return [last, last.data.length];
}}
function rangeFor(span) {{
  const [sn, so] = locate(span[0]), [en, eo] = locate(span[1]);
  const r = new Range(); r.setStart(sn, so); r.setEnd(en, eo);
  return r;
}}
if (window.Highlight && CSS.highlights) {{
  const hl = new Highlight();
  for (const span of recordedSpans) {{
    try {{ hl.add(rangeFor(span)); }} catch (err) {{}}
  }}
  CSS.highlights.set('recorded', hl);
  if (editingSpan) {{
    try {{ CSS.highlights.set('editing', new Highlight(rangeFor(editingSpan))); }} catch (err) {{}}
  }}
}}
// In boundary-correction mode, scroll straight to the annotation being
// corrected (independent of Highlight API support).
if (editingSpan) {{
  try {{
    const rect = rangeFor(editingSpan).getBoundingClientRect();
    window.scrollTo({{top: rect.top + window.scrollY - 160, behavior: 'smooth'}});
  }} catch (err) {{}}
}}
</script>"""
        self.send_html("Full article", content)

    def render_help(self):
        content = """<h1>What am I annotating?</h1><div class="panel"><p>Include marked direct quotations of at least five words from Pynchon's fiction. Exclude titles, scare quotes, other authors and critics, Pynchon's nonfiction, paraphrase, allusion, and summary.</p>
<h2>Important rules</h2><ul><li>Automatically suggested annotations and source suggestions are aids, not answers.</li><li>Block quotations keep their formatting in the plaintext; consult the published PDF whenever anything looks unclear, and add missed block quotations with "Add a missed quotation".</li><li>Record uncertainty rather than guessing.</li><li>Read the complete article before marking it complete.</li></ul>
<p>Your work saves automatically. Use <strong>Finish for now</strong> when you want to stop.</p></div>"""
        self.send_html("Help", content)

    def serve_pdf(self, query):
        _path, data, _body, _manifest = self.store.load(query.get("name", ""))
        pdf = self.store.pdf_path(data)
        if not pdf:
            raise FileNotFoundError("Published PDF is unavailable")
        size = pdf.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'inline; filename="{pdf.name}"')
        self.end_headers()
        with pdf.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                self.wfile.write(chunk)

    # Process a completed annotation decision (in-scope/out-scope) and advance to the next unresolved span.
    def save_decision(self, form):
        path, data, _body, _manifest = self.store.load(form["name"])
        rows = annotation_rows(data)
        index = int(form.get("i", "0"))
        row = rows[index]
        decision = form.get("decision", "unreviewed")
        notes = form.get("notes", "").strip()
        quotation_form = form.get("quotation_form", "").strip()
        alterations = [x.strip().lower() for x in form.get("alterations", "").split(",") if x.strip()]
        if decision == "in" and form.get("pid", "").strip() == "__unlocated__":
            set_positive_unlocated(row, alterations)
        elif decision == "in":
            source = form.get("pid", "").strip()
            pid, separator, work = source.partition("::")
            pid = pid.upper()
            valid_works = {
                ("P01", "Slow Learner"),
                ("P01", "Mortality and Mercy in Vienna"),
                *((candidate, WORK_TITLES[candidate]) for candidate in PIDS if candidate != "P01"),
            }
            if source and (not separator or (pid, work) not in valid_works):
                raise ValueError("Choose a valid source work")
            section_text = form.get("section_n", "").strip()
            section_n = int(section_text) if section_text.isdigit() else None
            set_positive(row, pid, work if pid else "", alterations, section_n)
        elif decision == "out":
            category = form.get("exclusion", "")
            if category not in {key for key, _description in EXCLUSIONS}:
                raise ValueError("Choose an exclusion reason")
            set_negative(row, category)
        else:
            row.update({"is_in_scope": None, "exclusion_category": "", "true_pid": "", "true_work": "", "true_section_n": None, "true_section_label": "", "in_primary_corpus": None, "alterations": []})
        row["quotation_form"] = quotation_form
        row["notes"] = notes
        self.store.commit(path, data)
        next_i = index
        for offset in range(1, len(rows)+1):
            candidate = (index+offset) % len(rows)
            if rows[candidate].get("is_in_scope") is None:
                next_i = candidate
                break
        message = "Decision saved"
        if decision == "in" and row.get("in_primary_corpus") is False:
            if not (row.get("notes") or "").strip():
                message += "; marked not locatable — an explanatory note is required before completion"
        elif decision == "in" and not row.get("true_pid"):
            message += "; source remains unresolved and must be revisited before completion"
        elif decision == "in" and not row.get("true_section_label"):
            message += "; chapter/episode remains unresolved and must be revisited before completion"
        self.redirect("/article?" + urllib.parse.urlencode({"name":form["name"],"i":next_i}), message)

    def _selected_span(self, body, form, near=None):
        start_text, end_text = form.get("start", ""), form.get("end", "")
        if start_text.isdigit() and end_text.isdigit():
            start, end = int(start_text), int(end_text)
            if 0 <= start < end <= len(body):
                return start, end
        quote = form.get("quote", "").strip()
        if not quote:
            raise ValueError(
                "No selection was received. Highlight the passage in the article, "
                "click 'Use my highlighted text' so it appears in the box, then submit.")
        start, end, count = find_quote_offsets(body, quote)
        if start is None:
            raise ValueError("The selected text could not be aligned to the article")
        if count > 1 and near is not None:
            spans = [find_quote_offsets(body, quote, occurrence_index=i)[:2] for i in range(count)]
            start, end = min(spans, key=lambda pair: abs(pair[0]-near))
        elif count > 1:
            raise ValueError("That wording occurs more than once; make the selection slightly longer")
        return start, end

    # Update a span's character offsets after the annotator corrects its boundaries in the full-article view.
    def save_boundary(self, form):
        path, data, body, _manifest = self.store.load(form["name"])
        rows = annotation_rows(data)
        index = int(form.get("i", "0"))
        row = rows[index]
        start, end = self._selected_span(body, form, near=row.get("start_char"))
        if word_count(body[start:end]) < 5:
            raise ValueError("The corrected quotation has fewer than five words")
        # Excluded duplicates may legitimately sit inside a merged span, so only
        # rows still in play block the correction.
        clashes = [(j + 1, r.get("annotation_id")) for j, r in enumerate(rows)
                   if j != index and r.get("is_in_scope") is not False
                   and start < r["end_char"] and r["start_char"] < end]
        if clashes:
            listed = ", ".join(f"annotation {n} ({aid})" for n, aid in clashes)
            raise ValueError(
                f"The corrected boundaries would overlap {listed}, leaving the same "
                "article text recorded twice. If one passage really contains both, "
                "highlight the full passage on the 'Full article' page and use 'Add "
                "selected text as a missed quotation' — it will offer to merge them.")
        row["start_char"], row["end_char"], row["quote"] = start, end, body[start:end]
        update_context(row, body)
        self.store.commit(path, data)
        self.redirect("/article?" + urllib.parse.urlencode({"name":form["name"],"i":index}), "Quotation boundaries updated")

    @staticmethod
    def _span_state(row):
        """One-line human summary of a row's current decision, for merge provenance."""
        if row.get("is_in_scope") is True:
            if row.get("in_primary_corpus") is False:
                return "in scope — not locatable in the primary texts"
            where = row.get("true_work") or "source unresolved"
            if row.get("true_section_label"):
                where += f", {row['true_section_label']}"
            return f"in scope — {where}"
        if row.get("is_in_scope") is False:
            return f"excluded — {row.get('exclusion_category') or 'uncategorized'}"
        return "undecided"

    # Add a missed quotation or merge overlapping spans when manually selected text overlaps existing annotations.
    def add_manual(self, form):
        path, data, body, _manifest = self.store.load(form["name"])
        start, end = self._selected_span(body, form)
        overlaps = [(i, r) for i, r in enumerate(annotation_rows(data))
                    if start < r["end_char"] and r["start_char"] < end]
        if len(overlaps) == 1:
            i, r = overlaps[0]
            raise ValueError(
                f"This selection overlaps annotation {i + 1} ({r.get('annotation_id')}). "
                "Go to that annotation and use "
                "'Correct boundaries using the full article' instead.")
        if len(overlaps) > 1:
            if form.get("confirm_merge") == "yes":
                self.merge_spans(path, data, body, form, start, end, overlaps)
            else:
                self.render_merge_confirmation(form, body, start, end, overlaps)
            return
        form_name = form.get("quotation_form", "block")
        # Match the CLI's manual-span vocabulary (make_manual_span defaults to
        # "manual_block") so the two interfaces never fragment the
        # quotation_form strata reported by eval_compare.
        if not form_name.startswith("manual_"):
            form_name = "manual_" + form_name
        row = make_manual_span(data, body, start, end, form_name)
        self.store.commit(path, data)
        index = annotation_rows(data).index(row)
        self.redirect("/article?" + urllib.parse.urlencode({"name":form["name"],"i":index}), "Missed quotation added; now decide whether it is in scope")

    @staticmethod
    def _merge_target(overlaps):
        """Pick which overlapped row survives a merge: prefer a row that already
        carries an in-scope decision (its source tags stay useful), then an
        undecided row, then an excluded one; earliest in the article on ties."""
        priority = {True: 0, None: 1, False: 2}
        return min(overlaps, key=lambda pair: (
            priority[pair[1].get("is_in_scope")], pair[1]["start_char"], pair[0]))

    def render_merge_confirmation(self, form, body, start, end, overlaps):
        target_i, target = self._merge_target(overlaps)
        merged_start = min([start] + [r["start_char"] for _, r in overlaps])
        merged_end = max([end] + [r["end_char"] for _, r in overlaps])
        items = []
        for i, r in overlaps:
            role = ("boundaries will be extended to the full selection"
                    if i == target_i else
                    "will be excluded as a nested duplicate (its current decision is "
                    "preserved in its notes)")
            excerpt = r.get("quote", "")
            if len(excerpt) > 110:
                excerpt = excerpt[:110] + "…"
            items.append(
                f"<li><strong>Annotation {i + 1} ({esc(r.get('annotation_id'))})</strong>"
                f" — {esc(self._span_state(r))}; {esc(role)}.<br>"
                f"<span class='muted'>“{esc(excerpt)}”</span></li>")
        excerpt = body[merged_start:merged_end]
        if len(excerpt) > 220:
            excerpt = excerpt[:110] + " […] " + excerpt[-110:]
        hidden = "".join(
            f'<input type="hidden" name="{k}" value="{esc(v)}">'
            for k, v in [("token", self.server.token), ("name", form["name"]),
                         ("start", str(start)), ("end", str(end)),
                         ("quotation_form", form.get("quotation_form", "block")),
                         ("confirm_merge", "yes")])
        back = "/full?" + urllib.parse.urlencode({"name": form["name"]})
        content = f"""<div class="panel warning"><h1>Merge overlapping annotations?</h1>
<p>Your selection contains {len(overlaps)} previously recorded annotations. A quotation
must be recorded exactly once, so they should become one span:</p>
<ul>{''.join(items)}</ul>
<p>The merged quotation will read:</p>
<p class="muted">“{esc(excerpt)}”</p>
<p>Afterwards you will be taken to the surviving annotation to re-verify its scope
decision, source, and alterations against the longer text.</p>
<form action="/manual" method="post">{hidden}
<p><button class="success" type="submit">Merge into annotation {target_i + 1}</button>
<a class="button secondary" href="{back}">Cancel</a></p></form></div>"""
        self.send_html("Merge annotations", content)

    # Merge overlapping spans into a single annotation and exclude nested duplicates with explanatory notes.
    def merge_spans(self, path, data, body, form, start, end, overlaps):
        target_i, target = self._merge_target(overlaps)
        # The union, not the bare selection: a partially overlapped span may
        # extend past the selection, and no recorded text should be dropped.
        merged_start = min([start] + [r["start_char"] for _, r in overlaps])
        merged_end = max([end] + [r["end_char"] for _, r in overlaps])
        absorbed = []
        for i, r in overlaps:
            if i == target_i:
                continue
            prior = self._span_state(r)
            set_negative(r, "duplicate_or_boundary_artifact")
            note = (f"Merged into {target.get('annotation_id')}: nested within its "
                    f"block quotation. Prior state: {prior}.")
            existing = (r.get("notes") or "").strip()
            r["notes"] = f"{existing} {note}".strip()
            absorbed.append(f"annotation {i + 1} ({r.get('annotation_id')})")
        target["start_char"], target["end_char"] = merged_start, merged_end
        target["quote"] = body[merged_start:merged_end]
        update_context(target, body)
        self.store.commit(path, data)
        self.redirect(
            "/article?" + urllib.parse.urlencode({"name": form["name"], "i": target_i}),
            f"Merged: boundaries of annotation {target_i + 1} now cover the full "
            f"quotation; {', '.join(absorbed)} excluded as nested duplicates. "
            "Re-verify this annotation's scope decision, source, and alterations.")

    def complete(self, form):
        path, data, body, _manifest = self.store.load(form["name"])
        if form.get("confirmed") != "yes":
            raise ValueError("Confirm that you checked the complete article")
        issues = completion_issues(data, body)
        if issues:
            summary = "; ".join(issues[:5])
            if len(issues) > 5:
                summary += f"; and {len(issues)-5} more"
            raise ValueError("Document is not ready: " + summary)
        data["review_status"] = "complete"
        data["full_document_reviewed"] = True
        data["completed_at"] = now_utc()
        data["last_saved_at"] = data["completed_at"]
        save_annotation(path, data)
        self.redirect("/", "Document marked complete")


def main():
    parser = argparse.ArgumentParser(description="Local browser annotation interface")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--manifest", default=str(SAMPLE_MANIFEST))
    parser.add_argument("--corpus-root", default=str(DEFAULT_CORPUS))
    parser.add_argument("--port", type=int, default=8517,
                        help="local port (default 8517, stable across restarts so "
                             "old tabs keep working; 0 selects a random free port)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the default browser")
    args = parser.parse_args()

    store = AnnotationStore(args.annotations, args.manifest, args.corpus_root, args.reviewer)
    try:
        server = AnnotationServer(("127.0.0.1", args.port), Handler, store)
    except OSError:
        # The fixed port is taken. If it's an annotation session, reuse it
        # instead of starting a second copy (two instances writing the same
        # files can overwrite each other's saves); otherwise fall back to a
        # random free port.
        probe_url = f"http://127.0.0.1:{args.port}/"
        try:
            import urllib.request
            with urllib.request.urlopen(probe_url, timeout=2) as response:
                page = response.read(4096).decode("utf-8", "replace")
        except Exception:
            page = ""
        if "CHR annotation" in page or "Assigned documents" in page:
            print(f"An annotation session is already running at {probe_url}")
            print("Opening it instead of starting a second copy.")
            if not args.no_browser:
                webbrowser.open(probe_url)
            return
        server = AnnotationServer(("127.0.0.1", 0), Handler, store)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print("CHR annotation is running locally.")
    print(f"Open {url} if the browser does not appear automatically.")
    print("Keep this window open while annotating. Use 'Finish for now' in the browser when done.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Annotation interface stopped; saved work is retained.")


if __name__ == "__main__":
    main()
