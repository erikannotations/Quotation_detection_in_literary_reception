#!/usr/bin/env python3
"""Generic Chrome (CDP) fetcher past Cloudflare. Usage:
    python3 chrome_fetch.py <url> <out_basename> [content_marker]
Connects to Chrome on --remote-debugging-port=9233; writes <out>.txt and <out>.html.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path
from project_paths import AUXILIARY_DIR, ROOT

sys.path.insert(0, str(ROOT.parent / "corpus_articles" / ".vendor"))
import websocket

PORT = 9233
BLOCK = ("Just a moment", "Sorry, you have been blocked", "Checking your browser",
         "cf-browser-verification", "Enable JavaScript and cookies")


class Chrome:
    # Chrome DevTools Protocol client for remote browser control.
    def __init__(self, port=PORT):
        pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=5))
        page = next(p for p in pages if p.get("type") == "page")
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=25)
        self.i = 0

    def call(self, method, params=None):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": params or {}}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.i:
                return m

    def eval(self, expr):
        r = self.call("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return r.get("result", {}).get("result", {}).get("value")


def fetch(url, out, marker=""):
    # Navigate to URL, wait for load, detect Cloudflare blocks, extract text and HTML.
    c = Chrome()
    c.call("Page.navigate", {"url": url})
    deadline = time.time() + 90
    text = html = ""
    while time.time() < deadline:
        time.sleep(2)
        if c.eval("document.readyState") != "complete":
            continue
        text = c.eval("document.body ? document.body.innerText : ''") or ""
        blocked = any(b in text for b in BLOCK) or len(text) < 1200
        good = (marker in text) if marker else (not blocked)
        print(f"  len={len(text)} blocked={blocked} marker={'y' if (marker and marker in text) else '-'}")
        if good and not blocked:
            html = c.eval("(document.querySelector('#mw-content-text') || document.body).innerHTML") or ""
            break
    destination = Path(out)
    if not destination.is_absolute():
        destination = AUXILIARY_DIR / "downloads" / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".txt").write_text(text, encoding="utf-8")
    destination.with_suffix(".html").write_text(html, encoding="utf-8")
    print(f"wrote {destination}.txt ({len(text)}) / {destination}.html ({len(html)})")
    return text


if __name__ == "__main__":
    url, out = sys.argv[1], sys.argv[2]
    marker = sys.argv[3] if len(sys.argv) > 3 else ""
    t = fetch(url, out, marker)
    print("---- first 1500 chars ----")
    print(t[:1500])
