#!/usr/bin/env python3
"""Reeve's `text-matcher` lexical text-reuse baseline across all primary works.

Install the pinned optional dependency from `requirements-textmatcher.txt` into
`.venv-textmatcher`. This is the Anglophone counterpart to `baseline_quid.py`:
text-matcher was built for exactly this task (quotations of a literary work inside
English-language scholarly prose), so it is the closest prior system to ours and the
one a CHR referee is most likely to ask about. Added after the 26 July 2026 freeze;
see `paper/audit_implementation_plan_2026-07-18.md` and NEXT_STEPS §I.

Like Quid, text-matcher compares each primary work (source, text A) with each
scholarly article (target, text B) and reports character spans in both. We keep the
spans in the article, merge overlapping ones, and record which primary works proposed
each span, so the output is interchangeable with every other system's candidates file.

Two configurations are provided, because "text-matcher's settings" is ambiguous in the
sources and the difference is the whole point of Reeve, Eckert and Terlunen's
corpus-adaptation argument:

  reeve_defaults  threshold 3, cutoff 5, ngrams 3, minDistance 8, stopwords removed.
                  The library defaults, which is what the DSH methods paper's
                  production run used (`notebooks/jstor-match.ipynb`, "Experiment 2A",
                  calls `Matcher(mm, articleText)` with no arguments). The paper
                  reports precision 100% / recall ~30% at this operating point and
                  argues the defaults were themselves arrived at "after a long period
                  of experimentation" on scholarly prose — i.e. here, running at
                  library defaults IS the corpus-adapted configuration.

  reeve_eval      threshold 2, cutoff 3, ngrams 2, minDistance 3, stopwords removed.
                  The looser operating point the project's own evaluation notebook
                  scores (`hyperparameter_test_name = "t2-c3-n2-m3-no-stops"` in
                  `notebooks/evaluation-metrics-notebook.ipynb`) and the setting behind
                  the released `middlemarch_fontsizeviz-t2-c3-n2-m3-no-stops.html`.

Running only the strict configuration would reproduce the criticism Reeve et al. level
at everyone else — judging a tool at an operating point its authors did not choose for
evaluation — so both are run and both are reported.

Two properties of the method are preserved deliberately rather than patched:

  * `difflib.SequenceMatcher` returns monotonically ordered matching blocks, so an
    article that quotes chapter 10 before chapter 2 can only keep one of them. This
    order-dependence is the limitation Arnold and Jäschke's benchmark identifies, and
    it is a property of the published method, not a defect of this wrapper.
  * Stopwords are dropped before matching, so a reported span runs from the first to
    the last *content* word. Leading and trailing stopwords of a quotation therefore
    fall outside the span. Interior stopwords are retained, because the span is a
    character range in the article, not a bag of tokens.
"""
import argparse
import bisect
import importlib.metadata
import json
import platform
import re
import resource
import sys
import time
from pathlib import Path

from locate_quotes import PIDS, PRIMARY_FILES
from project_paths import CANDIDATES_DIR
from quotation_spans import article_body

# text-matcher's extend_matches() recurses once per round of single-word boundary
# growth, so a long quotation can nest several hundred frames deep. The default 1000
# is not generous enough for novel-length sources; a pair that still overflows is
# recorded as an error for that pair rather than aborting the run.
RECURSION_LIMIT = 20000

# text-matcher's Text.preprocess() rejoins words broken across a line ("ac- commodate"
# -> "accommodate") and then tokenises the REWRITTEN string, so every span it reports
# is an offset into the healed text, not into the article as we hold it. Each heal
# removes exactly two characters ("- "), so a match after the first heal in an article
# is reported two characters early, after the second heal four, and so on. Slicing the
# article with those offsets silently returns quotations shifted off their word
# boundaries. The same regex is replayed here to map the offsets back.
HYPHEN_HEAL = re.compile(r"([A-Za-z])- ([a-z])")
PRESETS = {
    "reeve_defaults": {
        "threshold": 3, "cutoff": 5, "ngrams": 3, "min_distance": 8,
        "remove_stopwords": True,
    },
    "reeve_eval": {
        "threshold": 2, "cutoff": 3, "ngrams": 2, "min_distance": 3,
        "remove_stopwords": True,
    },
}


def heal_shifts(original):
    """Positions, in healed coordinates, at or after which two characters were cut.

    `re.sub` and `finditer` both scan left to right without overlap, so replaying the
    pattern reproduces the substitutions in the same order the tool made them.
    """
    shifts, removed = [], 0
    for match in HYPHEN_HEAL.finditer(original):
        # The hyphen and space sit between the two kept letters, so the second letter
        # is where healed coordinates start running short.
        shifts.append(match.start() + 1 - removed)
        removed += 2
    return shifts


def to_article_offset(position, shifts):
    return position + 2 * bisect.bisect_right(shifts, position)


def merge_matches(rows):
    """Merge overlapping article spans, retaining every proposing primary work.

    Identical to `baseline_quid.py`'s merge so the two lexical baselines are scored
    on the same span granularity.
    """
    merged = []
    for row in sorted(rows, key=lambda r: (r["start"], r["end"], r["pid"])):
        hit = next((old for old in merged
                    if row["start"] < old["end"] and old["start"] < row["end"]), None)
        if hit is None:
            item = dict(row)
            item["candidate_pids"] = [row["pid"]]
            merged.append(item)
        else:
            hit["start"] = min(hit["start"], row["start"])
            hit["end"] = max(hit["end"], row["end"])
            if row["pid"] not in hit["candidate_pids"]:
                hit["candidate_pids"].append(row["pid"])
    return merged


def target_spans(matcher, text_obj, ngram_size, shifts):
    """Character spans of the extended matches, as offsets into the original article.

    `Matcher.match()` is not used: it writes every match to stdout with ANSI colour and
    rebuilds a context string per match, which is expensive over 10 works x 16 articles
    and useless here. This reproduces exactly what `match()` records in `locationsB`,
    minus the printing, and then undoes the hyphen-healing offset shift.

    A match whose boundary walked below index 0 (extend_matches() decrements without a
    floor, so `spans[-1:]` would silently return a span from the *end* of the article)
    is dropped rather than turned into a spurious quotation; the count is reported.
    """
    spans, dropped = [], 0
    length_offset = ngram_size - 1
    for match in matcher.extended_matches:
        start = match.b
        if start < 0:
            dropped += 1
            continue
        located = matcher.getLocations(text_obj, start, match.sizeB + length_offset)
        if located is None:
            dropped += 1
            continue
        first, last = int(located[0]), int(located[1])
        if last > first:
            spans.append((to_article_offset(first, shifts),
                          to_article_offset(last, shifts)))
        else:
            dropped += 1
    return spans, dropped


def build_tag(version, settings):
    """Model tag for the candidates filename.

    The settings come before anything else variable because `eval_compare.py` prints
    `tag[:25]`: with the version string first, the two presets would truncate to the
    same 25 characters and appear as two indistinguishable rows in the results table.
    The pinned commit is not in the tag — it is in the `.runinfo.json` sidecar written
    beside every candidates file, as Quid's full settings are.
    """
    stops = "nostops" if settings["remove_stopwords"] else "stops"
    return (
        f"textmatcher_t{settings['threshold']}c{settings['cutoff']}"
        f"n{settings['ngrams']}m{settings['min_distance']}_{stops}_{version}"
    )


def installed_commit():
    """Recover the pinned git commit recorded by pip at install time."""
    try:
        direct = importlib.metadata.distribution("text-matcher").read_text(
            "direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
    if not direct:
        return "unknown"
    info = json.loads(direct)
    return info.get("vcs_info", {}).get("commit_id", "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("articles", nargs="+")
    parser.add_argument(
        "--preset", choices=sorted(PRESETS), default="reeve_defaults",
        help="named configuration from the text-matcher literature (default: "
             "reeve_defaults, the library defaults used by the DSH methods paper)",
    )
    parser.add_argument("--threshold", type=int, help="override preset threshold")
    parser.add_argument("--cutoff", type=int, help="override preset cutoff")
    parser.add_argument("--ngrams", type=int, help="override preset ngram size")
    parser.add_argument("--min-distance", type=int, help="override preset healing distance")
    parser.add_argument(
        "--keep-stopwords", action="store_true",
        help="match with stopwords retained (text-matcher's removeStopwords=False)",
    )
    parser.add_argument("--out-dir", default=str(CANDIDATES_DIR))
    parser.add_argument(
        "--pids", nargs="+", choices=PIDS, default=PIDS,
        help="primary works to compare (default: all; useful for smoke tests)",
    )
    args = parser.parse_args()

    try:
        from text_matcher.matcher import Matcher, Text
    except ImportError:
        raise SystemExit(
            "text-matcher is not installed. Run:\n"
            "  python3 -m venv .venv-textmatcher\n"
            "  .venv-textmatcher/bin/python -m pip install -r requirements-textmatcher.txt\n"
            "  .venv-textmatcher/bin/python -c \"import nltk; nltk.download('stopwords')\""
        )
    try:
        import nltk
        nltk.corpus.stopwords.words("english")
    except LookupError:
        raise SystemExit(
            "NLTK's stopword corpus is missing. Run:\n"
            "  .venv-textmatcher/bin/python -c \"import nltk; nltk.download('stopwords')\""
        )

    settings = dict(PRESETS[args.preset])
    for name in ("threshold", "cutoff", "ngrams", "min_distance"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    if args.keep_stopwords:
        settings["remove_stopwords"] = False

    version = importlib.metadata.version("text-matcher")
    commit = installed_commit()
    tag = build_tag(version, settings)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.setrecursionlimit(RECURSION_LIMIT)

    articles = []
    for filename in args.articles:
        path = Path(filename)
        body = article_body(path.read_text(encoding="utf-8"))
        articles.append({
            "path": path,
            "body": body,
            "text": Text(body, path.stem, removeStopwords=settings["remove_stopwords"]),
            "shifts": heal_shifts(body),
            "matches": [],
        })

    # Source-outer order, as in baseline_quid.py: the primary works are the expensive
    # objects (tokenising and Lancaster-stemming a novel), the articles are cheap, so
    # each work is built once, compared with every article, and released before the
    # next. Only one novel is resident at a time.
    run_started = time.monotonic()
    per_pid_seconds, per_pid_matches = {}, {}
    dropped_total = 0
    pair_errors = []
    for pid in args.pids:
        pid_started = time.monotonic()
        source = Text(
            PRIMARY_FILES[pid].read_text(encoding="utf-8"), pid,
            removeStopwords=settings["remove_stopwords"],
        )
        # Text.__init__ always precomputes a trigram list that Matcher never reads
        # (it builds its own at the requested n). On ten novel-length sources that is
        # tens of megabytes of dead weight; dropping it changes no result.
        source.trigrams = None
        found = 0
        for article in articles:
            try:
                matcher = Matcher(
                    source, article["text"],
                    threshold=settings["threshold"],
                    cutoff=settings["cutoff"],
                    ngramSize=settings["ngrams"],
                    removeStopwords=settings["remove_stopwords"],
                    minDistance=settings["min_distance"],
                    silent=True,
                )
            except (RecursionError, IndexError, ValueError) as exc:
                # Reeve's own CLI skips a failing pair; recorded here instead of
                # swallowed, because a silent skip would be indistinguishable from a
                # work that genuinely goes unquoted.
                pair_errors.append({
                    "pid": pid, "article": article["path"].stem,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            spans, dropped = target_spans(
                matcher, article["text"], settings["ngrams"], article["shifts"])
            dropped_total += dropped
            found += len(spans)
            article["matches"].extend(
                {"start": start, "end": end, "pid": pid} for start, end in spans
            )
        per_pid_seconds[pid] = round(time.monotonic() - pid_started, 2)
        per_pid_matches[pid] = found
        print(f"{pid}: {found} raw matches across {len(articles)} articles "
              f"({per_pid_seconds[pid]}s)", flush=True)
        del source

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    scale = 1 << 20 if platform.system() == "Darwin" else 1 << 10
    runinfo = {
        "tool": "text-matcher",
        "version": version,
        "git_commit": commit,
        "preset": args.preset,
        "settings": settings,
        "pids": list(args.pids),
        "wall_seconds_total": round(time.monotonic() - run_started, 2),
        "wall_seconds_by_pid": per_pid_seconds,
        "raw_matches_by_pid": per_pid_matches,
        "dropped_unlocatable_matches": dropped_total,
        "pair_errors": pair_errors,
        "peak_rss_mb_self": round(usage.ru_maxrss / scale, 1),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nltk": importlib.metadata.version("nltk"),
        "articles": [article["path"].stem for article in articles],
        "run_status": "post-freeze addition (NEXT_STEPS §I); not part of the "
                      "26 July 2026 frozen system matrix",
    }

    for article in articles:
        path, body = article["path"], article["body"]
        rows = []
        for i, item in enumerate(merge_matches(article["matches"]), 1):
            pids = sorted(item["candidate_pids"])
            rows.append({
                "id": f"{path.stem}-tm-{i:04d}",
                "article": path.stem,
                "model_tag": tag,
                "quote": body[item["start"]:item["end"]],
                "detector_guess": "|".join(pids),
                "attributed_work": "",
                "attributed_page": "",
                "confidence": "baseline",
                "article_start_char": item["start"],
                "article_end_char": item["end"],
                "textmatcher_candidate_pids": pids,
            })
        out = out_dir / f"{path.stem}__{tag}.candidates.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name}: {len(rows)} text-matcher matches -> {out}")
        (out_dir / f"{path.stem}__{tag}.runinfo.json").write_text(
            json.dumps(runinfo, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
