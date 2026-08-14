#!/usr/bin/env python3
"""Quid 2.6.7 lexical text-reuse baseline across all primary works.

Install the pinned optional dependency from `requirements-baselines.txt`. Quid compares
each primary work (source) with each scholarly article (target). Overlapping target
matches are merged into a single candidate while retaining all proposed source pids.
Novel-length sources are processed as overlapping chunks, with each source index
cached across the whole article batch. This bounds memory without changing Quid's
default MinHash or fuzzy-token settings.
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import platform
import resource
import time
from pathlib import Path

from locate_quotes import PIDS, PRIMARY_FILES
from project_paths import CANDIDATES_DIR
from quotation_spans import article_body


def span_value(span, name, default=None):
    if isinstance(span, dict):
        return span.get(name, default)
    return getattr(span, name, default)


def merge_matches(rows):
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


def source_chunks(text, max_chars, overlap=1000):
    """Yield whitespace-aligned, overlapping source chunks."""
    if len(text) <= max_chars:
        yield text
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            next_space = text.rfind(" ", start, end)
            if next_space > start:
                end = next_space
        yield text[start:end]
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def compare_chunk(task):
    """Index one bounded source chunk in a disposable worker process."""
    chunk, bodies, settings = task
    from quid.core.Quid import Quid
    from quid.core.QuidMatcher import QuidMatcher

    matcher = Quid(
        min_match_length=settings["min_match_length"],
        keep_ambiguous_matches=True,
        min_levenshtein_similarity=settings["token_similarity"],
        min_levenshtein_similarity_short=settings["token_similarity"],
    )
    matcher.HASH_PERM = settings["hash_permutations"]
    QuidMatcher.HASH_PERM = settings["hash_permutations"]
    positions, hashes = matcher.prepare_source_data(chunk)
    output = [[] for _body in bodies]
    for article_i, body in enumerate(bodies):
        for match in matcher.compare(chunk, body, positions, hashes):
            target = span_value(match, "target_span")
            start = int(span_value(target, "start", 0))
            end = int(span_value(target, "end", 0))
            if end > start:
                output[article_i].append((start, end))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("articles", nargs="+")
    parser.add_argument("--min-match-length", type=int, default=5)
    parser.add_argument("--hash-permutations", type=int, default=128)
    parser.add_argument(
        "--token-similarity", type=float, default=0.85,
        help="Quid token/initial-ngram similarity threshold (default: 0.85)",
    )
    parser.add_argument("--source-chunk-chars", type=int, default=100000)
    parser.add_argument("--out-dir", default=str(CANDIDATES_DIR))
    parser.add_argument(
        "--pids", nargs="+", choices=PIDS, default=PIDS,
        help="primary works to compare (default: all; useful for smoke tests)",
    )
    args = parser.parse_args()
    if importlib.util.find_spec("quid") is None:
        raise SystemExit(
            "Quid is not installed. Run: python3 -m pip install -r requirements-baselines.txt"
        )
    if args.hash_permutations < 8:
        raise SystemExit("--hash-permutations must be at least 8")
    similarity_tag = str(args.token_similarity).replace(".", "p")
    tag = (
        f"quid_2.6.7_hp{args.hash_permutations}_s{similarity_tag}"
        f"_m{args.min_match_length}_c{args.source_chunk_chars}"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    articles = []
    for filename in args.articles:
        path = Path(filename)
        body = article_body(path.read_text(encoding="utf-8"))
        articles.append({"path": path, "body": body, "matches": []})

    # Source-outer order lets all target articles reuse the expensive MinHash
    # index. Each chunk runs in a fresh child process because Quid/datasketch do
    # not reliably return their large MinHash allocations to long-lived processes.
    bodies = [article["body"] for article in articles]
    settings = {
        "min_match_length": args.min_match_length,
        "hash_permutations": args.hash_permutations,
        "token_similarity": args.token_similarity,
    }
    # Plan §4.2 requires reporting Quid's wall time and peak memory alongside its
    # chunking and MinHash settings. Nothing recorded them, so the frozen run would
    # have had to reconstruct them by hand afterwards. Captured here instead.
    run_started = time.monotonic()
    per_pid_seconds = {}
    for pid in args.pids:
        pid_started = time.monotonic()
        source = PRIMARY_FILES[pid].read_text(encoding="utf-8")
        chunks = list(source_chunks(source, args.source_chunk_chars))
        print(f"indexing {pid}: {len(chunks)} source chunks", flush=True)
        context = mp.get_context("spawn")
        tasks = ((chunk, bodies, settings) for chunk in chunks)
        with context.Pool(processes=1, maxtasksperchild=1) as pool:
            for chunk_result in pool.imap(compare_chunk, tasks):
                for article_i, matches in enumerate(chunk_result):
                    articles[article_i]["matches"].extend(
                        {"start": start, "end": end, "pid": pid}
                        for start, end in matches
                    )
        per_pid_seconds[pid] = round(time.monotonic() - pid_started, 2)

    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    scale = 1 << 20 if platform.system() == "Darwin" else 1 << 10
    runinfo = {
        "tool": "Quid",
        "settings": settings,
        "source_chunk_chars": args.source_chunk_chars,
        "pids": list(args.pids),
        "wall_seconds_total": round(time.monotonic() - run_started, 2),
        "wall_seconds_by_pid": per_pid_seconds,
        "peak_rss_mb_self": round(usage.ru_maxrss / scale, 1),
        "peak_rss_mb_largest_child": round(children.ru_maxrss / scale, 1),
        "platform": platform.platform(),
        "articles": [article["path"].stem for article in articles],
    }

    for article in articles:
        path, body = article["path"], article["body"]
        raw_matches = article["matches"]
        rows = []
        for i, item in enumerate(merge_matches(raw_matches), 1):
            pids = sorted(item["candidate_pids"])
            rows.append({
                "id": f"{path.stem}-quid-{i:04d}",
                "article": path.stem,
                "model_tag": tag,
                "quote": body[item["start"]:item["end"]],
                "detector_guess": "|".join(pids),
                "attributed_work": "",
                "attributed_page": "",
                "confidence": "baseline",
                "article_start_char": item["start"],
                "article_end_char": item["end"],
                "quid_candidate_pids": pids,
            })
        out = out_dir / f"{path.stem}__{tag}.candidates.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name}: {len(rows)} Quid matches -> {out}")
        (out_dir / f"{path.stem}__{tag}.runinfo.json").write_text(
            json.dumps(runinfo, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
