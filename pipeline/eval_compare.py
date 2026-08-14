#!/usr/bin/env python3
"""Evaluate quotation detection and source linking against an exhaustive census.

The evaluator performs a one-to-one bipartite alignment between predicted spans and
gold spans. This prevents duplicate model outputs from being counted repeatedly and
separates three questions:

1. detection: did the system extract the marked direct quotation?
2. end-to-end linking: was it confirmed and linked to the correct primary work?
3. source ranking: was the gold work top-ranked, or at least retained among tied/
   near-tied source candidates?

Preferred gold format (`evaluation/gold/<article>.gold.json`):

{
  "schema_version": "2.0",
  "article": "<CORPUS text stem>",
  "quotes": [
    {
      "gold_id": "<stable id>",
      "quote": "exact span as printed in the scholarly text",
      "true_pid": "P04",
      "true_work": "Gravity's Rainbow",
      "true_section_n": 9,
      "true_section_label": "Episode 9",
      "in_primary_corpus": true,
      "quotation_form": "quotation_marks|block",
      "alterations": ["ellipsis"],
      "notes": ""
    }
  ]
}

Legacy `evaluation/legacy/gold_census/*.census.json` files remain readable.
"""
import argparse
import difflib
import json
import random
import zlib
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from locate_quotes import normalize_query
from quotation_spans import MIN_QUOTE_WORDS, under_minimum
from project_paths import CANDIDATES_DIR, GOLD_DIR, LEGACY_CENSUS_DIR

BASE = Path(__file__).resolve().parent
DEFAULT_MATCH = 0.60


def sim(a, b):
    """Token-level span similarity; partial containment is not automatically perfect."""
    aa, bb = normalize_query(a).split(), normalize_query(b).split()
    if not aa or not bb:
        return 0.0
    # autojunk=False for the same reason as in locate_quotes.py: past 200 elements
    # difflib treats frequent items as junk. Ordinary prose survives it, but a long
    # repetitive passage does not — 280 tokens of Pynchon-style repeated dialogue
    # score 0.357 with the default against 0.650 without it. Below 0.60 an aligned
    # span becomes a false positive AND a false negative for every system at once.
    # Development gold peaks at 135 tokens so current numbers are unchanged; the
    # test articles are up to 2.3x longer.
    return difflib.SequenceMatcher(None, aa, bb, autojunk=False).ratio()


def norm_work(s):
    return "".join(c for c in (s or "").lower() if c.isalnum() or c == " ").strip()


def work_match(pred, gold):
    p, g = norm_work(pred), norm_work(gold)
    return bool(p) and bool(g) and (g in p or p in g)


def source_match(result, gold, include_alternatives=False):
    true_pid = gold.get("true_pid", "")
    if true_pid:
        if include_alternatives:
            pids = set(filter(None, (result.get("candidate_pids") or "").split("|")))
            pids.add(result.get("pid", ""))
            return true_pid in pids
        return result.get("pid", "") == true_pid
    return work_match(result.get("novel", ""), gold.get("true_work", ""))


def section_match(result, gold):
    """Exact-locus match: correct work AND correct chapter/episode index.

    A section number is only meaningful together with its work (Chapter 5 of one
    novel is not "close to" Chapter 5 of another), so this does not credit a
    correct section under a wrong work, and is undefined (None) rather than False
    when gold carries no ``true_section_n`` -- e.g. legacy pre-migration gold
    files -- so callers can exclude those rows from the denominator instead of
    silently scoring them wrong.
    """
    true_section_n = gold.get("true_section_n")
    if true_section_n is None:
        return None
    if not source_match(result, gold):
        return False
    return result.get("section_n") == true_section_n


def maximum_cardinality_alignment(predictions, gold, threshold):
    """Return one-to-one (prediction index, gold index, score) alignments.

    Kuhn's augmenting-path algorithm guarantees maximum cardinality. Edges and
    predictions are traversed by descending similarity so equally large matchings
    normally retain the strongest span correspondences.
    """
    edges = {}
    for pi, pred in enumerate(predictions):
        candidates = []
        for gi, item in enumerate(gold):
            score = sim(pred.get("quote", ""), item.get("quote", ""))
            if score >= threshold:
                candidates.append((gi, score))
        edges[pi] = sorted(candidates, key=lambda row: (-row[1], row[0]))

    order = sorted(
        range(len(predictions)),
        key=lambda pi: (-(edges[pi][0][1] if edges[pi] else -1), pi),
    )
    gold_to_pred = {}

    def augment(pi, seen):
        for gi, _score in edges[pi]:
            if gi in seen:
                continue
            seen.add(gi)
            if gi not in gold_to_pred or augment(gold_to_pred[gi], seen):
                gold_to_pred[gi] = pi
                return True
        return False

    for pi in order:
        augment(pi, set())

    return sorted(
        ((pi, gi, sim(predictions[pi]["quote"], gold[gi]["quote"]))
         for gi, pi in gold_to_pred.items()),
        key=lambda row: row[0],
    )


def load_gold(stem, allow_legacy=True):
    """Load v2 gold for an article stem, optionally falling back to legacy census.

    The legacy census files use a different schema (no `true_section_n`, no
    `quotation_form`), so a silent fallback mixes two gold generations inside one
    metrics table. The fallback now announces itself, and the frozen run disables
    it entirely with --no-legacy-gold.
    """
    preferred = GOLD_DIR / f"{stem}.gold.json"
    legacy = LEGACY_CENSUS_DIR / f"{stem}.census.json"
    if preferred.exists():
        path = preferred
    elif allow_legacy and legacy.exists():
        path = legacy
        print(f"WARNING: {stem} has no v2 gold; falling back to the LEGACY census "
              f"schema. Metrics for this article are not comparable to v2 rows.")
    else:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(data.get("quotes", []), 1):
        row = dict(item)
        row.setdefault("gold_id", f"{stem}-g{i:03d}")
        if "in_primary_corpus" not in row:
            row["in_primary_corpus"] = row.get("in_corpus", True)
        rows.append(row)
    return rows


ALLOW_LEGACY_GOLD = True


OUT_OF_SCOPE_BY_TAG = defaultdict(int)
#: Whether predictions below the frozen minimum are excluded from scoring. Default on.
#: Turn it off (``--score-out-of-scope``) to report the sensitivity: the choice moves
#: several systems and moves the open-weight detector most, so the paper should be able
#: to show both readings rather than assert one.
EXCLUDE_OUT_OF_SCOPE = True


def discover():
    """Return model tag -> article bundles with gold, candidates, and results.

    Predictions below the frozen minimum are removed before any scoring. Gold cannot
    contain such a span (`compile_gold_v2.py` excludes it), so a short prediction can
    only ever be a guaranteed false positive — it is scored not on whether the system
    found a quotation but on whether it answered a question the task did not ask.
    Detection precision should measure detection.

    This is NOT a way of hiding the behaviour, which is real and differs sharply by
    system: the count removed per system is tallied here and reported alongside the
    metrics, so a detector that emits out-of-scope spans is still visibly doing so.
    """
    OUT_OF_SCOPE_BY_TAG.clear()
    by_tag = defaultdict(list)
    for result_path in sorted(CANDIDATES_DIR.glob("*__*.candidates.results.json")):
        name = result_path.name[:-len(".candidates.results.json")]
        stem, separator, tag = name.partition("__")
        if not separator:
            continue
        candidate_path = CANDIDATES_DIR / f"{name}.candidates.json"
        gold = load_gold(stem, allow_legacy=ALLOW_LEGACY_GOLD)
        if gold is None or not candidate_path.exists():
            continue
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        kept = [row for row in candidates if not under_minimum(row.get("quote", ""))]
        OUT_OF_SCOPE_BY_TAG[tag] += len(candidates) - len(kept)
        if EXCLUDE_OUT_OF_SCOPE:
            candidates = kept
        results = {
            row["id"]: row
            for row in json.loads(result_path.read_text(encoding="utf-8"))
        }
        by_tag[tag].append((stem, gold, candidates, results))
    return by_tag


def div(num, den):
    return num / den if den else float("nan")


def f1(precision, recall):
    if precision != precision or recall != recall or precision + recall == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


def score_article(bundle, threshold):
    stem, gold, candidates, results = bundle
    pairs = maximum_cardinality_alignment(candidates, gold, threshold)
    pred_to_gold = {pi: gi for pi, gi, _score in pairs}

    det_tp = len(pairs)
    det_fp = len(candidates) - det_tp
    det_fn = len(gold) - det_tp
    locatable = sum(bool(row.get("in_primary_corpus", True)) for row in gold)
    end_tp = end_fp = top_denom = top_ok = alt_ok = 0
    section_denom = section_ok = 0

    for pi, pred in enumerate(candidates):
        result = results.get(pred.get("id", ""), {})
        confirmed = bool(result.get("confirmed_pynchon_fiction"))
        gi = pred_to_gold.get(pi)
        if gi is None:
            if confirmed:
                end_fp += 1
            continue
        item = gold[gi]
        if not confirmed:
            continue
        if not item.get("in_primary_corpus", True):
            end_fp += 1
            continue
        top_denom += 1
        top_correct = source_match(result, item)
        alternative_correct = source_match(result, item, include_alternatives=True)
        top_ok += int(top_correct)
        alt_ok += int(alternative_correct)
        if top_correct:
            end_tp += 1
        else:
            end_fp += 1
        # Locus (chapter/episode) accuracy is conditioned on top-source being
        # correct -- a section number under the wrong work is not comparable --
        # and only counted when gold carries a confirmed section (section_match
        # returns None for legacy gold rows that predate this field).
        locus = section_match(result, item)
        if locus is not None:
            section_denom += 1
            section_ok += int(locus)

    end_fn = locatable - end_tp

    # Detection recall strata over gold: quotation form (descriptive reporting).
    matched_gold = set(pred_to_gold.values())
    strata = {}
    for gi, item in enumerate(gold):
        form = item.get("quotation_form") or "unspecified"
        # Provenance: spans recovered by the adjudication-stage pooling audit
        # (notes token `pooling_recovered`, predeclared 23 July 2026) are
        # stratified apart so recall is reportable with and without them —
        # `provenance:annotator` alone is annotator-only recall.
        provenance = ("system_recovered"
                      if "pooling_recovered" in (item.get("notes") or "")
                      else "annotator")
        for key in (f"form:{form}", f"provenance:{provenance}"):
            bucket = strata.setdefault(key, {"tp": 0, "fn": 0})
            bucket["tp" if gi in matched_gold else "fn"] += 1

    det_p, det_r = div(det_tp, det_tp + det_fp), div(det_tp, det_tp + det_fn)
    end_p, end_r = div(end_tp, end_tp + end_fp), div(end_tp, end_tp + end_fn)
    return {
        "strata": strata,
        "stem": stem,
        "articles": 1,
        "candidates": len(candidates),
        "gold": len(gold),
        "det_tp": det_tp, "det_fp": det_fp, "det_fn": det_fn,
        "end_tp": end_tp, "end_fp": end_fp, "end_fn": end_fn,
        "top_ok": top_ok, "alt_ok": alt_ok, "top_denom": top_denom,
        "section_ok": section_ok, "section_denom": section_denom,
        # A system that returns NOTHING for an article with gold has failed that
        # article; it has not declined to be measured. Precision is 0/0 there, so
        # f1() yields NaN and the macro average silently DROPS the article — which
        # inflates the macro of exactly the systems that crashed or timed out. Score
        # it 0.0 instead. NaN is reserved for the genuinely undefined case: no gold
        # and no candidates.
        "det_f1": 0.0 if (gold and det_tp == 0) else f1(det_p, det_r),
    }


# Roll up per-article metrics into corpus-level aggregates and compute precision/recall rates.
def aggregate_articles(articles):
    summed = {}
    for key in ("articles", "candidates", "gold", "det_tp", "det_fp", "det_fn",
                "end_tp", "end_fp", "end_fn", "top_ok", "alt_ok", "top_denom",
                "section_ok", "section_denom"):
        summed[key] = sum(row[key] for row in articles)

    det_p = div(summed["det_tp"], summed["det_tp"] + summed["det_fp"])
    det_r = div(summed["det_tp"], summed["det_tp"] + summed["det_fn"])
    end_p = div(summed["end_tp"], summed["end_tp"] + summed["end_fp"])
    end_r = div(summed["end_tp"], summed["end_tp"] + summed["end_fn"])
    strata = {}
    for row in articles:
        for key, bucket in row.get("strata", {}).items():
            merged = strata.setdefault(key, {"tp": 0, "fn": 0})
            merged["tp"] += bucket["tp"]
            merged["fn"] += bucket["fn"]
    recall_by_stratum = {
        key: {"tp": bucket["tp"], "fn": bucket["fn"],
              "recall": div(bucket["tp"], bucket["tp"] + bucket["fn"])}
        for key, bucket in sorted(strata.items())
    }

    valid_macro = [row["det_f1"] for row in articles if row["det_f1"] == row["det_f1"]]
    summed.update({
        "recall_by_stratum": recall_by_stratum,
        "det_p": det_p, "det_r": det_r, "det_f1": f1(det_p, det_r),
        "end_p": end_p, "end_r": end_r, "end_f1": f1(end_p, end_r),
        "top_acc": div(summed["top_ok"], summed["top_denom"]),
        "alt_acc": div(summed["alt_ok"], summed["top_denom"]),
        "section_acc": div(summed["section_ok"], summed["section_denom"]),
        "macro_det_f1": div(sum(valid_macro), len(valid_macro)),
        "article_metrics": articles,
    })
    return summed


def score_tag(bundles, threshold):
    return aggregate_articles([score_article(bundle, threshold) for bundle in bundles])


def percentile(values, probability):
    values = sorted(value for value in values if value == value)
    if not values:
        return float("nan")
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def bootstrap_intervals(article_metrics, iterations=2000, seed=2027):
    """Article-level nonparametric 95% intervals for headline metrics."""
    if not article_metrics or iterations <= 0:
        return {}
    rng = random.Random(seed)
    draws = {key: [] for key in (
        "det_p", "det_r", "det_f1", "end_p", "end_r", "end_f1",
        "top_acc", "alt_acc", "section_acc", "macro_det_f1",
    )}
    n = len(article_metrics)
    for _ in range(iterations):
        sampled = [article_metrics[rng.randrange(n)] for _i in range(n)]
        metrics = aggregate_articles(sampled)
        for key in draws:
            draws[key].append(metrics[key])
    return {
        key: {"low": percentile(values, 0.025), "high": percentile(values, 0.975)}
        for key, values in draws.items()
    }


def paired_bootstrap_difference(a_articles, b_articles, iterations=2000, seed=2027):
    """Paired article-resampling intervals for A minus B on shared documents."""
    a_by_stem = {row["stem"]: row for row in a_articles}
    b_by_stem = {row["stem"]: row for row in b_articles}
    common = sorted(set(a_by_stem) & set(b_by_stem))
    if not common or iterations <= 0:
        return {"shared_articles": len(common), "differences": {}}
    rng = random.Random(seed)
    keys = ("det_f1", "end_f1", "section_acc")
    draws = {key: [] for key in keys}
    point_a = aggregate_articles([a_by_stem[stem] for stem in common])
    point_b = aggregate_articles([b_by_stem[stem] for stem in common])
    for _ in range(iterations):
        sampled_stems = [common[rng.randrange(len(common))] for _i in common]
        a_metrics = aggregate_articles([a_by_stem[stem] for stem in sampled_stems])
        b_metrics = aggregate_articles([b_by_stem[stem] for stem in sampled_stems])
        for key in keys:
            draws[key].append(a_metrics[key] - b_metrics[key])
    return {
        "shared_articles": len(common),
        "differences": {
            key: {
                "point": point_a[key] - point_b[key],
                "low": percentile(values, 0.025),
                "high": percentile(values, 0.975),
                "probability_a_better": sum(value > 0 for value in values) / len(values),
            }
            for key, values in draws.items()
        },
    }


def fmt(value):
    return f"{value:.3f}" if value == value else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH)
    parser.add_argument("--json-out", help="optional path for machine-readable metrics")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="article-bootstrap iterations (recommended final run: 5000)")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--score-out-of-scope", action="store_true",
        help="count predictions below the five-word minimum as false positives "
             "instead of excluding them; for reporting the sensitivity of the "
             "exclusion, which moves several systems",
    )
    parser.add_argument("--no-legacy-gold", action="store_true",
                        help="fail rather than fall back to the legacy census schema "
                             "(use for the frozen run)")
    parser.add_argument("--paired", action="store_true",
                        help="also compare every model pair on their shared articles")
    args = parser.parse_args()

    global ALLOW_LEGACY_GOLD, EXCLUDE_OUT_OF_SCOPE
    ALLOW_LEGACY_GOLD = not args.no_legacy_gold
    EXCLUDE_OUT_OF_SCOPE = not args.score_out_of_scope
    by_tag = discover()
    if not by_tag:
        raise SystemExit(
            "No matched gold/candidate/result bundles found. Add evaluation/gold/<stem>.gold.json "
            "and run detection plus run_pipeline.py."
        )

    # discover() silently skips any (article, tag) missing a candidates or results
    # file, and detect_quotes.py deliberately continues a batch past a transport or
    # malformed-output failure without writing candidates. Together those mean a
    # detector that failed on some articles is scored only over the ones it covered
    # -- which INFLATES its row -- with nothing but the 'art' column to show it. Say
    # so loudly; on the frozen run this is the difference between a reported result
    # and a wrong one.
    gold_stems = {path.name[:-len(".gold.json")] for path in GOLD_DIR.glob("*.gold.json")}
    for tag in sorted(by_tag):
        covered = {bundle[0] for bundle in by_tag[tag]}
        missing = gold_stems - covered
        if missing:
            print(f"WARNING: {tag} covers {len(covered)}/{len(gold_stems)} articles with "
                  f"gold; missing {', '.join(sorted(missing))}. Its scores are computed "
                  f"only over the articles it covers.")

    header = (
        f"{'model_tag':25} {'art':>3} {'cand':>4} {'gold':>4} "
        f"{'dP':>5} {'dR':>5} {'dF1':>5} {'eP':>5} {'eR':>5} {'eF1':>5} "
        f"{'topA':>5} {'altA':>5} {'secA':>5} {'macF1':>5} {'oos':>4}"
    )
    print(header)
    print("-" * len(header))
    output = {
        "match_threshold": args.match_threshold,
        "minimum_quote_words": MIN_QUOTE_WORDS,
        "out_of_scope_predictions_excluded": EXCLUDE_OUT_OF_SCOPE,
        "models": {},
    }
    for tag in sorted(by_tag):
        metrics = score_tag(by_tag[tag], args.match_threshold)
        if args.bootstrap:
            metrics["bootstrap_95"] = bootstrap_intervals(
                metrics["article_metrics"], args.bootstrap, args.seed
            )
        # Out-of-scope proposals are excluded from scoring but reported per system:
        # the count is the compliance statistic, the metrics are the detection result.
        metrics["out_of_scope_predictions_excluded"] = OUT_OF_SCOPE_BY_TAG.get(tag, 0)
        output["models"][tag] = metrics
        print(
            f"{tag[:25]:25} {metrics['articles']:>3} {metrics['candidates']:>4} "
            f"{metrics['gold']:>4} {fmt(metrics['det_p']):>5} {fmt(metrics['det_r']):>5} "
            f"{fmt(metrics['det_f1']):>5} {fmt(metrics['end_p']):>5} "
            f"{fmt(metrics['end_r']):>5} {fmt(metrics['end_f1']):>5} "
            f"{fmt(metrics['top_acc']):>5} {fmt(metrics['alt_acc']):>5} "
            f"{fmt(metrics['section_acc']):>5} {fmt(metrics['macro_det_f1']):>5} "
            f"{metrics['out_of_scope_predictions_excluded']:>4}"
        )
        strata_line = "  ".join(
            f"{key}={fmt(bucket['recall'])} ({bucket['tp']}/{bucket['tp'] + bucket['fn']})"
            for key, bucket in metrics["recall_by_stratum"].items()
        )
        if strata_line:
            print(f"  detection recall by stratum: {strata_line}")
        if args.bootstrap:
            det_ci = metrics["bootstrap_95"]["det_f1"]
            end_ci = metrics["bootstrap_95"]["end_f1"]
            sec_ci = metrics["bootstrap_95"]["section_acc"]
            print(
                f"  95% article-bootstrap CI: dF1 [{fmt(det_ci['low'])}, "
                f"{fmt(det_ci['high'])}]  eF1 [{fmt(end_ci['low'])}, "
                f"{fmt(end_ci['high'])}]  secA [{fmt(sec_ci['low'])}, "
                f"{fmt(sec_ci['high'])}]"
            )

    if args.paired:
        if not args.bootstrap:
            raise SystemExit("--paired requires --bootstrap")
        output["paired_comparisons"] = {}
        for tag_a, tag_b in combinations(sorted(output["models"]), 2):
            # Seed from the PAIR'S IDENTITY, not its enumeration index. Deriving it
            # from position (seed + pair_i) made every reported interval depend on
            # which other systems happened to be present: adding the hybrid row
            # shifted the Fable-vs-oracle interval from [-0.076, 0.000] to
            # [-0.070, 0.000] without either system changing. Since archiving stale
            # candidate files also changes the tag set, published CIs would move for
            # reasons that have nothing to do with the comparison. crc32 is used for
            # a stable cross-run digest, not for cryptographic strength.
            pair_seed = args.seed + zlib.crc32(
                f"{tag_a}|{tag_b}".encode("utf-8")) % 1_000_003
            comparison = paired_bootstrap_difference(
                output["models"][tag_a]["article_metrics"],
                output["models"][tag_b]["article_metrics"],
                args.bootstrap,
                pair_seed,
            )
            key = f"{tag_a}__minus__{tag_b}"
            output["paired_comparisons"][key] = comparison
            if comparison["differences"]:
                det = comparison["differences"]["det_f1"]
                end = comparison["differences"]["end_f1"]
                sec = comparison["differences"]["section_acc"]
                print(
                    f"pair {tag_a} - {tag_b} (n={comparison['shared_articles']}): "
                    f"ΔdF1={det['point']:.3f} [{det['low']:.3f}, {det['high']:.3f}] "
                    f"ΔeF1={end['point']:.3f} [{end['low']:.3f}, {end['high']:.3f}] "
                    f"ΔsecA={sec['point']:.3f} [{sec['low']:.3f}, {sec['high']:.3f}]"
                )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
