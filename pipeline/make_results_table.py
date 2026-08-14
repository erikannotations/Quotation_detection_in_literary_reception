#!/usr/bin/env python3
"""Render the CHR 2027 short paper's Table 1 from evaluation_metrics.json.

Reads the machine-readable metrics file produced by ``eval_compare.py
--json-out`` (see that script's ``main()`` for the exact structure this
depends on: ``{"match_threshold": ..., "models": {<tag>: {..., "bootstrap_95":
{...}}}}``) and formats the seven-column results table used in the paper:

    System | Detection P | Detection R | Detection F1 | End-to-end F1 |
    Top-source accuracy | Chapter/episode accuracy

Model tags in the JSON come from whatever detector/eval scripts produced the
candidate files (e.g. ``oracle_gold``, ``heuristic_marked``,
``quid_2.6.7_hp128_s0p85``, ``anthropic_claude-3-5-...``,
``ollama_llama3.1-70b``, ``hybrid``). This script maps those raw tags onto
the paper's six fixed systems (O, M, Q, G-C, G-O, H) with a small set of
keyword heuristics, and never fabricates a value: a system with no matching
tag is skipped (with a stderr warning) rather than filled in, and an
unrecognised tag is appended as its own row rather than silently dropped.

Usage:
    python3 make_results_table.py [--metrics-path PATH] [--format md|tex]
                                   [--output PATH] [--system-map PATH]
    python3 make_results_table.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# The frozen run writes dev_metrics_<date>.json / test_metrics_<date>.json.
# Pick the newest metrics file actually present, and keep --metrics-path explicit.
def _default_metrics_path():
    candidates = sorted((SCRIPT_DIR / "evaluation").glob("*metrics*.json"))
    return candidates[-1] if candidates else (
        SCRIPT_DIR / "evaluation" / "evaluation_metrics.json")


DEFAULT_METRICS_PATH = _default_metrics_path()

# Fixed row order for Table 1, and the exact system labels used in the paper.
# Each hybrid is its own row: the variants answer different questions, so they
# are not interchangeable.
CANONICAL_ORDER = ["O", "M", "Q", "T", "T-STRICT", "G-C", "G-O",
                   "H", "H-QG", "H-MGC", "H-MGO", "H-GCGO", "H-XQGC", "H-ALL",
                   "H-TGC", "H-TGO", "H-QT", "H-XTGC"]
# The draft splits the results across two tables so the main one stays readable in a
# short paper: Table 1 carries the systems declared at the freeze plus the second
# lexical baseline, Table 2 the four detector combinations involving that baseline.
# The split is presentational only — every declared variant is still reported, which is
# what the predeclaration commits to.
SUPPLEMENTARY_ORDER = ["H-TGC", "H-TGO", "H-QT", "H-XTGC"]
CANONICAL_LABELS_MD = {
    "O": "Oracle gold spans → locator",
    "M": "Marked-span heuristic",
    "Q": "Quid 2.6.7",
    "T": "text-matcher (evaluation setting)",
    "T-STRICT": "text-matcher (authors' defaults)",
    "G-C": "Closed generative detector",
    "G-O": "Open-weight generative detector",
    "H": "Hybrid: Quid ∪ closed",
    "H-QG": "Hybrid: marked-gated Quid ∪ closed",
    "H-MGC": "Hybrid: heuristic ∪ closed",
    "H-MGO": "Hybrid: heuristic ∪ open (no API)",
    "H-GCGO": "Hybrid: closed ∪ open",
    "H-XQGC": "Hybrid: Quid ∩ closed (agreement only)",
    "H-ALL": "Hybrid: all four detectors",
    "H-TGC": "Hybrid: text-matcher ∪ closed",
    "H-TGO": "Hybrid: text-matcher ∪ open (no API)",
    "H-QT": "Hybrid: Quid ∪ text-matcher (no LLM)",
    "H-XTGC": "Hybrid: text-matcher ∩ closed (agreement only)",
}
CANONICAL_LABELS_TEX = {
    "O": r"Oracle gold spans $\to$ locator",
    "M": "Marked-span heuristic",
    "Q": "Quid 2.6.7",
    "T": "text-matcher (evaluation setting)",
    "T-STRICT": "text-matcher (authors' defaults)",
    "G-C": "Closed generative detector",
    "G-O": "Open-weight generative detector",
    "H": r"Hybrid: Quid $\cup$ closed",
    "H-QG": r"Hybrid: marked-gated Quid $\cup$ closed",
    "H-MGC": r"Hybrid: heuristic $\cup$ closed",
    "H-MGO": r"Hybrid: heuristic $\cup$ open (no API)",
    "H-GCGO": r"Hybrid: closed $\cup$ open",
    "H-XQGC": r"Hybrid: Quid $\cap$ closed (agreement only)",
    "H-ALL": "Hybrid: all four detectors",
    "H-TGC": r"Hybrid: text-matcher $\cup$ closed",
    "H-TGO": r"Hybrid: text-matcher $\cup$ open (no API)",
    "H-QT": r"Hybrid: Quid $\cup$ text-matcher (no LLM)",
    "H-XTGC": r"Hybrid: text-matcher $\cap$ closed (agreement only)",
}

# (column header, metrics key, eligible for a bootstrap-CI suffix)
COLUMNS = [
    ("Detection P", "det_p", False),
    ("Detection R", "det_r", False),
    ("Detection F1", "det_f1", True),
    ("End-to-end F1", "end_f1", True),
    ("Top-source accuracy", "top_acc", False),
    ("Chapter/episode accuracy", "section_acc", False),
]

# Oracle bypasses detection by construction (its candidates ARE the gold
# spans), so the paper's Table 1 renders those three cells as an em dash
# rather than a tautological 1.000. This mirrors the paper's own table
# design, not an invented number: End-to-end F1 and Top-source accuracy for
# Oracle still come straight out of the JSON.
ORACLE_NA_KEYS = {"det_p", "det_r", "det_f1"}

CLOSED_MARKERS = (
    "anthropic", "claude", "openai", "gpt-", "gpt3", "gpt4", "gpt5",
    "gemini", "google", "cohere", "azure",
)
OPEN_MARKERS = (
    "ollama", "vllm", "together", "groq", "huggingface", "hf_", "llama",
    "qwen", "mixtral", "mistral", "deepseek", "gemma", "phi-", "phi3",
    "local_", "open-weight", "open_weight",
)


def classify_tag(tag, overrides=None):
    """Map a raw model_tag from the JSON onto a canonical system code.

    Returns one of CANONICAL_ORDER, or None if the tag is not recognised
    (in which case the caller appends it as its own row rather than
    dropping or misattributing it).
    """
    if overrides and tag in overrides:
        return overrides[tag]
    t = tag.lower()
    if "oracle" in t:
        return "O"
    if "heuristic" in t or "marked" in t:
        return "M"
    if t.startswith("quid") or "quid" in t:
        return "Q"
    if "hybrid" in t:
        # No "H" fallback. An unrecognised hybrid returns None and is appended as its
        # own row: defaulting to "H" is what previously collapsed every variant onto
        # one line and silently kept whichever sorted first.
        return {"hybrid": "H", "hybrid_qgated": "H-QG", "hybrid_mgc": "H-MGC",
                "hybrid_mgo": "H-MGO", "hybrid_gcgo": "H-GCGO",
                "hybrid_xqgc": "H-XQGC", "hybrid_all": "H-ALL",
                "hybrid_tgc": "H-TGC", "hybrid_tgo": "H-TGO",
                "hybrid_qt": "H-QT", "hybrid_xtgc": "H-XTGC"}.get(t)
    if "textmatcher" in t:
        return "T-STRICT" if "t3c5n3m8" in t else "T"
    if any(marker in t for marker in CLOSED_MARKERS):
        return "G-C"
    if any(marker in t for marker in OPEN_MARKERS):
        return "G-O"
    return None


def load_metrics(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"error: metrics file not found: {p}\n"
            "The evaluation has not produced results yet. Run, e.g.:\n"
            f"  python3 eval_compare.py --bootstrap 5000 --json-out {p}"
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: could not parse JSON at {p}: {exc}")
    if "models" not in data or not isinstance(data["models"], dict):
        raise SystemExit(
            f"error: {p} does not look like eval_compare.py --json-out output "
            "(no top-level 'models' object)"
        )
    return data


def fmt_num(value):
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        return None
    return f"{value:.3f}"


def format_cell(metrics, key, ci_eligible, na_text):
    text = fmt_num(metrics.get(key))
    if text is None:
        text = na_text
    if ci_eligible:
        boot = metrics.get("bootstrap_95") or {}
        ci = boot.get(key)
        if ci:
            low = fmt_num(ci.get("low"))
            high = fmt_num(ci.get("high"))
            if low is not None and high is not None:
                text = f"{text} [{low}, {high}]"
    return text


def build_rows(data, system_map=None, order=None, expected=None):
    """Return (rows, warnings). rows = list of (label_key, code, metrics).

    `order` selects and orders the rows; `expected` is the set whose absence is worth a
    warning. Unrecognised tags are appended to the main table only, so a supplementary
    selection cannot silently duplicate them.
    """
    models = data.get("models", {})
    warnings = []
    classified = {}
    unknown = []
    for tag, metrics in models.items():
        code = classify_tag(tag, system_map)
        if code is None:
            unknown.append((tag, metrics))
            continue
        if code in classified:
            prev_tag, _ = classified[code]
            msg = (
                f"warning: both '{prev_tag}' and '{tag}' map to system '{code}'; "
                f"keeping '{prev_tag}' and ignoring '{tag}'"
            )
            warnings.append(msg)
            continue
        classified[code] = (tag, metrics)

    order = CANONICAL_ORDER if order is None else order
    expected = set(order if expected is None else expected)
    rows = []
    for code in order:
        if code not in classified:
            if code in expected:
                msg = (f"warning: no model_tag mapped to system '{code}' "
                       f"({CANONICAL_LABELS_MD[code]}); skipping row")
                warnings.append(msg)
            continue
        tag, metrics = classified[code]
        rows.append((code, tag, metrics))
    if order is CANONICAL_ORDER or set(order) == set(CANONICAL_ORDER):
        for tag, metrics in unknown:
            rows.append((None, tag, metrics))

    for msg in warnings:
        print(msg, file=sys.stderr)
    return rows, warnings


def row_values(code, metrics, na_text):
    values = []
    for _header, key, ci_eligible in COLUMNS:
        if code == "O" and key in ORACLE_NA_KEYS:
            values.append(na_text)
        else:
            values.append(format_cell(metrics, key, ci_eligible, na_text))
    return values


def render_markdown(rows, first_header="System", strip_prefix=""):
    headers = [first_header] + [h for h, _k, _c in COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" + "---:|" * len(COLUMNS),
    ]
    for code, tag, metrics in rows:
        label = CANONICAL_LABELS_MD.get(code, tag).removeprefix(strip_prefix)
        values = row_values(code, metrics, na_text="—")
        lines.append("| " + " | ".join([label] + values) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "→": r"$\to$",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def render_latex(rows, first_header="System", strip_prefix=""):
    headers = [first_header] + [h for h, _k, _c in COLUMNS]
    col_spec = "l" + "r" * len(COLUMNS)
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for code, tag, metrics in rows:
        if code in CANONICAL_LABELS_TEX:
            label = CANONICAL_LABELS_TEX[code].removeprefix(strip_prefix)
        else:
            label = latex_escape(tag)
        values = row_values(code, metrics, na_text="---")
        lines.append(" & ".join([label] + values) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def render_table(data, fmt, system_map=None, table="main"):
    strip = ""
    if table == "supplementary":
        order = SUPPLEMENTARY_ORDER
        header = "Combination"
        # The column header already says these are combinations.
        strip = "Hybrid: "
    elif table == "main":
        order = [c for c in CANONICAL_ORDER if c not in SUPPLEMENTARY_ORDER]
        header = "System"
    else:
        order, header = CANONICAL_ORDER, "System"
    rows, _warnings = build_rows(data, system_map, order=order, expected=order)
    if not rows:
        raise SystemExit("error: no systems in the JSON matched any known or unknown row; nothing to render")
    if fmt == "md":
        return render_markdown(rows, header, strip)
    if fmt == "tex":
        return render_latex(rows, header, strip)
    raise SystemExit(f"error: unknown --format \'{fmt}\' (expected md or tex)")


def synthetic_metrics():
    """A small in-memory metrics dict matching eval_compare.py's --json-out shape.

    Deliberately omits a Quid tag (to exercise the missing-row warning) and
    includes one unrecognised tag (to exercise the unknown-row append path).
    """
    def article(stem, det_f1, end_f1):
        return {"stem": stem, "articles": 1, "det_f1": det_f1}

    def block(det_p, det_r, det_f1, end_p, end_r, end_f1, top_acc, alt_acc,
              section_acc, macro_det_f1, with_ci=False):
        m = {
            "articles": 4, "candidates": 40, "gold": 40,
            "det_tp": 36, "det_fp": 4, "det_fn": 4,
            "end_tp": 30, "end_fp": 6, "end_fn": 4,
            "top_ok": 30, "alt_ok": 33, "top_denom": 36,
            "section_ok": 27, "section_denom": 36,
            "det_p": det_p, "det_r": det_r, "det_f1": det_f1,
            "end_p": end_p, "end_r": end_r, "end_f1": end_f1,
            "top_acc": top_acc, "alt_acc": alt_acc, "section_acc": section_acc,
            "macro_det_f1": macro_det_f1,
            "article_metrics": [article("a1", det_f1, end_f1), article("a2", det_f1, end_f1)],
        }
        if with_ci:
            m["bootstrap_95"] = {
                "det_p": {"low": det_p - 0.05, "high": det_p + 0.04},
                "det_r": {"low": det_r - 0.05, "high": det_r + 0.04},
                "det_f1": {"low": round(det_f1 - 0.05, 3), "high": round(det_f1 + 0.04, 3)},
                "end_p": {"low": end_p - 0.05, "high": end_p + 0.04},
                "end_r": {"low": end_r - 0.05, "high": end_r + 0.04},
                "end_f1": {"low": round(end_f1 - 0.06, 3), "high": round(end_f1 + 0.05, 3)},
                "top_acc": {"low": top_acc - 0.05, "high": top_acc + 0.04},
                "alt_acc": {"low": alt_acc - 0.05, "high": alt_acc + 0.04},
                "section_acc": {"low": section_acc - 0.05, "high": section_acc + 0.04},
                "macro_det_f1": {"low": macro_det_f1 - 0.05, "high": macro_det_f1 + 0.04},
            }
        return m

    return {
        "match_threshold": 0.6,
        "models": {
            "oracle_gold": block(1.000, 1.000, 1.000, 0.944, 0.944, 0.944, 0.944, 1.000, 0.861, 1.000),
            "heuristic_marked": block(0.850, 0.700, 0.768, 0.700, 0.583, 0.636, 0.900, 0.950, 0.750, 0.760, with_ci=True),
            # No Quid tag on purpose: exercises the missing-row warning.
            "anthropic_claude-3-5-sonnet-20241022": block(0.900, 0.880, 0.890, 0.820, 0.800, 0.810, 0.930, 0.960, 0.780, 0.885, with_ci=True),
            "ollama_llama3.1-70b-instruct": block(0.860, 0.840, 0.850, 0.760, 0.740, 0.750, 0.880, 0.920, 0.730, 0.845, with_ci=True),
            "hybrid": block(0.870, 0.910, 0.890, 0.790, 0.820, 0.805, 0.900, 0.940, 0.760, 0.880, with_ci=True),
            "some_future_detector_v9": block(0.500, 0.500, 0.500, 0.400, 0.400, 0.400, 0.600, 0.650, 0.450, 0.500),
        },
    }


def run_self_test():
    data = synthetic_metrics()

    rows, warnings = build_rows(data)
    codes = [code for code, _tag, _m in rows]
    assert "Q" not in codes, "Quid was deliberately omitted from synthetic data"
    assert any("no model_tag mapped to system 'Q'" in w for w in warnings), (
        "expected a stderr warning about the missing Quid row"
    )
    assert codes[-1] is None, "the unrecognised tag should be appended as the last row"
    unknown_tag = rows[-1][1]
    assert unknown_tag == "some_future_detector_v9"

    md = render_markdown(rows)
    tex = render_latex(rows)

    for text, label in ((md, "markdown"), (tex, "latex")):
        assert "some_future_detector_v9" in text or latex_escape("some_future_detector_v9") in text, (
            f"unknown system missing from {label} output"
        )

    assert "Oracle gold spans" in md
    assert "Oracle gold spans $\\to$ locator" in tex
    # Oracle detection cells must be the structural N/A marker, not 1.000.
    oracle_md_line = [line for line in md.splitlines() if line.startswith("| Oracle")][0]
    assert oracle_md_line.count("—") == 3, f"expected 3 em dashes in Oracle md row, got: {oracle_md_line}"
    oracle_tex_line = [line for line in tex.splitlines() if line.startswith("Oracle")][0]
    assert oracle_tex_line.count("---") == 3, f"expected 3 --- markers in Oracle tex row, got: {oracle_tex_line}"

    # A system with bootstrap CIs must show a bracketed interval on F1 columns.
    assert "0.890 [0." in md, "expected a CI-annotated F1 value in markdown output"
    assert "0.890 [0." in tex, "expected a CI-annotated F1 value in latex output"

    # Top-source and chapter/episode accuracy must NOT carry a CI suffix even
    # though bootstrap_95 contains one for both (CIs are appended to F1 columns
    # only).
    for line in md.splitlines():
        if line.startswith("| Hybrid"):
            assert "0.900 [" not in line, f"top-source accuracy should have no CI suffix: {line}"
            assert "0.760 [" not in line, f"chapter/episode accuracy should have no CI suffix: {line}"

    # System label appears in both formats.
    assert "Marked-span heuristic" in md and "Marked-span heuristic" in tex

    print("self-test: OK (markdown and LaTeX renders both passed all assertions)")
    print()
    print("--- markdown sample ---")
    print(md)
    print("--- latex sample ---")
    print(tex)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-path", default=str(DEFAULT_METRICS_PATH),
        help="path to evaluation_metrics.json (default: evaluation/evaluation_metrics.json next to this script)",
    )
    parser.add_argument("--format", choices=["md", "tex"], default="md", help="output format (default: md)")
    parser.add_argument("--output", help="write to this file instead of stdout")
    parser.add_argument(
        "--system-map", help="optional JSON file of {raw_model_tag: system_code} overrides for classify_tag()",
    )
    parser.add_argument(
        "--table", choices=["main", "supplementary", "all"], default="main",
        help="which table to render: 'main' is Table 1 (systems declared at the freeze "
             "plus the second lexical baseline), 'supplementary' is Table 2 (the four "
             "combinations involving that baseline), 'all' is one combined table "
             "(default: main)",
    )
    parser.add_argument("--self-test", action="store_true", help="run the built-in self-test and exit")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    system_map = None
    if args.system_map:
        system_map = json.loads(Path(args.system_map).read_text(encoding="utf-8"))

    data = load_metrics(args.metrics_path)
    table_text = render_table(data, args.format, system_map, args.table)

    if args.output:
        Path(args.output).write_text(table_text, encoding="utf-8")
    else:
        sys.stdout.write(table_text)


if __name__ == "__main__":
    main()
