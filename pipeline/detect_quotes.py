#!/usr/bin/env python3
"""Detection step (backend-agnostic): extract marked direct Pynchon-fiction quotations
from a scholarly article, for the model-comparison evaluation.

Identical prompt and required output fields across backends so only the *model*
varies; the downstream locate step (run_pipeline.py) is deterministic and shared.
Each backend uses its own idiomatic mechanism for getting that structured output
out reliably (see the note on --backend ollama below) rather than an identical
wire protocol, which is not the same claim.

Backends:
  --backend anthropic   Claude via the official SDK (needs `pip install anthropic`
                        + ANTHROPIC_API_KEY). messages.parse → validated output.
  --backend ollama      Local model via Ollama's REST API at OLLAMA_HOST
                        (default http://localhost:11434). Uses native tool-calling
                        (frozen policy, 22 July 2026 — see TOOL_DEFINITION below
                        and paper/audit_implementation_plan_2026-07-18.md), not
                        the `format` JSON-schema hint, which gemma4:26b-mlx was
                        observed to ignore even under a direct retry.

Output: <out-dir>/<article-stem>__<tag>.candidates.json, where <tag> identifies the
model (e.g. anthropic_claude-opus-4-8, ollama_gpt-oss-20b). run_pipeline.py then
locates it, and eval_compare.py scores all tags against a census.

Usage:
    python3 detect_quotes.py <article.txt> [more.txt ...] \\
        --backend ollama --model gpt-oss:20b
    python3 detect_quotes.py <article.txt> --backend anthropic --model claude-opus-4-8
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from project_paths import CANDIDATES_DIR
from quotation_spans import find_quote_offsets
from locate_quotes import normalize_query

SEP_RE = re.compile(r"^={10,}\s*$", re.M)

SYSTEM = """\
You are a precise literary-citation extractor for a corpus project on the reception \
of the novelist Thomas Pynchon.

You are given the body text of a scholarly article or book chapter about Pynchon. \
Extract every MARKED DIRECT QUOTATION the author reproduces from THOMAS PYNCHON'S \
FICTION — his novels and short stories:
V.; The Crying of Lot 49; Gravity's Rainbow; Vineland; Mason & Dixon; Against the \
Day; Inherent Vice; Bleeding Edge; Shadow Ticket; "Mortality and Mercy in Vienna"; \
and the Slow Learner stories ("The Small Rain", \
"Low-lands", "Entropy", "Under the Rose", "The Secret Integration").

Rules:
- MARKED DIRECT QUOTATIONS ONLY: text in quotation marks or set as a block quote. \
Never extract paraphrase, allusion, or the author's own prose.
- FIVE WORDS OR MORE. Do not extract single words or short phrases below five words, \
even if they are distinctively Pynchon's diction — the task is scoped to quotations of \
at least five words.
- Reproduce the quoted text EXACTLY as printed: keep ellipses (… or [...]), bracketed \
editorial insertions like [w]here, original/archaic spelling. Do not normalise it.
- Be GENEROUS / high recall. If a marked quotation plausibly comes from Pynchon's \
fiction, include it. A downstream step verifies every candidate against the actual \
novels, so borderline cases are fine to include.
- EXCLUDE quotations you can clearly tell are NOT Pynchon's fiction: quotations of \
critics/scholars, of other novelists (Kerouac, Melville…), of theorists (Foucault…), \
or of Pynchon's NON-fiction (essays, the Slow Learner Introduction, liner notes). \
When genuinely unsure, include it and mark confidence "low".
- Ignore scare-quotes, quoted titles, and single quoted words that are not fictional text.
- Capture the work the author attributes each quote to (if stated) and any page number.

Return the full list; return an empty list if there are none."""


def build_schema():
    from pydantic import BaseModel, field_validator

    class PynchonQuote(BaseModel):
        quote: str
        detector_guess: str = ""
        attributed_work: str = ""
        attributed_page: str = ""
        confidence: str = ""

        # Models idiomatically send explicit null (or a bare page number) for
        # optional fields they have no value for; a strict `str` annotation
        # rejected an otherwise perfectly-formed extract_quotes tool call for
        # exactly this on 22 July 2026 (Eve dev article: `"attributed_page":
        # null`). Coerce rather than fail — applies identically to both
        # backends, which validate through this same model.
        @field_validator("detector_guess", "attributed_work", "attributed_page",
                         "confidence", mode="before")
        @classmethod
        def _none_to_empty(cls, value):
            return "" if value is None else str(value)

    class Detection(BaseModel):
        quotes: list[PynchonQuote]

    return Detection


def article_body_and_key(path: Path):
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^Key:\s*(\S+)", text, re.M)
    key = m.group(1) if m else (path.stem.split("_")[0] or path.stem)
    parts = SEP_RE.split(text, maxsplit=1)
    body = parts[1] if len(parts) == 2 else text
    return body.strip(), key


# ---- backends: each returns a list of quote dicts ----
def detect_anthropic(model, body, Detection):
    """Frontier backend. Returns (quotes, runinfo).


    Predeclared policy, fixed here before any test article is opened: a response
    that stops for `max_tokens` (i.e. was truncated mid-output) gets exactly ONE
    retry at double the output budget; a second failure is recorded as a failure
    and raised. Nothing else is retried.
    """
    import time

    import anthropic
    client = anthropic.Anthropic()
    runinfo = {"backend": "anthropic", "model": model, "attempts": []}

    def attempt(max_tokens):
        started = time.monotonic()
        resp = client.messages.parse(
            model=model, max_tokens=max_tokens, thinking={"type": "adaptive"},
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "Article body follows.\n\n" + body}],
            output_format=Detection,
        )
        usage = getattr(resp, "usage", None)
        record = {
            "max_tokens": max_tokens,
            "stop_reason": getattr(resp, "stop_reason", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(
                usage, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(
                usage, "cache_creation_input_tokens", None),
            "wall_seconds": round(time.monotonic() - started, 3),
            "response_id": getattr(resp, "id", None),
        }
        runinfo["attempts"].append(record)
        return resp, record

    resp, record = attempt(16000)
    if record["stop_reason"] == "max_tokens":
        print("note: response truncated at max_tokens; one retry at double the "
              "output budget (predeclared policy)", file=sys.stderr)
        resp, record = attempt(32000)
        if record["stop_reason"] == "max_tokens":
            runinfo["failed"] = True
            raise ValueError(
                "anthropic response truncated at max_tokens twice; recorded as a "
                "failure rather than silently returning a partial quote list")

    quotes = [q.model_dump() for q in resp.parsed_output.quotes]
    runinfo["n_quotes"] = len(quotes)
    return quotes, runinfo


def _ollama_chat_once(host, payload):
    req = urllib.request.Request(
        host + "/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    content, tool_calls, final = [], [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            if msg.get("content"):
                content.append(msg["content"])
            if msg.get("tool_calls"):
                tool_calls.extend(msg["tool_calls"])
            if chunk.get("done"):
                final = {k: v for k, v in chunk.items() if k != "message"}
    final["message"] = {"role": "assistant", "content": "".join(content)}
    if tool_calls:
        final["message"]["tool_calls"] = tool_calls
    return final


def _ollama_chat(host, payload):
    """POST /api/chat with stream=True, aggregating the chunks into one response.

    Streaming is load-bearing, not cosmetic (22 July 2026): in non-streaming mode
    this Ollama version's gemma4 response parser can return a completely empty
    message — no content, no tool_calls — for a response whose eval_count shows
    thousands of generated tokens, silently discarding output it fails to parse
    (observed on the Jiménez Heffernan article, where the model emitted its
    quote list as a markdown-fenced JSON block instead of a native tool call).
    With stream=True the same output arrives as content deltas, so a
    format-drifted response stays inspectable and salvageable downstream.
    """
    payload = {**payload, "stream": True}
    try:
        return _ollama_chat_once(host, payload)
    except urllib.error.HTTPError:
        if "think" not in payload:
            raise
        # Older Ollama versions reject the top-level `think` key.
        return _ollama_chat_once(host, {k: v for k, v in payload.items()
                                        if k != "think"})


TOOL_NAME = "extract_quotes"

# Frozen backend policy (decided 22 July 2026, applied uniformly from the first
# dev run onward — not a per-failure patch): Ollama uses native tool-calling
# rather than the `format` JSON-schema hint. The hint is a soft constraint that
# gemma4:26b-mlx was observed to ignore outright, including on a direct
# "return ONLY JSON" retry (see legacy/ollama_format_schema_superseded/). A live
# comparison against this project's own Ollama instance showed tool-calling
# producing clean structured output, with no prose fallback, on a test
# including a distractor quotation that should be excluded.
#
# The tool's parameter schema is hand-flattened (no $ref/$defs) for maximum
# local-model compatibility; the full nested Detection pydantic model still
# does the actual validation after the call, so the rest of the pipeline is
# unaffected by this choice.
#
# Amended 22 July 2026 (uniform, applied before any final dev run; see the
# audit log's "gemma4 structured-output diagnosis" entry): a response that
# contains no native tool call but does carry the quote list as JSON in plain
# content (observed drift mode: a markdown-fenced block) is accepted through
# salvage_detection() below, validated against the same schema, and recorded
# per-attempt in the .runinfo.json sidecar as a format deviation. The one
# reminder retry then applies only when neither mechanism yields a valid
# Detection.
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Record every marked direct quotation of Thomas Pynchon's fiction found in the article.",
        "parameters": {
            "type": "object",
            "properties": {
                "quotes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "quote": {"type": "string", "description": "The quotation exactly as printed, including ellipses and archaic spelling."},
                            "detector_guess": {"type": "string"},
                            "attributed_work": {"type": "string", "description": "The Pynchon work the article attributes this quote to, if stated."},
                            "attributed_page": {"type": "string", "description": "Page number cited, if any."},
                            "confidence": {"type": "string", "description": "'low' if genuinely unsure this is Pynchon's fiction."},
                        },
                        "required": ["quote"],
                    },
                },
            },
            "required": ["quotes"],
        },
    },
}

# tool_choice="required" is accepted by this Ollama version's API but is NOT
# reliably enforced by gemma4:26b-mlx in practice (confirmed empirically: the
# model still answered in plain prose, ignoring it, on a distractor prompt
# during development). It is sent anyway as a harmless hint; the real
# enforcement is the explicit tool_calls check below plus its one retry.
TOOL_RETRY_REMINDER = (
    "You did not call the extract_quotes tool. Call it now with every "
    "quotation you found, even if you already described them in text. Do not "
    "respond with plain text.")

# Server-side tool-call swallow (diagnosed 26 July 2026; see the audit log entry
# "gemma4 native tool-call swallow"). Intermittently (~1 run in 3, nondeterministic
# despite temperature 0 on this MLX build) gemma4 answers through its NATIVE
# tool-call channel rather than its usual markdown-fenced JSON. Ollama 0.32.1's
# gemma4 parser routes those tokens away from the content stream, buffers them,
# attempts a JSON repair, and — when the repair fails — DISCARDS the entire
# generation: no content, no tool_calls, no error, done_reason "stop".
#
# The repair fails on the model's known tic, the very one _close_unterminated_strings()
# handles on the content path: one string value whose closing delimiter is omitted
# because the quoted text itself ends in a typographic quote. In the captured
# failure exactly 1 of 143 values was unterminated, and 11,016 characters holding
# 48 correct quote entries were thrown away server-side.
#
# Streaming does not protect against this. Streaming fixed the CONTENT-path swallow
# (22 July); tokens inside a candidate tool call are never emitted as content deltas
# at all, so the loss is upstream of chunk emission and nothing client-side can
# recover the bytes — they were never sent.
#
# The reminder retry is actively counterproductive here on two counts: the retry
# conversation carries an EMPTY assistant message (the model cannot see its own
# answer), and the reminder's "Do not respond with plain text" pushes it straight
# back into the broken native channel. Both observed attempts of the failing run
# were swallowed identically, making the failure total.
#
# Recovery is therefore to re-ask WITHOUT the tools declaration, which deterministically
# routes the answer down the content path that the existing salvage machinery already
# handles (verified live). This replaces the reminder retry rather than adding to it,
# holding the attempt budget at two.
SWALLOW_MIN_EVAL_TOKENS = 200


def tool_call_swallowed(record):
    """True if this attempt shows the Ollama gemma4 tool-call swallow signature.

    Substantial generation (eval_count) that produced neither content nor a tool
    call, and stopped of its own accord rather than hitting the output cap.
    """
    return (record.get("done_reason") != "length"
            and not record.get("content_chars")
            and not record.get("n_tool_calls")
            and (record.get("eval_count") or 0) >= SWALLOW_MIN_EVAL_TOKENS)


def parse_tool_call(data, Detection):
    """Extract and validate the extract_quotes tool call, if the model made one."""
    message = data.get("message", {})
    for call in message.get("tool_calls") or []:
        if call.get("function", {}).get("name") == TOOL_NAME:
            return Detection.model_validate(call["function"].get("arguments", {}))
    raise ValueError("no extract_quotes tool call in response")


_ENTRY_RE = re.compile(r"\{[^{}]*\}", re.S)
FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_STRING_PROP_RE = re.compile(
    r'^\s*"(?:quote|detector_guess|attributed_work|attributed_page|confidence)":\s*"')
_TERMINATED_RE = re.compile(r'"\s*,?\s*$')


def _close_unterminated_strings(raw):
    """Deterministically repair the one malformation gemma4:26b-mlx was observed
    to produce inside otherwise well-formed JSON: a string value whose closing
    '"' is missing because the quoted text itself ends in a typographic quote
    (e.g. `…small boys inside”` followed by a bare newline). Only lines that
    open a known string property, do not terminate it, AND are followed by a
    structural line (`"`, `{`, or `}`) are touched — a line followed by prose is
    the continuation of a legitimately multi-line block quotation and must not
    be truncated.
    """
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if _STRING_PROP_RE.match(line) and not _TERMINATED_RE.search(line.rstrip()):
            following = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
            if following.startswith(('"', "{", "}")):
                lines[i] = line.rstrip() + '"' + ("," if following.startswith('"') else "")
    return "\n".join(lines)


def _strip_json_line_comments(raw):
    """Remove JavaScript-style `// …` line comments, which are not legal JSON.

    gemma4:26b-mlx annotates its own output when it is unsure, e.g.
    `"attributed_page": "753", // Note: text says 753 but contextually likely …`.
    One such comment invalidates the WHOLE document, and because the comment sits
    inside an entry's braces it also defeats per-entry recovery for that entry —
    so each annotated quotation was being dropped silently. Observed 26 July 2026
    on Smith: comment-bearing runs returned 32-33 candidates where comment-free
    runs on the same article returned 36-37.

    The scan is string-aware: a `//` inside a JSON string (a URL, or Pynchon's own
    punctuation) is left alone, and backslash escapes are honoured so an escaped
    quote does not appear to end a string.
    """
    out = []
    for line in raw.split("\n"):
        in_string = escaped = False
        cut = None
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = not in_string
            elif (not in_string and ch == "/"
                  and i + 1 < len(line) and line[i + 1] == "/"):
                cut = i
                break
        # Only strip when a comment was found OUTSIDE a string. A line that ends
        # mid-string is a multi-line block quotation; leave it for
        # _close_unterminated_strings to judge.
        out.append(line[:cut].rstrip() if cut is not None else line)
    return "\n".join(out)


def _candidate_json_payloads(content):
    """Yield candidate JSON strings from response content, most likely first:
    the outermost {...} of each fenced block in turn, then of the whole content.
    Trying each fenced block separately matters when the model emits more than
    one block (e.g. an example before its real answer) — a single greedy match
    across blocks would span them and parse as garbage.
    """
    def _spans(text):
        # Both shapes, object first. The model sometimes answers with a BARE ARRAY
        # of quote objects instead of the {"quotes": [...]} envelope; scanning only
        # for {...} then spans from the first entry's brace to the last entry's,
        # which is "Extra data", never valid JSON. Observed 26 July 2026 on a
        # windowed run where 29 perfectly good quotations were discarded for this
        # reason alone. Arrays are wrapped into the envelope in salvage_detection.
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = text.find(opener), text.rfind(closer)
            if start != -1 and end > start:
                yield text[start:end + 1]

    for match in FENCED_BLOCK_RE.finditer(content):
        yield from _spans(match.group(1))
    yield from _spans(content)


def salvage_detection(content, Detection, stats=None):
    """Salvage acceptance path (uniform policy amendment, 22 July 2026): when the
    model returns its quote list as JSON in plain content — typically a
    markdown-fenced block — instead of a native tool call, extract it, parse it
    (strict=False tolerates literal newlines inside block quotations, plus the
    single deterministic repair above), and validate it against the SAME pydantic
    schema the tool call would have passed through. Acceptance via this path is
    recorded as a format deviation in the run info, never silently.

    Returns (Detection, mechanism_label); raises ValueError if no usable JSON.
    """
    tried = False
    for raw in _candidate_json_payloads(content):
        tried = True
        decommented = _strip_json_line_comments(raw)
        for candidate, label in ((raw, "content_json"),
                                 (decommented, "content_json_repaired"),
                                 (_close_unterminated_strings(raw),
                                  "content_json_repaired"),
                                 (_close_unterminated_strings(decommented),
                                  "content_json_repaired")):
            try:
                data = json.loads(candidate, strict=False)
                if isinstance(data, list):
                    data = {"quotes": data}   # bare-array shape
                return Detection.model_validate(data), label
            except Exception:
                continue  # try the repaired form, then any later payload

    # Last resort: per-entry recovery. Everything above parses the response as ONE
    # document, so a single corrupted entry discards every good entry around it.
    # Measured 26 July 2026 on four windowed responses the whole-document paths
    # rejected outright: 106 of 108 entries were individually valid — one glitched
    # line (`de: "Everybody...` for `{"quote": "Everybody...`) cost the other 28 in
    # its window. Each entry is validated against the SAME schema, so this is more
    # forgiving about STRUCTURE, never about content, and every surviving candidate
    # still faces the locator. Runs only after the whole-document paths fail, so
    # any article they already accept is untouched — development results cannot
    # move. `stats` (optional) receives the recovered/dropped counts for the sidecar.
    entries, dropped = [], 0
    for block in _ENTRY_RE.finditer(content):
        item = None
        # The `// …` annotation sits INSIDE the entry's braces, so an entry the
        # model commented is invalid on its own too. Strip comments here as well or
        # per-entry recovery drops exactly the entries the model was least sure of.
        for text in (block.group(0), _strip_json_line_comments(block.group(0))):
            try:
                item = json.loads(text, strict=False)
                break
            except Exception:
                continue
        if item is None:
            dropped += 1
            continue
        if isinstance(item, dict) and str(item.get("quote", "")).strip():
            entries.append(item)
        else:
            dropped += 1
    if entries:
        try:
            detection = Detection.model_validate({"quotes": entries})
            if stats is not None:
                stats.update(recovered=len(entries), dropped=dropped)
            return detection, "content_json_per_entry"
        except Exception:
            pass

    if not tried and not entries:
        raise ValueError("no JSON object in response content")
    raise ValueError("no content JSON validated against the schema, "
                     "even after repair")


def detect_ollama(model, body, Detection, num_ctx=98304, num_predict=24576):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Article body follows.\n\n" + body}]
    base_payload = {
        "model": model,
        "stream": False,
        # Structured extraction: reasoning tokens on a whole article can exhaust the
        # response budget and leave the tool call unmade, so thinking is disabled.
        "think": False,
        "tools": [TOOL_DEFINITION],
        "tool_choice": "required",
        # num_predict caps a degenerate/repetition-loop generation at a bounded
        # wall-clock cost instead of running unbounded up to num_ctx (observed on
        # the Jiménez Heffernan dev article: generation exceeded 20 minutes with no
        # output token cap set, versus ~20-50s on Mentz/Eve, and was killed as
        # stuck rather than genuinely progressing).
        #
        # Raised 8192 -> 24576 on 26 July 2026. "Comfortably covers the densest
        # dev article" was true of development and false of test: the densest dev
        # article (Smith) spends 0.399 output tokens per word, which projects the
        # longest TEST article (Thomas, 20,267 words) to ~8,089 tokens — 99% of the
        # old 8192 cap. A verification run on a 23,593-word out-of-sample article
        # hit the cap exactly (done_reason "length", 8192/8192, 39k chars of
        # truncated JSON) and the article failed outright. The input side was never
        # the problem there: 39,977 prompt tokens fit the raised num_ctx with room
        # to spare. num_ctx goes to 98304 so prompt + prediction still fit.
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    runinfo = {"attempts": []}

    def attempt(msgs, tools_enabled=True):
        payload = {**base_payload, "messages": msgs}
        if not tools_enabled:
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
        data = _ollama_chat(host, payload)
        message = data.get("message", {})
        record = {
            "done_reason": data.get("done_reason"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "content_chars": len(message.get("content") or ""),
            "n_tool_calls": len(message.get("tool_calls") or []),
            "tools_enabled": tools_enabled,
            "accepted_via": None,
        }
        # Ollama TRUNCATES a prompt longer than num_ctx silently and still returns
        # well-formed output, so a truncated run is indistinguishable from a real
        # one in the candidates file. At the previous default (num_ctx 32768, input
        # budget 24,576) the longest test article -- Thomas, 20,267 words, ~30,700
        # tokens at the 1.51 tokens/word measured from these very runs -- would have
        # overflowed by a quarter, destroying recall on the largest article of the
        # single permitted frozen run. num_ctx is now 65536; this check makes any
        # remaining overflow loud instead of silent.
        # Output-side truncation. done_reason "length" means generation stopped
        # because it hit num_predict, not because the model finished — the JSON is
        # cut mid-structure and salvage cannot recover it. Distinct from the input
        # check below, and the failure A1 actually found: input fit fine.
        if record.get("done_reason") == "length":
            record["output_truncated"] = True
            print(f"WARNING: generation stopped at the num_predict cap "
                  f"({num_predict} tokens) — the response is truncated mid-output "
                  f"and cannot be parsed. Raise --num-predict for this article.",
                  file=sys.stderr)

        prompt_tokens = record["prompt_eval_count"]
        budget = num_ctx - num_predict
        if prompt_tokens and prompt_tokens >= budget:
            record["prompt_truncation_suspected"] = True
            print(f"WARNING: prompt_eval_count {prompt_tokens} reached the input "
                  f"budget {budget} (num_ctx {num_ctx} - num_predict {num_predict}). "
                  f"Ollama truncates silently; this run may be INVALID.",
                  file=sys.stderr)
        detection = None
        try:
            detection = parse_tool_call(data, Detection)
            record["accepted_via"] = "tool_call"
        except Exception as tool_error:
            record["tool_call_error"] = str(tool_error)[:300]
            try:
                detection, label = salvage_detection(
                    message.get("content") or "", Detection)
                record["accepted_via"] = label
                print(f"note: no usable {TOOL_NAME} tool call; accepted via "
                      f"content salvage ({label})", file=sys.stderr)
            except Exception as salvage_error:
                record["salvage_error"] = str(salvage_error)[:300]
        # A generation stopped by the num_predict cap is severed mid-structure, so
        # whatever parsed out of it is a PREFIX of the model's answer, not the
        # answer. Salvage is good enough to extract entries from such a prefix
        # (observed on Smith: 119,032 truncated characters yielded 43 valid
        # entries), which meant a truncated run was being accepted and scored as
        # though complete. Reject it instead and let the retry stand: accepting a
        # prefix makes an article's score depend on how far generation happened to
        # get before the cap, which is not a property of the model's competence.
        # Marked whenever the cap was hit, INDEPENDENTLY of whether salvage found
        # anything: a truncated attempt must route to the truncation retry either
        # way. Gating this on a successful salvage would leave the no-salvage case
        # falling through to the reminder retry, which appends the assistant turn —
        # i.e. the entire runaway — to the next prompt.
        if record.get("output_truncated"):
            record["rejected_truncated"] = True
            if detection is not None:
                record["rejected_accepted_via"] = record["accepted_via"]
                record["accepted_via"] = None
                detection = None
                print("note: discarding a TRUNCATED generation that salvage could "
                      "otherwise parse — the answer is a prefix, not a result",
                      file=sys.stderr)
        if tool_call_swallowed(record):
            record["tool_call_swallow_suspected"] = True
        runinfo["attempts"].append(record)
        # Keep the raw generation for any attempt that was not a clean
        # whole-document accept. Without it a truncated or salvaged run leaves no
        # evidence of WHAT the model actually emitted, which is precisely the
        # question when a generation runs away (observed on Smith: 119,032
        # characters yielding 43 entries). Held under a private key and split out
        # to its own file by the CLI so the reported sidecar stays small.
        if record["accepted_via"] != "content_json" or record.get("output_truncated"):
            runinfo.setdefault("_raw", []).append({
                "attempt": len(runinfo["attempts"]),
                "accepted_via": record["accepted_via"],
                "content": message.get("content") or "",
            })
        return data, detection

    data, detection = attempt(messages)
    if detection is None:
        # Predeclared malformed-output policy: exactly one retry, counted via
        # these log lines; a second failure is recorded and raised. Which retry
        # is warranted depends on how the first attempt failed.
        if tool_call_swallowed(runinfo["attempts"][-1]):
            # Ollama swallowed a native tool call server-side (see the note above
            # TOOL_RETRY_REMINDER). The reminder would push the model back into the
            # same broken channel, so re-ask without tools instead: that routes the
            # answer down the content path, which salvage already handles.
            record = runinfo["attempts"][-1]
            print(f"note: no content and no tool call despite "
                  f"{record.get('eval_count')} generated tokens — Ollama appears to "
                  f"have discarded a native tool call server-side; retrying once "
                  f"with tools disabled", file=sys.stderr)
            runinfo["tool_call_swallow_recovery"] = True
            data, detection = attempt(messages, tools_enabled=False)
        elif runinfo["attempts"][-1].get("rejected_truncated"):
            # A runaway is stochastic, not a format misunderstanding: the same
            # article completes normally on most attempts. Retry the SAME request
            # fresh. The reminder retry would be actively harmful here — it appends
            # the assistant turn, which on a runaway is the entire truncated
            # generation (119,032 characters when this was found), inflating the
            # prompt for no benefit. num_predict is deliberately NOT raised: that
            # would retune frozen decoding parameters in response to one failure.
            print("note: retrying once after a truncated generation "
                  "(same request, same settings)", file=sys.stderr)
            runinfo["truncation_retry"] = True
            data, detection = attempt(messages)
        else:
            print("note: model did not call extract_quotes; sending one reminder retry",
                  file=sys.stderr)
            # Only carry the assistant turn if it actually said something. Appending an
            # empty message leaves the model unable to see its own prior answer while
            # being told to correct it.
            prior = data.get("message", {})
            retry_messages = list(messages)
            if (prior.get("content") or prior.get("tool_calls")):
                retry_messages.append(prior)
            retry_messages.append({"role": "user", "content": TOOL_RETRY_REMINDER})
            data, detection = attempt(retry_messages)
        if detection is None:
            debug = Path("evaluation/candidates") / "ollama_last_failure.json"
            debug.parent.mkdir(parents=True, exist_ok=True)
            debug.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
            print(f"no usable tool call or salvageable content after retry; "
                  f"raw response saved to {debug}", file=sys.stderr)
            runinfo["failed"] = True
            return None, runinfo
    return [q.model_dump() for q in detection.quotes], runinfo


def candidate_rows(quotes, body, key, stem, tag):
    rows = []
    seen_quote_occurrences = {}
    for i, q in enumerate(quotes, 1):
        normalized = normalize_query(q["quote"])
        occurrence_index = seen_quote_occurrences.get(normalized, 0)
        start, end, occurrences = find_quote_offsets(
            body, q["quote"], occurrence_index=occurrence_index
        )
        seen_quote_occurrences[normalized] = occurrence_index + 1
        rows.append({
            "id": f"{key}-{i:02d}", "article": stem, "model_tag": tag,
            "quote": q["quote"], "detector_guess": q["detector_guess"],
            "attributed_work": q["attributed_work"],
            "attributed_page": q["attributed_page"], "confidence": q["confidence"],
            "article_start_char": start, "article_end_char": end,
            "article_offset_occurrences": occurrences,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("articles", nargs="+")
    ap.add_argument("--backend", choices=["anthropic", "ollama"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=str(CANDIDATES_DIR))
    ap.add_argument("--num-ctx", type=int, default=98304, help="Ollama context window")
    ap.add_argument("--num-predict", type=int, default=24576,
                    help="Ollama output token cap (the truncation warning names this flag)")
    args = ap.parse_args()

    Detection = build_schema()
    if args.backend == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            sys.exit("`pip install anthropic` and set ANTHROPIC_API_KEY first.")
        # An unset ANTHROPIC_API_KEY does not mean there are no credentials. The
        # SDK resolves, in order: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, an
        # OAuth profile stored by `ant auth login`, workload-identity env vars,
        # then the default profile on disk. Rejecting the run on the env vars
        # alone would refuse a perfectly good profile, so probe the resolved
        # client instead and only bail when nothing resolves.
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            try:
                import anthropic
                anthropic.Anthropic()
            except Exception:
                sys.exit(
                    "No Anthropic credentials resolved. Either export "
                    "ANTHROPIC_API_KEY, or run `ant auth login` to store a "
                    "profile the SDK picks up automatically.")

    tag = f"{args.backend}_{args.model}".replace(":", "-").replace("/", "-")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for art in args.articles:
        path = Path(art)
        body, key = article_body_and_key(path)
        runinfo = None
        if args.backend == "anthropic":
            quotes, runinfo = detect_anthropic(args.model, body, Detection)
        else:
            try:
                quotes, runinfo = detect_ollama(args.model, body, Detection,
                                                args.num_ctx, args.num_predict)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                # Transport failure (server down, connection drop) — distinct
                # from the malformed-output policy: record it and continue the
                # batch rather than losing every remaining article.
                failed.append(path.name)
                print(f"{path.name}  [{tag}]: TRANSPORT FAILURE ({error}); "
                      f"continuing batch", file=sys.stderr)
                continue
        if runinfo is not None:
            # Per-article record of how structured output was obtained (tool call
            # vs. content salvage, retries, truncation, failure) — the basis for
            # the malformed-output and retry rates reported in the paper. Written
            # even when the article fails its two attempts, so a batch records
            # every article's outcome.
            raw_attempts = runinfo.pop("_raw", None)
            if raw_attempts:
                raw_path = out_dir / f"{path.stem}__{tag}.rawresponse.txt"
                raw_path.write_text(
                    "\n".join(
                        f"===== attempt {r['attempt']} "
                        f"(accepted_via={r['accepted_via']}, "
                        f"{len(r['content'])} chars) =====\n{r['content']}"
                        for r in raw_attempts),
                    encoding="utf-8")
                runinfo["raw_response_file"] = raw_path.name
            info_path = out_dir / f"{path.stem}__{tag}.runinfo.json"
            info_path.write_text(json.dumps(runinfo, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        if quotes is None:
            failed.append(path.name)
            print(f"{path.name}  [{tag}]: FAILED malformed-output policy "
                  f"(recorded; continuing batch)")
            continue
        rows = candidate_rows(quotes, body, key, path.stem, tag)
        out = out_dir / f"{path.stem}__{tag}.candidates.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name}  [{tag}]: {len(rows)} candidates -> {out}")
    if failed:
        sys.exit("malformed-output policy failures: " + ", ".join(failed))


if __name__ == "__main__":
    main()
