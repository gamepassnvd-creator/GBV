"""Bounded autonomous agent + classification verifier.

The agent is a limited plan→act→reflect loop whose tools are all READ-only analytics
of the local dataset, except two gated ones:
  - expand_comments (scrapes one permalink; requires cookies in GBV_FB_COOKIES env)
  - propose_target   (never scrapes; queues a URL for HUMAN approval in the DB)

Nothing posts, likes, follows, or messages anyone. Budgets (tool calls, wall time)
are enforced in code and cannot be overridden by the model's output.

Also here: the verifier pass — a second adversarial classification of every flagged
item plus a sample of unflagged ones; disagreements are written as separate verdict
rows (verifier_kind='verifier') and surfaced in the triage UI.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import storage
from analyzer import analyze_items  # noqa: F401 (re-exported for the UI/lmstudio users)
from analyzer import SYSTEM_PROMPT
from lmstudio import LMStudioClient

Progress = Optional[Callable[[str, float], None]]

TOOLS_DOC = """You are the autonomous analyst of a GBV-monitoring case. You have read-only
tool access to the collected dataset plus two gated actions (propose_target requires human
approval; expand_comments is rate-limited by your budget).

TOOLS (call exactly one per reply, JSON object only, no prose):
{"tool": "search_corpus", "query": "...", "flagged_only": false, "kind": null}
{"tool": "get_item", "item_key": "..."}
{"tool": "case_add", "observations": ["fact/hypothesis anchored to item_key", ...]}
{"tool": "propose_target", "url": "https://www.facebook.com/...", "why": "..."}
{"tool": "expand_comments", "item_key": "..."}
{"tool": "finish", "summary": "..."}

Rules:
- The corpus is mostly Arabic / Tunisian Darija: SEARCH IN ARABIC (e.g. سيداو، معاهدة الخراب، الأسرة).
- One silent thinking step is allowed: {"tool": "none", "thought": "..."}.
- Always anchor observations to item_key values.
- propose_target only for Facebook URLs you actually saw in collected items, never invented.
- finish quickly; your budget is limited.
"""


class AgentBudget:
    def __init__(self, max_tool_calls: int = 12, max_seconds: float = 600.0,
                 max_expand_comments: int = 3):
        self.max_tool_calls = max_tool_calls
        self.max_seconds = max_seconds
        self.max_expand_comments = max_expand_comments
        self.tool_calls = 0
        self.expands = 0
        self.stop_reason = ""

    def allow(self, tool: str) -> bool:
        if self.tool_calls >= self.max_tool_calls:
            self.stop_reason = "budget: tool calls exhausted"
            return False
        if time.monotonic() - self.t0 > self.max_seconds:
            self.stop_reason = "budget: time exhausted"
            return False
        if tool == "expand_comments" and self.expands >= self.max_expand_comments:
            self.stop_reason = "budget: comment expansions exhausted"
            return False
        return True

    def start(self):
        self.t0 = time.monotonic()


# ------------------------------------------------------------------ verifier pass
VERIFIER_PROMPT = """You are an adversarial reviewer checking an automated classification of
social media content documents (possible gender-based cyberviolence against women, incl.
Arabic/Darija). Reclassify each item independently; do not trust the previous verdict.

The original analyst works under this policy:
%s

For EACH item reply with one JSON object in this shape (and nothing else):
{"id": "...", "agree": true|false, "action": "keep"|"upgrade"|"downgrade"|"drop",
 "severity": "...", "categories": [...], "reason": "1-2 sentences"}
"drop" = the previous flag was wrongly applied (or wrongly withheld when reviewing an
unflagged item). Set agree=false and give the reason. Be equally strict about
false positives and false negatives.
"""


def verifier(client: LMStudioClient, rows: list[dict], model: str,
             run_id: int | None = None, key_of=None,
             sample_unflagged: float = 0.15, progress: Progress = None) -> list[dict]:
    """Adversarial second pass. Mutates copies; returns the reviewed rows with
    _verifier fields; also writes 'verifier' verdict rows to the DB."""
    reviewed: list[dict] = []
    flagged = [r for r in rows if r.get("is_violation")]
    calm = [r for r in rows if not r.get("is_violation")]
    k = min(len(calm), max(1, int(len(calm) * sample_unflagged))) if calm else 0
    targets = flagged + random.sample(calm, k=k)

    prompt = VERIFIER_PROMPT % SYSTEM_PROMPT
    n = len(targets)
    for i, r in enumerate(targets):
        r = dict(r)
        r["_verifier"] = ""
        payload = json.dumps({
            "id": r["id"], "text": r.get("text", "")[:2200],
            "previous": {"is_violation": r.get("is_violation"), "severity": r.get("severity"),
                         "categories": r.get("categories"), "evidence": r.get("evidence", "")},
        }, ensure_ascii=False)
        try:
            raw = client.chat(model, payload, system_prompt=prompt,
                              temperature=0.0, max_output_tokens=700)
            m = re.search(r"\{[\s\S]*\}", raw)
            v = json.loads(m.group(0)) if m else {}
        except Exception as e:
            r["_verifier"] = f"verifier error: {e}"
            reviewed.append(r)
            if progress:
                progress("verifier", (i + 1) / n)
            continue
        action = v.get("action", "")
        r["_verifier"] = f"agree={v.get('agree')} action={action} — {v.get('reason', '')}"[:400]
        r["_verifier_action"] = action
        if v.get("agree") is False:
            if isinstance(v.get("severity"), str) and v["severity"]:
                r["severity"] = v["severity"]
            if isinstance(v.get("categories"), list):
                r["categories"] = v["categories"]
            if action == "drop":
                r["is_violation"] = False
            elif action in ("upgrade", "keep_flag") and v.get("severity"):
                r["is_violation"] = True
        if run_id is not None and key_of is not None:
            storage.add_verdicts(run_id, [r], key_of=key_of, model=model,
                                 verifier_kind="verifier")
        reviewed.append(r)
        if progress:
            progress(f"verified {r['id']}", (i + 1) / n)
    return reviewed


# ------------------------------------------------------------------ agent tools
def _tool_search_corpus(args: dict, ctx: dict) -> str:
    rows = storage.search_corpus(
        args.get("query", ""), page_id=ctx["page_id"],
        flagged_only=bool(args.get("flagged_only", False)),
        kind=args.get("kind"), limit=min(int(args.get("limit", 8) or 8), 15))
    out = [{"item_key": r["item_key"], "kind": r["kind"], "author": r["author"],
            "excerpt": r["excerpt"][:240], "flagged": r["is_violation"],
            "severity": r["severity"]} for r in rows]
    return json.dumps({"matches": out}, ensure_ascii=False)


def _tool_get_item(args: dict, ctx: dict) -> str:
    it = storage.item_full(args.get("item_key", ""))
    if not it:
        return json.dumps({"error": "item_key not found"})
    v = it.get("verdict") or {}
    return json.dumps({"item_key": it["item_key"], "kind": it["kind"], "author": it["author"],
                       "text": it["text"][:1800], "permalink": it["permalink"],
                       "flagged": bool(v.get("is_violation")), "severity": v.get("severity"),
                       "categories": v.get("categories"), "explanation": v.get("explanation")},
                      ensure_ascii=False)


def _tool_case_add(args: dict, ctx: dict) -> str:
    body = storage.get_case_file(ctx["page_id"])
    lines = []
    for obs in (args.get("observations") or [])[:8]:
        if isinstance(obs, str) and obs.strip():
            lines.append(f"- [{ctx['ts']}]{(' ' + obs.strip())[:600]}")
    storage.write_case_file(ctx["page_id"], (body + ("\n" if body else "") + "\n".join(lines)).strip())
    return json.dumps({"added": len(lines), "case_file_chars": len(storage.get_case_file(ctx["page_id"]))})


def _tool_propose_target(args: dict, ctx: dict) -> str:
    url = (args.get("url") or "").strip()
    if "facebook.com" not in url:
        return json.dumps({"error": "only facebook.com URLs may be proposed"})
    pid = storage.add_proposal("target", url, (args.get("why") or "")[:500])
    return json.dumps({"proposal_id": pid, "status": "pending — awaiting human approval"})


def _tool_expand_comments(args: dict, ctx: dict) -> str:
    cookies = os.environ.get("GBV_FB_COOKIES", "")
    if not cookies:
        return json.dumps({"error": "expand_comments unavailable: set GBV_FB_COOKIES env "
                                    "var first (cookies are NOT stored in the database)"})
    it = storage.item_full(args.get("item_key", ""))
    if not it or it["kind"] != "post" or not it["permalink"]:
        return json.dumps({"error": "item not found or not a post with a permalink"})
    from scraper import scrape_page
    data = scrape_page(it["permalink"], cookies, max_posts=1, include_comments=True,
                       deep_comments=True, capture_screenshots=False)
    if not data["posts"]:
        return json.dumps({"error": "scrape returned nothing (cookies expired?)"})
    items = _flatten_and_store(ctx["page_id"], data)
    newly = _classify_new_in_db(ctx)
    return json.dumps({"collected_items": len(items), "classified": newly},
                      ensure_ascii=False)


def _flatten_and_store(page_id: int, data: dict) -> list[dict]:
    from analyzer import flatten
    import analyzer
    run_id = storage.start_run(page_id, notes="agent expand_comments")
    items = flatten(data, pseudonymize=True)
    if items:
        storage.upsert_items(page_id, run_id, items)
    storage.finish_run(run_id, "done", data["meta"]["post_count"],
                       data["meta"]["comment_count"])
    return items


def _classify_new_in_db(ctx: dict) -> int:
    """Classify DB items of this page that have no verdict yet; add verdict rows."""
    with storage.db() as c:
        rows = c.execute("""SELECT i.* FROM items i
                            LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
                            WHERE i.page_id=? AND v.item_key IS NULL""", (ctx["page_id"],)).fetchall()
    if not rows:
        return 0
    lm, model = ctx["lm"], ctx["model"]
    batch_items = []
    keys = []
    for r in rows:
        keys.append(r["item_key"])
        batch_items.append({"id": r["item_key"], "kind": r["kind"], "parent_id": r["parent_key"],
                            "author": r["author"], "text": r["text"], "permalink": r["permalink"],
                            "time_text": r["time_text"], "sha256": r["sha256"],
                            "collected_at": r["collected_at"]})
    out = analyze_items(batch_items, lm, model, batch_size=ctx.get("batch_size", 2))
    run_id = storage.start_run(ctx["page_id"], notes="agent reclassify")
    storage.add_verdicts(run_id, out, key_of=lambda r, p=ctx["page_id"]: f"{p}:{r['id']}",
                         model=model)
    storage.finish_run(run_id, "done", model=model)
    return len(out)


TOOLS = {
    "search_corpus": _tool_search_corpus,
    "get_item": _tool_get_item,
    "case_add": _tool_case_add,
    "propose_target": _tool_propose_target,
    "expand_comments": _tool_expand_comments,
}


def _parse_tool_call(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return {"tool": "finish", "summary": "no tool call parsed"}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {"tool": "finish", "summary": "unparseable"}
    except json.JSONDecodeError:
        return {"tool": "finish", "summary": "unparseable tool call"}


# ------------------------------------------------------------------ the loop
def agent_run(page_id: int, goal: str, lm: LMStudioClient, model: str,
              budget: AgentBudget | None = None, batch_size: int = 2,
              progress: Progress = None):
    """Bounded plan→act→reflect loop. Returns (log, budget). Read-only except the
    human-gated proposal queue and the case file."""
    budget = budget or AgentBudget()
    budget.start()
    log: list[dict] = []
    ctx = {"page_id": page_id, "lm": lm, "model": model,
           "batch_size": batch_size, "ts": ""}

    with storage.db() as c:
        n_items = c.execute("SELECT COUNT(*) n FROM items WHERE page_id=?", (page_id,)).fetchone()["n"]
        n_flag = c.execute("""SELECT COUNT(*) n FROM items i JOIN verdicts v ON v.item_key=i.item_key
                              WHERE i.page_id=? AND v.is_violation=1 AND v.verifier_kind=''""", (page_id,)).fetchone()["n"]
        runs = c.execute("SELECT COUNT(*) n FROM runs WHERE page_id=?", (page_id,)).fetchone()["n"]
    stats = {"items": n_items, "flagged_current": n_flag, "runs": runs}
    system = TOOLS_DOC + f"\n\nCASE FILE (yours):\n{storage.get_case_file(page_id) or '(empty)'}\n\nCORPUS STATS: {json.dumps(stats)}"

    transcript = f"GOAL: {goal}\n"
    stop = ""
    while True:
        if progress:
            progress(f"agent step {budget.tool_calls + 1}", min(1.0, budget.tool_calls / max(1, budget.max_tool_calls)))
        try:
            reply = lm.chat(model, transcript + "\nYOUR NEXT TOOL CALL (JSON only):",
                            system_prompt=system, temperature=0.0, max_output_tokens=400)
        except Exception as e:
            stop = f"model error: {e}"
            break
        call = _parse_tool_call(reply)
        tool = call.get("tool", "finish")
        log.append({"call": call, "reply": reply})
        if tool == "finish" or tool not in TOOLS:
            stop = call.get("summary") or tool
            break
        if not budget.allow(tool):
            stop = budget.stop_reason
            break
        budget.tool_calls += 1
        ctx["ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            result = TOOLS[tool](call, ctx)
        except Exception as e:
            result = json.dumps({"error": f"{type(e).__name__}: {e}"})
        if tool == "expand_comments":
            budget.expands += 1
        transcript += f"\n> {tool} {json.dumps(call.get('query') or call.get('item_key') or call.get('url') or '', ensure_ascii=False)}\n{result[:1500]}\n"
        if stop:
            break

    if not stop:
        stop = budget.stop_reason or "budget reached"
    storage.audit("agent", "agent_run", f"page {page_id}: {stop} ({budget.tool_calls} tool calls)")
    return log, budget, stop
