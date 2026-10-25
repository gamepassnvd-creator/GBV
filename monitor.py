"""CLI monitor — the autonomous crank that works when Streamlit is closed.

2026 built. Run from the project folder (DB defaults to output/gbv.db):

  add a watched page:
    python monitor.py add-watch https://www.facebook.com/somepage --label "Hate page X" --cadence-hours 12

  run every watchlist that is due once:
    python monitor.py run-due

  continuous watcher (loop forever, sleep between checks):
    python monitor.py watch --interval 600

  autonomous agent investigation of a page already in the DB:
    python monitor.py agent 2 "find coordinated anti-rights campaigns in page 2"

Cookies for scrape-type jobs live in GBV_FB_COOKIES (env) or a file argument: --cookies cookies.json.
Everything else (LM Studio) must be running, as usual — or use the bundled engine:
  GBVMonitor.exe monitor run-due --enrich          (inside the portable exe, LM Studio not needed)

Intended cron usage:
  */30 * * * * cd /path/to/fb_gbv_monitor && python monitor.py run-due >> output/monitor.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import storage


def _resolve_cookies(args) -> str:
    if args.cookies and os.path.isfile(args.cookies):
        return open(args.cookies, encoding="utf-8").read()
    return os.environ.get("GBV_FB_COOKIES", "")


def _page_url_of(watch: dict) -> str:
    import re
    return re.sub(r"[?&]fref=.*$", "", watch["url"])


def run_due(args) -> int:
    """Scrape + classify + enrich every due watchlist once. Returns 0 on success."""
    from analyzer import analyze_items, flatten
    from lmstudio import create_client
    from scraper import scrape_page
    import osint
    from agent import verifier
    import alerts

    lm = create_client(args.lm_url, "")
    lm.list_models()  # fail fast with the friendly message if the server is off
    llms = [m for m in lm.list_models() if m.get("type") == "llm"]
    if not llms:
        print("The local engine has no language models loaded yet.")
        return 2
    model = args.model or llms[0]["key"]
    print(f"[.] model: {model}")

    due = storage.due_watchlists()
    if not due:
        print("Nothing due.")
        return 0
    cookies = _resolve_cookies(args)
    failures = 0
    for w in due:
        print(f"[.] watch {w['label']} ({w['url']})")
        page_id = storage.get_or_create_page(_page_url_of(w), w["label"])
        known = storage.known_sha_keys(page_id)
        run_id = storage.start_run(page_id, notes=f"watchlist {w['id']}")
        try:
            if not cookies:
                raise RuntimeError("no cookies: set GBV_FB_COOKIES or pass --cookies file")
            data = scrape_page(
                _page_url_of(w), cookies, max_posts=w["max_posts"],
                include_comments=bool(w["include_comments"]), deep_comments=bool(w["deep_comments"]),
                headless=True, capture_screenshots=bool(w["capture_shots"]))
            items = flatten(data, pseudonymize=True)
            storage.upsert_items(page_id, run_id, items)
            storage.finish_run(run_id, "done", data["meta"]["post_count"],
                               data["meta"]["comment_count"], model=model)
            new_items = [it for it in items if f"{page_id}:{it['sha256']}" not in known]
            print(f"    collected {len(items)} items, {len(new_items)} new")

            # classify: ALL items missing a verdict (cheap because of storage)
            with storage.db() as c:
                pending = c.execute("""SELECT i.* FROM items i
                    LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
                    WHERE i.page_id=? AND v.item_key IS NULL""", (page_id,)).fetchall()
            pend = [dict(r) for r in pending]
            for r in pend:
                r.update({"id": r["item_key"], "parent_id": r["parent_key"], "sha256": r["sha256"],
                          "time_text": r["time_text"], "collected_at": r["collected_at"]})
            if pend:
                rows = analyze_items(pend, lm, model, batch_size=args.batch_size)
                storage.add_verdicts(run_id, rows, key_of=lambda r: r["id"], model=model)
                new_flagged = [r for r in rows if r["is_violation"] and r["confidence"] >= 0.7
                               and r["severity"] in ("high", "critical")]
                if new_flagged:
                    txt = "\n".join(f"- [{r['severity']}] {r['evidence'][:120]}" for r in new_flagged[:10])
                    print(alerts.send(f"fb_gbv_monitor: {len(new_flagged)} new high/critical item(s)\n{txt}"))
                    storage.audit("monitor", "alert_sent", f"{len(new_flagged)} items")
            if args.enrich:
                print(f"    osint: {len(osint.run_osint(run_id, page_id, lm, llm_model=model)['near_dups'])} near-dup pairs")
            if args.verify:
                rows_for_verify = rows if pend else []
                verifier(lm, rows_for_verify, model, run_id=run_id,
                         key_of=lambda r: r["id"], progress=lambda m, f: None)
            storage.mark_watchlist_run(w["id"], run_id, w["cadence_hours"])
        except Exception as e:
            storage.finish_run(run_id, "failed", model=model)
            storage.audit("monitor", "watchlist_failed", f"{w['url']}: {type(e).__name__}: {e}")
            print(f"    FAILED: {type(e).__name__}: {e}")
            failures += 1
    return 0 if not failures else 1


def watch_loop(args) -> None:
    interval = max(60, args.interval)
    while True:
        try:
            due = storage.due_watchlists()
        except Exception as e:
            print(f"[!] DB error: {e}")
            due = []
        if due:
            run_due(args)
        time.sleep(interval)


def agent_cmd(args) -> int:
    from lmstudio import create_client
    from agent import AgentBudget, agent_run
    lm = create_client(args.lm_url, "")
    budget = AgentBudget(max_tool_calls=args.max_steps, max_seconds=args.max_seconds,
                         max_expand_comments=args.max_expand)
    log, budget, stop = agent_run(args.page_id, args.goal, lm, args.model or "", budget,
                                  batch_size=args.batch_size,
                                  progress=lambda m, f: print(f"    {m}"))
    for i, entry in enumerate(log, 1):
        print(f"--- step {i}: {json.dumps(entry['call'], ensure_ascii=False)[:200]}")
    print(f"[.] agent stopped: {stop}")
    print(f"[.] case file:\n{storage.get_case_file(args.page_id)}")
    return 0


def add_watch(args) -> int:
    wid = storage.add_watchlist(args.url, args.label, args.cadence_hours, args.max_posts,
                                not args.no_comments, args.deep, not args.no_shots)
    print(f"watchlist #{wid} added for {args.url} (cadence {args.cadence_hours}h)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fb_gbv_monitor background monitor")
    ap.add_argument("--lm-url", default=os.environ.get("LMSTUDIO_URL", "http://localhost:1234"))
    ap.add_argument("--model", default="", help="LM Studio model key (default: first available)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--cookies", default="", help="cookies file for scrape-type jobs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run-due", help="process every due watchlist once")
    p.add_argument("--enrich", action="store_true", help="also run OSINT enrichment")
    p.add_argument("--verify", action="store_true", help="also run the adversarial verifier")
    p.set_defaults(func=run_due)

    p = sub.add_parser("watch", help="loop forever, sleep between checks (for cron-like usage)")
    p.add_argument("--interval", type=int, default=600, help="seconds between checks")
    p.add_argument("--enrich", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.set_defaults(func=watch_loop)

    p = sub.add_parser("agent", help="run the autonomous agent on a page in the DB")
    p.add_argument("page_id", type=int)
    p.add_argument("goal", help="investigation goal in one sentence")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--max-expand", type=int, default=3)
    p.set_defaults(func=agent_cmd)

    p = sub.add_parser("add-watch", help="register a watchlist target")
    p.add_argument("url")
    p.add_argument("--label", default="")
    p.add_argument("--cadence-hours", type=float, default=24.0)
    p.add_argument("--max-posts", type=int, default=50)
    p.add_argument("--no-comments", action="store_true")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--no-shots", action="store_true")
    p.set_defaults(func=add_watch)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
