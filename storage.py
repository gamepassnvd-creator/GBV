"""SQLite dataset layer — cases → runs → items → verdicts → reviews → watchlists.

Every collection/classification/review action lands here, so the tool gains memory:
delta scraping, longitudinal reports, triage and the agent all read/write this one
file (output/gbv.db by default; override with GBV_DB).

Only the stdlib is used (sqlite3 + WAL). One short-lived connection per operation
keeps it safe from Streamlit's threading model.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

DB_PATH = os.environ.get("GBV_DB", os.path.join("output", "gbv.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages(
  id INTEGER PRIMARY KEY, url TEXT UNIQUE, label TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, page_id INT, started_at TEXT, finished_at TEXT,
  status TEXT, post_count INT DEFAULT 0, comment_count INT DEFAULT 0,
  model TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS items(
  item_key TEXT PRIMARY KEY,
  page_id INT, kind TEXT, parent_key TEXT, author TEXT, text TEXT,
  sha256 TEXT, permalink TEXT, time_text TEXT, collected_at TEXT, screenshot TEXT,
  first_seen_run INT, last_seen_run INT, times_seen INT DEFAULT 1);
CREATE INDEX IF NOT EXISTS idx_items_page ON items(page_id);
CREATE TABLE IF NOT EXISTS verdicts(
  id INTEGER PRIMARY KEY, item_key TEXT, run_id INT, model TEXT,
  is_violation INT, categories TEXT, severity TEXT, target TEXT,
  language TEXT, evidence TEXT, explanation TEXT, confidence REAL,
  analysis_error TEXT, verifier_kind TEXT DEFAULT '', verifier_summary TEXT,
  created_at TEXT);
CREATE INDEX IF NOT EXISTS idx_verdicts_item ON verdicts(item_key);
CREATE TABLE IF NOT EXISTS reviews(
  item_key TEXT PRIMARY KEY, action TEXT, reviewer TEXT, note TEXT,
  reviewed_at TEXT, verdict_id INT);
CREATE TABLE IF NOT EXISTS watchlists(
  id INTEGER PRIMARY KEY, url TEXT UNIQUE, label TEXT, cadence_hours REAL DEFAULT 24,
  max_posts INT DEFAULT 50, include_comments INT DEFAULT 1, deep_comments INT DEFAULT 0,
  capture_shots INT DEFAULT 1, active INT DEFAULT 1,
  last_run_id INT, next_due TEXT);
CREATE TABLE IF NOT EXISTS proposals(
  id INTEGER PRIMARY KEY, created_at TEXT, type TEXT, value TEXT, why TEXT,
  status TEXT DEFAULT 'pending', decided_by TEXT, decided_at TEXT);
CREATE TABLE IF NOT EXISTS case_files(
  page_id INT PRIMARY KEY, name TEXT, body TEXT DEFAULT '', updated_at TEXT);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY, ts TEXT, actor TEXT, action TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS embeddings(
  item_key TEXT PRIMARY KEY, model TEXT, dim INT, vec BLOB);
CREATE TABLE IF NOT EXISTS osint_runs(
  run_id INT PRIMARY KEY, clusters TEXT, near_dups TEXT, entities TEXT,
  cadence TEXT, amplifiers TEXT, graphml_path TEXT, done_at TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def db(path: str = None):
    conn = sqlite3.connect(path or DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def audit(actor: str, action: str, detail: str = "", path: str = None) -> None:
    with db(path) as c:
        c.execute("INSERT INTO audit(ts, actor, action, detail) VALUES (?,?,?,?)",
                  (_now(), actor, action, detail))


# ---------------------------------------------------------------------- pages/runs
def get_or_create_page(url: str, label: str = "", path: str = None) -> int:
    url = url.strip()
    with db(path) as c:
        row = c.execute("SELECT id FROM pages WHERE url=?", (url,)).fetchone()
        if row:
            return row["id"]
        cur = c.execute("INSERT INTO pages(url, label, created_at) VALUES (?,?,?)",
                        (url, label or url, _now()))
        return cur.lastrowid


def start_run(page_id: int, page_url: str = "", notes: str = "") -> int:
    with db() as c:
        cur = c.execute(
            "INSERT INTO runs(page_id, started_at, status, notes) VALUES (?,?, 'running', ?)",
            (page_id, _now(), notes))
        return cur.lastrowid


def finish_run(run_id: int, status: str = "done", post_count: int = 0,
               comment_count: int = 0, model: str = "") -> None:
    with db() as c:
        c.execute("UPDATE runs SET finished_at=?, status=?, post_count=?, comment_count=?, model=? "
                  "WHERE id=?", (_now(), status, post_count, comment_count, model, run_id))


def list_runs(page_id: int, limit: int = 50) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT * FROM runs WHERE page_id=? ORDER BY id DESC LIMIT ?",
                         (page_id, limit)).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int) -> dict | None:
    with db() as c:
        r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(r) if r else None


def latest_run(page_id: int) -> dict | None:
    runs = list_runs(page_id, limit=1)
    return runs[0] if runs else None


# ---------------------------------------------------------------------- items
def upsert_items(page_id: int, run_id: int, items: list[dict], path: str = None) -> dict:
    """items = analyzer.flatten(...) output (adding 'id','kind','parent_id',...).

    Dedupe key = page_id + sha256 of raw text (scraper-integrity anchored).
    Returns {"added": int, "updated": int}.
    """
    stats = {"added": 0, "updated": 0}
    with db(path) as c:
        # first pass: capture post ids → keys so comments can link to their parent post
        post_key_by_id: dict[str, str] = {}
        for it in items:
            if it["kind"] == "post":
                post_key_by_id[it["id"]] = f"{page_id}:{it['sha256'] or it['id']}"
        for it in items:
            key = f"{page_id}:{it['sha256'] or it['id']}"
            parent_key = ""
            if it["kind"] == "comment":
                post_id = (it.get("parent_id") or "").split("c")[0]
                parent_key = post_key_by_id.get(post_id, "")
            row = c.execute("SELECT * FROM items WHERE item_key=?", (key,)).fetchone()
            if row:
                new = max(row["text"] or "", it["text"] or "")
                c.execute("""UPDATE items SET text=?, parent_key=?, screenshot=CASE WHEN ?<>'' THEN ? ELSE screenshot END,
                             last_seen_run=?, times_seen=times_seen+1 WHERE item_key=?""",
                          (new, parent_key, it.get("screenshot", ""), it.get("screenshot", ""), run_id, key))
                stats["updated"] += 1
            else:
                c.execute("""INSERT INTO items(item_key, page_id, kind, parent_key, author, text,
                             sha256, permalink, time_text, collected_at, screenshot,
                             first_seen_run, last_seen_run, times_seen)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                          (key, page_id, it["kind"], parent_key, it["author"], it["text"], it["sha256"],
                           it["permalink"], it.get("time_text", ""), it.get("collected_at", ""),
                           it.get("screenshot", ""), run_id, run_id, 1))
                stats["added"] += 1
    return stats


def add_verdicts(run_id: int, rows: list[dict], key_of, model: str,
                 verifier_kind: str = "") -> None:
    """rows: analyzer verdicts (must carry id/text refs); key_of(row) -> item_key."""
    with db() as c:
        for r in rows:
            c.execute("""INSERT INTO verdicts(item_key, run_id, model, is_violation, categories,
                         severity, target, language, evidence, explanation, confidence,
                         analysis_error, verifier_kind, verifier_summary, created_at)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (key_of(r), run_id, model + (f" [{verifier_kind}]" if verifier_kind else ""),
                       1 if r.get("is_violation") else 0,
                       json.dumps(r.get("categories", [])), r.get("severity", "none"),
                       r.get("target", "other"), r.get("language", "other"),
                       r.get("evidence", ""), r.get("explanation", ""),
                       float(r.get("confidence", 0) or 0), r.get("analysis_error", ""),
                       verifier_kind, r.get("verifier_summary", ""), _now()))


def current_verdicts(page_id: int, run_id: int | None = None,
                     flagged_only: bool = False, limit: int = 500) -> list[dict]:
    """Join latest items with their latest (non-verifier) verdict for triage."""
    with db() as c:
        q = """
        SELECT i.*, v.run_id AS v_run, v.is_violation AS is_violation, v.categories AS categories,
               v.severity, v.target, v.language, v.evidence, v.explanation, v.confidence,
               v.analysis_error, v.model AS verdict_model, r.action AS review_action,
               r.reviewer, r.note
        FROM items i
        LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
          AND v.id IN (SELECT MAX(id) FROM verdicts WHERE verifier_kind='' GROUP BY item_key)
        LEFT JOIN reviews r ON r.item_key=i.item_key
        WHERE i.page_id=? """
        args: list = [page_id]
        if run_id:
            q += " AND i.last_seen_run=? "
            args.append(run_id)
        if flagged_only:
            q += " AND v.is_violation=1 "
        q += " ORDER BY v.is_violation DESC, v.confidence DESC LIMIT ?"
        args.append(limit)
        out = []
        for row in c.execute(q, args):
            d = dict(row)
            try:
                d["categories"] = json.loads(d["categories"] or "[]")
            except json.JSONDecodeError:
                d["categories"] = []
            d["is_violation"] = bool(d["is_violation"])
            out.append(d)
        return out


def known_sha_keys(page_id: int) -> set[str]:
    with db() as c:
        return {r["item_key"] for r in c.execute(
            "SELECT item_key FROM items WHERE page_id=?", (page_id,))}


def set_review(item_key: str, action: str, reviewer: str = "", note: str = "",
               verdict_id: int | None = None) -> None:
    if action == "unset":
        with db() as c:
            c.execute("DELETE FROM reviews WHERE item_key=?", (item_key,))
        return
    with db() as c:
        c.execute("""INSERT INTO reviews(item_key, action, reviewer, note, reviewed_at, verdict_id)
                     VALUES (?,?,?,?,?,?)
                     ON CONFLICT(item_key) DO UPDATE SET action=excluded.action,
                       reviewer=excluded.reviewer, note=excluded.note,
                       reviewed_at=excluded.reviewed_at, verdict_id=excluded.verdict_id""",
                  (item_key, action, reviewer, note, _now(), verdict_id))


def pending_review(page_id: int, run_id: int | None = None) -> list[dict]:
    rows = [r for r in current_verdicts(page_id, run_id)
            if r.get("is_violation") and not r.get("review_action")]
    return rows


def review_stats(page_id: int) -> dict:
    with db() as c:
        n_conf = c.execute("""SELECT COUNT(*) n FROM reviews r JOIN items i
                              ON i.item_key=r.item_key WHERE i.page_id=? AND r.action='confirm'""",
                           (page_id,)).fetchone()["n"]
        n_rej = c.execute("""SELECT COUNT(*) n FROM reviews r JOIN items i
                             ON i.item_key=r.item_key WHERE i.page_id=? AND r.action='reject'""",
                          (page_id,)).fetchone()["n"]
        return {"confirmed": n_conf, "rejected": n_rej}


# ---------------------------------------------------------------------- watchlists
def add_watchlist(url: str, label: str = "", cadence_hours: float = 24,
                  max_posts: int = 50, include_comments: bool = True,
                  deep_comments: bool = False, capture_shots: bool = True,
                  active: bool = True) -> int:
    with db() as c:
        try:
            cur = c.execute("""INSERT INTO watchlists(url,label,cadence_hours,max_posts,
                               include_comments,deep_comments,capture_shots,active,next_due)
                               VALUES (?,?,?,?,?,?,?,?,?)""",
                            (url.strip(), label or url.strip(),
                             float(cadence_hours), int(max_posts), int(include_comments),
                             int(deep_comments), int(capture_shots), int(active), _now()))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            c.execute("""UPDATE watchlists SET label=?, cadence_hours=?, max_posts=?, active=1
                         WHERE url=?""", (label or url, float(cadence_hours), int(max_posts), url.strip()))
            return c.execute("SELECT id FROM watchlists WHERE url=?", (url.strip(),)).fetchone()["id"]


def list_watchlists(active_only: bool = False) -> list[dict]:
    with db() as c:
        q = "SELECT * FROM watchlists" + (" WHERE active=1" if active_only else "") + " ORDER BY id"
        return [dict(r) for r in c.execute(q)]


def due_watchlists(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    out = []
    for w in list_watchlists(active_only=True):
        if not w["next_due"]:
            out.append(w)
            continue
        try:
            due = datetime.fromisoformat(w["next_due"])
            if due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            if due <= now:
                out.append(w)
        except ValueError:
            out.append(w)
    return out


def mark_watchlist_run(watch_id: int, run_id: int, cadence_hours: float) -> None:
    nxt = (datetime.now(timezone.utc) + timedelta(hours=cadence_hours)).isoformat()
    with db() as c:
        c.execute("UPDATE watchlists SET last_run_id=?, next_due=? WHERE id=?",
                  (run_id, nxt, watch_id))


def toggle_watchlist(watch_id: int, active: bool) -> None:
    with db() as c:
        c.execute("UPDATE watchlists SET active=? WHERE id=?", (int(active), watch_id))


def delete_watchlist(watch_id: int) -> None:
    with db() as c:
        c.execute("DELETE FROM watchlists WHERE id=?", (watch_id,))


# ---------------------------------------------------------------------- agent
def add_proposal(p_type: str, value: str, why: str) -> int:
    with db() as c:
        dup = c.execute("SELECT id FROM proposals WHERE type=? AND value=? AND status='pending'",
                        (p_type, value)).fetchone()
        if dup:
            return dup["id"]
        cur = c.execute("INSERT INTO proposals(created_at, type, value, why) VALUES (?,?,?,?)",
                        (_now(), p_type, value, why))
        return cur.lastrowid


def list_proposals(status: str | None = "pending") -> list[dict]:
    with db() as c:
        q = "SELECT * FROM proposals"
        args: list = []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY id DESC LIMIT 100"
        return [dict(r) for r in c.execute(q, args)]


def decide_proposal(prop_id: int, action: str, by: str) -> None:
    if action not in ("approved", "dismissed"):
        raise ValueError("action must be approved/dismissed")
    with db() as c:
        c.execute("UPDATE proposals SET status=?, decided_by=?, decided_at=? WHERE id=?",
                  (action, by, _now(), prop_id))
        row = c.execute("SELECT type, value FROM proposals WHERE id=?", (prop_id,)).fetchone()
    audit(by, f"proposal_{action}", f"{row['type']}: {row['value']}")


def get_case_file(page_id: int, name: str = "") -> str:
    with db() as c:
        r = c.execute("SELECT body FROM case_files WHERE page_id=?", (page_id,)).fetchone()
        return r["body"] if r else ""


def write_case_file(page_id: int, body: str, name: str = "Autonomous case file") -> None:
    with db() as c:
        c.execute("""INSERT INTO case_files(page_id, name, body, updated_at) VALUES (?,?,?,?)
                     ON CONFLICT(page_id) DO UPDATE SET body=excluded.body,
                       updated_at=excluded.updated_at""", (page_id, name, body, _now()))
    audit("agent", "case_file_update", f"page {page_id}, {len(body)} chars")


# ---------------------------------------------------------------------- agent helpers
def longitudinal(page_id: int, run_id: int) -> dict:
    """What changed vs the previous run: new items, new flags, severity flips."""
    with db() as c:
        prev = c.execute("""SELECT id FROM runs WHERE page_id=? AND id<? AND status='done'
                            ORDER BY id DESC LIMIT 1""", (page_id, run_id)).fetchone()
        if not prev:
            return {"first_run": True}
        p = prev["id"]
        new_items = c.execute("SELECT COUNT(*) n FROM items WHERE page_id=? AND first_seen_run=?",
                              (page_id, run_id)).fetchone()["n"]
        q = """SELECT i.item_key, v.is_violation, v.severity FROM verdicts v JOIN items i
               ON i.item_key=v.item_key WHERE v.run_id=? AND v.verifier_kind=''"""
        cur = {r["item_key"]: dict(is_violation=r["is_violation"], severity=r["severity"])
               for r in c.execute(q, (run_id,))}
        old = {r["item_key"]: dict(is_violation=r["is_violation"], severity=r["severity"])
               for r in c.execute(q, (p,))}
        new_flags = [k for k, v in cur.items() if v["is_violation"] and not old.get(k, {}).get("is_violation")]
        dropped = [k for k, v in old.items() if v["is_violation"] and not cur.get(k, {}).get("is_violation")]
        flips = [k for k in set(cur) & set(old) if cur[k]["is_violation"] and old[k]["is_violation"]
                 and cur[k]["severity"] != old[k]["severity"]]
        return {"first_run": False, "prev_run": p, "new_items": new_items,
                "new_flags": len(new_flags), "dropped_flags": len(dropped),
                "severity_flips": len(flips)}


# ---------------------------------------------------------------------- reporting helpers
def search_corpus(query: str, page_id: int | None = None, flagged_only: bool = False,
                  kind: str | None = None, limit: int = 10) -> list[dict]:
    """LIKE search over items joined with their latest verdict (agent tool + UI)."""
    with db() as c:
        q = """
        SELECT i.item_key, i.page_id, i.kind, i.author, substr(i.text,1,300) AS excerpt,
               i.permalink, v.is_violation, v.severity, v.categories, v.explanation
        FROM items i
        LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
          AND v.id IN (SELECT MAX(id) FROM verdicts WHERE verifier_kind='' GROUP BY item_key)
        WHERE i.text LIKE ? """
        args: list = [f"%{query}%"]
        if page_id:
            q += " AND i.page_id=? "
            args.append(page_id)
        if flagged_only:
            q += " AND v.is_violation=1 "
        if kind in ("post", "comment"):
            q += " AND i.kind=? "
            args.append(kind)
        q += " ORDER BY i.times_seen DESC LIMIT ?"
        args.append(limit)
        out = []
        for r in c.execute(q, args):
            d = dict(r)
            try:
                d["categories"] = json.loads(d["categories"] or "[]")
            except json.JSONDecodeError:
                d["categories"] = []
            out.append(d)
        return out


def item_full(item_key: str) -> dict | None:
    with db() as c:
        r = c.execute("SELECT * FROM items WHERE item_key=?", (item_key,)).fetchone()
        if not r:
            return None
        d = dict(r)
        v = c.execute("""SELECT * FROM verdicts WHERE item_key=? AND verifier_kind=''
                         ORDER BY id DESC LIMIT 1""", (item_key,)).fetchone()
        d["verdict"] = dict(v) if v else None
        return d
