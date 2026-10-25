"""OSINT enrichment — all local: embeddings, near-duplicate/template detection,
theme clustering, entity extraction, posting cadence and the amplification network.

Everything computed here runs against the local SQLite dataset (storage.py) and the
local LM Studio embedding model. No data leaves the machine.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np

import storage
from lmstudio import LMStudioClient

DUP_THRESHOLD = 0.925     # cosine similarity above which two items are near-duplicates
CLUSTER_THRESHOLD = 0.66  # leader-cluster theme threshold
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"

ENTITY_TYPES = ("person", "org", "law_or_treaty", "slur_or_insult", "place", "url")
ENTITY_PROMPT = """You will receive social media items (posts/comments) about women's-rights debates.
Extract named entities. Only names EXPLICITLY present in the text; never invent names.

Allowed types: person (individuals), org (organisations, groups, pages), law_or_treaty
(laws, conventions, articles — e.g. CEDAW, مجلة الأحوال الشخصية), slur_or_insult
(derogatory labels applied to groups, incl. Darija), place (countries, cities), url.

Return ONLY a JSON array, one object per input item, in the same order:
[{"id": "...", "entities": [{"type": "...", "name": "..."}]}]
An item without extractable names gets "entities": [].

Items:
"""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


# ------------------------------------------------------------------ embeddings
def embed_items(client: LMStudioClient, keys: list[str], texts: list[str],
                model: str = EMBED_MODEL, dim_cap: int = 512) -> dict[str, np.ndarray]:
    """Embed texts (batched), cache vectors in SQLite, return {key: unit vector}."""
    vecs: dict[str, np.ndarray] = {}
    todo = [i for i, k in enumerate(keys) if not _cached(k, model, vecs)]
    for start in range(0, len(todo), 32):
        chunk = todo[start:start + 32]
        batch = [texts[i][:1024] for i in chunk]
        for i, vec in zip(chunk, client.embed(batch, model)):
            vec = np.asarray(vec[:dim_cap], dtype=np.float32)
            n = float(np.linalg.norm(vec)) or 1.0
            vec = vec / n
            vecs[keys[i]] = vec
            with storage.db() as c:
                c.execute("""INSERT INTO embeddings(item_key, model, dim, vec) VALUES (?,?,?,?)
                             ON CONFLICT(item_key) DO UPDATE SET model=excluded.model,
                               vec=excluded.vec""",
                          (keys[i], model, vec.shape[0], vec.astype(np.float32).tobytes()))
    return vecs


def _cached(key: str, model: str, store: dict[str, np.ndarray]) -> bool:
    with storage.db() as c:
        row = c.execute("SELECT model, vec FROM embeddings WHERE item_key=?", (key,)).fetchone()
    if row and row["model"] == model:
        store[key] = np.frombuffer(row["vec"], dtype=np.float32)
        return True
    return False


# ------------------------------------------------------------------ similarity & clusters
def pair_similarities(vecs: dict[str, np.ndarray]) -> list[tuple[str, str, float]]:
    keys = sorted(vecs)
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sim = float(np.dot(vecs[keys[i]], vecs[keys[j]]))
            if sim >= CLUSTER_THRESHOLD:
                out.append((keys[i], keys[j], round(sim, 4)))
    out.sort(key=lambda t: -t[2])
    return out


def theme_clusters(vecs: dict[str, np.ndarray], texts: dict[str, str],
                   threshold: float = CLUSTER_THRESHOLD) -> list[dict]:
    """Greedy leader clustering; label = most common 3-word opening among members."""
    clusters: list[dict] = []
    for key in sorted(vecs):
        v = vecs[key]
        best, best_sim = None, threshold
        for cl in clusters:
            sim = float(np.dot(v, cl["_centroid"]))
            if sim > best_sim:
                best, best_sim = cl, sim
        if best:
            best["members"].append(key)
            n = len(best["members"])
            best["_centroid"] = (best["_centroid"] * (n - 1) + v) / n
        else:
            clusters.append({"_centroid": v.copy(), "members": [key]})
    for cl in clusters:
        cl["size"] = len(cl["members"])
        cl["label"] = _label([texts.get(m, "") for m in cl["members"]])
        cl.pop("_centroid", None)
    clusters.sort(key=lambda c: -c["size"])
    return clusters


def _label(variants: list[str]) -> str:
    openers = []
    for t in variants[:20]:
        ws = _norm(t)[:90].split()
        if len(ws) >= 2:
            openers.append(" ".join(ws[:3]))
    if not openers:
        return "(empty cluster)"
    top = Counter(openers).most_common(1)[0]
    return top[0] if top[1] > 1 else openers[0][:80]


# ------------------------------------------------------------------ cadence
def cadence_counts(items: list[dict]) -> dict:
    """Parse Facebook's coarse time_text ('2 d', 'Just now', 'Yesterday', …) into buckets."""
    order = ["Just now", "hours", "1 day", "2-6 days", "1-4 weeks", "older", "unknown"]
    counts: Counter = Counter()
    for it in items:
        tt = _norm(it.get("time_text", "")).lower()
        m = re.search(r"(\d+)\s*(m|min|minutes|j|jr?)\b", tt)
        h = re.search(r"(\d+)\s*(h|hr|heures?|س|\u0633\u0627\u0639\u0629)\b", tt)
        d = re.search(r"(\d+)\s*(d|days?|jours?)\b", tt)
        w = re.search(r"(\d+)\s*(w|weeks?|semaines?)\b", tt)
        if re.search(r"just|now|maintenant|الآن|few seconds", tt):
            counts["Just now"] += 1
        elif m:
            counts["Just now" if int(m.group(1)) <= 30 else "hours"] += 1
        elif re.search(r"yesterday|hier|البارح|البارحه", tt):
            counts["1 day"] += 1
        elif h:
            counts["hours"] += 1
        elif d:
            n = int(d.group(1))
            counts["1 day" if n == 1 else "2-6 days" if n <= 6 else
                   "1-4 weeks" if n <= 28 else "older"] += 1
        elif w:
            n = int(w.group(1))
            counts["1-4 weeks" if n <= 4 else "older"] += 1
        elif re.search(r"\d+\s*(mo|months?|mois|ans|years?|y)\b", tt):
            counts["older"] += 1
        else:
            counts["unknown"] += 1
    if not items:
        return {}
    return {b: counts.get(b, 0) for b in order if counts.get(b, 0)}


# ------------------------------------------------------------------ amplification network
def amplification_edges(page_id: int, run_id: int | None = None) -> list[dict]:
    """Edges = (comment author) → (post author), weighted by comment volume."""
    with storage.db() as c:
        q = "SELECT item_key, author FROM items WHERE page_id=? AND kind='post'"
        args: list = [page_id]
        if run_id:
            q += " AND last_seen_run=?"
            args.append(run_id)
        page_of = {r["item_key"]: r["author"] for r in c.execute(q, args)}

        q = "SELECT author, parent_key FROM items WHERE page_id=? AND kind='comment'"
        args = [page_id]
        if run_id:
            q += " AND last_seen_run=?"
            args.append(run_id)
        edges: Counter = Counter()
        for r in c.execute(q, args):
            target = page_of.get(r["parent_key"], "")
            if r["author"] and target and r["author"] != target:
                edges[(r["author"], target)] += 1
    return [{"source": s, "target": t, "weight": w} for (s, t), w in edges.most_common(200)]


def graph_to_networkx(edges: list[dict]):
    import networkx as nx
    g = nx.DiGraph()
    for e in edges:
        if g.has_edge(e["source"], e["target"]):
            g[e["source"]][e["target"]]["weight"] += e["weight"]
        else:
            g.add_edge(e["source"], e["target"], weight=e["weight"])
    return g


def network_summary(edges: list[dict], top: int = 8) -> dict:
    g = graph_to_networkx(edges) if edges else None
    if not g or not g.number_of_nodes():
        return {}
    top_in = [{"node": n, "comments_received": int(v)}
              for n, v in sorted(g.in_degree(weight="weight"), key=lambda kv: -kv[1])[:top]]
    top_out = [{"node": n, "comments_given": int(v)}
               for n, v in sorted(g.out_degree(weight="weight"), key=lambda kv: -kv[1])[:top]]
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
            "most_commented_authors": top_in, "most_active_commenters": top_out}


# ------------------------------------------------------------------ entity extraction
def extract_entities(client: LMStudioClient, model: str, items: list[dict],
                     batch: int = 4, max_chars: int = 800) -> tuple[Counter, list[dict]]:
    """LLM pass over items; returns (Counter[(type, name)], per-item rows)."""
    counts: Counter = Counter()
    rows: list[dict] = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        payload = "\n\n".join(
            '{"id": "%s", "text": %s}' % (it["item_key"], json.dumps(_norm(it["text"])[:max_chars],
                                                                    ensure_ascii=False))
            for it in chunk)
        try:
            raw = client.chat(model, payload, system_prompt=ENTITY_PROMPT, temperature=0.0)
            arr = _loads_json_array(raw)
        except Exception:
            continue
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            ikey = obj.get("id", "")
            for ent in obj.get("entities") or []:
                if not isinstance(ent, dict):
                    continue
                etype = (ent.get("type") or "").strip().lower().replace(" ", "_")
                name = _norm(ent.get("name") or "")[:80]
                if etype in ENTITY_TYPES and len(name) >= 2:
                    counts[(etype, name)] += 1
                    rows.append({"item_key": ikey, "type": etype, "name": name})
    return counts, rows


def _loads_json_array(text: str) -> list:
    m = re.search(r"\[[\s\S]*\]", text or "")
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


# ------------------------------------------------------------------ run packaging
def run_osint(run_id: int, page_id: int, lm: LMStudioClient,
              embed_model: str = EMBED_MODEL, llm_model: str | None = None) -> dict:
    """Compute the full enrichment for a completed run and store results."""
    with storage.db() as c:
        rows = c.execute("""SELECT item_key, kind, author, text, time_text FROM items
                            WHERE page_id=? AND last_seen_run=?""",
                         (page_id, run_id)).fetchall()
    items = [dict(r) for r in rows]
    keys = [it["item_key"] for it in items]
    texts = {it["item_key"]: it["text"] for it in items}
    vecs = embed_items(lm, keys, [texts[k] for k in keys], embed_model)
    dups = [(a, b, s) for a, b, s in pair_similarities(vecs) if s >= DUP_THRESHOLD]
    clusters = theme_clusters(vecs, texts)
    edges = amplification_edges(page_id, run_id)
    net = network_summary(edges)
    cad = cadence_counts(items)
    entity_counter, _rows = (Counter(), [])
    if llm_model:
        entity_counter, _rows = extract_entities(lm, llm_model, items[:120])
    graphml_path = export_graphml(run_id, page_id)
    out = {
        "run_id": run_id,
        "clusters": [{"label": c["label"], "size": c["size"], "members": c["members"]}
                     for c in clusters],
        "near_dups": [{"a": a, "b": b, "sim": s} for a, b, s in dups],
        "entities": [{"type": t, "name": n, "count": k}
                     for (t, n), k in entity_counter.most_common(40)],
        "cadence": cad,
        "amplifiers": net,
        "graphml_path": graphml_path,
    }
    with storage.db() as c:
        c.execute("""INSERT INTO osint_runs(run_id, clusters, near_dups, entities, cadence,
                     amplifiers, graphml_path, done_at) VALUES (?,?,?,?,?,?,?,?)
                     ON CONFLICT(run_id) DO UPDATE SET clusters=excluded.clusters,
                       near_dups=excluded.near_dups, entities=excluded.entities,
                       cadence=excluded.cadence, amplifiers=excluded.amplifiers,
                       done_at=excluded.done_at""",
                  (run_id, json.dumps(out["clusters"], ensure_ascii=False),
                   json.dumps(out["near_dups"]), json.dumps(out["entities"], ensure_ascii=False),
                   json.dumps(cad), json.dumps(net), graphml_path, storage._now()))
    return out


def get_osint(run_id: int) -> dict | None:
    with storage.db() as c:
        r = c.execute("SELECT * FROM osint_runs WHERE run_id=?", (run_id,)).fetchone()
        if not r:
            return None
        out = {"run_id": r["run_id"], "graphml_path": r["graphml_path"], "done_at": r["done_at"]}
        for k in ("clusters", "near_dups", "entities", "cadence", "amplifiers"):
            out[k] = json.loads(r[k] or "[]")
        out["amplifiers"] = out["amplifiers"] or {}
        return out


def export_graphml(run_id: int, page_id: int, out_dir: str = "output") -> str:
    """Write the amplification graph as GraphML (GraphML → Gephi/Cytoscape/yEd import)."""
    edges = amplification_edges(page_id, run_id)
    if not edges:
        return ""
    try:
        import networkx as nx
    except ImportError:
        return ""
    path = f"{out_dir}/run_{run_id}_graph.graphml"
    nx.write_graphml(graph_to_networkx(edges), path, named_key_ids=True)
    return path
