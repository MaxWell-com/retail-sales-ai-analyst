"""Agent web server: chat, live calculation progress (SSE), history, Excel and PowerPoint export.

Deployment contract: port from $PORT, unauthenticated GET /health, user identity from the X-Auth-Request-*
headers set by any authenticating reverse proxy (e.g. oauth2-proxy); run locally, the user is "local".
Run: python -m app.server
"""
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import agent
from . import entities as E
from . import exports
from . import semantic as S

WEB = Path(__file__).parent / "web"
STORE = Path(os.environ.get("VT_STORE", Path(__file__).resolve().parents[1] / "data" / "app_data"))
(STORE / "answers").mkdir(parents=True, exist_ok=True)
(STORE / "conv").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ИИ-аналитик: продажи в сетях")
log = logging.getLogger("analyst.server")
jobs = {}            # id -> {"events": [...], "done": bool, "answer": dict|None, "cond": Condition}
work = queue.Queue()  # a single CPU-bound model: questions are processed one at a time
HISTORY_TURNS = 6    # past turns passed to the agent (agent.run decides how many go into the prompt)
ID_RE = re.compile(r"[0-9a-f]{12}")   # answer and conversation ids are file names in STORE; anything else is rejected


def user_of(req: Request):
    return req.headers.get("X-Auth-Request-Email") or req.headers.get("X-Auth-Request-User") or "local"


def _read_json(p):
    """Parsed JSON file or None. A corrupt file (interrupted write, manual edit) must not crash the server."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:      # ValueError covers JSONDecodeError and UnicodeDecodeError
        log.warning("cannot read %s: %s", p, e)
        return None


def _write_json(p, obj, indent=None):
    """Write via a temp file and os.replace: after a crash the disk holds the old or the new version, never a truncated one."""
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent, default=str), encoding="utf-8")
    os.replace(tmp, p)


def _conv_path(cid):
    return STORE / "conv" / f"{cid}.json"


def load_conv(cid):
    c = _read_json(_conv_path(cid))
    return c if isinstance(c, dict) and isinstance(c.get("turns"), list) else None


def save_conv(c):
    _write_json(_conv_path(c["id"]), c, indent=1)


def _load_answer(jid):
    a = _read_json(STORE / "answers" / f"{jid}.json") if ID_RE.fullmatch(str(jid)) else None
    return a if isinstance(a, dict) else None


def _memory(conv):
    """Past turns for the agent: question, slice and code-computed key facts — never the model's text. The model
    used to copy its own earlier text from history (made-up numbers, «[Вызовы: …]»), so it is left out. Answers
    saved by older versions have no slice, facts or ctx — only the question is kept for them."""
    out = []
    for t in (conv["turns"] if conv else [])[-HISTORY_TURNS:]:
        a = _load_answer(t.get("id")) or {}
        out.append({"q": t.get("question") or "", "slice": a.get("slice") or "", "headline": a.get("headline") or [],
                    "ctx": t.get("ctx") or a.get("ctx") or {}})
    return out


def _error_answer(question, e):
    """Answer on failure: a plain reason for the user, technical details in the error field (the «Как считалось» panel)."""
    return {"question": question, "answer": "Ответ не получен: на сервере произошла ошибка. Задайте вопрос еще раз; "
            "если ошибка повторится, переформулируйте его.", "slice": "", "headline": [], "results": [], "caveats": [],
            "calls": [], "next": [], "ctx": None, "error": f"{type(e).__name__}: {e}"[:1000],
            "check": {"total": 0, "checked": 0, "unmatched": [], "banned": [], "removed": [], "status": "none"}}


def _run_job(jid, job):
    def emit(ev):
        ev["t"] = round(time.time() - job["t0"], 1)
        with job["cond"]:
            job["events"].append(ev)
            job["cond"].notify_all()

    job["t0"] = time.time()
    emit({"type": "start", "text": "Начинаю расчет"})
    cid = job["cid"]
    hist = _memory(load_conv(cid))
    try:
        ans = agent.run(job["question"], hist, emit, ctx=(hist[-1]["ctx"] or None) if hist else None,
                        forced=job["forced"])
    except Exception as e:
        log.exception("agent.run: %s", job["question"])
        ans = _error_answer(job["question"], e)
    ans.update({"id": jid, "cid": cid, "user": job["user"], "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    job["answer"] = ans            # the UI gets the answer even if writing to disk fails
    _write_json(STORE / "answers" / f"{jid}.json", ans)
    conv = load_conv(cid)
    if conv is None:
        p = _conv_path(cid)
        if p.exists():             # conversation file exists but is unreadable: don't overwrite it, set it aside for inspection
            os.replace(p, p.with_suffix(".bad"))
        conv = {"id": cid, "user": job["user"], "title": job["question"][:80], "turns": []}
    conv["turns"].append({"id": jid, "question": job["question"], "ts": ans["ts"], "ctx": ans.get("ctx")})
    conv["updated"] = ans["ts"]
    save_conv(conv)


def worker():
    """Question queue. A failure on one question must not stop the thread, otherwise every later question of every
    user would hang forever at «Начинаю расчет» ("starting calculation")."""
    while True:
        jid = work.get()
        job = jobs[jid]
        try:
            _run_job(jid, job)
        except Exception as e:
            log.exception("question %s", jid)
            if not job["answer"]:
                job["answer"] = {**_error_answer(job["question"], e), "id": jid, "cid": job["cid"],
                                 "user": job["user"], "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        finally:
            with job["cond"]:
                job["done"] = True
                job["events"].append({"type": "done", "id": jid, "t": round(time.time() - job["t0"], 1)})
                job["cond"].notify_all()


threading.Thread(target=worker, daemon=True).start()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta(req: Request):
    lo, hi = S.data_bounds()
    d = E.dictionary()
    nets = S.q("""SELECT network, sum(rub) rub FROM net_sku GROUP BY 1 ORDER BY rub DESC""")
    return {"user": user_of(req), "model": agent.MODEL, "from": str(lo), "to": str(hi),
            "mat": S.mat().label(), "ytd": S.ytd().label(),
            "networks": [S.net_label(n) for n in nets.network],
            "banners": list(E.BANNER_ALIASES), "subgroups": sorted(d.subgroups),
            "districts": d.districts, "top_brands": [b for b, _, _ in sorted(d.brands, key=lambda x: -x[2])[:300]],
            "top_manufacturers": [m for m, _ in sorted(d.manufacturers, key=lambda x: -x[1])[:200]]}


@app.post("/api/ask")
async def ask(req: Request):
    body = await req.json()
    q = (body.get("question") or "").strip()
    if not q:
        raise HTTPException(400, "Пустой вопрос")
    cid = body.get("cid") or uuid.uuid4().hex[:12]
    if not ID_RE.fullmatch(str(cid)):
        raise HTTPException(400, "Неизвестный диалог")
    forced = None
    if body.get("tool"):
        # next-step button: tool and arguments are fixed, the model does not choose them (buttons never offer SQL)
        tool, args = body["tool"], body.get("args") or {}
        if not isinstance(tool, str) or tool == "sql" or not isinstance(args, dict):
            raise HTTPException(400, "Неверный следующий шаг")
        forced = {"tool": tool, "args": args}
    jid = uuid.uuid4().hex[:12]
    jobs[jid] = {"events": [], "done": False, "answer": None, "cond": threading.Condition(), "cid": cid,
                 "question": q, "user": user_of(req), "t0": time.time(), "forced": forced}
    jobs[jid]["events"].append({"type": "queued", "pos": work.qsize() + 1, "t": 0})
    work.put(jid)
    return {"id": jid, "cid": cid}


@app.get("/api/ask/{jid}/events")
def events(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404)

    def gen():
        i = 0
        while True:
            with job["cond"]:
                while i >= len(job["events"]) and not job["done"]:
                    job["cond"].wait(timeout=15)
                    if i >= len(job["events"]):
                        break
                batch = job["events"][i:]
                i += len(batch)
                done = job["done"] and i >= len(job["events"])
            for ev in batch:
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
            if not batch:
                yield ": ping\n\n"
            if done:
                return

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _answer(jid):
    a = _load_answer(jid)
    if a is None:
        job = jobs.get(jid)
        a = job and job["answer"]
    if not a:
        raise HTTPException(404)
    return a


def _view(a):
    """Answer plus the view shared with the exports: number-check badge, model text with unverified numbers marked
    (\\x01…\\x02 — app.js turns them into <mark>), technical details, plain text for copying."""
    v = exports.view(a)
    v["marked"] = exports.mark(v["text"], (a.get("check") or {}).get("unmatched"), "\x01", "\x02")
    v["copy"] = exports.plain_text(a)
    return {**a, "view": v}


@app.get("/api/ask/{jid}")
def answer(jid: str):
    return JSONResponse(_view(_answer(jid)))


@app.get("/api/history")
def history(req: Request):
    u = user_of(req)
    out = []
    for p in sorted((STORE / "conv").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
        c = _read_json(p)
        if not isinstance(c, dict) or not c.get("id"):
            continue      # an unreadable conversation must not break the history list for everyone
        if c.get("user") in (u, "local") or u == "local":
            out.append({"id": c["id"], "title": c.get("title") or "", "updated": c.get("updated"), "n": len(c.get("turns") or [])})
    return out


@app.get("/api/conv/{cid}")
def conv(cid: str):
    c = load_conv(cid) if ID_RE.fullmatch(cid) else None
    if not c:
        raise HTTPException(404)
    c["answers"] = []
    for t in c["turns"]:
        try:
            c["answers"].append(_view(_answer(t.get("id"))))
        except HTTPException:
            pass
    return c


@app.get("/api/ask/{jid}/xlsx")
def xlsx(jid: str):
    data = exports.to_xlsx(_answer(jid))
    return Response(data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename=answer_{jid}.xlsx"})


@app.get("/api/ask/{jid}/pptx")
def pptx(jid: str):
    data = exports.to_pptx(_answer(jid))
    return Response(data, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    headers={"Content-Disposition": f"attachment; filename=answer_{jid}.pptx"})


@app.get("/api/eval")
def eval_results():
    p = Path(__file__).parent.parent / "eval" / "results" / "latest.json"
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"runs": []})


@app.get("/api/info")
def info():
    from . import tools
    return JSONResponse(json.loads(json.dumps(tools.t_data_info({}).to_dict(), default=str)))


@app.get("/api/docs/{name}")
def docs(name: str):
    p = Path(__file__).parent.parent / "docs" / f"{name}.md"
    if name not in ("methodology", "data_dictionary") or not p.exists():
        raise HTTPException(404)
    return Response(p.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    E.dictionary()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8765")), log_level="warning")
