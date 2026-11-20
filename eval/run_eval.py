"""Eval runner: each question (or dialogue) goes through the agent, then the answer is checked strictly.

Run: python -m eval.run_eval [--ids G1,N3] [--model qwen3:...]
     python -m eval.run_eval --rescore [--ids G1,N3] — re-check the last saved run without the model
Output: eval/results/latest.json (shown on the «Проверка» page) and a timestamped run file.

Checks (the type field in questions.json):
  tool         — a call to one of the tools in any succeeded and the answer is not «Посчитать не удалось»;
  no_tools     — no calculations were made;
  fact         — a tool fact mentioning entity contains the number value in unit (and/or matches re);
                 in: "caveats" — search the caveats; tool — only that tool's results;
  num_text     — the number value in unit appears in the answer text (entity in the same sentence);
  answer, forbid — regex re must / must not match the answer text;
  args_has     — some call of tool ('*' — any tool) has the arguments has;
  args_absent  — no call has any of the argument keys or values has;
  list_len     — after «Топ-N» the answer lists at least N items, otherwise it must refer to the table;
  status       — the answer's check.status is in any (default ok);
  slice        — regex re against the «Срез» line.
entity: a string «A|B» means any of the variants, a list means all elements. Units: %, п.п., руб., шт., руб./шт.;
value is in base units (руб., шт.), signed. Tolerance is half of the last displayed digit
(«85,2 тыс. шт.» — ±50 шт.) unless tol is set. The sign comes from «−»/«+» or from «снижение/упали ... на».
Dialogue: the turns field instead of q; turns go through agent.run with the previous turn's history and ctx,
checks apply to the last turn.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "eval" / "results"

# number: «85 234», «−3,3», «1 568,3»; the sign counts only if no letter precedes it («топ-10» is not −10)
NUM_RE = re.compile(r"(?<![\w.,])([+−-]?)(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](\d+))?")
UNIT_RE = re.compile(r"\s*(?:(трлн|млрд|млн|тыс)\.?\s*)?(%|п\.\s?п\.?|руб\.?\s*(?:/|за)\s*шт|руб|₽|шт|штук)?", re.I)
# «снижение на 3,3%», «упали продажи на 15,3%» — negative without a sign before the number
NEG_RE = re.compile(r"(?:(?:сниж|сниз|паден|пада|упал|упад|сократ|сокращ|уменьш|потер)\w*(?:\s+[^\s\d]+){0,3}?\s+на|минус)\s*$", re.I)
SCALE = {"трлн": 1e12, "млрд": 1e9, "млн": 1e6, "тыс": 1e3}
YO = "ё"
# judgements that no tool fact supports (L6-14); «стабильн» and «недоступн» can be legitimate, so they are not listed
BAN_RE = re.compile(r"(?i)преми|аутсайдер|доминир|сохраня\w* лидерств|снижени\w* роста|сильн\w* бренд|собственн\w* сет")
TECH_RE = re.compile(r"\[Вызовы|<tool_call>|Traceback|\w+Error\b|Ошибка расчета")
ITEM_RE = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+\S")
TOP_RE = re.compile(r"(?i)топ[\s‑–-]?(\d+)")


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower().replace(YO, "е")).strip()


def _unit(u, k):
    u = (u or "").lower()
    if u == "%":
        return "%"
    if u.startswith("п"):
        return "п.п."
    if u.startswith(("руб", "₽")):
        return "руб./шт." if "шт" in u else "руб."
    if u.startswith("шт"):
        return "шт."
    return "~" if k != 1 else ""   # «21,46 млрд» — scale without a unit


def pairs(text):
    """Numbers in the text with units: [(signed value in base units, unit, tolerance from the displayed digit)]."""
    out = []
    for m in NUM_RE.finditer(text):
        sign, whole, frac = m.groups()
        u = UNIT_RE.match(text, m.end())
        k = SCALE.get((u.group(1) or "").lower(), 1)
        v = (int(re.sub(r"\D", "", whole)) + (int(frac) / 10 ** len(frac) if frac else 0)) * k
        if sign in ("−", "-") or (not sign and NEG_RE.search(text[max(0, m.start() - 80):m.start()])):
            v = -v
        out.append((v, _unit(u.group(2), k), 0.5 * 10 ** -len(frac or "") * k))
    return out


def has_num(text, value, unit, tol=None):
    def unit_ok(got):
        return got == unit or (got == "~" and unit in ("руб.", "шт.")) or (unit == "руб./шт." and got == "руб.")
    return any(unit_ok(u) and abs(v - value) <= (t if tol is None else tol) + 1e-9 * abs(value) for v, u, t in pairs(text))


def mentions(text, entity):
    """entity: «A|B» means any of the variants, a list means all elements."""
    if not entity:
        return True
    t = norm(text)
    return all(any(norm(a) in t for a in str(e).split("|")) for e in (entity if isinstance(entity, list) else [entity]))


def sentences(text):
    return [s for line in text.splitlines() for s in re.split(r"(?<=[.!?])\s+(?=[А-ЯA-Z«\"])", line) if s.strip()]


def _vals(v):
    """Argument value -> set of strings: «2026-09» = «2026-09..2026-09», «Нива (сеть)» = «Нива»."""
    if v in (None, "", [], {}):
        return set()
    out = set()
    for x in (v if isinstance(v, list) else [v]):
        s = norm(x).replace(" (сеть)", "")
        out.add(f"{s}..{s}" if re.fullmatch(r"\d{4}-\d{2}", s) else s)
    return out


def same(actual, want):
    if isinstance(want, dict):
        return isinstance(actual, dict) and all(same(actual.get(k), v) for k, v in want.items())
    return _vals(actual) == _vals(want)


def overlap(actual, bad):
    if isinstance(bad, dict):
        return isinstance(actual, dict) and any(overlap(actual.get(k), v) for k, v in bad.items())
    return bool(_vals(actual) & _vals(bad))


def forbidden_args(args, ch):
    """The call violates args_absent: it has a key from keys or a value from has."""
    return any(_vals(args.get(k)) for k in ch.get("keys", [])) or any(overlap(args.get(k), v) for k, v in ch.get("has", {}).items())


def list_ok(answer):
    """At least N items follow «Топ-N», or the text refers to the table. -> (ok, explanation)."""
    if re.search(r"(?i)таблиц", answer):
        return True, ""
    lines = answer.splitlines()
    for i, line in enumerate(lines):
        m = TOP_RE.search(line)
        if not m or int(m.group(1)) < 2:
            continue
        cnt = 0
        for x in lines[i + 1:]:
            if ITEM_RE.match(x):
                cnt += 1
            elif x.strip():
                break
        if 0 < cnt < int(m.group(1)):
            return False, f": «{m.group(0)}», а пунктов {cnt}"
    return True, ""


def status_of(ans):
    chk = ans.get("check") or {}
    if chk.get("status"):
        return chk["status"]
    if not chk.get("checked"):
        return "none"
    return "partial" if chk.get("unmatched") else "ok"


def slice_of(ans):
    return ans.get("slice") or next((r["slice"] for r in ans.get("results") or [] if r.get("slice")), "")


def run_checks(item, ans):
    out = []
    answer = ans.get("answer") or ""
    calls = ans.get("calls") or []
    status = status_of(ans)
    for ch in item["checks"]:
        t, name = ch["type"], ch.get("name", "")
        cs = [c for c in calls if ch.get("tool", "*") in ("*", c["name"])]
        if t == "tool":
            ok = any(c["name"] in ch["any"] and c.get("ok") for c in calls) and not answer.lstrip().startswith("Посчитать не удалось")
            called = ", ".join(c["name"] + ("" if c.get("ok") else " (ошибка)") for c in calls) or "нет"
            name = f"инструмент без ошибки: {' или '.join(ch['any'])} (вызваны: {called})"
        elif t == "no_tools":
            ok, name = not calls, "отказ без расчетов"
        elif t == "fact":
            src = "caveats" if ch.get("in") == "caveats" else "facts"
            lines = [x for r in ans.get("results") or [] if ch.get("tool") in (None, r.get("tool")) for x in r.get(src) or []]
            ok = any(mentions(x, ch.get("entity")) and (not ch.get("re") or re.search(ch["re"], x))
                     and ("value" not in ch or has_num(x, ch["value"], ch.get("unit", ""), ch.get("tol"))) for x in lines)
            name += "" if lines else " (фактов нет)"
        elif t == "num_text":
            ok = any(has_num(s, ch["value"], ch.get("unit", ""), ch.get("tol")) for s in sentences(answer) if mentions(s, ch.get("entity")))
        elif t in ("answer", "forbid"):
            hit = re.search(ch["re"], answer)
            ok = bool(hit) if t == "answer" else not hit
            name += f" (найдено «{hit.group(0)}»)" if t == "forbid" and hit else ""
        elif t == "args_has":
            opts = ch.get("any_of") or [ch["has"]]     # any_of — several valid ways (filter or focus)
            ok = any(all(same((c.get("args") or {}).get(k), v) for k, v in o.items()) for o in opts for c in cs)
            name = name or f"аргументы {ch.get('tool', '*')}: {json.dumps(opts, ensure_ascii=False)}"
        elif t == "args_absent":
            ok = not any(forbidden_args(c.get("args") or {}, ch) for c in cs)
            name = name or f"нет в аргументах {ch.get('tool', '*')}: {json.dumps(ch.get('keys') or ch.get('has'), ensure_ascii=False)}"
        elif t == "list_len":
            ok, why = list_ok(answer)
            name = (name or "список «Топ-N» полный или есть ссылка на таблицу") + why
        elif t == "status":
            ok = status in ch.get("any", ["ok"])
            name = name or f"проверка чисел: status {status}, нужен {' или '.join(ch.get('any', ['ok']))}"
        elif t == "slice":
            ok = bool(re.search(ch["re"], slice_of(ans)))
            name = name or f"срез: {ch['re']}"
        else:
            ok, name = False, f"неизвестный тип проверки {t}"
        out.append({"name": name, "ok": bool(ok)})
    return out + general_checks(item, ans, status)


def general_checks(item, ans, status):
    """Rules for every question. Only violations and the number-check result are listed."""
    answer = ans.get("answer") or ""
    out = []
    if YO in answer.lower():
        out.append({"name": "в тексте ответа нет буквы е с точками", "ok": False})
    for rx, what in ((TECH_RE, "в ответе нет вызовов текстом и технических ошибок"), (BAN_RE, "нет оценок без фактов")):
        m = rx.search(answer)
        if m:
            out.append({"name": f"{what} (найдено «{m.group(0)}»)", "ok": False})
    bad = [c["name"] for c in ans.get("calls") or [] if not c.get("ok") and c["name"] not in ("data_info", "find_entities")]
    if bad and not item.get("allow_errors"):
        out.append({"name": f"вызовы без ошибок (упали: {', '.join(bad)})", "ok": False})
    if status != "none" and not any(c["type"] == "status" for c in item["checks"]):
        chk = ans.get("check") or {}
        found = f", найдено {chk['checked'] - len(chk.get('unmatched') or [])} из {chk['checked']}" if chk.get("checked") else ""
        out.append({"name": f"проверка чисел: {status}{found}", "ok": status == "ok"})
    return out


def ask(agent, item):
    """A question or a dialogue. Returns the last turn's answer and a short record of previous turns."""
    hist, ctx, prev, ans = [], None, [], None
    for q in item.get("turns") or [item["q"]]:
        if ans is not None:
            prev.append({"q": hist[-1]["q"], "answer": ans.get("answer"), "slice": ans.get("slice", ""),
                         "tools": [c["name"] for c in ans.get("calls") or []], "ctx": ctx})
        try:
            # history format: [{q, slice, headline, ctx}]; the model's text is not added to the history
            ans = agent.run(q, history=hist, ctx=ctx) if hist else agent.run(q)
        except Exception as e:
            ans = {"answer": f"Ошибка: {type(e).__name__}: {e}", "results": [], "calls": [], "check": {}}
        hist.append({"q": q, "slice": ans.get("slice", ""), "headline": ans.get("headline") or [], "ctx": ans.get("ctx")})
        ctx = ans.get("ctx")
    return ans, prev


def record(item, ans, prev, sec):
    checks = run_checks(item, ans)
    return {"id": item["id"], "src": item["src"], "q": " → ".join(item.get("turns") or [item["q"]]), "sec": round(sec, 1),
            "tools": [c["name"] for c in ans.get("calls") or []], "calls": ans.get("calls") or [], "checks": checks,
            "pass": all(c["ok"] for c in checks), "status": status_of(ans), "answer": ans.get("answer") or "",
            "slice": ans.get("slice", ""), "headline": ans.get("headline") or [], "check": ans.get("check") or {},
            "ctx": ans.get("ctx"), "recognized": ans.get("recognized"),
            # facts and caveats are saved so --rescore can re-check the answer without the model
            "results": [{"tool": r.get("tool"), "slice": r.get("slice", ""), "facts": r.get("facts") or [],
                         "caveats": r.get("caveats") or []} for r in ans.get("results") or []],
            "prev": prev, "tokens": ans.get("tokens")}


def show(rec):
    print(f"{rec['id']}: {'OK ' if rec['pass'] else 'FAIL'} {rec['sec']:.0f}s [{rec['status']}] {rec['tools']} "
          + "; ".join(("+" if c["ok"] else "-") + c["name"] for c in rec["checks"]), flush=True)


def summary(recs):
    n = len(recs)
    print(f"Total: {sum(x['pass'] for x in recs)} of {n}, mean time {sum(x['sec'] for x in recs) / max(1, n):.0f} s; "
          f"status ok: {sum(x['status'] == 'ok' for x in recs)} of {n}")


def rescore(items, ids):
    """Apply current checks to the answers of the last saved run; the model is not called, no files are changed."""
    run = json.loads((RESULTS / "latest.json").read_text(encoding="utf-8"))["runs"][-1]
    spec = {x["id"]: x for x in items}
    print(f"Run {run['ts']}, model {run['model']}")
    recs = []
    for old in run["items"]:
        if ids and old["id"] not in ids:
            continue
        it = spec.get(old["id"])
        if not it:
            print(f"{old['id']}: not in questions.json, skipped")
            continue
        q = " → ".join(it.get("turns") or [it["q"]])
        if q != old["q"]:
            print(f"{old['id']}: question changed in the set, saved answer is for «{old['q']}»")
        rec = dict(old, checks=run_checks(it, old), status=status_of(old))
        rec["pass"] = all(c["ok"] for c in rec["checks"])
        show(rec)
        recs.append(rec)
    summary(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma-separated question ids: G1,N3")
    ap.add_argument("--model")
    ap.add_argument("--rescore", action="store_true", help="re-check the last saved run without calling the model")
    a = ap.parse_args()
    items = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    ids = set(a.ids.split(",")) if a.ids else None
    if a.rescore:
        return rescore(items, ids)
    if a.model:
        os.environ["VT_MODEL"] = a.model
    from app import agent
    items = [x for x in items if not ids or x["id"] in ids]
    run = {"ts": time.strftime("%Y-%m-%d %H:%M"), "model": agent.MODEL, "items": []}
    RESULTS.mkdir(exist_ok=True)
    for it in items:
        t = time.time()
        ans, prev = ask(agent, it)
        rec = record(it, ans, prev, time.time() - t)
        run["items"].append(rec)
        show(rec)
        (RESULTS / f"run_{run['ts'].replace(':', '').replace(' ', '_')}.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
        latest = RESULTS / "latest.json"
        hist = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else {"runs": []}
        hist["runs"] = ([r for r in hist["runs"] if r["ts"] != run["ts"]] + [run])[-10:]
        latest.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    summary(run["items"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
