"""Tool result: blocks for the UI (tables, charts), facts for the model, and caveats."""
import math
import re

from . import semantic as S


def _clean(v):
    """numpy/pandas scalars -> plain Python values, NaN -> None (for JSON)."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


class ToolError(Exception):
    """Error worth showing to the model: it can fix its arguments and retry."""


class Result:
    def __init__(self, tool, title):
        self.tool = tool
        self.title = title
        self.subtitle = ""
        self.blocks = []
        self.facts = []
        self.caveats = []
        self.args = {}
        self.slice_line = ""      # «Срез: ...» — what was calculated; built by code, shown above the answer
        self.meta = {}            # resolved slice: filters, periods, measure — for conversation history and follow-up questions
        self.headline = []        # 2–4 key facts: fallback «Главное» (key points) and conversation memory
        self.next = []            # next steps: [{text, tool, args}] — buttons in the UI

    def set_slice(self, scope, period, gcur=None, gbase=None, measure="rub", cmp=None, note=None):
        """Record the calculation slice. Each tool calls this once, after filters and periods are resolved.
        Adds caveats about incomplete chain months inside the periods."""
        self.slice_line = S.slice_text(scope, period, gcur, gbase, measure, cmp, note)
        self.meta = {"filters": {k: S._as_list(v) for k, v in scope.f.items() if S._as_list(v)},
                     "exclude": {k: S._as_list(v) for k, v in scope.ex.items() if S._as_list(v)},
                     "portal": scope.portal, "kids": scope.kids, "period": period.to_str(),
                     "period_label": period.label(), "measure": measure,
                     "gcur": gcur.to_str() if gcur else None, "gbase": gbase.to_str() if gbase else None, "cmp": cmp}
        for c in S.coverage_notes(scope, [period, gcur, gbase], growth=bool(gbase)):
            self.caveat(c)

    def kpis(self, items):
        """items: [(label, value string, sub-label or None, change sign or None)]"""
        self.blocks.append({"kind": "kpis", "items": [
            {"label": l, "value": v, "sub": s, "sign": g} for l, v, s, g in items]})

    def table(self, title, columns, rows, note=None, total=None, key=None):
        """columns: [(key, label, format)], rows: list of dicts.
        Formats: text, int, num1, num2, mln, th, pct, pct2, gpct, pp, heat."""
        cols = [{"key": k, "label": l, "fmt": f} for k, l, f in columns]
        clean_rows = [{c["key"]: _clean(r.get(c["key"])) for c in cols} | ({"_cls": r["_cls"]} if r.get("_cls") else {})
                      for r in rows]
        block = {"kind": "table", "title": title, "columns": cols, "rows": clean_rows, "note": note}
        if total:
            block["total"] = {c["key"]: _clean(total.get(c["key"])) for c in cols}
        if key:
            block["id"] = key
        self.blocks.append(block)

    def chart(self, ctype, title, x, series, fmt="num1", note=None, unit=""):
        """ctype: bar (horizontal bars), column, line, waterfall.
        series: [{"name": ..., "values": [...]}]"""
        self.blocks.append({"kind": "chart", "type": ctype, "title": title, "x": [_clean(v) for v in x],
                            "series": [{"name": s["name"], "values": [_clean(v) for v in s["values"]],
                                        **({"kind": s["kind"]} if s.get("kind") else {})} for s in series],
                            "fmt": fmt, "note": note, "unit": unit})

    def text(self, md):
        self.blocks.append({"kind": "text", "md": md})

    def fact(self, s, key=False):
        """Fact for the model. Order matters: totals and conclusions before ranking rows (long lists are cut from the end).
        key=True also puts the fact into headline (fallback «Главное» and conversation memory)."""
        s = re.sub(r"(?<=\.)\.(?=\s|$)", "", s)   # «85,2 тыс. шт..» — unit abbreviation dot plus sentence-final dot
        self.facts.append(s)
        if key and s not in self.headline:
            self.headline.append(s)

    def caveat(self, s):
        if s not in self.caveats:
            self.caveats.append(s)

    def suggest(self, text, tool, args):
        """Next step: a question button that the tool is guaranteed to compute with these arguments."""
        if len(self.next) < 4 and all(x["text"] != text for x in self.next):
            self.next.append({"text": text, "tool": tool, "args": {k: _clean(v) for k, v in args.items() if v not in (None, "", [])}})

    def to_dict(self):
        return {"tool": self.tool, "title": self.title, "subtitle": self.subtitle, "blocks": self.blocks,
                "facts": self.facts, "caveats": self.caveats, "args": self.args, "slice": self.slice_line,
                "meta": self.meta, "headline": self.headline, "next": self.next}

    def facts_text(self, budget=5000):
        """Text for the model: title, slice, caveats, then whole fact lines while they fit the budget.
        If facts were cut, the model is told explicitly how many lines it cannot see, so it does not invent them."""
        head = [f"[{self.title}] {self.subtitle}".strip()]
        if self.slice_line:
            head.append(self.slice_line)
        if self.caveats:
            head.append("Оговорки: " + " ".join(self.caveats))
        out, used, shown = list(head), sum(len(x) + 1 for x in head), 0
        for f in self.facts:
            if used + len(f) + 1 > budget - 150:
                break
            out.append(f)
            used += len(f) + 1
            shown += 1
        if shown < len(self.facts):
            out.append(f"Показаны {shown} из {len(self.facts)} фактов; остальные — в таблице интерфейса, "
                       "в тексте их не перечисляй и не досчитывай.")
        return "\n".join(out)
