"""Answer export: Excel (all tables) and PowerPoint (editable charts and tables).

Also the answer view shared by the chat, the exports and copy-to-clipboard (view): slice, number-check badge,
model text with unverified numbers marked, code-computed key figures, caveats — so every channel says the same thing.
"""
import io
import re

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.util import Emu, Pt

NUMFMT = {"mln": '#,##0.0', "th": '#,##0.0', "pct": '0.0%', "pct2": '0.00%', "gpct": '+0.0%;-0.0%;0.0%',
          "pp": '+0.00;-0.00;0.00', "int": '#,##0', "num0": '#,##0', "num1": '#,##0.0', "num2": '#,##0.00', "heat": '0.0%'}
SCALE = {"mln": 1e-6, "th": 1e-3, "pp": 100}
ACCENT = "D14400"
SERIES = ["2563EB", "8B94A3", "0E9384", "8B3FD9", "D97706", "DC2626", "15803D", "0284C7", "1E3A8A", "B45309"]
LEVEL_RGB = {"ok": "14713A", "warn": "96600A", "bad": "BC2417", "": "635C55"}   # number-check badge colours, same as in app.css
MAX_ROWS, MAX_COLS = 16, 8        # table on a slide: more rows/columns become unreadable
UNV = ("[не проверено: ", "]")    # marker for an unverified number in exports and copied text
NUM_RE = re.compile(r"(?<![\w.])[−-]?\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?|(?<![\w.])[−-]?\d+(?:[.,]\d+)?")
UNIT_TAIL = re.compile(r"[ \u00a0]?(?:%|п\.\s?п\.?|(?:млрд|млн|тыс)\.?(?:[ \u00a0]?(?:руб|шт)\.?)?|руб\.?|шт\.?)")


# ——— Answer view: shared by chat, Excel, PowerPoint and copy ———

def _plural(n, one, few, many):
    n = abs(n) % 100
    return one if n % 10 == 1 and n != 11 else few if 2 <= n % 10 <= 4 and not 12 <= n <= 14 else many


def _val(s):
    return float(s.replace("−", "-").replace(" ", "").replace("\u00a0", "").replace(",", "."))


def _num(v):
    """Number from the legacy check (78.5, 12500.0) -> «78,5», «12 500»."""
    s = f"{v:,.2f}".rstrip("0").rstrip(".")
    return s.replace(",", " ").replace(".", ",").replace("-", "−")


def _unmatched(ans):
    return (ans.get("check") or {}).get("unmatched") or []


def _spans(text, unmatched):
    """Where the unverified numbers are in the text: non-overlapping [(start, end, index in unmatched)].
    The current check provides positions and the fragment; the legacy one only values (78.5), in which case
    numbers in the text with the same value are matched. A unit after the number is part of the span («78,5%»)."""
    found = []
    for k, u in enumerate(unmatched):
        if isinstance(u, dict):
            t, s, e = str(u.get("text") or "").strip(), u.get("start"), u.get("end")
            if t and isinstance(s, int) and isinstance(e, int) and text[s:e].strip() == t:
                found.append((s, e, k))
            elif t:   # no positions, or the text changed since: search for the fragment itself, not inside another number
                pat = r"[\s\u00a0\u202f]+".join(re.escape(w).replace("\\-", "[−-]").replace("−", "[−-]") for w in t.split())
                end = r"(?!\d)(?![.,]\d)" if t[-1].isdigit() else ""
                found += [(m.start(), m.end(), k) for m in re.finditer(r"(?<!\d)(?<!\d[.,])" + pat + end, text)]
        elif isinstance(u, (int, float)):
            found += [(m.start(), m.end(), k) for m in NUM_RE.finditer(text) if abs(_val(m.group(0)) - u) < 1e-6]
    out, pos = [], 0
    for s, e, k in sorted(found):
        if s >= pos:
            tail = UNIT_TAIL.match(text, e)
            pos = tail.end() if tail else e
            out.append((s, pos, k))
    return out


def mark(text, unmatched, open_=UNV[0], close=UNV[1]):
    """Mark unverified numbers in the model text (in exports: «[не проверено: 78,5%]»)."""
    out, pos = [], 0
    for s, e, _ in _spans(text, unmatched or []):
        out += [text[pos:s], open_, text[s:e], close]
        pos = e
    return "".join(out) + text[pos:]


def user_text(ans):
    """Model text for the user, with technical details split out. Older answers embedded a Python exception in the
    text («Техническая причина: …», sometimes repeated) and tool calls the model wrote as text («[Вызовы: …]»,
    never executed) — these move to the «Как считалось» panel."""
    text, tech = ans.get("answer") or "", [ans.get("error") or ""]
    if "Техническая причина:" in text:
        text, _, t = text.partition("Техническая причина:")
        text = text.rstrip()
        tech += [x.strip() for x in t.split(";")]
    fake = re.findall(r"^\s*\[Вызовы:.*$", text, re.M)
    if fake:
        text = re.sub(r"^\s*\[Вызовы:.*(?:\n|$)", "", text, flags=re.M).lstrip()
        tech.append("модель написала вызов текстом, расчета по нему не было: " + " ".join(x.strip() for x in fake))
    return text, "; ".join(dict.fromkeys(x for x in tech if x))


def check_view(ans, text=None):
    """Number-check badge: {status, level, text}. Green only for status ok; wording is not checked by code, and the
    badge says so. Answers saved by older versions (no status) used the legacy check: only numbers above 12."""
    ch, calls = ans.get("check") or {}, ans.get("calls") or []
    un = _unmatched(ans)
    if text is None:
        text = user_text(ans)[0]
    shown = {k: text[s:e] for s, e, k in reversed(_spans(text, un))}
    names = list(dict.fromkeys(shown.get(k) or (u.get("text") if isinstance(u, dict) else _num(u)) for k, u in enumerate(un)))
    lst = "; ".join(names[:6]) + ("…" if len(names) > 6 else "")
    status = ch.get("status") or ("fail" if un else "old")
    tail = "; формулировки не проверяются"
    if not calls:
        if un:
            return {"status": "fail", "level": "bad", "text": f"Расчетов не было — числа в тексте не из данных: {lst}"}
        return {"status": status, "level": "", "text": "Расчетов не было"}
    if not any(c.get("ok") for c in calls):
        return {"status": status, "level": "bad", "text": "Расчет не удался"}
    if not ch.get("status"):
        n = ch.get("checked") or 0
        return {"status": status, "level": "bad" if un else "",
                "text": f"Старая проверка: сверены только числа больше 12, подтверждены {n - len(un)} из {n}"
                        + (f"; не найдены в расчетах: {lst}" if un else "") + tail}
    if status == "none":
        return {"status": status, "level": "", "text": "Чисел с единицами в тексте нет" + tail}
    total = ch.get("total") or 0
    s = f"Проверено чисел с единицами: {(ch.get('checked', total) or 0) - len(un)} из {total}"
    if un:
        s += f"; не подтверждены расчетом: {lst}"
    rm = len(ch.get("removed") or [])
    if rm:
        s += f"; код убрал из текста модели {rm} {_plural(rm, 'фрагмент', 'фрагмента', 'фрагментов')} без опоры на расчет"
    return {"status": status, "level": {"ok": "ok", "fail": "bad"}.get(status, "warn"), "text": s + tail}


def view(ans):
    """Answer view: slice, check badge, model text (without technical details), code key figures, caveats.
    Key figures are not repeated if the model text already contains them verbatim (a «Главное» built by code)."""
    text, tech = user_text(ans)
    chk = check_view(ans, text)
    key = list(ans.get("headline") or [])
    if chk["status"] != "fail" and all(h in text for h in key):
        key = []
    sl = ans.get("slice") or next((r.get("slice") for r in ans.get("results") or [] if r.get("slice")), "")
    return {"slice": sl, "check": chk, "text": text, "tech": tech, "key": key, "caveats": ans.get("caveats") or []}


def plain_text(ans):
    """Text for the «Копировать текст» (copy) button: slice, number check, marked model text, key figures, caveats."""
    v = view(ans)
    out = [x for x in (v["slice"], v["check"]["text"]) if x] + [""]
    if v["check"]["status"] == "fail":
        out.append("Текст модели не прошел проверку чисел:")
    out += [mark(v["text"], _unmatched(ans)).replace("**", ""), ""]
    if v["key"]:
        out += ["Ключевые цифры расчета:"] + [f"- {h}" for h in v["key"]] + [""]
    if v["caveats"]:
        out += ["Ограничения:"] + [f"- {c}" for c in v["caveats"]]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ——— Excel ———

def _safe(s, n=31):
    s = re.sub(r"[\[\]:*?/\\]", " ", str(s or "Таблица")).strip()
    return s[:n] or "Таблица"


def to_xlsx(ans):
    vw = view(ans)
    wb = Workbook()
    ws = wb.active
    ws.title = "Ответ"
    ws["A1"] = ans["question"]
    ws["A1"].font = Font(bold=True, size=12)
    lines = [(vw["slice"], Font(italic=True, color="635C55")),
             (vw["check"]["text"], Font(bold=True, color=LEVEL_RGB[vw["check"]["level"]])), ("", None)]
    if vw["check"]["status"] == "fail":
        lines.append(("Текст модели не прошел проверку чисел:", Font(bold=True, color=LEVEL_RGB["bad"])))
    lines += [(x.replace("**", ""), None) for x in mark(vw["text"], _unmatched(ans)).splitlines()]
    if vw["key"]:
        lines += [("", None), ("Ключевые цифры расчета", Font(bold=True))] + [("• " + h, None) for h in vw["key"]]
    if vw["caveats"]:
        lines += [("", None), ("Ограничения", Font(bold=True))] + [("• " + c, None) for c in vw["caveats"]]
    for i, (text, font) in enumerate(lines, 2):
        if text:
            cell = ws.cell(row=i, column=1, value=text)
            if font:
                cell.font = font
    ws.column_dimensions["A"].width = 120
    names = set()
    for r in ans.get("results", []):
        for b in r["blocks"]:
            if b["kind"] != "table" or not b["rows"]:
                continue
            name = _safe(b["title"])
            base, k = name, 2
            while name in names:
                name = _safe(f"{base[:27]} {k}")
                k += 1
            names.add(name)
            sh = wb.create_sheet(name)
            sh["A1"] = b["title"]
            sh["A1"].font = Font(bold=True)
            if r.get("slice"):
                sh["A2"] = r["slice"]
                sh["A2"].font = Font(italic=True, color="635C55")
            for j, c in enumerate(b["columns"], 1):
                cell = sh.cell(row=3, column=j, value=c["label"])
                cell.font = Font(bold=True, color="635C55")
                cell.fill = PatternFill("solid", fgColor="F8F5F1")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
                sh.column_dimensions[get_column_letter(j)].width = 42 if c["fmt"] in ("text", "txt") and j <= 2 else 16
            for i, rr in enumerate(b["rows"], 4):
                for j, c in enumerate(b["columns"], 1):
                    v = rr.get(c["key"])
                    if isinstance(v, (int, float)) and c["fmt"] in SCALE:
                        v = v * SCALE[c["fmt"]]
                    cell = sh.cell(row=i, column=j, value=v)
                    if c["fmt"] in NUMFMT and isinstance(v, (int, float)):
                        cell.number_format = NUMFMT[c["fmt"]]
                    if rr.get("_cls") in ("tot", "lvl"):
                        cell.font = Font(bold=True)
            sh.freeze_panes = "B4"
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


# ——— PowerPoint ———

def _rgb(h):
    return RGBColor.from_string(h)


def _is_months(b):
    """Monthly table: the slide shows the last rows (most recent months), not the first ones."""
    c = b["columns"][0]
    return c["key"] in ("month", "period") or c["label"].strip().lower() in ("месяц", "период")


def _table(s, cols, rows):
    shape = s.shapes.add_table(len(rows) + 1, len(cols), Emu(457200), Emu(1250000), Emu(11277600), Emu(300000 * (len(rows) + 1)))
    t = shape.table
    for j, c in enumerate(cols):
        cell = t.cell(0, j)
        cell.text = c["label"]
        cell.text_frame.paragraphs[0].font.size = Pt(10)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.fill.solid()
        cell.fill.fore_color.rgb = _rgb("F8F5F1")
        cell.text_frame.paragraphs[0].font.color.rgb = _rgb("635C55")
    for i, rr in enumerate(rows, 1):
        for j, c in enumerate(cols):
            cell = t.cell(i, j)
            cell.text = _fmt(rr.get(c["key"]), c["fmt"])
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(10)
            para.font.bold = rr.get("_cls") in ("tot", "lvl")
            cell.fill.solid()
            cell.fill.fore_color.rgb = _rgb("FFFFFF")


def _table_slides(prs, layout, title, b, sub):
    """Table to slides: at most MAX_ROWS rows (the last ones for monthly tables) and MAX_COLS columns; extra columns
    go to following slides with the same first column. What did not fit is stated in the subtitle, never cut silently."""
    rows, cols = b["rows"], b["columns"]
    last = _is_months(b)
    shown = rows[-MAX_ROWS:] if last else rows[:MAX_ROWS]
    notes = [sub] if sub else []
    if len(shown) < len(rows):
        notes.append(f"показаны {'последние' if last else 'первые'} {len(shown)} из {len(rows)} строк, полностью — в Excel")
    step = MAX_COLS - 1
    parts = [cols[1:][i:i + step] for i in range(0, len(cols) - 1, step)] or [[]]
    for k, part in enumerate(parts):
        s = prs.slides.add_slide(layout)
        if len(parts) == 1:
            cn = []
        elif k == 0:
            cn = [f"показаны {len(part) + 1} из {len(cols)} колонок, остальные — на следующем слайде"]
        else:
            lo = 2 + k * step
            cn = [f"колонка {lo} из {len(cols)}" if len(part) == 1 else f"колонки {lo}–{lo + len(part) - 1} из {len(cols)}"]
        title(s, b["title"] + (" (продолжение)" if k else ""), " · ".join(notes + cn))
        _table(s, [cols[0]] + part, shown)


def to_pptx(ans):
    vw = view(ans)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
    blank = prs.slide_layouts[6]

    def title(slide, text, sub=None):
        tb = slide.shapes.add_textbox(Emu(457200), Emu(280000), Emu(11277600), Emu(600000))
        p = tb.text_frame.paragraphs[0]
        p.text = text
        p.font.size, p.font.bold = Pt(24), True
        p.font.color.rgb = _rgb("191614")
        if sub:
            p2 = tb.text_frame.add_paragraph()
            p2.text = sub
            p2.font.size = Pt(12)
            p2.font.color.rgb = _rgb("7A736B")
        bar = slide.shapes.add_shape(1, Emu(457200), Emu(230000), Emu(900000), Emu(40000))
        bar.fill.solid()
        bar.fill.fore_color.rgb = _rgb(ACCENT)
        bar.line.fill.background()

    # first slide: slice and number check, then model text (unverified numbers marked) and code key figures
    s = prs.slides.add_slide(blank)
    title(s, ans["question"][:120])
    paras = [(vw["slice"], 11, False, "635C55")] if vw["slice"] else []   # (text, font size, bold, colour)
    paras.append((vw["check"]["text"], 11, True, LEVEL_RGB[vw["check"]["level"]]))
    if vw["check"]["status"] == "fail":
        paras.append(("Текст модели не прошел проверку чисел:", 13, True, LEVEL_RGB["bad"]))
    for line in mark(vw["text"], _unmatched(ans)).splitlines():
        if line.strip():
            head = line.startswith("**")
            paras.append((re.sub(r"^\s*[-*•]\s+", "", line.replace("**", "")).strip(), 15 if head else 13, head, None))
    if vw["key"]:
        paras += [("Ключевые цифры расчета", 15, True, None)] + [(h, 13, False, None) for h in vw["key"]]
    n = sum(len(p[0]) for p in paras)
    k = 1 if n < 1400 else 0.85 if n < 2200 else 0.75    # shrink long text so it stays on the slide
    tb = s.shapes.add_textbox(Emu(457200), Emu(1250000), Emu(11277600), Emu(5300000))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (text, size, bold, color) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(round(size * k))
        p.font.bold = bold
        if color:
            p.font.color.rgb = _rgb(color)
    for r in ans.get("results", []):
        for b in r["blocks"]:
            if b["kind"] == "chart" and b["x"] and b["series"]:
                s = prs.slides.add_slide(blank)
                title(s, b["title"], r.get("subtitle"))
                cd = CategoryChartData()
                cd.categories = [str(x) for x in b["x"]]
                for ser in b["series"]:
                    cd.add_series(ser["name"], [None if v is None else float(v) for v in ser["values"]])
                kind = {"bar": XL_CHART_TYPE.BAR_CLUSTERED, "column": XL_CHART_TYPE.COLUMN_CLUSTERED,
                        "line": XL_CHART_TYPE.LINE_MARKERS, "waterfall": XL_CHART_TYPE.COLUMN_CLUSTERED}[b["type"]]
                gf = s.shapes.add_chart(kind, Emu(457200), Emu(1250000), Emu(11277600), Emu(5300000), cd)
                ch = gf.chart
                ch.has_legend = len(b["series"]) > 1
                if ch.has_legend:
                    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
                    ch.legend.include_in_layout = False
                for i, ser in enumerate(ch.series):
                    fill = SERIES[i % len(SERIES)]
                    if kind == XL_CHART_TYPE.LINE_MARKERS:
                        ser.format.line.color.rgb = _rgb(fill)
                        ser.smooth = False
                    else:
                        ser.format.fill.solid()
                        ser.format.fill.fore_color.rgb = _rgb(fill)
                if b["type"] == "bar":
                    ch.plots[0].categories  # noqa
                    ch.category_axis.reverse_order = True
                fmt = b.get("fmt")
                ch.value_axis.tick_labels.number_format = '0%' if fmt == "pct" else '#,##0.0'
                ch.value_axis.tick_labels.number_format_is_linked = False
                ch.font.size = Pt(11)
            elif b["kind"] == "table" and b["rows"]:
                _table_slides(prs, blank, title, b, r.get("subtitle"))
    if ans.get("caveats"):
        s = prs.slides.add_slide(blank)
        title(s, "Ограничения")
        tb = s.shapes.add_textbox(Emu(457200), Emu(1300000), Emu(11277600), Emu(5000000))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, c in enumerate(ans["caveats"]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = "• " + c
            p.font.size = Pt(12)
    bio = io.BytesIO()
    prs.save(bio)
    return bio.getvalue()


def _fmt(v, fmt):
    if v is None:
        return "—"
    if not isinstance(v, (int, float)):
        return str(v)
    if fmt == "mln":
        return f"{v / 1e6:,.1f}".replace(",", " ").replace(".", ",")
    if fmt == "th":
        return f"{v / 1e3:,.1f}".replace(",", " ").replace(".", ",")
    if fmt in ("pct", "heat"):
        return f"{v * 100:.1f}%".replace(".", ",")
    if fmt == "pct2":
        return f"{v * 100:.2f}%".replace(".", ",")
    if fmt == "gpct":
        return f"{v * 100:+.1f}%".replace(".", ",")
    if fmt == "pp":
        return f"{v * 100:+.2f} п.п.".replace(".", ",")
    if fmt in ("int", "num0"):
        return f"{v:,.0f}".replace(",", " ")
    if fmt == "num2":
        return f"{v:,.2f}".replace(",", " ").replace(".", ",")
    return f"{v:,.1f}".replace(",", " ").replace(".", ",")
