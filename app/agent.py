"""Agent loop: code-side recognition -> model picks a tool -> calculation -> model writes the answer -> code check.

The model is local (Ollama). Raw rows never reach the model: it only sees facts computed by code.
Code overrides the model wherever it is confident: recognised filters, period, measure and sort override the model's
arguments; every number and judgement in the text is checked against the facts, the model rewrites what is not
confirmed, and code removes whatever is still unfixed. Conversation history stores the slice and the code's key facts,
not the model's text.
"""
import json
import os
import re
import time
import urllib.request

from . import entities as E
from . import semantic as S
from . import tools as T
from . import tools2 as T2
from .result import ToolError

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("VT_MODEL", "qwen2.5:7b")
# same context size as Ollama chat's default: otherwise the model reloads and loses its prompt cache
NUM_CTX = int(os.environ.get("VT_NUM_CTX", "32768"))
MAX_STEPS = 4

TOOLS = {
    "data_info": T.t_data_info, "find_entities": T.t_find_entities, "ranking": T.t_ranking, "trend": T.t_trend,
    "profile": T.t_profile, "factors": T2.t_factors, "distribution": T2.t_distribution, "prices": T2.t_prices,
    "abc": T2.t_abc, "assortment": T2.t_assortment, "forecast": T2.t_forecast, "competitors": T2.t_competitors,
    "sql": T.t_sql,
}
TOOL_RU = {"data_info": "описание данных", "find_entities": "поиск названий", "ranking": "рейтинг", "trend": "динамика",
           "profile": "обзор", "factors": "факторный анализ", "distribution": "представленность", "prices": "цены",
           "abc": "ABC-анализ", "assortment": "ротация ассортимента", "forecast": "прогноз", "competitors": "конкуренты",
           "sql": "запрос"}
SERVICE = ("data_info", "find_entities")

# Tool descriptions are short: on CPU the prompt is read at ~40 tokens/s, so every 1000 tokens cost ~25 s of waiting.
# Filters are a single `filters` object; code puts its ready-made value into the user message.
_COMMON = {
    "filters": {"type": "object", "description": "фильтры из «Готовых аргументов»"},
    "exclude": {"type": "object", "description": "исключения из «Готовых аргументов»"},
    "period": {"type": "string", "description": "2026-01..2026-06, 2026-09, 2025, mat, ytd"},
    "base_period": {"type": "string"},
    "measure": {"type": "string", "enum": ["rub", "qty", "kg"]},
}


def _tool(name, desc, props=None, required=None, filters=True):
    p = dict(_COMMON) if filters else {}
    p.update(props or {})
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": p, "required": required or []}}}


_LV = {"type": "string", "description": "manufacturer|brand|sku|subgroup|product_type|network|banner|format|district|region"}
TOOL_SPECS = [
    _tool("data_info", "Что за данные, что умеешь, объем, сети, категории.", filters=False),
    _tool("find_entities", "Найти точное название бренда/производителя/SKU.", {"query": {"type": "string"}}, ["query"], filters=False),
    _tool("ranking", "Топ и доли за период, рост, изменение доли. sort: growth — по темпу роста, gain — по приросту, "
                     "decline — по падению, share_change — по изменению доли. by — топ внутри групп. focus — соседи объекта по рейтингу.",
          {"level": _LV, "by": _LV, "top": {"type": "integer"}, "focus": {"type": "string"},
           "sort": {"type": "string", "enum": ["sales", "growth", "gain", "decline", "share_change"]}}, ["level"]),
    _tool("trend", "Динамика по месяцам; level+top — доли топ-N; seasonality=true — сезонность.",
          {"level": _LV, "top": {"type": "integer"}, "seasonality": {"type": "boolean"}}),
    _tool("profile", "Обзор производителя/бренда: рост, доля, взрослые/детские, бренды, сети."),
    _tool("factors", "Факторный анализ изменения продаж двух периодов: почему выросли или упали."),
    _tool("distribution", "Представленность: доля точек сети с продажами объекта, потенциал.", {"level": _LV, "by": _LV}),
    _tool("prices", "Цена продажи и закупочная цена сети за штуку, наценка, цена за литр.", {"level": _LV, "by": _LV}),
    _tool("abc", "ABC-анализ SKU или брендов.", {"level": _LV}),
    _tool("assortment", "Ротация в сети: mode=delist — на вывод, introduce — на ввод.",
          {"mode": {"type": "string", "enum": ["delist", "introduce"]}, "n": {"type": "integer"},
           "min_coverage": {"type": "number"}}, ["mode"]),
    _tool("forecast", "Прогноз на месяц после конца данных.", {"target_month": {"type": "string"}}),
    _tool("competitors", "Конкуренты производителя или бренда в его сегментах."),
    _tool("sql", "Крайний случай: SELECT к таблицам net_sku, sku, store, fact.", {"query": {"type": "string"}}, ["query"], filters=False),
]

SYSTEM = """Ты — ИИ-аналитик компании {own} (производитель косметики) по продажам розничных сетей (sell-out), 01.2024–{last}, помесячно: руб., шт., кг/л, закупка сети; сети, баннеры, форматы, регионы, категории, производители, бренды, SKU, точки.
Отвечай только про эти данные. Другие темы — откажи одной фразой.
Правила:
1. Цифры — только из результатов инструментов. Сам ничего не считай: не складывай, не вычитай, не сортируй, не выводи «единственный», «все», «большинство» — бери готовые итоги и агрегаты из фактов. Числа нет в фактах — не пиши его.
2. Инструмент вызывай только через вызов инструмента, никогда не пиши вызов текстом. «Готовые аргументы» — точные значения, передай их как есть. Названия нет в справочнике — find_entities.
3. Обычно один инструмент. Ошибка — поправь аргументы; не вышло — скажи, что посчитать не удалось, без чисел.
4. «Топ по росту», «растущие», «падающие» — ranking с sort. «На уровне X», «сопоставимые с X» — ranking с focus=X.
5. Период не назван — не передавай period: код возьмет период по умолчанию ({mat} для долей, {ytd} для роста).
6. Не давай оценок, которых нет в фактах: премиум, устойчивый, стабильный, доминирует, аутсайдер, сильный бренд, собственные сети, сохраняет лидерство. Падение называй снижением, не «снижением роста». Причины — только с пометкой «гипотеза».
7. Таблицы и графики интерфейс покажет сам: не пересказывай таблицу — назови не больше 3 позиций и напиши, что полный список в таблице ниже.
Ответ по-русски, без буквы «ё», коротко: **Главное** — 2–3 пункта, прямо отвечающих на вопрос, с цифрами и периодом; **Выводы** — 0–3 пункта, сравнения чисел из фактов. Раздел «Что посмотреть дальше» не пиши. НДС не упоминай, пиши «руб.».
"""


def system_prompt():
    return SYSTEM.format(own=S.OWN, last=f"{S.last_month():%m.%Y}", mat=S.mat().label(), ytd=S.ytd().label())


def _ollama(messages, tools=None, think=True, on_delta=None, temperature=0.2):
    body = {"model": MODEL, "messages": messages, "stream": True, "think": think, "keep_alive": "60m",
            "options": {"num_ctx": NUM_CTX, "temperature": temperature}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    content, thinking, calls, stats = [], [], [], {}
    with urllib.request.urlopen(req, timeout=1200) as resp:
        for line in resp:
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("error"):
                raise RuntimeError(ev["error"])
            msg = ev.get("message") or {}
            if msg.get("thinking"):
                thinking.append(msg["thinking"])
                on_delta and on_delta("thinking", msg["thinking"])
            if msg.get("content"):
                content.append(msg["content"])
                on_delta and on_delta("content", msg["content"])
            if msg.get("tool_calls"):
                calls.extend(msg["tool_calls"])
            if ev.get("done"):
                stats = {k: ev.get(k) for k in ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration", "total_duration")}
    text = "".join(content)
    # model without a separate thinking channel: drop the reasoning up to </think>
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return {"content": text.strip(), "thinking": "".join(thinking), "tool_calls": calls, "stats": stats}


_caps = None


def model_thinks():
    global _caps
    if _caps is None:
        try:
            req = urllib.request.Request(f"{OLLAMA}/api/show", data=json.dumps({"model": MODEL}).encode(),
                                         headers={"Content-Type": "application/json"})
            _caps = json.load(urllib.request.urlopen(req, timeout=30)).get("capabilities", [])
        except Exception:
            _caps = []
    return "thinking" in _caps


# ——— Question parsing ———

# Questions about the data itself: answered by the code-computed data description
META_RE = re.compile(
    r"что (ты )?(умеешь|можешь)( делать)?\s*\??$|какие (у тебя )?данные|что за данные|откуда данные|"
    r"какие данные (у тебя )?есть|в (твоей |вашей |этой )?(баз[еуы]|панел)|какой период (данных|есть)|за какой период|"
    r"за какие месяцы|с какого месяца|по какой месяц|есть ли (у тебя )?данные|есть ли в (базе|данных)|"
    r"как (ты )?(можешь|сможешь|мог бы|умеешь) (по)?счита|можешь ли (ты )?(по)?счита|"
    r"(какие|сколько) (бренд|производител|категори|подгрупп|тип|sku|ску|сет|формат|регион|город|точ)\w*\s*"
    r"(есть|представлен|в баз|в данн|у тебя|\?|$)")
LIST_RE = re.compile(r"(какие|сколько|список|перечисл)\w*\s+(бренд|производител|марк)")
# time words: without them, a period from the model is invented (the default period is used)
TIME_RE = re.compile(r"20\d\d|квартал|\bкв\b|полугод|месяц|\bгод|недел|сезон|зим|весн|лет[оа]|осен|период|ytd|mat\b|"
                     r"январ|феврал|март|апрел|\bма[йяе]\b|июн|июл|август|сентябр|октябр|ноябр|декабр")
LEVEL_SYNONYMS = {"category": "subgroup", "subcategory": "subgroup", "group": "category_group", "type": "product_type",
                  "company": "manufacturer", "producer": "manufacturer", "vendor": "manufacturer", "chain": "network",
                  "retailer": "network", "store": "network", "item": "sku", "product_name": "sku"}
BY_RX = re.compile(r"\bпо\s+(?:\w+\s+)?(?:сет|баннер|формат|округ|регион|город|категори|подгрупп|бренд|производ|sku|ску|тип)|"
                   r"в каждой|каждого|кажд\w+ сет|разбивк|в разрезе|внутри")
SKU_RX = re.compile(r"\bsku\b|\bску\b|позици|артикул|наименован|\bтовар\w*\b")
# «по всей панели» ("across the whole panel"): lift the restriction to NORD categories (the default scope)
ALL_CATS_RE = re.compile(r"вс[яеюй]\w* панел|(?:по )?все[мх]? категори|всех категори|всего рынка|все товары|"
                         r"включая (?:бумаг|салфет|туалетн|медицин|гигиену медицин)")
COMPARE_RE = re.compile(r"на уровне|сопостав|рядом с|сосед|похож|сравнимы")
ANALYTIC_RE = re.compile(r"продаж|продал|доля|доли|рост|упал|паден|топ|рейтинг|лидер|динамик|сколько|цен[аыуе]|наценк|"
                         r"представлен|дистрибуц|точк|прогноз|конкурент|abc|авс|вывод|ввод|выручк|объем")
TOOL_HINTS = [
    (re.compile(r"прогноз"), "forecast"),
    (re.compile(r"конкурент"), "competitors"),
    (re.compile(r"\b(abc|авс)\b"), "abc"),
    (re.compile(r"на вывод|вывести|вывода из|на ввод|ввести|ввода в|ротаци"), "assortment"),
    (re.compile(r"представлен|дистрибуц|в скольких точках|сколько точек|число точек"), "distribution"),
    (re.compile(r"почему|причин|фактор|декомпоз|за счет чего|из-за чего"), "factors"),
    (re.compile(r"\bцен[аыуе]?\b|наценк|закупочн|закупк"), "prices"),
    (re.compile(r"сезонн|по месяцам|помесячн|динамик"), "trend"),
    (re.compile(r"топ|рейтинг|лидер|доля|доли|крупнейш|по росту|растущ|падающ|сопостав|на уровне"), "ranking"),
]


def _vals(v):
    return v if isinstance(v, list) else ([] if v in (None, "", []) else [v])


def _nset(v):
    return {E.norm(x).strip() for x in _vals(v)}


def _fld(found, name, default=None):
    v = getattr(found, name, default)
    return default if v is None else v


def _inherit(found, ctx):
    """A follow-up question («а в штуках?», «какие именно SKU…») inherits from the previous turn what it lacks:
    period, filters per group (product, category, chains), exclusions, measure."""
    if not ctx or not _fld(found, "followup", False):
        return {}
    out = {"filters": {}}
    fl = found.filters
    dims = set(_fld(found, "dims", set()))
    for group in (S.OBJECT_KEYS, S.CATEGORY_KEYS, S.STORE_KEYS):
        if any(fl.get(k) for k in group):
            continue
        for k in group:
            if (ctx.get("filters") or {}).get(k) and k not in dims:
                out["filters"][k] = ctx["filters"][k]
    if not found.periods and not found.period_kind and ctx.get("period"):
        out["period"] = ctx["period"]
        if ctx.get("base_period"):
            out["base_period"] = ctx["base_period"]
        out["period_label"] = ctx.get("period_label") or ctx["period"]
    if not _fld(found, "exclude", {}) and ctx.get("exclude"):
        out["exclude"] = ctx["exclude"]
    if not found.measure and ctx.get("measure") and ctx.get("measure") != "rub":
        out["measure"] = ctx["measure"]
    return out if (out["filters"] or len(out) > 1) else {}


def _inherit_note(inh):
    parts = [f"{S.FILTER_NAMES.get(k, k)}: {', '.join(map(str, _vals(v)))}" for k, v in inh.get("filters", {}).items()]
    if inh.get("period"):
        parts.append(f"период {inh['period_label']}")
    if inh.get("measure"):
        parts.append(f"мера {S.MEASURE_NAME.get(inh['measure'], inh['measure'])}")
    if inh.get("exclude"):
        parts.append("исключения прошлого вопроса")
    return "из прошлого вопроса: " + "; ".join(parts) if parts else ""


def _hint(found, inh, question=""):
    """«Готовые аргументы» (ready-made arguments) for the model: what code recognised, plus what was inherited from the previous turn."""
    flat = found.as_args()
    filters = {k: v for k, v in flat.items() if k in S.FILTERS}
    for k, v in inh.get("filters", {}).items():
        filters.setdefault(k, v)
    hint = {"filters": filters}
    exc = flat.get("exclude") or inh.get("exclude")
    if exc:
        hint["exclude"] = exc
    src = flat if flat.get("period") else inh
    for k in ("period", "base_period"):
        if src.get(k):
            hint[k] = src[k]
    if flat.get("measure") or inh.get("measure"):
        hint["measure"] = flat.get("measure") or inh["measure"]
    for k in ("sort", "focus", "level", "portal", "exclude_pl"):
        if flat.get(k) not in (None, "", False):
            hint[k] = flat[k]
    if found.kids is not None:
        hint["kids"] = found.kids
    if ALL_CATS_RE.search(question.lower().replace("ё", "е")):
        hint["all_categories"] = True
    if found.future_month:
        hint["target_month"] = S.ym(found.future_month)
    return hint


def _tool_hint(question, found):
    ql = question.lower().replace("ё", "е")
    if _fld(found, "sort"):
        return "ranking"
    for rx, name in TOOL_HINTS:
        if rx.search(ql):
            return name
    return None


# ——— Call arguments: code overrides the model ———

def _flatten(args):
    for key in ("filters", "exclude"):
        v = args.get(key)
        if isinstance(v, str):
            try:
                args[key] = json.loads(v)
            except json.JSONDecodeError:
                args[key] = {}
    filt = args.get("filters")
    if isinstance(filt, dict):
        args = {**filt, **{k: v for k, v in args.items() if k != "filters"}}
    else:
        args.pop("filters", None)
    return args


def _normalize(name, args, found, hint, question, ctx):
    """Align the model's arguments with what code recognised. Returns (args, filled, overridden)."""
    args = _flatten(dict(args or {}))
    filled, overridden = {}, {}
    if name in ("data_info", "find_entities", "sql"):
        return args, filled, overridden

    def put(k, v):
        cur = args.get(k)
        if cur in (None, "", []):
            filled[k] = v
        elif _nset(cur) != _nset(v):
            overridden[k] = [cur, v]
        else:
            return
        args[k] = v

    ql_low = question.lower().replace("ё", "е")
    # junk in filters: the model inserts objects and slice text («{'in': ['12 подгрупп…']}»)
    for k in list(args):
        if k in S.FILTERS and (isinstance(args[k], dict) or any(not isinstance(x, (str, int, float)) for x in _vals(args[k]))):
            overridden[k] = [args.pop(k), None]
    # a category the model added on its own is not a filter when the reference portal's «Косметика» category applies
    for k in S.CATEGORY_KEYS:
        if args.get(k) and k not in hint.get("filters", {}) and found.portal:
            overridden[k] = [args.pop(k), None]
    for k in ("level", "by"):
        if args.get(k) in LEVEL_SYNONYMS:
            overridden[k] = [args[k], LEVEL_SYNONYMS[args[k]]]
            args[k] = LEVEL_SYNONYMS[args[k]]
    if name in ("prices", "distribution", "abc") and args.get("level") == "sku" and not SKU_RX.search(ql_low):
        overridden["level"] = [args.pop("level"), None]     # default level is the narrowest named object
    if name == "prices" and len(_vals(hint.get("filters", {}).get("network"))) > 1 and args.get("by") != "network":
        put("by", "network")                                # «в Ниве и Граните» (two chains) — prices per chain
    if name == "trend" and args.get("seasonality") and "сезон" not in ql_low:
        overridden["seasonality"] = [args.pop("seasonality"), None]   # «доли по месяцам» (monthly shares) — a trend, not seasonality
    if name == "ranking":
        # sort and breakdown absent from the question are invented by the model: «доля NORD без Гранита» used to
        # become "share change per chain", and the overall share was missing from the calculation
        if args.get("sort") not in (None, "sales") and not hint.get("sort"):
            overridden["sort"] = [args.pop("sort"), None]
        if args.get("by") and args["by"] not in set(_fld(found, "dims", set()) or set()) and not BY_RX.search(ql_low):
            overridden["by"] = [args.pop("by"), None]
    level, by = args.get("level"), args.get("by")
    # «доля NORD в …»: rank at the named object's level (NORD's place and share), not subgroups inside NORD
    if name == "ranking" and not hint.get("level"):
        objs = [(k, v) for k, v in hint.get("filters", {}).items() if k in ("manufacturer", "brand") and len(_vals(v)) == 1]
        m = re.search(r"\bдол[яиюе]\s+(?:бренда\s+|производителя\s+|компании\s+)?(\S+)", ql_low)
        if len(objs) == 1 and m and E.norm(_vals(objs[0][1])[0]).strip().startswith(E.norm(m.group(1)).strip()[:3]) \
                and level != objs[0][0]:
            overridden["level"] = [level, objs[0][0]]
            level = args["level"] = objs[0][0]
    if name == "ranking" and hint.get("level") and hint["level"] not in (level, by):
        if level:
            overridden["level"] = [level, hint["level"]]
        else:
            filled["level"] = hint["level"]
        level = args["level"] = hint["level"]
    focus = hint.get("focus") if name == "ranking" else None
    fnorm = E.norm(focus).strip() if focus else None
    for k, v in hint.get("filters", {}).items():
        vals = _vals(v)
        if k in (level, by) and len(vals) == 1 and name == "ranking":
            # «доля Гранита среди сетей», «доля NORD среди производителей»: a single object ranked at its own level is
            # not a filter (it would get 100%) but a reference point: the ranking shows its place, share and the leaders
            if args.get(k):
                overridden[k] = [args.pop(k), None]
            if k == level and not focus:
                focus, fnorm = vals[0], E.norm(vals[0]).strip()
            continue
        if k in (level, by) and len(vals) == 1:
            # «динамика NORD» but the model chose level "manufacturer": drop the level, not the NORD filter
            # (otherwise the whole market is computed while the answer is labelled NORD)
            dim = "level" if k == level else "by"
            overridden[dim] = [args.pop(dim), None]
            level, by = args.get("level"), args.get("by")
        if fnorm and k in S.OBJECT_KEYS and fnorm in _nset(vals):
            continue   # comparison object «на уровне X» ("on par with X") is not a filter
        put(k, v)
    # excluded («кроме Гранита») and named-as-dimension («доля Гранита в продажах NORD») values are not filters
    for k, vals in (_fld(found, "exclude", {}) or {}).items():
        if args.get(k):
            rest = [x for x in _vals(args[k]) if E.norm(x).strip() not in _nset(vals)]
            if len(rest) != len(_vals(args[k])):
                overridden[k] = [args[k], rest or None]
                if rest:
                    args[k] = rest
                else:
                    args.pop(k)
    mentioned = _fld(found, "mentioned", {}) or {}
    for k in set(_fld(found, "dims", set()) or set()) | {level, by}:
        if k and args.get(k) and k not in hint.get("filters", {}) and _nset(args[k]) <= _nset(mentioned.get(k)):
            overridden[k] = [args.pop(k), None]
    # a filter the model carried over from the previous question although this is a new question
    if ctx and not _fld(found, "followup", False):
        for k, v in (ctx.get("filters") or {}).items():
            if k not in hint.get("filters", {}) and args.get(k) and _nset(args[k]) == _nset(v):
                overridden[k] = [args.pop(k), None]
    if hint.get("exclude"):
        put("exclude", hint["exclude"])
    # period: code's recognition wins; if neither the question nor the context has one, the model may not invent it
    if hint.get("period"):
        put("period", hint["period"])
        if hint.get("base_period"):
            put("base_period", hint["base_period"])
        elif args.get("base_period"):
            overridden["base_period"] = [args.pop("base_period"), None]
    elif not TIME_RE.search(question.lower()):
        for k in ("period", "base_period"):
            if args.get(k):
                overridden[k] = [args.pop(k), None]
    if hint.get("measure"):
        put("measure", hint["measure"])
    if name == "ranking" and hint.get("sort"):
        put("sort", hint["sort"])
    if name == "ranking" and not focus and level and level not in hint.get("filters", {}):
        named = _vals(mentioned.get(level))
        if len(named) == 1:
            # «доля NORD», «доля Гранита»: the named object is the ranking's reference point (its place and share always shown)
            focus = named[0]
            fnorm = E.norm(focus).strip()
    if name == "ranking":
        if focus:
            put("focus", focus)
            for k in S.OBJECT_KEYS:   # remove the comparison object from filters (otherwise the ranking holds NORD alone)
                if args.get(k) and fnorm in _nset(args[k]):
                    rest = [x for x in _vals(args[k]) if E.norm(x).strip() != fnorm]
                    overridden[k] = [args[k], rest or None]
                    if rest:
                        args[k] = rest
                    else:
                        args.pop(k)
        elif args.get("focus"):
            # model's focus: only an object (manufacturer/brand/SKU) code found in the question, and only for comparisons
            known = set()
            for src in (found.filters, _fld(found, "mentioned", {})):
                for k in S.OBJECT_KEYS:
                    known |= _nset(src.get(k))
            if not (COMPARE_RE.search(question.lower()) and E.norm(args["focus"]).strip() in known):
                overridden["focus"] = [args.pop("focus"), None]
    else:
        args.pop("focus", None)
    if found.portal and not args.get("portal"):
        put("portal", True)
    if hint.get("exclude_pl") and not args.get("exclude_pl"):
        put("exclude_pl", True)
    if hint.get("all_categories") and not args.get("all_categories"):
        put("all_categories", True)
    if found.kids is not None and args.get("kids") != found.kids:
        put("kids", found.kids)
    if found.future_month and name == "forecast":
        put("target_month", S.ym(found.future_month))
    for k in list(filled):
        if filled[k] in (None, "", []):
            filled.pop(k)
    return args, filled, overridden


# ——— Tool calls written as text ———

def _scan_calls(s):
    """name({...}) inside a string: tool name and arguments with balanced parentheses."""
    out = []
    for m in re.finditer(r"(\w+)\s*\(", s):
        name = m.group(1)
        if name not in TOOLS:
            continue
        i, depth, j = m.end(), 1, m.end()
        while j < len(s) and depth:
            depth += {"(": 1, ")": -1}.get(s[j], 0)
            j += 1
        raw = s[i:j - 1].strip()
        args = {}
        if raw:
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                continue
        out.append({"function": {"name": name, "arguments": args if isinstance(args, dict) else {}}})
    return out


def _json_call(s):
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return None
    if isinstance(d, dict) and d.get("name") in TOOLS:
        a = d.get("arguments") or d.get("parameters") or {}
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except json.JSONDecodeError:
                a = {}
        return {"function": {"name": d["name"], "arguments": a if isinstance(a, dict) else {}}}
    return None


def _text_calls(content):
    """Calls the model wrote as text instead of a tool call: <tool_call>{...}</tool_call>,
    {"name": ..., "arguments": ...}, «[Вызовы: ranking({...})]», «ranking({...})». Returns (calls, text without them)."""
    calls = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.S):
        c = _json_call(m.group(1))
        if c:
            calls.append(c)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.S)
    keep = []
    for line in text.splitlines():
        s = line.strip().strip("`").strip()
        if s.startswith("{") and '"name"' in s:
            c = _json_call(s)
            if c:
                calls.append(c)
                continue
        if re.match(r"^\[?\s*Вызовы\s*:", s):
            calls += _scan_calls(s)
            continue
        m = re.match(r"^(\w+)\s*\((.*)\)\s*;?$", s)
        if m and m.group(1) in TOOLS:
            calls += _scan_calls(s)
            continue
        keep.append(line)
    return calls, "\n".join(keep).strip()


def _clean_text(text):
    _, text = _text_calls(text or "")
    # «Что посмотреть дальше» (what to look at next) comes from code buttons; the model suggests such questions at random
    text = re.split(r"\n\s*(?:\*\*|#+\s*)?Что посмотреть дальше", text)[0]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ——— Number and wording check ———

UNITS = [  # (pattern after the number, kind, multiplier)
    (r"\s*%", "pct", 1), (r"\s*п\.\s?п\.?", "pp", 1),
    (r"\s*млрд\.?\s*руб", "rub", 1e9), (r"\s*млн\.?\s*руб", "rub", 1e6), (r"\s*тыс\.?\s*руб", "rub", 1e3),
    (r"\s*руб\.?\s*(?:/|за)\s*(?:шт|л|кг|точк)", "price", 1), (r"\s*руб", "rub", 1),
    (r"\s*млн\.?\s*шт", "qty", 1e6), (r"\s*тыс\.?\s*шт", "qty", 1e3), (r"\s*шт", "qty", 1),
    (r"\s*тыс\.?\s*т\b", "kg", 1e6), (r"\s*т\b", "kg", 1e3), (r"\s*кг", "kg", 1),
    (r"\s*млрд", "num", 1e9), (r"\s*млн", "num", 1e6), (r"\s*тыс", "num", 1e3),
    (r"\s*раз", "times", 1),
    (r"\s*(?:sku|ску|позици|бренд|производител|сет[ьеия]|сетях|точ[ек]|точк|магазин|категори|подгрупп|регион|город|месяц)", "count", 1),
]
UNIT_RX = [(re.compile(p, re.I), k, m) for p, k, m in UNITS]
COUNT_KIND = [(re.compile(p, re.I), k) for p, k in (
    (r"\s*(sku|ску|позици)", "sku"), (r"\s*бренд", "brand"), (r"\s*производител", "man"), (r"\s*сет", "net"),
    (r"\s*(точ|магазин)", "store"), (r"\s*(категори|подгрупп)", "cat"), (r"\s*регион", "reg"), (r"\s*город", "city"),
    (r"\s*месяц", "month"))]
# a sign must touch the number: «- 10 брендов» at the start of a list item is a bullet, not a minus
NUM_RX = re.compile(r"(?<![\w.,])([+−-]?)(\d{1,3}(?:[  ]\d{3})+|\d+)(?:[.,](\d+))?")
DATE_RX = re.compile(r"\b\d{1,2}\.\d{4}\b|\b\d{4}-\d{2}(?:-\d{2})?\b")
PLUS_RX = re.compile(r"(рост|вырос|увелич|прирост|прибав|выше)\w*\s*(на\s*)?$")
MINUS_RX = re.compile(r"(сниж|паден|упал|сократ|уменьш|потер|минус|ниже)\w*\s*(на\s*)?$")
# judgements the data cannot support: the model likes them, the check catches them
BANNED = [r"преми", r"устойчив", r"стабильн", r"доминир", r"лидер\w* рынка", r"аутсайдер", r"собственн\w* сет",
          r"сильн\w* бренд", r"сохран\w* лидерств", r"снижени\w* роста", r"нов\w* игрок", r"единственн", r"крайне",
          r"недоступн", r"не определен", r"уточнени\w* источник"]
BANNED_RX = [re.compile(b, re.I) for b in BANNED]
# a cause without the «гипотеза» (hypothesis) label: data shows what changed, not why (methodology §1.3).
# Conclusions drawn from the numbers themselves («доля снизилась при росте — рынок рос быстрее», «снижение дали точки
# и SKU на точку» in factor analysis) are not invented causes; external explanations and unsupported denials are caught.
CAUSE_RX = re.compile(r"не связан\w* с|не из-за|из-за |благодаря|обусловлен|вызван\w* ", re.I)
PROMO_RX = re.compile(r"акци|промо", re.I)


def _tokens(text, pool=False):
    """Numbers with units: [{val (in base units), kind, tol, sign, start, end, raw}].
    pool=True parses facts: small unitless integers are kept (counters «8 из 10», ranks «места 1–10»)."""
    text = DATE_RX.sub(lambda m: " " * len(m.group(0)), text)
    out = []
    for m in NUM_RX.finditer(text):
        sign_ch, intpart, frac = m.group(1), m.group(2), m.group(3)
        after = text[m.end():m.end() + 24]
        before = text[max(0, m.start() - 30):m.start()]
        if after[:1].isalpha():
            continue   # 300мл, 8шт inside a product name
        if after[:1] == "-" and after[1:2].isalpha() and not pool:
            continue   # «10-е место» (10th place) in the text; in facts «37-е место из 59» is a rank kept for matching
        if re.search(r"топ\s*-?\s*$", before.lower()) and not pool:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        if not pool and re.match(r"^\s*(?:[-*]\s*)?$", text[line_start:m.start()]) and re.match(r"^[.)]\s", after):
            continue   # numbered-list marker
        v = float(intpart.replace(" ", "").replace(" ", "") + ("." + frac if frac else ""))
        kind, mult = None, 1
        for rx, k, mu in UNIT_RX:
            if rx.match(after):
                kind, mult = k, mu
                break
        if kind == "count":   # «32 SKU», «61 производитель»: a counter is matched against a counter of the same kind
            kind = "count:" + next((k for rx, k in COUNT_KIND if rx.match(after)), "x")
            if kind == "count:month" and not pool:
                continue      # «за 12 месяцев» (over 12 months) describes the period, not a data value
        if kind is None and pool:
            # «порог прошли 59 из 978 производителей», «в 70 из 70 точек»: the first number has the same kind as the second
            m2 = re.match(r"\s*(?:-\w+\s+)?(?:\w+\s+)?из\s+[\d  ]+", after)
            if m2:
                tail = text[m.end() + m2.end():m.end() + m2.end() + 24]
                ck = next((k for rx, k in COUNT_KIND if rx.match(tail)), None)
                kind = f"count:{ck}" if ck else None
        if kind is None:
            if not pool and (v <= 12 and not frac or (2020 <= v <= 2030 and not frac)):
                continue   # ordinals and years
            kind = "num"
        digits = len(frac) if frac else 0
        sign = None
        if sign_ch in ("-", "−"):
            sign = -1
        elif sign_ch == "+":
            sign = 1
        else:
            b = before.lower()
            sign = 1 if PLUS_RX.search(b) else (-1 if MINUS_RX.search(b) else None)
        out.append({"val": v * mult, "kind": kind, "tol": 10 ** -digits * mult, "sign": sign,
                    "start": m.start(), "end": m.end(), "raw": text[m.start():m.end()].strip()})
    return out


_SUBJ_RX = re.compile(r"^\s*(?:\[[^\]]+\]\s*)?(?:\d+\.\s+|(?:сеть|бренд|производитель|SKU|баннер|формат|подгруппа|категория)\s+)(.{2,160}?):\s")


def _subject(line):
    """Subject of a fact line: «3. TOFFI (STM(X7)): ...» -> TOFFI; «сеть Нива: ...» -> Нива. Total lines -> None."""
    m = _SUBJ_RX.match(line)
    if not m:
        return None
    name = m.group(1).strip()
    # the last top-level parenthesised group is the manufacturer/brand annotation
    depth, cut = 0, None
    for i in range(len(name) - 1, -1, -1):
        ch = name[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                cut = i
                break
    if cut and name.endswith(")") and cut > 0 and name[cut - 1] == " ":
        name = name[:cut].strip()
    return name.replace(" [СТМ]", "").strip()


def _name_keys(name):
    keys = {name}
    keys.add(re.sub(r"\s*\[[^\]]*\]", "", name).strip())
    m = re.search(r"\[([^\]]+)\]", name)
    if m:
        keys.add(m.group(1).strip())
    return {k.lower() for k in keys if len(k) >= 2}   # «ЛАД», «НУ» are objects too; matched as whole words


def _pool(results):
    """Fact numbers: (token, line subject or None). Tables are excluded: the model never sees them."""
    pool, names = [], {}
    for r in results:
        lines = list(r.facts) + list(r.headline)
        for b in r.blocks:
            if b["kind"] == "kpis":
                lines += [f"{it['label']}: {it['value']} {it.get('sub') or ''}" for it in b["items"]]
        for line in lines:
            subj = _subject(line)
            if subj:
                for k in _name_keys(subj):
                    names[k] = subj
            for t in _tokens(line, pool=True):
                pool.append((t, subj))
    return pool, names


def _close(a, b):
    return abs(a["val"] - b["val"]) <= a["tol"] + 0.5 * b["tol"] + 1e-9


def _kinds_ok(a, b, strict_counts=()):
    if a["kind"] == b["kind"]:
        return True
    if a["kind"] in strict_counts:
        return False   # facts hold a counter of this kind («В выборке 32 SKU») — match against it only
    loose = ("rub", "qty", "kg", "num")   # "num" is a unitless number: matches any quantity
    return ((a["kind"] == "num" and (b["kind"] in loose or b["kind"].startswith("count")))
            or (b["kind"] == "num" and (a["kind"] in loose or a["kind"].startswith("count"))))


def _units(text):
    """Semantic units of the answer: list items and paragraph sentences (start, end)."""
    out, pos = [], 0
    for line in text.splitlines(keepends=True):
        body = line.strip()
        if body and re.match(r"^([-*•]|\d+[.)])\s", body):
            out.append((pos, pos + len(line)))
        elif body:
            s = 0
            for m in re.finditer(r"(?<=[.!?])\s+(?=[А-ЯA-Z«\"*])", line):
                out.append((pos + s, pos + m.end()))
                s = m.end()
            out.append((pos + s, pos + len(line)))
        pos += len(line)
    return out


def check_answer(answer, results, question=""):
    """Check the model text: every number with a unit must be in the facts (to the precision shown), its sign must
    match, and a number next to an object must belong to that object or to the totals; unsupported judgements
    are flagged. Returns {total, checked, unmatched: [{text, reason, start, end}], banned: [...]}."""
    pool, names = _pool(results)
    facts_low = " ".join(f.lower() for r in results for f in r.facts)
    toks = _tokens(answer)
    units = _units(answer)
    low = answer.lower()
    name_pos = []
    for k in sorted(names, key=len, reverse=True):
        # whole word: «ок» must not match inside «покупатель»; «гранит» does match «Граните» (inflected ending)
        for m in re.finditer(r"(?<!\w)" + re.escape(k) + (r"(?!\w)" if len(k) < 4 else ""), low):
            if not any(a <= m.start() < b for a, b, _ in name_pos):
                name_pos.append((m.start(), m.end(), names[k]))
    bad, checked = [], 0
    strict = {p["kind"] for p, _ in pool if p["kind"].startswith("count:")}
    for t in toks:
        cand = [(p, s) for p, s in pool if _kinds_ok(t, p, strict) and _close(t, p)]
        checked += 1
        if not cand:
            # complement to 100% («на остальные — 21,6%») — only next to «остальн/прочи» ("the rest/others")
            ctx_txt = low[max(0, t["start"] - 40):t["start"]]
            if t["kind"] == "pct" and re.search(r"остальн|прочи", ctx_txt) and any(
                    p["kind"] == "pct" and abs((100 - p["val"]) - t["val"]) <= t["tol"] + 0.5 * p["tol"] for p, _ in pool):
                continue
            bad.append({"text": t["raw"], "reason": "нет в расчетах", "start": t["start"], "end": t["end"]})
            continue
        if t["sign"] is not None:
            signed = [(p, s) for p, s in cand if p["sign"] is not None]
            if signed and all(p["sign"] != t["sign"] for p, _ in signed) and len(signed) == len(cand):
                fact = signed[0][0]["raw"]
                bad.append({"text": t["raw"], "reason": f"знак: в расчете {fact}", "start": t["start"], "end": t["end"]})
                continue
        # object binding: objects in the same answer unit
        u = next(((a, b) for a, b in units if a <= t["start"] < b), (0, len(answer)))
        objs = {n for a, b, n in name_pos if u[0] <= a < u[1] and a < t["start"]}
        if objs and not any(s is None or s in objs for _, s in cand):
            owner = cand[0][1]
            bad.append({"text": t["raw"], "reason": f"это число в расчете у «{owner}», а не у «{', '.join(sorted(objs))}»",
                        "start": t["start"], "end": t["end"]})
    banned = []
    for rx in BANNED_RX:
        for m in rx.finditer(answer):
            # a one-word judgement is allowed if its stem is in the facts; a phrase («снижение роста») only verbatim
            hit = m.group(0).lower()
            if (hit if " " in hit else hit[:6]) not in facts_low:
                banned.append({"text": m.group(0), "start": m.start(), "end": m.end()})
    if not PROMO_RX.search(question):
        for m in PROMO_RX.finditer(answer):
            banned.append({"text": m.group(0), "start": m.start(), "end": m.end()})
    for a, b in (_units(answer) if not any(r.tool == "factors" for r in results) else []):
        unit = answer[a:b]
        m = CAUSE_RX.search(unit)
        if m and not re.search(r"гипотез", unit, re.I):
            banned.append({"text": m.group(0).strip() + " (причина без пометки «гипотеза»)", "start": a + m.start(), "end": a + m.end()})
    return {"total": len(toks), "checked": checked, "unmatched": bad, "banned": banned}


def _remove_units(answer, spans):
    """Remove list items and sentences holding unconfirmed numbers or judgements. Returns (text, removed)."""
    units = _units(answer)
    drop = set()
    for s in spans:
        for i, (a, b) in enumerate(units):
            if a <= s["start"] < b:
                drop.add(i)
    removed = [answer[a:b].strip() for i, (a, b) in enumerate(units) if i in drop]
    keep, pos = [], 0
    for i, (a, b) in enumerate(units):
        keep.append(answer[pos:a])
        if i not in drop:
            keep.append(answer[a:b])
        pos = b
    keep.append(answer[pos:])
    text = _renumber(re.sub(r"\n{3,}", "\n\n", "".join(keep)))
    return text.strip(), removed


def _renumber(text):
    """After removals a numbered list must not start at «2.»: numbering restarts in each block."""
    out, n = [], 0
    for line in text.split("\n"):
        m = re.match(r"^(\d+)([.)])(\s+)", line)
        if m:
            n += 1
            line = f"{n}{m.group(2)}{m.group(3)}" + line[m.end():]
        elif not line.strip() or line.lstrip().startswith("**"):
            n = 0
        out.append(line)
    return "\n".join(out)


def _ensure_main(answer, results):
    """If «Главное» (key points) has no items left (all removed by the check), code rebuilds it from the key facts."""
    hl = [h for r in results for h in r.headline][:3] or [f for r in results[:1] for f in r.facts[:2]]
    if not hl:
        return answer
    bullets = "\n".join(f"- {h}" for h in hl)
    m = re.search(r"\*\*Главное\*\*:?[ \t]*", answer)
    if not m:
        return answer if len(answer.strip()) > 40 else "**Главное**\n" + bullets + ("\n\n" + answer if answer.strip() else "")
    rest = answer[m.end():]
    nxt = re.search(r"(?:^|\n)[ \t]*\*\*[^*\n]+\*\*", rest)
    body = rest[:nxt.start()] if nxt else rest
    if body.strip():
        return answer
    tail = rest[nxt.start():].lstrip("\n") if nxt else ""
    return answer[:m.end()] + "\n" + bullets + ("\n\n" + tail if tail else "")


def _fix_yo(text, results):
    """«ё» -> «е» in the generated text, but not inside names coming from the data."""
    names = {w for r in results for f in r.facts for w in re.findall(r"[^\s,;:()«»]*[ёЁ][^\s,;:()«»]*", f)}
    keep = {}
    for i, w in enumerate(sorted(names, key=len, reverse=True)):
        ph = f"\x00{i}\x00"
        if w in text:
            text = text.replace(w, ph)
            keep[ph] = w
    text = text.replace("ё", "е").replace("Ё", "Е")
    for ph, w in keep.items():
        text = text.replace(ph, w)
    return text


# ——— Loop ———

def _user_error(err):
    """Failure reason for the user: without hints aimed at the model (find_entities, argument formats)
    and without technical exceptions — those stay in the «Как считалось» panel."""
    if not err or err.startswith(("Ошибка расчета", "Ошибка: нет инструмента")):
        return None
    sents = re.split(r"(?<=[.!?])\s+", err.replace("Ошибка: ", "", 1))
    keep = [s for s in sents if not re.search(r"find_entities|Используй|level должен|Формат:", s)]
    out = " ".join(keep).strip()
    if S.OWN_CATS_LABEL in out and "нет продаж" in out:
        out += (" По умолчанию анализ идет только по категориям, где есть товары NORD; чтобы посчитать другую категорию, "
                "назовите ее в вопросе или попросите всю панель.")
    return out or None


def _exec(name, args):
    """Run a tool. Returns (Result|None, text for the model, ok)."""
    if name not in TOOLS:
        return None, f"Ошибка: нет инструмента {name}. Есть: {', '.join(TOOLS)}.", False
    try:
        r = TOOLS[name](args)
        r.args = args
        return r, r.facts_text(), True
    except ToolError as e:
        return None, f"Ошибка: {e}", False
    except Exception as e:
        return None, f"Ошибка расчета ({TOOL_RU.get(name, name)}): {type(e).__name__}: {str(e)[:300]}", False


def _history_msgs(history):
    msgs = []
    for h in (history or [])[-3:]:
        msgs.append({"role": "user", "content": h.get("q") or ""})
        if h.get("slice") or h.get("headline"):
            body = "\n".join(x for x in [h.get("slice") or "",
                                         ("Ключевые цифры: " + "; ".join(h["headline"][:4])) if h.get("headline") else ""] if x)
        else:
            body = "(ответ без подтвержденного расчета; его цифры не использовать)"
        msgs.append({"role": "assistant", "content": body})
    return msgs


def run(question, history=None, emit=lambda e: None, ctx=None, forced=None):
    """Answer a question. history: past turns [{q, slice, headline, ctx}]; ctx: previous turn's slice (for follow-ups);
    forced: {tool, args} — calculation without the model choosing the tool (next-step button)."""
    t0 = time.time()
    found = E.dictionary().find(question)
    inh = _inherit(found, ctx)
    hint = _hint(found, inh, question)
    rec = found.summary()
    note = _inherit_note(inh)
    if note:
        rec = f"{rec}; {note}" if rec and rec != "ничего не распознано" else note
    emit({"type": "step", "text": f"Распознано: {rec}"})
    ql = question.lower().replace("ё", "е")

    def delta(kind, text):
        emit({"type": "delta", "kind": kind, "text": text})

    results, calls_log, seen, stats = [], [], {}, []
    msgs = [{"role": "system", "content": system_prompt()}] + _history_msgs(history)
    th = _tool_hint(question, found)
    user = f"{question}\n\n[Распознано кодом: {rec}]\n[Готовые аргументы: {json.dumps(hint, ensure_ascii=False)}]"
    if th and not forced:
        user += f"\n[Подходит инструмент: {th}]"
    msgs.append({"role": "user", "content": user})
    think = model_thinks()

    def call(name, args, source="model"):
        args, filled, overridden = _normalize(name, args, found, hint, question, ctx) if source == "model" else (args, {}, {})
        key = name + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        emit({"type": "tool", "name": name, "args": args})
        ts = time.time()
        if key in seen:
            r, err, ok = None, seen[key][0], seen[key][1]
            content = ("Этот вызов уже был, результат выше. Напиши ответ." if ok else
                       f"Этот вызов уже вернул ошибку: {err} Исправь аргументы или ответь, что посчитать не удалось, без чисел.")
        else:
            r, content, ok = _exec(name, args)
            err = content
            seen[key] = (content, ok)
            if r:
                results.append(r)
        sec = round(time.time() - ts, 1)
        calls_log.append({"name": name, "args": args, "sec": sec, "ok": ok, "filled": filled, "overridden": overridden,
                          "source": source, **({} if ok else {"error": err[:400]})})
        emit({"type": "tool_done", "name": name, "sec": sec, "ok": ok, "note": "" if ok else content[:300]})
        return content

    final, early = None, None
    out_of_range = _fld(found, "out_of_range", [])
    meta = bool(META_RE.search(ql)) and not (any(found.filters.get(k) for k in S.OBJECT_KEYS + S.CATEGORY_KEYS)
                                             or found.periods or found.period_kind)
    if out_of_range and not found.periods and not forced:
        lo, hi = S.data_bounds()
        early = (f"Данных за {', '.join(out_of_range)} нет: в базе есть месяцы {lo:%m.%Y}–{hi:%m.%Y}. "
                 "Уточните период в этих границах.")
    elif forced and forced.get("tool") in TOOLS:
        content = call(forced["tool"], dict(forced.get("args") or {}), source="forced")
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": forced["tool"], "arguments": forced.get("args") or {}}}]})
        msgs.append({"role": "tool", "content": content, "tool_name": forced["tool"]})
        emit({"type": "llm", "text": "Модель пишет вывод"})
        out = _ollama(msgs, None, think=think, on_delta=delta)
        stats.append(out["stats"])
        final = out["content"]
    elif meta:
        # questions about the data itself: code builds the description, the model retells it without new calculations
        extra = [("data_info", {})]
        m = LIST_RE.search(ql)
        if m:
            lv = "manufacturer" if m.group(2).startswith("производител") else "brand"
            extra.append(("ranking", {"level": lv, "top": 30, **{k: v for k, v in hint["filters"].items() if k in S.STORE_KEYS}}))
        for name, args in extra:
            content = call(name, args, source="code")
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]})
            msgs.append({"role": "tool", "content": content, "tool_name": name})
        ask = "Ответь на мой вопрос прямо по этим результатам. Инструменты не нужны."
        if m:
            ask += " Сначала назови, сколько их всего (из описания данных), затем крупнейшие (из рейтинга)."
        msgs.append({"role": "user", "content": ask})
        emit({"type": "llm", "text": "Модель пишет вывод"})
        out = _ollama(msgs, None, think=think, on_delta=delta)
        stats.append(out["stats"])
        final = out["content"]
    else:
        nudged = False
        for step in range(MAX_STEPS + 1):
            use_tools = step < MAX_STEPS
            emit({"type": "llm", "text": "Модель выбирает анализ" if not results else "Модель пишет вывод"})
            out = _ollama(msgs, TOOL_SPECS if use_tools else None, think=think, on_delta=delta)
            stats.append(out["stats"])
            tcalls = out["tool_calls"]
            text = out["content"]
            if not tcalls:
                tcalls, text = _text_calls(text)   # a call written as text is executed, not shown
                if tcalls and not use_tools:
                    tcalls = []
            if not tcalls:
                analytic_done = any(r.tool not in SERVICE for r in results)
                if (not analytic_done and not nudged and use_tools and not any(not c["ok"] for c in calls_log)
                        and (hint["filters"] or ANALYTIC_RE.search(ql)) and _tokens(text)):
                    # numbers without a calculation: demand a tool call once
                    nudged = True
                    msgs.append({"role": "assistant", "content": text})
                    msgs.append({"role": "user", "content": "Цифры без расчета показывать нельзя. Вызови подходящий инструмент "
                                                            "через вызов инструмента с «Готовыми аргументами»."})
                    continue
                final = text
                break
            msgs.append({"role": "assistant", "content": text, "tool_calls": tcalls})
            for c in tcalls:
                fn = c.get("function", {})
                name, args = fn.get("name"), fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                content = call(name, args)
                msgs.append({"role": "tool", "content": content, "tool_name": name})
    if final is None:
        final = early or "Не удалось получить ответ модели."
    final = _clean_text(final)

    analytic = [r for r in results if r.tool not in SERVICE]
    failed = [c for c in calls_log if not c["ok"]]
    check = {"total": 0, "checked": 0, "unmatched": [], "banned": [], "removed": [], "status": "none"}
    if early:
        pass
    elif failed and not analytic and any(c["name"] not in SERVICE for c in calls_log):
        # calculation failed — don't show the model's answer: without results it makes up numbers
        user_errs = list(dict.fromkeys(e for e in (_user_error(c.get("error", "")) for c in failed) if e))
        final = ("Посчитать не удалось. " + (" ".join(user_errs[:2]) if user_errs else
                 "Внутренняя ошибка расчета; подробности — в «Как считалось».") +
                 "\n\nПереформулируйте вопрос или уточните название, период, сеть.")
        results = [r for r in results if r.tool == "find_entities"]
    else:
        check.update(check_answer(final, results, question))
        if not results and check["total"]:
            final = ("Посчитать не удалось: модель ответила без расчета, такие цифры показывать нельзя. "
                     "Переформулируйте вопрос: назовите объект, сеть, период.")
            check.update({"total": 0, "checked": 0, "unmatched": [], "banned": []})
        elif (check["unmatched"] or check["banned"]) and results:
            # self-check: ask once to fix the specific spots
            n = len(check["unmatched"]) + len(check["banned"])
            emit({"type": "llm", "text": f"Проверка нашла {n} мест(а) без опоры на расчет — модель переписывает ответ"})
            probs = [f"«{u['text']}» — {u['reason']}" for u in check["unmatched"][:8]]
            probs += [f"«{b['text']}» — оценка, которой нет в данных" for b in check["banned"][:6]]
            msgs.append({"role": "assistant", "content": final})
            msgs.append({"role": "user", "content": "Проверка ответа:\n" + "\n".join(probs) +
                         "\nПерепиши ответ целиком: исправь или убери эти места, используй только числа и формулировки "
                         "из результатов инструментов. Инструменты не вызывай."})
            out = _ollama(msgs, None, think=think, on_delta=delta, temperature=0)
            stats.append(out["stats"])
            retry = _clean_text(out["content"])
            if retry:
                rc = check_answer(retry, results, question)
                if len(rc["unmatched"]) + len(rc["banned"]) <= n:
                    final, check = retry, {**check, **rc}
        if check["unmatched"] or check["banned"]:
            # whatever is still unfixed after the retry is removed by code: unverified numbers never reach the user
            final, removed = _remove_units(final, check["unmatched"] + check["banned"])
            check["removed"] = removed
            final = _ensure_main(final, analytic)
            rc = check_answer(final, results, question)
            check.update({"total": rc["total"], "checked": rc["checked"], "unmatched": rc["unmatched"], "banned": rc["banned"]})
        if check["total"] or check["removed"]:
            check["status"] = ("ok" if not check["removed"] and not check["unmatched"] and not check["banned"]
                               else ("partial" if not check["unmatched"] else "fail"))
    # «Топ-10:» followed by two items and no pointer to the table reads as the complete list
    mt = re.search(r"топ[-\s]?(\d+)", final, re.I)
    if (mt and "таблиц" not in final.lower() and any(b["kind"] == "table" for r in analytic for b in r.blocks)
            and len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s", final, re.M)) < int(mt.group(1))):
        final += "\n\nПолный список — в таблице ниже."
    final = _fix_yo(final, results)
    caveats = []
    for r in results:
        for c in r.caveats:
            if c not in caveats:
                caveats.append(c)
    main = analytic[0] if analytic else None
    main_call = next((c for c in calls_log if c["ok"] and c["name"] not in SERVICE), None)
    ctx_out = None
    if main is not None and main_call:
        a = main_call["args"]
        ctx_out = {"filters": main.meta.get("filters") or {}, "exclude": main.meta.get("exclude") or {},
                   "period": a.get("period"), "base_period": a.get("base_period"),
                   "period_label": main.meta.get("period_label") if a.get("period") else None,
                   "measure": main.meta.get("measure"), "level": a.get("level")}
    nxt = []
    for r in analytic:
        for x in r.next:
            if len(nxt) < 4 and all(x["text"] != y["text"] for y in nxt):
                nxt.append(x)
    return {"question": question, "answer": final, "slice": main.slice_line if main else "",
            "headline": [h for r in analytic for h in r.headline][:4],
            "results": [r.to_dict() for r in results], "caveats": caveats, "recognized": rec, "calls": calls_log,
            "check": check, "ctx": ctx_out, "next": nxt, "forced": bool(forced), "model": MODEL,
            "sec": round(time.time() - t0, 1),
            "tokens": {"prompt": sum((s.get("prompt_eval_count") or 0) for s in stats),
                       "gen": sum((s.get("eval_count") or 0) for s in stats)}}


def summary_for_history(ans):
    """Short memory of a turn: slice and code key figures (the model text never goes into history)."""
    hl = ans.get("headline") or []
    return "\n".join(x for x in [ans.get("slice") or "", ("Ключевые цифры: " + "; ".join(hl[:4])) if hl else ""] if x)


def history_item(ans):
    """Conversation turn for the next question (the history format of run)."""
    return {"q": ans.get("question", ""), "slice": ans.get("slice") or "", "headline": ans.get("headline") or [],
            "ctx": ans.get("ctx")}
