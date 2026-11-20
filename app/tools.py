"""Analysis tools of the agent. Each takes the model's arguments (dict) and returns a Result.

Methodology rules (docs/methodology.md) are enforced here, not in the prompt: comparable networks for growth,
identical months, kids' subgroups as a separate row, private labels and «не указан» (unspecified) in competitor
selection, Depo without purchases. Facts are written so the model has nothing left to compute: totals, object
counts, rank windows, row aggregates.
"""
import difflib
import math
import re
import threading

import numpy as np
import pandas as pd

from . import entities as E
from . import semantic as S
from .result import Result, ToolError

LEVEL = {
    "manufacturer": "k.manufacturer", "brand": "k.brand", "subbrand": "k.subbrand", "sku": "k.sku_name",
    "subgroup": "k.subgroup_name", "subgroup_base": "k.subgroup_base", "product_type": "k.product_type",
    "product": "k.product", "category_group": "k.group_name",
    "network": "n.network", "banner": "n.banner", "format": "n.format", "district": "n.district",
    "region": "n.region", "city": "n.city", "month": "n.month",
}
LEVEL_RU = {"manufacturer": "Производитель", "brand": "Бренд", "subbrand": "Подбренд", "sku": "SKU",
            "subgroup": "Подгруппа", "subgroup_base": "Категория", "product_type": "Тип товара", "product": "Товар",
            "category_group": "Группа", "network": "Сеть", "banner": "Сеть / баннер", "format": "Формат",
            "district": "Округ", "region": "Регион", "city": "Город", "month": "Месяц"}
LEVEL_GEN = {"manufacturer": "производителей", "brand": "брендов", "subbrand": "подбрендов", "sku": "SKU",
             "subgroup": "подгрупп", "subgroup_base": "категорий", "product_type": "типов товара", "product": "товаров",
             "category_group": "групп", "network": "сетей", "banner": "сетей и баннеров", "format": "форматов",
             "district": "округов", "region": "регионов", "city": "городов"}
# «без метки бренда» ("without a brand label"): genitive singular
LEVEL_GEN1 = {"manufacturer": "производителя", "brand": "бренда", "subbrand": "подбренда", "sku": "SKU",
              "subgroup": "подгруппы", "subgroup_base": "категории", "product_type": "типа товара", "product": "товара",
              "category_group": "группы", "network": "сети", "banner": "баннера", "format": "формата",
              "district": "округа", "region": "региона", "city": "города"}
UNIT = {"rub": ("mln", "млн руб."), "qty": ("num0", "шт."), "kg": ("th", "т (тыс. л)")}
DIV = {"rub": 1e6, "qty": 1, "kg": 1e3}   # chart divisor: mln RUB, units, tonnes
MEASURE_RU = {"rub": "в рублях", "qty": "в штуках", "kg": "в кг/л"}
BLANK = {".", "", "не указан", "Не указан", "нет", "Нет"}
SORTS = {"sales": "по продажам", "growth": "по росту продаж", "gain": "по приросту продаж",
         "decline": "по снижению продаж", "share_change": "по изменению доли"}
FILLER = {"для", "и", "с", "со", "в", "на", "по", "за", "от"}    # function words in category names


# ——— argument parsing ———

def _norm_simple(s):
    return str(s).lower().replace("ё", "е").replace("^", "").strip()


def _words(s):
    return E.norm(s).split()


def _prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _same_stem(a, b):
    """Same word in a different case or number: «маска» / «маски», «тульской» / «тульская»."""
    if a == b:
        return True
    k = _prefix_len(a, b)
    return k >= 3 and k >= min(len(a), len(b)) - 2


def _stem_match(value, pool, prefix=False):
    """Values of pool whose words match the words of value by stem, in the same order.
    prefix=True — value may be the beginning of a name («гигиена волос» -> «Гигиена волос и кожи головы»).
    Of several matches the best are kept: longer common stems, fewer extra words."""
    vw = _words(value)
    if not vw:
        return []
    scored = []
    for x in pool:
        xw = _words(x)
        if len(xw) < len(vw) or (len(xw) > len(vw) and not prefix):
            continue
        if all(_same_stem(p, q) for p, q in zip(vw, xw)):
            scored.append((sum(_prefix_len(p, q) for p, q in zip(vw, xw)) - (len(xw) - len(vw)), x))
    if not scored:
        return []
    best = max(s for s, _ in scored)
    return [x for s, x in scored if s == best]


def _head_match(value, pool):
    """A name from pool that is the beginning of value: «шампунь для волос» -> «Шампунь». The longest one wins.
    The head must cover at least half of the significant words: «Гигиена» does not match «гигиена волос детская»."""
    vw = _words(value)
    n_sig = len([w for w in vw if w not in FILLER])
    hits = []
    for x in pool:
        xw = _words(x)
        if (xw and len(xw) < len(vw) and all(_same_stem(p, q) for p, q in zip(xw, vw))
                and 2 * len([w for w in xw if w not in FILLER]) >= n_sig):
            hits.append((len(xw), x))
    if not hits:
        return []
    best = max(n for n, _ in hits)
    return [x for n, x in hits if n == best]


def _keyword_match(value, key):
    """Recognition keywords (E.PRODUCT_KEYWORDS) for a single value. A keyword is accepted only if the value's
    other significant words also occur in the matched category: «кондиционер для волос» -> «Бальзам для волос»,
    but «кондиционер для белья» does not match (no such category in the panel)."""
    vw = _words(value)
    for kw, k2, val in E.PRODUCT_KEYWORDS:
        if k2 != key:
            continue
        ks = kw.split()
        for i in range(len(vw) - len(ks) + 1):
            if all(vw[i + j].startswith(ks[j]) for j in range(len(ks))):
                rest = [w for j, w in enumerate(vw) if not i <= j < i + len(ks) and w not in FILLER]
                tw = _words(val)
                if all(any(_same_stem(w, t) for t in tw) for w in rest):
                    return val
    return None


def _nearest(value, pool, n=3):
    low = {str(x).lower(): x for x in pool}
    return [low[x] for x in difflib.get_close_matches(str(value).lower(), list(low), n=n, cutoff=0.3)]


_cache = {}
_cache_lock = threading.Lock()


def _cached(name, fn):
    with _cache_lock:
        if name not in _cache:
            _cache[name] = fn()
        return _cache[name]


def _banners():
    """Banners from the data: {banner: network}. Granit carries the Mayak banners; for other networks banner = short network name."""
    return _cached("banners", lambda: dict(S.con().execute("SELECT DISTINCT banner, network FROM store").fetchall()))


def _city_pool():
    """Cities as understood by the city filter: store_x.city_eff (Moscow, St Petersburg, Sevastopol are region-cities)."""
    return _cached("cities", lambda: [r[0] for r in S.con().execute(
        "SELECT DISTINCT city_eff FROM store_x WHERE city_eff IS NOT NULL").fetchall()])


def _brand_parts(b):
    """Brand lookup keys: full name, part before the bracket, part in [..], name without (PL); spaces around & collapsed.
    «СВОНЛИ [SWANLY]» is found by SWANLY, «СИНЯЯ ЦЕНА (PL)» by СИНЯЯ ЦЕНА, HILL&SHORE by HILL & SHORE."""
    s = re.sub(r"\s*&\s*", "&", str(b))
    out = {s}
    m = re.match(r"^(.*?)\s*[\[(]", s)
    if m:
        out.add(m.group(1))
    out.update(re.findall(r"\[([^\]]+)\]", s))
    out.add(re.sub(r"\s*\(PL\)\s*", " ", s))
    return {E.norm(x).strip() for x in out if E.norm(x).strip()}


def _brand_index():
    def build():
        idx = {}
        for b, _, rub in E.dictionary().brands:
            for k in _brand_parts(b):
                idx.setdefault(k, []).append((b, rub))
        return {k: [b for b, _ in sorted(v, key=lambda x: -(x[1] or 0))] for k, v in idx.items()}
    return _cached("brand_index", build)


def _find_banner(v):
    nv = _norm_simple(v)
    for b in _banners():
        if b and _norm_simple(b) == nv:
            return b
    nn = E.norm(v).strip()
    for b, al in E.BANNER_ALIASES.items():
        if any(E.norm(x).strip() == nn for x in al):
            return b
    return None


def _find_network(v):
    d = E.dictionary()
    nv = _norm_simple(v)
    for n in d.networks:
        if _norm_simple(n) == nv or _norm_simple(S.net_label(n)) == nv:
            return [n]
    nn = E.norm(v)
    if nn.strip() in E.NETWORK_GROUPS:             # «Х7» means both Semerochka and Perelesok
        return list(E.NETWORK_GROUPS[nn.strip()])
    for n, al in E.NETWORK_ALIASES.items():
        if any(E.norm(x) == nn for x in al):
            return [n]
    pool = {E.norm(x).strip(): n for n, al in E.NETWORK_ALIASES.items() for x in al}
    m = difflib.get_close_matches(nn.strip(), list(pool), n=1, cutoff=0.6)
    return [pool[m[0]]] if m else None


def _region_base(s):
    """Region base word: «Московская область» -> «московская», «Республика Татарстан» -> «татарстан», «Москва г» -> «москва»."""
    w = [x for x in _words(str(s).lower().replace("автономный округ", "ао").replace("автономная область", "аобл"))
         if x not in ("область", "обл", "край", "республика", "респ", "ао", "аобл", "г", "город")]
    return " ".join(w)


CITY_REGION_ALIASES = {"спб": "Санкт-Петербург г", "питер": "Санкт-Петербург г", "петербург": "Санкт-Петербург г"}
DISTRICT_ALIASES = {"центральный": "ЦФО", "приволжский": "ПФО", "северо западный": "СЗФО", "южный": "ЮФО",
                    "уральский": "УФО", "сибирский": "СФО", "северо кавказский": "СКФО", "дальневосточный": "ДФО",
                    "крымский": "КФО"}


def _find_region(v):
    d = E.dictionary()
    nv = _norm_simple(v)
    base = _region_base(v)
    for x in d.regions:
        if _norm_simple(x) == nv or _region_base(x) == base:
            return x
    if base in CITY_REGION_ALIASES:
        return CITY_REGION_ALIASES[base]
    bases = {_region_base(x): x for x in d.regions}
    hits = _stem_match(base, list(bases))
    if not hits:   # «Якутия» -> «Саха (Якутия)»: the value's words occur among the region's words
        bw = base.split()
        hits = [b for b in bases if bw and all(any(_same_stem(p, q) for q in b.split()) for p in bw)]
    return bases[hits[0]] if len(hits) == 1 else None


def _find_city(v):
    pool = _city_pool()
    nv = _norm_simple(v)
    base = " ".join(x for x in _words(v) if x not in ("г", "город"))
    for x in pool:
        if _norm_simple(x) == nv or " ".join(w for w in _words(x) if w != "г") == base:
            return x
    bases = {" ".join(w for w in _words(x) if w != "г"): x for x in pool if not x.startswith("город не указан")}
    hits = _stem_match(base, list(bases))
    return bases[hits[0]] if len(hits) == 1 else None


def _category(v, key, pool):
    """Product type or subgroup without false substitutions: exact name, same words in another case/number, name head,
    recognition keyword (with the remaining words checked), close spelling (difflib >= 0.85)."""
    nv = _norm_simple(v)
    for x in pool:
        if _norm_simple(x) == nv:
            return [x]
    # residual «…_Прочее» ("other") buckets match only by exact name: «Косметика» must not become «Косметика_Прочее»
    pool = [x for x in pool if "прочее" not in str(x).lower()]
    for prefix in (False, True):
        hits = _stem_match(v, pool, prefix=prefix)
        if len(hits) == 1:
            return hits
        if len(hits) > 1:
            raise ToolError(f"«{v}» подходит к нескольким значениям ({FILTER_RU[key]}): " + ", ".join(sorted(hits)[:8])
                            + ". Передай одно точное значение или список.")
    hits = _head_match(v, pool)
    if len(hits) == 1:
        return hits
    kw = _keyword_match(v, key)
    if kw and kw in pool:
        return [kw]
    low = {str(x).lower().strip(): x for x in pool}
    m = difflib.get_close_matches(nv, list(low), n=1, cutoff=0.85)
    if m:
        return [low[m[0]]]
    return None


FILTER_RU = {"product_type": "тип товара", "subgroup": "подгруппа", "subgroup_exact": "подгруппа",
             "category_group": "группа", "product": "товар"}


def _sku_values(col):
    return _cached(f"sku_{col}", lambda: [r[0] for r in S.con().execute(
        f"SELECT DISTINCT {col} FROM sku WHERE {col} IS NOT NULL").fetchall()])


def resolve_filter(key, value):
    """Map a filter value to exact dictionary values: (key, [values]).
    The key may change if the value belongs to another level: a Mayak banner passed as a network -> banner;
    a network passed as a banner -> network; «Москва» as a city -> region «Москва г» (Moscow's city is empty in the data);
    a region passed as a city and vice versa; a brand passed as a manufacturer -> brand; a subgroup as a type -> subgroup.
    Not found — ToolError with a hint."""
    d = E.dictionary()
    v = str(value).strip()
    nv = _norm_simple(v)
    if key in ("network", "banner"):
        b = _find_banner(v)
        if b and (key == "banner" or b not in {S.net_label(n) for n in d.networks}):
            return "banner", [b]
        n = _find_network(v)
        if n:
            return "network", n
        if key == "banner":
            raise ToolError(f"Баннер «{v}» не найден. Баннеры в данных: " + ", ".join(sorted(x for x in _banners() if x)))
        raise ToolError(f"Сеть «{v}» не найдена. Есть: " + ", ".join(sorted(S.net_label(n) for n in d.networks)))
    if key == "format":
        for f, al in E.FORMAT_ALIASES.items():
            if nv == f.lower() or any(E.norm(x).strip() == E.norm(v).strip() for x in al):
                return key, [f]
        if nv == "рц":
            return key, ["РЦ"]
        raise ToolError(f"Формат «{v}» не найден. Есть: ММ (у дома), СМ (супермаркет), ГМ (гипермаркет), Интернет Магазин, РЦ")
    if key == "manufacturer":
        for m, _ in d.manufacturers:
            if _norm_simple(m) == nv:
                return key, [m]
        if E.norm(v) in d.man_index:
            return key, [d.man_index[E.norm(v)]]
        for b, _, _ in d.brands:           # «L'AURELLE PARIS» is a brand, not a manufacturer
            if _norm_simple(b) == nv:
                return "brand", [b]
        names = [m for m, _ in d.manufacturers]
        m = difflib.get_close_matches(v.upper(), names, n=1, cutoff=0.85)
        if m:
            return key, [m[0]]
        raise ToolError(f"Производитель «{v}» не найден. Используй find_entities, чтобы найти точное название.")
    if key == "brand":
        for b, _, _ in d.brands:
            if _norm_simple(b) == nv:
                return key, [b]
        hits = _brand_index().get(E.norm(re.sub(r"\s*&\s*", "&", v)).strip())
        if hits:   # several labels of one brand (methodology §6.2) are summed; all are visible in the slice
            return key, hits
        row = S.con().execute("SELECT DISTINCT brand FROM sku WHERE lower(brand) = lower(?)", [v]).fetchall()
        if row:
            return key, [row[0][0]]
        m = difflib.get_close_matches(v.upper(), [b for b, _, _ in d.brands], n=1, cutoff=0.85)
        if m:
            return key, [m[0]]
        raise ToolError(f"Бренд «{v}» не найден. Используй find_entities, чтобы найти точное название.")
    if key == "subbrand":
        row = S.con().execute("SELECT DISTINCT subbrand FROM sku WHERE lower(subbrand) = lower(?)", [v]).fetchall()
        if row:
            return key, [row[0][0]]
        raise ToolError(f"Подбренд «{v}» не найден. Используй find_entities, чтобы найти точное название.")
    if key in ("subgroup", "subgroup_exact"):
        pool = d.subgroups if key == "subgroup" else _sku_values("subgroup_name")
        val = v
        if key == "subgroup":          # «Гигиена волос детская» (kids') -> base subgroup (make_scope sets kids)
            val = re.sub(r"\s+детск\S*$", "", v, flags=re.I).strip() or v
        hit = _category(val, key, pool)
        if hit:
            return key, hit
        if key == "subgroup":
            t = _category(v, "product_type", d.types)       # a type passed as a subgroup («Шампунь»)
            if t:
                return "product_type", t
        raise ToolError(f"Подгруппа «{v}» не найдена. Есть: " + ", ".join(sorted(x.strip() for x in d.subgroups if x)))
    if key == "product_type":
        hit = _category(v, key, d.types)
        if hit:
            return key, hit
        try:                                # a subgroup passed as a type («Средства для окраски волос»)
            k2, sg = resolve_filter("subgroup", v)
            if k2 == "subgroup":
                return k2, sg
        except ToolError:
            pass
        raise ToolError(f"Тип товара «{v}» не найден, такой категории в панели может не быть. Ближайшие: "
                        + (", ".join(_nearest(v, d.types)) or "нет") + ".")
    if key in ("category_group", "product"):
        pool = _sku_values("group_name" if key == "category_group" else "product")
        hit = _category(v, key, pool)
        if hit:
            return key, hit
        raise ToolError(f"Значение «{v}» ({FILTER_RU[key]}) не найдено. Ближайшие: " + (", ".join(_nearest(v, pool)) or "нет") + ".")
    if key == "district":
        for x in d.districts:
            if _norm_simple(x) == nv:
                return key, [x]
        base = " ".join(w for w in _words(v) if w not in ("федеральный", "округ", "фо"))
        if base in DISTRICT_ALIASES:
            return key, [DISTRICT_ALIASES[base]]
        raise ToolError(f"Округ «{v}» не найден. Есть: " + ", ".join(d.districts))
    if key == "region":
        x = _find_region(v)
        if x:
            return key, [x]
        c = _find_city(v)
        if c:
            return ("region", [c]) if c in S.CITY_REGIONS else ("city", [c])
        raise ToolError(f"Регион «{v}» не найден. Ближайшие: " + (", ".join(_nearest(v, d.regions)) or "нет") + ".")
    if key == "city":
        base = " ".join(x for x in _words(v) if x not in ("г", "город"))
        if base in CITY_REGION_ALIASES:
            return "region", [CITY_REGION_ALIASES[base]]
        c = _find_city(v)
        if c:
            # for Moscow, St Petersburg and Sevastopol the city is almost always empty: a region filter takes all stores
            return ("region", [c]) if c in S.CITY_REGIONS else ("city", [c])
        x = _find_region(v)
        if x:
            return "region", [x]
        raise ToolError(f"Город «{v}» не найден. Ближайшие: " + (", ".join(_nearest(v, _city_pool())) or "нет") + ".")
    return key, [v]


def resolve(key, value):
    """First exact dictionary value for a filter (the key may change, see resolve_filter)."""
    return resolve_filter(key, value)[1][0]


def make_scope(a):
    a = dict(a)
    f, ex = {}, {}
    kids_hint = []      # subgroup given as its kids' variant («Гигиена волос детская») -> base subgroup + kids only

    def add(dst, key, vals):
        for v in vals:
            if v is None or not str(v).strip():
                continue
            if key == "sku":
                if str(v) not in dst.setdefault(key, []):
                    dst[key].append(str(v))
                continue
            k2, vs = resolve_filter(key, v)
            if dst is f and k2 == "subgroup":
                kids_hint.append(bool(re.search(r"детск", str(v), re.I)) and not any(re.search(r"детск", x, re.I) for x in vs))
            for x in vs:
                if x not in dst.setdefault(k2, []):
                    dst[k2].append(x)

    for key in S.FILTERS:
        vals = a.get(key)
        if vals in (None, "", [], "все", "all"):
            continue
        add(f, key, vals if isinstance(vals, list) else [vals])
    # exclusions: «все сети, кроме Гранита» ("all networks except Granit"), «без СТМ» ("without private labels")
    exc = a.get("exclude") or {}
    if isinstance(exc, dict):
        for key, vals in exc.items():
            if key in S.FILTERS and vals not in (None, "", []):
                add(ex, key, vals if isinstance(vals, list) else [vals])
    kids = a.get("kids")
    if isinstance(kids, str):
        kids = {"true": True, "false": False}.get(kids.lower())
    if kids is None and kids_hint and all(kids_hint):
        kids = True
    # all_categories — the user explicitly asked for the whole panel; otherwise, with no category named, the slice is NORD's categories
    return S.Scope({k: v for k, v in f.items() if v}, portal=bool(a.get("portal")),
                   kids=kids if isinstance(kids, bool) else None,
                   exclude_pl=bool(a.get("exclude_pl")), ex={k: v for k, v in ex.items() if v},
                   universe=not a.get("all_categories"))


def net_lbl(scope):
    """Where growth was computed: «9 сопоставимых сетей» without a network filter, otherwise the selected networks and banners."""
    if not scope.has_network_filter():
        ex = [S.net_label(x) for x in scope.excluded("network")] + list(scope.excluded("banner"))
        return "9 сопоставимых сетей" + (f" без: {', '.join(ex)}" if ex else "")
    return ", ".join([S.net_label(x) for x in scope.values("network")] + list(scope.values("banner")))


def get_measure(a):
    m = str(a.get("measure") or "rub").lower().strip()
    m = {"руб": "rub", "руб.": "rub", "рубли": "rub", "шт": "qty", "шт.": "qty", "штуки": "qty", "упаковки": "qty",
         "кг": "kg", "литры": "kg", "кг/л": "kg"}.get(m, m)
    if m not in UNIT:
        raise ToolError("measure должен быть rub, qty или kg.")
    return m


def main_period(a, default="mat"):
    """Analysis period: (Period, explicitly_given)."""
    p = a.get("period")
    if p:
        try:
            per = S.clip(S.parse_period(p, default))
        except Exception:
            raise ToolError(f"Не понял период «{p}». Формат: 2026-01..2026-06, 2026-09, 2025, mat, ytd, all.")
        if not per:
            lo, hi = S.data_bounds()
            raise ToolError(f"Период {p} вне данных. Данные есть с {lo:%m.%Y} по {hi:%m.%Y}.")
        # «за последние 12 месяцев» ("last 12 months") is an explicit choice: growth is also 12 months vs the previous 12, not YTD
        return per, True
    return S.parse_period(default), False


def growth_pair(a, period, explicit):
    """Growth periods (gcur, gbase) of equal length.
    No base: the same period a year earlier; no period given — year to date (methodology §2.3).
    Explicit base: later than the period — the two are swapped; different length — aligned to the same months
    of the base year («2026 к 2025» -> Jan–Sep 2026 vs Jan–Sep 2025), and if that fails — ToolError;
    same length but different months («январь и июнь») — computed, comparable_note warns about seasonality.
    Base not in the data (before 01.2024) — (gcur, None)."""
    lo, _ = S.data_bounds()
    raw = a.get("base_period")
    base = None
    if raw:
        try:
            base = S.parse_period(raw)
        except Exception:
            raise ToolError(f"Не понял базовый период «{raw}». Формат: 2025-01..2025-09, 2025-09, 2025.")
    cur = period
    if not explicit:
        nxt = S.clip(base.shift(12)) if base else None
        cur = nxt or S.ytd()
    if base is None or base == cur:
        b = cur.shift(-12)
        return (cur, b) if b.start >= lo else (cur, None)
    if base.start > cur.start:          # base later than the period: order swapped
        cur, base = S.clip(base) or base, cur
    if base.n != cur.n:
        k = cur.start.year - base.start.year
        aligned = cur.shift(-12 * k) if k >= 1 else None
        if not aligned or aligned.start < base.start or aligned.end > base.end:
            raise ToolError(f"Периоды разной длины: {cur.label()} ({cur.n} мес.) и {base.label()} ({base.n} мес.). "
                            f"Сравнивать нужно одинаковые месяцы, например {cur.vs(cur.shift(-12))}.")
        base = aligned
    if base.start < lo:
        return cur, None
    return cur, base


def _periods(a, default):
    """Sales/share period and the growth period pair. If the explicit base is later than the period (order swapped),
    the share period is the later of the two."""
    period, explicit = main_period(a, default)
    gcur, gbase = growth_pair(a, period, explicit)
    if explicit and a.get("base_period") and gcur.start > period.start:
        period = gcur
    return period, explicit, gcur, gbase


def base_note(a, gcur, gbase):
    """Caveat when the explicit base was aligned to the same months or the periods were swapped."""
    raw = a.get("base_period")
    if not raw or not gbase:
        return None
    try:
        rb = S.parse_period(raw)
    except Exception:
        return None
    if rb == gcur:
        return f"Периоды переставлены: текущий период более поздний ({gcur.label()}), база {gbase.label()}."
    if rb != gbase:
        return f"База {rb.label()} взята за те же месяцы, что и текущий период: {gbase.label()}."
    return None


def no_base_reason(gcur, gap=None):
    lo, _ = S.data_bounds()
    if gap:
        return "; ".join(f"{n} есть в данных не за все месяцы сравнения (месяцы сети в данных: {seg})" for n, seg in gap)
    return f"базы за {gcur.shift(-12).label()} нет в данных (данные с {lo:%m.%Y})"


def agg(scope, period, levels, measure="rub", comparable_only=False, where=None, geo=False, growth=False):
    """Sales sum by levels. growth=True — one side of a year-over-year comparison: incomplete network months
    are excluded on both sides (S.growth_exclusions). For a period pair growth_agg is more precise."""
    geo = geo or any(l in ("region", "city") for l in levels)
    src = S.source(scope, period, comparable_only, geo=geo, growth=growth)
    sel = ", ".join(f"{LEVEL[l]} AS {l}" for l in levels)
    grp = ", ".join(LEVEL[l] for l in levels)
    sql = f"""SELECT {sel + ',' if sel else ''} sum(n.rub) rub, sum(n.qty) qty, sum(n.kg) kg, sum(n.cost) cost,
                     sum(n.stores) pairs, count(DISTINCT n.month) n_months
              FROM {src} n JOIN sku k USING (sku_id)
              {('WHERE ' + where) if where else ''}
              {('GROUP BY ' + grp) if grp else ''}"""
    df = S.q(sql)
    df["v"] = df[measure].astype(float).fillna(0.0)
    return df


def _cmp_where():
    return "n.network IN (" + ", ".join(S._lit(x) for x in S.COMPARABLE) + ")"


def growth_agg(scope, gcur, gbase, levels, measure="rub", cmp_only=True, where=None):
    """Both sides of the gcur vs gbase comparison: (g1, g0) by levels, totals are g1.v.sum(), g0.v.sum().
    cmp_only — only the 9 comparable networks. A month loaded incompletely for a network (S.bad_pairs) is removed
    for that network in it and in the paired month on the other side. This compares identical months of the same
    networks; the shared S.growth_exclusions set is asymmetric for pairs like 2025 vs 2024 (09.2025 dropped, 09.2024 not)."""
    bad = S.bad_pairs()
    ex1, ex0 = [], []
    for c, b in zip(gcur.months, gbase.months):
        for n in sorted({n for n, x in bad if x in (c, b)}):
            ex1.append((n, c))
            ex0.append((n, b))

    def side(per, ex):
        parts = [f"({where})"] if where else []
        if cmp_only:
            parts.append(_cmp_where())
        parts += [f"NOT (n.network = {S._lit(n)} AND n.month = DATE '{m}')" for n, m in ex]
        return agg(scope, per, levels, measure, where=" AND ".join(parts) or None)

    return side(gcur, ex1), side(gbase, ex0)


def dropped_pairs(scope, gcur, gbase):
    """Slice networks whose months were dropped from the comparison as incomplete: [(network, current-period month)]."""
    nets = set(S.scope_networks(scope) or S.COMPARABLE)
    bad = S.bad_pairs()
    return [(n, c) for c, b in zip(gcur.months, gbase.months) for n, x in sorted(bad) if x in (c, b) and n in nets]


# ——— network coverage ———

def net_months():
    """Network -> months in which the network is present in the data (stores with sales; no product filters)."""
    return _cached("net_months", lambda: {n: sorted(d.month) for n, d in S.coverage().groupby("network")})


def month_segments(months):
    """[date] -> «03–12.2024, 08–09.2025, 02–03.2026»: consecutive months merged into one range."""
    runs = []
    for d in sorted(months):
        if runs and S.add_months(runs[-1][-1], 1) == d:
            runs[-1].append(d)
        else:
            runs.append([d])

    def seg(r):
        x, y = r[0], r[-1]
        if x == y:
            return f"{x:%m.%Y}"
        return f"{x:%m}–{y:%m.%Y}" if x.year == y.year else f"{x:%m.%Y}–{y:%m.%Y}"
    return ", ".join(seg(r) for r in runs)


def partial_networks(scope, period):
    """Networks with incomplete coverage in the selection: [(network, network months in data for the period, slice sales)].
    Incompleteness is judged by the network's presence in the data, not by the object's sales: a niche brand may simply not sell."""
    df = S.q(f"SELECT network, sum(rub) rub FROM {S.source(scope, period)} n GROUP BY 1")
    nm = net_months()
    out = []
    for x in df.itertuples():
        k = sum(1 for d in nm.get(x.network, []) if period.start <= d <= period.end)
        if x.rub and x.rub > 0 and (k < period.n or x.network not in S.COMPARABLE):
            out.append((S.net_label(x.network), k, x.rub))
    return sorted(out, key=lambda t: -t[2])


def coverage_gap(scope, gcur, gbase):
    """Selected networks outside the 9 comparable ones that are missing for part of the comparison months: [(network, month ranges)].
    Growth is not computed for them (methodology §3.2, §3.4): otherwise different months would be compared."""
    if not gbase or not scope.has_network_filter():
        return []
    need = set(gcur.months) | set(gbase.months)
    nm = net_months()
    return [(S.net_label(n), month_segments(nm.get(n, []))) for n in (S.scope_networks(scope) or [])
            if n not in S.COMPARABLE and not need <= set(nm.get(n, []))]


def _covered(network, gcur, gbase):
    return set(gcur.months) | set(gbase.months) <= set(net_months().get(network, []))


def mayak_other_note(scope, level=None, by=None):
    """Caveat about the «Маяк прочие» banner in breakdowns and filters by Granit banners.
    The banner is fixed by the methodology (§5) and is not relabelled; this only warns that its growth is not comparable."""
    banners = set(scope.values("banner"))
    nets = S.scope_networks(scope) or []
    granit = not scope.has_network_filter() or "Гранит" in nets
    if not (banners & {"Маяк прочие", "Маяк у дома"} or ("banner" in (level, by) and granit)):
        return None
    df = _cached("mayak_other", lambda: S.q(
        """SELECT ss.month, count(DISTINCT s.store_key) st FROM store_sub ss JOIN store_x s USING (store_id)
           WHERE s.banner = 'Маяк прочие' AND ss.qty > 0 GROUP BY 1 ORDER BY 1"""))
    if df.empty:
        return None
    last = df.iloc[-1]
    return (f"«Маяк прочие»: точки Гранита без кода формата сети, появились с {df.month.min():%m.%Y}, "
            f"в {last.month:%m.%Y} их {int(last.st)}. Число таких точек быстро растет, поэтому рост баннера "
            "«Маяк прочие» и Маяк у дома несопоставим (новые точки без кода по профилю похожи на магазины у дома). "
            "Точки не переносились в Маяк у дома: баннер оставлен отдельным по методике.")


# ——— labels ———

def _label(level, v):
    if level == "network":
        return S.net_label(v)
    if level == "month" and hasattr(v, "month"):
        return f"{S.MONTHS_RU[v.month - 1]} {v.year}"
    return v


def _blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() in BLANK


def _fdelta(v, m):
    """Signed change: «+1,2 млн шт.», «−0,81 млрд руб.»."""
    return ("+" if v and v > 0 else "") + S.fmoney(v or 0.0, m)


def _dyn(cur, prev, m, total_prev=None):
    """Growth in words for facts: «рост на 3,5%»; small base — a label instead of a percentage (S.fsmall)."""
    cur = 0.0 if cur is None or cur != cur else float(cur)
    prev = 0.0 if prev is None or prev != prev else float(prev)
    if prev <= 0 and cur <= 0:
        return "продаж нет ни в одном из периодов"
    if cur <= 0 < prev:
        return f"продаж в текущем периоде нет (в базовом {S.fmoney(prev, m)})"
    if S.small_base(cur, prev, total_prev):
        return S.fsmall(cur, prev, m)
    return S.fdyn(S.growth(cur, prev))


def _g(cur, prev):
    """Growth for tables: None for a small base (the label goes to the «Примечание» column)."""
    return None if S.small_base(cur, prev) else S.growth(cur, prev)


def _obj_name(scope):
    """Object filter as one string for the share label: «ESTRA», «NORD», «SKU «Объем»»."""
    parts = [str(x) for k in ("manufacturer", "brand", "subbrand") for x in scope.values(k)]
    parts += [f"SKU «{x}»" for x in scope.values("sku")]
    return ", ".join(parts)


def _share_lbl(scope):
    """«доля в выборке» (share of selection) without an object filter; with one — «доля внутри ESTRA»: not a market share."""
    return f"доля внутри {_obj_name(scope)}" if scope.has_object() else "доля в выборке"


def _place(s):
    """Descending rank: ties share a rank (1, 2, 2, 4); no sales — no rank."""
    return s.where(s > 0).rank(ascending=False, method="min")


def _stable(df, col, level, ascending=False):
    """Stable sort: ties broken by name so that ranks do not drift between calls."""
    return df.sort_values([col, level], ascending=[ascending, True], kind="mergesort")


def _labels(scope, period, level, names):
    """Manufacturer (and brand for SKUs) from slice sales for the period: labels with at least 1% of the object's sales, descending.
    The full dictionary is unsuitable: it links L'AURELLE PARIS to NTS GRADIENT on negligible sales."""
    if level not in ("brand", "subbrand", "sku") or not names:
        return {}
    col = LEVEL[level]
    df = S.q(f"""SELECT {col} AS name, k.brand, k.manufacturer, sum(n.rub) v FROM {S.source(scope, period)} n
                 JOIN sku k USING (sku_id) WHERE {col} IN ({', '.join(S._lit(x) for x in names)}) GROUP BY 1, 2, 3""")
    out = {}
    for nm, d in df.groupby("name"):
        tot = d.v.sum()
        mans = d.groupby("manufacturer").v.sum().sort_values(ascending=False)
        man = [x for x, v in mans.items() if tot <= 0 or v >= 0.01 * tot] or list(mans.index[:1])
        brands = d.groupby("brand").v.sum().sort_values(ascending=False)
        out[nm] = {"man": man, "brand": list(brands.index[:1])}
    return out


def _is_stm(level, name, info):
    """Network private label (СТМ): manufacturer STM(...) or a brand with (PL) (methodology §6.4)."""
    if level == "manufacturer":
        return str(name).startswith("STM(")
    if level not in ("brand", "subbrand", "sku"):
        return False
    i = info.get(name) or {}
    brands = [name] if level == "brand" else i.get("brand", [])
    return any(str(x).startswith("STM(") for x in i.get("man", [])) or any("(PL)" in str(b) for b in brands)


def _man_text(level, name, info, scope):
    i = info.get(name)
    if not i:
        return ""
    if level in ("brand", "subbrand"):
        return ", ".join(i["man"])
    if level == "sku" and not scope.values("brand"):    # brand is set by a filter — do not repeat it on every row
        return f"{', '.join(i['brand'])} / {', '.join(i['man'])}"
    return ""


def comparable_note(scope, gcur, gbase):
    same = [d.month for d in gcur.months] == [d.month for d in gbase.months]
    months = "одинаковые месяцы" if same else "разные месяцы одной длины: на сравнение влияет сезонность"
    if scope.has_network_filter():
        return f"Динамика: {gcur.vs(gbase)}, {months}, по выбранным сетям: {net_lbl(scope)}."
    nets = [S.net_label(n) for n in S.COMPARABLE if n not in scope.excluded("network")]
    return f"Динамика: {gcur.vs(gbase)}, {months}, только сети с полным покрытием ({', '.join(nets)})."


def _flat(scope):
    """Slice back into flat tool arguments (for next-step buttons)."""
    out = {k: (v if len(v) > 1 else v[0]) for k, v in scope.f.items() if v}
    if scope.ex:
        out["exclude"] = {k: v for k, v in scope.ex.items() if v}
    if scope.portal:
        out["portal"] = True
    if scope.kids is not None:
        out["kids"] = scope.kids
    if scope.exclude_pl:
        out["exclude_pl"] = True
    return out


def _next_args(scope, m, a, **kw):
    out = _flat(scope)
    out["measure"] = m
    for k in ("period", "base_period"):
        if a.get(k):
            out[k] = a[k]
    out.update(kw)
    return {k: v for k, v in out.items() if v is not None}


def _share_arg(v, default=0.001):
    if v in (None, ""):
        return default
    s = str(v).strip().replace(",", ".")
    try:
        x = float(s.rstrip("%")) / (100 if s.endswith("%") else 1)
    except ValueError:
        raise ToolError("min_share — доля в обоих периодах, например 0.001 (это 0,1%).")
    if not 0 <= x < 1:
        raise ToolError("min_share — доля от 0 до 1, например 0.001 (это 0,1%).")
    return x


def kids_split(scope, period, measure, focus_keys=("manufacturer", "brand", "subbrand", "sku"), base=None):
    """Rule §2.4: for each category with a kids' counterpart — rows «взрослые / детские / итог» (adult / kids / total).
    share — share for the period across all networks; dshare — share change period vs base on one comparison base
    (9 comparable networks or the selected networks, without incomplete months)."""
    market = scope.without(*focus_keys)
    has_focus = any(scope.values(k) for k in focus_keys)
    lv = ["subgroup_base", "subgroup"]
    m1 = agg(market, period, lv, measure)
    o1 = agg(scope, period, lv, measure) if has_focus else None
    m1c = m0c = o1c = o0c = None
    if base and has_focus and base.n == period.n:
        cmp = not scope.has_network_filter()
        m1c, m0c = growth_agg(market, period, base, lv, measure, cmp)
        o1c, o0c = growth_agg(scope, period, base, lv, measure, cmp)
    # only categories that have both an adult and a kids' subgroup
    if has_focus:
        bases = o1.groupby("subgroup_base").v.sum()
        bases = bases[bases > 0.01 * bases.sum()].index
    else:
        bases = m1.subgroup_base.dropna().unique()
    rows = []
    for b in bases:
        sub = m1[m1.subgroup_base == b]
        kids = sub.subgroup.str.contains("етск", na=False)
        if not kids.any() or kids.all():
            continue

        def val(df, mask):
            if df is None:
                return None
            d = df[(df.subgroup_base == b)]
            if mask == "kids":
                d = d[d.subgroup.str.contains("етск", na=False)]
            elif mask == "adult":
                d = d[~d.subgroup.str.contains("етск", na=False)]
            return float(d.v.sum())

        for kind, name in (("adult", "взрослые"), ("kids", "детские"), ("all", "итого")):
            r = {"cat": f"{b}: {name}", "market": val(m1, kind)}
            if has_focus:
                r["obj"] = val(o1, kind)
                r["share"] = r["obj"] / r["market"] if r["market"] else None
                if m1c is not None:
                    mk1, mk0, ob1, ob0 = val(m1c, kind), val(m0c, kind), val(o1c, kind), val(o0c, kind)
                    s1 = ob1 / mk1 if mk1 else None
                    r["share0"] = ob0 / mk0 if mk0 else None
                    r["dshare"] = (s1 - r["share0"]) if (s1 is not None and r["share0"] is not None) else None
            if kind == "all":
                r["_cls"] = "tot"
            rows.append(r)
    return rows


def add_kids_table(r, scope, period, measure, base=None, title=None, base_label=None):
    """base_label — base label for facts («к июлю 2026»); defaults to «к прошлому году» (vs last year)."""
    rows = kids_split(scope, period, measure, base=base)
    if not rows:
        return
    unit_fmt, unit = UNIT[measure]
    has_focus = "obj" in rows[0]
    cols = [("cat", "Категория", "text"), ("market", f"Рынок, {unit}", unit_fmt)]
    if has_focus:
        cols += [("obj", f"Объект, {unit}", unit_fmt), ("share", "Доля", "pct2")]
        if base:
            cols += [("dshare", "Изм. доли", "pp")]
    r.table(title or f"Взрослые и детские подгруппы, {period.label()}", cols, rows)
    if has_focus:
        for x in rows:
            if x.get("share") is None:
                continue
            s = f"{x['cat']}: рынок {S.fmoney(x['market'], measure)}, объект {S.fmoney(x['obj'], measure)}, доля {S.fpct(x['share'], 2)}"
            if x.get("dshare") is not None:
                s += f" ({S.fpp(x['dshare'])} {base_label or 'к прошлому году'})"
            r.fact(s)
        return
    for i in range(0, len(rows), 3):           # no object: one row per category
        ad, kd, tot = rows[i:i + 3]
        cat = tot["cat"].rsplit(":", 1)[0]
        r.fact(f"{cat}, {period.label()}: взрослые {S.fmoney(ad['market'], measure)}, детские {S.fmoney(kd['market'], measure)}"
               + (f" ({S.fpct(kd['market'] / tot['market'])} категории)" if tot["market"] else "")
               + f", итого {S.fmoney(tot['market'], measure)}.")


# ——— tools ———

def _similar(q, name):
    """Query-to-name similarity: the query words are consecutive prefixes of the name's words (or of its bracketed part),
    or the difflib ratio is at least 0.8. «Олвиа» is not similar to «НЕГА»; SWANLY is similar to «СВОНЛИ [SWANLY]»."""
    qw = _words(q)
    if not qw:
        return False
    for part in _brand_parts(name) | {E.norm(name).strip()}:
        pw = part.split()
        for i in range(len(pw) - len(qw) + 1):
            if all(pw[i + j].startswith(qw[j]) and (len(qw[j]) >= 3 or pw[i + j] == qw[j]) for j in range(len(qw))):
                return True
        if difflib.SequenceMatcher(None, " ".join(qw), part).ratio() >= 0.8:
            return True
    return False


def t_find_entities(a):
    """Look up exact names in the dictionaries. Candidates are similar ones only (query words in the name or similarity >= 0.8)."""
    text = str(a.get("query") or a.get("text") or "").strip()
    kind = a.get("kind")
    d = E.dictionary()
    lo, hi = S.data_bounds()
    r = Result("find_entities", f"Поиск в справочниках: «{text}»")
    r.set_slice(S.Scope(universe=False), S.Period(lo, hi), note="поиск названий; продажи кандидатов за весь период данных")
    found = d.find(text)
    out = []
    if kind in (None, "brand"):
        cands = sorted([(b, m, rub) for b, m, rub in d.brands if _similar(text, b)], key=lambda x: -(x[2] or 0))[:15]
        out += [{"type": "бренд", "name": b, "extra": m, "rub": rub} for b, m, rub in cands]
    if kind in (None, "manufacturer"):
        cands = sorted([(m, rub) for m, rub in d.manufacturers if _similar(text, m)], key=lambda x: -(x[1] or 0))[:10]
        out += [{"type": "производитель", "name": m, "extra": "", "rub": rub} for m, rub in cands]
    if kind in (None, "sku") and text:
        df = S.q(f"""SELECT k.sku_name, any_value(k.brand) brand, sum(n.rub) rub FROM net_sku n JOIN sku k USING (sku_id)
                    WHERE k.sku_name ILIKE {S._lit('%' + text + '%')} GROUP BY 1 ORDER BY rub DESC LIMIT 10""")
        out += [{"type": "SKU", "name": x.sku_name, "extra": x.brand, "rub": x.rub} for x in df.itertuples()]
    r.subtitle = "распознано: " + found.summary()
    r.table("Найдено", [("type", "Тип", "text"), ("name", "Название", "text"), ("extra", "Производитель / бренд", "text"),
                        ("rub", "Продажи с 2024, млн руб.", "mln")], out)
    r.fact("Распознано в тексте: " + found.summary())
    if not out:
        r.fact(f"Похожих названий не найдено (порог похожести 0,8): «{text}».", key=True)
    for i, x in enumerate(out[:20]):
        r.fact(f"{x['type']}: {x['name']}" + (f" ({x['extra']})" if x["extra"] else "")
               + f", продажи с {lo:%m.%Y} по {hi:%m.%Y}: {S.fmoney(x['rub'] or 0.0)}", key=i < 3)
    b = next((x["name"] for x in out if x["type"] == "бренд"), None)
    mn = next((x["name"] for x in out if x["type"] == "производитель"), None)
    if b:
        r.suggest(f"Обзор бренда {b}", "profile", {"brand": b})
        r.suggest(f"Топ SKU бренда {b}", "ranking", {"level": "sku", "brand": b})
    if mn:
        r.suggest(f"Обзор производителя {mn}", "profile", {"manufacturer": mn})
        r.suggest(f"Бренды производителя {mn}", "ranking", {"level": "brand", "manufacturer": mn})
    return r


def t_data_info(a):
    """What the data is: period, volume, networks, categories, measures."""
    r = Result("data_info", "Какие данные есть у агента")
    lo, hi = S.data_bounds()
    mat = S.mat()

    def load():
        rows = S.q1("SELECT count(*) FROM fact")[0]
        cnt = S.q1("""SELECT count(DISTINCT manufacturer) FILTER (WHERE manufacturer NOT IN ('.', 'не указан')),
                             count(DISTINCT brand) FILTER (WHERE brand NOT IN ('.', 'не указан')),
                             count(DISTINCT sku_name), count(DISTINCT subgroup_base), count(DISTINCT subgroup_name),
                             count(DISTINCT product_type) FROM sku""")
        act = S.q1(" UNION ALL ".join(
            f"""SELECT count(*) FROM (SELECT k.{c} FROM net_sku n JOIN sku k USING (sku_id)
                                     WHERE {mat.sql('n.month')} AND k.{c} NOT IN ('.', 'не указан')
                                     GROUP BY 1 HAVING sum(n.rub) > 0)""" for c in ("brand", "manufacturer")).join(["SELECT list(c) FROM (", ") t(c)"]))[0]
        geo = S.q1("""SELECT count(DISTINCT store_key), count(*), count(DISTINCT region),
                             count(DISTINCT city_eff) FILTER (WHERE city_eff NOT LIKE 'город не указан%'),
                             count(DISTINCT format), count(DISTINCT district) FILTER (WHERE district <> 'Нет') FROM store_x""")
        nets = S.q(f"""SELECT network, sum(rub) FILTER (WHERE {mat.sql()}) rub12 FROM net_sku GROUP BY 1
                       ORDER BY rub12 DESC NULLS LAST""")
        st = S.q(f"""SELECT s.network, count(DISTINCT s.store_key) stores FROM store_sub ss JOIN store_x s USING (store_id)
                     WHERE ss.month = DATE '{hi}' AND ss.qty > 0 GROUP BY 1""")
        nets = nets.merge(st, on="network", how="left")
        subs = S.q(f"""SELECT k.group_name, k.subgroup_name, sum(n.rub) rub FROM net_sku n JOIN sku k USING (sku_id)
                       WHERE {mat.sql('n.month')} GROUP BY 1, 2 ORDER BY rub DESC""")
        return rows, cnt, act, geo, nets, subs
    rows, cnt, act, geo, nets, subs = _cached("data_info", load)
    r.set_slice(S.Scope(universe=False), S.Period(lo, hi), note="описание данных")
    n_months = (hi.year - lo.year) * 12 + hi.month - lo.month + 1
    nm = net_months()
    r.subtitle = f"{lo:%m.%Y}–{hi:%m.%Y}, помесячно"
    tot12 = nets.rub12.sum()
    r.kpis([("Период", f"{lo:%m.%Y}–{hi:%m.%Y}", f"{n_months} мес.", None),
            ("Строк факта", f"{rows / 1e6:,.0f} млн".replace(",", " "), "месяц × точка × SKU", None),
            ("Продажи за 12 мес.", S.fmoney(tot12), mat.label(), None),
            ("Брендов с продажами за 12 мес.", f"{act[0]:,}".replace(",", " "), f"в справочнике {cnt[1]:,}".replace(",", " "), None),
            ("Точек", f"{geo[0]:,}".replace(",", " "), f"кодов точек {geo[1]:,}".replace(",", " "), None)])
    nrows = []
    for x in nets.itertuples():
        months = nm.get(x.network, [])
        nrows.append({"net": S.net_label(x.network), "months": len(months), "span": month_segments(months),
                      "share": (x.rub12 or 0) / tot12 if tot12 else None,
                      "stores": None if pd.isna(x.stores) else int(x.stores),
                      "cmp": "да" if x.network in S.COMPARABLE else "нет"})
    r.table("Сети", [("net", "Сеть", "text"), ("months", "Месяцев в данных", "int"), ("span", "Месяцы", "text"),
                     ("share", "Доля продаж за 12 мес.", "pct"), ("stores", f"Точек в {hi:%m.%Y}", "int"),
                     ("cmp", "Сопоставимая", "text")], nrows)
    grp = subs.groupby("group_name").rub.sum().sort_values(ascending=False)
    srows = [{"grp": g, "subs": ", ".join(subs[subs.group_name == g].subgroup_name.head(8)),
              "share": v / grp.sum()} for g, v in grp.items() if v > 0]
    r.table("Товарные группы и подгруппы", [("grp", "Группа", "text"), ("subs", "Крупнейшие подгруппы", "txt"),
                                            ("share", "Доля продаж", "pct")], srows)
    r.fact(f"Данные: продажи сетей (sell-out), {lo:%m.%Y}–{hi:%m.%Y} ({n_months} мес.), помесячно; гранула месяц × точка × SKU, "
           f"{rows / 1e6:.0f} млн строк.", key=True)
    r.fact(f"Охват: сетей — {len(nm)} (9 с полным покрытием для динамики), форматов — {geo[4]} (ММ, СМ, ГМ, интернет-магазин, РЦ), "
           f"федеральных округов — {geo[5]}, регионов — {geo[2]}, городов — {geo[3]}, точек — {geo[0]} (кодов точек {geo[1]}).", key=True)
    r.fact(f"Брендов с продажами за 12 мес. ({mat.label()}): {act[0]} (в справочнике {cnt[1]}); "
           f"производителей с продажами за 12 мес.: {act[1]} (в справочнике {cnt[0]}); SKU по названию: {cnt[2]}.", key=True)
    r.fact(f"Товары: базовых подгрупп — {cnt[3]} ({cnt[4]} с детскими вариантами), типов товара — {cnt[5]}.")
    r.fact(f"По умолчанию анализ идет по категориям NORD — подгруппам, где у NORD есть товары ({len(S.own_categories())}): "
           + ", ".join(S.own_categories()) + ". Другие категории (гигиена медицинская, полость рта, парфюмерия и др.) "
             "считаются, только если назвать их в вопросе или попросить всю панель.", key=True)
    r.fact(f"Продажи за 12 мес. ({mat.label()}), все сети: {S.fmoney(tot12)}.")
    r.fact("Показатели: продажи в рублях, в штуках, в кг/л, закупочная стоимость сети (себестоимость продаж для сети) в рублях.")
    r.fact("Сети с полным покрытием (для динамики): " + ", ".join(S.net_label(n) for n in S.COMPARABLE) + ".")
    part = [f"{x['net']} ({x['span']})" for x in nrows if x["cmp"] == "нет"]
    r.fact("Сети с неполным покрытием и их месяцы в данных: " + "; ".join(part) + ".")
    cov = S.coverage()
    badm = [f"{S.net_label(x.network)} {x.month:%m.%Y} (точек с продажами {x.st} при обычных ~{int(round(x.med_st))})"
            for x in cov[~cov.ok].itertuples()]
    if badm:
        r.fact("Месяцы, загруженные поставщиком не полностью (в сравнениях с прошлым годом исключаются): " + "; ".join(badm) + ".")
    r.fact("Чего нет: дней и недель, промо-признака, остатков, цен на полке, координат точек, связи с нашей номенклатурой 1С.")
    r.fact("Что можно посчитать: доли и рейтинги, динамику, сезонность, дистрибуцию по точкам, цены продажи и закупки сети и наценку, ABC, факторный анализ, ротацию ассортимента, простой прогноз.")
    r.fact("Представленность (дистрибуция) есть: по каждой точке и месяцу видно, продавался ли SKU; можно посчитать число точек с продажами, "
           "долю точек сети, где продается бренд или производитель, по сетям, баннерам Маяка, форматам и регионам, и потенциал: точки с категорией, но без бренда. "
           "Остатков на полке нет: точка без продаж не равна точке без товара.")
    r.suggest("Доли сетей в продажах за 12 месяцев", "ranking", {"level": "network"})
    r.suggest("Топ-10 производителей в категории Косметика", "ranking", {"level": "manufacturer", "portal": True})
    r.suggest("Динамика продаж по месяцам", "trend", {})
    return r


def t_ranking(a):
    level = a.get("level") or "brand"
    if level not in LEVEL or level == "month":
        raise ToolError(f"level должен быть одним из: {', '.join(k for k in LEVEL if k != 'month')}")
    by = a.get("by")
    if by and (by not in LEVEL or by in (level, "month")):
        by = None
    sort = str(a.get("sort") or "sales").lower()
    if sort not in SORTS:
        raise ToolError(f"sort должен быть одним из: {', '.join(SORTS)}.")
    min_share = _share_arg(a.get("min_share"))
    m = get_measure(a)
    top = max(1, int(a.get("top") or 10))
    focus = a.get("focus")
    if isinstance(focus, list):
        focus = focus[0] if focus else None
    if focus:
        # the comparison object must not narrow the ranking: «на уровне NORD» ("at NORD's level") — manufacturer ranking without a NORD filter
        for key in ("manufacturer", "brand"):
            vals = a.get(key) if isinstance(a.get(key), list) else [a.get(key)]
            if any(v and _norm_simple(v) == _norm_simple(focus) for v in vals):
                a = {k: v for k, v in a.items() if k != key}
                if level not in ("manufacturer", "brand"):
                    level = key
                elif level != key and key == "manufacturer":
                    level = "manufacturer"
    scope = make_scope(a)
    fval = _focus_value(level, focus) if focus else None
    period, explicit, gcur, gbase = _periods(a, "mat")
    gap = coverage_gap(scope, gcur, gbase)
    if gap:
        gbase = None
    if sort != "sales" and not gbase:
        raise ToolError(f"Рейтинг {SORTS[sort]} требует базы год назад, а ее нет: {no_base_reason(gcur, gap)}.")
    cmp_only = not scope.has_network_filter()
    unit_fmt, unit = UNIT[m]
    thr = f"доля не меньше {S.fpct(min_share)} в обоих периодах" if sort in ("growth", "decline", "share_change") else ""
    if focus:
        title = f"{LEVEL_RU[level]}: соседи {focus} по рейтингу {SORTS[sort]} {MEASURE_RU[m]}"
    else:
        title = f"Топ-{top} {LEVEL_GEN[level]} {SORTS[sort]} {MEASURE_RU[m]}" + (f" ({thr})" if thr else "")
    r = Result("ranking", title)
    r.subtitle = f"{scope.describe()}; {period.label()}"
    note = ", ".join(x for x in ((SORTS[sort] + (f", {thr}" if thr else "")) if sort != "sales" else "",
                                 f"соседи по рейтингу: {focus}" if focus else "") if x) or None
    r.set_slice(scope, period, gcur if gbase else None, gbase, m, cmp=cmp_only, note=note)

    keys = [level] + ([by] if by else [])
    cur = agg(scope, period, keys, m)
    if cur.empty or not (cur.v > 0).any():
        raise ToolError(f"В выборке ({scope.describe()}) нет продаж за {period.label()}.")
    T = float(cur.v.sum())
    df = cur[keys + ["v"]]
    T1 = T0 = 0.0
    if gbase:
        g1, g0 = growth_agg(scope, gcur, gbase, keys, m, cmp_only)
        T1, T0 = float(g1.v.sum()), float(g0.v.sum())
        g = g1[keys + ["v"]].merge(g0[keys + ["v"]], on=keys, how="outer", suffixes=("1", "0"))
        df = df.merge(g, on=keys, how="outer")
        df[["v", "v1", "v0"]] = df[["v", "v1", "v0"]].fillna(0.0)
    blank = df[level].map(_blank)
    blank_v = float(df.loc[blank, "v"].sum())
    obj = df[~blank]

    groups, gt = [None], None
    if by:
        gt = cur.groupby(by).v.sum()
        gt = gt[[not _blank(x) for x in gt.index]]
        gt = gt[gt > 0].sort_values(ascending=False)
        groups = list(gt.index[:12])
        G1 = g1.groupby(by).v.sum() if gbase else None
        G0 = g0.groupby(by).v.sum() if gbase else None
    blocks = []
    for gname in groups:
        part = (obj if gname is None else obj[obj[by] == gname]).copy()
        total = T if gname is None else float(gt[gname])
        t1 = T1 if gname is None or not gbase else float(G1.get(gname, 0.0))
        t0 = T0 if gname is None or not gbase else float(G0.get(gname, 0.0))
        part["share"] = part.v / total if total else np.nan
        if gbase:
            part["s1"] = part.v1 / t1 if t1 > 0 else np.nan
            part["s0"] = part.v0 / t0 if t0 > 0 else np.nan
            part["dsh"] = part.s1 - part.s0
            part["r1"] = _place(part.v1)
            part["r0"] = _place(part.v0)
            part["gr"] = part.v1 / part.v0.where(part.v0 > 0) - 1
            part["d"] = part.v1 - part.v0
        n_obj = int((part.v > 0).sum())
        n_any = int(((part.v1 > 0) | (part.v0 > 0)).sum()) if gbase else n_obj
        if sort == "sales":
            pool, keycol, asc = part[part.v > 0], "v", False
        elif sort == "gain":
            pool, keycol, asc = part[(part.v1 > 0) | (part.v0 > 0)], "d", False
        else:
            pool = part[(part.v0 > 0) & (part.v1 >= min_share * t1) & (part.v0 >= min_share * t0)] if t1 > 0 and t0 > 0 else part.iloc[0:0]
            keycol, asc = {"growth": ("gr", False), "decline": ("gr", True), "share_change": ("dsh", False)}[sort]
        ranked = _stable(pool, keycol, level, ascending=asc).reset_index(drop=True)
        ranked["place"] = ranked[keycol].rank(ascending=asc, method="min")
        start, hit = 0, None
        if focus:
            idx = [i for i, x in enumerate(ranked[level]) if x == fval or _norm_simple(x) == _norm_simple(fval)]
            if idx:
                hit = ranked.iloc[idx[0]]
                start = max(0, min(idx[0] - top // 2, len(ranked) - top))
        blocks.append({"gname": gname, "part": part, "ranked": ranked, "shown": ranked.iloc[start:start + top],
                       "total": total, "t1": t1, "t0": t0, "n_obj": n_obj, "n_any": n_any, "start": start, "hit": hit})

    names = set()
    for b in blocks:
        names |= set(b["shown"][level]) | set(b["ranked"][level].head(3))
    info = _labels(scope, S.Period(min(period.start, gbase.start if gbase else period.start), max(period.end, gcur.end)),
                   level, sorted(names))
    lbl = lambda name: f"{_label(level, name)}{' [СТМ]' if _is_stm(level, name, info) else ''}"
    share_lbl = _share_lbl(scope)
    share_scope = "все сети" if cmp_only else net_lbl(scope)
    where = net_lbl(scope)

    def growth_txt(x):
        # give the base and current values of the growth period ready-made: otherwise the model derives them from the percentage
        s = f"; {_dyn(x.v1, x.v0, m)}"
        if x.v0 > 0 and x.v1 > 0:
            s += f" ({S.fmoney(x.v0, m)} → {S.fmoney(x.v1, m)})"
        if sort == "gain":
            s += f", прирост {_fdelta(x.d, m)}"
        if x.dsh == x.dsh:
            s += f", доля {S.fpct(x.s0, 2)} → {S.fpct(x.s1, 2)} ({S.fpp(x.dsh)})" if x.s0 == x.s0 else f", доля {S.fpp(x.dsh)}"
        s += f", место {int(x.r1)}" if x.r1 == x.r1 else f", в {gcur.label()} продаж нет"
        s += f" (год назад {int(x.r0)})" if x.r0 == x.r0 else " (год назад продаж не было)"
        return s

    # market totals and growth over the whole slice: including blank labels and objects that stopped selling
    if gbase:
        r.fact(f"Продажи и доли посчитаны за {period.label()} ({share_scope}); рост, изменение доли и места: "
               f"{gcur.vs(gbase)} ({where}).")
    head = blocks[0] if not by else None
    r.fact(f"Всего в выборке за {period.label()}: {S.fmoney(T, m)}; {LEVEL_GEN[level]} с продажами: "
           f"{head['n_obj'] if head else int((obj.groupby(level).v.sum() > 0).sum())}"
           + (" (SKU считаются по названию)" if level == "sku" else "") + ".", key=True)
    if blank_v:
        r.fact(f"Без метки {LEVEL_GEN1[level]}: {S.fmoney(blank_v, m)} ({S.fpct(blank_v / T)} продаж выборки); "
               "в рейтинг не включено, в итог включено.")
    mk = "Продажи выборки" if scope.has_object() else "Рынок"
    if gbase:
        r.fact(f"{mk} ({scope.describe()}) {gcur.vs(gbase)}: {_dyn(T1, T0, m)} "
               f"({S.fmoney(T0, m)} → {S.fmoney(T1, m)}; {where}).", key=True)
    else:
        r.fact(f"{mk} ({scope.describe()}): рост к прошлому году не считается: {no_base_reason(gcur, gap)}.", key=True)
    if thr:
        r.fact(f"Порог рейтинга {SORTS[sort]}: доля не меньше {S.fpct(min_share)} продаж выборки в обоих периодах "
               f"({gcur.label()} и {gbase.label()}); порог прошли {len(blocks[0]['ranked']) if not by else sum(len(b['ranked']) for b in blocks)} "
               f"из {blocks[0]['n_any'] if not by else sum(b['n_any'] for b in blocks)} {LEVEL_GEN[level]} с продажами хотя бы в одном периоде.")
    elif sort == "gain":
        r.fact(f"Рейтинг по приросту: разница продаж {gcur.vs(gbase)} ({where}), порог доли не применяется; "
               "у объектов без продаж в базе вместо роста стоит «новый в периоде».")

    rows = []
    for b in blocks:
        gname, ranked, shown, total = b["gname"], b["ranked"], b["shown"], b["total"]
        glbl = None if gname is None else str(_label(by, gname))
        sl = share_lbl if gname is None else f"доля в {glbl}"
        pre = "" if gname is None else f"[{glbl}] "
        if gname is not None:
            s = f"[{glbl}] всего {S.fmoney(total, m)}, {LEVEL_GEN[level]} с продажами: {b['n_obj']}"
            if gbase:
                nocov = by == "network" and cmp_only and gname not in S.COMPARABLE
                s += ("; рост не считается: сеть с неполным покрытием (динамика только по 9 сопоставимым сетям)" if nocov
                      else f"; {gcur.vs(gbase)}: {_dyn(b['t1'], b['t0'], m)}")
            if len(shown):
                s += f"; показаны места {b['start'] + 1}–{b['start'] + len(shown)} из {len(ranked)}, на них {S.fpct(shown.v.sum() / total)}"
            r.fact(s + ".")
            rows.append({"name": glbl, "_cls": "lvl", "v": total, "share": 1.0})
        elif focus:
            hit = b["hit"]
            if hit is not None:
                s = (f"{lbl(hit[level])}: {int(hit.place)}-е место из {len(ranked)} {SORTS[sort]}, {S.fmoney(hit.v, m)}, "
                     f"{sl} {S.fpct(hit.share, 2)}")
                r.fact(s + (growth_txt(hit) if gbase else "") + ".", key=True)
            else:
                r.fact(f"«{focus}» нет в рейтинге этой выборки: нет продаж за {period.label()}"
                       + (" или не прошел порог" if thr else "") + ".", key=True)
            lead = ranked.head(3)
            if len(lead):
                r.fact("Лидеры: " + ", ".join(f"{int(x.place)}. {lbl(getattr(x, level))} {S.fmoney(x.v, m)} ({S.fpct(x.share)})"
                                               for x in lead.itertuples())
                       + f"; на первые {len(lead)} приходится {S.fpct(lead.v.sum() / total)} продаж выборки.", key=True)
        elif len(ranked):
            x = ranked.iloc[0]
            if sort == "sales":
                r.fact(f"Лидер: {lbl(x[level])}: {S.fmoney(x.v, m)}, {sl} {S.fpct(x.share)}"
                       + (f"; {gcur.vs(gbase)}: {_dyn(x.v1, x.v0, m)}" if gbase else "") + ".", key=True)
            else:
                r.fact(f"Первое место {SORTS[sort]}: {lbl(x[level])}: {_dyn(x.v1, x.v0, m)} "
                       f"({S.fmoney(x.v0, m)} → {S.fmoney(x.v1, m)}).", key=True)
        if gname is None:
            if len(shown):
                r.fact(f"Показаны места {b['start'] + 1}–{b['start'] + len(shown)} из {len(ranked)}"
                       + (" прошедших порог" if thr else "")
                       + f"; на них приходится {S.fpct(shown.v.sum() / total)} продаж выборки за {period.label()}, "
                       f"на все остальное {S.fpct(1 - shown.v.sum() / total)}.")
            elif thr:
                r.fact("Порог не прошел ни один объект: рейтинг пуст. Можно снизить порог параметром min_share.")
            _shown_aggregates(r, shown, level, lbl, m, gbase, info)
        b["facts"] = []
        for x in shown.itertuples():
            name = getattr(x, level)
            row = {"rank": int(x.place), "name": lbl(name), "man": _man_text(level, name, info, scope),
                   "v": x.v, "share": x.share}
            if gbase:
                row.update(g=_g(x.v1, x.v0), dsh=None if x.dsh != x.dsh else x.dsh,
                           r1=None if x.r1 != x.r1 else int(x.r1), r0=None if x.r0 != x.r0 else int(x.r0), d=x.d)
                if S.small_base(x.v1, x.v0):
                    row["note"] = _dyn(x.v1, x.v0, m)
            rows.append(row)
            b["facts"].append(f"{pre}{int(x.place)}. {row['name']}{' (' + row['man'] + ')' if row['man'] else ''}: "
                              f"{S.fmoney(x.v, m)}, {sl} {S.fpct(x.share)}" + (growth_txt(x) if gbase else ""))
    # ranking rows go last: long lists are truncated from the end; with by, up to 3 rows per group in facts
    for b in blocks:
        for s in (b["facts"] if not by else b["facts"][:3]):
            r.fact(s)

    share_col = "Доля в группе" if by else ("Доля внутри " + _obj_name(scope) if scope.has_object() else "Доля в выборке")
    cols = [("rank", "#", "int"), ("name", LEVEL_RU[level], "text")]
    if level in ("brand", "subbrand"):
        cols.append(("man", "Производитель", "text"))
    if level == "sku" and not scope.values("brand"):
        cols.append(("man", "Бренд / производитель", "text"))
    cols += [("v", f"Продажи, {unit}", unit_fmt), ("share", share_col, "pct")]
    if gbase:
        cols += [("g", f"Рост {gcur.vs(gbase)}", "gpct"), ("dsh", "Изм. доли, п.п.", "pp"),
                 ("r1", f"Место, {gcur.label()}", "int"), ("r0", f"Место, {gbase.label()}", "int")]
        if sort == "gain":
            cols.append(("d", f"Прирост, {unit}", unit_fmt))
        if any(x.get("note") for x in rows):
            cols.append(("note", "Примечание", "text"))
    b0 = blocks[0] if blocks else None
    ttl = (f"{LEVEL_RU[level]}: места {b0['start'] + 1}–{b0['start'] + len(b0['shown'])} из {len(b0['ranked'])}, соседи {focus}"
           if focus and not by and b0 is not None and len(b0["shown"]) else f"{r.title}, {period.label()}")
    r.table(ttl, cols, rows)
    if not by:
        top_rows = [x for x in rows if x.get("rank")][:15]
        if sort == "sales" or not gbase:
            r.chart("bar", f"Доля в продажах, {period.label()}", [x["name"] for x in top_rows],
                    [{"name": "Доля", "values": [x["share"] for x in top_rows]}], fmt="pct")
        else:
            r.chart("bar", f"Рост {gcur.vs(gbase)}", [x["name"] for x in top_rows],
                    [{"name": "Рост", "values": [x.get("g") for x in top_rows]}], fmt="pct")
    if gbase:
        r.caveat(comparable_note(scope, gcur, gbase))
    bn = base_note(a, gcur, gbase)
    if bn:
        r.caveat(bn)
    if gap:
        r.caveat("Рост не считается: " + no_base_reason(gcur, gap) + ".")
    pn = partial_networks(scope, period)
    if pn and not scope.has_network_filter():
        r.caveat("Доли за период посчитаны по всем сетям, включая сети с неполным покрытием: "
                 + ", ".join(f"{n} ({k} из {period.n} мес.)" for n, k, _ in pn[:6]) + ".")
    elif pn:
        r.caveat("Выбранные сети с неполным покрытием: " + ", ".join(f"{n} есть в данных за {k} из {period.n} мес. периода"
                                                                     for n, k, _ in pn) + ".")
    if level in ("manufacturer", "brand"):
        r.caveat("Производитель и бренд — метки поставщика данных; варианты написания не объединялись.")
    mo = mayak_other_note(scope, level, by)
    if mo:
        r.caveat(mo)
    if thr:
        r.caveat(f"Порог {S.fpct(min_share)} доли в обоих периодах — допущение (методика п. 8.3): без него в топ роста "
                 "попадают объекты с базой в единицы штук. Порог задается параметром min_share.")
    if scope.values("subgroup") or scope.portal or level == "subgroup":
        add_kids_table(r, scope, period, m, base=None)
    # next steps
    lead = blocks[0]["ranked"].head(1) if blocks else None
    leader = None if lead is None or lead.empty else lead.iloc[0][level]
    nxt = lambda **kw: _next_args(scope, m, a, **kw)
    if level in ("brand", "manufacturer", "sku", "network", "banner", "product_type", "subgroup"):
        r.suggest(f"Динамика долей топ-5 {LEVEL_GEN[level]} по месяцам", "trend", nxt(level=level, top=5))
    if leader is not None and level in ("brand", "manufacturer", "network", "banner"):
        r.suggest(f"Факторный анализ: {_label(level, leader)}", "factors", nxt(**{level: leader}))
    if level in ("brand", "manufacturer", "sku"):
        r.suggest(f"Цены {LEVEL_GEN[level]} по сетям", "prices", nxt(level=level))
    if leader is not None and level in ("brand", "manufacturer"):
        r.suggest(f"Представленность {leader} по сетям", "distribution", nxt(**{level: leader}, by="network"))
    if sort == "sales" and gbase and not focus:
        r.suggest(f"Топ-{top} {LEVEL_GEN[level]} по росту продаж", "ranking", nxt(level=level, by=by, top=top, sort="growth"))
    return r


def _focus_value(level, focus):
    """Exact value of the focus object at the ranking level (SWANLY -> «СВОНЛИ [SWANLY]»); not found — returned as passed."""
    key = {"subgroup": "subgroup_exact", "subgroup_base": "subgroup"}.get(level, level)
    if key in S.FILTERS and key != "sku":
        try:
            k, vals = resolve_filter(key, focus)
            if k == key:
                return vals[0]
        except ToolError:
            pass
    return focus


def _shown_aggregates(r, shown, level, lbl, m, gbase, info):
    """Aggregates over the shown rows are computed in code: the model must not sort or sum on its own («единственный»,
    «все растут», «максимальный рост»)."""
    if not gbase or len(shown) < 2:
        if len(shown) >= 2 and level in ("brand", "manufacturer", "sku", "subbrand"):
            stm = [lbl(x) for x in shown[level] if _is_stm(level, x, info)]
            r.fact("СТМ среди показанных: " + (", ".join(stm) if stm else "нет") + ".")
        return
    ok = [x for x in shown.itertuples() if not S.small_base(x.v1, x.v0)]
    up = [x for x in ok if round(x.gr * 100, 1) > 0]
    down = [x for x in ok if round(x.gr * 100, 1) < 0]
    flat = len(ok) - len(up) - len(down)
    nb = len(shown) - len(ok)
    s = f"Среди показанных {len(shown)}: растут {len(up)}, снижаются {len(down)}"
    s += (f", без изменений {flat}" if flat else "") + (f", рост не считается (малая база или нет базы) {nb}" if nb else "")
    if down and len(down) <= 5:
        s += ". Снижаются: " + ", ".join(f"{lbl(getattr(x, level))} ({S.fdyn(x.gr)})" for x in down)
    elif up and len(up) <= 5:
        s += ". Растут: " + ", ".join(f"{lbl(getattr(x, level))} ({S.fdyn(x.gr)})" for x in up)
    r.fact(s + ".")
    if len(ok) >= 2:
        hi_ = max(ok, key=lambda x: x.gr)
        lo_ = min(ok, key=lambda x: x.gr)
        r.fact(f"Максимальный рост среди показанных: {lbl(getattr(hi_, level))} ({S.fdyn(hi_.gr)}); "
               f"минимальный: {lbl(getattr(lo_, level))} ({S.fdyn(lo_.gr)}).")
    ds = [x for x in shown.itertuples() if x.dsh == x.dsh]
    if len(ds) >= 2:
        hi_ = max(ds, key=lambda x: x.dsh)
        lo_ = min(ds, key=lambda x: x.dsh)
        # «потеря доли» (share loss) only if the share actually fell; otherwise it is the smallest gain
        tail = (f"наибольшая потеря доли: {lbl(getattr(lo_, level))} ({S.fpp(lo_.dsh)})" if lo_.dsh < 0 else
                f"наименьший прирост доли: {lbl(getattr(lo_, level))} ({S.fpp(lo_.dsh)})")
        head = (f"Наибольший прирост доли среди показанных: {lbl(getattr(hi_, level))} ({S.fpp(hi_.dsh)})" if hi_.dsh > 0 else
                f"Доля снизилась у всех показанных; меньше всего у {lbl(getattr(hi_, level))} ({S.fpp(hi_.dsh)})")
        r.fact(f"{head}; {tail}.")
    if level in ("brand", "manufacturer", "sku", "subbrand"):
        stm = [lbl(x) for x in shown[level] if _is_stm(level, x, info)]
        r.fact("СТМ среди показанных: " + (", ".join(stm) if stm else "нет") + ".")


def t_trend(a):
    level = a.get("level")
    if level and (level not in LEVEL or level == "month"):
        raise ToolError(f"level должен быть одним из: {', '.join(k for k in LEVEL if k != 'month')}")
    scope = make_scope(a)
    m = get_measure(a)
    lo, hi = S.data_bounds()
    period, explicit, gcur, gbase = _periods(a, "all")
    if not explicit:
        period = S.Period(lo, hi)
    gap = coverage_gap(scope, gcur, gbase)
    if gap:
        gbase = None
    top = max(1, int(a.get("top") or 5))
    metric = a.get("metric") or ("share" if level else "sales")
    smooth = max(1, int(a.get("smooth") or 1))
    cmp_only = not scope.has_network_filter()
    r = Result("trend", "Динамика продаж по месяцам" if not level else f"Динамика топ-{top} {LEVEL_GEN[level]}")
    r.subtitle = f"{scope.describe()}; {period.label()}"
    if a.get("seasonality"):
        r.set_slice(scope, S.Period(lo, hi), measure=m, cmp=cmp_only, note="сезонность по полным годам")
        return _seasonality(r, scope, m, cmp_only, a)
    r.set_slice(scope, period, gcur if gbase else None, gbase, m, cmp=cmp_only)
    if gbase:
        r.caveat(comparable_note(scope, gcur, gbase))
        bn = base_note(a, gcur, gbase)
        if bn:
            r.caveat(bn)
    if gap:
        r.caveat("Рост не считается: " + no_base_reason(gcur, gap) + ".")
    if not level:
        _trend_total(r, scope, m, period, gcur, gbase, gap, cmp_only)
    else:
        _trend_level(r, scope, m, period, gcur, gbase, gap, cmp_only, level, top, metric, smooth)
    nxt = lambda **kw: _next_args(scope, m, a, **kw)
    if not level:
        r.suggest("Какие бренды дали изменение", "ranking", nxt(level="brand", sort="gain") if gbase else nxt(level="brand"))
        if gbase:
            r.suggest(f"Факторный анализ: {gcur.vs(gbase)}", "factors",
                      nxt(period=gcur.to_str(), base_period=gbase.to_str()))
        r.suggest("Сезонность продаж", "trend", nxt(seasonality=True, period=None, base_period=None))
        r.suggest("Динамика долей топ-5 брендов", "trend", nxt(level="brand", top=5))
    else:
        r.suggest(f"Рейтинг {LEVEL_GEN[level]} с ростом и долями", "ranking", nxt(level=level))
        r.suggest("Динамика продаж без разбивки", "trend", nxt())
        if gbase:
            r.suggest(f"Топ {LEVEL_GEN[level]} по росту", "ranking", nxt(level=level, sort="growth"))
    return r


def _trend_total(r, scope, m, period, gcur, gbase, gap, cmp_only):
    """Monthly sales: all networks and the comparison base; month-over-same-month-last-year growth and period growth
    use networks complete in both months of the pair (an incomplete network month is removed on both sides)."""
    lo, _ = S.data_bounds()
    unit_fmt, unit = UNIT[m]
    where = net_lbl(scope)
    prev = S.clip(period.shift(-12))
    starts = [period.start] + [x.start for x in (prev, gbase) if x]
    load = S.Period(min(starts), max(period.end, gcur.end))
    d = agg(scope, load, ["month", "network"], m)
    if d.empty or not (d[d.month.isin(period.months)].v != 0).any():
        raise ToolError(f"В выборке ({scope.describe()}) нет продаж за {period.label()}.")
    allv = d.groupby("month").v.sum()
    bn = d[d.network.isin(S.COMPARABLE)] if cmp_only else d
    cmpv = bn.groupby("month").v.sum()
    vals = {(x.network, x.month): x.v for x in bn.itertuples()}
    nets = set(bn.network)
    bad = S.bad_pairs()

    def pair(c, b):
        drop = {n for n, x in bad if x in (c, b)} & nets
        keep = nets - drop
        return sum(vals.get((n, c), 0.0) for n in keep), sum(vals.get((n, b), 0.0) for n in keep), drop

    rows = []
    for mm in period.months:
        pm = S.add_months(mm, -12)
        row = {"mm": mm, "month": _label("month", mm), "v": float(allv.get(mm, 0.0))}
        if cmp_only:
            row["vc"] = float(cmpv.get(mm, 0.0))
        if pm >= lo:
            c, b, drop = pair(mm, pm)
            row.update(c=c, b=b, drop=drop, g=_g(c, b))
            if S.small_base(c, b):
                row["note"] = _dyn(c, b, m)
        rows.append(row)
    tot_all = sum(x["v"] for x in rows)
    if cmp_only:
        r.fact(f"Продажи за {period.label()}: все сети {S.fmoney(tot_all, m)}; "
               f"9 сопоставимых сетей {S.fmoney(sum(x['vc'] for x in rows), m)}.", key=True)
    else:
        r.fact(f"Продажи за {period.label()} ({where}): {S.fmoney(tot_all, m)}.", key=True)
    if gbase:
        T1 = T0 = 0.0
        drops = set()
        for c, b in zip(gcur.months, gbase.months):
            x1, x0, dr = pair(c, b)
            T1, T0 = T1 + x1, T0 + x0
            drops |= {(n, c) for n in dr}
        s = f"{gcur.vs(gbase)} ({where}): {_dyn(T1, T0, m)} ({S.fmoney(T0, m)} → {S.fmoney(T1, m)})"
        if drops:
            s += "; без неполных месяцев: " + ", ".join(f"{S.net_label(n)} {c:%m.%Y}" for n, c in sorted(drops))
        r.fact(s + ".", key=True)
    else:
        r.fact(f"Рост за {gcur.label()} к прошлому году не считается: {no_base_reason(gcur, gap)}.", key=True)
    if any("c" not in x for x in rows):
        r.fact(f"Для месяцев до {S.add_months(lo, 12):%m.%Y} сравнения с прошлым годом нет: данные начинаются с {lo:%m.%Y}.")
    shown = rows[-15:]
    for i, x in enumerate(shown):
        lbl = S.Period(x["mm"], x["mm"]).label()
        s = (f"{lbl}: все сети {S.fmoney(x['v'], m)}; 9 сопоставимых сетей {S.fmoney(x['vc'], m)}" if cmp_only
             else f"{lbl} ({where}): {S.fmoney(x['v'], m)}")
        if "c" in x:
            pm = S.add_months(x["mm"], -12)
            s += f"; к {S.Period(pm, pm).label('dat')}: {_dyn(x['c'], x['b'], m)}"
            if x["drop"]:
                s += f" (без {', '.join(sorted(S.net_label(n) for n in x['drop']))}: месяц загружен не полностью)"
        r.fact(s + ".", key=i == len(shown) - 1)
    cols = [("month", "Месяц", "text"), ("v", f"Все сети, {unit}" if cmp_only else f"{where}, {unit}", unit_fmt)]
    if cmp_only:
        cols.append(("vc", f"9 сопоставимых сетей, {unit}", unit_fmt))
    cols.append(("g", f"К тому же месяцу год назад ({where})", "gpct"))
    if any(x.get("note") for x in rows):
        cols.append(("note", "Примечание", "text"))
    r.table("По месяцам", cols, rows)
    series = cmpv if cmp_only else allv
    years = sorted({x.year for x in period.months})
    r.chart("line", f"Продажи по месяцам, {unit} ({where})", S.MONTHS_RU,
            [{"name": str(y), "values": [series.get(S.date(y, mm, 1)) / DIV[m] if S.date(y, mm, 1) in period.months
                                         and series.get(S.date(y, mm, 1)) is not None else None for mm in range(1, 13)]}
             for y in years], fmt="num1", unit=unit)


def _trend_level(r, scope, m, period, gcur, gbase, gap, cmp_only, level, top, metric, smooth):
    """Top-N objects: selected by sales for the period (all slice networks), monthly shares on the comparison base,
    growth and share change for gcur vs gbase."""
    unit_fmt, unit = UNIT[m]
    where = net_lbl(scope)
    per_all = agg(scope, period, [level], m)
    per = per_all[~per_all[level].map(_blank) & (per_all.v > 0)]
    if per.empty:
        raise ToolError(f"В выборке ({scope.describe()}) нет продаж за {period.label()}.")
    total = float(per_all.v.sum())
    per = _stable(per, "v", level)
    names = list(per[level].head(top))
    pv = dict(zip(per[level], per.v))
    info = _labels(scope, period, level, names)
    lbl = lambda n: f"{_label(level, n)}{' [СТМ]' if _is_stm(level, n, info) else ''}"
    dm = agg(scope, period, ["month", level], m, where=_cmp_where() if cmp_only else None)
    months = period.months
    tot_m = dm.groupby("month").v.sum().reindex(months)
    piv = (dm[dm[level].isin(names)].pivot_table(index="month", columns=level, values="v", aggfunc="sum")
           .reindex(index=months, columns=names).fillna(0.0))
    if metric == "share":
        piv = piv.div(tot_m.where(tot_m != 0), axis=0)
    if smooth > 1:
        piv = piv.rolling(smooth, min_periods=smooth).mean()
    xs = [_label("month", x) for x in months]
    div = 1 if metric == "share" else DIV[m]
    r.chart("line", (f"Доля в продажах ({where})" if metric == "share" else f"Продажи, {unit} ({where})")
            + (f", скользящее среднее {smooth} мес." if smooth > 1 else ""),
            xs, [{"name": lbl(n), "values": [None if pd.isna(v) else v / div for v in piv[n]]} for n in names],
            fmt="pct" if metric == "share" else "num1", unit="" if metric == "share" else unit)
    r.fact(f"Продажи за {period.label()} ({'все сети' if cmp_only else where}): {S.fmoney(total, m)}; "
           f"{LEVEL_GEN[level]} с продажами: {len(per)}; показаны первые {len(names)} по продажам.", key=True)
    v1 = v0 = {}
    T1 = T0 = 0.0
    if gbase:
        g1, g0 = growth_agg(scope, gcur, gbase, [level], m, cmp_only)
        T1, T0 = float(g1.v.sum()), float(g0.v.sum())
        v1, v0 = dict(zip(g1[level], g1.v)), dict(zip(g0[level], g0.v))
        mk = "Продажи выборки" if scope.has_object() else "Рынок"
        r.fact(f"{mk} ({scope.describe()}) {gcur.vs(gbase)} ({where}): {_dyn(T1, T0, m)} "
               f"({S.fmoney(T0, m)} → {S.fmoney(T1, m)}).", key=True)
    else:
        r.fact(f"Рост за {gcur.label()} к прошлому году не считается: {no_base_reason(gcur, gap)}.", key=True)
    rows = []
    for i, n in enumerate(names):
        row = {"name": lbl(n), "v": pv[n], "share": pv[n] / total if total else None}
        s = f"{lbl(n)}: за {period.label()} {S.fmoney(pv[n], m)}, доля {S.fpct(row['share'])}"
        if gbase:
            a1, a0 = v1.get(n, 0.0), v0.get(n, 0.0)
            sh1 = a1 / T1 if T1 > 0 else None
            sh0 = a0 / T0 if T0 > 0 else None
            row.update(g=_g(a1, a0), sh1=sh1, dsh=(sh1 - sh0) if sh1 is not None and sh0 is not None else None)
            s += f"; {gcur.vs(gbase)} ({where}): {_dyn(a1, a0, m)}"
            if row["dsh"] is not None:
                s += f", доля {S.fpct(sh1)} ({S.fpp(row['dsh'])})"
        rows.append(row)
        r.fact(s + ".", key=i == 0)
        ser = piv[n].dropna() if metric == "share" else None
        if ser is not None and len(ser):
            r.fact(f"доля {lbl(n)} по месяцам ({where}): минимум {S.fpct(ser.min())} ({_label('month', ser.idxmin())}), "
                   f"максимум {S.fpct(ser.max())} ({_label('month', ser.idxmax())}), "
                   f"последний месяц {S.fpct(ser.iloc[-1])} ({_label('month', ser.index[-1])}).")
    cols = [("name", LEVEL_RU[level], "text"), ("v", f"Продажи за {period.label()}, {unit}", unit_fmt),
            ("share", "Доля", "pct")]
    if gbase:
        cols += [("g", f"Рост {gcur.vs(gbase)}", "gpct"), ("sh1", f"Доля, {gcur.label()} ({where})", "pct"),
                 ("dsh", "Изм. доли", "pp")]
    r.table(f"Итоги, {period.label()}" + (f"; рост {gcur.vs(gbase)}" if gbase else ""), cols, rows)


def _seasonality(r, scope, m, cmp_only, a=None):
    r.title = "Сезонность продаж"
    lo, hi = S.data_bounds()
    full_years = [y for y in range(lo.year, hi.year + 1) if S.date(y, 1, 1) >= lo and S.date(y, 12, 1) <= hi]
    per = S.Period(lo, hi)
    # 9 networks without month exclusions: the yearly profile is not compared with the previous year
    df = agg(scope, per, ["month", "subgroup"], m, where=_cmp_where() if cmp_only else None)
    if df.empty or not (df.v > 0).any():
        raise ToolError(f"В выборке ({scope.describe()}) нет продаж.")
    df["y"] = df.month.map(lambda d: d.year)
    df["mm"] = df.month.map(lambda d: d.month)
    kids_pair = df.subgroup.str.contains("етск", na=False).any() and (~df.subgroup.str.contains("етск", na=False)).any()
    groups = [("итого", df)]
    if kids_pair:
        groups = [("взрослые", df[~df.subgroup.str.contains("етск", na=False)]),
                  ("детские", df[df.subgroup.str.contains("етск", na=False)]), ("итого", df)]
    rows = {mm: {"month": S.MONTHS_FULL[mm - 1]} for mm in range(1, 13)}
    idx_series = []
    for gname, d in groups:
        prof = []
        for y in full_years:
            dy = d[d.y == y].groupby("mm").v.sum()
            if dy.sum() > 0:
                prof.append(dy.reindex(range(1, 13)).fillna(0) / dy.sum())
        if not prof:
            continue
        avg = sum(prof) / len(prof)
        idx = avg * 12
        for mm in range(1, 13):
            rows[mm][f"i_{gname}"] = idx[mm]
            rows[mm][f"s_{gname}"] = avg[mm]
        idx_series.append({"name": gname, "values": [idx[mm] for mm in range(1, 13)]})
        peak = idx.sort_values(ascending=False)
        top3 = peak.head(4)
        r.fact(f"{gname}: пиковые месяцы {', '.join(S.MONTHS_FULL[i - 1] + ' (индекс ' + S.fnum(v, 2) + ')' for i, v in top3.items())}; "
               f"минимум {S.MONTHS_FULL[peak.index[-1] - 1]} (индекс {S.fnum(peak.iloc[-1], 2)}).", key=gname == "итого")
        hi_months = [i for i, v in idx.items() if v >= 1.2]
        if hi_months:
            r.fact(f"{gname}: месяцы с индексом ≥1,2 дают {S.fpct(sum(avg[i] for i in hi_months))} годовых продаж.")
    cols = [("month", "Месяц", "text")]
    for gname, _ in groups:
        cols += [(f"s_{gname}", f"Доля года, {gname}", "pct"), (f"i_{gname}", f"Индекс, {gname}", "num2")]
    r.table(f"Сезонный профиль по полным годам ({', '.join(map(str, full_years))})", cols, list(rows.values()),
            note="Индекс = доля месяца в годовых продажах × 12; 1,0 — средний месяц.")
    r.chart("column", "Индекс сезонности по месяцам", S.MONTHS_RU, idx_series, fmt="num2")
    tot = df.groupby(["y", "mm"]).v.sum()
    for y in sorted(df.y.unique()):
        vals = [tot.get((y, mm)) for mm in range(1, 13)]
        r.fact(f"{y}: " + ", ".join(f"{S.MONTHS_RU[i]} {S.fmoney(v, m)}" for i, v in enumerate(vals) if v is not None))
    r.caveat("Сезонность — по полным годам " + ", ".join(map(str, full_years)) + "; двух лет мало, чтобы считать профиль доказанным.")
    if cmp_only:
        r.caveat("Расчет по 9 сетям с полным покрытием.")
    if a is not None:
        nxt = lambda **kw: _next_args(scope, m, a, **kw)
        r.suggest("Динамика продаж по месяцам", "trend", nxt(seasonality=None))
        r.suggest("Топ брендов в этом срезе", "ranking", nxt(level="brand", seasonality=None))
    return r


def t_profile(a):
    """Manufacturer or brand profile: totals, growth, share in its own categories, subgroup contribution, kids/adult,
    brands, networks."""
    scope = make_scope(a)
    obj_key = "manufacturer" if scope.values("manufacturer") else ("brand" if scope.values("brand") else None)
    if not obj_key:
        raise ToolError("Для profile нужен manufacturer или brand.")
    m = get_measure(a)
    unit_fmt, unit = UNIT[m]
    obj = ", ".join(scope.values(obj_key))
    period, explicit, gcur, gbase = _periods(a, "mat")
    gap = coverage_gap(scope, gcur, gbase)
    if gap:
        gbase = None
    cmp_only = not scope.has_network_filter()
    where = net_lbl(scope)
    r = Result("profile", f"Обзор: {obj}")
    r.subtitle = f"{scope.describe()}; {period.label()}"
    r.set_slice(scope, period, gcur if gbase else None, gbase, m, cmp=cmp_only)
    tot = float(agg(scope, period, [], m).v.sum())
    if not tot:
        raise ToolError(f"В выборке ({scope.describe()}) нет продаж за {period.label()}.")
    # market = the object's subgroups (at least 1% of its sales for the period) when no category is given;
    # the share numerator uses the same subgroups, otherwise sales outside the market inflate the share
    sg = agg(scope, period, ["subgroup_base"], m)
    sg = sg[sg.subgroup_base.notna() & (sg.v > 0.01 * sg.v.sum())]
    own = not any(scope.values(k) for k in S.CATEGORY_KEYS)
    market_scope = scope.without(*S.OBJECT_KEYS)
    obj_mk = scope
    if own:
        market_scope = market_scope.with_(subgroup=list(sg.subgroup_base))
        obj_mk = scope.with_(subgroup=list(sg.subgroup_base))
    cats = ", ".join(sg.sort_values("v", ascending=False).subgroup_base) if own else S.cat_desc(scope)
    kp = [(f"Продажи, {period.label()}", S.fmoney(tot, m), scope.describe(), None)]
    r.fact(f"{obj}: продажи за {period.label()} {S.fmoney(tot, m)} ({'все сети' if cmp_only else where}).", key=True)
    sh1 = sh0 = None
    if gbase:
        a1, a0 = (x.v.sum() for x in growth_agg(scope, gcur, gbase, [], m, cmp_only=False))
        mk1, mk0 = (x.v.sum() for x in growth_agg(market_scope, gcur, gbase, [], m, cmp_only))
        o1, o0 = (x.v.sum() for x in growth_agg(obj_mk, gcur, gbase, [], m, cmp_only))
        sh1 = o1 / mk1 if mk1 > 0 else None
        sh0 = o0 / mk0 if mk0 > 0 else None
        if cmp_only:
            c1, c0 = (x.v.sum() for x in growth_agg(scope, gcur, gbase, [], m, cmp_only=True))
            kp += [(f"Рост {gcur.vs(gbase)}, все сети", S.fpct(_g(a1, a0), sign=True) if _g(a1, a0) is not None else "не считается",
                    f"{S.fmoney(a0, m)} → {S.fmoney(a1, m)}", _g(a1, a0)),
                   (f"Рост {gcur.vs(gbase)}, 9 сетей", S.fpct(_g(c1, c0), sign=True) if _g(c1, c0) is not None else "не считается",
                    f"рынок {S.fpct(S.growth(mk1, mk0), sign=True)}" if S.growth(mk1, mk0) is not None else "", _g(c1, c0))]
            r.fact(f"{gcur.vs(gbase)}: все сети {S.fmoney(a0, m)} → {S.fmoney(a1, m)}, {_dyn(a1, a0, m)}; "
                   f"9 сопоставимых сетей {S.fmoney(c0, m)} → {S.fmoney(c1, m)}, {_dyn(c1, c0, m)}.", key=True)
        else:
            kp.append((f"Рост {gcur.vs(gbase)}, {where}", S.fpct(_g(a1, a0), sign=True) if _g(a1, a0) is not None else "не считается",
                       f"{S.fmoney(a0, m)} → {S.fmoney(a1, m)}", _g(a1, a0)))
            r.fact(f"{gcur.vs(gbase)} ({where}): {S.fmoney(a0, m)} → {S.fmoney(a1, m)}, {_dyn(a1, a0, m)}.", key=True)
        if sh1 is not None and sh0 is not None:
            kp.append((f"Доля в своих категориях ({where})", S.fpct(sh1, 2), f"год назад {S.fpct(sh0, 2)}", sh1 - sh0))
            r.fact(f"Рынок в категориях объекта ({cats}; {where}) {gcur.vs(gbase)}: {_dyn(mk1, mk0, m)}; "
                   f"доля {obj} в этих категориях {S.fpct(sh0, 2)} → {S.fpct(sh1, 2)} ({S.fpp(sh1 - sh0)}).", key=True)
    else:
        r.fact(f"Рост к прошлому году не считается: {no_base_reason(gcur, gap)}.", key=True)
    r.kpis(kp)
    if gbase:
        _contrib_table(r, scope, gcur, gbase, m, cmp_only, where, obj)
    add_kids_table(r, scope, period, m, base=None, title=f"Доля по категориям: взрослые и детские, {period.label()}")
    if obj_key == "manufacturer":
        b = _stable(agg(scope, period, ["brand"], m), "v", "brand")
        b = b[~b.brand.map(_blank)]
        bv1, bv0 = {}, {}
        if gbase:
            bg1, bg0 = growth_agg(scope, gcur, gbase, ["brand"], m, cmp_only)
            bv1, bv0 = dict(zip(bg1.brand, bg1.v)), dict(zip(bg0.brand, bg0.v))
        rows = []
        for x in b.head(15).itertuples():
            v1, v0 = bv1.get(x.brand, 0.0), bv0.get(x.brand, 0.0)
            pl = " [СТМ]" if "(PL)" in str(x.brand) else ""
            rows.append({"name": x.brand + pl, "v": x.v, "share": x.v / tot, "g": _g(v1, v0) if gbase else None,
                         "d": (v1 - v0) if gbase else None})
            r.fact(f"бренд {x.brand}{pl}: {S.fmoney(x.v, m)} ({S.fpct(x.v / tot)} продаж {obj})"
                   + (f"; {gcur.vs(gbase)} ({where}): {_dyn(v1, v0, m)} ({_fdelta(v1 - v0, m)})" if gbase else ""))
        cols = [("name", "Бренд", "text"), ("v", f"Продажи, {unit}", unit_fmt), ("share", f"Доля в {obj}", "pct")]
        if gbase:
            cols += [("g", f"Рост {gcur.vs(gbase)} ({where})", "gpct"), ("d", f"Изменение, {unit}", unit_fmt)]
        r.table(f"Бренды, {period.label()}", cols, rows)
    _profile_networks(r, scope, obj_mk, market_scope, period, gcur, gbase, m, obj)
    if gbase:
        r.caveat(comparable_note(scope, gcur, gbase)
                 + (" «Все сети» включают сети с неполным покрытием: на рост влияет появление и пропадание сети в данных."
                    if cmp_only else ""))
        bn = base_note(a, gcur, gbase)
        if bn:
            r.caveat(bn)
    if gap:
        r.caveat("Рост не считается: " + no_base_reason(gcur, gap) + ".")
    if own:
        r.caveat(f"Доля в своих категориях — доля объекта в продажах подгрупп, где у него не меньше 1% продаж за {period.label()}; "
                 "числитель и знаменатель посчитаны в одних и тех же подгруппах.")
    nxt = lambda **kw: _next_args(scope, m, a, **kw)
    r.suggest(f"Факторный анализ изменения продаж {obj}", "factors", nxt())
    if obj_key == "manufacturer":
        r.suggest(f"Конкуренты {obj} в его сегментах", "competitors", nxt())
    r.suggest(f"Представленность {obj} по сетям", "distribution", nxt(by="network"))
    r.suggest(f"Цены {obj}" + (" по брендам" if obj_key == "manufacturer" else " по SKU"), "prices",
              nxt(level="brand" if obj_key == "manufacturer" else "sku"))
    return r


def _contrib_table(r, scope, gcur, gbase, m, cmp_only, where, obj):
    """Subgroup contribution to the object's sales change (on the growth base): where the change came from.
    Example: for L'AURELLE PARIS 2026 almost the whole decline came from exiting decorative cosmetics."""
    unit_fmt, unit = UNIT[m]
    s1, s0 = growth_agg(scope, gcur, gbase, ["subgroup_base"], m, cmp_only)
    d = s1[["subgroup_base", "v"]].merge(s0[["subgroup_base", "v"]], on="subgroup_base", how="outer",
                                         suffixes=("1", "0")).fillna({"v1": 0.0, "v0": 0.0})
    d["subgroup_base"] = d.subgroup_base.fillna("подгруппа не указана")
    d["d"] = d.v1 - d.v0
    D = float(d.d.sum())
    d = d.reindex(d.d.abs().sort_values(ascending=False).index)
    d = d[(d.v1 != 0) | (d.v0 != 0)]
    if d.empty:
        return
    rows = [{"name": x.subgroup_base, "v0": x.v0, "v1": x.v1, "d": x.d, "g": _g(x.v1, x.v0),
             "part": x.d / D if D else None} for x in d.head(12).itertuples()]
    r.table(f"Вклад подгрупп в изменение, {gcur.vs(gbase)} ({where})",
            [("name", "Подгруппа", "text"), ("v0", f"{gbase.label()}, {unit}", unit_fmt), ("v1", f"{gcur.label()}, {unit}", unit_fmt),
             ("d", f"Изменение, {unit}", unit_fmt), ("g", "Рост", "gpct"), ("part", "Доля в изменении", "pct")], rows)
    r.fact(f"Изменение продаж {obj} {gcur.vs(gbase)} ({where}): {_fdelta(D, m)}; крупнейшие вклады подгрупп ниже.")
    for x in d.head(5).itertuples():
        s = f"вклад подгруппы {x.subgroup_base}: {S.fmoney(x.v0, m)} → {S.fmoney(x.v1, m)} ({_fdelta(x.d, m)}"
        if D and x.d * D > 0:
            s += f", {S.fpct(x.d / D)} общего изменения"
        elif D:
            s += ", против общего изменения"
        r.fact(s + ").")


def _profile_networks(r, scope, obj_mk, market_scope, period, gcur, gbase, m, obj):
    """Networks: object sales, share in the network's categories, network's share of object sales by year, growth.
    Network growth is computed only if the network is present for all comparison months (methodology §3.2, §3.4)."""
    unit_fmt, unit = UNIT[m]
    lo, hi = S.data_bounds()
    years = [S.Period(max(S.date(y, 1, 1), lo), min(S.date(y, 12, 1), hi)) for y in range(lo.year, hi.year + 1)]
    net_rows = {}
    for y in years:
        d = agg(scope, y, ["network"], m)
        t = d.v.sum()
        for x in d.itertuples():
            net_rows.setdefault(x.network, {"name": S.net_label(x.network)})[f"y{y.start.year}"] = x.v / t if t else None
    d_obj = agg(scope, period, ["network"], m)
    # one query per dict: networks and values from two separate GROUP BYs without ORDER BY may come back in different orders
    d_om = agg(obj_mk, period, ["network"], m).set_index("network").v.to_dict()
    d_mk = agg(market_scope, period, ["network"], m).set_index("network").v.to_dict()
    for x in d_obj.itertuples():
        row = net_rows.setdefault(x.network, {"name": S.net_label(x.network)})
        row["v"] = x.v
        mk = d_mk.get(x.network, 0.0)
        row["msh"] = d_om.get(x.network, 0.0) / mk if mk else None
    if gbase:
        n1, n0 = growth_agg(scope, gcur, gbase, ["network"], m, cmp_only=False)
        v1, v0 = dict(zip(n1.network, n1.v)), dict(zip(n0.network, n0.v))
        for nm, row in net_rows.items():
            row["cov"] = _covered(nm, gcur, gbase)
            row["v1"], row["v0"] = v1.get(nm, 0.0), v0.get(nm, 0.0)
            row["g"] = _g(row["v1"], row["v0"]) if row["cov"] else None
    nrows = sorted(net_rows.values(), key=lambda x: -(x.get("v") or 0))
    cols = [("name", "Сеть", "text"), ("v", f"Продажи, {unit}", unit_fmt), ("msh", "Доля в категориях сети", "pct2")]
    cols += [(f"y{y.start.year}", f"Доля сети в продажах, {y.label()}", "pct") for y in years]
    if gbase:
        cols.append(("g", f"Рост {gcur.vs(gbase)}", "gpct"))
    r.table(f"Сети, {period.label()}", cols, nrows)
    for row in nrows[:10]:
        s = f"сеть {row['name']}: {S.fmoney(row.get('v') or 0.0, m)} за {period.label()}"
        if row.get("msh") is not None:
            s += f", доля {obj} в категориях сети {S.fpct(row['msh'], 2)}"
        s += f"; доля сети в продажах {obj}: " + ", ".join(
            f"{y.label()} {S.fpct(row[f'y{y.start.year}'])}" if row.get(f"y{y.start.year}") is not None
            else f"{y.label()} продаж нет" for y in years)
        if gbase:
            s += (f"; {gcur.vs(gbase)}: {_dyn(row['v1'], row['v0'], m)}" if row["cov"]
                  else "; рост не считается: сеть есть в данных не за все месяцы сравнения")
        r.fact(s + ".")
    r.chart("bar", f"Продажи по сетям, {unit}", [x["name"] for x in nrows[:12]],
            [{"name": "Продажи", "values": [(x.get("v") or 0) / DIV[m] for x in nrows[:12]]}], fmt="num1", unit=unit)
    if gbase and any(not x.get("cov", True) for x in nrows):
        r.caveat("Рост по сети считается, только если сеть есть в данных за все месяцы сравнения: "
                 + ", ".join(f"{x['name']} ({month_segments(net_months().get(n, []))})" for n, x in net_rows.items()
                             if not x.get("cov", True)) + ".")


def _strip_literals(sql):
    """SQL without string literals: forbidden words are searched only in the query itself, not in 'set' inside a string."""
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def t_sql(a):
    """Free-form read-only SQL with row and time limits."""
    sql = str(a.get("query") or "").strip().rstrip(";")
    low = _strip_literals(sql).lower()
    if not re.match(r"^(select|with)\b", low) or ";" in low:
        raise ToolError("Разрешен один запрос SELECT или WITH.")
    if re.search(r"\b(read_\w+|glob|copy|attach|install|load|pragma|export|import|create|insert|update|delete|drop|alter|set)\b|http", low):
        raise ToolError("Запрос содержит запрещенные операции. Можно только читать таблицы fact, net_sku, store_sub, sku, store.")
    limit = 200
    c = S.con()
    timer = threading.Timer(60, c.interrupt)
    timer.start()
    n_total = None
    try:
        df = c.execute(f"SELECT * FROM ({sql}) LIMIT {limit + 1}").df()
        if len(df) <= limit:
            n_total = len(df)
        else:
            df = df.head(limit)
            try:
                n_total = c.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0]
            except Exception:
                n_total = None
    except Exception as e:
        raise ToolError(f"Ошибка SQL: {str(e)[:400]}")
    finally:
        timer.cancel()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date
    truncated = n_total is None or n_total > limit
    r = Result("sql", "Расчет запросом")
    r.slice_line = "Срез: задан самим запросом (свободный SQL); правила методики применены, только если они есть в запросе."
    r.subtitle = f"строк в результате: {n_total if n_total is not None else f'больше {limit}'}" + (f"; показаны первые {limit}" if truncated else "")
    cols = []
    for c_ in df.columns:
        fmt = "num2" if pd.api.types.is_float_dtype(df[c_]) else ("int" if pd.api.types.is_integer_dtype(df[c_]) else "text")
        cols.append((c_, c_, fmt))
    r.table("Результат", cols, df.to_dict("records"))
    shown = min(len(df), 40)
    r.fact((f"Строк в результате: {n_total}" if n_total is not None else f"Строк в результате больше {limit} (полный подсчет не успел)")
           + (f"; в таблице первые {len(df)}" if truncated else "") + f"; в фактах строки 1–{shown}"
           + (" (все)." if not truncated and shown == len(df) else "."), key=True)

    def cell(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "нет значения"
        if isinstance(v, float):
            return S.fnum(v, 2)
        return str(v)
    for i, row in enumerate(df.head(40).itertuples(index=False), 1):
        r.fact(f"строка {i}: " + "; ".join(f"{col} = {cell(v)}" for col, v in zip(df.columns, row)))
    r.caveat("Свободный запрос: правила методики (сопоставимые сети, одинаковые месяцы) применены только если они есть в запросе.")
    r.args = {"query": sql}
    return r
