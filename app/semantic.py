"""Semantic layer: periods, filters and methodology rules expressed as code.

The model does not write SQL for typical questions: it picks an analysis and filters, and this module
turns them into DuckDB queries following the rules in docs/methodology.md.
"""
import os
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(os.environ.get("VT_DB", Path(__file__).resolve().parents[1] / "data" / "retail.duckdb"))
THREADS = int(os.environ.get("VT_THREADS", str(os.cpu_count() or 4)))
MEMORY = os.environ.get("VT_MEMORY", "4GB")

COMPARABLE = ("Гранит", "Семерочка Х7", "Нива (сеть)", "Перелесок ТД", "Ажур",
              "Депо Кэш Энд Керри", "ЛАД (сеть)", "Гиперкосмос", "Детский Остров (сеть)")
MONTHS_RU = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
MONTHS_FULL = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август",
               "сентябрь", "октябрь", "ноябрь", "декабрь"]
MONTHS_DAT = ["январю", "февралю", "марту", "апрелю", "маю", "июню", "июлю", "августу",
              "сентябрю", "октябрю", "ноябрю", "декабрю"]
MEASURES = {"rub": ("руб.", "млн руб."), "qty": ("шт.", "тыс. шт."), "kg": ("кг/л", "т/тыс. л")}
MEASURE_NAME = {"rub": "рубли", "qty": "штуки", "kg": "кг/л"}
CITY_REGIONS = ("Москва г", "Санкт-Петербург г", "Севастополь г")

# Store: one store_key per physical store (network + store code). In some networks and months one store is
# registered under several store_ids at once (same address) — without merging them the store count is inflated.
# city_eff: the city; for Moscow, St. Petersburg and Sevastopol the city is empty, so the city-region is used;
# other empty cities become «город не указан (регион)» so a city ranking does not silently lose sales.
STORE_X_SQL = f"""
CREATE TEMP TABLE IF NOT EXISTS store_x AS
SELECT s.*,
       min(s.store_id) OVER (PARTITION BY s.network, coalesce(nullif(trim(s.store_code), ''), s.store_id::VARCHAR)) AS store_key,
       coalesce(nullif(trim(s.city), ''),
                CASE WHEN s.region IN ({', '.join("'" + x + "'" for x in CITY_REGIONS)}) THEN s.region
                     ELSE 'город не указан (' || coalesce(s.region, 'регион не указан') || ')' END) AS city_eff
FROM store s"""

_local = threading.local()


def con():
    """One connection per thread: a DuckDB connection is not thread-safe for concurrent queries."""
    c = getattr(_local, "con", None)
    if c is None:
        c = duckdb.connect(str(DB_PATH), read_only=True)
        c.execute(f"SET threads = {THREADS}")
        c.execute(f"SET memory_limit = '{MEMORY}'")
        c.execute(STORE_X_SQL)          # the temp table lives in the connection; the database itself stays read-only
        _local.con = c
    return c


def q(sql, params=None):
    """Query -> DataFrame. Dates are cast to datetime.date: DuckDB 2 + pandas 3 return Timestamp,
    while Period.months holds date, and without the cast month comparisons silently fail to match."""
    df = con().execute(sql, params or []).df()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.date
    return df


def q1(sql, params=None):
    return con().execute(sql, params or []).fetchone()


# ——— Periods ———

def ym(d):
    return f"{d.year:04d}-{d.month:02d}"


def parse_ym(s):
    y, m = s.split("-")[:2]
    return date(int(y), int(m), 1)


def add_months(d, n):
    k = d.year * 12 + d.month - 1 + n
    return date(k // 12, k % 12 + 1, 1)


@dataclass(frozen=True)
class Period:
    start: date
    end: date

    @property
    def months(self):
        out, d = [], self.start
        while d <= self.end:
            out.append(d)
            d = add_months(d, 1)
        return out

    @property
    def n(self):
        return len(self.months)

    def shift(self, months):
        return Period(add_months(self.start, months), add_months(self.end, months))

    def label(self, case="nom"):
        """Period label. case='dat' is the dative for the «к ...» ("vs ...") construction: «к сентябрю 2025», «к 2025 году»."""
        s, e = self.start, self.end
        if s == e:
            return f"{(MONTHS_DAT if case == 'dat' else MONTHS_FULL)[s.month - 1]} {s.year}"
        if s.year == e.year:
            if s.month == 1 and e.month == 12:
                return f"{s.year} году" if case == "dat" else f"{s.year} год"
            return f"{MONTHS_RU[s.month - 1]}–{MONTHS_RU[e.month - 1]} {s.year}"
        return f"{MONTHS_RU[s.month - 1]} {s.year} – {MONTHS_RU[e.month - 1]} {e.year}"

    def vs(self, base):
        """Comparison label, e.g. «янв–сен 2026 к янв–сен 2025», «сентябрь 2026 к сентябрю 2025»."""
        return f"{self.label()} к {base.label('dat')}"

    def short(self):
        return f"{self.start:%m.%Y}–{self.end:%m.%Y}" if self.start != self.end else f"{self.start:%m.%Y}"

    def sql(self, col="month"):
        return f"{col} BETWEEN DATE '{self.start}' AND DATE '{self.end}'"

    def to_str(self):
        return f"{ym(self.start)}..{ym(self.end)}"


_bounds = None


def data_bounds():
    global _bounds
    if _bounds is None:
        a, b = q1("SELECT min(month), max(month) FROM net_sku")
        _bounds = (a, b)
    return _bounds


def last_month():
    return data_bounds()[1]


def mat(end=None):
    end = end or last_month()
    return Period(add_months(end, -11), end)


def ytd(end=None):
    end = end or last_month()
    return Period(date(end.year, 1, 1), end)


def parse_period(s, default="mat"):
    """'2026-01..2026-06', '2026-09', '2025', 'mat', 'ytd', 'all', 'last_month' -> Period."""
    s = (s or default or "mat").strip().lower()
    lo, hi = data_bounds()
    if s in ("mat", "12m", "last12"):
        return mat()
    if s == "ytd":
        return ytd()
    if s in ("all", "весь"):
        return Period(lo, hi)
    if s in ("last_month", "month"):
        return Period(hi, hi)
    if ".." in s:
        a, b = s.split("..")
        a, b = a.strip(), b.strip()
        pa = parse_ym(a) if len(a) > 4 else date(int(a), 1, 1)
        pb = parse_ym(b) if len(b) > 4 else date(int(b), 12, 1)
        return Period(pa, pb)
    if len(s) == 4 and s.isdigit():
        return Period(date(int(s), 1, 1), min(date(int(s), 12, 1), hi))
    return Period(parse_ym(s), parse_ym(s))


def clip(p):
    """Clip a period to the data bounds; None if there is no overlap."""
    lo, hi = data_bounds()
    s, e = max(p.start, lo), min(p.end, hi)
    return Period(s, e) if s <= e else None


# ——— Data completeness: network × month ———

_cov = None
_cov_lock = threading.Lock()


def coverage():
    """Load completeness by network and month: stores with sales (by store_key), revenue, ok.

    A network's month is incomplete (ok=False) if stores fall below 85% or revenue below 75% of the median of the
    three previous months, AND it is either the network's last month in the data or the next month recovers to
    the previous level. If the low level persists, it is a real shift (e.g. a network shrinking), not a partial load.
    Example: the last month of a network loaded with under half of its usual stores."""
    global _cov
    with _cov_lock:
        if _cov is None:
            df = q("""SELECT s.network, ss.month, count(DISTINCT s.store_key) st, sum(ss.rub) rub
                      FROM store_sub ss JOIN store_x s USING (store_id)
                      WHERE ss.qty > 0 GROUP BY 1, 2 ORDER BY 1, 2""")
            parts = []
            for _, d in df.groupby("network", sort=False):
                d = d.sort_values("month").reset_index(drop=True)
                med_st = d.st.shift(1).rolling(3, min_periods=2).median()
                med_rub = d.rub.shift(1).rolling(3, min_periods=2).median()
                ok = []
                for i in range(len(d)):
                    if pd.isna(med_st[i]) or not (d.st[i] < 0.85 * med_st[i] or d.rub[i] < 0.75 * med_rub[i]):
                        ok.append(True)
                        continue
                    last = i == len(d) - 1
                    recovered = not last and d.st[i + 1] >= 0.9 * med_st[i] and d.rub[i + 1] >= 0.85 * med_rub[i]
                    ok.append(not (last or recovered))
                parts.append(d.assign(med_st=med_st, med_rub=med_rub, ok=ok))
            _cov = pd.concat(parts, ignore_index=True)
    return _cov


def bad_pairs():
    """{(network, month)}: months that were loaded only partially."""
    return {(r.network, r.month) for r in coverage().itertuples() if not r.ok}


def growth_exclusions():
    """(network, month) pairs removed from year-over-year comparisons on both sides:
    the incomplete month, the same month a year earlier and a year later. This keeps comparisons like-for-like."""
    out = set()
    for n, m in bad_pairs():
        for k in (-12, 0, 12):
            out.add((n, add_months(m, k)))
    return out


def _excl_sql(period, net_col="network", month_col="month"):
    pairs = sorted((n, m) for n, m in growth_exclusions() if period.start <= m <= period.end)
    return " AND ".join(f"NOT ({net_col} = {_lit(n)} AND {month_col} = DATE '{m}')" for n, m in pairs)


def scope_networks(scope):
    """Networks the scope touches: from the network filter, via the banner filter, or None for all."""
    nets = list(scope.values("network"))
    if scope.values("banner"):
        df = q(f"SELECT DISTINCT network FROM store_x WHERE banner IN ({', '.join(_lit(b) for b in scope.values('banner'))})")
        nets += [x for x in df.network if x not in nets]
    return nets or None


def last_ok_month(networks=None):
    """Last month that is complete for all given networks (default: the 9 comparable networks)."""
    nets = set(networks or COMPARABLE)
    cov = coverage()
    hi = data_bounds()[1]
    bad = {m for n, m in bad_pairs() if n in nets}
    m = hi
    while m in bad and m > cov.month.min():
        m = add_months(m, -1)
    return m


def coverage_notes(scope, periods, growth=False):
    """Caveat about incomplete months within the scope's periods, as one short line (facts passed to the model are
    length-limited and long caveats crowded out ranking rows). growth=True means the calculation compares with last year."""
    nets = scope_networks(scope)
    months = {m for p in periods if p for m in p.months}
    cov = coverage()
    parts = []
    for r in cov[~cov.ok].itertuples():
        if r.month not in months or (nets and r.network not in nets) or (not nets and r.network not in COMPARABLE):
            continue
        parts.append(f"{net_label(r.network)} {r.month:%m.%Y} ({r.st} из ~{int(round(r.med_st))} точек, "
                     f"выручка {fpct(r.rub / r.med_rub, 0)} от обычной)")
    if not parts:
        return []
    txt = "Месяцы загружены поставщиком не полностью: " + "; ".join(parts)
    if growth:
        txt += " — в сравнении с прошлым годом они исключены с обеих сторон"
    return [txt + "."]


# ——— Own-company (NORD) categories ———

# Design decision: by default the analysis covers only subgroups where NORD has products. A subgroup is included if
# it holds at least OWN_MIN_SHARE of NORD's all-time sales (adult and kids paired via subgroup_base). Without the
# threshold «Гигиена медицинская» would qualify: NORD has a single near-zero SKU there, while the market is huge.
OWN = os.environ.get("VT_OWN_MANUFACTURER", "NORD")      # the company the analyst works for
OWN_MIN_SHARE = float(os.environ.get("VT_OWN_MIN_SHARE", "0.001"))
OWN_CATS_LABEL = f"категории {OWN}"
_own_cats = None
_own_lock = threading.Lock()


def own_categories():
    """Base subgroups (subgroup_base) where NORD has products: by default the scope is limited to them."""
    global _own_cats
    with _own_lock:
        if _own_cats is None:
            d = q("""SELECT k.subgroup_base b, sum(n.rub) FILTER (WHERE k.manufacturer = ?) nord
                     FROM net_sku n JOIN sku k USING (sku_id) GROUP BY 1""", [OWN])
            d = d[d.nord.notna() & (d.nord > 0) & d.b.notna()]
            _own_cats = sorted(d[d.nord >= OWN_MIN_SHARE * d.nord.sum()].b.tolist())
    return _own_cats


# ——— Filters ———

# filter key -> (table, column, match mode)
FILTERS = {
    "manufacturer": ("sku", "manufacturer", "eq"),
    "brand": ("sku", "brand", "eq"),
    "subbrand": ("sku", "subbrand", "eq"),
    "sku": ("sku", "sku_name", "like"),
    "category_group": ("sku", "group_name", "eq"),
    "subgroup": ("sku", "subgroup_base", "eq"),      # adult and kids together (rule 2.4)
    "subgroup_exact": ("sku", "subgroup_name", "eq"),
    "product_type": ("sku", "product_type", "eq"),
    "product": ("sku", "product", "eq"),
    "network": ("store", "network", "eq"),
    "banner": ("store", "banner", "eq"),
    "format": ("store", "format", "eq"),
    "district": ("store", "district", "eq"),
    "region": ("store", "region", "eq"),
    "city": ("store", "city_eff", "eq"),     # column of the store_x temp table (see STORE_X_SQL)
}
# net_sku mart columns that can be filtered without the fact table
NET_SKU_STORE_COLS = {"network", "banner", "format", "district"}


def _lit(v):
    return "'" + str(v).replace("'", "''") + "'"


def _as_list(v):
    if v is None or v == "" or v == []:
        return []
    return v if isinstance(v, (list, tuple)) else [v]


FILTER_NAMES = {"manufacturer": "производитель", "brand": "бренд", "subbrand": "подбренд", "sku": "SKU",
                "category_group": "группа", "subgroup": "подгруппа", "subgroup_exact": "подгруппа",
                "product_type": "тип", "product": "товар", "network": "сеть", "banner": "баннер",
                "format": "формат", "district": "округ", "region": "регион", "city": "город"}
OBJECT_KEYS = ("manufacturer", "brand", "subbrand", "sku")
CATEGORY_KEYS = ("category_group", "subgroup", "subgroup_exact", "product_type", "product")
STORE_KEYS = ("network", "banner", "format", "district", "region", "city")


def _cond(col, how, vals, neg=False):
    if how == "like":
        c = "(" + " OR ".join(f"{col} ILIKE {_lit('%' + v + '%')}" for v in vals) + ")"
        return f"NOT {c}" if neg else c
    return f"{col} {'NOT IN' if neg else 'IN'} ({', '.join(_lit(v) for v in vals)})"


@dataclass
class Scope:
    """Set of analysis filters. Values are already resolved to exact dictionary values.
    ex — exclusions («все сети, кроме Гранита», "all networks except Granit"): filter key -> values removed from the sample."""
    f: dict = field(default_factory=dict)
    portal: bool = False          # only the reference portal's 6 subgroups
    kids: bool | None = None      # True — kids only, False — adult only
    exclude_pl: bool = False
    ex: dict = field(default_factory=dict)
    universe: bool = True         # default: NORD categories only (own_categories) when no category is named

    def values(self, key):
        return _as_list(self.f.get(key))

    def excluded(self, key):
        return _as_list(self.ex.get(key))

    def in_own_categories(self):
        """Scope is limited to NORD categories: no category was named and the whole panel was not requested."""
        return self.universe and not self.portal and not any(self.values(k) for k in CATEGORY_KEYS)

    def sku_where(self):
        parts = []
        for key, (tab, col, how) in FILTERS.items():
            if tab != "sku":
                continue
            if self.values(key):
                parts.append(_cond(col, how, self.values(key)))
            if self.excluded(key):
                parts.append(_cond(col, how, self.excluded(key), neg=True))
        if self.portal:
            parts.append("in_portal")
        if self.in_own_categories():
            parts.append(f"subgroup_base IN ({', '.join(_lit(x) for x in own_categories())})")
        if self.kids is True:
            parts.append("is_kids")
        elif self.kids is False:
            parts.append("NOT is_kids")
        if self.exclude_pl:
            parts.append("NOT is_pl")
        return " AND ".join(parts)

    def store_where(self, alias=""):
        parts = []
        for key, (tab, col, how) in FILTERS.items():
            if tab != "store":
                continue
            if self.values(key):
                parts.append(_cond(f"{alias}{col}", how, self.values(key)))
            if self.excluded(key):
                parts.append(_cond(f"{alias}{col}", how, self.excluded(key), neg=True))
        return " AND ".join(parts)

    def needs_fact(self):
        """Region and city exist only in the fact table (via stores); everything else is in the mart."""
        return any(self.values(k) or self.excluded(k) for k in ("region", "city"))

    def has_network_filter(self):
        """Scope targets specific networks or banners: growth is computed on them, not on the 9 comparable networks."""
        return bool(self.values("network") or self.values("banner"))

    def has_object(self):
        return any(self.values(k) for k in OBJECT_KEYS)

    def without(self, *keys):
        f = {k: v for k, v in self.f.items() if k not in keys}
        ex = {k: v for k, v in self.ex.items() if k not in keys}
        return Scope(f, self.portal, self.kids, self.exclude_pl, ex, self.universe)

    def with_(self, **kw):
        f = dict(self.f)
        f.update({k: v for k, v in kw.items()})
        return Scope(f, self.portal, self.kids, self.exclude_pl, dict(self.ex), self.universe)

    def with_kids(self, kids):
        return Scope(dict(self.f), self.portal, kids, self.exclude_pl, dict(self.ex), self.universe)

    def describe(self, keys=None):
        out = [f"{FILTER_NAMES[k]}: {', '.join(map(str, self.values(k)))}"
               for k in FILTERS if self.values(k) and (keys is None or k in keys)]
        out += [f"исключено — {FILTER_NAMES[k]}: {', '.join(map(str, self.excluded(k)))}"
                for k in FILTERS if self.excluded(k) and (keys is None or k in keys)]
        if keys is None or any(k in keys for k in CATEGORY_KEYS):
            if self.portal:
                out.append("категория «Косметика» портала (6 подгрупп)")
            if self.in_own_categories():
                out.append(OWN_CATS_LABEL)
            if self.kids is True:
                out.append("только детские подгруппы")
            elif self.kids is False:
                out.append("только взрослые подгруппы")
            if self.exclude_pl:
                out.append("без СТМ")
        return "; ".join(out) or "вся панель"


def _cmp_sql(alias=""):
    return f"{alias}network IN (" + ", ".join(map(_lit, COMPARABLE)) + ")"


def source(scope, period, comparable_only=False, extra_sku=None, extra_store=None, geo=False, growth=False):
    """FROM clause: the net_sku mart (fast) or the fact table joined to stores (when region or city is needed).
    Returns subquery SQL with columns month, sku_id, network, banner, format, district, qty, rub, kg, cost, stores
    (+ region, city from the fact table; city is store_x.city_eff).
    comparable_only — only the 9 networks with full coverage. growth=True or comparable_only — a year-over-year
    calculation: incomplete network months (and the same months a year earlier) are removed on both sides (growth_exclusions)."""
    sw = " AND ".join(x for x in (scope.sku_where(), extra_sku) if x)
    excl = (comparable_only or growth)
    if scope.needs_fact() or geo:
        stw = " AND ".join(x for x in (scope.store_where("s."), extra_store, _cmp_sql("s.") if comparable_only else "",
                                       _excl_sql(period, "s.network", "f.month") if excl else "") if x)
        return f"""(SELECT f.month, f.sku_id, s.network, s.banner, s.format, s.district, s.region, s.city_eff AS city,
                          f.qty, f.rub, f.kg, f.cost, (f.qty > 0)::INT AS stores
                   FROM fact f JOIN store_x s USING (store_id)
                   WHERE {period.sql('f.month')}
                     {('AND f.sku_id IN (SELECT sku_id FROM sku WHERE ' + sw + ')') if sw else ''}
                     {('AND ' + stw) if stw else ''})"""
    stw = " AND ".join(x for x in (scope.store_where(), extra_store, _cmp_sql() if comparable_only else "",
                                   _excl_sql(period) if excl else "") if x)
    return f"""(SELECT * FROM net_sku
               WHERE {period.sql()}
                 {('AND sku_id IN (SELECT sku_id FROM sku WHERE ' + sw + ')') if sw else ''}
                 {('AND ' + stw) if stw else ''})"""


def fact_source(scope, period, comparable_only=False, growth=False):
    """Store-level fact — for distribution and the stores × SKU-per-store decomposition.
    Count stores by store_key (count(DISTINCT store_key)): one store may have several store_ids."""
    sw = scope.sku_where()
    excl = (comparable_only or growth)
    stw = " AND ".join(x for x in (scope.store_where("s."), _cmp_sql("s.") if comparable_only else "",
                                   _excl_sql(period, "s.network", "f.month") if excl else "") if x)
    return f"""(SELECT f.month, f.store_id, s.store_key, f.sku_id, s.network, s.banner, s.format, s.district, s.region,
                      s.city_eff AS city, f.qty, f.rub, f.kg, f.cost
               FROM fact f JOIN store_x s USING (store_id)
               WHERE {period.sql('f.month')}
                 {('AND f.sku_id IN (SELECT sku_id FROM sku WHERE ' + sw + ')') if sw else ''}
                 {('AND ' + stw) if stw else ''})"""


def cat_desc(scope):
    """Which categories the scope includes — for the «Срез» (scope) line (methodology 2.1: always state what is included)."""
    parts = [f"{FILTER_NAMES[k]}: {', '.join(map(str, scope.values(k)))}" for k in CATEGORY_KEYS if scope.values(k)]
    parts += [f"исключено — {FILTER_NAMES[k]}: {', '.join(map(str, scope.excluded(k)))}" for k in CATEGORY_KEYS if scope.excluded(k)]
    if scope.portal:
        parts.append("категория «Косметика» портала (6 подгрупп)")
    if scope.in_own_categories():
        parts.insert(0, f"{OWN_CATS_LABEL} ({len(own_categories())} подгрупп, где у {OWN} есть товары)")
    if not parts:
        parts.append("все категории панели, включая гигиену медицинскую (бумага, салфетки)")
    if scope.kids is True:
        parts.append("только детские подгруппы")
    elif scope.kids is False:
        parts.append("только взрослые подгруппы")
    if scope.exclude_pl:
        parts.append("без СТМ")
    return "; ".join(parts)


def slice_text(scope, period, gcur=None, gbase=None, measure="rub", cmp=None, note=None):
    """The «Срез» (scope) line under an answer: exactly what was computed. Built by code; the model does not write it."""
    out = [cat_desc(scope)]
    obj = scope.describe(OBJECT_KEYS)
    if obj != "вся панель":
        out.append(obj)
    nets = scope.describe(STORE_KEYS)
    out.append(nets if nets != "вся панель" else "все сети")
    out.append(f"период {period.label()}")
    if gcur and gbase:
        where = "9 сопоставимых сетей" if cmp else "выбранные сети"
        out.append(f"рост {gcur.vs(gbase)} ({where})")
    out.append(f"мера: {MEASURE_NAME.get(measure, measure)}")
    if note:
        out.append(note)
    return "Срез: " + "; ".join(out) + "."


NET_SHORT = {
    "Нива (сеть)": "Нива", "Детский Остров (сеть)": "Детский Остров", "ЛАД (сеть)": "ЛАД",
    "Маятник (сеть)": "Маятник", "Добрый (сеть)": "Добрый", "ПАРУС (сеть)": "ПАРУС",
    "НУ (сеть)": "НУ", "Камелия (сеть)": "Камелия", "Депо Кэш Энд Керри": "Депо",
    "Семерочка Х7": "Семерочка", "Перелесок ТД": "Перелесок",
}


def net_label(name):
    return NET_SHORT.get(name, name)


# ——— Number formatting for text and facts ———

def fnum(v, digits=1):
    if v is None:
        return "—"
    try:
        if v != v:
            return "—"
    except TypeError:
        return str(v)
    s = f"{v:,.{digits}f}".replace(",", " ").replace(".", ",")
    return s.replace("-", "−")


def fmoney(v, measure="rub"):
    """Sales in readable units: bn/mn/thousand rubles, mn/thousand units, thousand t/t/kg.
    The unit is chosen after rounding to avoid «1 000,0 млн» and «0,0 млн» for small amounts."""
    if v is None or v != v:
        return "—"
    a = abs(v)
    if measure == "rub":
        if round(a / 1e9, 2) >= 1:
            return f"{fnum(v / 1e9, 2)} млрд руб."
        if round(a / 1e6, 1) >= 1:
            return f"{fnum(v / 1e6, 1)} млн руб."
        if round(a / 1e3, 1) >= 1:
            return f"{fnum(v / 1e3, 1)} тыс. руб."
        return f"{fnum(v, 0)} руб."
    if measure == "qty":
        if round(a / 1e6, 2) >= 1:
            return f"{fnum(v / 1e6, 2)} млн шт."
        if a >= 1e4:
            return f"{fnum(v / 1e3, 1)} тыс. шт."
        return f"{fnum(v, 0)} шт."
    if round(a / 1e6, 2) >= 1:
        return f"{fnum(v / 1e6, 2)} тыс. т (млн л)"
    if round(a / 1e3, 1) >= 1:
        return f"{fnum(v / 1e3, 1)} т (тыс. л)"
    return f"{fnum(v, 0)} кг (л)"


def fpct(v, digits=1, sign=False):
    if v is None or v != v:
        return "—"
    s = fnum(v * 100, digits) + "%"
    return ("+" + s) if sign and v > 0 else s


def fpp(v, digits=2):
    if v is None or v != v:
        return "—"
    s = fnum(v * 100, digits) + " п.п."
    return ("+" + s) if v > 0 else s


def growth(cur, prev):
    """Growth rate. None when the base is missing, zero or negative (returns): a percentage is meaningless then."""
    if prev is None or cur is None or prev != prev or cur != cur or prev <= 0:
        return None
    return cur / prev - 1


def fdyn(g, digits=1):
    """Change in words: «рост на 3,5%» / «снижение на 3,3%» (up / down), so the model never writes «снижение роста (−3,3%)»."""
    if g is None or g != g:
        return None
    if round(g * 100, digits) == 0:
        return "без изменений"
    return ("рост на " if g > 0 else "снижение на ") + fpct(abs(g), digits)


def small_base(cur, prev, total_prev=None):
    """Base too small for a meaningful growth percentage: a new item, an SKU relaunch, a network with a partial year.
    Rule: base <= 0, or base below 10% of the current value, or below 0.05% of the sample total for the base period."""
    if prev is None or prev != prev or prev <= 0:
        return True
    if cur is not None and cur == cur and prev < 0.1 * cur:
        return True
    return bool(total_prev) and prev < 0.0005 * total_prev


def fsmall(cur, prev, measure="rub"):
    """Label shown instead of a percentage when the base is small."""
    if prev is None or prev != prev or prev <= 0:
        return "новый в периоде (базы нет)"
    k = cur / prev if prev else None
    return f"база мала ({fmoney(prev, measure)} в базовом периоде), рост в {fnum(k, 1)} раза не показателен"
