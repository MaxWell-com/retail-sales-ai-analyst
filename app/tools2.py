"""Analysis tools, part 2: factor analysis, distribution, prices, ABC, assortment rotation, forecast, competitors.

Same methodology rules as tools.py: growth on the 9 comparable networks without incomplete months (growth_agg),
a store is store_key (network + store code), kids subgroups on a separate row, private labels (СТМ) flagged,
no purchase cost for Depo from 2025 on.
Facts: totals and aggregate conclusions first, long breakdowns compressed (up to 5 rows per group, up to 6 groups),
the rest goes into tables.
"""
import math

import numpy as np
import pandas as pd

from . import semantic as S
from .result import Result, ToolError
from .tools import (DIV, LEVEL, LEVEL_RU, LEVEL_GEN, UNIT, MEASURE_RU, agg, growth_agg, comparable_note, net_lbl,
                    base_note, no_base_reason, coverage_gap, get_measure, growth_pair, main_period, make_scope,
                    kids_split, mayak_other_note, _periods, _blank, _label, _dyn, _fdelta, _g, _is_stm, _labels,
                    _man_text, _obj_name, _stable, _next_args)

DEPO = "Депо Кэш Энд Керри"
GEO = ("network", "banner", "format", "district", "region")     # store-level breakdowns
GEO_ONE = {"network": "сеть", "banner": "баннер", "format": "формат", "district": "округ", "region": "регион"}
GEO_DAT = {"network": "сетям", "banner": "баннерам", "format": "форматам", "district": "округам", "region": "регионам"}
GEO_PL = {"network": "сети", "banner": "баннеры", "format": "форматы", "district": "округа", "region": "регионы"}
ALL = "__all__"                                                  # total across all groups in _dist_stats
_bnet = {}


# ——— shared helpers ———

def _empty(scope, period):
    return ToolError(f"В выборке ({scope.describe()}) нет продаж за {period.label()}.")


def _cnt(v):
    """Counter for facts: «21 328»."""
    return S.fnum(float(v or 0), 0)


def _pl(n, one, few, many):
    """Russian plural agreement: «1 подгруппа», «4 подгруппы», «7 подгрупп»."""
    n = abs(int(n)) % 100
    if 11 <= n <= 19:
        return many
    n %= 10
    return one if n == 1 else (few if 2 <= n <= 4 else many)


def _of(n):
    """Out-of-N phrase: «из 21 точки», «из 25 точек»."""
    return f"из {_cnt(n)} {_pl(n, 'точки', 'точек', 'точек')}"


def _num2(v):
    return S.fnum(v, 2) if v is not None and v == v else "нет"


def _fpp(v):
    """Share change in percentage points; rounds to zero as «0,00 п.п.» (never «−0,00»)."""
    return "0,00 п.п." if v is None or v != v or round(v * 100, 2) == 0 else S.fpp(v)


def _gw(cur, prev):
    """Change in words for unitless quantities (stores, SKU per store, price): «рост на 3,7%»."""
    g = S.growth(cur, prev)
    return S.fdyn(g) if g is not None else "рост не считается (база нулевая)"


def _fper(v, m):
    """Sales per store: «4 912 руб.», «2,38 шт.»."""
    if v is None or v != v:
        return "нет продаж"
    return f"{S.fnum(v, 0)} руб." if m == "rub" else f"{S.fnum(v, 2)} {'шт.' if m == 'qty' else 'кг (л)'}"


def _pair(d1, d0, keys):
    """Merge both sides of a comparison into one table: keys, v1, v0 (missing row -> 0); rows with no sales in either period dropped."""
    d = d1[keys + ["v"]].merge(d0[keys + ["v"]], on=keys, how="outer", suffixes=("1", "0"))
    d[["v1", "v0"]] = d[["v1", "v0"]].fillna(0.0)
    return d[(d.v1 != 0) | (d.v0 != 0)].reset_index(drop=True)


def _pair_ex(gcur, gbase):
    """Incomplete network months for a period pair (as in growth_agg): the month is removed for that network on both sides."""
    bad = S.bad_pairs()
    ex1, ex0 = [], []
    for c, b in zip(gcur.months, gbase.months):
        for n in sorted({n for n, x in bad if x in (c, b)}):
            ex1.append((n, c))
            ex0.append((n, b))
    return ex1, ex0


def _banner_net(b):
    if not _bnet:
        _bnet.update(dict(S.q("SELECT DISTINCT banner, network FROM store_x WHERE banner IS NOT NULL")
                          .itertuples(index=False, name=None)))
    return _bnet.get(b, b)


def _stores_desc(scope):
    d = scope.describe(S.STORE_KEYS)
    return "все сети" if d == "вся панель" else d


def _own_subgroups(scope, period, m="rub", base=None):
    """Object's subgroups above 1% of its sales — its categories when no category is given (as in profile).
    base — fallback period: the object has stopped selling but its categories are still needed (factor analysis)."""
    if not scope.has_object() or scope.portal or any(scope.values(k) for k in S.CATEGORY_KEYS):
        return []
    for per in (period, base):
        if per is None:
            continue
        sg = agg(scope, per, ["subgroup_base"], m)
        sg = sg[sg.subgroup_base.notna() & (sg.v > 0)]
        sg = sg[sg.v > 0.01 * sg.v.sum()]
        if len(sg):
            return list(sg.sort_values("v", ascending=False).subgroup_base)
    return []


def _kids_pair(scope):
    """The scope's categories contain a kids pair: one subgroup_base has both kids and adult subgroups (rule 2.4)."""
    w = scope.sku_where()
    return bool(S.q1(f"""SELECT count(*) FROM (SELECT subgroup_base FROM sku {('WHERE ' + w) if w else ''}
                           GROUP BY 1 HAVING bool_or(is_kids) AND bool_or(NOT is_kids))""")[0])


# ——— factor analysis ———

def _store_stats(scope, period, comparable_only=True, by="network", ex=()):
    """Monthly by network (or banner) plus a total (g = ALL): stores with sales (store_key), store × SKU pairs,
    sales over rows with qty > 0. comparable_only — only the 9 comparable networks; ex — (network, month) pairs
    removed from the comparison as incomplete. The total is computed separately: one store can be in two banners."""
    cond = ["qty > 0"]
    if comparable_only:
        cond.append("network IN (" + ", ".join(S._lit(x) for x in S.COMPARABLE) + ")")
    cond += [f"NOT (network = {S._lit(n)} AND month = DATE '{d}')" for n, d in ex]
    df = S.q(f"""SELECT g, month, GROUPING(g) AS gg, count(DISTINCT store_key) stores, count(DISTINCT (store_key, sku_id)) pairs,
                        sum(rub) rub, sum(qty) qty, sum(kg) kg
                 FROM (SELECT {by} AS g, month, store_key, sku_id, rub, qty, kg
                       FROM {S.fact_source(scope, period)} WHERE {' AND '.join(cond)})
                 GROUP BY GROUPING SETS ((g, month), (month))""")
    df["g"] = df.g.where(df.gg == 0, ALL)
    return df


def _lmdi(name, v1, v0, a1, a0, m, nm):
    """Network sales change = stores × SKU per store × sales per store-SKU pair (LMDI, no residual) + other records.
    a1, a0 — monthly stats (_store_stats); nm — the network's number of compared months (excluding incomplete ones)."""
    p1, p0 = float(a1[m].sum()), float(a0[m].sum())
    H1, H0 = float(a1.stores.sum()), float(a0.stores.sum())      # store-months with sales
    P1, P0 = float(a1.pairs.sum()), float(a0.pairs.sum())        # store × SKU × month pairs
    row = {"name": name, "v1": v1, "v0": v0, "p1": p1, "p0": p0, "H1": H1, "H0": H0, "P1": P1, "P0": P0,
           "A1": H1 / nm if nm else 0.0, "A0": H0 / nm if nm else 0.0,
           "k1": P1 / H1 if H1 else None, "k0": P0 / H0 if H0 else None,
           "u1": p1 / P1 if P1 else None, "u0": p0 / P0 if P0 else None, "e_o": (v1 - p1) - (v0 - p0)}
    if p1 > 0 and p0 > 0 and H1 > 0 and H0 > 0:
        L = (p1 - p0) / (math.log(p1) - math.log(p0)) if abs(p1 - p0) > 1e-9 * p0 else p1
        row.update(e_st=L * math.log(H1 / H0), e_w=L * math.log(row["k1"] / row["k0"]),
                   e_p=L * math.log(row["u1"] / row["u0"]))
    else:
        row.update(e_st=p1 - p0, e_w=0.0, e_p=0.0)
    return row


def t_factors(a):
    """Factor analysis of the sales change between two periods: networks with incomplete coverage, stores, SKU per store,
    sales per store × SKU pair (LMDI per network), market and share, contributions of brands, subgroups, SKUs and districts.
    Compared on the 9 comparable networks without incomplete months (methodology 3.2, 3.7) or on the selected networks."""
    scope = make_scope(a)
    m = get_measure(a)
    unit_fmt, unit = UNIT[m]
    _, _, gcur, gbase = _periods(a, "ytd")
    if not gbase:
        raise ToolError(f"Факторный анализ сравнивает два периода, а базы нет: {no_base_reason(gcur)}.")
    gap = coverage_gap(scope, gcur, gbase)
    if gap:
        raise ToolError(f"Факторный анализ не считается: {no_base_reason(gcur, gap)}.")
    cmp_only = not scope.has_network_filter()
    where = net_lbl(scope)
    # decomposition unit: network; with a banner filter or Granit alone — banner (Mayak Kosmetik on its own row, rule 5)
    ucol = "banner" if scope.values("banner") or S.scope_networks(scope) == ["Гранит"] else "network"
    uname = GEO_ONE[ucol]
    obj = _obj_name(scope)
    who = obj or "выборки"
    vs = gcur.vs(gbase)
    r = Result("factors", "Факторный анализ изменения продаж")
    r.subtitle = f"{scope.describe()}; {vs}"
    r.set_slice(scope, gcur, gcur, gbase, m, cmp=cmp_only)

    # network totals: both sides without incomplete months; networks with incomplete coverage are a separate factor
    n1, n0 = growth_agg(scope, gcur, gbase, [ucol], m, cmp_only=False)
    nets = _pair(n1, n0, [ucol])
    if nets.empty:
        raise ToolError(f"В выборке ({scope.describe()}) нет продаж за {gcur.label()} и {gbase.label()}.")
    is_c = nets[ucol].isin(S.COMPARABLE) if cmp_only else pd.Series(True, index=nets.index)
    nc = nets[~is_c]
    S1, S0 = float(nets.v1.sum()), float(nets.v0.sum())
    o1, o0 = float(nets[is_c].v1.sum()), float(nets[is_c].v0.sum())
    nc_eff = (S1 - S0) - (o1 - o0)
    nc_txt = ", ".join(f"{S.net_label(x.network)} {_fdelta(x.v1 - x.v0, m)}" for x in nc.itertuples() if x.v1 != x.v0)

    # per-network (per-banner) decomposition: stores × SKU per store × sales per pair
    ex1, ex0 = _pair_ex(gcur, gbase)
    st1 = _store_stats(scope, gcur, cmp_only, ucol, ex1)
    st0 = _store_stats(scope, gbase, cmp_only, ucol, ex0)
    per = []
    for x in nets[is_c].itertuples(index=False):
        g = getattr(x, ucol)
        net = _banner_net(g) if ucol == "banner" else g
        per.append(_lmdi(_label(ucol, g), x.v1, x.v0, st1[st1.g == g], st0[st0.g == g], m,
                         gcur.n - sum(1 for n, _ in ex1 if n == net)))
    tot = {k: sum(x[k] for x in per) for k in ("e_st", "e_w", "e_p", "e_o")}     # contributions are summed over networks
    for sfx, st, n in (("1", st1, gcur.n), ("0", st0, gbase.n)):                  # store totals without double counting
        a_ = st[st.g == ALL]
        tot.update({"H" + sfx: float(a_.stores.sum()), "P" + sfx: float(a_.pairs.sum()), "p" + sfx: float(a_[m].sum())})
        tot["A" + sfx] = tot["H" + sfx] / n
    k1t = tot["P1"] / tot["H1"] if tot["H1"] else None
    k0t = tot["P0"] / tot["H0"] if tot["H0"] else None
    u1t = tot["p1"] / tot["P1"] if tot["P1"] else None
    u0t = tot["p0"] / tot["P0"] if tot["P0"] else None
    fac = [{"f": "Сети с неполным покрытием", "d": nc_eff, "c": nc_txt or "нет"}] if cmp_only else []
    fac += [{"f": "Число точек с продажами", "d": tot["e_st"], "c": f"в среднем в месяц {_cnt(tot['A0'])} → {_cnt(tot['A1'])}"},
            {"f": "Ширина полки (SKU на точку)", "d": tot["e_w"], "c": f"{_num2(k0t)} → {_num2(k1t)} SKU на точку"},
            {"f": "Продажи одной пары точка × SKU", "d": tot["e_p"], "c": f"{_gw(u1t, u0t)} на пару в месяц"},
            {"f": "Прочие записи", "d": tot["e_o"], "c": "строки без продаж в штуках: нулевые и отрицательные (возвраты)"}]
    for x in fac:
        x["p"] = x["d"] / S0 if S0 > 0 else None

    r.fact(f"Продажи {who} {vs} ({where}): {S.fmoney(o0, m)} → {S.fmoney(o1, m)}, {_dyn(o1, o0, m)} ({_fdelta(o1 - o0, m)}).", key=True)
    if cmp_only and len(nc):
        r.fact(f"С учетом сетей с неполным покрытием (все сети): {S.fmoney(S0, m)} → {S.fmoney(S1, m)}, {_dyn(S1, S0, m)} "
               f"({_fdelta(S1 - S0, m)}); сети с неполным покрытием дали {_fdelta(nc_eff, m)}"
               + (f" ({nc_txt})" if nc_txt else "") + ": это граница данных, а не изменение продаж.")
    if tot["H1"] and tot["H0"]:
        r.fact(f"Разложение изменения ({where}): число точек с продажами {_fdelta(tot['e_st'], m)} (в среднем "
               f"{_cnt(tot['A0'])} → {_cnt(tot['A1'])} точек в месяц, {_gw(tot['A1'], tot['A0'])}); SKU на точку "
               f"{_fdelta(tot['e_w'], m)} ({_num2(k0t)} → {_num2(k1t)}); продажи одной пары точка × SKU {_fdelta(tot['e_p'], m)} "
               f"({_gw(u1t, u0t)}); прочие записи {_fdelta(tot['e_o'], m)}; сумма {_fdelta(o1 - o0, m)}.", key=True)
    subs = _factors_market(r, scope, gcur, gbase, m, cmp_only, where, obj, o1, o0)
    per.sort(key=lambda x: -abs(x["v1"] - x["v0"]))
    if len(per) > 1:
        up, down = sum(x["v1"] > x["v0"] for x in per), sum(x["v1"] < x["v0"] for x in per)
        r.fact(f"По {'баннерам' if ucol == 'banner' else 'сетям'} ({where}): растут {up}, снижаются {down}; крупнейшие изменения: "
               + "; ".join(f"{x['name']} {_fdelta(x['v1'] - x['v0'], m)} ({_dyn(x['v1'], x['v0'], m, o0)})" for x in per[:4]) + ".")

    # decomposition tables and chart
    total = {"f": "Итого" + (" (все сети)" if cmp_only else ""), "d": S1 - S0, "p": (S1 - S0) / S0 if S0 > 0 else None,
             "c": f"{S.fmoney(S0, m)} → {S.fmoney(S1, m)}", "_cls": "tot"}
    r.kpis([(f"Продажи, {gbase.label()}", S.fmoney(o0, m), where, None),
            (f"Продажи, {gcur.label()}", S.fmoney(o1, m), where, None),
            ("Изменение", S.fpct(_g(o1, o0), sign=True) if _g(o1, o0) is not None else "не считается", _fdelta(o1 - o0, m), o1 - o0)]
           + ([("Все сети", _fdelta(S1 - S0, m), f"{S.fmoney(S0, m)} → {S.fmoney(S1, m)}", S1 - S0)] if cmp_only and len(nc) else []))
    r.table("Разложение изменения продаж", [("f", "Фактор", "text"), ("d", f"Вклад, {unit}", unit_fmt),
                                            ("p", "В % к базе", "gpct"), ("c", "Что за этим стоит", "txt")], fac + [total])
    div = DIV[m]
    r.chart("waterfall", f"От {gbase.label()} к {gcur.label()}, {unit}", [gbase.label()] + [x["f"] for x in fac] + [gcur.label()],
            [{"name": "Вклад", "values": [S0 / div] + [x["d"] / div for x in fac] + [S1 / div]}], fmt="num1", unit=unit)
    r.table(f"Разложение по {'баннерам' if ucol == 'banner' else 'сетям'} ({where})", [
        ("name", LEVEL_RU[ucol], "text"), ("v0", f"База, {unit}", unit_fmt), ("v1", f"Текущий, {unit}", unit_fmt),
        ("A0", "Точек в мес., база", "num0"), ("A1", "Точек в мес.", "num0"), ("k0", "SKU на точку, база", "num2"),
        ("k1", "SKU на точку", "num2"), ("e_st", f"Точки, {unit}", unit_fmt), ("e_w", f"Ширина полки, {unit}", unit_fmt),
        ("e_p", f"Продажи пары, {unit}", unit_fmt), ("e_o", f"Прочие, {unit}", unit_fmt)], per)

    # breakdown contributions and SKU lifecycle — on the same comparison base
    nc_row = {"name": "Сети с неполным покрытием", "v0": float(nc.v0.sum()), "v1": float(nc.v1.sum()), "d": nc_eff,
              "_cls": "lvl"} if cmp_only and len(nc) else None
    for level, title in (("brand", "брендам"), ("subgroup", "подгруппам")):
        _factors_cut(r, scope, gcur, gbase, m, cmp_only, where, level, title, nc_row)
    sku_details = _factors_sku(r, scope, gcur, gbase, m, cmp_only, where, nc_row)
    for x in per[:3] if len(per) > 1 else []:
        if x["H1"] and x["H0"]:
            r.fact(f"{uname} {x['name']}: {S.fmoney(x['v0'], m)} → {S.fmoney(x['v1'], m)}; вклад точек {_fdelta(x['e_st'], m)} "
                   f"({_cnt(x['A0'])} → {_cnt(x['A1'])} в месяц), SKU на точку {_fdelta(x['e_w'], m)} ({_num2(x['k0'])} → {_num2(x['k1'])}), "
                   f"продаж пары точка × SKU {_fdelta(x['e_p'], m)}.")
    for f in sku_details:
        r.fact(f)
    _factors_cut(r, scope, gcur, gbase, m, cmp_only, where, "district", "округам", nc_row)
    if obj:
        _factors_kids(r, scope, gcur, gbase, m, where, obj)
    if gcur.n > 1 and gbase == gcur.shift(-12):
        mm1, mm0 = growth_agg(scope, gcur, gbase, ["month"], m, cmp_only)
        v1, v0 = dict(zip(mm1.month, mm1.v)), dict(zip(mm0.month, mm0.v))
        r.fact(f"По месяцам к тем же месяцам год назад ({where}): " + "; ".join(
            f"{S.MONTHS_RU[c.month - 1]} {_dyn(v1.get(c, 0.0), v0.get(b, 0.0), m)}" for c, b in zip(gcur.months, gbase.months)) + ".")
    if subs:
        r.fact(f"Категории {obj} для доли и рынка (подгруппы с долей больше 1% его продаж): {', '.join(subs)}.")

    r.caveat(comparable_note(scope, gcur, gbase))
    bn = base_note(a, gcur, gbase)
    if bn:
        r.caveat(bn)
    r.caveat("Продажи = точки × SKU на точку × продажи пары точка × SKU (LMDI по каждой сети, без остатка); точка — сеть + код точки.")
    mo = mayak_other_note(scope, by=ucol)
    if mo:
        r.caveat(mo)
    if len(r.headline) < 2 and len(r.facts) > 1:       # object stopped selling: no decomposition, so the next fact becomes the second headline
        r.headline.append(r.facts[1])
    nxt = lambda **kw: _next_args(scope, m, a, **kw)
    if scope.values("manufacturer") or scope.values("brand"):
        r.suggest(f"Обзор {obj}: доля, бренды, сети", "profile", nxt())
    r.suggest(f"Цены {who}: {vs}", "prices", nxt(period=gcur.to_str(), base_period=gbase.to_str(), by="network"))
    r.suggest(f"Представленность {who} по сетям", "distribution", nxt(period=None, base_period=None, by="network"))
    if scope.values("manufacturer") or scope.values("brand"):
        r.suggest(f"Конкуренты {obj} в его сегментах", "competitors", nxt(period=None, base_period=None))
    return r


def _factors_market(r, scope, gcur, gbase, m, cmp_only, where, obj, o1, o0):
    """Market in the object's categories and its share: what the object's sales change would be at market pace (methodology 3).
    Returns the object's subgroups when no category is given (listed as a separate fact at the end)."""
    if not obj:
        return []
    market, obj_mk = scope.without(*S.OBJECT_KEYS), scope
    subs = _own_subgroups(scope, gcur, m, base=gbase)
    if subs:
        market, obj_mk = market.with_(subgroup=subs), scope.with_(subgroup=subs)
    mk1, mk0 = (float(x.v.sum()) for x in growth_agg(market, gcur, gbase, [], m, cmp_only))
    om1, om0 = (float(x.v.sum()) for x in growth_agg(obj_mk, gcur, gbase, [], m, cmp_only))
    gm = S.growth(mk1, mk0)
    if gm is None or mk1 <= 0:
        return subs
    cats = f"{len(subs)} {_pl(len(subs), 'подгруппа', 'подгруппы', 'подгрупп')} объекта" if subs else S.cat_desc(scope)
    s0, s1 = om0 / mk0, om1 / mk1
    exp = o0 * gm
    diff = (o1 - o0) - exp
    if abs(diff) < 0.0005 * max(abs(o0), 1.0):
        cmp_txt = "то есть на уровне рынка"
    else:
        cmp_txt = (f"то есть на {S.fmoney(abs(diff), m)} {'выше' if diff > 0 else 'ниже'} рынка "
                   f"({'выигрыш' if diff > 0 else 'потеря'} доли)")
    r.fact(f"Рынок в категориях {obj} ({cats}; {where}) {gcur.vs(gbase)}: {_dyn(mk1, mk0, m)} ({S.fmoney(mk0, m)} → "
           f"{S.fmoney(mk1, m)}); доля {obj} в этих категориях {S.fpct(s0, 2)} → {S.fpct(s1, 2)} ({_fpp(s1 - s0)}). "
           f"При темпе рынка изменение продаж {obj} было бы {_fdelta(exp, m)}, фактическое {_fdelta(o1 - o0, m)}, {cmp_txt}.", key=True)
    return subs


def _factors_cut(r, scope, gcur, gbase, m, cmp_only, where, level, title, nc_row):
    """Contribution of a breakdown (brands, subgroups, districts) to the change: full table; facts list the largest changes."""
    unit_fmt, unit = UNIT[m]
    d1, d0 = growth_agg(scope, gcur, gbase, [level], m, cmp_only)
    d = _pair(d1, d0, [level])
    d = d[~d[level].map(_blank)]
    if len(d) < 2:
        return
    d["d"] = d.v1 - d.v0
    d = d.reindex(d.d.abs().sort_values(ascending=False).index)
    T0 = float(d.v0.sum())
    rows = [{"name": _label(level, getattr(x, level)), "v0": x.v0, "v1": x.v1, "d": x.d, "g": _g(x.v1, x.v0)} for x in d.itertuples()]
    r.table(f"Вклад в изменение по {title} ({where})", [
        ("name", LEVEL_RU[level], "text"), ("v0", f"База, {unit}", unit_fmt), ("v1", f"Текущий, {unit}", unit_fmt),
        ("d", f"Изменение, {unit}", unit_fmt), ("g", "Рост", "gpct")], rows[:30] + ([nc_row] if nc_row else []))
    up, down = int((d.d > 0).sum()), int((d.d < 0).sum())
    r.fact(f"По {title} ({where}): {LEVEL_GEN[level]} с продажами {len(d)}, растут {up}, снижаются {down}; крупнейшие изменения: "
           + "; ".join(f"{x['name']} {_fdelta(x['d'], m)} ({_dyn(x['v1'], x['v0'], m, T0)})" for x in rows[:4]) + ".")


def _factors_sku(r, scope, gcur, gbase, m, cmp_only, where, nc_row):
    """SKU lifecycle: sold in both periods, delisted, new (same comparison base as the total).
    The summary is written immediately; the largest SKUs per group are returned as strings for the facts below."""
    unit_fmt, unit = UNIT[m]
    s1, s0 = growth_agg(scope, gcur, gbase, ["sku"], m, cmp_only)
    sk = _pair(s1, s0, ["sku"])
    if sk.empty:
        return []
    sk["d"] = sk.v1 - sk.v0
    groups = [("продавались в обоих периодах", "у SKU, продававшихся в обоих периодах", sk[(sk.v1 > 0) & (sk.v0 > 0)]),
              ("выведены (продажи были в базовом периоде, в текущем нет)", "у выведенных SKU", sk[(sk.v1 <= 0) & (sk.v0 > 0)]),
              ("новые (не было в базе)", "у новых SKU", sk[(sk.v1 > 0) & (sk.v0 <= 0)])]
    rows = []
    for g, who, d in groups:
        top = d.reindex(d.d.abs().sort_values(ascending=False).index).head(3)
        rows.append({"g": g, "who": who, "n": len(d), "d": float(d.d.sum()),
                     "txt": "; ".join(f"{x.sku} ({_fdelta(x.d, m)})" for x in top.itertuples())})
    r.table(f"Жизненный цикл SKU ({where})", [("g", "Группа SKU", "text"), ("n", "Число SKU", "int"),
                                             ("d", f"Вклад, {unit}", unit_fmt), ("top", "Крупнейшие изменения", "txt")],
            [{"g": x["g"], "n": x["n"], "d": x["d"], "top": x["txt"]} for x in rows]
            + ([{"g": nc_row["name"], "d": nc_row["d"], "_cls": "lvl"}] if nc_row else []))
    r.fact(f"Жизненный цикл SKU ({where}, SKU по названию): " + "; ".join(f"{x['g']} {x['n']} SKU, вклад {_fdelta(x['d'], m)}"
                                                                         for x in rows) + ".")
    return [f"Крупнейшие изменения {x['who']}: {x['txt']}." for x in rows if x["n"]]


def _factors_kids(r, scope, gcur, gbase, m, where, obj):
    """Rule 2.4, compressed: one line per category with a kids pair — adult / kids / total share and its change."""
    rows = kids_split(scope, gcur, m, base=gbase)
    if not rows:
        return
    unit_fmt, unit = UNIT[m]
    for x in rows:
        x["share1"] = x["share0"] + x["dshare"] if x.get("share0") is not None and x.get("dshare") is not None else None
    r.table(f"Доля {obj} в категориях: взрослые и детские ({where})", [
        ("cat", "Категория", "text"), ("market", f"Рынок, {gcur.label()}, {unit}", unit_fmt), ("obj", f"{obj}, {unit}", unit_fmt),
        ("share0", f"Доля, {gbase.label()}", "pct2"), ("share1", f"Доля, {gcur.label()}", "pct2"), ("dshare", "Изм. доли", "pp")], rows)

    def part(x, nm):
        if x.get("share1") is not None:
            return f"{nm} {S.fpct(x['share0'], 2)} → {S.fpct(x['share1'], 2)} ({_fpp(x['dshare'])})"
        if x.get("share") is not None:
            return f"{nm} {S.fpct(x['share'], 2)} за {gcur.label()} (изменение не считается)"
        return f"{nm}: продаж рынка нет"
    for i in range(0, len(rows) - 2, 3):
        ad, kd, tt = rows[i:i + 3]
        r.fact(f"Доля {obj} в категории {tt['cat'].rsplit(':', 1)[0]} ({where}, {gcur.vs(gbase)}): {part(ad, 'взрослые')}; "
               f"{part(kd, 'детские')}; {part(tt, 'итого')}.")


# ——— distribution ———

def _dist_stats(sc, period, by, split):
    """Stores with sales by group and category part (adult / kids / all). g = ALL is the total across groups.
    stores — distinct stores over the period (store_key); sm — store-months with sales (for sales per store per month)."""
    part = "CASE WHEN k.is_kids THEN 'kids' ELSE 'adult' END" if split else "'all'"
    sets, psel = ("(g, part), (g), (part), ()", "part, GROUPING(part) AS gp") if split else ("(g), ()", "'all' AS part, 1 AS gp")
    df = S.q(f"""SELECT g, {psel}, GROUPING(g) AS gg, count(DISTINCT store_key) stores,
                        count(DISTINCT (store_key, month)) sm, sum(rub) rub, sum(qty) qty, sum(kg) kg
                 FROM (SELECT f.{by} AS g, {part} AS part, f.store_key, f.month, f.rub, f.qty, f.kg
                       FROM {S.fact_source(sc, period)} f {'JOIN sku k USING (sku_id)' if split else ''} WHERE f.qty > 0)
                 GROUP BY GROUPING SETS ({sets})""")
    df["g"] = df.g.where(df.gg == 0, ALL)
    df["part"] = df.part.where(df.gp == 0, "all")
    return df.drop(columns=["gg", "gp"])


def t_distribution(a):
    """Numeric distribution: the share of stores selling the object's category that also sell the object itself;
    potential — stores with the category but without the object. Store = store_key; kids and adult subgroups separately (rule 2.4)."""
    scope = make_scope(a)
    m = get_measure(a)
    level, by = a.get("level"), a.get("by")
    if level in GEO:                      # «дистрибуция по сетям» ("distribution by network") with level=network is a breakdown, not a product level
        by, level = (level if not by or by == "network" else by), None
    if level and (level not in LEVEL or level in ("month", "city")):
        raise ToolError(f"level должен быть уровнем товара: {', '.join(k for k in LEVEL if k not in GEO + ('month', 'city'))}.")
    by = by if by in GEO else "network"
    if by == "network" and scope.values("banner"):
        by = "banner"
    period, explicit = main_period(a, "last_month")
    if not explicit:
        d = S.last_ok_month(S.scope_networks(scope))
        period = S.Period(d, d)
    top = max(1, int(a.get("top") or 20))
    has_obj = scope.has_object()
    num_sc = scope
    subs = _own_subgroups(scope, period)
    if has_obj:
        cat = scope.without(*S.OBJECT_KEYS)
        if subs:
            cat, num_sc = cat.with_(subgroup=subs), scope.with_(subgroup=subs)
    else:                                 # no object: the category among all of the network's stores with sales
        cat = S.Scope({k: v for k, v in scope.f.items() if k in S.STORE_KEYS},
                      ex={k: v for k, v in scope.ex.items() if k in S.STORE_KEYS}, universe=scope.universe)
    split = has_obj and scope.kids is None and _kids_pair(cat)
    obj = _obj_name(scope) or scope.describe()
    gl = (lambda v: S.net_label(v)) if by == "network" else (lambda v: "не указан" if _blank(v) else str(v))
    r = Result("distribution", "Представленность в точках (числовая дистрибуция)")
    r.subtitle = f"{scope.describe()}; {period.label()}; знаменатель — точки, где продается {S.cat_desc(cat) if has_obj else 'что-либо из панели'}"
    r.set_slice(scope, period, None, None, m, note=f"представленность по {GEO_DAT[by]}")

    num = _dist_stats(num_sc, period, by, split)
    den = _dist_stats(cat, period, by, split)
    t = den[["g", "part", "stores"]].rename(columns={"stores": "den"}).merge(num, on=["g", "part"], how="left")
    t[["stores", "sm", "rub", "qty", "kg"]] = t[["stores", "sm", "rub", "qty", "kg"]].fillna(0)
    t["pct"] = t.stores / t.den.where(t.den > 0)
    t["pot"] = t.den - t.stores
    t["per"] = t[m] / t.sm.where(t.sm > 0)
    tot = t[(t.g == ALL) & (t.part == "all")]
    if tot.empty or not tot.stores.iloc[0]:
        raise _empty(scope, period)
    tot = tot.iloc[0]
    groups = t[(t.g != ALL) & (t.part == "all") & (t.den > 0)].sort_values("den", ascending=False)

    def part_txt(g, nm, kind):
        x = t[(t.g == g) & (t.part == kind)]
        if x.empty or not x.den.iloc[0]:
            return f"{nm}: точек с категорией нет"
        x = x.iloc[0]
        if not x.stores:
            return f"{nm}: объекта нет ни в одной {_of(x.den)}"
        return f"{nm} {_cnt(x.stores)} {_of(x.den)} ({S.fpct(x.pct)})"

    def line(x):
        if not x.stores:
            return f"объекта нет ни в одной {_of(x.den)} с категорией"
        s = (f"объект в {_cnt(x.stores)} {_of(x.den)} с категорией ({S.fpct(x.pct)}), потенциал {_cnt(x.pot)} "
             f"{_pl(x.pot, 'точка', 'точки', 'точек')}"
             f"{'; ' + part_txt(x.g, 'взрослые подгруппы', 'adult') + '; ' + part_txt(x.g, 'детские', 'kids') if split else ''}")
        return s + f"; в среднем {_fper(x.per, m)} на точку с продажами в месяц"

    r.fact(f"Итого ({_stores_desc(scope)}), {period.label()}: {line(tot)}; продажи объекта {S.fmoney(tot[m], m)}.", key=True)
    if not has_obj:
        r.fact(f"Объекта (производителя, бренда, SKU) в срезе нет: считается доля точек с продажами выборки ({scope.describe()}) "
               "среди всех точек сети с продажами.", key=True)
    else:
        r.fact(f"Объект: {obj}; категория для знаменателя: {S.cat_desc(cat)}"
               + (" (подгруппы объекта с долей больше 1% его продаж; продажи объекта в других подгруппах не учитываются)" if subs else "") + ".", key=True)
    if len(groups) > 1:
        ok = groups[groups.den >= 10]
        if len(ok) > 1:
            hi_, lo_ = ok.loc[ok.pct.idxmax()], ok.loc[ok.pct.idxmin()]
            r.fact(f"По {GEO_DAT[by]} с категорией ({len(groups)}, из них {int((groups.stores > 0).sum())} с объектом): выше всего доля точек "
                   f"у {gl(hi_.g)} ({S.fpct(hi_.pct)}, {_cnt(hi_.stores)} {_of(hi_.den)}), ниже всего у {gl(lo_.g)} "
                   f"({S.fpct(lo_.pct)}, {_cnt(lo_.stores)} {_of(lo_.den)}); сравнивались группы, где не меньше 10 точек с категорией.", key=True)
        for x in groups.head(6).itertuples():
            r.fact(f"{GEO_ONE[by]} {gl(x.g)}: {line(x)}.")
        if len(groups) > 6:
            r.fact(f"Остальные {GEO_PL[by]} ({len(groups) - 6}): в таблице интерфейса.")
    rows = []
    for x in groups.head(top).itertuples():
        rows.append({"g": gl(x.g), "den": x.den, "stores": x.stores, "pct": x.pct, "pot": x.pot, "per": x.per, "_cls": "lvl" if split else None})
        if split:
            for kind, nm in (("adult", "взрослые"), ("kids", "детские")):
                y = t[(t.g == x.g) & (t.part == kind)]
                if len(y):
                    y = y.iloc[0]
                    rows.append({"g": f"   {nm}", "den": y.den, "stores": y.stores, "pct": y.pct, "pot": y.pot, "per": y.per})
    rows.append({"g": "Итого", "den": tot.den, "stores": tot.stores, "pct": tot.pct, "pot": tot.pot, "per": tot.per, "_cls": "tot"})
    per_lbl = {"rub": "руб.", "qty": "шт.", "kg": "кг (л)"}[m]
    r.table(f"Представленность по {GEO_DAT[by]}, {period.label()}", [
        ("g", LEVEL_RU[by], "text"), ("den", "Точек с категорией", "int"), ("stores", "Точек с объектом", "int"),
        ("pct", "Доля точек", "heat"), ("pot", "Потенциал, точек", "int"), ("per", f"На точку с продажами в мес., {per_lbl}", "num0" if m == "rub" else "num2")], rows)
    r.chart("bar", "Доля точек с объектом", [gl(x.g) for x in groups.head(12).itertuples()],
            [{"name": "Доля точек", "values": [x.pct for x in groups.head(12).itertuples()]}], fmt="pct")
    if level:
        _dist_level(r, scope, num_sc, period, by, level, t, gl)
    r.caveat("Точка — сеть + код точки (store_key) с продажами в штуках больше нуля; за период длиннее месяца точка считается один раз. "
             "Продажи на точку — на точку с продажами в месяц: продажи / сумма помесячных точек с продажами.")
    r.caveat("Потенциал — точки, где категория продается, а объекта нет; это верхняя граница, а не прогноз.")
    if not explicit:
        r.caveat(f"Период по умолчанию — {period.label()}: последний месяц, полный у сетей среза (методика п. 5.5).")
    mo = mayak_other_note(scope, by=by)
    if mo:
        r.caveat(mo)
    nxt = lambda **kw: _next_args(scope, m, a, **kw)
    nets = S.scope_networks(scope) or []
    if by == "network" and (not nets or "Гранит" in nets):
        r.suggest("Представленность по баннерам (Маяк Косметик и другие отдельно)", "distribution", nxt(by="banner", level=level))
    if not level:
        lv = "brand" if scope.values("manufacturer") else ("sku" if has_obj else "manufacturer")
        r.suggest(f"Представленность {LEVEL_GEN[lv]}{' ' + obj if has_obj else ''} по {GEO_DAT[by]}", "distribution", nxt(by=by, level=lv))
    if has_obj:
        r.suggest(f"Факторный анализ продаж {obj}", "factors", nxt(period=None, base_period=None))
    else:
        r.suggest("Рейтинг производителей выборки", "ranking", nxt(level="manufacturer", period=None))
    r.suggest(f"Цены {obj if has_obj else 'выборки'} по {GEO_DAT[by]}", "prices", nxt(by=by))
    return r


def _dist_level(r, scope, num_sc, period, by, level, t, gl):
    """Level-object × group matrix: share of the group's stores with the category that sell the brand (SKU, etc.)."""
    df = S.q(f"""SELECT g, name, GROUPING(g) AS gg, count(DISTINCT store_key) stores, sum(rub) rub
                 FROM (SELECT f.{by} AS g, {LEVEL[level]} AS name, f.store_key, f.rub
                       FROM {S.fact_source(num_sc, period)} f JOIN sku k USING (sku_id) WHERE f.qty > 0)
                 GROUP BY GROUPING SETS ((g, name), (name))""")
    df = df[~df.name.map(_blank)]
    if df.empty:
        return
    den = dict(zip(t[t.part == "all"].g, t[t.part == "all"].den))
    names = df[df.gg == 1].sort_values(["rub", "name"], ascending=[False, True]).name.tolist()
    info = _labels(scope, period, level, names[:15])
    lbl = lambda n: f"{n}{' [СТМ]' if _is_stm(level, n, info) else ''}"
    part = df[df.gg == 0].copy()
    part["den"] = part.g.map(den)
    part["pct"] = part.stores / part.den
    gorder = t[(t.g != ALL) & (t.part == "all")].sort_values("den", ascending=False).g.tolist()[:12]
    rows = []
    for nm in names[:15]:
        d = part[part.name == nm]
        row = {"name": lbl(nm), "all": float(df[(df.gg == 1) & (df.name == nm)].stores.iloc[0])}
        for i, g in enumerate(gorder):
            v = d[d.g == g]
            row[f"g{i}"] = float(v.pct.iloc[0]) if len(v) else 0.0
        rows.append(row)
    r.table(f"Доля точек с категорией, где продается {LEVEL_RU[level].lower()}, {period.label()}",
            [("name", LEVEL_RU[level], "text"), ("all", "Точек всего", "int")] + [(f"g{i}", gl(g), "heat") for i, g in enumerate(gorder)],
            rows, note="Доля точек = точки с продажами объекта / точки с продажами категории в группе.")
    r.fact(f"{LEVEL_GEN[level].capitalize()} объекта с продажами: {len(names)}. Ниже доля точек с категорией, где продается каждый из "
           f"{min(5, len(names))} крупнейших по продажам (до 6 групп в строке; полная матрица в таблице).")
    for nm in names[:5]:
        d = part[part.name == nm].sort_values("rub", ascending=False).head(6)
        r.fact(f"{LEVEL_RU[level].lower()} {lbl(nm)}: " + "; ".join(
            f"{gl(x.g)} {S.fpct(x.pct)} ({_cnt(x.stores)} {_of(x.den)})" for x in d.itertuples() if x.den == x.den) + ".")


# ——— prices ———

def t_prices(a):
    """Retail price and the network's purchase price per unit, network markup, price per litre (kg) and price change vs base:
    average price and a price index on identical SKUs in the same networks (no assortment effect). Purchase cost is within-network only."""
    scope = make_scope(a)
    level, by = a.get("level"), a.get("by")
    if level in GEO:
        by, level = (level if not by or by == "network" else by), None
    level = level or next((k for k in ("sku", "subbrand", "brand", "manufacturer") if scope.values(k)), "manufacturer")
    if level not in LEVEL or level in GEO + ("month", "city"):
        raise ToolError(f"level должен быть уровнем товара: {', '.join(k for k in LEVEL if k not in GEO + ('month', 'city'))}.")
    by = by if by in GEO else "network"
    if by == "network" and scope.values("banner"):
        by = "banner"
    period, explicit = main_period(a, "last_month")
    gcur, gbase = growth_pair(a, period, True)          # price change over the same period as the prices
    if gcur.start > period.start:
        period = gcur
    gap = coverage_gap(scope, gcur, gbase)
    if gap:
        gbase = None
    cmp_only = not scope.has_network_filter()
    where = net_lbl(scope)
    one_net = by in ("network", "banner") or len(S.scope_networks(scope) or []) == 1
    r = Result("prices", "Цены продажи и закупки сети")
    r.subtitle = f"{scope.describe()}; {period.label()}"
    r.set_slice(scope, period, gcur if gbase else None, gbase, "rub", cmp=cmp_only,
                note=f"цены: {LEVEL_RU[by].lower()} × {LEVEL_RU[level].lower()}")
    ok = f"n.cost <> 0 AND NOT (n.network = {S._lit(DEPO)} AND n.month >= DATE '2025-01-01')"
    df = S.q(f"""SELECT {LEVEL[by]} AS g, {LEVEL[level]} AS name, sum(n.rub) rub, sum(n.qty) qty, sum(n.kg) kg,
                        sum(n.rub) FILTER (WHERE {ok}) rub_k, sum(n.qty) FILTER (WHERE {ok}) qty_k,
                        sum(n.cost) FILTER (WHERE {ok}) cost_k, sum(n.kg) FILTER (WHERE {ok}) kg_k
                 FROM {S.source(scope, period, geo=by == 'region')} n JOIN sku k USING (sku_id) GROUP BY 1, 2""")
    df[["rub_k", "qty_k", "cost_k", "kg_k"]] = df[["rub_k", "qty_k", "cost_k", "kg_k"]].fillna(0.0)
    R, Q = float(df.rub.sum()), float(df.qty.sum())
    df = df[~df.name.map(_blank) & ~df.g.map(_blank) & (df.qty > 0) & (df.rub > 0)]
    if df.empty or Q <= 0:
        raise _empty(scope, period)
    df["price"] = df.rub / df.qty
    if one_net:
        df["cost_u"] = (df.cost_k / df.qty_k).where((df.qty_k > 0) & (df.cost_k > 0))
        df["markup"] = (df.rub_k / df.cost_k - 1).where(df.cost_k > 0)
        df["cost_l"] = (df.cost_k / df.kg_k).where((df.kg_k > 0) & (df.cost_k > 0))
    else:
        df["cost_u"] = df["markup"] = df["cost_l"] = None
    df["ml"] = (1000 * df.kg / df.qty).where(df.kg > 0)
    df["price_l"] = (df.rub / df.kg).where(df.kg > 0)

    # change: both sides without incomplete months; price index on identical SKUs in the same networks (Laspeyres)
    p1g = p0g = idx_g = {}
    dyn = {}
    if gbase:
        d1, d0 = growth_agg(scope, gcur, gbase, [by, level], "rub", cmp_only)
        dd = d1[[by, level, "rub", "qty"]].merge(d0[[by, level, "rub", "qty"]], on=[by, level], suffixes=("1", "0"))
        dd = dd[(dd.qty1 > 0) & (dd.qty0 > 0) & (dd.rub1 > 0) & (dd.rub0 > 0)]
        dyn = {(x[0], x[1]): (x[2] / x[3], x[4] / x[5]) for x in dd[[by, level, "rub1", "qty1", "rub0", "qty0"]].itertuples(index=False)}
        keys = ["sku", by] if by in ("network", "banner") else ["sku", "network", by]
        i1, i0 = growth_agg(scope, gcur, gbase, keys, "rub", cmp_only)
        ii = i1[keys + ["rub", "qty"]].merge(i0[keys + ["rub", "qty"]], on=keys, suffixes=("1", "0"))
        ii = ii[(ii.qty1 > 0) & (ii.qty0 > 0) & (ii.rub1 > 0) & (ii.rub0 > 0)]

        def lasp(d):
            return float((d.rub1 / d.qty1 * d.qty0).sum() / d.rub0.sum() - 1) if len(d) and d.rub0.sum() > 0 else None
        R1, Q1, R0, Q0 = (float(x) for x in (i1.rub.sum(), i1.qty.sum(), i0.rub.sum(), i0.qty.sum()))
        P1, P0 = (R1 / Q1 if Q1 > 0 else None), (R0 / Q0 if Q0 > 0 else None)
        idx, cov = lasp(ii), (float(ii.rub0.sum()) / R0 if R0 > 0 else None)
        p1g = {g: (x.rub.sum() / x.qty.sum() if x.qty.sum() > 0 else None) for g, x in i1.groupby(by)}
        p0g = {g: (x.rub.sum() / x.qty.sum() if x.qty.sum() > 0 else None) for g, x in i0.groupby(by)}
        idx_g = {g: lasp(x) for g, x in ii.groupby(by)}

    # facts: total, change, purchase-cost rule, then groups (up to 6) and rows (up to 5)
    lv_n = df.groupby("name").rub.sum()
    r.fact(f"Всего в выборке за {period.label()} ({_stores_desc(scope)}): продажи {S.fmoney(R)}, {S.fmoney(Q, 'qty')}, "
           f"средняя цена {S.fnum(R / Q, 1)} руб./шт. (средняя цена зависит от ассортимента); {LEVEL_GEN[level]} с продажами: {len(lv_n)}.", key=True)
    if not gbase:
        r.fact(f"Динамика цен не считается: {no_base_reason(gcur, gap)}.", key=True)
    elif P1 is None or P0 is None:
        r.fact(f"Динамика цен {gcur.vs(gbase)} ({where}) не считается: нет продаж в одном из периодов.", key=True)
    else:
        s = (f"Цены {gcur.vs(gbase)} ({where}): средняя цена выборки {S.fnum(P0, 1)} → {S.fnum(P1, 1)} руб./шт., {_gw(P1, P0)}; ")
        if idx is not None:
            mix = (P1 / P0 - 1) - idx
            s += (f"на одинаковых SKU в тех же сетях (индекс цен) {S.fdyn(idx)}; разница {_fpp(mix)}: эффект ассортимента и структуры сетей; "
                  f"SKU с продажами в обоих периодах дают {S.fpct(cov)} продаж базы.")
        else:
            s += "индекс цен на одинаковых SKU не считается: нет SKU с продажами в обоих периодах."
        r.fact(s, key=True)
    if one_net:
        r.fact("Закупка за штуку и наценка сети (продажи / закупка − 1) посчитаны по строкам с закупочной стоимостью: без Депо с 01.2025 "
               "и без строк с нулевой закупкой; сравнивать их можно внутри одной сети и одного периода (методика п. 7).")
    else:
        r.fact(f"Закупка и наценка не считаются: строки по {GEO_DAT[by]} смешивают разные сети, а учет закупки у сетей разный "
               "(методика п. 7.2). Для наценки нужен разрез по сетям (by=network) или одна сеть.")
    gt = df.groupby("g").agg(rub=("rub", "sum"), qty=("qty", "sum"), n=("name", "count")).sort_values("rub", ascending=False)
    gorder = list(gt.index[:12])
    shown_g = gorder[:6]
    per_g = max(2, min(5, 12 // max(1, len(shown_g))))
    names = sorted({x for g in gorder for x in df[df.g == g].sort_values("rub", ascending=False).name.head(10)})
    info = _labels(scope, period, level, names)
    lbl = lambda n: f"{n}{' [СТМ]' if _is_stm(level, n, info) else ''}"
    gl = lambda v: _label(by, v)
    top = max(1, int(a.get("top") or 10))

    cut = {n for n, _ in _pair_ex(gcur, gbase)[0]} if gbase else set()

    def no_dyn(g):
        net = _banner_net(g) if by == "banner" else (g if by == "network" else None)
        if cmp_only and net and net not in S.COMPARABLE:
            return "не считается: сеть с неполным покрытием, динамика только по 9 сопоставимым сетям"
        if net in cut and not p1g.get(g):
            return "не считается: месяц сети загружен не полностью и исключен из сравнения"
        return "не считается: нет продаж в базе"
    rows = []
    for g in gorder:
        d = df[df.g == g].sort_values("rub", ascending=False)
        tg = float(d.rub.sum())
        hdr = {"name": gl(g), "_cls": "lvl", "rub": tg, "share": 1.0, "price": float(gt.loc[g, "rub"] / gt.loc[g, "qty"])}
        if gbase:
            hdr.update(price0=p0g.get(g), gp=S.growth(p1g.get(g), p0g.get(g)), idx=idx_g.get(g))
        rows.append(hdr)
        if g in shown_g and len(d) > 1:      # a single-row group is fully described by its row
            s = f"[{gl(g)}] всего {S.fmoney(tg)}, средняя цена {S.fnum(hdr['price'], 1)} руб./шт."
            if gbase:
                s += (f"; к {gbase.label('dat')}: средняя цена {_gw(p1g.get(g), p0g.get(g))}" if p0g.get(g) and p1g.get(g) else
                      f"; динамика к {gbase.label('dat')} {no_dyn(g)}")
                if idx_g.get(g) is not None:
                    s += f", на одинаковых SKU {S.fdyn(idx_g[g])}"
            s += f"; {LEVEL_GEN[level]} с продажами: {len(d)}"
            if len(d) > per_g:
                s += f", в фактах первые {per_g} (на них {S.fpct(d.rub.head(per_g).sum() / tg)} продаж группы), остальные в таблице"
            r.fact(s + ".")
        for i, x in enumerate(d.head(top).itertuples()):
            row = {"name": lbl(x.name), "rub": x.rub, "share": x.rub / tg if tg else None, "qty": x.qty, "price": x.price,
                   "cost_u": x.cost_u, "markup": x.markup, "ml": x.ml, "price_l": x.price_l, "cost_l": x.cost_l}
            pp = dyn.get((g, x.name))
            if pp:
                row.update(price0=pp[1], gp=pp[0] / pp[1] - 1)
            rows.append(row)
            if g not in shown_g or i >= per_g:
                continue
            s = f"{gl(g)} / {lbl(x.name)}: продажи {S.fmoney(x.rub)}"
            if len(d) > 1:
                s += f", доля в продажах группы {S.fpct(row['share'])}"
            s += f", цена {S.fnum(x.price, 1)} руб./шт."
            if gbase:
                s += f" ({_gw(*pp)} к {gbase.label('dat')})" if pp else f" (динамика {no_dyn(g)})"
                if len(d) == 1 and idx_g.get(g) is not None:
                    s += f", на одинаковых SKU цены {S.fdyn(idx_g[g])}"
            if one_net:
                if x.cost_u is None or x.cost_u != x.cost_u:
                    s += ", закупки в данных нет" + (" (Депо с 01.2025)" if g == DEPO or gl(g) == "Депо" else "")
                else:
                    # price minus purchase cost as a ready number: otherwise the model computes it itself
                    s += (f", закупка {S.fnum(x.cost_u, 1)} руб./шт., разница цены и закупки "
                          f"{S.fnum(x.price - x.cost_u, 1)} руб./шт., наценка сети {S.fpct(x.markup)}")
                    if x.markup < 0:
                        s += " (закупка выше цены продажи: особенность учета источника, не реальная наценка)"
            if x.price_l == x.price_l and x.price_l is not None:
                s += f", {S.fnum(x.price_l, 0)} руб./л(кг)"
            r.fact(s + ".")
    if len(gorder) > len(shown_g):
        r.fact(f"В фактах {len(shown_g)} крупнейших групп из {len(gt)}; остальные в таблице интерфейса.")
    # key columns first: a slide export keeps only the first 8
    cols = [("name", LEVEL_RU[level], "text"), ("rub", "Продажи, млн руб.", "mln"), ("price", "Цена, руб./шт.", "num1")]
    if one_net:
        cols += [("cost_u", "Закупка, руб./шт.", "num1"), ("markup", "Наценка сети", "pct")]
    if gbase:
        cols += [("gp", "Рост цены", "gpct"), ("price0", f"Цена {gbase.label()}, руб./шт.", "num1"), ("idx", "Индекс на одинаковых SKU", "gpct")]
    cols += [("share", "Доля в группе", "pct"), ("price_l", "Цена, руб./л(кг)", "num0"), ("ml", "Объем шт., мл/г", "num0"),
             ("qty", "Продажи, тыс. шт.", "th")]
    if one_net:
        cols.append(("cost_l", "Закупка, руб./л(кг)", "num0"))
    r.table(f"Цены: {LEVEL_RU[by].lower()} × {LEVEL_RU[level].lower()}, {period.label()}", cols, rows)
    bad = S.q1(f"""SELECT count(*) FILTER (WHERE cost > rub AND rub > 0) * 1.0 / nullif(count(*) FILTER (WHERE rub > 0), 0)
                   FROM {S.source(scope, period)} n WHERE NOT (network = {S._lit(DEPO)} AND month >= DATE '2025-01-01')""")[0]
    if one_net and bad:
        r.caveat(f"Закупка выше цены продажи в {S.fpct(bad)} строк витрины: так в источнике, отрицательная наценка не реальная (п. 7.5).")
    r.caveat("Средняя цена = руб. / шт., выше уровня SKU зависит от ассортимента; индекс цен — цены тех же SKU в тех же сетях с весами базы.")
    if gbase:
        r.caveat(comparable_note(scope, gcur, gbase))
        bn = base_note(a, gcur, gbase)
        if bn:
            r.caveat(bn)
    if one_net:
        r.caveat("Закупку и наценку сравнивать внутри одной сети и одного периода: учет закупки у сетей и по годам разный."
                 + (" У Депо с 01.2025 закупки в данных нет." if DEPO in set(df.g) and period.end >= S.date(2025, 1, 1) else ""))
    nxt = lambda **kw: _next_args(scope, "rub", a, **kw)
    if not a.get("base_period"):
        r.suggest(f"Цены: {period.vs(period.shift(-12))}", "prices",
                  nxt(level=level, by=by, period=period.to_str(), base_period=period.shift(-12).to_str()))
    if level != "sku":
        r.suggest(f"Цены по SKU, разрез по {GEO_DAT[by]}", "prices", nxt(level="sku", by=by))
    if not one_net:
        r.suggest("Закупка и наценка по сетям", "prices", nxt(level=level, by="network"))
    if scope.has_object():
        r.suggest(f"Факторный анализ продаж {_obj_name(scope)}", "factors", nxt(period=None, base_period=None))
    r.suggest(f"Рейтинг {LEVEL_GEN[level]} по продажам", "ranking", nxt(level=level, by=None, period=None, base_period=None))
    return r


# ——— ABC ———

def t_abc(a):
    """ABC analysis: classes by cumulative sales share (A — up to 80%, B — up to 95%, C — the rest);
    for SKUs, classes are also shown separately for adult and kids subgroups (methodology 2.4)."""
    scope = make_scope(a)
    level = a.get("level") or "sku"
    if level not in LEVEL or level == "month":
        raise ToolError(f"level должен быть одним из: {', '.join(k for k in LEVEL if k != 'month')}")
    m = get_measure(a)
    unit_fmt, unit = UNIT[m]
    period, explicit = main_period(a, "mat")
    ca, cb = float(a.get("a_share") or 0.8), float(a.get("b_share") or 0.95)
    r = Result("abc", f"ABC-анализ {LEVEL_GEN[level]} по продажам {MEASURE_RU[m]}")
    r.subtitle = f"{scope.describe()}; {period.label()}; A — до {S.fpct(ca, 0)} продаж, B — до {S.fpct(cb, 0)}, C — остальное"
    r.set_slice(scope, period, None, None, m, note=f"ABC {LEVEL_GEN[level]}: A — до {S.fpct(ca, 0)} продаж, B — до {S.fpct(cb, 0)}")
    raw = agg(scope, period, [level], m)
    df = _stable(raw[~raw[level].map(_blank) & (raw.v > 0)], "v", level).reset_index(drop=True)
    if df.empty:
        raise _empty(scope, period)
    tot = float(df.v.sum())
    df["cum"] = df.v.cumsum() / tot
    prev = df.cum.shift(1).fillna(0)
    df["cls"] = ["A" if p < ca else ("B" if p < cb else "C") for p in prev]
    df["share"] = df.v / tot
    n = len(df)
    summ = []
    for c in "ABC":
        d = df[df.cls == c]
        summ.append({"cls": c, "n": len(d), "pn": len(d) / n, "v": float(d.v.sum()), "pv": float(d.v.sum()) / tot})
    shown = df.head(30)
    info = _labels(scope, period, level, shown[level].tolist())
    lbl = lambda x: f"{_label(level, x)}{' [СТМ]' if _is_stm(level, x, info) else ''}"
    blank_v = float(raw[raw[level].map(_blank)].v.sum())
    r.fact(f"Всего в выборке за {period.label()}: {S.fmoney(tot, m)} (позиции с положительными продажами); {LEVEL_GEN[level]} с продажами: {n}"
           + (" (SKU считаются по названию)" if level == "sku" else "") + ".", key=True)
    r.fact("Классы ABC: " + "; ".join(f"{x['cls']}: {x['n']} {_pl(x['n'], 'позиция', 'позиции', 'позиций')} ({S.fpct(x['pn'])} позиций), "
                                      f"{S.fmoney(x['v'], m)} ({S.fpct(x['pv'])} продаж)" for x in summ) + ".", key=True)
    x0 = df.iloc[0]
    r.fact(f"Первое место: {lbl(x0[level])}, {S.fmoney(x0.v, m)}, доля {S.fpct(x0.share, 2)}; на первые {min(10, n)} позиций приходится "
           f"{S.fpct(df.share.head(10).sum())} продаж.", key=True)
    if blank_v:
        r.fact(f"Без метки ({LEVEL_RU[level].lower()} не указан): {S.fmoney(blank_v, m)}, в классы не включено.")
    if level == "sku":
        _abc_kids(r, scope, period, m, df)
    rows = []
    for i, x in enumerate(shown.itertuples()):
        name = getattr(x, level)
        man = _man_text(level, name, info, scope)
        rows.append({"rank": i + 1, "name": lbl(name), "extra": man, "v": x.v, "share": x.share, "cum": x.cum, "cls": x.cls})
    r.fact(f"Показаны места 1–{min(10, n)} из {n}; полный топ-30 в таблице.")
    for x in rows[:10]:
        r.fact(f"{x['rank']}. {x['name']}{' (' + x['extra'] + ')' if x['extra'] else ''}: {S.fmoney(x['v'], m)}, доля {S.fpct(x['share'], 2)}, "
               f"накопленная {S.fpct(x['cum'])}, класс {x['cls']}.")
    if level in ("brand", "manufacturer", "sku", "subbrand"):
        stm = [x["name"] for x in rows[:10] if x["name"].endswith("[СТМ]")]
        r.fact("СТМ среди показанных: " + (", ".join(stm) if stm else "нет") + ".")
    r.kpis([(f"Класс {x['cls']}", _cnt(x["n"]), f"{S.fpct(x['pn'])} позиций · {S.fpct(x['pv'])} продаж", None) for x in summ])
    r.table("Классы ABC", [("cls", "Класс", "text"), ("n", "Позиций", "int"), ("pn", "Доля позиций", "pct"),
                          ("v", f"Продажи, {unit}", unit_fmt), ("pv", "Доля продаж", "pct")], summ)
    r.table("Топ-30 позиций", [("rank", "#", "int"), ("name", LEVEL_RU[level], "text"), ("extra", "Бренд / производитель", "text"),
                               ("v", f"Продажи, {unit}", unit_fmt), ("share", "Доля", "pct2"), ("cum", "Накопленная доля", "pct"),
                               ("cls", "Класс", "text")], rows)
    pts = sorted(set(np.linspace(0, n - 1, min(60, n)).astype(int).tolist()))
    r.chart("line", "Кривая Парето: накопленная доля продаж", [f"{S.fnum(100 * (i + 1) / n, 0)}%" for i in pts],
            [{"name": "Накопленная доля продаж", "values": [float(df.cum.iloc[i]) for i in pts]}], fmt="pct")
    if level == "sku":
        r.caveat("SKU объединены по нормализованному названию из справочника: одна позиция в разных сетях считается одной.")
    if level in ("manufacturer", "brand"):
        r.caveat("Производитель и бренд — метки поставщика данных; варианты написания не объединялись.")
    nxt = lambda **kw: _next_args(scope, m, a, **kw)
    if scope.has_network_filter() and level == "sku":
        r.suggest("Кандидаты на вывод из ассортимента сети", "assortment", nxt(mode="delist", period=None))
    r.suggest(f"Топ {LEVEL_GEN[level]} по росту продаж", "ranking", nxt(level=level, sort="growth", period=None))
    if level != "brand":
        r.suggest("ABC-анализ брендов", "abc", nxt(level="brand"))
    r.suggest(f"Цены {LEVEL_GEN[level]} по сетям", "prices", nxt(level=level if level not in GEO else None, by="network", period=None))
    return r


def _abc_kids(r, scope, period, m, df):
    """SKU classes split by adult and kids subgroups (an SKU belongs to its highest-selling subgroup)."""
    sg = agg(scope, period, ["sku", "subgroup"], m)
    sg = sg[sg.sku.isin(df.sku) & sg.subgroup.notna()]
    if sg.empty:
        return
    main = sg.loc[sg.groupby("sku").v.idxmax(), ["sku", "subgroup"]]
    dd = df.merge(main, on="sku", how="left")
    kid = dd.subgroup.str.contains("етск", na=False)
    if not kid.any() or kid.all():
        return
    rows = []
    for name, mask in (("взрослые подгруппы", ~kid), ("детские подгруппы", kid)):
        d = dd[mask]
        rows.append({"g": name, "A": int((d.cls == "A").sum()), "B": int((d.cls == "B").sum()), "C": int((d.cls == "C").sum()),
                     "v": float(d.v.sum())})
    rows.append({"g": "итого", "A": int((dd.cls == "A").sum()), "B": int((dd.cls == "B").sum()), "C": int((dd.cls == "C").sum()),
                 "v": float(dd.v.sum()), "_cls": "tot"})
    r.fact("Классы по взрослым и детским подгруппам (SKU по подгруппе с наибольшими продажами): " + "; ".join(
        f"{x['g']}: A {x['A']}, B {x['B']}, C {x['C']} SKU, продажи {S.fmoney(x['v'], m)}" for x in rows) + ".")
    unit_fmt, unit = UNIT[m]
    r.table("Классы по взрослым и детским подгруппам", [("g", "Подгруппы", "text"), ("A", "A", "int"), ("B", "B", "int"),
                                                        ("C", "C", "int"), ("v", f"Продажи, {unit}", unit_fmt)], rows)


# ——— assortment rotation ———

def t_assortment(a):
    """Assortment rotation in a network: delisting candidates (weak sales per store despite wide presence)
    and introduction candidates (strong sales per store in other networks or formats, absent in the target network)."""
    scope = make_scope(a)
    if not scope.has_network_filter():
        raise ToolError("Для ротации нужна сеть (network) или баннер (banner).")
    mode = a.get("mode") or "delist"
    if mode not in ("delist", "introduce"):
        raise ToolError("mode должен быть delist (на вывод) или introduce (на ввод).")
    n = max(1, int(a.get("n") or 5))
    m = get_measure(a) if a.get("measure") else "qty"
    u1 = {"rub": "руб.", "qty": "шт.", "kg": "кг (л)"}[m]
    period, explicit = main_period(a, "mat")
    s = str(a.get("min_coverage") or "0.7").strip().replace(",", ".")
    try:
        cov_min = float(s.rstrip("%")) / (100 if s.endswith("%") else 1)
    except ValueError:
        raise ToolError("min_coverage — доля точек от 0 до 1, например 0.5.")
    if 1 < cov_min <= 100:                # «70» means percent
        cov_min /= 100
    if not 0 < cov_min <= 1:
        raise ToolError("min_coverage — доля точек от 0 до 1, например 0.5.")
    nets = S.scope_networks(scope) or []
    last = period.end
    ok_m = S.last_ok_month(nets)
    if period.start <= ok_m < last:
        last = ok_m                      # presence is measured in the network's last complete month
    target = net_lbl(scope)
    r = Result("assortment", "Кандидаты на вывод из ассортимента" if mode == "delist" else "Кандидаты на ввод в ассортимент")
    r.subtitle = f"{scope.describe()}; {period.label()}; мера — {MEASURE_RU[m]}"
    r.set_slice(scope, period, None, None, m, note="на вывод" if mode == "delist" else "на ввод")
    # denominator: the network's stores selling the object's category in the last complete month
    cat = scope.without(*S.OBJECT_KEYS)
    subs = _own_subgroups(scope, period, m)
    if subs:
        cat = cat.with_(subgroup=subs)
    lastp = S.Period(last, last)
    den = int(S.q1(f"SELECT count(DISTINCT store_key) FROM {S.fact_source(cat, lastp)} WHERE qty > 0")[0] or 0)
    tgt = S.q(f"""SELECT k.sku_name sku, sum(f.{m}) v, count(DISTINCT (f.store_key, f.month)) FILTER (WHERE f.qty > 0) sm,
                         count(DISTINCT f.store_key) FILTER (WHERE f.qty > 0 AND f.month = DATE '{last}') st_last
                  FROM {S.fact_source(scope, period)} f JOIN sku k USING (sku_id) GROUP BY 1""")
    tgt = tgt[(tgt.v > 0) & (tgt.sm > 0)].copy()
    if tgt.empty or not den:
        raise _empty(scope, period)
    tgt["cov_share"] = tgt.st_last / den
    tgt["ros"] = tgt.v / tgt.sm
    cat_txt = f"подгруппы объекта: {', '.join(subs)}" if subs else S.cat_desc(cat)
    r.fact(f"Целевая сеть {target}: {len(tgt)} SKU с продажами за {period.label()}; точек с категорией ({cat_txt}) в {last:%m.%Y}: {_cnt(den)}.", key=True)
    nxt = lambda **kw: _next_args(scope, m, a, **kw)
    if mode == "delist":
        _delist(r, scope, tgt, n, cov_min, last, m, u1)
        r.suggest("Кандидаты на ввод в эту сеть", "assortment", nxt(mode="introduce", n=n))
    else:
        _introduce(r, scope, a, tgt, n, period, m, u1)
        r.suggest("Кандидаты на вывод из этой сети", "assortment", nxt(mode="delist", n=n))
    r.caveat("Точка — сеть + код точки (store_key). Продажи на точку = продажи / число пар точка × месяц с продажами.")
    r.caveat("SKU сопоставлены между сетями по нормализованному названию из справочника.")
    r.caveat("Решение о ротации требует данных, которых в панели нет: маржа поставщика, промо, условия контракта, остатки.")
    r.suggest("ABC-анализ SKU в этой сети", "abc", nxt(level="sku", mode=None, n=None))
    r.suggest("Представленность по сетям", "distribution", nxt(mode=None, n=None, period=None, by="network", level="sku"))
    return r


def _delist(r, scope, tgt, n, cov_min, last, m, u1):
    wide = tgt[tgt.cov_share >= cov_min].sort_values(["ros", "sku"])
    if wide.empty:
        mx = tgt.cov_share.max()
        raise ToolError(f"Нет SKU с представленностью не меньше {S.fpct(cov_min, 0)} точек сети с категорией в {last:%m.%Y} "
                        f"(максимум {S.fpct(mx)}). Уменьши min_coverage (например {max(0.1, math.floor(mx * 10) / 10):.1f}).")
    cand = wide.head(n).copy()
    cand["rank"] = range(1, len(cand) + 1)
    trend, l6, p6 = _sku_trend(scope, cand.sku.tolist(), last, m)
    info = _labels(scope, S.Period(S.add_months(last, -11), last), "sku", cand.sku.tolist())
    med = float(wide.ros.median())
    r.fact(f"Широко представлены (не меньше {S.fpct(cov_min, 0)} точек с категорией в {last:%m.%Y}): {len(wide)} SKU; "
           f"кандидатов на вывод {len(cand)}: с самыми низкими продажами на точку в месяц среди них "
           f"(медиана у широко представленных {S.fnum(med, 2)} {u1}).", key=True)
    if len(wide) <= n:
        r.fact(f"Широко представленных SKU не больше, чем запрошено кандидатов ({n}): в список попали все, включая сильные; "
               "для выбора можно уменьшить min_coverage.")
    rows = []
    for x in cand.itertuples():
        tr = trend.get(x.sku)
        man = _man_text("sku", x.sku, info, scope)
        name = f"{x.sku}{' [СТМ]' if _is_stm('sku', x.sku, info) else ''}"
        rows.append({"name": name, "brand": man, "cov": x.cov_share, "ros": x.ros, "v": x.v, "trend": tr, "pos": f"{x.rank} из {len(wide)}"})
        tr_txt = (f"тренд на точку {l6.vs(p6)}: {S.fdyn(tr)}" if tr is not None else "тренд не считается (нет продаж в одном из полугодий)") if l6 else "тренд не считается (нет 12 месяцев истории)"
        r.fact(f"Кандидат на вывод {x.rank} из {len(wide)} снизу: {name}{' (' + man + ')' if man else ''}: в {S.fpct(x.cov_share)} точек, "
               f"{S.fnum(x.ros, 2)} {u1} на точку в месяц, всего {S.fnum(x.v, 0)} {u1}; {tr_txt}.")
    r.table(f"Кандидаты на вывод: {len(cand)} SKU", [
        ("name", "SKU", "text"), ("brand", "Бренд / производитель", "text"), ("cov", f"Доля точек, {last:%m.%Y}", "pct"),
        ("ros", f"Продажи на точку в мес., {u1}", "num2"), ("v", f"Продажи за период, {u1}", "num0"),
        ("trend", f"Тренд на точку: {l6.vs(p6)}" if l6 else "Тренд на точку", "gpct"), ("pos", "Место снизу", "text")], rows)
    r.caveat("Отбор на вывод: SKU, которые есть в большой доле точек сети с категорией, но продаются хуже всех на одну точку.")


def _sku_trend(scope, names, last, m):
    """Sales per store, last 6 months vs the previous 6 (a full 12 months up to last, not clipped by the period)."""
    lo, _ = S.data_bounds()
    l6, p6 = S.Period(S.add_months(last, -5), last), S.Period(S.add_months(last, -11), S.add_months(last, -6))
    if p6.start < lo or not names:
        return {}, None, None
    df = S.q(f"""SELECT k.sku_name sku,
                        sum(f.{m}) FILTER (WHERE f.month >= DATE '{l6.start}') v1,
                        count(DISTINCT (f.store_key, f.month)) FILTER (WHERE f.qty > 0 AND f.month >= DATE '{l6.start}') s1,
                        sum(f.{m}) FILTER (WHERE f.month < DATE '{l6.start}') v0,
                        count(DISTINCT (f.store_key, f.month)) FILTER (WHERE f.qty > 0 AND f.month < DATE '{l6.start}') s0
                 FROM {S.fact_source(scope, S.Period(p6.start, last))} f JOIN sku k USING (sku_id)
                 WHERE k.sku_name IN ({', '.join(S._lit(x) for x in names)}) GROUP BY 1""")
    out = {}
    for x in df.itertuples():
        if x.s1 and x.s0 and x.v0 and x.v0 > 0:
            out[x.sku] = (x.v1 / x.s1) / (x.v0 / x.s0) - 1
    return out, l6, p6


def _introduce(r, scope, a, tgt, n, period, m, u1):
    ref_scope = scope.without("network", "banner")
    ref_nets = a.get("reference_networks")
    if ref_nets:
        ref_scope = ref_scope.with_(network=[make_scope({"network": x}).values("network")[0]
                                             for x in (ref_nets if isinstance(ref_nets, list) else [ref_nets])])
    excl = []
    if scope.values("network"):
        excl.append("f.network NOT IN (" + ", ".join(S._lit(x) for x in scope.values("network")) + ")")
    if scope.values("banner"):
        excl.append("f.banner NOT IN (" + ", ".join(S._lit(x) for x in scope.values("banner")) + ")")
    min_sm = int(a.get("min_store_months") or 50)
    present = set(tgt[tgt.cov_share >= 0.1].sku)
    excl.append("NOT coalesce(k.is_pl, FALSE)")      # another network's private label cannot be introduced
    base = f"FROM {S.fact_source(ref_scope, period)} f JOIN sku k USING (sku_id) WHERE {' AND '.join(excl)}"
    # an exact store-month count over the whole category is expensive (a very large fact scan): first a rough cut by
    # row count (rows >= store-months), then an exact count for the shortlist
    pre = S.q(f"SELECT k.sku_name sku, sum(f.{m}) v, count(*) FILTER (WHERE f.qty > 0) sm0 {base} GROUP BY 1")
    n_ref = int((pre.v > 0).sum())
    pre = pre[(pre.v > 0) & (pre.sm0 >= min_sm) & ~pre.sku.isin(present)]
    short = pre.assign(r0=pre.v / pre.sm0).sort_values("r0", ascending=False).sku.head(max(200, 20 * n)).tolist()
    ref = S.q(f"""SELECT k.sku_name sku, sum(f.{m}) v, count(DISTINCT (f.store_key, f.month)) FILTER (WHERE f.qty > 0) sm,
                         count(DISTINCT f.store_key) FILTER (WHERE f.qty > 0) stores, string_agg(DISTINCT f.network, '|') nets,
                         sum(f.rub) / nullif(sum(f.qty), 0) price
                  {base} AND k.sku_name IN ({', '.join(S._lit(x) for x in short) or "''"}) GROUP BY 1""")
    ref = ref[(ref.v > 0) & (ref.sm >= min_sm)].copy()
    ref["ros"] = ref.v / ref.sm
    cand = ref.sort_values(["ros", "sku"], ascending=[False, True]).head(n)
    r.fact(f"Эталон для подбора: {ref_scope.describe()}, кроме целевой сети; SKU с продажами: {n_ref}; кандидаты на ввод ({len(cand)}): "
           f"SKU с самыми высокими продажами на точку в месяц среди продававшихся не меньше чем в {min_sm} парах точка × месяц, "
           "которых в целевой сети нет или они есть меньше чем в 10% точек.", key=True)
    if cand.empty:
        raise ToolError("Нет SKU для ввода: все сильные SKU эталона уже есть в целевой сети. Можно уменьшить min_store_months.")
    info = _labels(ref_scope, period, "sku", cand.sku.tolist())
    rows = []
    for i, x in enumerate(cand.itertuples()):
        here = tgt[tgt.sku == x.sku]
        cv = None if here.empty else float(here.cov_share.iloc[0])
        here_txt = "нет" if cv is None else ("меньше чем в 0,1% точек" if cv < 0.001 else f"в {S.fpct(cv)} точек")
        man = _man_text("sku", x.sku, info, scope)
        nets = ", ".join(sorted(S.net_label(s) for s in str(x.nets).split("|")))
        rows.append({"name": x.sku, "brand": man, "ros": x.ros, "stores": x.stores, "nets": nets, "price": x.price, "here": here_txt})
        r.fact(f"Кандидат на ввод {i + 1}: {x.sku}{' (' + man + ')' if man else ''}: {S.fnum(x.ros, 2)} {u1} на точку в месяц в "
               f"{_cnt(x.stores)} точках ({nets}), цена {S.fnum(x.price, 1)} руб./шт.; в целевой сети: {here_txt}.")
    r.table(f"Кандидаты на ввод: {len(cand)} SKU", [
        ("name", "SKU", "text"), ("brand", "Бренд / производитель", "text"), ("ros", f"Продажи на точку в мес., {u1}", "num2"),
        ("stores", "Точек в эталонных сетях", "int"), ("nets", "Где продается", "txt"), ("price", "Цена, руб./шт.", "num1"),
        ("here", "В целевой сети", "text")], rows)
    r.caveat("Отбор на ввод: SKU с самыми высокими продажами на точку в других сетях (или форматах), которых нет в целевой сети "
             "(представленность меньше 10%); СТМ сетей не предлагаются. Эталоном может стать другой формат (гипермаркет против "
             "магазинов у дома): формат задается фильтром format.")


# ——— forecast ———

def t_forecast(a):
    """Simple 1–6 month forecast: two methods (seasonal and annual trend), their mean and a range from the backtest error.
    Series: the 9 comparable networks without incomplete months, or the selected networks up to their last complete month."""
    scope = make_scope(a)
    m = get_measure(a)
    unit_fmt, unit = UNIT[m]
    lo, hi_data = S.data_bounds()
    cmp_only = not scope.has_network_filter()
    nets = S.scope_networks(scope)
    hi = S.last_ok_month(nets) if nets else hi_data       # an incomplete last month of the selected network is left out of the series
    raw = a.get("target_month")
    try:
        tgt = S.parse_ym(str(raw)) if raw else S.add_months(hi_data, 1)
    except (ValueError, TypeError):
        raise ToolError("target_month — месяц в формате 2026-10.")
    if not S.add_months(hi_data, 1) <= tgt <= S.add_months(hi_data, 6):
        raise ToolError(f"Прогноз делается на 1–6 месяцев после {hi_data:%m.%Y}.")
    h = (tgt.year - hi.year) * 12 + tgt.month - hi.month
    hist = S.Period(lo, hi)
    where = net_lbl(scope)
    tname = f"{S.MONTHS_FULL[tgt.month - 1]} {tgt.year}"
    r = Result("forecast", f"Прогноз продаж на {tname}")
    r.subtitle = f"{scope.describe()}; мера — {MEASURE_RU[m]}; ряд: {where}"
    r.set_slice(scope, hist, None, None, m, cmp=cmp_only, note=f"прогноз на {tname}")
    df = agg(scope, hist, ["month"], m, comparable_only=cmp_only)
    ser = pd.Series(df.v.values, index=df.month).reindex(hist.months).fillna(0.0)
    if ser.tail(12).sum() <= 0:
        raise _empty(scope, S.Period(S.add_months(hi, -11), hi))
    last_sale = max(d for d, v in ser.items() if v > 0)
    if last_sale < hi:
        raise ToolError(f"Продажи выборки ({scope.describe()}) заканчиваются в {last_sale:%m.%Y}: прогноз не строится.")
    idx = dict(ser.items())

    def fc(origin, target):
        """Both methods from the last known month origin: seasonal and annual trend."""
        l3 = [idx.get(S.add_months(origin, -i)) for i in range(3)]
        l3p = [idx.get(S.add_months(origin, -12 - i)) for i in range(3)]
        ly = idx.get(S.add_months(target, -12))
        l12 = [idx.get(S.add_months(origin, -i)) for i in range(12)]
        p12 = [idx.get(S.add_months(origin, -12 - i)) for i in range(12)]
        fa = ly * sum(l3) / sum(l3p) if None not in l3 + l3p and ly is not None and sum(l3p) > 0 else None
        fb = ly * sum(l12) / sum(p12) if None not in l12 + p12 and ly is not None and sum(p12) > 0 else None
        return fa, fb

    fa, fb = fc(hi, tgt)
    if fa is None and fb is None:
        raise ToolError("Мало истории для прогноза: нужен тот же месяц год назад и 3–12 месяцев перед ним.")
    errs = []
    for back in range(6):
        t_ = S.add_months(hi, -back)
        a_, b_ = fc(S.add_months(t_, -h), t_)
        act = idx.get(t_)
        if act and act > 0 and a_ is not None and b_ is not None:
            errs.append(abs((a_ + b_) / 2 / act - 1))
    vals = [v for v in (fa, fb) if v is not None]
    f_mean = sum(vals) / len(vals)
    mape = sum(errs) / len(errs) if errs else None
    spread = abs(fa - fb) / f_mean if fa is not None and fb is not None and f_mean > 0 else None
    fm = lambda v: S.fmoney(v, m) if v is not None else "не считается"
    if f_mean <= 0 or mape is None or mape > 0.3 or (spread is not None and spread > 0.3):
        raise ToolError(f"Прогноз ненадежен: сезонный метод {fm(fa)}, годовой тренд {fm(fb)}"
                        + (f" (расходятся на {S.fpct(spread)})" if spread is not None else "")
                        + (f", средняя ошибка на истории {S.fpct(mape)}" if mape is not None else ", ошибку на истории проверить нельзя")
                        + ". Похоже на новую или нестабильную позицию: прогноз не строится.")
    lo_v, hi_v = max(0.0, f_mean * (1 - mape)), f_mean * (1 + mape)
    ly = idx.get(S.add_months(tgt, -12))
    r.kpis([(f"Прогноз, {tname}", S.fmoney(f_mean, m), f"диапазон {S.fmoney(lo_v, m)} – {S.fmoney(hi_v, m)}", None),
            ("Тот же месяц год назад", fm(ly), where, None),
            ("Ошибка на истории", S.fpct(mape), f"средняя по {len(errs)} проверкам", None)])
    r.fact(f"Прогноз на {tname} ({where}): {S.fmoney(f_mean, m)} (среднее двух методов); диапазон {S.fmoney(lo_v, m)} – "
           f"{S.fmoney(hi_v, m)} по средней ошибке на истории {S.fpct(mape)} ({len(errs)} проверок).", key=True)
    r.fact(f"Методы: сезонный (тот же месяц год назад × последние 3 месяца к тем же 3 месяцам год назад) {fm(fa)}; годовой тренд "
           f"(тот же месяц год назад × последние 12 месяцев к предыдущим 12) {fm(fb)}"
           + (f"; расходятся на {S.fpct(spread)}" if spread is not None else "") + ".", key=True)
    if ly:
        r.fact(f"Тот же месяц год назад ({S.MONTHS_FULL[tgt.month - 1]} {tgt.year - 1}): {S.fmoney(ly, m)}; прогноз к нему: {_gw(f_mean, ly)}.", key=True)
    if hi < hi_data:
        r.fact(f"Последний полный месяц у выбранных сетей: {hi:%m.%Y}; прогноз строится от него на {h} мес. вперед.")
    hrows = [{"month": _label("month", d), "v": v, "ly": idx.get(S.add_months(d, -12))} for d, v in list(ser.items())[-15:]]
    r.fact("Последние месяцы: " + "; ".join(f"{x['month']} {S.fmoney(x['v'], m)} (год назад {fm(x['ly'])})" for x in hrows[-6:]) + ".")
    r.table("Методы", [("meth", "Метод", "txt"), ("v", f"Прогноз, {unit}", unit_fmt)], [
        {"meth": "Сезонный: тот же месяц год назад × (последние 3 мес. / те же 3 мес. год назад)", "v": fa},
        {"meth": "Годовой тренд: тот же месяц год назад × (последние 12 мес. / предыдущие 12 мес.)", "v": fb},
        {"meth": "Итог: среднее двух методов", "v": f_mean, "_cls": "tot"}])
    r.table("История", [("month", "Месяц", "text"), ("v", f"Продажи, {unit}", unit_fmt), ("ly", f"Год назад, {unit}", unit_fmt)], hrows)
    div = DIV[m]
    xs = [x["month"] for x in hrows] + [_label("month", tgt)]
    r.chart("line", f"Факт и прогноз, {unit}", xs, [
        {"name": "Факт", "values": [x["v"] / div for x in hrows] + [None]},
        {"name": "Прогноз", "values": [None] * (len(hrows) - 1) + [hrows[-1]["v"] / div, f_mean / div], "kind": "dash"}], fmt="num1", unit=unit)
    r.caveat("Прогноз статистический: не учитывает промо, изменения ассортимента и представленности, цены. "
             "Диапазон — по средней ошибке метода на последних месяцах; при ошибке или расхождении методов больше 30% прогноз не строится.")
    if cmp_only:
        r.caveat("Ряд — 9 сопоставимых сетей; неполные месяцы сетей (сейчас 09.2026 у ЛАД и Гиперкосмоса) и те же месяцы год назад "
                 "убраны из ряда с обеих сторон, поэтому сезонные отношения сравнивают одинаковый состав сетей.")
    nxt = lambda **kw: _next_args(scope, m, a, **kw)
    r.suggest("Динамика по месяцам и сезонность", "trend", nxt(seasonality=True, target_month=None))
    r.suggest("Факторный анализ продаж с начала года", "factors", nxt(target_month=None, period=None, base_period=None))
    return r


# ——— competitors ———

def t_competitors(a):
    """Competitors of a manufacturer or brand per methodology 8: segments are subgroup × «Товар» (product) with at least 0.5%
    of the target's sales; competitors are manufacturers with 0.5x–2x the target's sales in those segments, excluding
    private labels and «не указан» (unspecified). Default period is year-to-date, as on the reference portal (benchmark G2)."""
    scope = make_scope(a)
    tkey = "brand" if scope.values("brand") else ("manufacturer" if scope.values("manufacturer") else None)
    if not tkey:
        raise ToolError("Для конкурентов нужен производитель (manufacturer) или бренд (brand).")
    m = get_measure(a)
    unit_fmt, unit = UNIT[m]
    period, explicit = main_period(a, "ytd")
    gcur, gbase = growth_pair(a, period, True)
    if gcur.start > period.start:
        period = gcur
    gap = coverage_gap(scope, gcur, gbase)
    if gap:
        gbase = None
    cmp_only = not scope.has_network_filter()
    where = net_lbl(scope)
    seg_min = float(a.get("segment_min") or 0.005)
    lo_k, hi_k = float(a.get("low") or 0.5), float(a.get("high") or 2.0)
    seg_lv = a.get("segment_level") or "product"            # as on the reference portal: subgroup × «Товар»
    if seg_lv not in ("product", "product_type"):
        raise ToolError("segment_level — product (как на портале) или product_type.")
    seg_ru = "товар" if seg_lv == "product" else "тип товара"
    target = ", ".join(scope.values(tkey))
    kind = "бренд" if tkey == "brand" else "производитель"
    r = Result("competitors", f"Ближайшие конкуренты: {target}")
    r.subtitle = f"{scope.describe()}; {period.label()}"
    k_txt = f"{str(lo_k).replace('.', ',')}–{str(hi_k).replace('.', ',')}×"
    r.set_slice(scope, period, gcur if gbase else None, gbase, m, cmp=cmp_only, note=f"конкуренты: {k_txt} от цели в ее сегментах")
    seg = agg(scope, period, ["subgroup", seg_lv], m).rename(columns={seg_lv: "seg2"})
    tot = float(seg.v.sum())
    if tot <= 0:
        raise _empty(scope, period)
    mans = agg(scope, period, ["manufacturer"], m)
    # target manufacturers: for a brand, manufacturer labels with >= 1% of its sales (drops stray dictionary labels with tiny sales)
    tmans = list(mans[mans.v >= 0.01 * mans.v.sum()].sort_values("v", ascending=False).manufacturer)
    seg = seg[seg.subgroup.notna() & seg.seg2.notna() & (seg.v >= seg_min * tot)].sort_values("v", ascending=False)
    T = float(seg.v.sum())
    seg_where = "(" + " OR ".join(f"(k.subgroup_name = {S._lit(x.subgroup)} AND {LEVEL[seg_lv]} = {S._lit(x.seg2)})"
                                  for x in seg.itertuples()) + ")"
    mk = scope.without(*S.OBJECT_KEYS)                       # market: without the object filters
    cur = agg(mk, period, ["manufacturer"], m, where=seg_where)
    pool = cur[~cur.manufacturer.isin(tmans + scope.values("manufacturer")) & ~cur.manufacturer.str.startswith("STM(", na=False)
               & ~cur.manufacturer.map(_blank)]
    comp = _stable(pool[(pool.v >= lo_k * T) & (pool.v <= hi_k * T)], "v", "manufacturer")
    period_txt = period.label() + ("" if explicit else ", с начала года, как на портале")
    r.fact(f"Цель: {kind} {target}" + (f" (производитель: {', '.join(tmans[:3])})" if tkey == "brand" else "")
           + f". Сегменты за {period_txt}: {len(seg)} {_pl(len(seg), 'сочетание', 'сочетания', 'сочетаний')} подгруппа × {seg_ru} с долей не меньше {S.fpct(seg_min, 1)} продаж цели, "
           f"на них {S.fpct(T / tot)} ее продаж ({S.fmoney(T, m)}). Конкуренты: производители с продажами в этих сегментах от "
           f"{S.fmoney(lo_k * T, m)} до {S.fmoney(hi_k * T, m)} ({k_txt} от цели), без СТМ и «не указан».", key=True)
    if len(comp):
        r.fact(f"Конкуренты ({len(comp)}), продажи в сегментах {target} за {period.label()}: " + "; ".join(
            f"{x.manufacturer} {S.fmoney(x.v, m)} ({S.fnum(x.v / T, 2)}×)" for x in comp.itertuples()) + ".", key=True)
    else:
        near = _stable(pool, "v", "manufacturer").head(3)
        r.fact(f"Конкурентов в диапазоне нет; крупнейшие производители в сегментах {target}: "
               + "; ".join(f"{x.manufacturer} {S.fmoney(x.v, m)} ({S.fnum(x.v / T, 2)}×)" for x in near.itertuples()) + ".", key=True)

    # change: both sides without incomplete months, 9 comparable networks or the selected networks
    g_seg = g_all = {}
    if gbase:
        vs = gcur.vs(gbase)
        a1, a0 = (float(x.v.sum()) for x in growth_agg(scope, gcur, gbase, [], m, cmp_only))
        t1, t0 = (float(x.v.sum()) for x in growth_agg(scope, gcur, gbase, [], m, cmp_only, where=seg_where))
        mk1, mk0 = (float(x.v.sum()) for x in growth_agg(mk, gcur, gbase, [], m, cmp_only, where=seg_where))
        r.fact(f"{target} {vs} ({where}): всего {_dyn(a1, a0, m)} ({S.fmoney(a0, m)} → {S.fmoney(a1, m)}); в своих сегментах "
               f"{_dyn(t1, t0, m)}; сегменты в целом (все производители) {_dyn(mk1, mk0, m)}.", key=True)
        c1, c0 = growth_agg(mk, gcur, gbase, ["manufacturer"], m, cmp_only, where=seg_where)
        g_seg = {x.manufacturer: (x.v1, x.v0) for x in _pair(c1, c0, ["manufacturer"]).itertuples()}
        c1, c0 = growth_agg(mk, gcur, gbase, ["manufacturer"], m, cmp_only)
        g_all = {x.manufacturer: (x.v1, x.v0) for x in _pair(c1, c0, ["manufacturer"]).itertuples()}
        _dropped_segments(r, scope, gcur, gbase, m, cmp_only, seg, seg_lv, seg_min, target)
    else:
        r.fact(f"Рост {target} и конкурентов не считается: {no_base_reason(gcur, gap)}.", key=True)
    brands = S.q(f"""SELECT k.manufacturer, k.brand, sum(n.rub) rub FROM {S.source(mk, period)} n JOIN sku k USING (sku_id)
                     WHERE {seg_where} GROUP BY 1, 2""")
    rows = [{"name": f"{target} (цель)", "v": T, "ratio": 1.0, "_cls": "lvl"}]
    for i, x in enumerate(comp.itertuples()):
        br = brands[brands.manufacturer == x.manufacturer].sort_values("rub", ascending=False).brand.head(3).tolist()
        row = {"name": x.manufacturer, "v": x.v, "ratio": x.v / T, "brands": ", ".join(br)}
        s = f"производитель {x.manufacturer}: в сегментах {target} {S.fmoney(x.v, m)} ({S.fnum(x.v / T, 2)}× от {target})"
        if gbase:
            v1, v0 = g_seg.get(x.manufacturer, (0.0, 0.0))
            w1, w0 = g_all.get(x.manufacturer, (0.0, 0.0))
            row.update(g=_g(v1, v0), ga=_g(w1, w0))
            s += f"; {gcur.vs(gbase)}: в сегментах {target} {_dyn(v1, v0, m)}, всего {_dyn(w1, w0, m)}"
        if i < 6:
            r.fact(s + (f"; бренды: {', '.join(br)}" if br else "") + ".")
        rows.append(row)
    seg_rows = [{"sg": x.subgroup, "tp": x.seg2, "v": x.v, "share": x.v / tot} for x in seg.itertuples()]
    for x in seg_rows[:5]:
        r.fact(f"Сегмент {x['sg']} / {x['tp']}: {target} {S.fmoney(x['v'], m)} ({S.fpct(x['share'])} ее продаж).")
    if len(seg_rows) > 5:
        r.fact(f"Остальные сегменты ({len(seg_rows) - 5}, на них {S.fpct(sum(x['v'] for x in seg_rows[5:]) / tot)} продаж цели) в таблице.")
    _segment_leaders(r, mk, period, m, seg, seg_lv)
    r.table(f"Сегменты {target} (не меньше {S.fpct(seg_min, 1)} продаж цели), {period.label()}", [
        ("sg", "Подгруппа", "text"), ("tp", "Товар" if seg_lv == "product" else "Тип", "text"),
        ("v", f"Продажи {target}, {unit}", unit_fmt), ("share", f"Доля в продажах {target}", "pct")], seg_rows)
    cols = [("name", "Производитель", "text"), ("v", f"Продажи в сегментах, {unit}", unit_fmt), ("ratio", f"К {target}", "num2")]
    if gbase:
        cols += [("g", f"Рост в сегментах, {gcur.vs(gbase)}", "gpct"), ("ga", "Рост всего", "gpct")]
    r.table(f"Конкуренты: {str(lo_k).replace('.', ',')}–{str(hi_k).replace('.', ',')}× от {target} в его сегментах",
            cols + [("brands", "Основные бренды", "txt")], rows)
    r.chart("bar", f"Продажи в сегментах {target}, {unit}", [x["name"] for x in rows],
            [{"name": "Продажи", "values": [x["v"] / DIV[m] for x in rows]}], fmt="num1", unit=unit)
    r.caveat("Сегменты (подгруппа × «Товар», от 0,5% продаж цели) и диапазон конкурентов (0,5–2× от цели) — допущения методики п. 8, "
             "их можно поменять (segment_min, low, high).")
    if not explicit:
        r.caveat(f"Период по умолчанию — {period.label()}: с начала года, как считает конкурентов портал.")
    if gbase:
        r.caveat(comparable_note(scope, gcur, gbase))
    nxt = lambda **kw: _next_args(scope, m, a, **kw)
    if not explicit:
        r.suggest("Конкуренты за последние 12 месяцев", "competitors", nxt(period="mat"))
    if len(comp):
        first = comp.iloc[0].manufacturer
        r.suggest(f"Обзор {first}", "profile", _next_args(mk, m, a, manufacturer=first))
        r.suggest(f"Цены {target} и конкурентов по сетям", "prices",
                  _next_args(mk, m, a, manufacturer=tmans[:1] + comp.manufacturer.head(4).tolist(), level="manufacturer", by="network",
                             period=None, base_period=None))
    r.suggest(f"Факторный анализ продаж {target}", "factors", nxt())
    r.suggest(f"Обзор {target}: доля, бренды, сети", "profile", nxt())
    return r


def _dropped_segments(r, scope, gcur, gbase, m, cmp_only, seg, seg_lv, seg_min, target):
    """Segments where the target had at least seg_min of base sales but which have dropped off the list: the decline there is invisible in its "own segments"."""
    s1, s0 = growth_agg(scope, gcur, gbase, ["subgroup", seg_lv], m, cmp_only)
    d = _pair(s1, s0, ["subgroup", seg_lv])
    T0 = float(d.v0.sum())
    if T0 <= 0:
        return
    now = set(zip(seg.subgroup, seg.seg2))
    out = pd.Series([(x, y) not in now for x, y in zip(d.subgroup, d[seg_lv])], index=d.index)
    gone = d[(d.v0 >= seg_min * T0) & out & d.subgroup.notna()].sort_values("v0", ascending=False)
    if gone.empty:
        return
    r.fact(f"Сегменты, выпавшие из списка {target} (в {gbase.label()} не меньше {S.fpct(seg_min, 1)} ее продаж, сейчас меньше): "
           + "; ".join(f"{x.subgroup} / {getattr(x, seg_lv)} {S.fmoney(x.v0, m)} → {S.fmoney(x.v1, m)} ({_fdelta(x.v1 - x.v0, m)})"
                       for x in gone.head(5).itertuples())
           + f"; вместе {_fdelta(float((gone.v1 - gone.v0).sum()), m)}.", key=True)


def _segment_leaders(r, mk, period, m, seg, seg_lv):
    """Leaders of the largest segments by segment share; «не указан» (unspecified) is never a leader (rule 6.5) but a separate tail; private labels flagged."""
    rows = []
    for x in seg.head(4).itertuples():
        d = agg(mk, period, ["manufacturer"], m,
                where=f"k.subgroup_name = {S._lit(x.subgroup)} AND {LEVEL[seg_lv]} = {S._lit(x.seg2)}")
        st = float(d.v.sum())
        if st <= 0:
            continue
        unk = float(d[d.manufacturer.map(_blank)].v.sum())
        d = _stable(d[~d.manufacturer.map(_blank)], "v", "manufacturer").head(5)
        txt = "; ".join(f"{y.manufacturer}{' [СТМ]' if str(y.manufacturer).startswith('STM(') else ''} {S.fpct(y.v / st)}"
                        for y in d.itertuples())
        if unk >= 0.0005 * st:
            txt += f"; производитель не указан (разные производители) {S.fpct(unk / st)}"
        rows.append({"seg": f"{x.subgroup} / {x.seg2}", "lead": txt})
        r.fact(f"Лидеры сегмента {x.subgroup} / {x.seg2} (доля в сегменте, {period.label()}): {txt}.")
    r.table("Лидеры крупнейших сегментов (доля в сегменте)", [("seg", "Сегмент", "text"), ("lead", "Производители", "txt")], rows)
