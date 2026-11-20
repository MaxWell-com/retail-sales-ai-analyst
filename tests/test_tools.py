"""First-group tools (app/tools.py) checked against independent SQL on the database. No LLM.

Run: python -m pytest -q tests/test_tools.py
"""
import os
import re

os.environ.setdefault("VT_THREADS", "6")
os.environ.setdefault("VT_MEMORY", "8GB")

import pytest

from app import semantic as S
from app import tools as T
from app.result import ToolError

DASH = re.compile(r": —|, —|\(—\)| —\.|: —$")
YT = S.Period(S.date(2026, 1, 1), S.date(2026, 9, 1))
YB = YT.shift(-12)


def sql1(q):
    return S.q1(q)[0]


def num(s):
    return float(s.replace(" ", "").replace(" ", "").replace(",", "."))


def has(facts, rx):
    return any(re.search(rx, f) for f in facts)


def no_dash(r):
    bad = [f for f in r.facts if DASH.search(f)]
    assert not bad, bad[:3]


@pytest.mark.parametrize("level", ["brand", "sku", "manufacturer", "network"])
def test_market_growth_same_on_every_level(level):
    """Market growth does not depend on the ranking level: totals over the whole slice, incl. dropped-out objects and blank labels in the base."""
    v1 = sql1(f"SELECT sum(qty) FROM net_sku WHERE network = 'Гранит' AND {YT.sql()}")
    v0 = sql1(f"SELECT sum(qty) FROM net_sku WHERE network = 'Гранит' AND {YB.sql()}")
    g = round((v1 / v0 - 1) * 100, 1)                      # growth in %, one decimal
    r = T.t_ranking({"network": "Гранит", "level": level, "period": "2026-01..2026-09", "measure": "qty", "top": 3,
                     "all_categories": True})
    word = "снижение" if g < 0 else "рост"
    assert has(r.facts, rf"Рынок .*{word} на {S.fnum(abs(g), 1)}%"), r.facts[:5]
    assert has(r.facts, re.escape(S.fmoney(v1, "qty")))
    no_dash(r)


def test_estra_mayak_cosmetic_august():
    """K7: an explicit month gets growth vs. the same month a year ago, not «—»."""
    where = ("banner = 'Маяк Косметик' AND sku_id IN (SELECT sku_id FROM sku WHERE brand = 'ESTRA' "
             "AND product_type = 'Шампунь')")
    v1 = sql1(f"SELECT sum(qty) FROM net_sku WHERE {where} AND month = DATE '2026-08-01'")
    v0 = sql1(f"SELECT sum(qty) FROM net_sku WHERE {where} AND month = DATE '2025-08-01'")
    g = (v1 / v0 - 1) * 100
    r = T.t_trend({"banner": "Маяк Косметик", "brand": "ESTRA", "product_type": "Шампунь", "period": "2026-08", "measure": "qty"})
    assert has(r.facts, re.escape(S.fmoney(v1, "qty")))
    word = "снижение" if g < 0 else "рост"
    assert has(r.facts, rf"{word} на {S.fnum(abs(g), 1)}%"), r.facts
    assert "9 сопоставимых" not in r.slice_line   # the slice is one banner, not 9 chains
    no_dash(r)


def test_sku_count_is_stated():
    """«Какие SKU…»: the number of objects in the selection is stated as a fact, so the model doesn't present the top 10 as the count."""
    n = sql1(f"""SELECT count(DISTINCT k.sku_name) FROM net_sku n JOIN sku k USING (sku_id)
                 WHERE n.banner = 'Маяк Косметик' AND k.brand = 'ESTRA' AND k.product_type = 'Шампунь'
                   AND n.qty > 0 AND {S.mat().sql('n.month')}""")
    r = T.t_ranking({"banner": "Маяк Косметик", "brand": "ESTRA", "product_type": "Шампунь", "level": "sku", "measure": "qty", "top": 10})
    assert has(r.facts, rf"SKU с продажами: {n}\b"), r.facts[:3]
    assert has(r.facts, r"доля внутри ESTRA")


def test_trend_by_level_does_not_crash():
    r = T.t_trend({"network": "Гранит", "level": "brand", "top": 3, "measure": "qty"})
    assert r.facts and not any("Ошибка" in f for f in r.facts)
    no_dash(r)


def test_top_by_growth_with_threshold():
    """K3: «топ по росту» (top by growth) sorts by growth rate among objects with share >= 0.1% in both periods."""
    df = S.q(f"""SELECT k.brand, sum(n.qty) FILTER (WHERE {YT.sql('n.month')}) v1, sum(n.qty) FILTER (WHERE {YB.sql('n.month')}) v0
                 FROM net_sku n JOIN sku k USING (sku_id) WHERE n.network = 'Гранит' GROUP BY 1""")
    t1, t0 = df.v1.sum(), df.v0.sum()
    df = df[(df.brand.notna()) & (~df.brand.isin(list(T.BLANK))) & (df.v1 >= 0.001 * t1) & (df.v0 >= 0.001 * t0)]
    leader = (df.v1 / df.v0).idxmax()
    r = T.t_ranking({"network": "Гранит", "level": "brand", "measure": "qty", "sort": "growth", "top": 10, "all_categories": True})
    assert has(r.facts, rf"^1\. {re.escape(df.brand[leader])}\b"), [f for f in r.facts if f.startswith("1.")]
    assert has(r.facts, r"[Пп]орог")


def test_focus_shows_neighbours_and_leaders():
    """«На уровне NORD»: NORD's rank, a window of neighbours and the leaders separately (not «на топ-10 приходится 0,1%»)."""
    df = S.q("""SELECT k.manufacturer m, sum(n.qty) v FROM net_sku n JOIN sku k USING (sku_id)
                WHERE n.network = 'Ажур' AND k.product_type = 'Бальзам для волос' AND n.month = DATE '2026-09-01'
                GROUP BY 1 HAVING sum(n.qty) <> 0""")
    df = df[df.m.notna() & ~df.m.isin(list(T.BLANK))].sort_values("v", ascending=False).reset_index(drop=True)
    nord = float(df.v[df.m == "NORD"].iloc[0])
    places = range(int((df.v > nord).sum()) + 1, int((df.v >= nord).sum()) + 1)   # with tied sales any of these ranks is valid
    r = T.t_ranking({"network": "Ажур", "product_type": "Бальзам для волос", "level": "manufacturer", "period": "2026-09",
                     "measure": "qty", "focus": "NORD"})
    m = [re.search(r"NORD: (\d+)-е место из (\d+)", f) for f in r.facts]
    m = next((x for x in m if x), None)
    assert m and int(m.group(1)) in places and int(m.group(2)) == len(df), r.facts[:6]
    assert has(r.facts, r"^Лидеры: 1\. " + re.escape(df.m[0]))
    assert not has(r.facts, r"на топ-10 приходится")


def test_profile_share_base_uses_same_categories():
    """L3-7: share «in its own categories» in the base period — numerator over the same subgroups as the market.
    Subgroups are those with at least 1% of the brand's sales over 12 months; share is Jan–Sep 2026 vs. Jan–Sep 2025 over
    the 9 comparable chains, incomplete chain-months removed on both sides. The brand discontinued its decorative line
    in 09.2025: it is present in the base but not in the (12-month) subgroups, so a numerator over all subgroups
    overstates the share."""
    brand = "k.brand = 'L''AURELLE PARIS'"
    sg = S.q(f"""SELECT k.subgroup_base sb, sum(n.rub) v FROM net_sku n JOIN sku k USING (sku_id)
                 WHERE {brand} AND {S.mat().sql('n.month')} GROUP BY 1""")
    subs = sg[sg.sb.notna() & (sg.v > 0.01 * sg.v.sum())].sb.tolist()
    in_subs = "k.subgroup_base IN (" + ", ".join(S._lit(x) for x in subs) + ")"
    cmp = "n.network IN (" + ", ".join(S._lit(x) for x in S.COMPARABLE) + ")"

    def side(per, other):
        ex = [f"NOT (n.network = {S._lit(n)} AND n.month IN (DATE '{c}', DATE '{b}'))"
              for c, b in zip(per.months, other.months) for n, x in S.bad_pairs() if x in (c, b)]
        w = " AND ".join([per.sql("n.month"), cmp] + ex)
        return S.q1(f"""SELECT sum(n.rub) FILTER (WHERE {brand} AND {in_subs}), sum(n.rub) FILTER (WHERE {in_subs}),
                               sum(n.rub) FILTER (WHERE {brand})
                        FROM net_sku n JOIN sku k USING (sku_id) WHERE {w}""")
    o1, mk1, _ = side(YT, YB)
    o0, mk0, all0 = side(YB, YT)
    r = T.t_profile({"brand": "L'AURELLE PARIS", "all_categories": True})
    m = [re.search(r"доля L'AURELLE PARIS в этих категориях (\d+,\d+)% → (\d+,\d+)%", f) for f in r.facts]
    m = next((x for x in m if x), None)
    assert m, r.facts[:6]
    assert m.group(1) == S.fnum(o0 / mk0 * 100, 2) and m.group(2) == S.fnum(o1 / mk1 * 100, 2), m.group(0)
    assert S.fnum(all0 / mk0 * 100, 2) != m.group(1)     # former bug: numerator over all of the brand's subgroups
    assert has(r.facts, r"вклад подгруппы Декоративная косметика")


def test_period_alignment_and_order():
    """«2026 к 2025» uses the same months; a base later than the period is swapped."""
    r = T.t_ranking({"manufacturer": "NORD", "level": "brand", "period": "2026", "base_period": "2025", "top": 3})
    assert has(r.facts, r"янв–сен 2026 к янв–сен 2025")
    r2 = T.t_ranking({"manufacturer": "NORD", "level": "brand", "period": "2025-09", "base_period": "2026-09", "top": 3})
    assert has(r2.facts, r"сентябрь 2026 к сентябрю 2025")


@pytest.mark.parametrize("key,value,want", [
    ("product_type", "маска для лица", "Маски для лица"),
    ("product_type", "бальзам для губ", "Уход за губами"),
    ("brand", "SWANLY", "СВОНЛИ [SWANLY]"),
    ("brand", "СИНЯЯ ЦЕНА", "СИНЯЯ ЦЕНА (PL)"),
    ("brand", "HILL & SHORE", "HILL&SHORE"),
    ("region", "Московская область", "Московская обл"),
    ("city", "Москва", "Москва г"),
    ("network", "Семерочка", "Семерочка Х7"),
])
def test_resolve(key, value, want):
    assert T.resolve(key, value) == want


@pytest.mark.parametrize("key,value", [("product_type", "Кондиционер для белья")])
def test_resolve_no_false_substitution(key, value):
    """L3-6: better an error with a hint than a calculation over the wrong category."""
    with pytest.raises(ToolError):
        T.resolve(key, value)


def test_no_dashes_in_common_calls():
    for a in ({"level": "brand", "top": 10}, {"manufacturer": "NORD", "level": "network"},
              {"level": "brand", "by": "network", "product_type": "Шампунь", "top": 2}):
        no_dash(T.t_ranking(a))
    no_dash(T.t_profile({"manufacturer": "NORD", "portal": True}))
    no_dash(T.t_trend({"manufacturer": "NORD"}))


def test_default_scope_is_own_categories():
    """Product decision: with no category named, the slice is the subgroups where NORD has products."""
    cats = S.own_categories()
    assert "Гигиена" not in cats and "Средства для окраски волос" in cats
    lit = ", ".join(S._lit(c) for c in cats)
    tot = sql1(f"""SELECT sum(n.rub) FROM net_sku n JOIN sku k USING (sku_id)
                   WHERE k.subgroup_base IN ({lit}) AND {S.mat().sql('n.month')}""")
    r = T.t_ranking({"level": "brand", "top": 5})
    assert "категории NORD" in r.slice_line
    assert has(r.facts, re.escape(S.fmoney(tot)))
    r2 = T.t_ranking({"level": "brand", "top": 5, "all_categories": True})
    assert "все категории панели" in r2.slice_line
    with pytest.raises(ToolError, match="категории NORD"):
        T.t_ranking({"brand": "MELVA", "level": "sku"})
    # a named category outside NORD's is computed on request
    assert T.t_ranking({"product_type": "Зубная паста", "level": "brand", "top": 3}).facts
