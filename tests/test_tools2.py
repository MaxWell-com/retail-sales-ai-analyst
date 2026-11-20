"""Part-2 tools (app/tools2.py) checked with independent SQL queries against DuckDB.

Expected numbers are computed here with our own SQL (not via tools/tools2), including reference questions G2, G3 and M1.
Run: python -m pytest -q tests/test_tools2.py (from the repo root).
"""
import os
import re
from datetime import date
from pathlib import Path

os.environ.setdefault("VT_THREADS", "4")
os.environ.setdefault("VT_MEMORY", "6GB")

import pytest  # noqa: E402

from app import semantic as S  # noqa: E402
from app import tools2 as T2  # noqa: E402
from app.result import ToolError  # noqa: E402

CMP = ("Гранит", "Семерочка Х7", "Нива (сеть)", "Перелесок ТД", "Ажур",
       "Депо Кэш Энд Керри", "ЛАД (сеть)", "Гиперкосмос", "Детский Остров (сеть)")
CMP_SQL = "n.network IN (" + ", ".join(f"'{x}'" for x in CMP) + ")"
# store = chain + store code (docs/methodology.md §5.4), recomputed here independently, without store_x
SK = """(SELECT store_id, network, banner,
                min(store_id) OVER (PARTITION BY network, coalesce(nullif(trim(store_code), ''), store_id::VARCHAR)) AS skey
         FROM store)"""
# «—» in place of a value: after a colon, bracket or arrow, before punctuation or end of line
VALUE_DASH = re.compile(r"[:(→]\s*—|—\s*(?:$|[.,;)–])|—–|–—")


def q(sql):
    return S.q(sql)


def one(sql):
    return S.q1(sql)[0]


def months(y, m1, m2):
    return [date(y, m, 1) for m in range(m1, m2 + 1)]


def between(ms, col="n.month"):
    return f"{col} BETWEEN DATE '{ms[0]}' AND DATE '{ms[-1]}'"


def excl(cur, base, col_net="n.network", col_m="n.month"):
    """Incomplete chain-months are removed from both sides of the comparison (docs/methodology.md §3.7)."""
    pairs = []
    for c, b in zip(cur, base):
        for n, x in S.bad_pairs():
            if x in (c, b):
                pairs += [(n, c), (n, b)]
    return " AND ".join(f"NOT ({col_net} = '{n}' AND {col_m} = DATE '{m}')" for n, m in pairs) or "TRUE"


def num(v, d=0):
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


def mln(v):
    return f"{num(v / 1e6, 1)} млн руб."


def mlrd(v):
    return f"{num(v / 1e9, 2)} млрд руб."


def money(v):
    """Rubles in answer units: billions from 1 bn up, otherwise millions."""
    return mlrd(v) if round(abs(v) / 1e9, 2) >= 1 else mln(v)


def pct(v, d=1):
    return f"{num(abs(v) * 100, d)}%"


def dyn(g):
    return ("рост на " if g > 0 else "снижение на ") + pct(g)


def fact(r, start):
    hits = [f for f in r.facts if f.startswith(start)]
    assert hits, f"no fact «{start}…»: {r.facts[:4]}"
    return hits[0]


def table(r, start):
    hits = [b for b in r.blocks if b["kind"] == "table" and b["title"].startswith(start)]
    assert hits, f"no table «{start}…»"
    return hits[0]


def contract(r):
    """Common contract for all tools: slice line, 2–4 key facts, 2–4 next steps, no «—» in place of values."""
    assert r.slice_line.startswith("Срез:")
    assert 2 <= len(r.headline) <= 4 and all(h in r.facts for h in r.headline)
    assert 2 <= len(r.next) <= 4, r.next
    bad = [f for f in r.facts if VALUE_DASH.search(f)]
    assert not bad, bad
    assert len(r.facts_text()) <= 5000


def test_no_yo_in_own_text():
    for p in ("app/tools2.py", "tests/test_tools2.py"):
        assert "\u0451" not in Path(p).read_text(encoding="utf-8").lower(), p     # Cyrillic letter yo (U+0451)


# ——— factors ———

@pytest.fixture(scope="module")
def g3():
    return T2.t_factors({"manufacturer": "NORD", "network": "Гранит", "portal": True,
                         "period": "2026-01..2026-08", "base_period": "2025-01..2025-08"})


def test_factors_g3_reference(g3):
    """Reference G3: NORD, Гранит, reference-portal Cosmetics, Jan–Aug 2026 vs. Jan–Aug 2025: sales, growth, decomposition (SQL)."""
    contract(g3)
    head = g3.facts[0]
    cur, base = months(2026, 1, 8), months(2025, 1, 8)
    v = q(f"""SELECT year(n.month) y, sum(n.rub) v FROM net_sku n JOIN sku k USING (sku_id)
              WHERE k.manufacturer = 'NORD' AND k.in_portal AND n.network = 'Гранит'
                AND ({between(cur)} OR {between(base)}) GROUP BY 1""").set_index("y").v
    assert f"{mln(v[2025])} → {mln(v[2026])}" in head and dyn(v[2026] / v[2025] - 1) in head and "(Гранит)" in head
    assert "9 сопоставимых" not in "\n".join(g3.facts)               # L4-7: single-chain slice
    # average stores per month — by store_key, computed independently
    a = q(f"""SELECT year(f.month) y, count(DISTINCT (s.skey, f.month)) / 8.0 a FROM fact f JOIN {SK} s USING (store_id)
              JOIN sku k USING (sku_id) WHERE k.manufacturer = 'NORD' AND k.in_portal AND s.network = 'Гранит' AND f.qty > 0
                AND ({between(cur, 'f.month')} OR {between(base, 'f.month')}) GROUP BY 1""").set_index("y").a
    dec = fact(g3, "Разложение изменения (Гранит)")
    assert f"в среднем {num(a[2025])} → {num(a[2026])} точек в месяц" in dec
    # factor contributions add up to the change with no residual
    rows = table(g3, "Разложение изменения продаж")["rows"]
    assert abs(sum(x["d"] for x in rows[:-1]) - rows[-1]["d"]) < 1
    assert abs(rows[-1]["d"] - (v[2026] - v[2025])) < 1
    assert table(g3, "Разложение по баннерам (Гранит)")["rows"]                   # Маяк Косметик as a separate row
    mon = fact(g3, "По месяцам к тем же месяцам год назад (Гранит)")              # L4-16: no dashes
    assert mon.count(";") == 7


@pytest.fixture(scope="module")
def nord():
    return T2.t_factors({"manufacturer": "NORD", "all_categories": True})


def test_factors_comparable_networks(nord):
    """L4-3: headline growth and breakdowns over the 9 comparable chains without incomplete months; all chains as a separate line."""
    contract(nord)
    cur, base = months(2026, 1, 9), months(2025, 1, 9)
    ex = excl(cur, base)
    v = q(f"""SELECT year(n.month) y, sum(n.rub) FILTER (WHERE {CMP_SQL}) c, sum(n.rub) a FROM net_sku n JOIN sku k USING (sku_id)
              WHERE k.manufacturer = 'NORD' AND ({between(cur)} OR {between(base)}) AND {ex} GROUP BY 1""").set_index("y")
    head = nord.facts[0]
    assert "(9 сопоставимых сетей)" in head
    assert f"{mlrd(v.c[2025])} → {mlrd(v.c[2026])}" in head and dyn(v.c[2026] / v.c[2025] - 1) in head
    assert f"{mlrd(v.a[2025])} → {mlrd(v.a[2026])}" in fact(nord, "С учетом сетей с неполным покрытием (все сети)")
    # the ДФО district and delisted SKUs are also over the 9 chains
    d = q(f"""SELECT year(n.month) y, sum(n.rub) v FROM net_sku n JOIN sku k USING (sku_id)
              WHERE k.manufacturer = 'NORD' AND n.district = 'ДФО' AND {CMP_SQL} AND ({between(cur)} OR {between(base)}) AND {ex}
              GROUP BY 1""").set_index("y").v
    row = next(x for x in table(nord, "Вклад в изменение по округам")["rows"] if x["name"] == "ДФО")
    assert abs(row["v1"] - d[2026]) < 1 and abs(row["v0"] - d[2025]) < 1
    gone = one(f"""SELECT count(*) FROM (SELECT k.sku_name, sum(n.rub) FILTER (WHERE year(n.month) = 2026) v1,
                                         sum(n.rub) FILTER (WHERE year(n.month) = 2025) v0
                                  FROM net_sku n JOIN sku k USING (sku_id)
                                  WHERE k.manufacturer = 'NORD' AND {CMP_SQL} AND ({between(cur)} OR {between(base)}) AND {ex}
                                  GROUP BY 1) WHERE v0 > 0 AND coalesce(v1, 0) <= 0""")
    assert f"выведены (продажи были в базовом периоде, в текущем нет) {gone} SKU" in fact(nord, "Жизненный цикл SKU")
    mk = fact(nord, "Рынок в категориях NORD")                                      # L4-14: increment, not level
    assert "При темпе рынка изменение продаж NORD было бы" in mk and ("выше рынка" in mk or "ниже рынка" in mk)


def test_factors_month_to_month():
    """L4-15: base is the previous month: labels «к июлю 2026», no «к прошлому году» and no by-month line."""
    r = T2.t_factors({"manufacturer": "NORD", "network": "Гранит", "period": "2026-08", "base_period": "2026-07"})
    contract(r)
    text = "\n".join(r.facts)
    assert "август 2026 к июлю 2026" in r.facts[0] and "прошлому году" not in text and "По месяцам" not in text
    assert any(f.startswith("Доля NORD в категории") and "август 2026 к июлю 2026" in f for f in r.facts)


# ——— distribution ———

def test_distribution_kids_split():
    """L4-11: Гранит, NORD, hair hygiene — adult / kids / total; stores by store_key."""
    r = T2.t_distribution({"network": "Гранит", "subgroup": "Гигиена волос и кожи головы", "manufacturer": "NORD", "period": "2026-09"})
    contract(r)
    st = q(f"""SELECT k.is_kids kid, count(DISTINCT s.skey) FILTER (WHERE k.manufacturer = 'NORD') obj, count(DISTINCT s.skey) den
               FROM fact f JOIN {SK} s USING (store_id) JOIN sku k USING (sku_id)
               WHERE f.month = DATE '2026-09-01' AND f.qty > 0 AND s.network = 'Гранит' AND k.subgroup_base = 'Гигиена волос и кожи головы'
               GROUP BY ROLLUP (1)""")
    tot, ad, kd = (st[st.kid.isna()].iloc[0], st[st.kid == False].iloc[0], st[st.kid == True].iloc[0])  # noqa: E712
    f0 = r.facts[0]
    assert f"объект в {num(tot.obj)} из {num(tot.den)} точек с категорией ({pct(tot.obj / tot.den)})" in f0
    assert f"взрослые подгруппы {num(ad.obj)} из {num(ad.den)}" in f0 and f"детские {num(kd.obj)} из {num(kd.den)}" in f0


def test_distribution_banner_label_and_per_store():
    """L4-8: with a banner filter the label is the banner, not Гранит; sales per store checked with SQL."""
    r = T2.t_distribution({"brand": "ESTRA", "product_type": "Шампунь", "banner": "Маяк Косметик", "period": "2026-08"})
    contract(r)
    x = q(f"""SELECT count(DISTINCT s.skey) FILTER (WHERE k.brand = 'ESTRA') obj, count(DISTINCT s.skey) den,
                     sum(f.rub) FILTER (WHERE k.brand = 'ESTRA') rub
              FROM fact f JOIN {SK} s USING (store_id) JOIN sku k USING (sku_id)
              WHERE f.month = DATE '2026-08-01' AND f.qty > 0 AND s.banner = 'Маяк Косметик' AND k.product_type = 'Шампунь'""").iloc[0]
    f0 = r.facts[0]
    assert f0.startswith("Итого (баннер: Маяк Косметик)") and "Гранит" not in "\n".join(r.facts)
    assert f"в {num(x.obj)} из {num(x.den)} точек" in f0 and f"в среднем {num(x.rub / x.obj)} руб. на точку" in f0


def test_distribution_store_months():
    """L4-10: sales per store per month = sales / sum of monthly store counts (Перелесок, NORD, Jan–Sep 2026)."""
    r = T2.t_distribution({"network": "Перелесок ТД", "subgroup": "Гигиена волос и кожи головы", "manufacturer": "NORD",
                           "period": "2026-01..2026-09"})
    contract(r)
    x = q(f"""SELECT sum(f.rub) rub, count(DISTINCT (s.skey, f.month)) sm, count(DISTINCT s.skey) st
              FROM fact f JOIN {SK} s USING (store_id) JOIN sku k USING (sku_id)
              WHERE {between(months(2026, 1, 9), 'f.month')} AND f.qty > 0 AND s.network = 'Перелесок ТД'
                AND k.manufacturer = 'NORD' AND k.subgroup_base = 'Гигиена волос и кожи головы'""").iloc[0]
    assert f"в среднем {num(x.rub / x.sm)} руб. на точку с продажами в месяц" in r.facts[0]
    assert f"объект в {num(x.st)} из" in r.facts[0]


def test_distribution_apostrophe_level_network_and_default_month():
    """L4-9, L4-21, §5.5: apostrophe in a name, level=network as a breakdown, default month is the last one complete for the slice's chains."""
    r = T2.t_distribution({"brand": "L'AURELLE PARIS"})
    contract(r)
    assert f"{S.last_ok_month():%Y}" in r.facts[0] and S.MONTHS_FULL[S.last_ok_month().month - 1] in r.facts[0]
    r2 = T2.t_distribution({"manufacturer": "NORD", "level": "network", "network": "Ажур", "period": "2026-08"})
    contract(r2)
    with pytest.raises(ToolError, match="нет продаж за"):
        T2.t_distribution({"brand": "ESTRA", "network": "Детский Остров (сеть)"})


# ——— prices ———

def test_prices_m1_cost_and_markup():
    """Reference M1: kids' shampoos of NORD and AURIGA in Нива and Гранит, 09.2026: price, unit cost, the difference and chain markup."""
    r = T2.t_prices({"product_type": "Шампунь", "kids": True, "manufacturer": ["NORD", "AURIGA"],
                     "network": ["Нива (сеть)", "Гранит"], "period": "2026-09", "level": "manufacturer"})
    contract(r)
    d = q("""SELECT network, k.manufacturer man, sum(rub) / sum(qty) price,
                    sum(cost) FILTER (WHERE cost <> 0) / sum(qty) FILTER (WHERE cost <> 0) cost_u,
                    sum(rub) FILTER (WHERE cost <> 0) / sum(cost) FILTER (WHERE cost <> 0) - 1 markup
             FROM net_sku n JOIN sku k USING (sku_id)
             WHERE n.month = DATE '2026-09-01' AND k.product_type = 'Шампунь' AND k.is_kids
               AND k.manufacturer IN ('NORD', 'AURIGA') AND n.network IN ('Нива (сеть)', 'Гранит') GROUP BY 1, 2""")
    for x in d.itertuples():
        f = fact(r, f"{S.net_label(x.network)} / {x.man}:")
        assert f"цена {num(x.price, 1)} руб./шт." in f and f"закупка {num(x.cost_u, 1)} руб./шт." in f
        assert f"разница цены и закупки {num(x.price - x.cost_u, 1)} руб./шт." in f and f"наценка сети {pct(x.markup)}" in f
    assert {("Нива (сеть)", "NORD"), ("Гранит", "NORD")} <= set(zip(d.network, d.man))      # reference rows are checked above


def test_prices_no_markup_across_networks():
    """L4-4: a breakdown by format mixes chains, so cost and markup are not computed."""
    r = T2.t_prices({"product_type": "Шампунь", "period": "2026-08", "level": "manufacturer", "by": "format"})
    contract(r)
    assert any(f.startswith("Закупка и наценка не считаются") for f in r.facts)
    assert not any("наценка сети" in f or "закупка " in f for f in r.facts if " / " in f.split(":")[0])


def test_prices_dynamics_and_index():
    """L4-13, L4-24: price change vs. base, price index on identical SKUs in the same chains; level is brand; markup < 0 comes with a caveat."""
    r = T2.t_prices({"brand": "L'AURELLE PARIS", "period": "2026-01..2026-09", "base_period": "2025-01..2025-09", "all_categories": True})
    contract(r)
    cur, base = months(2026, 1, 9), months(2025, 1, 9)
    ex = excl(cur, base)
    p = q(f"""SELECT year(n.month) y, sum(n.rub) / sum(n.qty) p FROM net_sku n JOIN sku k USING (sku_id)
              WHERE k.brand = 'L''AURELLE PARIS' AND {CMP_SQL} AND ({between(cur)} OR {between(base)}) AND {ex} GROUP BY 1""").set_index("y").p
    idx = one(f"""SELECT sum(r1 / q1 * q0) / sum(r0) - 1 FROM (
                    SELECT k.sku_name, n.network, sum(n.rub) FILTER (WHERE year(n.month) = 2026) r1, sum(n.qty) FILTER (WHERE year(n.month) = 2026) q1,
                           sum(n.rub) FILTER (WHERE year(n.month) = 2025) r0, sum(n.qty) FILTER (WHERE year(n.month) = 2025) q0
                    FROM net_sku n JOIN sku k USING (sku_id)
                    WHERE k.brand = 'L''AURELLE PARIS' AND {CMP_SQL} AND ({between(cur)} OR {between(base)}) AND {ex} GROUP BY 1, 2)
                  WHERE q1 > 0 AND q0 > 0 AND r1 > 0 AND r0 > 0""")
    f = fact(r, "Цены янв–сен 2026 к янв–сен 2025 (9 сопоставимых сетей)")
    assert f"{num(p[2025], 1)} → {num(p[2026], 1)} руб./шт." in f and f"(индекс цен) {dyn(idx)}" in f
    t = fact(r, "Гранит / L'AURELLE PARIS:")                                        # level is brand, not manufacturer
    assert "доля" not in t and ("особенность учета источника" in t or "наценка сети −" not in t)


# ——— abc ———

def test_abc_classes():
    r = T2.t_abc({"network": "Гранит", "manufacturer": "NORD", "all_categories": True})
    contract(r)
    d = q("""SELECT k.sku_name, sum(n.rub) v FROM net_sku n JOIN sku k USING (sku_id)
             WHERE k.manufacturer = 'NORD' AND n.network = 'Гранит' AND n.month BETWEEN DATE '2025-10-01' AND DATE '2026-09-01'
             GROUP BY 1 HAVING sum(n.rub) > 0 ORDER BY v DESC, sku_name""")
    prev = (d.v.cumsum() / d.v.sum()).shift(1).fillna(0)
    cls = ["A" if x < 0.8 else ("B" if x < 0.95 else "C") for x in prev]
    f = fact(r, "Классы ABC:")
    for c in "ABC":
        assert f"{c}: {cls.count(c)} позиц" in f
    assert f"SKU с продажами: {len(d)}" in r.facts[0]
    assert fact(r, "1. ").startswith(f"1. {d.sku_name.iloc[0]}") and mln(d.v.iloc[0]) in fact(r, "1. ")
    with pytest.raises(ToolError, match="нет продаж за"):                          # L4-19
        T2.t_abc({"brand": "ESTRA", "network": "Детский Остров (сеть)"})


# ——— assortment ———

def test_assortment_both_modes():
    """L4-1, L4-20: both modes work; denominator is stores with the category; «кандидатов N» equals the table rows."""
    base = {"network": "Перелесок ТД", "subgroup": "Гигиена волос и кожи головы", "kids": True, "n": 3}
    r = T2.t_assortment(dict(base, mode="delist", period="2025-01..2026-09"))
    contract(r)
    den = one(f"""SELECT count(DISTINCT s.skey) FROM fact f JOIN {SK} s USING (store_id) JOIN sku k USING (sku_id)
                  WHERE f.month = DATE '2026-09-01' AND f.qty > 0 AND s.network = 'Перелесок ТД'
                    AND k.subgroup_base = 'Гигиена волос и кожи головы' AND k.is_kids""")
    assert f"в 09.2026: {num(den)}." in r.facts[0]
    rows = table(r, "Кандидаты на вывод")["rows"]
    assert rows and f"кандидатов на вывод {len(rows)}" in r.facts[1]
    sku = rows[0]["name"].replace(" [СТМ]", "")
    st = one(f"""SELECT count(DISTINCT s.skey) FROM fact f JOIN {SK} s USING (store_id) JOIN sku k USING (sku_id)
                 WHERE f.month = DATE '2026-09-01' AND f.qty > 0 AND s.network = 'Перелесок ТД' AND k.sku_name = '{sku.replace("'", "''")}'""")
    assert abs(rows[0]["cov"] - st / den) < 1e-9
    r2 = T2.t_assortment(dict(base, mode="introduce", format="СМ"))
    contract(r2)
    assert table(r2, "Кандидаты на ввод")["rows"]
    with pytest.raises(ToolError, match="min_coverage"):
        T2.t_assortment({"network": "Гранит", "manufacturer": "NORD", "mode": "delist"})


# ——— forecast ———

def test_forecast_nine_networks_and_refusals():
    """L4-17: series over the 9 chains; L4-6: an unreliable forecast or discontinued sales yield a refusal, not a number."""
    r = T2.t_forecast({"target_month": "2026-10", "all_categories": True})
    contract(r)
    ly = one(f"SELECT sum(rub) FROM net_sku n WHERE n.month = DATE '2025-10-01' AND {CMP_SQL}")
    assert fact(r, "Тот же месяц год назад").startswith(f"Тот же месяц год назад (октябрь 2025): {mlrd(ly)}")
    assert "(9 сопоставимых сетей)" in r.facts[0] and "млрд руб." in r.facts[0]
    with pytest.raises(ToolError, match="ненадежен"):
        T2.t_forecast({"brand": "SEDUCTEUR", "target_month": "2026-10"})
    with pytest.raises(ToolError, match="заканчиваются в 08.2026"):
        T2.t_forecast({"network": "Офис Х7"})


# ——— competitors ———

@pytest.fixture(scope="module")
def g2():
    return T2.t_competitors({"manufacturer": "NORD", "portal": True, "period": "2026-01..2026-08"})


def test_competitors_g2_reference(g2):
    """Reference G2: NORD's competitors in the reference-portal Cosmetics category, Jan–Aug 2026: list, sales and order (SQL)."""
    contract(g2)
    lst = fact(g2, "Конкуренты (")
    d = q("""WITH seg AS (SELECT k.subgroup_name sg, k.product pr, sum(n.rub) v FROM net_sku n JOIN sku k USING (sku_id)
                         WHERE k.manufacturer = 'NORD' AND k.in_portal AND n.month BETWEEN DATE '2026-01-01' AND DATE '2026-08-01'
                         GROUP BY 1, 2),
                  s AS (SELECT sg, pr, v FROM seg WHERE v >= 0.005 * (SELECT sum(v) FROM seg))
             SELECT k.manufacturer man, sum(n.rub) v, (SELECT sum(v) FROM s) t FROM net_sku n JOIN sku k USING (sku_id)
             JOIN s ON k.subgroup_name = s.sg AND k.product = s.pr
             WHERE k.in_portal AND n.month BETWEEN DATE '2026-01-01' AND DATE '2026-08-01' GROUP BY 1""")
    t = d.t.iloc[0]
    comp = d[(d.v >= 0.5 * t) & (d.v <= 2 * t) & (d.man != "NORD") & ~d.man.str.startswith("STM(") & (d.man != "не указан")]
    assert len(comp) >= 2 and lst.startswith(f"Конкуренты ({len(comp)})")
    for x in comp.itertuples():
        assert f"{x.man} {money(x.v)}" in lst
    comp = comp.sort_values(["v", "man"], ascending=[False, True])
    assert [lst.index(f"{x} ") for x in comp.man] == sorted(lst.index(f"{x} ") for x in comp.man)   # by descending sales
    assert money(t) in g2.facts[0]


def test_competitors_brand_target_and_leaders():
    """L4-2, L4-12, L4-23: brand as target, growth overall and within segments, dropped segments, «не указан» is never a leader, private label is marked."""
    r = T2.t_competitors({"manufacturer": "L'AURELLE", "brand": "L'AURELLE PARIS"})
    contract(r)
    lst = fact(r, "Конкуренты (")
    assert "L'AURELLE " not in lst.split(":", 1)[1]
    # leaders of the target's largest segment: shares of the whole market segment (not 100% for the target), «не указан» is not a leader
    ytd, cats = between(months(2026, 1, 9)), ", ".join(S._lit(c) for c in S.own_categories())
    seg = q(f"""SELECT k.subgroup_name sg, k.product pr, sum(n.rub) v FROM net_sku n JOIN sku k USING (sku_id)
                WHERE k.brand = 'L''AURELLE PARIS' AND k.subgroup_base IN ({cats}) AND {ytd}
                GROUP BY 1, 2 ORDER BY v DESC LIMIT 1""").iloc[0]
    sm = q(f"""SELECT k.manufacturer man, sum(n.rub) v FROM net_sku n JOIN sku k USING (sku_id)
               WHERE k.subgroup_name = {S._lit(seg.sg)} AND k.product = {S._lit(seg.pr)} AND {ytd} GROUP BY 1""")
    top = sm[~sm.man.isin(["не указан", "."])].sort_values(["v", "man"], ascending=[False, True]).iloc[0]
    lead = fact(r, f"Лидеры сегмента {seg.sg} / {seg.pr} (")
    assert lead.split(": ", 1)[1].startswith(f"{top.man} {pct(top.v / sm.v.sum())}") and "100,0%" not in lead
    g = fact(r, "L'AURELLE PARIS янв–сен 2026 к янв–сен 2025 (9 сопоставимых сетей)")
    assert "всего снижение" in g and "в своих сегментах" in g
    assert fact(r, "Сегменты, выпавшие из списка L'AURELLE PARIS")
    r2 = T2.t_competitors({"manufacturer": "AVANGARD"})
    contract(r2)
    for f in r2.facts:
        if f.startswith("Лидеры сегмента"):
            first = f.split(": ", 1)[1]
            assert not first.startswith("не указан") and not first.startswith("производитель не указан")
            assert all("[СТМ]" in x for x in re.findall(r"STM\([^)]*\)[^;]*", first))
