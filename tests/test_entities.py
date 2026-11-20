"""Question parsing (app/entities.py): a "question -> what the code must pass on" table — filters, exclusions,
period and base (as_args), measure, kids, sorting, comparison target, breakdown, follow-up, out of range. No LLM.

Run: python -m pytest -q tests/test_entities.py
Filters are compared as a whole (an extra filter is an error: the code would force it on the model). Periods without
a year are counted from the last month of data (currently 09.2026) via the helpers below.
"""
import json
import os
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("VT_THREADS", "4")
os.environ.setdefault("VT_MEMORY", "6GB")

import pytest

from app import entities as E
from app import semantic as S

D = E.dictionary()
LO, HI = S.data_bounds()
LAST = S.ym(HI)


def ps(a, b=None):
    return f"{a}..{b or a}"


def back(p, n=12):
    a, b = p.split("..")
    return ps(S.ym(S.add_months(S.parse_ym(a), -n)), S.ym(S.add_months(S.parse_ym(b), -n)))


def last_n(n):
    return ps(S.ym(S.add_months(HI, -n + 1)), LAST)


def yr(m):
    """Year of the latest month m in the data (month named without a year)."""
    return HI.year if m <= HI.month else HI.year - 1


def latest(m):
    return ps(f"{yr(m)}-{m:02d}")


def full(y, a, b):
    """Latest year for which months a..b (a > b wraps over the year end) are fully in the data."""
    y = HI.year if date(HI.year, b, 1) <= HI else HI.year - 1
    return ps(f"{y - (a > b)}-{a:02d}", f"{y}-{b:02d}")


def nxt(m):
    return f"{HI.year if m > HI.month else HI.year + 1}-{m:02d}"


Y26 = ps("2026-01", LAST)
Y26B = back(Y26)
Q_END = S.add_months(HI, -(HI.month % 3))
LAST_Q = ps(S.ym(S.add_months(Q_END, -2)), S.ym(Q_END))
NORD = {"manufacturer": "NORD"}
NIVA, GRANIT, AZHUR, X7 = "Нива (сеть)", "Гранит", "Ажур", ["Семерочка Х7", "Перелесок ТД"]
SOAPS = ["Мыло Жидкое", "Мыло кусковое", "Мыло хозяйственное"]
KIDS_HAIR = "Гигиена волос и кожи головы"
COLOR = "Средства для окраски волос"

CASES = [
    # ——— periods: current is the later one, base is the same months (L2-1, L2-2, L2-10, L2-13) ———
    ("Продажи NORD январь–август 2026", {"f": NORD, "p": "2026-01..2026-08"}),
    ("Как изменились продажи NORD по сетям: январь–август 2026 к январю–августу 2025",
     {"f": NORD, "p": "2026-01..2026-08", "b": "2025-01..2025-08", "lvl": "network"}),
    ("Продажи NORD по сетям в январе–августе 2026 к тому же периоду 2025",
     {"f": NORD, "p": "2026-01..2026-08", "b": "2025-01..2025-08", "lvl": "network"}),
    ("Продажи NORD 2026 к 2025", {"f": NORD, "p": Y26, "b": Y26B}),
    ("Продажи NORD 2026 г. к 2025 г.", {"f": NORD, "p": Y26, "b": Y26B}),
    ("Продажи NORD за 9 мес. 2026 к 9 мес. 2025", {"f": NORD, "p": "2026-01..2026-09", "b": "2025-01..2025-09"}),
    ("Продажи NORD сентябрь 2026 к сентябрю 2025", {"f": NORD, "p": "2026-09..2026-09", "b": "2025-09..2025-09"}),
    ("Продажи май 2026 к маю 2025", {"p": "2026-05..2026-05", "b": "2025-05..2025-05"}),
    ("Продажи NORD в 3 квартале 2026 к 3 кварталу 2025", {"f": NORD, "p": "2026-07..2026-09", "b": "2025-07..2025-09"}),
    ("Сравни январь, февраль и март 2026", {"p": "2026-01..2026-03", "note": "несколько периодов"}),
    ("Продажи NORD за последние 3 месяца 2026", {"f": NORD, "p": last_n(3)}),
    ("Как изменились продажи NORD к прошлому году?", {"f": NORD, "note": "теми же месяцами прошлого года"}),
    ("Сравни динамику продаж с прошлым годом", {"note": "теми же месяцами прошлого года"}),
    ("Продажи NORD в Х7 в 2026 году к прошлому году", {"f": {**NORD, "network": X7}, "p": Y26, "b": Y26B}),
    ("Продажи NORD в январе–августе 2026 к аналогичному периоду прошлого года",
     {"f": NORD, "p": "2026-01..2026-08", "b": "2025-01..2025-08"}),
    ("Продажи NORD в сентябре год к году", {"f": NORD, "p": latest(9), "b": back(latest(9))}),
    ("Продажи NORD в 2026 году к АППГ", {"f": NORD, "p": Y26, "b": Y26B}),
    ("Продажи NORD за 1 кв 2026", {"f": NORD, "p": "2026-01..2026-03"}),
    ("Продажи NORD за 1 кв. 2026", {"f": NORD, "p": "2026-01..2026-03"}),
    ("Продажи NORD 1Q 2026", {"f": NORD, "p": "2026-01..2026-03"}),
    ("Продажи NORD Q1 2026", {"f": NORD, "p": "2026-01..2026-03"}),
    ("Продажи NORD в первом квартале 2026", {"f": NORD, "p": "2026-01..2026-03"}),
    ("Продажи NORD в 1-м квартале 2026", {"f": NORD, "p": "2026-01..2026-03"}),
    ("Продажи NORD III квартал 2025", {"f": NORD, "p": "2025-07..2025-09"}),
    ("Продажи NORD 4 кв 2025", {"f": NORD, "p": "2025-10..2025-12"}),
    ("Продажи NORD в 1 квартале 2026 и 1 квартале 2025", {"f": NORD, "p": "2026-01..2026-03", "b": "2025-01..2025-03"}),
    ("Продажи NORD первое полугодие 2026", {"f": NORD, "p": "2026-01..2026-06"}),
    ("Продажи NORD в 1-м полугодии 2026", {"f": NORD, "p": "2026-01..2026-06"}),
    ("Продажи NORD 2 полугодие 2025", {"f": NORD, "p": "2025-07..2025-12"}),
    ("Продажи шампуней зима 2025", {"f": {"product_type": "Шампунь"}, "p": "2024-12..2025-02"}),
    ("Продажи шампуней осенью 2025", {"f": {"product_type": "Шампунь"}, "p": "2025-09..2025-11"}),
    ("Продажи шампуней летом 2025", {"f": {"product_type": "Шампунь"}, "p": "2025-06..2025-08"}),
    ("Продажи шампуней весна 2025", {"f": {"product_type": "Шампунь"}, "p": "2025-03..2025-05"}),
    ("Сравни два периода: весну и лето 2025", {"p": "2025-06..2025-08", "b": "2025-03..2025-05"}),
    ("Продажи солнцезащитных средств летом", {"f": {"subgroup": "Солнцезащитные средства и уход"}, "p": full(0, 6, 8)}),
    ("Продажи шампуней зимой", {"f": {"product_type": "Шампунь"}, "p": full(0, 12, 2)}),
    ("Продажи NORD во втором полугодии", {"f": NORD, "p": full(0, 7, 12)}),
    ("Продажи NORD за 2 квартал", {"f": NORD, "p": full(0, 4, 6)}),
    ("Продажи NORD за 6 месяцев 2026", {"f": NORD, "p": "2026-01..2026-06"}),
    ("Продажи NORD за 2026-08", {"f": NORD, "p": "2026-08..2026-08"}),
    ("Продажи NORD за 08.2026", {"f": NORD, "p": "2026-08..2026-08"}),
    ("Продажи NORD за 01.2026–06.2026", {"f": NORD, "p": "2026-01..2026-06"}),
    ("Продажи NORD с марта 2025 по май 2026", {"f": NORD, "p": "2025-03..2026-05"}),
    ("Продажи NORD за октябрь 2025 – март 2026", {"f": NORD, "p": "2025-10..2026-03"}),
    ("Продажи NORD с октября 2025 по март 2026", {"f": NORD, "p": "2025-10..2026-03"}),
    ("Продажи NORD за 2024–2026", {"f": NORD, "p": ps("2024-01", LAST)}),
    ("Продажи NORD за 2025–2026 годы", {"f": NORD, "p": ps("2025-01", LAST)}),
    ("Сравни 2025 и 2026 годы по NORD", {"f": NORD, "p": Y26, "b": Y26B}),
    ("Продажи NORD в 2025 vs 2024", {"f": NORD, "p": "2025-01..2025-12", "b": "2024-01..2024-12"}),
    ("Продажи NORD с января по март", {"f": NORD, "p": full(0, 1, 3)}),
    ("Продажи NORD с октября по март", {"f": NORD, "p": full(0, 10, 3)}),
    ("Продажи NORD за последний квартал", {"f": NORD, "p": last_n(3)}),
    ("Продажи NORD за прошлый квартал", {"f": NORD, "p": LAST_Q}),
    ("Продажи NORD за полгода", {"f": NORD, "p": last_n(6)}),
    ("Продажи NORD за последние 6 месяцев", {"f": NORD, "p": last_n(6)}),
    ("Продажи NORD за последние 3 месяца", {"f": NORD, "p": last_n(3)}),
    ("Продажи NORD за последние 12 месяцев", {"f": NORD, "p": "mat"}),
    ("Продажи NORD за последний год", {"f": NORD, "p": "mat"}),
    ("Продажи NORD в этом году", {"f": NORD, "p": "ytd"}),
    ("Продажи NORD с начала года", {"f": NORD, "p": "ytd"}),
    ("Продажи NORD за весь период", {"f": NORD, "p": "all"}),
    ("Продажи NORD за все время", {"f": NORD, "p": "all"}),
    ("Продажи NORD за прошлый месяц", {"f": NORD, "p": ps(LAST)}),
    ("Продажи NORD за прошлый год", {"f": NORD, "p": f"{HI.year - 1}-01..{HI.year - 1}-12"}),
    ("Продажи NORD в мае", {"f": NORD, "p": latest(5)}),
    ("Продажи NORD в декабре", {"f": NORD, "p": latest(12)}),
    ("Продажи NORD в октябре", {"f": NORD, "p": latest(10)}),
    ("Как изменились продажи NORD в августе по сравнению с 2025?", {"f": NORD, "p": latest(8), "b": back(latest(8))}),
    ("Сравни январь и июнь 2025", {"p": "2025-06..2025-06", "b": "2025-01..2025-01"}),
    ("Продажи NORD с 2025 года", {"f": NORD, "p": ps("2025-01", LAST)}),
    ("Продажи NORD с марта 2026", {"f": NORD, "p": ps("2026-03", LAST)}),
    ("Продажи NORD в 2023 году", {"f": NORD, "out": ["2023 год"]}),
    ("Продажи NORD в октябре 2030", {"f": NORD, "out": ["октябрь 2030"]}),
    ("Прогноз продаж NORD на ноябрь 2030", {"f": NORD, "fut": "2030-11"}),
    ("Прогноз продаж NORD на декабрь", {"f": NORD, "fut": nxt(12)}),
    ("Сколько NORD продаст в декабре?", {"f": NORD, "fut": nxt(12)}),
    ("Прогноз продаж NORD на 4 квартал 2030", {"f": NORD}),
    # ——— exclusions and kids' products (L2-3, L2-4) ———
    ("Продажи NORD во всех сетях кроме Гранита", {"f": NORD, "ex": {"network": GRANIT}}),
    ("Продажи всех производителей кроме NORD в Ажуре", {"f": {"network": AZHUR}, "ex": NORD}),
    ("Продажи NORD в Маяке без Маяк Косметик", {"f": {**NORD, "network": GRANIT}, "ex": {"banner": "Маяк Косметик"}}),
    ("Продажи NORD за исключением Нивы и Ажура", {"f": NORD, "ex": {"network": [NIVA, AZHUR]}}),
    ("Продажи шампуней кроме NORD и AURIGA в Граните",
     {"f": {"product_type": "Шампунь", "network": GRANIT}, "ex": {"manufacturer": ["NORD", "AURIGA"]}}),
    ("Продажи NORD не считая Семерочки", {"f": NORD, "ex": {"network": "Семерочка Х7"}}),
    ("Продажи без СТМ Маяка в Граните", {"f": {"network": GRANIT}, "ex": {"manufacturer": "STM(GRANIT)"}}),
    ("Продажи NORD без детских", {"f": NORD, "kids": False}),
    ("Доля NORD без учета детских шампуней", {"f": {**NORD, "product_type": "Шампунь"}, "kids": False}),
    ("Продажи NORD без взрослых шампуней", {"f": {**NORD, "product_type": "Шампунь"}, "kids": True}),
    ("Продажи шампуней без СТМ в Ниве", {"f": {"product_type": "Шампунь", "network": NIVA}, "pl": True}),
    ("Продажи NORD без собственных марок сетей", {"f": NORD, "pl": True}),
    ("Сравни взрослые и детские шампуни NORD", {"f": {**NORD, "product_type": "Шампунь"}, "both": True}),
    ("Доля детских и взрослых шампуней в Граните", {"f": {"product_type": "Шампунь", "network": GRANIT}, "both": True}),
    ("Топ-5 брендов в Детском острове", {"f": {"network": "Детский Остров (сеть)"}, "lvl": "brand"}),
    ("Продажи шампуней для детей в Ниве", {"f": {"product_type": "Шампунь", "network": NIVA}, "kids": True}),
    ("Продажи детских шампуней NORD", {"f": {**NORD, "product_type": "Шампунь"}, "kids": True}),
    # ——— chains, banners, formats: short aliases (L2-5, L2-18, L2-19) ———
    ("Верно ли, что продажи NORD растут в 2026 году?", {"f": NORD, "p": Y26}),
    ("Верно ли, что NORD падает в Граните?", {"f": {**NORD, "network": GRANIT}}),
    ("Верните прошлый ответ", {}),
    ("Ладно, теперь покажи по брендам в Граните", {"f": {"network": GRANIT}, "fu": True, "lvl": "brand"}),
    ("Ладно, а в штуках?", {"m": "qty", "fu": True}),
    ("Назови семерку лидеров среди брендов шампуней", {"f": {"product_type": "Шампунь"}, "lvl": "brand"}),
    ("Топ-семерка брендов в Ниве", {"f": {"network": NIVA}, "lvl": "brand"}),
    ("Продажи NORD в ЛАД", {"f": {**NORD, "network": "ЛАД (сеть)"}}),
    ("Продажи NORD в Ладно", {"f": {**NORD, "network": "ЛАД (сеть)"}}),
    ("Продажи в Ла'Дно", {"f": {"network": "ЛАД (сеть)"}}),
    ("Продажи NORD в Семерке", {"f": {**NORD, "network": "Семерочка Х7"}}),
    ("Продажи в Семёрочке", {"f": {"network": "Семерочка Х7"}}),
    ("Продажи в Детострове", {"f": {"network": "Детский Остров (сеть)"}}),
    ("Продажи в Депо C&C", {"f": {"network": "Депо Кэш Энд Керри"}}),
    ("Продажи NORD в X7", {"f": {**NORD, "network": X7}}),
    ("Продажи NORD в Х7 за 9 месяцев 2026", {"f": {**NORD, "network": X7}, "p": "2026-01..2026-09"}),
    ("Продажи NORD в Маяк Косметик и Маяк у дома", {"f": {**NORD, "banner": ["Маяк Косметик", "Маяк у дома"]}}),
    ("Продажи в гипермаркетах Нивы", {"f": {"format": "ГМ", "network": NIVA}}),
    ("Продажи NORD в Маяк Семейный", {"f": {**NORD, "banner": "Маяк гипермаркет"}}),
    ("Продажи NORD в Нашей Цене", {"f": {**NORD, "banner": "Наша Цена"}}),
    ("Продажи NORD в магазинах Наша Цена", {"f": {**NORD, "banner": "Наша Цена"}}),
    ("Сравни продажи NORD в Семерочке и Маяк у дома", {"f": {**NORD, "banner": ["Маяк у дома", "Семерочка"]},
                                                       "lvl": "banner"}),
    ("Сравни продажи NORD в Маяке и Маяк Косметик",
     {"f": {**NORD, "network": GRANIT}, "lvl": "banner", "ment": {"banner": ["Маяк Косметик"]}}),
    ("Продажи в Стрижике и Маятнике", {"f": {"network": ["Стрижик", "Маятник (сеть)"]}}),
    ("Продажи в Камелии и Запасе", {"f": {"network": ["Камелия (сеть)", "Запас"]}}),
    ("Продажи запас не нужны", {}),
    ("Продажи в сети Добрый и в Стрижике", {"f": {"network": ["Добрый (сеть)", "Стрижик"]}}),
    ("Продажи в сети НУ", {"f": {"network": "НУ (сеть)"}}),
    ("Да, покажи продажи в Ниве", {"f": {"network": NIVA}}),
    ("Продажи в Парусе", {"f": {"network": "ПАРУС (сеть)"}}),
    ("Продажи NORD в магазинах у дома", {"f": {**NORD, "format": "ММ"}}),
    ("Продажи NORD онлайн", {"f": {**NORD, "format": "Интернет Магазин"}}),
    ("Продажи NORD в супермаркетах Маяка", {"f": {**NORD, "format": "СМ", "network": GRANIT}}),
    # ——— manufacturers and brands: the longest key wins (L2-6, K6, L2-15, L2-16, L2-17, L2-22) ———
    ("Продажи GREEN SHOP в Ниве", {"f": {"brand": "GREEN SHOP", "network": NIVA}}),
    ("Продажи Green Kitchen в Граните", {"f": {"brand": "GREEN KITCHEN", "network": GRANIT}}),
    ("Продажи бренда TERRA BOREALIS в Ниве", {"f": {"brand": "TERRA BOREALIS", "network": NIVA}}),
    ("Продажи TERRA BOREALIS в Ниве", {"f": {"manufacturer": "TERRA BOREALIS", "network": NIVA},
                                         "note": "есть также бренд TERRA BOREALIS"}),
    ("Продажи Терра Бореалис в Маяке", {"f": {"manufacturer": "TERRA BOREALIS", "network": GRANIT}}),
    ("Продажи L'AURELLE PARIS в Граните", {"f": {"brand": "L'AURELLE PARIS", "network": GRANIT}}),
    ("Продажи бренда L'Aurelle Paris в Ниве в 2026 году", {"f": {"brand": "L'AURELLE PARIS", "network": NIVA}, "p": Y26}),
    ("Продажи MIXTA в Ниве", {"f": {"brand": "MIXTA", "network": NIVA}}),
    ("Продажи производителя MIXTA", {"f": {"manufacturer": "MIXTA"}}),
    ("Продажи FARMIX", {"f": {"manufacturer": "FARMIX"}, "note": "есть также бренд FARMIX"}),
    ("Продажи бренда FARMIX", {"f": {"brand": "FARMIX"}}),
    ("Продажи SHENZHEN NORD TECHNOLOGY в Ниве", {"f": {"manufacturer": "SHENZHEN NORD TECHNOLOGY CO.", "network": NIVA}}),
    ("Продажи NORDAKU в Ниве", {"f": {"brand": "NORDAKU", "network": NIVA}}),
    ("Продажи норд в Ниве", {"f": {**NORD, "network": NIVA}}),
    ("Продажи компании НОРД в Ниве", {"f": {**NORD, "network": NIVA}}),
    ("Продажи SWANLY в Ниве", {"f": {"brand": "СВОНЛИ [SWANLY]", "network": NIVA}}),
    ("Продажи Свонли в Ниве", {"f": {"brand": "СВОНЛИ [SWANLY]", "network": NIVA}}),
    ("Продажи MELVA в Ниве", {"f": {"brand": "MELLO [MELVA]", "network": NIVA}}),
    ("Продажи Palitra в Ниве", {"f": {"brand": "ПАЛИТР [PALITRA]", "network": NIVA}}),
    ("Продажи Tresille", {"f": {"brand": "ТРЕСИЛЬ [TRESILLE]"}}),
    ("Продажи Синяя цена в Семерочке", {"f": {"brand": "СИНЯЯ ЦЕНА (PL)", "network": "Семерочка Х7"}}),
    ("Продажи HILL & SHORE в Граните", {"f": {"brand": "HILL&SHORE", "network": GRANIT}}),
    ("Продажи Хилл энд Шор в Граните", {"f": {"brand": "HILL&SHORE", "network": GRANIT}}),
    ("Как продается Олвиа в Ажуре?", {"f": {"brand": "OLVIA", "network": AZHUR}}),
    ("Продажи Вердант, Спума и Эстра", {"f": {"brand": ["VERDANT", "SPUMA", "ESTRA"]},
                                         "note": "есть также: ESTRA PROFESSIONAL"}),
    ("Продажи Юнивера в Ниве", {"f": {"manufacturer": "UNIVERA", "network": NIVA}}),
    ("Продажи P&G и Parker and Grant", {"f": {"manufacturer": "PARKER & GRANT"}}),
    ("Продажи Фрут Драгонс в Граните", {"f": {"brand": "FRUIT DRAGONS", "network": GRANIT}}),
    ("Динамика продаж Лесной тропы в Семерочке", {"f": {"brand": "ЛЕСНАЯ ТРОПА", "network": "Семерочка Х7"}}),
    ("Продажи Forest Path в Семерочке", {"f": {"brand": "ЛЕСНАЯ ТРОПА", "network": "Семерочка Х7"}}),
    ("Продажи шампуней Королевны в Граните", {"f": {"product_type": "Шампунь", "brand": "КОРОЛЕВНА", "network": GRANIT}}),
    ("Продажи Гаммы в Граните", {"f": {"brand": "ГАММА", "network": GRANIT}}),
    ("Продажи Vector в Граните", {"f": {"brand": "VECTOR", "network": GRANIT}, "note": "VECTOR FUSION"}),
    ("Продажи бренда Atelier в Маяке", {"f": {"brand": "ATELIER PROFESSIONAL", "network": GRANIT}}),
    ("Продажи брендов NORD: Vector, Atelier, Гамма",
     {"f": {**NORD, "brand": ["VECTOR", "ATELIER PROFESSIONAL", "ГАММА"]}}),
    ("Продажи VECTOR PROWAY в Граните", {"f": {"brand": "VECTOR PROWAY", "network": GRANIT}}),
    ("Продажи СЕКРЕТЫ БАБУШКИ МАРФЫ", {"f": {"brand": "СЕКРЕТЫ БАБУШКИ МАРФЫ"}}),
    ("Продажи OLVIA MEN в Ниве", {"f": {"brand": "OLVIA", "network": NIVA}}),
    ("Продажи VERDANT VITALIS в Ниве", {"f": {"brand": "VERDANT", "network": NIVA}}),
    ("Продажи бренда ZU в Ниве", {"f": {"brand": "ZU", "network": NIVA}}),
    ("Продажи 365 ДНЕЙ (PL) в Ниве", {"f": {"brand": "365 ДНЕЙ (PL)", "network": NIVA}}),
    ("Продажи за 365 дней", {}),
    ("Продажи HOME в Ниве", {"f": {"brand": "HOME", "network": NIVA}}),
    # ——— brands that are common words (L2-7, L4-5) ———
    ("ABC-анализ SKU NORD в Граните", {"f": {**NORD, "network": GRANIT}}),
    ("Сделай ABC анализ брендов шампуней", {"f": {"product_type": "Шампунь"}}),
    ("XYZ анализ брендов", {}),
    ("Кто лидер в шампунях в Ниве?", {"f": {"product_type": "Шампунь", "network": NIVA}}),
    ("А еще покажи топ брендов в Ажуре", {"f": {"network": AZHUR}, "fu": True, "lvl": "brand"}),
    ("Просто дай топ-5 брендов шампуней", {"f": {"product_type": "Шампунь"}, "lvl": "brand"}),
    ("Точно ли NORD падает в Граните?", {"f": {**NORD, "network": GRANIT}}),
    ("Какой эффект дало сокращение точек у NORD?", {"f": NORD}),
    ("Есть ли новинка у NORD", {"f": NORD}),
    ("Какая основа роста PAPYRA", {"f": {"brand": "PAPYRA"}}),
    ("Продажи шампуней для всей семьи", {"f": {"product_type": "Шампунь"}}),
    ("Продажи тоника для лица в Ниве", {"f": {"product_type": "Тоник для лица", "network": NIVA}}),
    ("Продажи бренда «Лидер» в Ниве", {"f": {"brand": "ЛИДЕР", "network": NIVA}}),
    ("Какая динамика у бренда Весна?", {"f": {"brand": "ВЕСНА"}}),
    # ——— private label (СТМ): brand (PL) or manufacturer STM(...), not a chain (L2-18) ———
    ("Какая доля СТМ Наша Цена в Граните?", {"f": {"brand": "НАША ЦЕНА (PL)", "network": GRANIT}}),
    ("Продажи бренда НАША ЦЕНА в Маяке", {"f": {"brand": "НАША ЦЕНА (PL)", "network": GRANIT}}),
    ("Доля STM X7 в Семерочке", {"f": {"manufacturer": "STM(X7)", "network": "Семерочка Х7"}}),
    ("Доля СТМ Х7 в Семерочке", {"f": {"manufacturer": "STM(X7)", "network": "Семерочка Х7"}}),
    ("Продажи МАЯК(PL) в Граните", {"f": {"brand": "МАЯК(PL)", "network": GRANIT}}),
    ("Продажи бренда Маяк в Маяке", {"f": {"brand": "МАЯК(PL)", "network": GRANIT}}),
    ("Доля СТМ Маяка в Граните", {"f": {"manufacturer": "STM(GRANIT)", "network": GRANIT}}),
    # ——— geography: region before city, grammatical cases, short city names (L2-8, L2-11, L2-12) ———
    ("Продажи NORD в Ростовской области", {"f": {**NORD, "region": "Ростовская обл"}}),
    ("Продажи NORD в Ростове-на-Дону", {"f": {**NORD, "city": "Ростов-на-Дону г"}}),
    ("Продажи NORD в Ростове", {"f": {**NORD, "city": "Ростов-на-Дону г"}}),
    ("Продажи NORD в Ростове Великом", {"f": {**NORD, "city": "Ростов г"}}),
    ("Продажи NORD в Краснодарском крае", {"f": {**NORD, "region": "Краснодарский край"}}),
    ("Продажи NORD в Новосибирской области", {"f": {**NORD, "region": "Новосибирская обл"}}),
    ("Продажи NORD в Приволжском федеральном округе", {"f": {**NORD, "district": "ПФО"}}),
    ("Продажи NORD в Центральном ФО", {"f": {**NORD, "district": "ЦФО"}}),
    ("Продажи NORD в ЦФО и ПФО", {"f": {**NORD, "district": ["ЦФО", "ПФО"]}}),
    ("Продажи NORD в Москве", {"f": {**NORD, "region": "Москва г"}}),
    ("Продажи NORD в СПб и Мск", {"f": {**NORD, "region": ["Санкт-Петербург г", "Москва г"]}}),
    ("Продажи NORD в Питере", {"f": {**NORD, "region": "Санкт-Петербург г"}}),
    ("Продажи NORD в Санкт-Петербурге и Екатеринбурге",
     {"f": {**NORD, "region": "Санкт-Петербург г", "city": "Екатеринбург г"}}),
    ("Продажи NORD в Москве и Московской области", {"f": {**NORD, "region": ["Москва г", "Московская обл"]}}),
    ("Продажи NORD в Самаре, Казани и Перми", {"f": {**NORD, "city": ["Самара г", "Казань г", "Пермь г"]}}),
    ("Продажи NORD в Уфе и Туле", {"f": {**NORD, "city": ["Уфа г", "Тула г"]}}),
    ("Продажи NORD в Орле", {"f": {**NORD, "city": "Орел г"}}),
    ("Продажи NORD в Нижнем Новгороде", {"f": {**NORD, "city": "Нижний Новгород г"}}),
    ("Продажи NORD в Новосибирске", {"f": {**NORD, "city": "Новосибирск г"}}),
    ("Продажи NORD в Башкирии", {"f": {**NORD, "region": "Башкортостан"}}),
    ("Продажи в Республика Башкортостан", {"f": {"region": "Башкортостан"}}),
    ("Продажи в Татарстане", {"f": {"region": "Татарстан"}}),
    ("Продажи NORD на Кубани", {"f": {**NORD, "region": "Краснодарский край"}}),
    ("Продажи NORD на Урале", {"f": {**NORD, "district": "УФО"}}),
    ("Продажи NORD на Дальнем Востоке", {"f": {**NORD, "district": "ДФО"}}),
    ("Продажи NORD в Пермском крае", {"f": {**NORD, "region": "Пермский край"}}),
    ("Продажи NORD в г. Бор", {"f": {**NORD, "city": "Бор г"}}),
    ("Продажи NORD в Чите", {"f": {**NORD, "city": "Чита г"}}),
    ("Как NORD борется за долю в Граните?", {"f": {**NORD, "network": GRANIT}}),
    ("В первую очередь покажи топ брендов", {"lvl": "brand"}),
    ("Какой режим расчета у динамики?", {}),
    ("Как читать этот отчет по NORD?", {"f": NORD}),
    ("Какие любимые бренды покупателей Нивы?", {"f": {"network": NIVA}, "lvl": "brand"}),
    ("Мирный рост продаж NORD", {"f": NORD}),
    ("Продажи NORD в Советском районе", {"f": NORD}),
    ("Продажи в пластиковой упаковке", {}),
    # ——— products (L2-20) ———
    ("Продажи кондиционеров для белья в Ажуре", {"f": {"network": AZHUR}, "note": "нет в данных"}),
    ("Топ-5 брендов салфетки в Семерочке", {"f": {"product_type": ["Салфетки влажные", "Салфетки бумажные"],
                                                   "network": "Семерочка Х7"}, "lvl": "brand"}),
    ("Продажи салфетки влажные", {"f": {"product_type": "Салфетки влажные"}}),
    ("Продажи влажных салфеток в Семерочке", {"f": {"product_type": "Салфетки влажные", "network": "Семерочка Х7"}}),
    ("Продажи набор косметики", {"f": {"product_type": "Набор косметики"}}),
    ("Продажи красок для волос NORD в Маяке", {"f": {"subgroup": COLOR, **NORD, "network": GRANIT}}),
    ("Продажи краски для волос NORD в Маяке", {"f": {"subgroup": COLOR, **NORD, "network": GRANIT}}),
    ("Продажи лака для волос", {"f": {"product_type": "Лак для волос"}}),
    ("Продажи масок для лица", {"f": {"product_type": "Маски для лица"}}),
    ("Продажи крема для рук", {"f": {"product_type": "Крем для рук"}}),
    ("Продажи туалетной бумаги", {"f": {"product_type": "Туалетная бумага"}}),
    ("Продажи мицеллярной воды", {"f": {"product_type": "Мицеллярная вода"}}),
    ("Продажи антиперспирантов", {"f": {"subgroup": "Дезодоранты"}}),
    ("Продажи сухого шампуня, кондиционера для волос и геля для душа",
     {"f": {"product_type": ["Сухой шампунь", "Бальзам для волос", "Гель для душа"]}}),
    ("Продажи жидкого, кускового и хозяйственного мыла", {"f": {"product_type": SOAPS}}),
    ("Продажи дезодорантов и солнцезащитных средств",
     {"f": {"subgroup": ["Дезодоранты", "Солнцезащитные средства и уход"]}}),
    ("Продажи детской гигиены волос", {"f": {"subgroup": KIDS_HAIR}, "kids": True}),
    ("Продажи оттеночных бальзамов Гамма в Граните за 2025 год",
     {"f": {"product_type": "Оттеночные средства для волос", "brand": "ГАММА", "network": GRANIT},
      "p": "2025-01..2025-12"}),
    # ——— measures (L2-22) ———
    ("Продажи NORD в штуках и в рублях", {"f": NORD, "ms": ["qty", "rub"]}),
    ("Продажи NORD в денежном выражении", {"f": NORD, "m": "rub"}),
    ("Продажи NORD в натуральном выражении", {"f": NORD, "m": "qty"}),
    ("Средняя цена за штуку шампуней NORD", {"f": {**NORD, "product_type": "Шампунь"}}),
    ("Продажи в литрах и тоннах", {"m": "kg"}),
    ("Выручка NORD в деньгах", {"f": NORD, "m": "rub"}),
    ("Продажи NORD в упаковках", {"f": NORD, "m": "qty"}),
    # ——— sorting, comparison target, breakdown, follow-up, role (L2-9, L2-21) ———
    ("Какие бренды растут быстрее всех в Ниве?", {"f": {"network": NIVA}, "sort": "growth", "lvl": "brand"}),
    ("Топ брендов по приросту", {"sort": "gain", "lvl": "brand"}),
    ("Какие производители падают сильнее всего?", {"sort": "decline", "lvl": "manufacturer"}),
    ("Топ брендов по изменению доли", {"sort": "share_change", "lvl": "brand"}),
    ("Топ-10 растущих брендов шампуней", {"f": {"product_type": "Шампунь"}, "sort": "growth", "lvl": "brand"}),
    ("Падающие бренды в Граните", {"f": {"network": GRANIT}, "sort": "decline", "lvl": "brand"}),
    ("Какие бренды теряют долю?", {"sort": "share_change", "lvl": "brand"}),
    ("Назови соседей NORD по рейтингу в шампунях",
     {"f": {"product_type": "Шампунь"}, "focus": "NORD", "lvl": "manufacturer"}),
    ("Кто рядом с ESTRA в Ажуре?", {"f": {"network": AZHUR}, "focus": "ESTRA", "lvl": "brand"}),
    ("Бренды на уровне PAPYRA", {"focus": "PAPYRA", "lvl": "brand"}),
    ("Доля NORD по сетям", {"f": NORD, "lvl": "network", "dims": {"network"}}),
    ("Какая доля Маяк Косметик в Граните?",
     {"f": {"network": GRANIT}, "lvl": "banner", "ment": {"banner": ["Маяк Косметик"]}}),
    ("Доля NORD в Граните по баннерам", {"f": {**NORD, "network": GRANIT}, "lvl": "banner"}),
    ("Какие сети растут быстрее Гранита?", {"lvl": "network", "ment": {"network": [GRANIT]}}),
    ("А что по этим брендам в Ниве?", {"f": {"network": NIVA}, "fu": True, "lvl": "brand"}),
    ("И в штуках?", {"m": "qty", "fu": True}),
    ("Мы — NORD, покажи рынок шампуней", {"f": {"product_type": "Шампунь"}, "role": "NORD"}),
    ("Я КАМ NORD, покажи продажи NORD в Ниве", {"f": {**NORD, "network": NIVA}, "role": "NORD"}),
    ("Я вижу, что NORD падает", {"f": NORD}),
    ("Я работаю в NORD, покажи рынок", {"role": "NORD"}),
    ("Топ-10 SKU NORD", {"f": NORD, "lvl": "sku"}),
    ("Продажи шампуней по городам", {"f": {"product_type": "Шампунь"}, "lvl": "city"}),
    # ——— no entities: nothing extra ———
    ("Почему падает средний чек?", {}),
    ("Объясни методологию расчета роста", {}),
    ("Сколько точек продаж в данных?", {}),
    ("Это верно?", {}),
    ("Хорошо, спасибо", {}),
    ("Мы — производитель, покажи конкурентов", {}),
]

# eval/questions.json questions (field q and turns) with expected parsing results
EVAL = {
    "G1": {"f": NORD, "p": "2026-01..2026-08", "b": "2025-01..2025-08", "portal": True, "lvl": "network",
           "ment": {"network": [GRANIT]}},
    "G2": {"f": NORD, "p": "2026-01..2026-08", "portal": True},
    "G3": {"f": {**NORD, "network": GRANIT}, "p": "2026-01..2026-08", "b": "2025-01..2025-08", "portal": True,
           "lvl": None},
    "G4": {"f": NORD, "p": "2026-01..2026-08", "b": "2025-01..2025-08", "portal": True},
    "M1": {"f": {"product_type": "Шампунь", "manufacturer": ["NORD", "AURIGA"], "network": [NIVA, GRANIT]},
           "p": "2026-09..2026-09", "kids": True},
    "M2": {"f": {"product_type": "Бальзам для волос", "network": AZHUR}, "p": "2026-09..2026-09", "m": "qty",
           "focus": "NORD", "lvl": "manufacturer"},
    "D1": {},
    "D2": {"m": "rub", "portal": True, "lvl": "brand"},
    "D3": {"m": "rub", "portal": True},
    "D4": {"f": {"subgroup": "Солнцезащитные средства и уход"}},
    "D5": {},
    "D6": {},
    "D7": {"f": NORD, "p": "2026-01..2026-06", "b": "2025-01..2025-06", "portal": True},
    "D8": {"f": NORD, "m": "rub", "lvl": "brand"},
    "D9": {"f": {"network": GRANIT, "product_type": SOAPS}, "p": "2026-06..2026-06", "b": "2026-01..2026-01",
           "m": "qty", "role": "NORD"},
    "D10": {"f": {"network": "Перелесок ТД", "subgroup": KIDS_HAIR}, "p": ps("2025-01", LAST), "m": "qty",
            "kids": True, "lvl": "sku"},
    "D11": {"f": {"network": "Перелесок ТД", "subgroup": KIDS_HAIR, "format": "СМ"}, "kids": True, "lvl": "sku"},
    "D12": {"f": {"product_type": "Шампунь", "brand": "КОРОЛЕВНА", "network": GRANIT}, "p": ps("2025-01", LAST),
            "m": "qty", "fut": "2026-10"},
    "D13": {},
    "D14": {"f": NORD, "lvl": "network", "dims": {"network", "brand"}},
    "N1": {"f": {"product_type": "Шампунь", "brand": "ESTRA", "banner": "Маяк Косметик"}, "p": "2026-08..2026-08"},
    "N2": {"f": {"product_type": "Шампунь", "brand": "ESTRA", "banner": "Маяк Косметик"}, "p": "2026-01..2026-09",
           "b": "2025-01..2025-09", "m": "qty"},
    "N3": {"f": {"product_type": "Шампунь", "brand": "ESTRA", "banner": "Маяк Косметик"}, "fu": True, "lvl": "sku"},
    "N4": {"f": {"product_type": "Шампунь", "brand": "ESTRA", "banner": "Маяк Косметик"}, "p": "mat", "m": "qty"},
    "N5": {"m": "rub", "lvl": "brand"},
    "N6": {"p": Y26, "m": "rub", "sort": "growth", "lvl": "brand"},
    "N7": {"f": {"network": GRANIT}, "m": "qty", "sort": "growth", "lvl": "brand"},
    "N8": {"f": {"network": GRANIT}, "p": "2026-01..2026-09", "m": "qty", "lvl": "sku"},
    "N9": {"lvl": "brand"},
    "N10": {"f": {"brand": "L'AURELLE PARIS"}},
    "N11": {"f": {"product_type": "Бальзам для волос", "network": AZHUR}, "p": "2026-09..2026-09", "m": "qty",
            "focus": "NORD", "lvl": "manufacturer"},
    "N12": {"f": NORD, "p": "2026-09..2026-09", "b": "2025-09..2025-09"},
    "N13": {"f": NORD, "lvl": "brand"},
    "N14": {"f": {"network": GRANIT}, "p": "2026-01..2026-09", "m": "qty", "sort": "growth"},
    "R1": {"f": NORD, "p": "2026-09..2026-09", "b": "2025-09..2025-09"},
    "R2": {"f": {"brand": "GREEN SHOP", "network": NIVA}, "p": "2025-01..2025-12"},
    "R3": {"f": NORD, "ex": {"network": GRANIT}, "portal": True},
    "R4": {"f": {**NORD, "network": AZHUR}},
    "R5": {"f": {"product_type": "Шампунь", "network": "Детский Остров (сеть)"}, "lvl": "brand"},
}


def _eval_cases():
    qs = json.loads((Path(__file__).parent.parent / "eval" / "questions.json").read_text(encoding="utf-8"))
    return [(x.get("q") or x["turns"][-1], EVAL[x["id"]]) for x in qs if x["id"] in EVAL]


def _lst(v):
    return sorted(E.de_yo(x) for x in (v if isinstance(v, (list, tuple)) else [v]))


def _dict(d):
    return {k: _lst(v) for k, v in (d or {}).items()}


def check(q, exp):
    f = D.find(q)
    a = f.as_args()
    msg = f"{q!r} -> {f.summary()} | {a}"
    assert _dict(f.filters) == _dict(exp.get("f")), msg
    assert _dict(f.exclude) == _dict(exp.get("ex")), msg
    assert a.get("period") == exp.get("p"), msg
    assert a.get("base_period") == exp.get("b"), msg
    assert f.measure == exp.get("m"), msg
    assert f.measures == exp.get("ms", []), msg
    assert f.kids == exp.get("kids"), msg
    assert f.exclude_pl == exp.get("pl", False), msg
    assert f.sort == exp.get("sort"), msg
    assert f.focus == exp.get("focus"), msg
    assert f.followup == exp.get("fu", False), msg
    assert f.out_of_range == exp.get("out", []), msg
    assert (S.ym(f.future_month) if f.future_month else None) == exp.get("fut"), msg
    assert f.portal == exp.get("portal", False), msg
    assert f.role == exp.get("role"), msg
    if "lvl" in exp:
        assert f.level_hint == exp["lvl"], msg
    if "dims" in exp:
        assert f.dims == exp["dims"], msg
    if "ment" in exp:
        assert _dict(f.mentioned) == _dict(exp["ment"]), msg
    if "both" in exp:
        assert f.kids_both == exp["both"], msg
    if "note" in exp:
        assert exp["note"] in f.summary(), msg
    for k, vals in f.exclude.items():                     # excluded values and comparison target are not filters
        assert not set(vals) & set(f.filters.get(k, [])), msg
    if f.focus:
        assert all(f.focus not in v for v in f.filters.values()), msg


@pytest.mark.parametrize("q,exp", CASES, ids=[c[0][:60] for c in CASES])
def test_find(q, exp):
    check(q, exp)


@pytest.mark.parametrize("q,exp", _eval_cases(), ids=lambda x: x[:60] if isinstance(x, str) else "")
def test_eval_questions(q, exp):
    check(q, exp)


def test_eval_covered():
    """Every eval question is in the expectations table."""
    qs = json.loads((Path(__file__).parent.parent / "eval" / "questions.json").read_text(encoding="utf-8"))
    assert {x["id"] for x in qs} <= set(EVAL)


def test_aliases_exist():
    """Russian spellings and aliases map only to real brands and manufacturers (checked against sku)."""
    brands = {r[0] for r in S.con().execute("SELECT DISTINCT brand FROM sku").fetchall()}
    mans = {r[0] for r in S.con().execute("SELECT DISTINCT manufacturer FROM sku").fetchall()}
    assert set(E.BRAND_ALIASES) <= brands, set(E.BRAND_ALIASES) - brands
    assert set(E.MANUFACTURER_ALIASES) <= mans, set(E.MANUFACTURER_ALIASES) - mans


def test_top_brands_by_name():
    """«бренд X» finds brand X itself for the 400 largest brands (long key, brackets, (PL), grammatical cases)."""
    top = [b for b, _, _ in sorted(D.brands, key=lambda x: -x[2]) if E.norm(b).strip()][:400]
    bad = [b for b in top if b not in D.find(f"Продажи бренда {b} в Ниве").filters.get("brand", [])]
    assert not bad, bad[:10]


def test_parse_periods_api():
    """Legacy parse_periods API: (periods in order of mention, kind, forecast month)."""
    ps_, kind, fut = E.parse_periods("январь–август 2026 к январю–августу 2025")
    assert [p.to_str() for p in ps_] == ["2026-01..2026-08", "2025-01..2025-08"] and kind is None and fut is None


def test_speed():
    """The dictionary is built once; parsing a question takes milliseconds."""
    assert E.dictionary() is D
    t = time.perf_counter()
    for q, _ in CASES:
        D.find(q)
    assert (time.perf_counter() - t) / len(CASES) < 0.02
