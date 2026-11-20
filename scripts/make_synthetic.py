"""
Synthetic retail sell-out panel.

The production agent ran on a licensed third-party panel of store-level sales in
Russian retail chains (hundreds of millions of rows). That data cannot be published, so this script
generates a panel with the same schema, the same entity structure and — most
importantly — the same *data defects* that the agent's methodology layer handles:

* chains with partial coverage (a few months only, gaps, a chain that stops reporting);
* incomplete last month for two chains (one loaded a fraction of stores, one a fraction of each store);
* one physical store registered under several `store_id`s (three chains, different periods);
* a chain-level banner that is only inferable from a format code, plus a growing "other" bucket;
* ~40% of stores without a city (always empty for federal cities);
* a brand labelled "." with an unknown manufacturer;
* a cash-and-carry chain with zero purchase cost from 2025 on;
* private labels (manufacturer `STM(<chain>)`, brand `... (PL)`), sold only in their own chain;
* one brand under two labels (Latin and Cyrillic) and brands that look like common words;
* SKU life cycles: a brand line delisted at the end of 2025, a brand launched mid-2025;
* a per-chain core assortment (top SKUs of each subgroup) sold in most larger stores;
* a kids chain that stocks adult products only from family-care subgroups.

All chains, manufacturers, brands and SKUs are fictional. Geography and the
product taxonomy are generic. One seed drives everything.

Output: data/parquet/dim/{dim_sku,dim_store}.parquet and data/parquet/fact/month=YYYY-MM/part.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "parquet"
MONTHS = pd.period_range("2024-01", "2026-09", freq="M")

# --------------------------------------------------------------------------- #
# Chains: name, client_id, stores, formats (share), months active, price level, cost/revenue by year
# --------------------------------------------------------------------------- #
CHAINS = [
    # name,                  cid, stores, formats,                                 active,                 price, cost ratio 2024/25/26
    ("Гранит",                 1, 620, {"ММ": .80, "СМ": .05, "ГМ": .03, "Интернет Магазин": .0, "РЦ": .0}, None, 1.00, (.86, .95, .98)),
    ("Семерочка Х7",           5, 500, {"ММ": .95, "СМ": .05},                       None,                   1.02, (.87, .89, .92)),
    ("Нива (сеть)",            7,  40, {"ГМ": .55, "СМ": .45},                       None,                   0.97, (.95, .91, .91)),
    ("Перелесок ТД",           4,  36, {"СМ": .9, "Интернет Магазин": .1},           None,                   1.10, (.86, .87, .90)),
    ("Ажур",                   9,  10, {"ГМ": 1.0},                                  None,                   0.98, (.91, .93, .94)),
    ("Депо Кэш Энд Керри",    11,  10, {"ГМ": 1.0},                                  None,                   0.94, (1.05, 0.0, 0.0)),
    ("ЛАД (сеть)",            19,   8, {"ГМ": .7, "СМ": .3},                         None,                   1.00, (.95, .95, 1.01)),
    ("Гиперкосмос",           21,   4, {"ГМ": 1.0},                                  None,                   0.99, (.85, .85, .85)),
    ("Детский Остров (сеть)", 23,  50, {"СМ": 1.0},                                  None,                   1.08, (.84, .73, .71)),
    ("Запас",                 31,   2, {"Интернет Магазин": 1.0},                    ("2024-01", "2026-02"), 0.96, (.96, .96, 1.07)),
    ("Добрый (сеть)",         33,  40, {"ММ": 1.0}, ["2024-03", "2024-04", "2024-06", "2024-09", "2024-11", "2025-01", "2025-02",
                                                     "2025-04", "2025-07", "2025-08", "2025-10", "2025-12", "2026-02", "2026-03"], 0.93, (.84, .83, .81)),
    ("Маятник (сеть)",        35,  30, {"ММ": 1.0}, ["2025-04", "2025-05", "2025-08"], 0.90, (.83, .83, .83)),
    ("Стрижик",               37,  25, {"ММ": 1.0}, ["2025-05", "2025-06"],         0.92, (.83, .83, .83)),
    ("Камелия (сеть)",        39,  10, {"ММ": 1.0}, ("2024-03", "2024-12"),         1.00, (.57, .57, .57)),
    ("ПАРУС (сеть)",          41,  10, {"СМ": 1.0}, ("2024-03", "2024-12"),         1.00, (.81, .81, .81)),
    ("НУ (сеть)",             43,   6, {"ММ": 1.0}, ("2024-02", "2024-04"),         1.00, (.88, .88, .88)),
    ("Офис Х7",                4,   1, {"РЦ": 1.0}, ("2024-01", "2026-08"),         1.00, (.90, .90, .90)),
]
# Banner of the largest chain is encoded only in the chain's own format code (format_src).
GRANIT_BANNERS = {"МД": .80, "МК": .10, "МА": .03, "Наша Цена": .04, "ГМ": .03}
STM_CODE = {"Гранит": "GRANIT", "Семерочка Х7": "X7", "Перелесок ТД": "X7", "Нива (сеть)": "NIVA", "Ажур": "AZHUR",
            "Депо Кэш Энд Керри": "DEPO", "ЛАД (сеть)": "LAD", "Гиперкосмос": "KOSMOS", "Детский Остров (сеть)": "DETSKIY OSTROV",
            "Добрый (сеть)": "DOBRIY", "Камелия (сеть)": "KAMELIA", "ПАРУС (сеть)": "PARUS"}
FORMAT_SIZE = {"ММ": 1.0, "СМ": 2.6, "ГМ": 6.0, "Интернет Магазин": 3.0, "РЦ": 0.05}

# --------------------------------------------------------------------------- #
# Geography (public): district, region, cities (city "" = federal city stored without a city)
# --------------------------------------------------------------------------- #
GEO = [
    ("ЦФО", "Москва г", [""], 14), ("ЦФО", "Московская обл", ["Химки г", "Подольск г", "Балашиха г", "Мытищи г"], 9),
    ("ЦФО", "Тульская обл", ["Тула г", "Новомосковск г"], 2), ("ЦФО", "Воронежская обл", ["Воронеж г"], 2.5),
    ("ЦФО", "Ярославская обл", ["Ярославль г", "Ростов г"], 1.5), ("ЦФО", "Орловская обл", ["Орел г"], 1),
    ("СЗФО", "Санкт-Петербург г", [""], 7), ("СЗФО", "Ленинградская обл", ["Гатчина г", "Всеволожск г"], 2),
    ("СЗФО", "Карелия", ["Петрозаводск г"], 0.8),
    ("ПФО", "Татарстан", ["Казань г", "Набережные Челны г"], 4), ("ПФО", "Башкортостан", ["Уфа г", "Стерлитамак г"], 3.5),
    ("ПФО", "Самарская обл", ["Самара г", "Тольятти г"], 3), ("ПФО", "Нижегородская обл", ["Нижний Новгород г", "Бор г"], 3),
    ("ПФО", "Пермский край", ["Пермь г"], 2.2),
    ("ЮФО", "Краснодарский край", ["Краснодар г", "Сочи г", "Новороссийск г"], 6), ("ЮФО", "Ростовская обл", ["Ростов-на-Дону г", "Таганрог г"], 4),
    ("ЮФО", "Волгоградская обл", ["Волгоград г"], 2),
    ("УФО", "Свердловская обл", ["Екатеринбург г", "Нижний Тагил г"], 3.5), ("УФО", "Челябинская обл", ["Челябинск г", "Магнитогорск г"], 3),
    ("УФО", "Ханты-Мансийский АО", ["Сургут г"], 1.2),
    ("СФО", "Новосибирская обл", ["Новосибирск г"], 2.5), ("СФО", "Красноярский край", ["Красноярск г"], 2),
    ("СФО", "Омская обл", ["Омск г"], 1.5),
    ("ДФО", "Приморский край", ["Владивосток г"], 1), ("ДФО", "Забайкальский край", ["Чита г"], 0.6),
    ("ДФО", "Саха (Якутия)", ["Якутск г"], 0.4),
    ("СКФО", "Ставропольский край", ["Ставрополь г"], 1.5), ("СКФО", "РСО-Алания", ["Владикавказ г"], 0.5),
    ("КФО", "Севастополь г", [""], 0.3), ("КФО", "Крым Респ", ["Симферополь г"], 0.8),
]

# --------------------------------------------------------------------------- #
# Product taxonomy: (group, subgroup, weight, [product types], seasonal peak month or None)
# --------------------------------------------------------------------------- #
TAXONOMY = [
    ("^Лицо/Тело/Волосы", "^Гигиена волос и кожи головы", 13, ["Шампунь", "Бальзам для волос", "Сухой шампунь", "Шампунь для бороды"], None),
    ("^Гигиена медицинская", "^Гигиена", 30, ["Туалетная бумага", "Салфетки бумажные", "Подгузники детские", "Прокладки"], None),
    ("^Лицо/Тело/Волосы", "^Дезодоранты", 9, ["Дезодорант", "Антиперспирант"], 7),
    ("^Лицо/Тело/Волосы", "^Гигиена тела", 6.5, ["Гель для душа", "Пена для ванн", "Скраб для тела"], None),
    ("^Лицо/Тело/Волосы", "^Уход за кожей лица", 4, ["Крем для лица", "Маски для лица", "Сыворотка для лица", "Уход за губами"], None),
    ("^Лицо/Тело/Волосы", "^Уход за волосами и кожей головы", 5, ["Маски для волос", "Масло для волос", "Сыворотка для волос"], None),
    ("^Лицо/Тело/Волосы", "^Средства для окраски волос", 5, ["Краска для волос", "Оттеночные средства для волос", "Осветлитель"], None),
    ("^Лицо/Тело/Волосы", "^Мыло", 5, ["Мыло Жидкое", "Мыло кусковое", "Мыло хозяйственное"], None),
    ("^Лицо/Тело/Волосы", "^Очищение кожи лица", 2.8, ["Мицеллярная вода", "Тоник для лица", "Пенка для умывания"], None),
    ("^Гигиена медицинская", "^Гигиена с пропиткой", 6.5, ["Салфетки влажные"], 7),
    ("^Лицо/Тело/Волосы", "^Средства для укладки волос", 2.5, ["Лак для волос", "Гель для укладки", "Мусс для волос"], None),
    ("^Лицо/Тело/Волосы", "^Уход за кожей рук и ног", 1.1, ["Крем для рук", "Крем для ног"], 1),
    ("^Гигиена рта", "^Гигиена полости рта Детская", 1.3, ["Зубная паста", "Зубная щетка"], None),
    ("^Лицо/Тело/Волосы", "^Средства для ухода за телом", 1.0, ["Крем для тела", "Лосьон для тела"], None),
    ("^Лицо/Тело/Волосы", "^Гигиена тела Детская", 0.8, ["Гель для душа", "Пена для ванн"], None),
    ("^Лицо/Тело/Волосы", "^Мыло Детское", 0.7, ["Мыло Жидкое", "Мыло кусковое"], None),
    ("^Лицо/Тело/Волосы", "^Гигиена волос и кожи головы Детская", 0.5, ["Шампунь", "Бальзам для волос"], None),
    ("^Лицо/Тело/Волосы", "^Средства для ухода за телом Детские", 0.3, ["Крем для тела"], None),
    ("^Наборы", "^Набор косметический универсальный", 0.3, ["Набор косметики", "Подарочный набор"], 12),
    ("^Лицо/Тело/Волосы", "^Солнцезащитные средства и уход", 0.4, ["Солнцезащитный крем", "Средство после загара"], 6),
    ("^Декоративная косметика", "^Декоративная косметика для глаз и бровей", 0.12, ["Тушь для ресниц", "Карандаш для бровей"], None),
    ("^Декоративная косметика", "^Декоративная косметика для ногтей", 0.07, ["Лак для ногтей"], None),
    ("^Лицо/Тело/Волосы", "^Солнцезащитные средства и уход Детский", 0.05, ["Солнцезащитный крем"], 6),
    ("^Наборы", "^Набор косметический универсальный Детский", 0.03, ["Набор косметики"], 12),
    ("^Парфюмерия", "^Парфюмерия", 0.1, ["Туалетная вода", "Парфюмерная вода"], 12),
    ("^Декоративная косметика", "^Декоративная косметика для лица", 0.01, ["Пудра"], None),
    ("^Промтовары", "^Косметика_Прочее", 0.005, ["Прочее"], None),
    ("^Хозтовары", "^Хозтовары", 0.003, ["Тряпки/Салфетки"], None),
    ("", "", 0.001, ["Прочие"], None),
]
OWN_SUBGROUPS = {  # where the own company ("NORD") has products
    "^Средства для окраски волос": 30, "^Гигиена волос и кожи головы": 22, "^Гигиена тела": 9,
    "^Уход за волосами и кожей головы": 9, "^Декоративная косметика для глаз и бровей": 2,
    "^Набор косметический универсальный": 3, "^Уход за кожей рук и ног": 3, "^Дезодоранты": 4,
    "^Уход за кожей лица": 5, "^Мыло": 4, "^Средства для ухода за телом": 3, "^Средства для укладки волос": 4,
    "^Гигиена волос и кожи головы Детская": 1.5, "^Гигиена тела Детская": 1, "^Мыло Детское": 0.5,
}

# --------------------------------------------------------------------------- #
# Manufacturers and brands: manufacturer -> list of (brand, subgroups or None=any, strength[, options])
# options: life=(first, last) months the line is sold ("" = open end), weight = demand weight of the line
# instead of its subgroup's (a line that sells far above its small subgroup), types = product types of the line. The same brand may appear
# twice to give it a separate line.
# --------------------------------------------------------------------------- #
OWN = "NORD"
OWN_BRANDS = [
    ("ATELIER PROFESSIONAL", ["^Средства для окраски волос", "^Гигиена волос и кожи головы", "^Уход за волосами и кожей головы"], 3.0),
    ("КОРОЛЕВНА", ["^Гигиена волос и кожи головы Детская", "^Гигиена тела Детская", "^Мыло Детское", "^Набор косметический универсальный"], 1.6),
    ("ГАММА", ["^Средства для окраски волос"], 2.2), ("SILK", ["^Гигиена волос и кожи головы", "^Гигиена тела", "^Уход за кожей рук и ног"], 1.6),
    ("VECTOR ANTI-YELLOW", ["^Средства для окраски волос", "^Гигиена волос и кожи головы"], 1.2),
    ("VECTOR FUSION", ["^Гигиена волос и кожи головы", "^Уход за волосами и кожей головы"], 1.2),
    ("WILD GIRL", ["^Уход за кожей лица", "^Декоративная косметика для глаз и бровей", "^Гигиена тела"], 0.9),
    ("VECTOR", ["^Средства для окраски волос", "^Средства для укладки волос"], 1.0), ("ГОРКИ", ["^Набор косметический универсальный", "^Гигиена тела Детская"], 0.5),
    ("SEDUCTEUR", ["^Гигиена тела", "^Дезодоранты"], 0.6, {"life": ("2025-06", "")}),   # launched mid-2025
    ("AQUABERRY", ["^Мыло", "^Гигиена тела"], 0.6),
    ("LACTOVIA", ["^Уход за кожей лица", "^Средства для ухода за телом"], 0.4), ("MARVELOUS", ["^Уход за волосами и кожей головы"], 0.4),
    ("CERALUX", ["^Уход за кожей лица", "^Уход за кожей рук и ног"], 0.3), ("VECTOR PROWAY", ["^Средства для окраски волос"], 0.3),
    ("ПТИЧКА", ["^Мыло Детское", "^Гигиена волос и кожи головы Детская"], 0.3), ("МЕГАДРАКОНЫ", ["^Гигиена полости рта Детская", "^Гигиена тела Детская"], 0.2),
    ("MEGADRAGONS", ["^Гигиена тела Детская"], 0.1), ("COLORIA", ["^Средства для окраски волос"], 0.4), ("КОЛОРИЯ", ["^Средства для окраски волос"], 0.15),
    ("DERMALINA", ["^Уход за кожей лица", "^Средства для ухода за телом"], 0.3), ("DEO MOTION", ["^Дезодоранты"], 0.4),
    ("FRUIT DRAGONS", ["^Гигиена полости рта Детская", "^Гигиена тела Детская"], 0.8),
]
COMPETITORS = {
    "PARKER & GRANT": [("HILL&SHORE", ["^Гигиена волос и кожи головы"], 6), ("LUMINA", ["^Гигиена волос и кожи головы", "^Уход за волосами и кожей головы"], 4),
                       ("OLD HARBOR", ["^Дезодоранты", "^Гигиена тела"], 3), ("PUFFIES", ["^Гигиена"], 8), ("HERBAL SPIRIT", ["^Гигиена волос и кожи головы"], 1.5),
                       ("SECRET ROSE", ["^Дезодоранты"], 1.2)],
    "UNIVERA": [("MEADOWA", ["^Гигиена волос и кожи головы"], 1.5), ("CLARO", ["^Гигиена волос и кожи головы"], 3), ("DEFENDA", ["^Дезодоранты"], 4),
                ("ТРЕСИЛЬ [TRESILLE]", ["^Гигиена волос и кожи головы", "^Средства для укладки волос"], 2), ("ЛЕСНАЯ ТРОПА", ["^Уход за кожей лица", "^Гигиена тела", "^Уход за кожей рук и ног"], 3),
                ("СВОНЛИ [SWANLY]", ["^Гигиена тела", "^Мыло", "^Дезодоранты"], 5)],
    "HELMAN": [("SHINE KUR", ["^Гигиена волос и кожи головы", "^Уход за волосами и кожей головы"], 3), ("SYLVA", ["^Гигиена волос и кожи головы", "^Средства для окраски волос"], 2.5),
               ("FIXA", ["^Средства для укладки волос"], 3), ("SPUMA", ["^Гигиена волос и кожи головы"], 2), ("ПАЛИТР [PALITRA]", ["^Средства для окраски волос"], 3),
               ("ZU", ["^Гигиена тела", "^Дезодоранты"], 1.2)],
    "BERGSTAD": [("OLVIA", ["^Уход за кожей лица", "^Средства для ухода за телом", "^Дезодоранты", "^Солнцезащитные средства и уход", "^Уход за кожей рук и ног"], 6),
                 ("OLVIA SUN", ["^Солнцезащитные средства и уход"], 1)],
    "L'AURELLE": [("L'AURELLE PARIS", ["^Средства для окраски волос", "^Гигиена волос и кожи головы", "^Уход за кожей лица", "^Очищение кожи лица"], 6),
                  ("VERDANT", ["^Средства для окраски волос", "^Гигиена волос и кожи головы", "^Очищение кожи лица", "^Уход за кожей лица"], 5),
                  # delisted lines: the brand falls in 2026 although its remaining segments grow. Hair care ends with
                  # 2025 (dropped segments); decorative cosmetics end in 09.2025, so they are in the base of every
                  # comparison but not among the brand's categories of the last 12 months
                  ("L'AURELLE PARIS", ["^Уход за волосами и кожей головы"], 4, {"life": ("", "2025-12")}),
                  ("L'AURELLE PARIS", ["^Декоративная косметика для глаз и бровей"], 1.5, {"life": ("", "2025-09"), "weight": 6})],
    "KENDALL-CROSS LTD": [("CUDDLES", ["^Гигиена", "^Гигиена с пропиткой"], 6), ("KOTTA", ["^Гигиена"], 3)],
    "JAMESON & JAMESON": [("JAMESON BABY", ["^Гигиена тела Детская", "^Мыло Детское", "^Средства для ухода за телом Детские", "^Гигиена волос и кожи головы Детская"], 1.5), ("CLEARSKIN", ["^Очищение кожи лица"], 1)],
    "CORVIN-OLIVERA": [("COLDENT", ["^Гигиена полости рта Детская"], 4), ("OLIVERA", ["^Мыло", "^Гигиена тела"], 3), ("CAMEA", ["^Мыло"], 1)],
    "ESTRA LAB": [("ESTRA", ["^Средства для окраски волос", "^Гигиена волос и кожи головы", "^Уход за волосами и кожей головы", "^Средства для укладки волос"], 5),
                  ("ESTRA", ["^Гигиена волос и кожи головы"], 1.0, {"types": ["Шампунь"]}),   # shampoo line: sold in every chain
                  ("ESTRA PROFESSIONAL", ["^Средства для окраски волос", "^Уход за волосами и кожей головы"], 1.0)],
    "TERRA BOREALIS": [("TERRA BOREALIS", ["^Гигиена волос и кожи головы", "^Уход за кожей лица", "^Гигиена тела", "^Средства для ухода за телом"], 2),
                       ("ICE BY TERRA BOREALIS", ["^Гигиена волос и кожи головы", "^Уход за кожей лица"], 2), ("TERRA BOREALIS LAB", ["^Уход за кожей лица"], 1.5)],
    "MIXTA": [("MIXTA", ["^Уход за кожей лица", "^Гигиена тела", "^Уход за волосами и кожей головы"], 2)],
    "AURIGA": [("AURIGA", ["^Уход за кожей рук и ног"], 0.4), ("SUNVEIL", ["^Уход за кожей лица", "^Солнцезащитные средства и уход"], 1.5), ("FLORESAN", ["^Солнцезащитные средства и уход"], 1),
               ("АУРИКА МАЛЫШ", ["^Гигиена волос и кожи головы Детская", "^Гигиена тела Детская"], 1.2)],
    "SOLMARE": [("SOLMARE", ["^Солнцезащитные средства и уход", "^Уход за кожей рук и ног"], 1)],
    "FARMIX": [("FARMIX", ["^Средства для окраски волос", "^Уход за волосами и кожей головы"], 1.0), ("HAIRLAB", ["^Уход за волосами и кожей головы"], 1.5)],
    "SPLASH": [("SPLASH", ["^Гигиена полости рта Детская"], 2)],
    "PAPYRA GROUP": [("PAPYRA", ["^Гигиена"], 4), ("TOFFI", ["^Гигиена"], 2), ("MELLO [MELVA]", ["^Гигиена", "^Гигиена с пропиткой"], 3)],
    "BRISK": [("BRISK", ["^Хозтовары", "^Мыло"], 1)],
    "DERMOLIB": [("DERMOLIB", ["^Уход за кожей лица"], 1)], "OLLAR": [("OLLAR", ["^Средства для окраски волос", "^Уход за волосами и кожей головы"], 1)],
    "DEOLINA": [("DEOLINA", ["^Дезодоранты"], 1.5)], "CELINE MARAIS": [("CELINE MARAIS", ["^Декоративная косметика для глаз и бровей", "^Декоративная косметика для ногтей"], 1)],
    "LACTOFIN": [("LACTOFIN", ["^Гигиена тела"], 0.6)], "VELURA GROUP": [("VELURA", ["^Средства для окраски волос", "^Средства для укладки волос"], 2)],
    "GREEN": [("GREEN SHOP", ["^Гигиена волос и кожи головы", "^Гигиена тела", "^Уход за кожей лица"], 2), ("GREEN KITCHEN", ["^Гигиена волос и кожи головы", "^Гигиена тела"], 1.5)],
    "CLEVERLINE": [("DOLCE SILK", ["^Гигиена тела", "^Гигиена волос и кожи головы"], 1.2)],
    "ФАБРИКА ЛЕС": [("СЕКРЕТЫ БАБУШКИ МАРФЫ", ["^Гигиена волос и кожи головы", "^Уход за волосами и кожей головы", "^Гигиена тела"], 2.5),
                    ("СЕКРЕТЫ ЧИСТОТЫ", ["^Хозтовары", "^Мыло"], 0.8), ("ЛЕСНЫЕ МОТИВЫ", ["^Гигиена тела", "^Мыло"], 0.8),
                    ("ДОБРЫЕ ТРАДИЦИИ", ["^Мыло"], 0.5)],
    "AVANGARD": [("ЛИДЕР", ["^Гигиена", "^Мыло"], 1), ("ВЕСНА", ["^Мыло", "^Уход за кожей рук и ног"], 1.2), ("ПРЕЛЕСТЬ", ["^Средства для укладки волос"], 0.8),
                 ("САМАЯ", ["^Уход за кожей лица"], 0.3), ("МОЙ ЛУЧИК", ["^Гигиена тела Детская", "^Мыло Детское", "^Средства для ухода за телом Детские",
                                "^Гигиена волос и кожи головы Детская"], 0.8)],
    "SHENZHEN NORD TECHNOLOGY CO.": [("NORDAKU", ["^Уход за кожей лица"], 0.4)],
    "не указан": [(".", None, 3), ("HOME", ["^Гигиена"], 1.0), ("L", ["^Декоративная косметика для ногтей"], 0.05)],
}
PL = {  # private labels: chain -> brands
    "Гранит": ["СИНЯЯ ЦЕНА (PL)", "НАША ЦЕНА (PL)", "МАЯК(PL)"], "Семерочка Х7": ["КАЖДЫЙ ДЕНЬ (PL)", "SEMERKA HOME (PL)"],
    "Перелесок ТД": ["PERELES SELECT (PL)"], "Нива (сеть)": ["365 ДНЕЙ (PL)"], "Ажур": ["AZHUR (PL)"], "Депо Кэш Энд Керри": ["DEPO CHEF (PL)"],
    "ЛАД (сеть)": ["LAD (PL)"], "Детский Остров (сеть)": ["ОСТРОВОК (PL)"],
}
# the kids chain stocks adult products only from these family-care subgroups
FAMILY_SUBGROUPS = {"^Гигиена", "^Гигиена с пропиткой", "^Гигиена тела", "^Мыло", "^Средства для ухода за телом"}
CORE_PER_SUBGROUP = 3                       # core assortment: top SKUs of each subgroup in a chain
CORE_P_SELL = {"ММ": 0.45}                  # floor of the sell probability for core SKUs; other formats 0.9
MAX_P_SELL = {"ММ": 0.6}                    # small stores rotate a narrow shelf: no SKU sells in every one of them
SYLL = ["ka", "lo", "mi", "ra", "ve", "to", "zan", "dri", "pel", "sor", "vix", "bel", "cor", "lun", "mar", "nex", "qui", "tal", "vor", "zel"]


def _fill_brands(rng, n):
    """Long tail of minor fictional brands (Latin, uppercase): low recognition risk."""
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(SYLL, size=rng.integers(2, 4))).upper())
    return sorted(out)


# --------------------------------------------------------------------------- #
def make_stores(rng):
    rows, sid = [], 1_000_000
    geo_w = np.array([g[3] for g in GEO])
    geo_w = geo_w / geo_w.sum()
    for name, cid, n, formats, active, *_ in CHAINS:
        fmts, fw = zip(*formats.items())
        fw = np.array(fw) / sum(fw)
        for i in range(n):
            d, region, cities, _ = GEO[rng.choice(len(GEO), p=geo_w)]
            city = rng.choice(cities)
            if city and rng.random() < 0.22:        # many regional stores have no city either
                city = ""
            fmt = rng.choice(fmts, p=fw)
            fsrc = ""
            if name == "Гранит":
                fsrc = rng.choice(list(GRANIT_BANNERS), p=np.array(list(GRANIT_BANNERS.values())))
                fmt = {"МД": "ММ", "МК": "ММ", "МА": "ММ", "Наша Цена": "ММ", "ГМ": "ГМ"}[fsrc]
            sid += int(rng.integers(1, 40))
            code = f"{cid:02d}{i:05d}"
            rows.append(dict(store_id=sid, network=name, client_id=cid, store_code=code, store_code_in=code,
                             store_code_src=f"S{code}", format=fmt, format_src=fsrc or "не указан", district=d,
                             region=region, city=city, address=f"{region}, {city or region}, ул. Синтетическая, {i + 1}",
                             kladr=str(rng.integers(10**14, 10**15)), rc_supplier="не указан"))
    st = pd.DataFrame(rows)
    # "other" banner bucket: new largest-chain stores without a format code, opening over time
    extra = []
    for i in range(140):
        d, region, cities, _ = GEO[rng.choice(len(GEO), p=geo_w)]
        sid += int(rng.integers(1, 40))
        code = f"01{90000 + i:05d}"
        extra.append(dict(store_id=sid, network="Гранит", client_id=1, store_code=code, store_code_in=code,
                          store_code_src=f"S{code}", format="ММ", format_src="", district=d, region=region,
                          city=rng.choice(cities), address=f"{region}, ул. Новая, {i + 1}",
                          kladr=str(rng.integers(10**14, 10**15)), rc_supplier="не указан"))
    st = pd.concat([st, pd.DataFrame(extra)], ignore_index=True)
    st["open_from"] = pd.Period("2024-01", "M")
    st.loc[st.index[-140:], "open_from"] = [pd.Period("2024-03", "M") + int(k) for k in np.linspace(0, 30, 140)]
    # duplicate store_ids for the same physical store (same code and address)
    dups = []
    for net, months, k in [("Нива (сеть)", ("2026-06", "2026-08"), 30), ("Детский Остров (сеть)", ("2024-01", "2024-12"), 12),
                           ("ЛАД (сеть)", ("2025-05", "2025-05"), 4)]:
        base = st[st.network == net].sample(k, random_state=int(rng.integers(1e9)))
        for _, r in base.iterrows():
            sid += int(rng.integers(1, 40))
            r2 = r.copy()
            r2["store_id"], r2["dup_of"], r2["dup_months"] = sid, r["store_id"], months
            dups.append(r2)
    st = pd.concat([st, pd.DataFrame(dups)], ignore_index=True)
    st["size"] = st["format"].map(FORMAT_SIZE) * np.exp(rng.normal(0, 0.45, len(st)))
    return st


def make_skus(rng):
    rows, kid = [], 20_000_000
    tax = {s: (g, w, types, peak) for g, s, w, types, peak in TAXONOMY}

    def add(manufacturer, brand, subgroups, strength, n_sku, pl_chain=None, life=("", ""), weight=np.nan, types=None):
        nonlocal kid
        subs = subgroups or [s for s in tax if tax[s][1] >= 1]
        for _ in range(n_sku):
            sub = subs[rng.integers(len(subs))]
            g, w, sub_types, peak = tax[sub]
            ptypes = types or sub_types
            ptype = ptypes[rng.integers(len(ptypes))]
            mass = float(rng.choice([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]))
            size = f"{int(mass * 1000)}мл"
            variant = rng.choice(["Объем", "Питание", "Восстановление", "Свежесть", "Нежность", "Сияние", "Защита",
                                  "Баланс", "Классик", "Интенсив", "Для всей семьи", "Ромашка", "Алоэ", "Кокос"])
            kid += int(rng.integers(1, 9))
            base_price = 60 * np.exp(rng.normal(0.9, 0.5)) * (0.7 if pl_chain else 1) * (1.3 if manufacturer == "L'AURELLE" else 1)
            rows.append(dict(sku_id=kid, category_group=g, subgroup=sub, product_type=ptype, product=ptype.split()[0],
                             manufacturer=manufacturer, brand=brand, subbrand=f"{brand} {variant}",
                             sku_name=f"{ptype} {brand} {variant} {size}", sku_name_src=f"{ptype[:12]} {brand.lower()} {variant.lower()} {size}",
                             size_text=size, mass=mass, color="", country="", package="", quant="",
                             article_src=f"Б{kid}", barcode_src=f"Ш46{kid:010d}",
                             strength=strength, w_line=weight, life_from=life[0], life_to=life[1],
                             base_price=round(base_price, 1), pl_chain=pl_chain or "",
                             trend=rng.normal(0.0, 0.18), peak=peak or 0))

    for brand, subs, s, *opt in OWN_BRANDS:
        add(OWN, brand, subs, s, max(2, int(s * 14)), **(opt[0] if opt else {}))
    for m, brands in COMPETITORS.items():
        for brand, subs, s, *opt in brands:
            add(m, brand, subs, s, max(2, int(s * 10)), **(opt[0] if opt else {}))
    for chain, brands in PL.items():
        for b in brands:
            add(f"STM({STM_CODE[chain]})", b, None, 2.5, 40, pl_chain=chain)
    tail_m = [f"{x} LLC" for x in _fill_brands(rng, 60)]
    for i, b in enumerate(_fill_brands(rng, 320)):
        add(tail_m[i % len(tail_m)] if rng.random() > 0.25 else "не указан", b, None, float(rng.lognormal(-1.2, 0.8)), int(rng.integers(1, 6)))
    sk = pd.DataFrame(rows)
    sk["weight"] = sk["w_line"].fillna(sk["subgroup"].map({s: w for g, s, w, *_ in TAXONOMY}))
    return sk


def make_fact(rng, st, sk):
    tax_w = sk["weight"].to_numpy()
    sku_pop = sk["strength"].to_numpy() * np.exp(rng.normal(0, 0.8, len(sk))) * tax_w / tax_w.mean()
    # listing per chain: big manufacturers everywhere, tail more selective; PL only in own chain
    listed, core = {}, {}
    for name, *_ in CHAINS:
        p = np.clip(0.25 + 0.6 * (sk["strength"] > 1.0), 0, 1) * (1.0 if name not in ("Детский Остров (сеть)",) else 0.6)
        mask = (rng.random(len(sk)) < p) & ((sk["pl_chain"] == "") | (sk["pl_chain"] == name)).to_numpy()
        if name == "Детский Остров (сеть)":   # kids chain: mostly kids subgroups
            mask &= (sk["subgroup"].str.contains("Детск") | (sk["subgroup"].isin(FAMILY_SUBGROUPS) & (rng.random(len(sk)) < 0.15))).to_numpy()
        listed[name] = np.flatnonzero(mask)
        # core assortment: the most popular SKUs of established brands (strength >= 1) in each subgroup
        lst = pd.DataFrame({"sub": sk["subgroup"].to_numpy()[listed[name]], "pop": sku_pop[listed[name]],
                            "ok": sk["strength"].to_numpy()[listed[name]] >= 1.0})
        rank = lst["pop"].where(lst["ok"]).groupby(lst["sub"]).rank(ascending=False, method="first")
        core[name] = (rank <= CORE_PER_SUBGROUP).to_numpy()

    out = []
    for m in MONTHS:
        ms = str(m)
        t = (m - MONTHS[0]).n / 12
        season = 1 + 0.08 * np.cos(2 * np.pi * (m.month - 12) / 12)
        peak = sk["peak"].to_numpy()
        sku_season = np.where(peak > 0, 1 + 0.6 * np.cos(2 * np.pi * (m.month - peak) / 12), 1.0)
        sku_trend = np.exp(sk["trend"].to_numpy() * t)
        price = sk["base_price"].to_numpy() * 1.08 ** t
        alive = (((sk["life_from"] == "") | (sk["life_from"] <= ms)) & ((sk["life_to"] == "") | (ms <= sk["life_to"]))).to_numpy()
        parts = []
        for name, cid, n, formats, active, price_lvl, cost in CHAINS:
            if active is not None:
                if isinstance(active, tuple) and not (active[0] <= ms <= active[1]):
                    continue
                if isinstance(active, list) and ms not in active:
                    continue
            s = st[(st.network == name) & (st.open_from <= m)]
            # duplicates active only in their months; their originals are inactive then
            if "dup_of" in s:
                dup_on = s["dup_months"].apply(lambda x: isinstance(x, tuple) and x[0] <= ms <= x[1])
                originals_off = set(s.loc[dup_on, "dup_of"])
                s = s[(s["dup_of"].isna() & ~s["store_id"].isin(originals_off)) | dup_on]
            if name == "ЛАД (сеть)" and ms == "2026-09":
                s = s.sample(frac=0.4, random_state=1)
            frac = 0.6 if (name == "Гиперкосмос" and ms == "2026-09") else 1.0
            idx = listed[name]
            lam = (np.outer(s["size"].to_numpy(), sku_pop[idx] * sku_season[idx] * sku_trend[idx] * alive[idx]) * season * 0.9 * frac)
            p_sell = np.minimum(1 - np.exp(-lam * 0.16), s["format"].map(MAX_P_SELL).fillna(1.0).to_numpy()[:, None])
            floor = s["format"].map(CORE_P_SELL).fillna(0.9).to_numpy()[:, None] * frac
            p_sell = np.where(core[name][None, :] & alive[idx][None, :], np.maximum(p_sell, floor), p_sell)
            hit = rng.random(lam.shape) < p_sell
            r, c = np.nonzero(hit)
            qty = rng.poisson(lam[r, c] * 4.0) + 1
            sku_i = idx[c]
            rub = qty * price[sku_i] * price_lvl * np.exp(rng.normal(0, 0.04, len(qty)))
            yr = min(m.year - 2024, 2)
            ratio = cost[yr] * np.exp(rng.normal(0, 0.03, len(qty)))
            parts.append(pd.DataFrame({"store_id": s["store_id"].to_numpy()[r], "sku_id": sk["sku_id"].to_numpy()[sku_i],
                                       "qty": qty.astype(float), "rub": np.round(rub, 2),
                                       "kg": np.round(qty * sk["mass"].to_numpy()[sku_i], 3),
                                       "cost": np.round(rub * ratio, 2)}))
        f = pd.concat(parts, ignore_index=True)
        out.append((ms, f))
        print(f"  {ms}: {len(f):,} rows")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    st, sk = make_stores(rng), make_skus(rng)
    (OUT / "dim").mkdir(parents=True, exist_ok=True)
    dim_store = st.drop(columns=["size", "open_from", "dup_of", "dup_months"], errors="ignore")
    dim_sku = sk.drop(columns=["strength", "w_line", "life_from", "life_to", "base_price", "pl_chain", "trend", "peak", "weight"])
    dim_store.to_parquet(OUT / "dim" / "dim_store.parquet", index=False)
    dim_sku.to_parquet(OUT / "dim" / "dim_sku.parquet", index=False)
    print(f"dim_store {len(dim_store):,}  dim_sku {len(dim_sku):,}")
    for ms, f in make_fact(rng, st, sk):
        d = OUT / "fact" / f"month={ms}"
        d.mkdir(parents=True, exist_ok=True)
        f.to_parquet(d / "part.parquet", index=False)


if __name__ == "__main__":
    main()
