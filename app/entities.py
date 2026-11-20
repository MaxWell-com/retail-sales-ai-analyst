"""Rule-based question parsing before the LLM: networks, banners, formats, manufacturers, brands, categories,
geography, periods, measure, exclusions («кроме», «без»), breakdown, sorting, comparison object, follow-ups.

The local model garbles names («Гранит» -> «Granite»), so it is handed the exact dictionary values
(«Готовые аргументы», "ready-made arguments"), and the code then overrides the model's arguments with them.
Hence the rule: better to leave a doubtful word unrecognised than to inject a wrong filter
(«Верно ли» is not the network Добрый, «лидер» is not the brand ЛИДЕР).

How it works: the question is split into words keeping the original spelling (case, quotes, sentence start);
periods are parsed with regexes and their words are masked out; then the keys of all dictionaries (with case
endings) are matched against the words in a single pass: a longer key beats a shorter one («GREEN SHOP» is a
brand, not the manufacturer GREEN); on equal length the entity kind and context decide («бренд», «СТМ», «в», «г.»).
"""
import re
from dataclasses import dataclass, field
from datetime import date
from itertools import product

from . import semantic as S


YO = {"\u0451": "е", "\u0401": "Е"}    # \u0451 and \u0401 -> «е»: dictionaries and questions spell it inconsistently


def de_yo(s):
    return re.sub("[\u0451\u0401]", lambda m: YO[m.group()], str(s))


def norm(s):
    s = de_yo(s).lower()
    s = re.sub(r"[«»\"'`’]", "", s)
    s = re.sub(r"\s*&\s*", "&", s)            # HILL & SHORE = HILL&SHORE
    s = re.sub(r"[^0-9a-zа-я&+]+", " ", s)
    return " " + re.sub(r"\s+", " ", s).strip() + " "


# ——— Word forms ———
# A word pattern is (stem, endings): EXACT — the whole word, ANY4 — stem plus up to 4 letters (network aliases),
# ANY — stem plus anything (product keywords), otherwise — a set of allowed endings.
EXACT = frozenset([""])
ANY4, ANY = "4", "*"
VOWELS = "аеиоуыэюя"
ADJ = frozenset("ый ий ой ая яя ое ее ые ие ого его ому ему ым им ом ем ую юю ых их ыми ими ою ею".split())
N_CONS = frozenset(["", "а", "у", "е", "ом", "ем", "ы", "и", "ов", "ев", "ам", "ах", "ами"])   # Гранит, Омск, АИСТ
N_A = frozenset(["а", "ы", "и", "е", "у", "ой", "ою"])                                         # Королевна, Тула
N_YA = frozenset(["я", "и", "ю", "е", "ей", "ею"])                                             # Карелия, Лилия
N_SOFT = frozenset(["ь", "и", "ью", "я", "ю", "е", "ем"])                                      # Пермь, Ярославль
N_O = frozenset(["о", "а", "у", "е", "ом"])                                                    # Иваново, Морозко
N_E = frozenset(["е", "я", "ю", "ем"])                                                         # Поволжье
N_PL = frozenset(["ы", "и", "", "ов", "ев", "ам", "ям", "ах", "ях", "ами", "ями"])               # Химки, Мегадраконы
N_J = frozenset(["й", "я", "ю", "е", "ем", "и", "ев"])                                         # Алтай, Ладно
FLEET = frozenset(["ок", "ек"])                                                                # масок, салфеток
ADJ_TAIL = ("ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие")
PREP_CASE = frozenset(["", "е", "и", "о", "ах", "ях"])     # after «в» ("in"): «в Самаре», «в Перми», «в Химках»


def decl(w):
    """Word-pattern variants with case endings: «тула» -> туле, тулы; «краска» -> красок."""
    if not re.fullmatch(r"[а-я]{3,}", w):
        return [(w, EXACT)]
    if len(w) >= 5 and w.endswith(ADJ_TAIL):
        return [(w[:-2], ADJ)]
    last, stem = w[-1], w[:-1]
    fleet = [(w[:-2], FLEET)] if w[-2] == "к" and w[-3] not in VOWELS and last in "аи" else []
    if last == "а":
        return [(stem, N_A)] + fleet
    if last == "я":
        return [(stem, N_YA)]
    if last == "ь":
        return [(stem, N_SOFT)]
    if last == "о":
        return [(stem, N_O)]
    if last == "е":
        return [(stem, N_E)]
    if last in "ыи":
        return [(w, EXACT)] if len(w) <= 4 else [(stem, N_PL)] + fleet
    if last == "й":
        return [(stem, N_J)]
    if last in VOWELS:
        return [(w, EXACT)]
    out = [(w, N_CONS)]
    if re.search(r"[^аеиоуыэюя](ок|ек|ец)$", w):          # перелесок -> перелеска
        out.append((w[:-2] + w[-1], N_CONS - EXACT))
    return out


def _wm(tok, pat):
    stem, ends = pat
    if ends is EXACT:
        return tok == stem
    if not tok.startswith(stem):
        return False
    rest = tok[len(stem):]
    if ends == ANY:
        return True
    if ends == ANY4:
        return len(rest) <= 4 and (not rest or rest.isalpha())
    return rest in ends


FIXED_AFTER = {"для", "за", "с", "со", "от", "после", "под", "на", "по", "из", "и"}


def pats(text, mode="decl"):
    """Word patterns of a key: for each word, its (stem, endings) variants.
    decl — case forms (words after «для», «на», «и» are not declined: «крем для рук», «Ростов-на-Дону»);
    alias — stem + up to 4 letters, as for network aliases; kw — word prefix; exact — the whole word."""
    out, fixed = [], False
    for w in norm(text).split():
        if mode == "kw":
            alts = [(w, ANY)]
        elif mode == "exact" or fixed:
            alts = [(w, EXACT)]
        elif mode == "alias":
            alts = [(w, EXACT if len(w) <= 2 else ANY4)]
            if re.fullmatch(r"[а-я]+[йья]", w):
                alts += decl(w)
        else:
            alts = decl(w)
        out.append(alts)
        fixed = fixed or (mode == "decl" and w in FIXED_AFTER)
    return out


# ——— Aliases (also used by tools.resolve_filter) ———
# Network words match with endings of up to 4 letters: «гранит» finds «Граните», «нив» finds «Ниве»
NETWORK_ALIASES = {
    "Гранит": ["гранит", "маяк", "mayak", "granit"],
    "Семерочка Х7": ["семерочк", "семерк", "7ка", "semerochka"],
    "Перелесок ТД": ["перелес", "pereleski"],
    "Нива (сеть)": ["нив", "niva"],
    "Ажур": ["ажур", "azhur"],
    "Депо Кэш Энд Керри": ["депо", "depo"],
    "ЛАД (сеть)": ["ладно", "ла дно", "ladno", "la dno", "лад"],
    "Гиперкосмос": ["гиперкосмос", "космос", "kosmos"],
    "Детский Остров (сеть)": ["детск остров", "детостров"],
    "Запас": ["запас"],
    "Добрый (сеть)": ["добрый"],
    "Маятник (сеть)": ["маятник"],
    "Стрижик": ["стрижик"],
    "Камелия (сеть)": ["камели"],
    "ПАРУС (сеть)": ["парус", "parus"],
    "НУ (сеть)": ["ну"],
}
NETWORK_GROUPS = {"х7": ["Семерочка Х7", "Перелесок ТД"], "x7": ["Семерочка Х7", "Перелесок ТД"]}
BANNER_ALIASES = {
    "Маяк Косметик": ["маяк косметик", "маяк cosmetic"],
    "Маяк у дома": ["маяк у дома"],
    "Маяк Аптека": ["маяк аптек"],
    "Наша Цена": ["наша цен", "нашей цен"],
    "Маяк гипермаркет": ["маяк гипермаркет", "гипермаркет маяк", "маяк семейн", "маяк экстра"],
}
FORMAT_ALIASES = {
    "СМ": ["супермаркет", "см"],
    "ГМ": ["гипермаркет", "гм"],
    "ММ": ["магазин у дома", "малом формат", "малый формат", "мм"],
    "Интернет Магазин": ["интернет магазин", "онлайн"],
}
# Aliases that are common words («Верно ли», «Ладно, а в штуках?», «семерка лидеров»): a network only in context
STORE_CTX = {"семерк": "five", "ладно": "ok", "ла дно": "ok", "ladno": "ok", "la dno": "ok", "лад": "upper",
             "ну": "near_upper", "добрый": "cap", "запас": "cap", "наша цен": "banner", "нашей цен": "banner",
             "см": "upper", "гм": "upper", "мм": "upper"}
STORE_EXACT = {"депо", "depo", "лад", "ну", "см", "гм", "мм"}
# Network private labels: «СТМ Маяка», «СТМ X7» -> manufacturer STM(<code>)
NET_STM = {"Гранит": "GRANIT", "Семерочка Х7": "X7", "Перелесок ТД": "X7", "Нива (сеть)": "NIVA", "Ажур": "AZHUR",
           "Депо Кэш Энд Керри": "DEPO", "ЛАД (сеть)": "LADNO", "Гиперкосмос": "KOSMOS", "ПАРУС (сеть)": "PARUS",
           "Детский Остров (сеть)": "DETSKIY OSTROV", "Добрый (сеть)": "DOBRIY", "Камелия (сеть)": "KAMELIA"}

MANUFACTURER_ALIASES = {
    "NORD": ["nord", "норд", "норда", "норду", "норде", "нордом"],
    "PARKER & GRANT": ["p&g", "parker and grant", "parker grant", "паркер энд грант", "паркер энд гранд",
                         "паркер"],
    "UNIVERA": ["юнивера"],
    "HELMAN": ["хелман", "хелма"],
    "BERGSTAD": ["бергштад", "бергстад"],
    "L'AURELLE": ["лорель", "лорел"],
    "KENDALL-CROSS LTD": ["kendall cross", "кендалл кросс"],
    "JAMESON & JAMESON": ["jameson and jameson", "джеймсон энд джеймсон"],
    "CORVIN-OLIVERA": ["корвин оливера"],
}
# Cyrillic spellings of brands (only brands from the dictionary; checked in tests/test_entities.py)
BRAND_ALIASES = {
    "OLVIA": ["олвиа", "олвия"], "VERDANT": ["вердант", "верданн"], "SPUMA": ["спума"], "ESTRA": ["эстра"],
    "LUMINA": ["люмина", "люмине"], "HILL&SHORE": ["хилл энд шор", "хил энд шор", "хиллен шор",
                                                        "hill and shore"],
    "FRUIT DRAGONS": ["фрут драгонс"], "TERRA BOREALIS": ["терра бореалис"], "L'AURELLE PARIS": ["лорель париж"],
    "PAPYRA": ["папира"], "TOFFI": ["тоффи"], "DEFENDA": ["дефенда"], "OLD HARBOR": ["олд харбор"],
    "PUFFIES": ["паффис"], "OLIVERA": ["оливера"], "SYLVA": ["сильва", "сайлва"], "CLARO": ["кларо"],
    "MIXTA": ["микста"], "SHINE KUR": ["шайн кур", "шайн кюр"], "FIXA": ["фикса"], "CUDDLES": ["каддлс"],
    "CAMEA": ["камеа"], "SPLASH": ["сплэш"], "COLDENT": ["колдент"], "VELURA": ["велюра"], "MEADOWA": ["медова"],
    "BRISK": ["бриск"], "DERMOLIB": ["дермолиб"], "OLLAR": ["оллар"], "DEOLINA": ["деолина"],
    "ATELIER PROFESSIONAL": ["atelier", "ателье"], "VECTOR": ["вектор"], "SILK": ["силк"],
    "WILD GIRL": ["уайлд герл", "вайлд герл"], "DERMALINA": ["дермалина"], "COLORIA": ["колория"],
    "MEGADRAGONS": ["мегадраконы"], "МЕГАДРАКОНЫ": ["megadragons"], "MELLO [MELVA]": ["мелва"],
    "CELINE MARAIS": ["селин маре"], "HERBAL SPIRIT": ["хербал спирит"], "LACTOFIN": ["лактофин"],
    "ЛЕСНАЯ ТРОПА": ["forest path"],
}
# Brands that are common words: a brand only with an explicit signal (UPPER case, quotes, «бренд»/«марка» before
# the word, capital letter not at sentence start; for Latin script — only quotes and «бренд»: «ABC-анализ», «OLVIA MEN»)
WORD_BRANDS = set("""лидер еще просто точно эффект новинка весна основа семья мечта экстра спектр оптима луч альфа
радуга вечер домой грани сотка дыши малышам идеальная жемчужная свобода титан кристалл забава натур каре бриз
ласка норка алмаз трактор консул меридиан зодиак юпитер флора воевода пикник фасоль прелесть мила нега егоза
пома лора мэтр рассвет кокетка пантенол лилия пушинка алиса маруся солнышко лакомка золушка спасатель лекарь
аптекарь фельдшер знахарь травница батист грация удача спутник сокол
abc xyz men man home love time eco yes smart fresh help click focus trust lucky agent pure near merci top premium
sale promo spa life beauty style care natural organic bio kids kid baby woman lady clean basic total best gold
classic original plus pro max mini""".split()) | {
    "то что надо", "каждый день", "ваш выбор", "до и после", "так чисто", "правильный выбор", "для всей семьи",
    "особая серия", "две линии", "мой малыш", "я самая", "скорая помощь", "выгодная покупка", "натуральные масла",
    "мир детства", "в 60 как в 30", "из набережных челнов", "чисто дома", "чистый дом"}
# Analysis terms: UPPER case is not a brand signal for them («ABC-анализ», «топ SKU»)
TERMS = {"abc", "xyz", "sku", "mat", "ytd", "kpi", "edlp", "kam", "top", "spf", "max", "mini", "plus", "pro", "total"}
# Never brands: function words, sales-question words, kids words (those are the kids signal)
STOP_BRANDS = {"для", "мыло", "шампунь", "детский", "детская", "мир", "дом", "да", "ну", "нет", "про", "все", "nord", "био",
               "сеть", "сети", "топ", "доля", "рост", "год", "это", "как", "что", "где", "или", "без", "при", "так",
               "наш", "pl", "стм", "stm", "sku", "ску", "тм", "id", "by", "ok", "vs"}
# Brand key equal to a banner name: «Наша Цена» without «в»/«магазин» is the brand НАША ЦЕНА (PL)
BRAND_OVER_STORE = {"наша цена"}

# Product keywords (consecutive word prefixes); longer before shorter. Also used by tools.resolve_filter.
PRODUCT_KEYWORDS = [
    ("сух шампун", "product_type", "Сухой шампунь"),
    ("шампун для бород", "product_type", "Шампунь для бороды"),
    ("шампун", "product_type", "Шампунь"),
    ("кондиционер для волос", "product_type", "Бальзам для волос"),
    ("кондиционер", "product_type", "Бальзам для волос"),
    ("бальзам для волос", "product_type", "Бальзам для волос"),
    ("бальзам ополаскивател", "product_type", "Бальзам для волос"),
    ("ополаскивател для волос", "product_type", "Бальзам для волос"),
    ("бальзам для губ", "product_type", "Уход за губами"),
    ("оттеноч бальзам", "product_type", "Оттеночные средства для волос"),
    ("бальзам", "product_type", "Бальзам для волос"),
    ("маск для волос", "product_type", "Маски для волос"),
    ("масл для волос", "product_type", "Масло для волос"),
    ("жидк мыл", "product_type", "Мыло Жидкое"),
    ("кусков мыл", "product_type", "Мыло кусковое"),
    ("хозяйствен мыл", "product_type", "Мыло хозяйственное"),
    ("краск для волос", "subgroup", "Средства для окраски волос"),
    ("крас для волос", "subgroup", "Средства для окраски волос"),
    ("окрашиван", "subgroup", "Средства для окраски волос"),
    ("красител", "subgroup", "Средства для окраски волос"),
    ("краск", "subgroup", "Средства для окраски волос"),
    ("красок", "subgroup", "Средства для окраски волос"),
    ("оттеноч", "product_type", "Оттеночные средства для волос"),
    ("солнцезащит", "subgroup", "Солнцезащитные средства и уход"),
    ("spf", "subgroup", "Солнцезащитные средства и уход"),
    ("гигиен волос", "subgroup", "Гигиена волос и кожи головы"),
    ("уход за волос", "subgroup", "Уход за волосами и кожей головы"),
    ("укладк", "subgroup", "Средства для укладки волос"),
    ("гел для душ", "product_type", "Гель для душа"),
    ("гигиен тела", "subgroup", "Гигиена тела"),
    ("дезодорант", "subgroup", "Дезодоранты"),
    ("антиперспирант", "subgroup", "Дезодоранты"),
    ("зубн паст", "product_type", "Зубная паста"),
    ("зубн щетк", "product_type", "Зубная щетка"),
    ("подгузник", "product_type", "Подгузники детские"),
    ("уход за кож лица", "subgroup", "Уход за кожей лица"),
    ("уход за телом", "subgroup", "Средства для ухода за телом"),
    ("парфюм", "subgroup", "Парфюмерия"),
    ("мыл", "subgroup", "Мыло"),
]
# Products absent from the panel: their words are consumed so they do not become a wrong category («кондиционер для белья»)
PRODUCT_NONE = ["кондиционер для бель", "кондиционер для стир", "кондиционер для ткан", "кондиционер для одежд",
                "ополаскивател для бель"]
# Residual (near-empty) types: not recognised by name; «салфетки» means both wet and paper wipes
RESIDUAL_TYPES = {"Салфетки", "Набор", "Прочее", "Прочие", "Лечебный уход", "Тряпки/Салфетки", "Набор универсальный",
                  "Подарочный набор"}
PRODUCT_MULTI = [("салфетк", "product_type", ("Салфетки влажные", "Салфетки бумажные"))]

# Geography: colloquial names -> store values; «МО», «ЛО» only in upper case
REGION_ALIASES = {
    "Москва г": ["мск"], "Санкт-Петербург г": ["петербург", "питер", "спб"],
    "Московская обл": ["подмосковье", "МО"], "Ленинградская обл": ["ленобласть", "ЛО"],
    "Краснодарский край": ["кубань"], "Башкортостан": ["башкирия"], "Татарстан": ["татария"],
    "Саха (Якутия)": ["якутия"], "Ханты-Мансийский АО": ["хмао", "югра"], "Ямало-Ненецкий АО": ["янао"],
    "РСО-Алания": ["северная осетия", "алания"], "Кабардино-Балкария": ["кбр"], "Карачаево-Черкесия": ["кчр"],
}
DISTRICT_NAMES = {"ЦФО": "центральный", "ПФО": "приволжский", "СЗФО": "северо западный", "ЮФО": "южный",
                  "УФО": "уральский", "СФО": "сибирский", "СКФО": "северо кавказский", "ДФО": "дальневосточный",
                  "КФО": "крымский"}
DISTRICT_ALIASES = {"УФО": ["урал"], "СФО": ["сибирь"], "ДФО": ["дальний восток"], "ПФО": ["поволжье"],
                    "СЗФО": ["северо запад"], "СКФО": ["северный кавказ"]}
# cities keyed by normalised name (without «г»): in speech «ростов» almost always means Ростов-на-Дону
CITY_ALIASES = {"ростов на дону": ["ростов"], "ростов": ["ростов великий"], "екатеринбург": ["екб"],
                "нижний новгород": ["н новгород"], "орел": ["орла"]}
CITY_REGIONS = {"Москва г", "Санкт-Петербург г", "Севастополь г"}       # city = region (see semantic.store_x)
# «Ростовская» without «области» is not matched: a one-word regional adjective needs the region type
REGION_TYPES = {"обл": [pats("обл", "kw")],
                "край": [[[("кра", frozenset(["й", "я", "ю", "е", "ем"]))]]],
                "АО": [pats("ао", "exact"), pats("автономный округ")],
                "Аобл": [pats("аобл", "exact"), pats("автономная область"), pats("ао", "exact")]}

# Priority on equal key length (lower wins)
KIND_PRIO = {"banner": 1, "network": 2, "format": 3, "district": 4, "region": 5, "city": 6, "manufacturer": 7,
             "brand": 8, "kw": 9, "none": 9, "product_type": 10, "subgroup": 10, "category_group": 11}
OBJECT_KINDS = ("manufacturer", "brand")
STORE_KINDS = ("network", "banner", "format", "district", "region", "city")

TOK_RE = re.compile(r"[0-9a-zа-я&+]+(?:['’`][0-9a-zа-я]+)*", re.I)
PREP_LOC = {"в", "во", "из", "по", "у", "для", "с", "со", "о", "об"}
BRAND_CTX = {"марка", "марки", "марку", "маркой", "марке", "марок", "тм", "линейка", "линейки", "линейку",
             "линейкой", "линейке", "серия", "серии", "серию"}
STM_WORDS = {"стм", "stm", "pl", "смт"}
MAN_CTX = ("производител", "компани", "завод", "концерн", "холдинг", "поставщик", "вендор")


class _Text:
    """Question words: normalised form, original spelling, sentence start, quotes, comma after the word."""

    def __init__(self, text):
        s = re.sub(r"\s*&\s*", "&", de_yo(text))
        self.low = s.lower()
        self.w, self.o, self.pos, self.first, self.quoted, self.apos, gaps = [], [], [], [], [], [], []
        q, prev = False, 0
        for m in TOK_RE.finditer(s):
            gap = s[prev:m.start()]
            for ch in gap:
                if ch in "«“„":
                    q = True
                elif ch in "»”":
                    q = False
                elif ch in "\"'‘’":
                    q = not q
            gaps.append(gap)
            self.first.append(not self.w or bool(re.search(r"[.!?…:;\n]", gap)))
            self.quoted.append(q)
            o = re.sub(r"['’`]", "", m.group())
            self.apos.append(o != m.group())
            self.o.append(o)
            self.w.append(o.lower())
            self.pos.append((m.start(), m.end()))
            prev = m.end()
        self.comma = ["," in g for g in gaps[1:]] + [False]
        self.used = [False] * len(self.w)
        self.kind = [None] * len(self.w)        # kind of the entity accepted on this word
        self.line = " " + " ".join(self.w) + " "

    def mask(self, spans):
        for t, (a, b) in enumerate(self.pos):
            if any(a < y and x < b for x, y in spans):
                self.used[t] = True

    def prev(self, i, k=1):
        return self.w[i - k] if i - k >= 0 else ""

    def nxt(self, i):
        return self.w[i] if i < len(self.w) else ""

    def cap_mid(self, i):
        """Capitalised and not at sentence start — looks like a proper name."""
        return self.o[i][:1].isupper() and not self.first[i]

    def upper(self, i):
        o = re.sub(r"[^A-Za-zА-Яа-я]", "", self.o[i])
        return len(o) >= 2 and o.isupper()

    def brand_ctx(self, i):
        """«бренд Маяк», «марки Лидер»: the word «бренд» right before the name («брендов в Ниве» is not a signal)."""
        return self.prev(i).startswith("бренд") or self.prev(i) in BRAND_CTX

    def man_ctx(self, i):
        return self.prev(i).startswith(MAN_CTX)

    def stm_ctx(self, i):
        p1, p2 = self.prev(i), self.prev(i, 2)
        return (p1 in STM_WORDS or (p1.startswith("марк") and p2.startswith("собствен"))
                or (p1 == "label" and p2 == "private"))


@dataclass
class Found:
    filters: dict = field(default_factory=dict)
    periods: list = field(default_factory=list)       # [Period] in order of mention
    period_kind: str | None = None                    # mat / ytd / all / None
    measure: str | None = None
    kids: bool | None = None
    portal: bool = False
    future_month: date | None = None
    exclude: dict = field(default_factory=dict)       # «кроме Гранита», «без Маяк Косметик» ("except ...", "without ...")
    exclude_pl: bool = False                          # «без СТМ» (without private labels)
    out_of_range: list = field(default_factory=list)  # periods outside the data: «2023 год»
    sort: str | None = None                           # growth / gain / decline / share_change
    focus: str | None = None                          # «на уровне NORD», «соседи NORD» — a reference object, not a filter
    dims: set = field(default_factory=set)            # breakdown levels: «в каких сетях», «доля Гранита в ...»
    mentioned: dict = field(default_factory=dict)     # breakdown and comparison-object values (not filters)
    level_hint: str | None = None                     # main ranking level
    followup: bool = False                            # follow-up to the previous question
    measures: list = field(default_factory=list)      # two measures: «в штуках и в рублях»
    yoy: bool = False                                 # «к прошлому году»: the base is the same months a year earlier
    role: str | None = None                           # «я КАМ NORD»: who is asking, not a filter
    kids_both: bool = False                           # «взрослые и детские» (adults and kids)
    notes: list = field(default_factory=list)         # notes for the «Распознано» ("Recognised") line

    def add(self, key, value, where=None):
        vals = (self.filters if where is None else where).setdefault(key, [])
        if value not in vals:
            vals.append(value)

    def period_pair(self):
        """(current, base): current is the later period, base the earlier one aligned to the same calendar months.
        Nested periods have no base; three or more give their overall span with no base (monthly comparison)."""
        ps = sorted(self.periods, key=lambda p: (p.start, p.end))
        if not ps:
            return None, None
        if len(ps) >= 3:
            return S.Period(ps[0].start, max(p.end for p in ps)), None
        if len(ps) == 1:
            cur, base = ps[0], (S.clip(ps[0].shift(-12)) if self.yoy else None)
            return cur, (base if base and base.n == cur.n else None)
        base, cur = ps
        if base.start <= cur.start and cur.end <= base.end:
            return cur, None
        if cur.start <= base.start and base.end <= cur.end:
            return base, None
        if base.n != cur.n:
            k = cur.start.year - base.start.year
            b2 = S.clip(cur.shift(-12 * k)) if k > 0 else None
            return cur, (b2 if b2 and b2.n == cur.n else None)
        return cur, base

    def summary(self):
        out = [f"{FILTER_RU.get(k, k)} = {', '.join(v)}" for k, v in self.filters.items()]
        out += [f"исключить: {FILTER_RU.get(k, k)} {', '.join(v)}" for k, v in self.exclude.items()]
        if self.exclude_pl:
            out.append("исключить: СТМ (собственные марки сетей)")
        if self.role:
            out.append(f"пользователь — {self.role} (не фильтр)")
        if self.focus:
            out.append(f"объект сравнения (не фильтр) = {self.focus}")
        for k in sorted(self.dims):
            vals = self.mentioned.get(k)
            out.append(f"разрез: {LEVEL_RU.get(k, k)}" + (f" (упомянуты: {', '.join(vals)})" if vals else ""))
        if self.level_hint and self.level_hint not in self.dims:
            out.append(f"уровень рейтинга: {LEVEL_RU.get(self.level_hint, self.level_hint)}")
        cur, base = self.period_pair()
        if len(self.periods) >= 3:
            out.append("периоды: " + ", ".join(p.label() for p in self.periods)
                       + " — несколько периодов: сравнение помесячно (trend)")
        elif cur:
            out.append(f"период = {cur.label()} ({cur.to_str()})")
            if base:
                out.append(f"база сравнения = {base.label()} ({base.to_str()})")
        elif self.yoy:
            out.append("сравнение с теми же месяцами прошлого года")
        if self.period_kind and not self.periods:
            out.append("период = " + KIND_RU[self.period_kind])
        if self.out_of_range:
            lo, hi = S.data_bounds()
            out.append(f"вне данных: {', '.join(self.out_of_range)} (в базе {lo:%m.%Y}–{hi:%m.%Y})")
        if self.future_month:
            out.append(f"месяц в будущем (прогноз) = {S.ym(self.future_month)}")
        if self.measures:
            out.append("меры = " + ", ".join(S.MEASURE_NAME[m] for m in self.measures))
        elif self.measure:
            out.append(f"мера = {S.MEASURE_NAME[self.measure]}")
        if self.kids is not None:
            out.append("детские = " + ("да" if self.kids else "нет"))
        elif self.kids_both:
            out.append("взрослые и детские — обе группы")
        if self.portal:
            out.append("категория Косметика портала = да")
        if self.sort:
            out.append(f"сортировка = {SORT_RU[self.sort]}")
        if self.followup:
            out.append("уточнение прошлого вопроса")
        out += self.notes
        return "; ".join(out) if out else "ничего не распознано"

    def as_args(self):
        """Ready-made tool arguments — the model can use them as is."""
        a = {k: (v if len(v) > 1 else v[0]) for k, v in self.filters.items()}
        cur, base = self.period_pair()
        if cur:
            a["period"] = cur.to_str()
            if base:
                a["base_period"] = base.to_str()
        elif self.period_kind:
            a["period"] = self.period_kind
        if self.measure:
            a["measure"] = self.measure
        if self.kids is not None:
            a["kids"] = self.kids
        if self.portal:
            a["portal"] = True
        if self.exclude_pl:
            a["exclude_pl"] = True
        if self.exclude:
            a["exclude"] = {k: (v if len(v) > 1 else v[0]) for k, v in self.exclude.items()}
        if self.sort:
            a["sort"] = self.sort
        if self.focus:
            a["focus"] = self.focus
        if self.level_hint:
            a["level"] = self.level_hint
        return a


FILTER_RU = {"manufacturer": "производитель", "brand": "бренд", "subbrand": "подбренд", "subgroup": "подгруппа",
             "product_type": "тип товара", "category_group": "группа", "network": "сеть", "banner": "баннер",
             "format": "формат", "district": "округ", "region": "регион", "city": "город", "sku": "SKU"}
LEVEL_RU = {"manufacturer": "производители", "brand": "бренды", "sku": "SKU", "subgroup": "подгруппы",
            "product_type": "типы товара", "network": "сети", "banner": "баннеры", "format": "форматы",
            "district": "округа", "region": "регионы", "city": "города"}
KIND_RU = {"mat": "последние 12 месяцев (mat)", "ytd": "с начала года (ytd)", "all": "весь период данных (all)"}
SORT_RU = {"growth": "по росту", "gain": "по приросту", "decline": "по падению", "share_change": "по изменению доли"}


class Dictionary:
    """Recognition dictionaries, loaded once at startup."""

    def __init__(self):
        c = S.con()
        self.networks = [r[0] for r in c.execute("SELECT DISTINCT network FROM store").fetchall()]
        # manufacturers above the revenue threshold plus every «...NORD...»: the long key SHENZHEN NORD TECHNOLOGY beats «nord»
        self.manufacturers = c.execute("""
            SELECT k.manufacturer, sum(n.rub) rub FROM net_sku n JOIN sku k USING (sku_id)
            WHERE k.manufacturer IS NOT NULL GROUP BY 1
            HAVING sum(n.rub) > 1e6 OR regexp_matches(k.manufacturer, ?)""", [rf"\b{S.OWN}\b"]).fetchall()
        pairs = c.execute("""
            SELECT k.brand, k.manufacturer, sum(n.rub) FROM net_sku n JOIN sku k USING (sku_id)
            WHERE k.brand IS NOT NULL GROUP BY 1, 2""").fetchall()
        tot, mans = {}, {}
        for b, m, r in pairs:
            tot[b] = tot.get(b, 0) + (r or 0)
            if m:
                mans.setdefault(b, set()).add(m)
        self.brands = [(b, ", ".join(sorted(mans.get(b, ()))), r) for b, r in tot.items() if r > 1e5]
        self.brand_set = {b for b, _, _ in self.brands}
        self.man_names = {m for m, _ in self.manufacturers}
        self._pair_rub = {(b, m): r or 0 for b, m, r in pairs}
        self.subgroups = [r[0] for r in c.execute("SELECT DISTINCT subgroup_base FROM sku WHERE subgroup_base IS NOT NULL").fetchall()]
        self.types = [r[0] for r in c.execute("SELECT DISTINCT product_type FROM sku WHERE product_type IS NOT NULL").fetchall()]
        self.groups = [r[0] for r in c.execute("SELECT DISTINCT group_name FROM sku WHERE group_name IS NOT NULL").fetchall()]
        geo = c.execute("SELECT DISTINCT district, region, city FROM store").fetchall()
        self.districts = sorted({g[0] for g in geo if g[0] and g[0] != "Нет"})
        self.regions = sorted({g[1] for g in geo if g[1]})
        self.cities = sorted({g[2] for g in geo if g[2]})
        self.city_eff = sorted(r[0] for r in c.execute("SELECT DISTINCT city_eff FROM store_x").fetchall()
                               if r[0] and not r[0].startswith("город не указан"))
        bn = c.execute("SELECT DISTINCT banner, network FROM store WHERE banner IS NOT NULL").fetchall()
        self.banner_net = dict(bn)
        per_net = {}
        for b, n in bn:
            per_net.setdefault(n, []).append(b)
        self.net_banner = {n: bs[0] for n, bs in per_net.items() if len(bs) == 1}
        self._keys = {}
        self._build()

    # ——— key index ———

    def _key(self, alts, kind, val, flag=None, limit=16):
        """Key: pattern variants for each word; values of the same key accumulate (brand label variants)."""
        for combo in list(product(*alts))[:limit]:
            vals = self._keys.setdefault((combo, kind, flag), [])
            if val not in vals:
                vals.append(val)

    def _build(self):
        for kind, table in (("banner", BANNER_ALIASES), ("network", NETWORK_ALIASES), ("format", FORMAT_ALIASES)):
            for name, al in table.items():
                for a in al:
                    self._key(pats(a, "exact" if a in STORE_EXACT else "alias"), kind, name, STORE_CTX.get(a))
        for g, nets in NETWORK_GROUPS.items():
            for n in nets:
                self._key(pats(g, "exact"), "network", n)
        # manufacturers: name, name without legal form (CO., LTD), aliases
        self.man_index, man_key, man_rub = {}, {}, dict(self.manufacturers)
        for m, _ in self.manufacturers:
            if m == "не указан":
                continue
            for k in {norm(m).strip(), norm(re.sub(r"\b(CO|LTD|LLC|INC|GMBH|AG|CORP|CORPORATION)\b\.?", " ", m)).strip()}:
                if len(k) < 3 or k in STOP_BRANDS:
                    continue
                self.man_index[f" {k} "] = m
                man_key[k] = m
                self._key(pats(k, "exact"), "manufacturer", m, "risky" if k in WORD_BRANDS else None)
        for m, al in MANUFACTURER_ALIASES.items():
            if m in self.man_names:
                for a in al:
                    self.man_index[norm(a)] = m
                    self._key(pats(a, "exact" if re.search(r"[a-z]", a) else "decl"), "manufacturer", m)
        # brands: full name, part before the bracket, part in [..], name without (PL); by descending sales.
        # Brand named like a manufacturer: brand_tie is the brand's share of that manufacturer's sales (MIXTA ~100% —
        # take the brand, same numbers; TERRA BOREALIS ~45% — the manufacturer unless «бренд» is said)
        self.lines, self.brand_tie, self.man_alt = {}, {}, {}
        by_man = {}
        for b, mans, rub in sorted(self.brands, key=lambda x: -(x[2] or 0)):
            for m in mans.split(", "):
                if m and m != "не указан":
                    by_man.setdefault(m, []).append(b)
            for k in brand_keys(b):
                words = k.split()
                if not words or k in STOP_BRANDS or k.isdigit():
                    continue
                if k in man_key:
                    m = man_key[k]
                    self.brand_tie[b] = max(self.brand_tie.get(b, 0), self._pair_rub.get((b, m), 0) / (man_rub[m] or 1))
                    self.man_alt.setdefault(m, b)
                if len(k) < 3:
                    if not re.fullmatch(r"[a-z]{2}", k):
                        continue
                    flag = "upper"                       # ZU — upper case only
                elif k in WORD_BRANDS or re.fullmatch(r"\d+ [а-я].*(?<! pl)", k):
                    flag = "risky"                       # «лидер», «365 дней» (but not «365 ДНЕЙ (PL)»)
                elif k in BRAND_OVER_STORE:
                    flag = "default"
                else:
                    flag = None
                self._key(pats(k), "brand", b, flag)
        for b, al in BRAND_ALIASES.items():
            if b in self.brand_set:
                for a in al:
                    p = pats(a, "exact" if re.search(r"[a-z]", a) else "decl")
                    self._key(p, "brand", b)
                    if b in self.man_names:             # «Терра Бореалис» — same as Latin TERRA BOREALIS
                        self._key(p, "manufacturer", b)
        # «Vector»: other brands of the same manufacturer with the same prefix (product decision on brand lines)
        for bs in by_man.values():
            for b in bs:
                for x in bs:
                    if x.startswith(b + " ") and x not in self.lines.setdefault(b, []):
                        self.lines[b].append(x)
        self.lines = {b: v for b, v in self.lines.items() if v}
        # geography
        for d in self.districts:
            self._key(pats(d, "exact"), "district", d)
            if d in DISTRICT_NAMES:
                adj = pats(DISTRICT_NAMES[d])
                self._key(adj + pats("федеральный округ"), "district", d)
                self._key(adj + pats("фо", "exact"), "district", d)
            for a in DISTRICT_ALIASES.get(d, []):
                self._key(pats(a), "district", d)
        region_set = set(self.regions)
        for r in self.regions:
            m = re.match(r"^(.*?)\s+(обл|край|АО|Аобл|г)\.?$", r)
            base, typ = (m.group(1), m.group(2)) if m else (r, None)
            bw = norm(base).split()
            if typ in REGION_TYPES and len(bw) == 1:
                for t in REGION_TYPES[typ]:
                    self._key(pats(base) + t, "region", r)
            else:
                self._key(pats(base), "region", r)
                if typ is None:                     # «Республика Татарстан», «Саха (Якутия)»
                    self._key(pats("республика") + pats(base), "region", r)
                    if "(" in base:
                        self._key(pats(re.sub(r"\(.*?\)", "", base)), "region", r)
        for r, al in REGION_ALIASES.items():
            if r in region_set:
                for a in al:
                    self._key(pats(a, "exact" if a.isupper() else "decl"), "region", r, "upper" if a.isupper() else None)
        city_keys = {}
        for c in self.city_eff:
            if c not in CITY_REGIONS:
                city_keys[norm(re.sub(r"\s+г\.?$", "", c)).strip()] = c
        base = dict(city_keys)
        for name, al in CITY_ALIASES.items():
            for a in al:
                if name in base:
                    city_keys[a] = base[name]
        for k, c in city_keys.items():
            w = k.split()
            risky = len(w) == 1 and (len(w[0]) <= 5 or w[0].endswith(ADJ_TAIL))
            self._key(pats(k), "city", c, "short" if risky else None)
        # products: keywords, residual and absent ones, exact names of types, subgroups and groups
        for kw, key, val in PRODUCT_KEYWORDS:
            self._key(pats(kw, "kw"), "kw", (key, val))
        for kw, key, vals in PRODUCT_MULTI:
            for v in vals:
                self._key(pats(kw, "kw"), "kw", (key, v))
        for kw in PRODUCT_NONE:
            self._key(pats(kw, "kw"), "none", kw)
        for kind, pool in (("product_type", self.types), ("subgroup", self.subgroups), ("category_group", self.groups)):
            for v in pool:
                name = str(v).strip()
                if not name or name in RESIDUAL_TYPES or "/" in name or "_" in name or name in ("Гигиена", "nan"):
                    continue
                if kind == "category_group" and len(name.split()) < 2:
                    continue
                self._key(pats(name), kind, v)
                w = name.split()
                if len(w) == 2 and (w[0].lower().endswith(ADJ_TAIL) or w[1].lower().endswith(ADJ_TAIL)):
                    self._key(pats(f"{w[1]} {w[0]}"), kind, v)       # «влажные салфетки», «жидкое мыло» (reversed word order)
        # index by the first two letters of the first stem: (word patterns, kind, values, flag)
        self.index = {}
        for (combo, kind, flag), vals in self._keys.items():
            self.index.setdefault(combo[0][0][:2], []).append((combo, kind, tuple(vals), flag))
        self._keys = None

    # ——— matching ———

    def _match(self, tx):
        W, n = tx.w, len(tx.w)
        cands = []
        for i in range(n):
            if tx.used[i]:
                continue
            for e in self.index.get(W[i][:2], ()):
                combo = e[0]
                k = len(combo)
                if i + k <= n and all(not tx.used[i + j] and _wm(W[i + j], combo[j]) for j in range(k)):
                    cands.append((i, i + k, e))
        store_spans = {(i, j) for i, j, e in cands if e[1] in ("network", "banner", "format")}

        def prio(c):
            i, j, (_, kind, vals, flag) = c
            p = KIND_PRIO[kind]
            if kind == "brand" and tx.brand_ctx(i):
                p = 0                                   # «бренд Маяк», «марка Лидер»
            elif kind == "manufacturer" and tx.man_ctx(i):
                p = 0
            elif kind == "brand" and self.brand_tie.get(vals[0], 0) >= 0.9:
                p = 6.5                                 # brand = almost all of the same-named manufacturer
            return -(j - i), p, i

        out = []
        for i, j, e in sorted(cands, key=prio):
            if any(tx.used[i:j]):
                continue
            r = self._accept(i, j, e, tx, store_spans)
            if r is None:
                continue
            for x in range(i, j):
                tx.used[x] = True
                tx.kind[x] = r[0]
            out.append((i, j) + r)
        return sorted(out)

    @staticmethod
    def _looks_named(i, j, tx):
        """A common-word brand is written like a name: Cyrillic — UPPER case or capitalised not at sentence start;
        Latin — UPPER case, but not an analysis term (ABC, SKU) and not a continuation of another brand («OLVIA MEN»)."""
        words = [x for x in range(i, j) if re.search(r"[а-яa-z]", tx.w[x])]
        if not words:
            return False
        if re.search(r"[а-я]", tx.w[words[0]]):
            return all(tx.upper(x) for x in words) or tx.cap_mid(words[0])
        return (all(tx.upper(x) for x in words) and tx.w[i] not in TERMS
                and not (i and tx.kind[i - 1] in ("brand", "manufacturer")))

    def _accept(self, i, j, e, tx, store_spans):
        """Context check. Returns (kind, [values], note) or None."""
        combo, kind, vals, flag = e
        vals = list(vals)
        p, nx = tx.prev(i), tx.nxt(j)
        near = p in PREP_LOC or p.startswith(("сет", "магазин", "баннер", "формат"))
        if kind in ("network", "banner", "format"):
            if flag == "cap" and not (near or tx.cap_mid(i)):
                return None
            if flag == "ok" and not (tx.apos[i] or near or (tx.cap_mid(i) and not tx.comma[j - 1])):
                return None
            if flag == "upper" and not (tx.upper(i) and (near or not tx.first[i])):
                return None
            if flag == "near_upper" and not (tx.upper(i) and near):
                return None
            if flag == "banner" and not (p in ("в", "во", "из", "по") or p.startswith(("магазин", "баннер", "формат"))):
                return None
            if flag == "five" and not ((near or tx.cap_mid(i)) and p != "топ" and not re.match(
                    r"(лидер|бренд|марк|компани|производ|sku|ску|позиц|товар|игрок|крупн|лучш|сет)", nx)):
                return None
            if kind == "network" and tx.stm_ctx(i):     # «СТМ Маяка» — the network's private labels
                stm = sorted({f"STM({NET_STM[v]})" for v in vals if v in NET_STM})
                if stm and all(s in self.man_names for s in stm):
                    return "manufacturer", stm, None
            return kind, vals, None
        if kind in ("brand", "manufacturer"):
            sig = tx.brand_ctx(i) or (kind == "manufacturer" and tx.man_ctx(i))
            if (i, j) in store_spans and not (sig or flag == "default"):
                return None         # «Маяк» without «бренд» is the network; «СТМ Маяка» is the manufacturer STM(GRANIT)
            sig = sig or tx.stm_ctx(i)
            if flag == "upper" and not tx.upper(i):
                return None
            if flag == "risky" and not (sig or any(tx.quoted[i:j]) or self._looks_named(i, j, tx)):
                return None
            alt = self.man_alt.get(vals[0]) if kind == "manufacturer" else None
            note = f"есть также бренд {alt}" if alt and self.brand_tie.get(alt, 0) < 0.9 else None
            return kind, vals, note
        if kind == "city":
            if p in ("г", "гор", "город", "города", "городе", "городу", "городом"):
                return kind, vals, None
            if flag == "short" and not tx.cap_mid(i):
                return None
            if re.match(r"(район|округ|област|кра[йею]|улиц|проспект|микрорайон|поселк|сел[оа])", nx):
                return None                             # «в Советском районе» is not the city Советск
            if p in ("в", "во", "о", "об"):
                for x in range(i, j):
                    stem, ends = combo[x - i]
                    rest = tx.w[x][len(stem):]
                    if ends is not EXACT and ends != ADJ and rest not in PREP_CASE:
                        return None                     # «в Советском» is an adjective, not «в Советске»
            return kind, vals, None
        if kind in ("region", "district") and flag == "upper" and not tx.upper(i):
            return None
        if kind == "kw":                                # values are (filter key, value) pairs
            return vals[0][0], [v for _, v in vals], None
        if kind == "none":
            return kind, vals, f"нет в данных: {' '.join(tx.w[i:j])}"
        return kind, vals, None

    # ——— question parsing ———

    def find(self, text):
        f = Found()
        tx = _Text(text)
        pr = _periods(tx.low)
        f.periods, f.period_kind, f.future_month = pr["periods"], pr["kind"], pr["future"]
        f.out_of_range, f.yoy = pr["out"], pr["yoy"]
        tx.mask(pr["spans"])
        low = tx.low
        ms = _measures(low)
        if len(ms) > 1:
            f.measures = ms
        elif ms:
            f.measure = ms[0]
        f.portal = bool(PORTAL_RE.search(low))
        f.sort = _sort(low)
        f.followup = bool(FOLLOW_RE.search(tx.line))
        found = self._match(tx)
        start = {m[0]: k for k, m in enumerate(found)}
        ex, ex_tok, flags = _exclusions(tx, found, start)
        focus_k = _scan_after(tx, found, start, FOCUS_RE, OBJECT_KINDS, FOCUS_FILL, need_prev={"уровне": "на"})
        role_k = _role(tx, found, start)
        dims, level, share = _dims(tx, found, start)
        f.dims = dims
        # a single value of a breakdown level is a reference point, not a filter («какие сети растут быстрее Гранита»);
        # categories («какие типы шампуней») stay filters
        counts = {}
        for k, (i, j, kind, vals, note) in enumerate(found):
            if kind in dims and k not in ex and kind in OBJECT_KINDS + STORE_KINDS:
                counts[kind] = counts.get(kind, 0) + len(vals)
        for k, (i, j, kind, vals, note) in enumerate(found):
            if note and note not in f.notes:
                f.notes.append(note)
            if kind == "none":
                continue
            if k in ex:
                for v in vals:
                    f.add(kind, v, f.exclude)
            elif k == focus_k:
                f.focus = vals[0]
                f.add(kind, vals[0], f.mentioned)
            elif k == role_k:
                f.role = vals[0]
            elif k in share or (kind in dims and counts.get(kind, 0) == 1):
                f.dims.add(kind)
                for v in vals:
                    f.add(kind, v, f.mentioned)
            else:
                for v in vals:
                    f.add(kind, v)
        f.level_hint = level or (next((found[k][2] for k in sorted(share)), None)) or \
            (found[focus_k][2] if focus_k is not None else None)
        self._mixed_stores(f)
        self._soap(f, tx)
        # kids: words outside names («Детский остров») — after matching; «без детских» is an exclusion
        kid = [t for t in range(len(tx.w)) if not tx.used[t] and _kid_word(tx, t)]
        adult = [t for t in range(len(tx.w)) if not tx.used[t] and tx.w[t].startswith("взросл")]
        k_in = any(t not in ex_tok for t in kid)
        a_in = any(t not in ex_tok for t in adult)
        if k_in and a_in:
            f.kids_both = True
        elif "kids" in flags and not k_in:
            f.kids = False
        elif "adults" in flags and not a_in:
            f.kids = True
        elif k_in:
            f.kids = True
        elif a_in:
            f.kids = False
        f.exclude_pl = "pl" in flags
        for b in f.filters.get("brand", []):
            if b in self.lines:
                f.notes.append(f"есть также: {', '.join(self.lines[b])}")
        return f

    def _mixed_stores(self, f):
        """Network and banner together: «Маяк и Маяк Косметик» — network Гранит broken down by banner;
        «Семерочка и Маяк у дома» — all as banners (otherwise the intersection is empty)."""
        nets, bans = f.filters.get("network", []), f.filters.get("banner", [])
        if not (nets and bans):
            return
        own = [b for b in bans if self.banner_net.get(b) in nets]
        for b in own:
            bans.remove(b)
            f.add("banner", b, f.mentioned)
        if own:
            f.dims.add("banner")
            f.level_hint = f.level_hint or "banner"
        other = [n for n in nets if n not in {self.banner_net.get(b) for b in own}]
        if bans and other and all(n in self.net_banner for n in other):
            for n in other:
                nets.remove(n)
                bans.append(self.net_banner[n])
            f.level_hint = f.level_hint or "banner"
        for k in ("network", "banner"):
            if not f.filters.get(k):
                f.filters.pop(k, None)

    @staticmethod
    def _soap(f, tx):
        """«жидкого, кускового и хозяйственного мыла»: soap kinds listed with commas."""
        if not any(w.startswith("мыл") for w in tx.w):
            return
        kinds = [(st, v) for st, v in (("жидк", "Мыло Жидкое"), ("кусков", "Мыло кусковое"),
                                       ("хозяйствен", "Мыло хозяйственное"))
                 if any(not tx.used[t] and w.startswith(st) for t, w in enumerate(tx.w))]
        for _, v in kinds:
            f.add("product_type", v)
        if kinds and "Мыло" in f.filters.get("subgroup", []):
            f.filters["subgroup"].remove("Мыло")
            if not f.filters["subgroup"]:
                f.filters.pop("subgroup")


def brand_keys(b):
    """Brand lookup keys: full name, part before the bracket, part in [..], name without (PL).
    «СВОНЛИ [SWANLY]» is found by SWANLY and СВОНЛИ, «СИНЯЯ ЦЕНА (PL)» by «Синяя цена»."""
    s = str(b)
    out = {norm(s).strip()}
    m = re.match(r"^(.*?)\s*[\[(]", s)
    if m:
        out.add(norm(m.group(1)).strip())
    out.update(norm(x).strip() for x in re.findall(r"\[([^\]]+)\]", s))
    out.add(norm(re.sub(r"\(PL\)", " ", s)).strip())
    return {k for k in out if k}


def _kid_word(tx, t):
    w = tx.w[t]
    return w.startswith(("детск", "ребен", "малыш")) or w in ("детей", "детям", "детьми")


EX_FILL = re.compile(r"(учет|производител|бренд|компани|баннер|формат|регион|город|област|категори|подгрупп|магазин|"
                     r"продаж|товар|позици)\w*|сет(ь|и|ей|ям|ях)|марк(а|и|ой|е|ам|ах|ок)|sku|ску|тип(а|ов|ы)?|"
                     r"вид(а|ов|ы)?|все|всех|самого|самой|самих")
JOIN = {"и", "или", "а", "также", "плюс", "либо"}


def _exclusions(tx, found, start):
    """«кроме Гранита и Нивы», «без Маяк Косметик», «за исключением NORD», «без детских», «без СТМ»:
    entities right after the marker (joined by «и», commas, «сетей», «учета») are exclusions, not filters."""
    W, n = tx.w, len(tx.w)
    ex, toks, flags = set(), set(), set()
    for t, w in enumerate(W):
        if not (w in ("кроме", "без", "исключая", "исключив") or (w == "исключением" and tx.prev(t) == "за")
                or (w == "считая" and tx.prev(t) == "не")):
            continue
        u = t + 1
        while u < n:
            if u in start:
                k = start[u]
                ex.add(k)
                toks.update(range(found[k][0], found[k][1]))
                u = found[k][1]
                continue
            x = W[u]
            if _kid_word(tx, u):
                flags.add("kids")
                toks.add(u)
                break
            if x.startswith("взросл"):
                flags.add("adults")
                toks.add(u)
                break
            if x in STM_WORDS or x.startswith("собствен"):
                if u + 1 not in start or found[start[u + 1]][2] != "manufacturer":
                    flags.add("pl")
                u += 1
                continue
            if x in JOIN or EX_FILL.fullmatch(x):
                u += 1
                continue
            break
    return ex, toks, flags


FOCUS_RE = re.compile(r"(сопостав|сравним|сосед|похож|близк|рядом|уровне|уровня)")
FOCUS_FILL = {"с", "со", "на", "к", "ко", "по", "у", "рейтингу", "продажам", "объему", "размеру", "выручке",
              "масштабу", "доле"}


def _scan_after(tx, found, start, rx, kinds, fill, need_prev=None):
    """First entity of the wanted kind right after a marker word («на уровне, сопоставимом с NORD» -> NORD)."""
    W, n = tx.w, len(tx.w)
    for t, w in enumerate(W):
        if not rx.match(w) or (need_prev and w in need_prev and tx.prev(t) != need_prev[w]):
            continue
        for u in range(t + 1, min(t + 6, n)):
            if u in start:
                k = start[u]
                if found[k][2] in kinds:
                    return k
                break
            if not (W[u] in fill or rx.match(W[u])):
                break
    return None


ROLE_RE = re.compile(r"кам|kam|из|в|как|по|(менеджер|аналитик|категорийн|сотрудник|представител|работа|руководител|"
                     r"директор|специалист|трейд|маркетолог|ключев|аккаунт|отдел|компани|продаж)\w*")


def _role(tx, found, start):
    """«Я КАМ NORD», «мы — NORD»: a manufacturer after a role is who is asking, not a filter."""
    W, n = tx.w, len(tx.w)
    for t, w in enumerate(W):
        if w not in ("я", "мы"):
            continue
        for u in range(t + 1, min(t + 7, n)):
            if u in start:
                k = start[u]
                return k if found[k][2] == "manufacturer" else None
            if not ROLE_RE.fullmatch(W[u]):
                break
    return None


LEVEL_WORDS = {}
for _lv, _ws in {
    "sku": "sku ску позиция позиции позиций позициям позициях артикул артикулы артикулов артикулам товар товары "
           "товаров товарам товарах наименований",
    "brand": "бренд бренды брендов брендам брендах брендами марки марок маркам марках тм",
    "manufacturer": "производитель производители производителей производителям производителях компания компании "
                    "компаний компаниям компаниях поставщики поставщиков поставщикам игроки игроков игрокам",
    "network": "сеть сети сетей сетям сетях сетями ритейлеры ритейлеров ритейлерам",
    "banner": "баннер баннеры баннеров баннерам баннерах",
    "format": "формат форматы форматов форматам форматах",
    "region": "регион регионы регионов регионам регионах областям областях субъекты субъектов субъектам",
    "city": "город города городов городам городах",
    "district": "округ округа округов округам округах",
    "subgroup": "подгруппа подгруппы подгрупп подгруппам подгруппах категория категории категорий категориям категориях",
    "product_type": "тип типы типов типам типах вид виды видов видам видах",
}.items():
    for _w in _ws.split():
        LEVEL_WORDS[_w] = _lv
RANK_TRIG = {"топ", "рейтинг", "какие", "каких", "какой", "какая", "каким", "какому", "какую", "каком", "кто",
             "назови", "перечисли", "покажи", "выведи", "найди", "лидеры", "крупнейшие", "крупнейших", "растущие",
             "растущих", "падающие", "падающих", "быстрорастущие"}
DIM_TRIG = {"по", "среди", "между", "разрезе", "разбивке", "разбивкой", "каждой", "каждому", "каждого", "каждая",
            "каждый", "каждом", "каждую"}
DIM_SKIP = re.compile(r"(\d+|из|все[мх]?|его|ее|их|именно|сво[ие]\w*|(тройк|семерк|десятк|двадцатк)\w*|"
                      r"\w+(ым|им|ых|их|ые|ие|ой|ей|ую))$")
SHARE_WORDS = {"доля", "доли", "долю", "долей", "доле"}


def _dims(tx, found, start):
    """Breakdowns and the main ranking level: «в каких сетях», «по брендам», «топ-10 SKU», «3–4 компании»;
    «доля Гранита в продажах NORD» — the network as the share object (a breakdown, not a filter)."""
    W, n = tx.w, len(tx.w)
    dims, rank, dim_lv, share = set(), None, None, set()
    for t, w in enumerate(W):
        is_rank = w in RANK_TRIG or (w.isdigit() and int(w) <= 100 and not tx.used[t])   # «топ-10», «3 SKU»
        if not (is_rank or w in DIM_TRIG):
            continue
        u = t + 1
        while u < n and u - t <= 3 and DIM_SKIP.match(W[u]) and W[u] not in LEVEL_WORDS:
            u += 1
        lv = LEVEL_WORDS.get(W[u]) if u < n else None
        if not lv or (w.isdigit() and u == t + 1 and W[u] in ("сети", "сеть")):
            continue
        if u + 1 in start and found[start[u + 1]][2] == lv:
            continue                                    # «по сети Гранит», «в сети Гранит» — a filter, not a breakdown
        dims.add(lv)
        if is_rank and rank is None:
            rank = lv
        dim_lv = dim_lv or lv
    for k, (i, j, kind, vals, note) in enumerate(found):
        if kind not in STORE_KINDS:
            continue
        p = i - 1
        while p >= 0 and LEVEL_WORDS.get(W[p]) == kind:
            p -= 1
        if p >= 0 and W[p] in SHARE_WORDS:
            share.add(k)
        elif k - 1 in share and found[k - 1][2] == kind and all(W[x] in JOIN for x in range(found[k - 1][1], i)):
            share.add(k)                                # «доля Гранита и Семерочки в ...»
    for k in share:
        dims.add(found[k][2])
    return dims, rank or dim_lv, share


# ——— Measure, reference portal, sorting, follow-up ———

MEASURE_RX = [
    ("qty", r"в\s+штуках|штук|\bшт\b|в\s+упаковках|упаковок|в\s+упак\b|в\s+натуральном|натуральн[а-я]*\s+(?:выражени|единиц)"),
    ("kg", r"в\s+литрах|литр|в\s+кг\b|килограмм|в\s+тоннах|тонн"),
    ("rub", r"в\s+рублях|\bруб|выручк|в\s+деньгах|деньгах|в\s+денежном|денежн[а-я]*\s+выражени"),
]
PER_UNIT = re.compile(r"(?:за|на)\s+(?:одну\s+|1\s+)?(?:штук[уа]|шт\b\.?|единиц[уа]|упаковк[уа])|руб[а-я]*\.?\s*/\s*шт\.?")


def _measures(low):
    """Measures in order of mention; «цена за штуку» (price per unit) is not a measure."""
    s = PER_UNIT.sub(" ", low)
    hits = sorted((m.start(), name) for name, rx in MEASURE_RX for m in [re.search(rx, s)] if m)
    return [name for _, name in hits]


PORTAL_RE = re.compile(r"категори[яиюей]\s+[«\"]?косметик|косметика\s+и\s+гигиена|как\s+на\s+портале|портал|\bcosm\b")
SORT_RX = [
    ("share_change", r"изменени[а-я]*\s+дол|дол[а-я]*\s+изменени|прирост[а-я]*\s+дол|рост[а-я]*\s+дол[еи]|"
                     r"набира[а-я]*\s+дол|теря[а-я]*\s+дол|динамик[а-я]*\s+дол"),
    ("gain", r"по\s+(?:абсолютному\s+)?приросту|наибольш[а-я]*\s+(?:абсолютн[а-я]*\s+)?прирост|"
             r"больше\s+всего\s+прибав|лидер[а-я]*\s+прироста"),
    ("decline", r"по\s+падению|по\s+снижению|падающ[а-я]*(?!\s+рын)|(?:пада|упал|снизил|снижа|сократил|сокраща)[а-я]*\s+"
                r"(?:сильнее|больше)\s+всего|(?:сильнее|больше)\s+всего\s+(?:пада|упал|снизил|снижа|сократил|сокраща|"
                r"потерял)|наибольш[а-я]*\s+(?:падени|снижени)|аутсайдер"),
    ("growth", r"по\s+росту|по\s+темп[а-я]*\s+роста|растущ[а-я]*(?!\s+рын)|быстрорастущ|(?:быстрее|сильнее)\s+(?:всех|всего)"
               r"\s+раст|раст[а-я]*\s+(?:быстрее|сильнее)\s+(?:всех|всего)|лидер[а-я]*\s+роста|наибольш[а-я]*\s+(?:рост|темп)|"
               r"максимальн[а-я]*\s+рост"),
]


def _sort(low):
    for name, rx in SORT_RX:
        if re.search(rx, low):
            return name
    return None


FOLLOW_RE = re.compile(r"^ (?:(?:ок|ладно|хорошо|ладно|отлично|понятно|спасибо|ясно|супер) )?(?:а|и|теперь) |"
                       r" (?:эти|этих|этим|этими|их|там|у них|по ним|из них|какие именно|какой именно|какая именно|"
                       r"то же|тот же|та же|те же|том же|аналогично)(?= )")


# ——— Periods ———

MONTH_STEMS = ["янв", "фев", "мар", "апр", "ма", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
B = r"(?<![0-9a-zа-я])"
E = r"(?![0-9a-zа-я])"
MON = (r"(январ[а-я]*|янв|феврал[а-я]*|фев|март[а-я]*|мар|апрел[а-я]*|апр|ма[йяею]|июн[а-я]*|июл[а-я]*|"
       r"август[а-я]*|авг|сентябр[а-я]*|сент?|октябр[а-я]*|окт|ноябр[а-я]*|нояб?|декабр[а-я]*|дек)" + E + r"\.?")
YR = r"(20\d\d)(?!\d)"
NUMW = {"один": 1, "одного": 1, "два": 2, "двух": 2, "две": 2, "три": 3, "трех": 3, "четыре": 4, "четырех": 4,
        "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7, "восемь": 8, "восьми": 8,
        "девять": 9, "девяти": 9, "десять": 10, "десяти": 10, "одиннадцать": 11, "одиннадцати": 11,
        "двенадцать": 12, "двенадцати": 12}
NUM = r"(\d{1,2}|" + "|".join(sorted(NUMW, key=len, reverse=True)) + ")"
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
FORECAST_RE = re.compile(r"прогноз|спрогноз|будет|будут|ожида|план|продаст|продадут|предскаж|предсказ|forecast")
SEASONS = {"зим": (12, 2), "весн": (3, 5), "весен": (3, 5), "лет": (6, 8), "осен": (9, 11)}
SEASON_PREV = {"за", "этой", "этим", "этот", "эту", "прошлой", "прошлым", "прошлое", "прошлую", "последней",
               "последним", "последнее", "последнюю"}
YOY_RE = re.compile(
    r"(?:к|по\s+сравнению\s+с|в\s+сравнении\s+с|относительно|против|vs\.?|versus)\s+(?:прошл|предыдущ)[а-я]*\s+год[а-я]*|"
    r"с\s+(?:прошлым|предыдущим)\s+годом|год\s+к\s+году|г\s*/\s*г|\bг\s*к\s*г\b|\byoy\b|аппг|"
    r"(?:к|с|против|относительно)\s+(?:аналогичн[а-я]*|тем\s+же|тому\s+же|соответствующ[а-я]*)\s+период[а-я]*"
    r"(?:\s+(?:прошл|предыдущ)[а-я]*\s+год[а-я]*)?")


def month_no(word):
    w = word.lower()
    if w.startswith("ма") and not w.startswith("мар"):
        return 5
    for i, st in enumerate(MONTH_STEMS):
        if w.startswith(st):
            return i + 1
    return None


def _num(s):
    return int(s) if s.isdigit() else NUMW[s]


def _periods(low):
    """Question periods. Returns a dict: periods (in order of mention), kind, future, out (outside data), yoy, spans."""
    lo, hi = S.data_bounds()
    fc = bool(FORECAST_RE.search(low))
    found, spans, out = [], [], []
    res = {"kind": None, "future": None, "yoy": False}
    P = S.Period

    def free(m):
        return not any(m.start() < y and x < m.end() for x, y in spans)

    def put(m, p, month=False):
        """Period from text: a month after the data is a forecast (if asked) or "outside data"; clipped to the data."""
        spans.append((m.start(), m.end()))
        if p.start > hi:
            if fc and month:
                res["future"] = p.end
            elif not fc:                                # quarterly forecast — left to the model, there is no data
                out.append(p.label())
            return
        cp = S.clip(p)
        if cp is None:
            out.append(p.label())
            return
        if p.start < lo:
            out.append(P(p.start, S.add_months(lo, -1)).label())
        found.append((m.start(), cp))

    def latest(build):
        """No year given: the latest year for which the whole period is in the data."""
        for y in range(hi.year, lo.year - 1, -1):
            if build(y).end <= hi:
                return build(y)
        return build(hi.year)

    def span(y, a, b):                    # months a..b of year y; a > b starts in the previous year («с октября по март»)
        return P(date(y - (a > b), a, 1), date(y, b, 1))

    def iter_(rx):
        for m in re.finditer(rx, low):
            if free(m):
                yield m

    for m in iter_(rf"{B}(20\d\d)-(0[1-9]|1[0-2]){E}\s*(?:\.\.|[-–—]|по|до)\s*(20\d\d)-(0[1-9]|1[0-2]){E}"):
        put(m, P(date(int(m[1]), int(m[2]), 1), date(int(m[3]), int(m[4]), 1)))
    for m in iter_(rf"{B}(0?[1-9]|1[0-2])\.(20\d\d){E}\s*(?:[-–—]|по|до)\s*(0?[1-9]|1[0-2])\.(20\d\d){E}"):
        put(m, P(date(int(m[2]), int(m[1]), 1), date(int(m[4]), int(m[3]), 1)))
    for m in iter_(rf"{B}(20\d\d)-(0[1-9]|1[0-2]){E}|{B}(0?[1-9]|1[0-2])\.(20\d\d){E}"):
        y, mn = (int(m[1]), int(m[2])) if m[1] else (int(m[4]), int(m[3]))
        put(m, P(date(y, mn, 1), date(y, mn, 1)), month=True)
    # month range, possibly across years: «январь–август 2026», «с октября 2025 по март 2026»
    for m in iter_(rf"{B}(?:(?:с|со)\s+)?{MON}\s*(?:{YR}\s*(?:г\.?|года?)?)?\s*(?:[-–—]|по|до)\s*{MON}\s*(?:{YR})?"):
        a, b = month_no(m[1]), month_no(m[3])
        y1, y2 = (int(m[2]) if m[2] else None), (int(m[4]) if m[4] else None)
        if y1 and y2:
            p = P(date(y1, a, 1), date(y2, b, 1))
        elif y2 or y1:
            p = span(y2 or y1, a, b) if y2 else P(date(y1, a, 1), date(y1 + (a > b), b, 1))
        else:
            p = latest(lambda y: span(y, a, b))
        put(m, p, month=a == b)
    # quarters: «1 кв», «I квартал», «первый квартал», «в 1-м квартале», «Q1 2026», «1Q 2026»
    qrx = (rf"{B}(?:([1-4])|(iv|i{{1,3}})|(перв|втор|трет|четверт)[а-я]*)\s*(?:-?(?:й|ый|ий|ой|го|м|ом|ем))?\s*"
           rf"(?:кв\.?|квартал[а-я]*){E}\s*(?:{YR})?")
    for m in iter_(qrx):
        q = int(m[1]) if m[1] else (ROMAN[m[2]] if m[2] else ["перв", "втор", "трет", "четверт"].index(m[3]) + 1)
        build = (lambda y, q=q: P(date(y, 3 * q - 2, 1), date(y, 3 * q, 1)))
        put(m, build(int(m[4])) if m[4] else latest(build))
    for m in iter_(rf"{B}(?:q([1-4])|([1-4])\s*q){E}\s*(?:{YR})?"):
        q = int(m[1] or m[2])
        build = (lambda y, q=q: P(date(y, 3 * q - 2, 1), date(y, 3 * q, 1)))
        put(m, build(int(m[3])) if m[3] else latest(build))
    # half-years: «первое полугодие 2026», «в 1-м полугодии», «H1 2026»
    for m in iter_(rf"{B}(?:([12])|(перв|втор)[а-я]*)\s*(?:-?(?:е|ое|ом|м|го|й))?\s*(?:полугоди[а-я]*|пг){E}\s*(?:{YR})?|"
                   rf"{B}h([12]){E}\s*(?:{YR})?"):
        h = 1 if (m[1] == "1" or m[2] == "перв" or m[4] == "1") else 2
        yy = m[3] or m[5]
        build = (lambda y, h=h: P(date(y, 1 if h == 1 else 7, 1), date(y, 6 if h == 1 else 12, 1)))
        put(m, build(int(yy)) if yy else latest(build))
    # seasons: «зима 2025» = Dec 2024 – Feb 2025; without a year only «зимой», «за лето», «летний сезон», «этой весной»
    srx = (rf"{B}(зим[а-я]*|весн[а-я]*|весен[а-я]*|лет(?:о|ом|а|у|ний|нем|него|них|ние|няя|нюю|ней)|осен[а-я]*){E}"
           rf"(?:\s+сезон[а-я]*)?(?:\s*{YR}(?:\s*[-–—/]\s*{YR})?)?")
    for m in iter_(srx):
        word = m[1]
        a, b = next(v for k, v in SEASONS.items() if word.startswith(k))
        if m[2]:                                        # «зима 2025–2026» = Dec 2025 – Feb 2026
            put(m, span(int(m[3] or m[2]), a, b))
            continue
        ly = re.match(rf"(?:\s*(?:,|и|или)\s*(?:зим|весн|лет|осен)[а-я]*)+\s*{YR}", low[m.end():])
        if ly:                                          # «весну и лето 2025»
            put(m, span(int(ly[1]), a, b))
            continue
        prevw = (low[:m.start()].split()[-1:] or [""])[0]
        adj = re.match(r"(зимн|весенн|летн|осенн)", word) and (
            "сезон" in m[0] or re.match(r"\s*(месяц|период)", low[m.end():]))
        if word in ("зимой", "весной", "летом", "осенью") or adj or prevw in SEASON_PREV:
            put(m, latest(lambda y: span(y, a, b)))
    # last N months (with a year — the last N months of that year), last quarter, half-year
    for m in iter_(rf"{B}последни[еймх]\s+{NUM}\s+(?:полных\s+)?месяц[а-я]*(?:\s+{YR}(?:\s*(?:г\.?|года?))?)?"):
        k = _num(m[1])
        end = min(date(int(m[2]), 12, 1), hi) if m[2] else hi
        if k == 12 and not m[2]:
            res["kind"] = "mat"
            spans.append((m.start(), m.end()))
        elif 1 <= k <= 24:
            put(m, P(S.add_months(end, -k + 1), end))
    for m in iter_(rf"{B}(?:за\s+)?последн[а-я]*\s+квартал[а-я]*"):
        put(m, P(S.add_months(hi, -2), hi))
    for m in iter_(rf"{B}(?:за\s+)?(?:последн[а-я]*\s+)?(?:полгода|полугодие|полугодия)(?![а-я])"):
        if "последн" in m[0] or "полгода" in m[0]:
            put(m, P(S.add_months(hi, -5), hi))
    for m in iter_(rf"{B}(?:прошл|предыдущ)[а-я]*\s+квартал[а-я]*"):
        q_end = S.add_months(hi, -(hi.month % 3))       # last complete calendar quarter
        put(m, P(S.add_months(q_end, -2), q_end))
    # «за 9 мес. 2026», «первые 6 месяцев 2026» — from January
    for m in iter_(rf"{B}(?:(?:за|первые)\s+)?{NUM}\s*(?:мес\.?|месяц[а-я]*){E}\s*{YR}"):
        k, y = _num(m[1]), int(m[2])
        if 1 <= k <= 12:
            put(m, P(date(y, 1, 1), date(y, k, 1)))
    for m in iter_(rf"{B}(?:за\s+|в\s+)?(?:прошл|предыдущ|последн)[а-я]*\s+месяц(?:е|а|у)?{E}"):
        put(m, P(hi, hi))
    for m in iter_(YOY_RE.pattern):
        res["yoy"] = True
        spans.append((m.start(), m.end()))
    for m in iter_(rf"{B}(?:за|в|во)\s+(поза)?(?:прошл|предыдущ)[а-я]*\s+год[а-я]*"):
        y = hi.year - (2 if m[1] else 1)
        put(m, P(date(y, 1, 1), date(y, 12, 1)))
    for rx, kind in ((r"\bytd\b|с\s+начала\s+(?:этого\s+|текущего\s+)?года|в\s+этом\s+году|за\s+этот\s+год|"
                      r"в\s+текущем\s+году|за\s+текущий\s+год|этого\s+года|текущего\s+года", "ytd"),
                     (r"\bmat\b|\bмат\b|скользящ[а-я]*\s+год|последн(?:ий|ие|их|ем)\s+год(?![а-я])|за\s+год(?![а-я])|"
                      r"12\s+месяц", "mat"),
                     (r"(?:за\s+)?(?:весь|все|всю)\s+(?:доступн[а-я]*\s+)?(?:период|время|месяцы|историю)(?![а-я])",
                      "all")):
        for m in iter_(rx):
            res["kind"] = kind
            spans.append((m.start(), m.end()))
    # years: «2024–2026» and «с 2024 по 2026» are ranges; «2025 и 2026» is a range, but two years with «сравни»
    for m in iter_(rf"{B}(?:с\s+)?{YR}\s*(?:г\.?\s*)?(?:[-–—]|по|до|и)\s*{YR}(?:\s*(?:гг\.?|годы|годах|года|год|г\.?))?"):
        y1, y2 = int(m[1]), int(m[2])
        sep = re.search(r"\s(и)\s", m[0])
        if y2 <= y1:
            continue
        if sep and re.search(r"сравн|\bvs\b|против", low):
            continue
        put(m, P(date(y1, 1, 1), date(y2, 12, 1)))
    # «с 2025 года», «начиная с марта 2026» — up to the end of the data
    for m in iter_(rf"{B}(?:начиная\s+)?(?:с|со)\s+(?:{MON}\s*)?{YR}\s*(?:г\.?|года?)?(?!\s*(?:[-–—]|по|до))|"
                   rf"{B}(?:начиная\s+)?(?:с|со)\s+{MON}(?!\s*(?:20\d\d\s*)?(?:[-–—]|по|до))"):
        if re.search(r"(сравнени[юи]|сопоставлени[юи]|вместе|наряду)\s*$", low[:m.start()]):
            continue                                    # «по сравнению с 2025» is a base, not «с 2025 по сей день»
        mn = month_no(m[1] or m[3]) if (m[1] or m[3]) else 1
        start = date(int(m[2]), mn, 1) if m[2] else latest(lambda y: S.Period(date(y, mn, 1), date(y, mn, 1))).start
        put(m, P(start, hi))
    # single months: the year comes right after the month or after a list («январь и июнь 2026»)
    for m in iter_(rf"{B}{MON}(?:\s*{YR})?"):
        mn = month_no(m[1])
        if m[2]:
            put(m, P(date(int(m[2]), mn, 1), date(int(m[2]), mn, 1)), month=True)
            continue
        ly = re.match(rf"(?:\s*(?:,|и|или|/)\s*{MON})+\s*{YR}", low[m.end():])
        if ly:
            y = int(ly[2])
            put(m, P(date(y, mn, 1), date(y, mn, 1)), month=True)
            continue
        prevw = (low[:m.start()].split()[-1:] or [""])[0]
        if fc and prevw not in ("за", "с", "со", "по", "до", "после") and (prevw == "на" or mn > hi.month):
            y = hi.year if mn > hi.month else hi.year + 1
            put(m, P(date(y, mn, 1), date(y, mn, 1)), month=True)
        else:
            put(m, latest(lambda y: P(date(y, mn, 1), date(y, mn, 1))), month=True)
    for m in iter_(rf"{B}{YR}(?![-–—.,]\d)(?!\s*(?:руб|р\.|₽|шт|млн|млрд|тыс|%|точ|sku|ску))(?:\s*(?:г\.?|гг\.?|год[а-я]*))?"):
        y = int(m[1])
        put(m, P(date(y, 1, 1), date(y, 12, 1)))
    found.sort(key=lambda x: x[0])
    return {"periods": [p for _, p in found], "kind": res["kind"], "future": res["future"], "out": out,
            "yoy": res["yoy"], "spans": spans}


def parse_periods(text):
    """Periods in order of mention. Returns (periods, kind, future_month)."""
    r = _periods(de_yo(text).lower())
    return r["periods"], r["kind"], r["future"]


_dict = None


def dictionary():
    global _dict
    if _dict is None:
        _dict = Dictionary()
    return _dict
