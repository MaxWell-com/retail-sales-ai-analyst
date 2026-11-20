"""Build the agent's DuckDB database (data/retail.duckdb) from data/parquet produced by scripts/make_synthetic.py.

Steps (--only runs a single one):
  dims   — sku and store dimensions with derived fields (Гранит banners, kids' subgroups, private label,
           comparable chains);
  fact   — fact table loaded month by month, sorted by sku_id within a month (zone maps for product filters);
  marts  — net_sku: month × SKU × chain × banner × format × district, with the number of stores with sales;
           store_sub: month × store × subgroup, for distribution denominators and SKUs per store;
  check  — sanity check: no empty months, no orphan keys.

Run: python scripts/build_db.py [--only dims|fact|marts|check]
Paths can be overridden with VT_DATA (input folder) and VT_DB (output file).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get("VT_DATA", ROOT / "data"))
DB = Path(os.environ.get("VT_DB", LOCAL / "retail.duckdb"))
OWN = os.environ.get("VT_OWN_MANUFACTURER", "NORD")
PORTAL_SUBGROUPS = (
    "^Гигиена волос и кожи головы", "^Гигиена волос и кожи головы Детская",
    "^Мыло", "^Мыло Детское",
    "^Солнцезащитные средства и уход", "^Солнцезащитные средства и уход Детский",
)
COMPARABLE = ("Гранит", "Семерочка Х7", "Нива (сеть)", "Перелесок ТД", "Ажур",
              "Депо Кэш Энд Керри", "ЛАД (сеть)", "Гиперкосмос", "Детский Остров (сеть)")
# Short chain names for answers; the data keeps the original names
NET_SHORT = {
    "Нива (сеть)": "Нива", "Детский Остров (сеть)": "Детский Остров", "ЛАД (сеть)": "ЛАД",
    "Маятник (сеть)": "Маятник", "Добрый (сеть)": "Добрый", "ПАРУС (сеть)": "ПАРУС",
    "НУ (сеть)": "НУ", "Камелия (сеть)": "Камелия", "Депо Кэш Энд Керри": "Депо",
    "Семерочка Х7": "Семерочка", "Перелесок ТД": "Перелесок",
}
# Гранит banners by the chain's own format code (format_src)
GRANIT_BANNER = {"МД": "Маяк у дома", "МК": "Маяк Косметик", "МА": "Маяк Аптека",
                 "Наша Цена": "Наша Цена", "ГМ": "Маяк гипермаркет"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sql_list(items):
    return ", ".join("'" + s.replace("'", "''") + "'" for s in items)


def build_dims(con):
    pq = (LOCAL / "parquet" / "dim").as_posix()
    log("dim sku")
    con.execute(f"""
        CREATE OR REPLACE TABLE sku AS
        SELECT *,
               ltrim(subgroup, '^') AS subgroup_name,
               ltrim(category_group, '^') AS group_name,
               subgroup ILIKE '%детск%' AS is_kids,
               regexp_replace(ltrim(subgroup, '^'), '\\s+Детск\\S*$', '') AS subgroup_base,
               subgroup IN ({sql_list(PORTAL_SUBGROUPS)}) AS in_portal,
               (manufacturer LIKE 'STM(%' OR brand LIKE '%(PL)%') AS is_pl,
               manufacturer = '{OWN}' AS is_own
        FROM read_parquet('{pq}/dim_sku.parquet')
        ORDER BY sku_id""")
    log("dim store")
    short = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in NET_SHORT.items())
    banner = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in GRANIT_BANNER.items())
    con.execute(f"""
        CREATE OR REPLACE TABLE store AS
        SELECT *,
               CASE network {short} ELSE network END AS net,
               CASE WHEN network = 'Гранит' THEN CASE format_src {banner} ELSE 'Маяк прочие' END
                    ELSE CASE network {short} ELSE network END END AS banner,
               network IN ({sql_list(COMPARABLE)}) AS comparable
        FROM read_parquet('{pq}/dim_store.parquet')
        ORDER BY store_id""")


def build_fact(con):
    root = LOCAL / "parquet" / "fact"
    months = sorted(p.name[6:] for p in root.glob("month=*"))
    log(f"fact: {len(months)} months")
    con.execute("""CREATE OR REPLACE TABLE fact (month DATE, store_id INTEGER, sku_id INTEGER,
                   qty DOUBLE, rub DOUBLE, kg DOUBLE, cost DOUBLE)""")
    for m in months:
        t = time.time()
        con.execute(f"""
            INSERT INTO fact
            SELECT DATE '{m}-01', store_id::INTEGER, sku_id::INTEGER, qty, rub, kg, cost
            FROM read_parquet('{(root / f"month={m}").as_posix()}/*.parquet')
            ORDER BY sku_id, store_id""")
        n = con.execute(f"SELECT count(*) FROM fact WHERE month = DATE '{m}-01'").fetchone()[0]
        log(f"  {m}: {n:,} rows, {time.time() - t:.0f} s")


def build_marts(con):
    log("mart net_sku")
    con.execute("""
        CREATE OR REPLACE TABLE net_sku AS
        SELECT f.month, f.sku_id, s.network, s.banner, s.format, s.district,
               sum(f.qty) AS qty, sum(f.rub) AS rub, sum(f.kg) AS kg, sum(f.cost) AS cost,
               count(*) FILTER (WHERE f.qty > 0) AS stores
        FROM fact f JOIN store s USING (store_id)
        GROUP BY ALL
        ORDER BY month, sku_id""")
    log("mart store_sub")
    con.execute("""
        CREATE OR REPLACE TABLE store_sub AS
        SELECT f.month, f.store_id, k.subgroup,
               sum(f.rub) AS rub, sum(f.qty) AS qty,
               count(*) FILTER (WHERE f.qty > 0) AS skus
        FROM fact f JOIN sku k USING (sku_id)
        GROUP BY ALL
        ORDER BY month, store_id""")
    for t in ("fact", "net_sku", "store_sub", "sku", "store"):
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        log(f"  {t}: {n:,} rows")


def check(con):
    """Sanity check: every month has rows and revenue, store and SKU keys all resolve."""
    rows = con.execute("SELECT strftime(month, '%Y-%m'), count(*), sum(rub) FROM fact GROUP BY 1 ORDER BY 1").fetchall()
    orphans = con.execute("""SELECT (SELECT count(*) FROM fact f ANTI JOIN store USING (store_id)),
                                    (SELECT count(*) FROM fact f ANTI JOIN sku USING (sku_id))""").fetchone()
    assert all(n > 0 and r > 0 for _, n, r in rows), "empty month"
    assert orphans == (0, 0), f"orphan keys: {orphans}"
    log(f"check: {len(rows)} months, no orphan keys")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["dims", "fact", "marts", "check"])
    a = ap.parse_args()
    t0 = time.time()
    con = duckdb.connect(str(DB))
    con.execute(f"SET memory_limit = '{os.environ.get('VT_MEMORY', '4GB')}'")
    con.execute(f"SET temp_directory = '{(LOCAL / 'tmp').as_posix()}'")
    steps = {"dims": build_dims, "fact": build_fact, "marts": build_marts, "check": check}
    for name, fn in steps.items():
        if a.only in (None, name):
            fn(con)
    con.execute("CHECKPOINT")
    con.close()
    log(f"done in {(time.time() - t0) / 60:.1f} min, file {DB} {DB.stat().st_size / 2**30:.1f} GB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
