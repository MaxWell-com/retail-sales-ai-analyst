# Analysis methodology: rules the agent follows

These are the analytical rules of the agent. They live in code (`app/semantic.py`, `app/tools.py`, `app/tools2.py`, `app/agent.py`), not in the prompt; tests and code comments cite them as "methodology §N.N".

Every rule applies unless the user explicitly asks otherwise. Whenever a rule restricts an answer, the answer says so in its «Ограничения» (Limitations) block.

Examples below use the synthetic panel shipped with the repository (`scripts/make_synthetic.py`); see `data_dictionary.md`. Chain, manufacturer and brand names are data values and stay in Russian.

## 1. Scope of the agent

1. It answers only questions about retail sell-out in the panel: chains, regions, formats, categories, manufacturers, brands, SKUs, stores, retail and purchase prices, and how they change over time.
2. It does not answer questions outside the panel (general knowledge, advice, the company's own ERP sales, etc.). It declines politely and suggests what can be asked.
3. It does not invent causes. The data shows *what* changed and *where*, not *why*. Any hypothesis about a cause must be labelled as a hypothesis («гипотеза»); the answer checker removes unlabelled causal claims.
4. It never returns raw rows; output tables are aggregates.

## 2. Default scope

1. **Own-company categories.** By default the analysis covers only the categories of the own company (NORD, configurable via `VT_OWN_MANUFACTURER`): the base subgroups where NORD sells products. The rationale is that the relevant market is made of companies selling products similar to ours. A base subgroup is included if it holds at least 0.1% of NORD's all-time sales (`VT_OWN_MIN_SHARE`); adult and kids subgroups are paired (see 2.4). In the synthetic panel this yields 9 subgroups: hair colouring, hair hygiene, body hygiene, hair care, sets, hand and foot care, facial care, soap, hair styling — 32.5% of panel sales and 99.9% of NORD's sales. Not included: medical hygiene (paper, wet wipes), oral care, deodorants, body care, eye and brow make-up, sun care, face cleansing and others. The threshold matters: without it, subgroups where NORD has a few near-zero SKUs would pull large unrelated markets into every default answer. Naming a category in the question, or asking for "the whole panel", lifts the restriction. The «Срез» (Scope) line under every answer states which categories were included.
2. If the user compares with the reference analytics portal or asks for "the Cosmetics category", the portal's 6 subgroups are used (`sku.in_portal`, see `data_dictionary.md` §4).
3. **Default periods** follow the reference portal; the last month in the data is 09.2026. Shares, rankings and structure: the last 12 full months (moving annual total, MAT), 10.2025–09.2026. Growth: year to date versus the same months of the previous year, 01–09.2026 vs 01–09.2025, on the 9 comparable chains (§3.2). A period named in the question always overrides these defaults. "Last 12 months" (the «12 месяцев» button in the question builder) is an explicit period: both shares and growth cover 10.2025–09.2026, compared with 10.2024–09.2025.
4. **Kids subgroups.** When an answer involves a subgroup that has a kids counterpart (hair hygiene, soap, body hygiene, body care, oral care), it shows three rows: adult, kids, total. Example (MAT 10.2025–09.2026): NORD's share is 0.1% in adult soap, 92.0% in kids soap and 1.1% in soap overall — without the split the main point is lost.

## 3. Comparisons over time

1. Compare only identical months: Jan–Sep 2026 vs Jan–Sep 2025. A partial year is never compared with a full one.
2. **Comparable chains.** Growth (year over year, month over month) is computed only on the 9 chains with full coverage: Гранит, Семерочка Х7, Нива, Перелесок ТД, Ажур, Депо Кэш Энд Керри, ЛАД, Гиперкосмос, Детский Остров. Otherwise a chain appearing in or disappearing from the data shows up as growth or decline (Маятник, Стрижик, Добрый, Запас, Камелия, ПАРУС, НУ). This rule is strict.
3. Shares and structure for a period may use all chains; the «Ограничения» block then lists the partially covered chains present in the period.
4. A chain, brand or SKU "disappearing" at the edge of a chain's coverage is a data boundary, not a market exit. Growth for a selected chain is computed only if the chain is present in every compared month.
5. A chain is identified by `network`, not by the client code: one `client_id` can cover several chains (code 4 is shared by Перелесок ТД and Офис Х7).
6. "Holding level" means a change within ±2%. This is a wording convention (an assumption to be stated in the answer).
7. **Incomplete months.** Code checks load completeness for every chain and month (`semantic.coverage()`). A month is incomplete if stores with sales fall below 85%, or revenue below 75%, of the median of the three previous months, **and** it is either the chain's last month in the data or the next month recovers to the previous level (if the low level persists, it is a real shift — e.g. a chain that permanently shrinks — not a partial load). In year-over-year comparisons the incomplete chain-month and the same month a year earlier are removed **on both sides**; in shares the month stays and «Ограничения» notes that it is incomplete. In the synthetic panel two comparable chains are incomplete in 09.2026: ЛАД (3 of ~8 stores, 46% of usual revenue) and Гиперкосмос (all 4 stores, but each at ~60% of its August revenue — only part of the month was loaded).
8. **Period order.** The current period is the later one, the base is the earlier one, regardless of word order in the question. "Versus last year" means the same months a year earlier, not the whole previous year. Periods of different length are aligned to the same calendar months; otherwise no calculation is made.
9. **Small base.** If the base is ≤ 0, below 10% of the current value, or below 0.05% of the sample total for the base period, no growth percentage is shown. The answer says "new in the period" or "base too small, an N-fold increase is not meaningful" (typical cases: SKU delisting and relaunch, a chain with part of a year).

## 4. Sales and prices

1. `rub` is sales in rubles. The data carries no VAT flag, so answers say just "RUB" and do not comment on VAT (design decision). The same applies to `cost`.
2. Growth in `rub` mixes price and volume. They can only be separated via units: the change in average price (`rub/qty`) and in quantity at SKU level. At brand level and above the average price depends on the assortment (mix effect), and the answer says so.
3. `kg` is units × the SKU's mass from the product card; it mixes kilograms and litres. Use it to compare within one category, not across categories.
4. Negative values and possible duplicates are kept as delivered by the source. Negative values are not treated as returns without checking. (The synthetic panel has no negative values; the rule is kept for real data.)

## 5. Stores, formats and distribution

Formats: ММ — small neighbourhood store, СМ — supermarket, ГМ — hypermarket, plus РЦ (distribution centre) and online store. The largest chain, Гранит, mixes several store chains under ММ, so it is split into banners by the chain's own format code (`format_src`): Маяк у дома (МД), Маяк Косметик (МК), Маяк Аптека (МА), Наша Цена, Маяк гипермаркет (ГМ), Маяк прочие (no code). Маяк Косметик is a key cosmetics customer and is always a separate row in Гранит breakdowns. In the database this is `store.banner`.

1. "Stores" means the number of store codes with sales (`qty > 0`) in the period, not a confirmed count of shops.
2. Store counts are never summed across months or chains: for a period, count distinct stores; for "average stores per month", average the monthly counts. The answer states which one was computed.
3. If the question is about the drivers of growth, sales growth is decomposed into stores and sales per store (§8, `factors`).
4. **A store is chain + store code (`store_key`), not `store_id`.** The same physical store (same code and address) is registered under a different `store_id` for some months: Нива in 06–08.2026, Детский Остров in 2024, ЛАД in 05.2025. Without merging, Нива shows 70 stores over 01–09.2026 instead of 40, Детский Остров 62 instead of 50 over 2024–2025, and every multi-month distribution or per-store metric is distorted.
5. Distribution defaults to the last month that is complete for the chains in the slice (an incomplete month, such as 09.2026 for ЛАД and Гиперкосмос, would understate presence).
6. **«Маяк прочие»** are Гранит stores without a format code. Their number grew from 0 in 01–02.2024 to 140 in 09.2026; by assortment profile they look like Маяк у дома. Growth of this banner is not comparable (new stores); breakdowns by Гранит banner say so in «Ограничения». Whether to fold them into Маяк у дома is an open question (§10).
7. **City.** 40% of stores (642 of 1,588) have an empty city, including all stores in Moscow, St Petersburg and Sevastopol. For those three the city is the city-region («Москва г»); for the rest it is «город не указан (<region>)» (city not specified), so a city ranking never silently drops sales.

## 6. Manufacturers and brands

1. A manufacturer is a label assigned by the data vendor, not necessarily one legal entity. Label variants of the same company (e.g. "X" and "X TRADING") are not merged without an explicit decision by the user.
2. One brand can have several labels (МЕГАДРАКОНЫ / MEGADRAGONS, COLORIA / КОЛОРИЯ). Brand lookup ignores case and keyboard layout, shows every variant found and sums them, listing the variants explicitly.
3. NORD is only `manufacturer = 'NORD'`. NORDAKU (manufacturer SHENZHEN NORD TECHNOLOGY CO.) is not NORD.
4. **Private labels** (СТМ) are manufacturers `STM(<chain>)` and brands containing `(PL)`. They are shown in manufacturer rankings but flagged as СТМ; they are excluded from competitor selection unless the user asks otherwise.
5. **`manufacturer = 'не указан'`** (not specified) means the manufacturer is unknown (73 brands, 1.0% of panel sales). It is not one manufacturer: it never appears as a ranking leader and is analysed at brand level.
6. The same brand name may in principle belong to different manufacturers, and some brands look like common words (ВЕСНА, ЛИДЕР, HOME). When searching by name, check the manufacturer.

## 7. Purchase cost (chain's buying price)

1. `cost` is what the chain paid for the goods it sold: the chain's purchase price, not the manufacturer's cost.
2. Purchase prices are comparable only within one chain and one period: e.g. NORD's vs a competitor's purchase price per unit in Нива in 2026.
3. Cost levels compare poorly across chains and across years. The cost-to-revenue ratio of Гранит rose from 0.86 (2024) to 0.98 (2026), while Детский Остров's fell from 0.84 to 0.71 — most likely a change in the vendor's accounting, not in real markup. Such comparisons are made only with a caveat. Markup is therefore computed only within one chain (`prices` shows purchase price and markup only when the breakdown is by chain or banner, or the slice is a single chain).
4. Депо Кэш Энд Керри has no cost from 01.2025 (zeros). Zeros are missing data, not free goods; Депо is excluded from cost calculations from 2025 on.
5. Cost above revenue occurs in individual rows and in whole chains over a year (Депо 2024: 1.05; ЛАД 2026: 1.01; Запас 2026: 1.07). A negative chain markup is never presented as a fact without a caveat.

## 8. Competitors and segments (default assumptions)

1. A manufacturer's segment is a subgroup × «Товар» (`product`) pair holding at least 0.5% of the manufacturer's sales. This mirrors the reference portal's definition.
2. Competitors are manufacturers whose sales in those segments are 0.5× to 2× the target's sales in the same segments; private labels and «не указан» are excluded.
3. Thresholds are assumptions and are always stated in the answer. The user can set their own.
4. "Top by growth" ranks only objects with at least a 0.1% share in both periods (assumption, threshold stated in the answer, parameter `min_share`). Without it the top is filled by objects with a base of a few units — e.g. the brand ПТИЧКА: 3 → 23 units in Jan–Sep, +685%.
5. The default competitor period is year to date, as on the reference portal; the answer says so and offers a "last 12 months" button.
6. The target can be a brand: its manufacturers (labels with at least 1% of the brand's sales) are excluded from the competitor pool. Segments the target has left are listed separately. «не указан» is never a segment leader; private labels are flagged.
7. Assortment rotation: private labels are excluded from introduction candidates and flagged [СТМ] among delisting candidates.
8. The price index in `prices` uses identical SKUs in the same chains with base-period weights (Laspeyres); the difference from the change in average price is the assortment effect.

## 9. Answer format and verification

1. Short: the «Срез» (Scope) line (written by code: categories, object, chains, share and growth periods, measure), then **«Главное»** (Key points, 2–3 items) and **«Выводы»** (Takeaways, 0–3 items) written by the model, then tables and charts.
2. Tables carry units in headers (RUB, mln RUB, units, stores), the period, and the chain and category coverage.
3. A mandatory «Ограничения» block: which methodology rules were applied, which assumptions were made, which chains or categories were excluded and why.
4. Numbers: millions and billions with one or two decimals, shares and growth rates in percent with one decimal, share changes in percentage points. Direction in words: «рост на 3,5%» (up 3.5%), «снижение на 3,3%» (down 3.3%).
5. **Code verifies the model's text.** Every number with a unit must exist in the calculations to the precision shown, its sign must match, and a number next to an object must belong to that object (or to the totals). Judgements with no support in the data ("premium", "stable", "dominates", "outsider", "keeps leadership", etc.) are forbidden. Unconfirmed text is rewritten by the model once; whatever is still unfixed is removed by code, and the answer says what was removed. The model never computes sums, sorts or "the only / all" statements — tools provide them as aggregate facts.
6. «Что посмотреть дальше» (Next steps) are buttons built by code from what the tools can actually compute; the model does not suggest follow-up questions.
7. A follow-up question ("and in units?", "which SKUs exactly…") inherits from the previous turn the period and filters it lacks; this is visible in «Распознано» (Recognised), e.g. "from the previous question: period August 2026". A new question without such cues inherits nothing.

## 10. Design decisions and open questions

Design decisions:

- VAT: no flag in the data — answers say "RUB" without comment (§4.1).
- Formats: the decoding above is confirmed; Маяк Косметик is always shown separately (§5).
- Kids subgroups: always adult and kids rows plus a total (§2.4).
- Default periods: as on the reference portal (§2.3).
- Default scope: own-company categories (§2.1). The 0.1% presence threshold is an assumption and can be changed (`VT_OWN_MIN_SHARE`).

Open questions (the option in brackets applies until decided):

- Should «Маяк прочие» be folded into Маяк у дома? (No: a separate row with a caveat.)
- Is "Vector" the single brand VECTOR or the whole VECTOR* line (FUSION, ANTI-YELLOW, PROWAY)? (The single brand; the other line brands are listed in «Распознано».)
- The "top by growth" threshold (at least 0.1% share in both periods, §8.4).
- Reload incomplete months once the vendor completes them (until then they are excluded from comparisons, §3.7).
- Default competitor period: year to date as on the portal, or 12 months as for shares? (Year to date, §8.5.)
- Introduction candidates: allow private labels of the chain's own group (STM(X7) from Семерочка into Перелесок)? (No: all private labels are excluded, §8.7.)
- Factor analysis for Гранит alone: by banner or as one chain row? (By banner, Маяк Косметик separately.)
- Price index on identical SKUs in the same chains with base weights (§8.8): is this definition acceptable?
