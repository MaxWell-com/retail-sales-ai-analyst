/* Retail sales AI analyst: chat, streamed calculation progress, history, exports, pages «Данные», «Методика», «Проверка». */
(function () {
  const $ = s => document.querySelector(s);
  const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const nf = Charts.nf;
  const MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"];
  const TOOL_RU = {
    data_info: "описание данных", find_entities: "поиск в справочниках", ranking: "рейтинг и доли", trend: "динамика",
    profile: "обзор производителя", factors: "факторный анализ", distribution: "представленность в точках", prices: "цены и закупка",
    abc: "ABC-анализ", assortment: "ротация ассортимента", forecast: "прогноз", competitors: "конкуренты", sql: "SQL-запрос",
  };
  const EXAMPLES = [
    "Какие данные у тебя есть?",
    "Топ-10 брендов по выручке в категории Косметика",
    "Обзор NORD: категории, бренды, сети",
    "Факторный анализ продаж NORD: январь–июнь 2026 к январю–июню 2025",
    "Ближайшие конкуренты NORD в категории Косметика",
    "Определи сезонность солнцезащитных средств",
    "Карта представленности брендов NORD по сетям",
    "Сравни среднюю цену продажи и закупочную цену за штуку детских шампуней NORD и AURIGA в Ниве и Граните, сентябрь 2026",
  ];
  let META = null, CID = null, busy = false;

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let d = "";
      try { d = (await r.json()).detail; } catch (e) { /* response is not JSON */ }
      throw new Error(r.status === 400 && d ? d : `сервер не смог выполнить запрос (код ${r.status})`);
    }
    return r.json();
  }
  // Tool error during calculation: the user sees the gist; the Python exception only in the tooltip and «Как считалось»
  const humanErr = s => /^Ошибка расчета/.test(s || "") ? "внутренняя ошибка расчета" : String(s || "").replace(/^Ошибка:\s*/, "");

  // ——— formatting ———
  function fmt(v, f) {
    if (v == null || v === "") return f === "text" || f === "txt" ? "" : "—";
    if (typeof v !== "number") return esc(v);
    switch (f) {
      case "mln": return nf(v / 1e6, Math.abs(v) < 1e6 ? 2 : 1);
      case "th": return nf(v / 1e3, 1);
      case "pct": return nf(v * 100, 1) + "%";
      case "pct2": return nf(v * 100, 2) + "%";
      case "gpct": return `<span class="${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${v > 0 ? "+" : ""}${nf(v * 100, 1)}%</span>`;
      case "pp": return `<span class="${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${v > 0 ? "+" : ""}${nf(v * 100, 2)}</span>`;
      case "int": case "num0": return nf(v, 0);
      case "num2": return nf(v, 2);
      case "heat": {
        const a = Math.max(0, Math.min(1, v));
        return `<span class="heat" style="background:rgba(37,99,235,${(0.08 + a * 0.72).toFixed(2)});color:${a > 0.55 ? "#fff" : "#191614"}">${nf(v * 100, 0)}%</span>`;
      }
      default: return nf(v, 1);
    }
  }

  // ——— mini-Markdown for the model answer and docs ———
  function inline(s) {
    return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>").replace(/(^|\s)\*([^*\s][^*]*?)\*(?=\s|$|[.,;:])/g, "$1<i>$2</i>");
  }
  function md(text) {
    const lines = String(text || "").replace(/\r/g, "").split("\n");
    let html = "", list = null, i = 0;
    const close = () => { if (list) { html += `</${list}>`; list = null; } };
    while (i < lines.length) {
      const l = lines[i];
      if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
        close();
        const row = s => s.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
        html += "<table><thead><tr>" + row(l).map(c => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>";
        i += 2;
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) { html += "<tr>" + row(lines[i]).map(c => `<td>${inline(c)}</td>`).join("") + "</tr>"; i++; }
        html += "</tbody></table>";
        continue;
      }
      let m;
      if ((m = l.match(/^(#{1,4})\s+(.*)/))) { close(); const n = Math.min(4, m[1].length + 1); html += `<h${n}>${inline(m[2])}</h${n}>`; }
      else if ((m = l.match(/^\s*[-*•]\s+(.*)/))) { if (list !== "ul") { close(); html += "<ul>"; list = "ul"; } html += `<li>${inline(m[1])}</li>`; }
      else if ((m = l.match(/^\s*\d+[.)]\s+(.*)/))) { if (list !== "ol") { close(); html += "<ol>"; list = "ol"; } html += `<li>${inline(m[1])}</li>`; }
      else if (!l.trim()) { close(); }
      else if (/^\s*---+\s*$/.test(l)) { close(); html += "<hr>"; }
      else { close(); html += `<p>${inline(l)}</p>`; }
      i++;
    }
    close();
    return `<div class="md">${html}</div>`;
  }

  // ——— result blocks ———
  function tableHTML(b) {
    const cols = b.columns;
    let h = `<div class="scroll"><table><thead><tr>${cols.map(c => `<th class="${c.fmt === "text" || c.fmt === "txt" ? "txt" : ""}">${esc(c.label)}</th>`).join("")}</tr></thead><tbody>`;
    b.rows.forEach(r => {
      const cls = r._cls === "tot" ? "tot" : r._cls === "lvl" ? "lvl" : "";
      h += `<tr class="${cls}">` + cols.map(c => {
        const tcls = c.fmt === "txt" ? "txt long" : (c.fmt === "text" ? "txt" : "");
        return `<td class="${tcls}">${fmt(r[c.key], c.fmt)}</td>`;
      }).join("") + "</tr>";
    });
    return h + "</tbody></table></div>";
  }
  function kpisHTML(b) {
    return `<div class="kpis">` + b.items.map((k, i) => `${i ? '<div class="kpi-sep"></div>' : ""}<div class="kpi"><div class="l">${esc(k.label)}</div>
      <div class="v">${esc(k.value)}</div><div class="s">${k.sub ? `<span class="${k.sign > 0 ? "pos" : k.sign < 0 ? "neg" : ""}">${esc(k.sub)}</span>` : ""}</div></div>`).join("") + "</div>";
  }
  function resultEl(r) {
    const d = document.createElement("div"); d.className = "res";
    d.innerHTML = `<div class="rh"><b>${esc(r.title)}</b><span>${esc(r.subtitle || "")}</span></div>`;
    r.blocks.forEach(b => {
      if (b.kind === "kpis") d.insertAdjacentHTML("beforeend", kpisHTML(b));
      else if (b.kind === "table") {
        if (!b.rows.length) return;
        d.insertAdjacentHTML("beforeend", `<h4>${esc(b.title)}</h4>` + tableHTML(b) + (b.note ? `<p class="note">${esc(b.note)}</p>` : ""));
      } else if (b.kind === "chart") {
        if (!b.x.length) return;
        d.insertAdjacentHTML("beforeend", `<h4>${esc(b.title)}</h4>`);
        d.appendChild(Charts.render(b));
      } else if (b.kind === "text") d.insertAdjacentHTML("beforeend", md(b.md));
    });
    return d;
  }

  // ——— answer ———
  // Order: slice and number check (code) -> model text -> key figures, tables, charts (code) -> caveats.
  // The answer view (badge, unverified-number marks, copy text) is built by the server — the same one used in exports.
  function answerEl(a) {
    const v = a.view || { slice: a.slice || "", check: {}, marked: a.answer || "", key: [], caveats: a.caveats || [], tech: "", copy: a.answer || "" };
    const box = document.createElement("div");
    if (v.slice) box.insertAdjacentHTML("beforeend", `<div class="slice">${esc(v.slice)}</div>`);
    if (v.check.text) box.insertAdjacentHTML("beforeend", `<div class="chk"><span class="pill ${esc(v.check.level || "")}">${esc(v.check.text)}</span></div>`);
    if (v.marked.trim()) {
      const text = md(v.marked).replace(/\u0001/g, '<mark class="unv" title="Число не подтверждено расчетом">').replace(/\u0002/g, "</mark>");
      box.insertAdjacentHTML("beforeend", v.check.status === "fail"
        ? `<details class="unv-text"><summary>Текст модели не прошел проверку чисел</summary>${text}</details>` : text);
    }
    if (v.key.length) box.insertAdjacentHTML("beforeend", `<div class="keyfacts"><b>Ключевые цифры расчета</b><ul>${v.key.map(h => `<li>${esc(h)}</li>`).join("")}</ul></div>`);
    if (a.results && a.results.length) {
      const bl = document.createElement("div"); bl.className = "blocks";
      a.results.forEach(r => bl.appendChild(resultEl(r)));
      box.appendChild(bl);
    }
    if (v.caveats.length) {
      box.insertAdjacentHTML("beforeend", `<div class="callout warn limits"><span class="pic">!</span><div><b>Ограничения</b><ul>${v.caveats.map(c => `<li>${esc(c)}</li>`).join("")}</ul></div></div>`);
    }
    const steps = nextSteps(a);
    if (steps.length) box.appendChild(nextEl(steps));
    box.insertAdjacentHTML("beforeend", `<div class="foot">
      <details class="calls"><summary>Как считалось: ${esc(a.recognized || "")}</summary><pre>${esc(howText(a, v))}</pre></details>
      <span class="sp"></span>
      ${a.id && a.results && a.results.length ? `<a class="btn" href="/api/ask/${a.id}/xlsx">Excel</a><a class="btn" href="/api/ask/${a.id}/pptx">PowerPoint</a>` : ""}
      <button class="btn" data-copy>Копировать текст</button></div>`);
    const cp = box.querySelector("[data-copy]");
    cp.onclick = () => navigator.clipboard.writeText(v.copy || a.answer || "").then(() => {
      cp.textContent = "Скопировано"; setTimeout(() => { cp.textContent = "Копировать текст"; }, 1500);
    });
    return box;
  }

  // Next steps: ready-made calculations from tool results; on click they run without the model choosing
  function nextSteps(a) {
    const out = [];
    (a.next || [].concat(...(a.results || []).map(r => r.next || []))).forEach(n => {
      if (n && n.text && n.tool && !out.some(x => x.text === n.text)) out.push(n);
    });
    return out.slice(0, 6);
  }
  function nextEl(steps) {
    const d = document.createElement("div"); d.className = "next";
    d.innerHTML = `<span class="flabel">Что посмотреть дальше</span>`;
    steps.forEach(n => {
      const b = document.createElement("button");
      b.textContent = n.text;
      b.title = `Расчет «${TOOL_RU[n.tool] || n.tool}» с готовыми аргументами: ${JSON.stringify(n.args || {})}`;
      b.onclick = () => ask(n.text, { tool: n.tool, args: n.args || {} });
      d.appendChild(b);
    });
    return d;
  }

  // «Как считалось» (how it was calculated): calls and code overrides of model arguments, check findings, error details
  function howText(a, v) {
    const q = x => typeof x === "string" ? `«${x}»` : JSON.stringify(x);
    const SRC = { forced: " (кнопка следующего шага, без выбора модели)", code: " (вызвал код)" };
    const out = [];
    (a.calls || []).forEach(c => {
      out.push(`${c.ok ? "✓" : "✗"} ${c.name} ${JSON.stringify(c.args)} — ${c.sec} с${SRC[c.source] || ""}`);
      const f = Object.entries(c.filled || {}), o = Object.entries(c.overridden || {}).filter(([, x]) => Array.isArray(x));
      const rep = o.filter(([, x]) => x[1] != null), del = o.filter(([, x]) => x[1] == null);
      if (f.length) out.push(`   код дописал: ${f.map(([k, x]) => `${k} = ${q(x)}`).join("; ")}`);
      if (rep.length) out.push(`   код заменил: ${rep.map(([k, x]) => `${k} модели ${q(x[0])} на ${q(x[1])}`).join("; ")}`);
      if (del.length) out.push(`   код убрал: ${del.map(([k, x]) => `${k} модели ${q(x[0])}`).join("; ")}`);
      if (c.error) out.push(`   ошибка: ${c.error}`);
    });
    if (!out.length) out.push("Инструменты не вызывались.");
    const ch = a.check || {};
    (ch.unmatched || []).forEach(u => { if (u && typeof u === "object") out.push(`Не подтверждено: ${u.text}${u.reason ? " — " + u.reason : ""}`); });
    (ch.banned || []).forEach(b => out.push(`Оценка без опоры в данных: ${b && typeof b === "object" ? b.text : b}`));
    if ((ch.removed || []).length) out.push("Код убрал из текста модели:", ...ch.removed.map(s => `   «${s}»`));
    if (v.tech) out.push(`Техническая причина: ${v.tech}`);
    return out.join("\n");
  }

  function turnEl(question) {
    const t = document.createElement("div"); t.className = "turn";
    t.innerHTML = `<div class="q">${esc(question)}</div>
      <div class="panel ans"><div class="ph"><div class="who"><span class="dot"></span>Аналитик</div><div class="meta"></div></div>
      <div class="progress"></div><div class="body"></div></div>`;
    return t;
  }
  function setMeta(t, a) {
    const tok = a.tokens ? ` · токенов: ${nf(a.tokens.prompt, 0)} на входе, ${nf(a.tokens.gen, 0)} на выходе` : "";
    t.querySelector(".meta").textContent = `${a.ts || ""} · ${nf(a.sec || 0, 0)} с${tok}`;
  }

  // ——— sending a question and the event stream ———
  // nextStep — {tool, args} of a next-step button: the server runs this tool with these arguments, no model choice
  async function ask(question, nextStep) {
    if (busy || !question.trim()) return;
    busy = true; $("#send").disabled = true;
    const welcome = $("#turns .welcome"); if (welcome) welcome.remove();
    const t = turnEl(question);
    $("#turns").appendChild(t);
    t.scrollIntoView({ behavior: "smooth", block: "start" });
    const prog = t.querySelector(".progress"), body = t.querySelector(".body");
    let thinkChars = 0, thinkEl = null, live = null, liveText = "";
    const step = (text, cls) => {
      prog.querySelectorAll(".st.run").forEach(x => x.classList.remove("run"));
      const d = document.createElement("div"); d.className = "st " + (cls || "run");
      d.innerHTML = `<span class="t"></span><span>${text}</span>`;
      prog.appendChild(d); return d;
    };
    try {
      const { id, cid } = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, cid: CID, ...(nextStep || {}) }) });
      CID = cid;
      await new Promise((resolve, reject) => {
        const es = new EventSource(`/api/ask/${id}/events`);
        es.onmessage = m => {
          const ev = JSON.parse(m.data);
          const stamp = d => { if (d && ev.t != null) d.querySelector(".t").textContent = nf(ev.t, 0) + " с"; };
          if (ev.type === "queued" && ev.pos > 1) step(`В очереди: ${ev.pos}-й`);
          else if (ev.type === "step") stamp(step(esc(ev.text), "done"));
          else if (ev.type === "llm") { stamp(step(esc(ev.text))); thinkChars = 0; thinkEl = null; liveText = ""; if (live) { live.remove(); live = null; } }
          else if (ev.type === "delta" && ev.kind === "thinking") {
            thinkChars += ev.text.length;
            if (!thinkEl) { thinkEl = document.createElement("div"); thinkEl.className = "thinking"; prog.appendChild(thinkEl); }
            thinkEl.textContent = `модель рассуждает… ${nf(thinkChars, 0)} знаков`;
          } else if (ev.type === "delta" && ev.kind === "content") {
            liveText += ev.text;
            if (!live) { live = document.createElement("div"); body.appendChild(live); }
            live.innerHTML = `<div class="draft-note">Черновик модели: числа будут проверены, когда она допишет</div>` + md(liveText);
          } else if (ev.type === "tool") stamp(step(`Считаю: ${esc(TOOL_RU[ev.name] || ev.name)} <code>${esc(JSON.stringify(ev.args))}</code>`));
          else if (ev.type === "tool_done") {
            const last = prog.querySelector(".st.run");
            if (last) { last.classList.remove("run"); last.insertAdjacentHTML("beforeend", ` <span class="muted"${ev.ok ? "" : ` title="${esc(ev.note)}"`}>— ${nf(ev.sec, 1)} с${ev.ok ? "" : ". " + esc(humanErr(ev.note))}</span>`); if (!ev.ok) last.classList.add("err"); }
          } else if (ev.type === "done") { es.close(); resolve(); }
        };
        es.onerror = () => { es.close(); reject(new Error("связь с сервером прервалась; ответ, если он посчитан, появится в диалоге слева")); };
      });
      const a = await api(`/api/ask/${id}`);
      if (live) live.remove();
      prog.querySelectorAll(".st.run").forEach(x => x.classList.remove("run"));
      const det = document.createElement("details"); det.className = "calls";
      det.innerHTML = `<summary>Ход расчета</summary>`;
      while (prog.firstChild) det.appendChild(prog.firstChild);
      prog.appendChild(det);
      body.appendChild(answerEl(a));
      setMeta(t, a);
      loadHistory();
    } catch (e) {
      step("Ошибка: " + esc(e.message), "err");
    } finally {
      busy = false; $("#send").disabled = false;
    }
  }

  // ——— history ———
  async function loadHistory() {
    const h = await api("/api/history");
    $("#hist").innerHTML = h.length ? h.map(c => `<a href="#chat/${c.id}" class="${c.id === CID ? "active" : ""}">${esc(c.title)}<small>${esc(c.updated || "")} · вопросов: ${c.n}</small></a>`).join("")
      : `<div class="empty">Пока нет диалогов</div>`;
  }
  async function openConv(cid) {
    CID = cid;
    const c = await api(`/api/conv/${cid}`);
    $("#turns").innerHTML = "";
    c.answers.forEach(a => {
      const t = turnEl(a.question);
      t.querySelector(".progress").remove();
      t.querySelector(".body").appendChild(answerEl(a));
      setMeta(t, a);
      $("#turns").appendChild(t);
    });
    loadHistory();
  }
  function newConv() {
    CID = null;
    $("#turns").innerHTML = `<div class="panel welcome"><h2>Спросите о продажах в сетях</h2>
      <p>Аналитик отвечает по панели продаж розничных сетей за ${esc(META ? META.from.slice(0, 7).split("-").reverse().join(".") : "")}–${esc(META ? META.to.slice(0, 7).split("-").reverse().join(".") : "")}: доли, рейтинги, динамика, сезонность, представленность в точках, цены и закупка сети, факторный анализ, ротация ассортимента, прогноз. Цифры считает код по данным; текст модели проверяется, непроверенные числа подсвечиваются. Под каждым ответом — срез расчета, таблицы, графики, ограничения и выгрузка в Excel и PowerPoint.</p>
      <div class="examples">${EXAMPLES.map(q => `<button>${esc(q)}</button>`).join("")}</div></div>`;
    $("#turns").querySelectorAll(".examples button").forEach(b => b.onclick = () => { $("#q").value = b.textContent; $("#q").focus(); });
    loadHistory();
  }

  // ——— query builder ———
  function segInit(id, cb) {
    $(id).querySelectorAll("button").forEach(b => b.onclick = () => {
      $(id).querySelectorAll("button").forEach(x => x.classList.toggle("active", x === b)); cb && cb(b.dataset.v);
    });
  }
  const segVal = id => $(id).querySelector("button.active").dataset.v;
  function monthText(v) { const [y, m] = v.split("-"); return `${MONTHS[+m - 1]} ${y}`; }
  function consPhrase() {
    const parts = [];
    const obj = $("#c-obj").value.trim();
    if (obj) parts.push(obj);
    const net = $("#c-net").value;
    if (net) parts.push(`в сети ${net}`);
    const p = segVal("#c-per");
    if (p === "mat") parts.push("за последние 12 месяцев");
    else if (p === "ytd") parts.push("с начала года к тому же периоду прошлого года");
    else if (p === "last") parts.push(`за ${monthText(META.to.slice(0, 7))}`);
    else if (p === "own" && $("#c-from").value && $("#c-to").value) {
      const a = $("#c-from").value, b = $("#c-to").value;
      parts.push(a === b ? `за ${monthText(a)}` : `${monthText(a).split(" ")[0]}–${monthText(b)}`.replace(/^(\S+)–(\S+) (\d+)$/, a.slice(0, 4) === b.slice(0, 4) ? "$1–$2 $3" : "$1 " + a.slice(0, 4) + " – $2 $3"));
    }
    const ms = segVal("#c-meas");
    if (ms) parts.push(ms);
    return parts.join(", ");
  }

  // ——— pages ———
  async function showPage(p) {
    document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", a.dataset.p === p));
    ["chat", "data", "method", "eval"].forEach(x => $("#p-" + x).hidden = x !== p);
    if (p === "data" && !$("#p-data").dataset.ok) {
      $("#p-data").innerHTML = `<div class="panel doc"><p class="empty">Загрузка…</p></div>`;
      const [info, dict] = await Promise.all([api("/api/info"), fetch("/api/docs/data_dictionary").then(r => r.text())]);
      $("#p-data").innerHTML = "";
      $("#p-data").appendChild(resultEl(info));
      $("#p-data").insertAdjacentHTML("beforeend", `<div class="panel doc" style="margin-top:14px">${md(dict)}</div>`);
      $("#p-data").dataset.ok = 1;
    }
    if (p === "method" && !$("#p-method").dataset.ok) {
      const t = await fetch("/api/docs/methodology").then(r => r.text());
      $("#p-method").innerHTML = `<div class="panel doc">${md(t)}</div>`;
      $("#p-method").dataset.ok = 1;
    }
    if (p === "eval") {
      const e = await api("/api/eval");
      if (!e.runs || !e.runs.length) { $("#p-eval").innerHTML = `<div class="panel doc"><p class="empty">Прогона eval-набора еще не было.</p></div>`; return; }
      const r = e.runs[e.runs.length - 1];
      const ok = r.items.filter(x => x.pass).length;
      let h = `<div class="panel doc"><h2 style="font:700 16px var(--font-h);margin:14px 0 4px">Проверка агента на eval-наборе</h2>
        <p class="sub" style="color:var(--text-subtle);font-size:12px">Прогон ${esc(r.ts)} · модель ${esc(r.model)} · пройдено ${ok} из ${r.items.length} · среднее время ${nf(r.items.reduce((s, x) => s + (x.sec || 0), 0) / r.items.length, 0)} с</p>
        <div class="scroll" style="max-height:none"><table><thead><tr><th class="txt">Вопрос</th><th>Итог</th><th class="txt">Инструменты</th><th class="txt">Проверки</th><th>Время, с</th></tr></thead><tbody>`;
      r.items.forEach(x => {
        h += `<tr><td class="txt" style="max-width:420px">${esc(x.q)}</td><td><span class="pill ${x.pass ? "ok" : "bad"}">${x.pass ? "пройден" : "нет"}</span></td>
          <td class="txt">${esc((x.tools || []).join(", "))}</td><td class="txt" style="white-space:normal;max-width:520px">${(x.checks || []).map(c => `<div class="${c.ok ? "ok-badge" : "bad-badge"}">${c.ok ? "✓" : "✗"} ${esc(c.name)}</div>`).join("")}</td>
          <td>${nf(x.sec, 0)}</td></tr>`;
      });
      $("#p-eval").innerHTML = h + "</tbody></table></div></div>";
    }
  }
  function route() {
    const h = location.hash.replace("#", "") || "chat";
    const [p, id] = h.split("/");
    showPage(p);
    if (p === "chat") { if (id && id !== CID) openConv(id); else if (!id && !$("#turns").children.length) newConv(); }
  }

  // ——— startup ———
  async function init() {
    META = await api("/api/meta");
    const per = d => d.slice(0, 7).split("-").reverse().join(".");
    $("#asof").textContent = per(META.to);
    $("#model").textContent = META.model;
    $("#scope").textContent = `Панель sell-out розничных сетей · ${per(META.from)}–${per(META.to)} · ${META.networks.length} сетей · считается локально`;
    $("#c-net").innerHTML = `<option value="">Все сети</option>` + META.networks.map(n => `<option>${esc(n)}</option>`).join("") + META.banners.map(b => `<option>${esc(b)}</option>`).join("");
    $("#objs").innerHTML = META.top_manufacturers.slice(0, 120).map(m => `<option value="${esc(m)}">`).join("") + META.top_brands.slice(0, 200).map(b => `<option value="${esc(b)}">`).join("");
    $("#c-from").value = META.to.slice(0, 4) + "-01"; $("#c-to").value = META.to.slice(0, 7);
    segInit("#c-per", v => $("#c-own").hidden = v !== "own");
    segInit("#c-meas");
    $("#c-add").onclick = () => { const ph = consPhrase(); if (ph) { $("#q").value = ($("#q").value.trim() + (ph ? (($("#q").value.trim() ? ", " : "") + ph) : "")); $("#q").focus(); } };
    $("#send").onclick = () => { const q = $("#q").value; $("#q").value = ""; ask(q); };
    $("#q").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#send").click(); } });
    $("#new").onclick = () => { location.hash = "#chat"; newConv(); };
    window.addEventListener("hashchange", route);
    route();
  }
  init().catch(e => { document.body.insertAdjacentHTML("beforeend", `<div class="callout bad" style="margin:20px">Не удалось загрузить: ${esc(e.message)}</div>`); });
})();
