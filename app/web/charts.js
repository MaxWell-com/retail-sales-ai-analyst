/* Answer charts in SVG: bars, columns, lines, waterfall. Data colours are blue/grey and the series palette;
   orange is not used (it is the interface colour). Hover tooltip is the shared .tip from app.css. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const PAL = ["#2563EB", "#8B94A3", "#0E9384", "#8B3FD9", "#D97706", "#DC2626", "#15803D", "#0284C7", "#1E3A8A", "#B45309"];
  const POS = "#14713A", NEG = "#BC2417";

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function nf(v, d) {
    if (v == null || isNaN(v)) return "—";
    return Number(v).toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d }).replace("-", "−");
  }
  function fmtv(v, fmt, tick) {
    if (v == null) return "—";
    if (fmt === "pct" && tick) return nf(v * 100, Math.abs(v) < 0.02 && v !== 0 ? 1 : 0) + "%";
    if (fmt === "pct") return nf(v * 100, Math.abs(v) < 0.1 ? 2 : 1) + "%";
    if (fmt === "num0") return nf(v, 0);
    if (fmt === "num2") return nf(v, 2);
    return nf(v, 1);
  }
  function niceTicks(lo, hi, n) {
    if (hi === lo) { hi = lo + 1; }
    const span = hi - lo, step0 = span / (n || 4), mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= (n || 4)) || mag * 10;
    const out = [];
    for (let t = Math.floor(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(+t.toFixed(10));
    return out;
  }
  const tip = () => document.getElementById("tip");
  function showTip(ev, html) {
    const t = tip(); t.innerHTML = html; t.style.opacity = 1;
    const x = Math.min(ev.clientX + 14, window.innerWidth - t.offsetWidth - 10);
    t.style.left = x + "px"; t.style.top = (ev.clientY + 14) + "px";
  }
  function hideTip() { tip().style.opacity = 0; }
  function legend(series, dashKinds) {
    const d = document.createElement("div"); d.className = "legend";
    series.forEach((s, i) => {
      const sp = document.createElement("span");
      sp.innerHTML = `<i style="background:${PAL[i % PAL.length]}" class="${s.kind === "dash" ? "dash" : ""}"></i>${esc(s.name)}`;
      d.appendChild(sp);
    });
    return d;
  }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
  function trunc(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

  function bar(c) {
    const box = document.createElement("div");
    const vals = c.series[0].values, n = c.x.length, rowH = 24, labW = 230, W = 820, padR = 70;
    const H = n * rowH + 24;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    const max = Math.max(0, ...vals.filter(v => v != null)), min = Math.min(0, ...vals.filter(v => v != null));
    const sx = v => labW + (v - min) / ((max - min) || 1) * (W - labW - padR);
    niceTicks(min, max, 5).forEach(t => {
      if (t < min || t > max * 1.0001) return;
      el("line", { x1: sx(t), x2: sx(t), y1: 0, y2: n * rowH, class: "grid-line" }, svg);
      const tx = el("text", { x: sx(t), y: n * rowH + 16, "text-anchor": "middle" }, svg); tx.textContent = fmtv(t, c.fmt, true);
    });
    c.x.forEach((name, i) => {
      const v = vals[i], y = i * rowH;
      const lab = el("text", { x: labW - 8, y: y + rowH / 2 + 4, "text-anchor": "end", class: "lbl" }, svg); lab.textContent = trunc(name, 34);
      if (v == null) return;
      const x0 = sx(Math.min(0, v)), x1 = sx(Math.max(0, v));
      const r = el("rect", { x: x0, y: y + 4, width: Math.max(1, x1 - x0), height: rowH - 8, rx: 3, fill: v < 0 ? NEG : PAL[0] }, svg);
      const t = el("text", { x: x1 + 6, y: y + rowH / 2 + 4, class: "lbl-strong" }, svg); t.textContent = fmtv(v, c.fmt);
      r.addEventListener("mousemove", e => showTip(e, `<b>${esc(name)}</b><br>${fmtv(v, c.fmt)} ${esc(c.unit || "")}`));
      r.addEventListener("mouseleave", hideTip);
    });
    box.appendChild(svg);
    return box;
  }

  function axesXY(svg, c, W, H, pad, lo, hi) {
    const sy = v => pad.t + (1 - (v - lo) / ((hi - lo) || 1)) * (H - pad.t - pad.b);
    niceTicks(lo, hi, 5).forEach(t => {
      if (t < lo - 1e-9 || t > hi + 1e-9) return;
      el("line", { x1: pad.l, x2: W - pad.r, y1: sy(t), y2: sy(t), class: "grid-line" }, svg);
      const tx = el("text", { x: pad.l - 6, y: sy(t) + 4, "text-anchor": "end" }, svg); tx.textContent = fmtv(t, c.fmt, true);
    });
    return sy;
  }

  function line(c) {
    const box = document.createElement("div");
    if (c.series.length > 1) box.appendChild(legend(c.series));
    const W = 860, H = 280, pad = { l: 58, r: 16, t: 10, b: 34 }, n = c.x.length;
    const all = c.series.flatMap(s => s.values).filter(v => v != null);
    let lo = Math.min(...all), hi = Math.max(...all);
    if (c.fmt !== "pct" && lo > 0) lo = 0;
    const pd = (hi - lo) * 0.06; hi += pd; if (lo !== 0) lo -= pd;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    const sy = axesXY(svg, c, W, H, pad, lo, hi);
    const sx = i => pad.l + (n === 1 ? 0.5 : i / (n - 1)) * (W - pad.l - pad.r);
    const every = Math.ceil(n / 14);
    c.x.forEach((x, i) => {
      if (i % every && i !== n - 1) return;
      const t = el("text", { x: sx(i), y: H - pad.b + 16, "text-anchor": "middle" }, svg); t.textContent = trunc(x, 10);
    });
    c.series.forEach((s, k) => {
      let d = "", pen = false;
      s.values.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        d += (pen ? "L" : "M") + sx(i).toFixed(1) + "," + sy(v).toFixed(1); pen = true;
      });
      el("path", { d, fill: "none", stroke: PAL[k % PAL.length], "stroke-width": 2, "stroke-dasharray": s.kind === "dash" ? "5 4" : "" }, svg);
      if (n <= 40) s.values.forEach((v, i) => { if (v != null) el("circle", { cx: sx(i), cy: sy(v), r: 2.6, fill: PAL[k % PAL.length] }, svg); });
    });
    const hl = el("line", { y1: pad.t, y2: H - pad.b, stroke: "#CFC7BC", opacity: 0 }, svg);
    const ov = el("rect", { x: pad.l, y: pad.t, width: W - pad.l - pad.r, height: H - pad.t - pad.b, fill: "transparent" }, svg);
    ov.addEventListener("mousemove", e => {
      const b = svg.getBoundingClientRect(), px = (e.clientX - b.left) / b.width * W;
      const i = Math.max(0, Math.min(n - 1, Math.round((px - pad.l) / ((W - pad.l - pad.r) || 1) * (n - 1))));
      hl.setAttribute("x1", sx(i)); hl.setAttribute("x2", sx(i)); hl.setAttribute("opacity", 1);
      showTip(e, `<b>${esc(c.x[i])}</b>` + c.series.map((s, k) =>
        `<div class="tr"><span><i style="background:${PAL[k % PAL.length]}"></i>${esc(s.name)}</span><b>${fmtv(s.values[i], c.fmt)}</b></div>`).join(""));
    });
    ov.addEventListener("mouseleave", () => { hl.setAttribute("opacity", 0); hideTip(); });
    box.appendChild(svg);
    return box;
  }

  function column(c) {
    const box = document.createElement("div");
    if (c.series.length > 1) box.appendChild(legend(c.series));
    const W = 860, H = 260, pad = { l: 58, r: 16, t: 10, b: 34 }, n = c.x.length, m = c.series.length;
    const all = c.series.flatMap(s => s.values).filter(v => v != null);
    const lo = Math.min(0, ...all), hi = Math.max(0, ...all) * 1.06;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    const sy = axesXY(svg, c, W, H, pad, lo, hi);
    const gw = (W - pad.l - pad.r) / n, bw = Math.max(3, (gw * 0.75) / m);
    c.x.forEach((x, i) => {
      const t = el("text", { x: pad.l + gw * (i + 0.5), y: H - pad.b + 16, "text-anchor": "middle" }, svg); t.textContent = trunc(x, 12);
      c.series.forEach((s, k) => {
        const v = s.values[i]; if (v == null) return;
        const x0 = pad.l + gw * i + gw * 0.125 + k * bw;
        const r = el("rect", { x: x0, y: Math.min(sy(v), sy(0)), width: bw - 1, height: Math.max(1, Math.abs(sy(v) - sy(0))), rx: 2, fill: PAL[k % PAL.length] }, svg);
        r.addEventListener("mousemove", e => showTip(e, `<b>${esc(x)}</b><br>${esc(s.name)}: ${fmtv(v, c.fmt)}`));
        r.addEventListener("mouseleave", hideTip);
      });
    });
    box.appendChild(svg);
    return box;
  }

  function waterfall(c) {
    const box = document.createElement("div");
    const v = c.series[0].values, n = v.length, W = 860, H = 280, pad = { l: 58, r: 16, t: 14, b: 54 };
    let run = 0; const bars = [];
    v.forEach((x, i) => {
      if (i === 0 || i === n - 1) { bars.push({ a: 0, b: x, tot: true }); run = x; }
      else { bars.push({ a: run, b: run + x, d: x }); run += x; }
    });
    const ext = bars.flatMap(b => [b.a, b.b]);
    const lo = Math.min(0, ...ext), hi = Math.max(...ext) * 1.06;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    const sy = axesXY(svg, c, W, H, pad, lo, hi);
    const gw = (W - pad.l - pad.r) / n;
    bars.forEach((b, i) => {
      const x0 = pad.l + gw * i + gw * 0.18, w = gw * 0.64;
      const fill = b.tot ? (i === 0 ? "#8B94A3" : "#2563EB") : (b.d >= 0 ? POS : NEG);
      const r = el("rect", { x: x0, y: sy(Math.max(b.a, b.b)), width: w, height: Math.max(1, Math.abs(sy(b.a) - sy(b.b))), rx: 2, fill }, svg);
      const val = b.tot ? b.b : b.d;
      const t = el("text", { x: x0 + w / 2, y: sy(Math.max(b.a, b.b)) - 4, "text-anchor": "middle", class: "lbl-strong" }, svg);
      t.textContent = (!b.tot && val > 0 ? "+" : "") + fmtv(val, c.fmt);
      const words = String(c.x[i]).split(" "); let line1 = "", line2 = "";
      words.forEach(wd => { if ((line1 + " " + wd).trim().length <= 16 && !line2) line1 = (line1 + " " + wd).trim(); else line2 = (line2 + " " + wd).trim(); });
      const l1 = el("text", { x: x0 + w / 2, y: H - pad.b + 14, "text-anchor": "middle" }, svg); l1.textContent = line1;
      const l2 = el("text", { x: x0 + w / 2, y: H - pad.b + 27, "text-anchor": "middle" }, svg); l2.textContent = trunc(line2, 18);
      r.addEventListener("mousemove", e => showTip(e, `<b>${esc(c.x[i])}</b><br>${fmtv(val, c.fmt)} ${esc(c.unit || "")}`));
      r.addEventListener("mouseleave", hideTip);
    });
    el("line", { x1: pad.l, x2: W - pad.r, y1: sy(0), y2: sy(0), class: "axis" }, svg);
    box.appendChild(svg);
    return box;
  }

  window.Charts = {
    render(c) {
      const f = { bar, line, column, waterfall }[c.type] || bar;
      const d = document.createElement("div"); d.className = "chart-box";
      d.appendChild(f(c));
      return d;
    },
    fmtv, nf, PAL,
  };
})();
