"""
History dashboard — visualize balance history and trade log.

Usage:
    python src/dashboard/history.py

Reads:
    src/state/balance_history.csv
    src/state/trade_log.csv

Writes & opens:
    src/dashboard/history.html
"""

import csv
import json
import os
import webbrowser

SRC_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BALANCE_FILE = os.path.join(SRC_DIR, "state", "balance_history.csv")
TRADE_FILE   = os.path.join(SRC_DIR, "state", "trade_log.csv")
OUT_FILE     = os.path.join(SRC_DIR, "dashboard", "history.html")

ROWS_PER_PAGE = 100


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_html(out_path=None):
    """
    Generate history.html from the current CSV state and write it to out_path.
    If out_path is None, writes to the default OUT_FILE location.
    Returns the path written.
    """
    out_path = out_path or OUT_FILE
    balance_rows = _read_csv(BALANCE_FILE)
    trade_rows   = _read_csv(TRADE_FILE)

    for r in balance_rows:
        r["cash_balance"]  = float(r.get("cash_balance",  0) or 0)
        r["market_value"]  = float(r.get("market_value",  0) or 0)
        r["total_balance"] = float(r.get("total_balance", 0) or 0)

    for r in trade_rows:
        r["quantity"] = float(r.get("quantity", 0) or 0)
        r["price"]    = float(r.get("price",    0) or 0)
        r["value"]    = float(r.get("value",    0) or 0)

    html = _HTML_TEMPLATE \
        .replace("__BALANCE_DATA__",  json.dumps(balance_rows)) \
        .replace("__TRADE_DATA__",    json.dumps(trade_rows)) \
        .replace("__ROWS_PER_PAGE__", str(ROWS_PER_PAGE))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path


def main():
    path = generate_html()
    url  = "file:///" + path.replace("\\", "/")
    print(f"Opening history dashboard: {url}")
    webbrowser.open(url)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>WebullTradeAI — History</title>
  <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f1117; color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      min-height: 100vh;
    }

    /* ── Header ── */
    #header {
      background: #1a1f2e; border-bottom: 3px solid #2ecc71;
      padding: 18px 32px;
    }
    #header h1 { font-size: 1.3rem; font-weight: 600; letter-spacing: 0.5px; }
    #header p  { font-size: 0.8rem; color: #7f8c8d; margin-top: 4px; }

    /* ── Tabs ── */
    #tabs {
      display: flex; background: #1a1f2e;
      border-bottom: 1px solid #2a2f3e; padding: 0 32px;
    }
    .tab-btn {
      padding: 14px 24px; font-size: 0.9rem; font-weight: 600; cursor: pointer;
      border: none; background: none; color: #7f8c8d;
      border-bottom: 3px solid transparent; margin-bottom: -1px;
      transition: color 0.2s, border-color 0.2s;
    }
    .tab-btn.active    { color: #2ecc71; border-bottom-color: #2ecc71; }
    .tab-btn:hover:not(.active) { color: #e0e0e0; }

    .tab-panel          { display: none; padding: 28px 32px; max-width: 1400px; margin: 0 auto; }
    .tab-panel.active   { display: block; }

    /* ── Balance tab ── */
    #balance-controls {
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 16px; margin-bottom: 20px;
    }

    #legend { display: flex; gap: 14px; flex-wrap: wrap; }
    .legend-item {
      display: flex; align-items: center; gap: 10px; cursor: pointer;
      padding: 8px 16px; border-radius: 8px; border: 1px solid #2a2f3e;
      background: #1a1f2e; transition: opacity 0.2s; user-select: none;
    }
    .legend-item.off    { opacity: 0.35; }
    .legend-dot         { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
    .legend-label       { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.7px; color: #7f8c8d; margin-bottom: 2px; }
    .legend-value       { font-size: 1rem; font-weight: 700; color: #fff; }

    #timeframe          { display: flex; gap: 6px; flex-wrap: wrap; }
    .tf-btn {
      padding: 6px 13px; border-radius: 6px; border: 1px solid #2a2f3e;
      background: #1a1f2e; color: #7f8c8d; font-size: 0.82rem; font-weight: 600;
      cursor: pointer; transition: background 0.15s, color 0.15s;
    }
    .tf-btn:hover          { background: #2a2f3e; color: #e0e0e0; }
    .tf-btn.active         { background: #2ecc71; color: #0f1117; border-color: #2ecc71; }

    #chart-container    { border-radius: 10px; border: 1px solid #2a2f3e; overflow: hidden; height: 460px; }
    #no-balance-data    {
      display: none; align-items: center; justify-content: center;
      height: 460px; color: #4a5568; font-size: 0.95rem;
      background: #1a1f2e; border-radius: 10px; border: 1px solid #2a2f3e;
    }

    /* ── Trade tab ── */
    #trade-filters      { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-end; margin-bottom: 14px; }
    .filter-group       { display: flex; flex-direction: column; gap: 5px; }
    .filter-group label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.7px; color: #7f8c8d; }

    input[type="text"], select {
      background: #1a1f2e; border: 1px solid #2a2f3e; color: #e0e0e0;
      padding: 7px 12px; border-radius: 6px; font-size: 0.875rem;
    }
    input[type="text"]           { width: 150px; }
    select                       { min-width: 110px; }
    input[type="text"]:focus,
    select:focus                 { outline: none; border-color: #3a4060; }

    #trade-count        { font-size: 0.8rem; color: #7f8c8d; margin-bottom: 12px; min-height: 18px; }

    .table-wrap         { overflow-x: auto; border-radius: 10px; border: 1px solid #2a2f3e; }
    table               { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
    th {
      text-align: left; color: #7f8c8d; font-weight: 500;
      padding: 10px 16px; border-bottom: 1px solid #2a2f3e;
      background: #1a1f2e; cursor: pointer; user-select: none; white-space: nowrap;
    }
    th:hover            { color: #e0e0e0; }
    th .sort-icon       { display: inline-block; margin-left: 4px; opacity: 0.4; font-size: 0.7rem; }
    th.sort-asc  .sort-icon,
    th.sort-desc .sort-icon { opacity: 1; color: #2ecc71; }
    td                  { padding: 9px 16px; border-bottom: 1px solid #1e2330; }
    tr:last-child td    { border-bottom: none; }
    tr:hover td         { background: rgba(255,255,255,0.02); }
    .side-buy           { color: #2ecc71; font-weight: 600; }
    .side-sell          { color: #e74c3c; font-weight: 600; }
    .no-rows td         { color: #4a5568; text-align: center; padding: 48px; }

    /* ── Pagination ── */
    #pagination {
      display: flex; align-items: center; justify-content: center;
      gap: 6px; margin-top: 18px; flex-wrap: wrap;
    }
    .page-btn {
      padding: 6px 12px; border-radius: 6px; border: 1px solid #2a2f3e;
      background: #1a1f2e; color: #7f8c8d; font-size: 0.82rem; cursor: pointer;
    }
    .page-btn:hover:not(:disabled):not(.active) { background: #2a2f3e; color: #e0e0e0; }
    .page-btn.active    { background: #2ecc71; color: #0f1117; border-color: #2ecc71; font-weight: 700; }
    .page-btn:disabled  { opacity: 0.3; cursor: default; }
    .page-ellipsis      { color: #4a5568; padding: 6px 4px; font-size: 0.82rem; }
  </style>
</head>
<body>

<div id="header">
  <h1>WebullTradeAI — History</h1>
  <p>Balance history &amp; trade log</p>
</div>

<div id="tabs">
  <button class="tab-btn active" onclick="switchTab(this,'balance')">Balance History</button>
  <button class="tab-btn"        onclick="switchTab(this,'trades')">Trade History</button>
</div>

<!-- ===== Balance Tab ===== -->
<div id="tab-balance" class="tab-panel active">
  <div id="balance-controls">
    <div id="legend">
      <div class="legend-item" id="leg-total"  onclick="toggleSeries('total')">
        <div class="legend-dot" style="background:#2ecc71"></div>
        <div>
          <div class="legend-label">Total Balance</div>
          <div class="legend-value" id="val-total">—</div>
        </div>
      </div>
      <div class="legend-item" id="leg-market" onclick="toggleSeries('market')">
        <div class="legend-dot" style="background:#3498db"></div>
        <div>
          <div class="legend-label">Market Value</div>
          <div class="legend-value" id="val-market">—</div>
        </div>
      </div>
      <div class="legend-item" id="leg-cash"   onclick="toggleSeries('cash')">
        <div class="legend-dot" style="background:#f0c040"></div>
        <div>
          <div class="legend-label">Cash Balance</div>
          <div class="legend-value" id="val-cash">—</div>
        </div>
      </div>
    </div>
    <div id="timeframe">
      <button class="tf-btn active" onclick="setTf(this,'ALL')">All</button>
      <button class="tf-btn"        onclick="setTf(this,'5Y')">5Y</button>
      <button class="tf-btn"        onclick="setTf(this,'1Y')">1Y</button>
      <button class="tf-btn"        onclick="setTf(this,'6M')">6M</button>
      <button class="tf-btn"        onclick="setTf(this,'3M')">3M</button>
      <button class="tf-btn"        onclick="setTf(this,'1M')">1M</button>
      <button class="tf-btn"        onclick="setTf(this,'2W')">2W</button>
      <button class="tf-btn"        onclick="setTf(this,'1W')">1W</button>
    </div>
  </div>

  <div id="chart-container"></div>
  <div id="no-balance-data">No balance history yet — run the bot to start recording.</div>
</div>

<!-- ===== Trade Tab ===== -->
<div id="tab-trades" class="tab-panel">
  <div id="trade-filters">
    <div class="filter-group">
      <label>Symbol</label>
      <input type="text" id="f-symbol" placeholder="e.g. AAPL" oninput="applyFilters()"/>
    </div>
    <div class="filter-group">
      <label>Side</label>
      <select id="f-side" onchange="applyFilters()">
        <option value="">All</option>
        <option value="BUY">BUY</option>
        <option value="SELL">SELL</option>
      </select>
    </div>
    <div class="filter-group">
      <label>Year</label>
      <select id="f-year" onchange="applyFilters()">
        <option value="">All</option>
      </select>
    </div>
    <div class="filter-group">
      <label>Month</label>
      <select id="f-month" onchange="applyFilters()">
        <option value="">All</option>
        <option value="01">Jan</option><option value="02">Feb</option>
        <option value="03">Mar</option><option value="04">Apr</option>
        <option value="05">May</option><option value="06">Jun</option>
        <option value="07">Jul</option><option value="08">Aug</option>
        <option value="09">Sep</option><option value="10">Oct</option>
        <option value="11">Nov</option><option value="12">Dec</option>
      </select>
    </div>
  </div>

  <div id="trade-count"></div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th onclick="sortBy('date')"     data-col="date">     Date     <span class="sort-icon">↕</span></th>
          <th onclick="sortBy('symbol')"   data-col="symbol">   Symbol   <span class="sort-icon">↕</span></th>
          <th onclick="sortBy('side')"     data-col="side">     Side     <span class="sort-icon">↕</span></th>
          <th onclick="sortBy('quantity')" data-col="quantity"> Qty      <span class="sort-icon">↕</span></th>
          <th onclick="sortBy('price')"    data-col="price">    Price    <span class="sort-icon">↕</span></th>
          <th onclick="sortBy('value')"    data-col="value">    Value    <span class="sort-icon">↕</span></th>
        </tr>
      </thead>
      <tbody id="trades-body"></tbody>
    </table>
  </div>
  <div id="pagination"></div>
</div>

<script>
  const balanceData   = __BALANCE_DATA__;
  const allTrades     = __TRADE_DATA__;
  const ROWS_PER_PAGE = __ROWS_PER_PAGE__;

  // ── Tabs ──────────────────────────────────────────────────────────────────
  function switchTab(btn, name) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn') .forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
  }

  // ── Balance chart ─────────────────────────────────────────────────────────
  const seriesVisible = { total: true, market: true, cash: true };
  let chart, totalSeries, marketSeries, cashSeries;

  function fmtDollar(n) {
    if (n == null) return '—';
    return '$' + (+n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function initChart() {
    const container = document.getElementById('chart-container');
    if (!balanceData.length) {
      container.style.display = 'none';
      document.getElementById('no-balance-data').style.display = 'flex';
      return;
    }

    chart = LightweightCharts.createChart(container, {
      width:  container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { color: '#0f1117' },
        textColor:  '#7f8c8d',
      },
      grid: {
        vertLines: { color: '#1e2330' },
        horzLines: { color: '#1e2330' },
      },
      crosshair:       { mode: LightweightCharts.CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#2a2f3e' },
      timeScale:       { borderColor: '#2a2f3e', timeVisible: false },
    });

    window.addEventListener('resize', () => {
      chart.applyOptions({ width: container.clientWidth });
    });

    const seriesOpts = { lineWidth: 2, priceLineVisible: false, lastValueVisible: false };
    totalSeries  = chart.addLineSeries({ ...seriesOpts, color: '#2ecc71' });
    marketSeries = chart.addLineSeries({ ...seriesOpts, color: '#3498db' });
    cashSeries   = chart.addLineSeries({ ...seriesOpts, color: '#f0c040' });

    const pts = key => balanceData.map(r => ({ time: r.date, value: r[key] }));
    totalSeries .setData(pts('total_balance'));
    marketSeries.setData(pts('market_value'));
    cashSeries  .setData(pts('cash_balance'));

    chart.timeScale().fitContent();

    // Show last values in legend; update on crosshair move
    function updateLegend(t, m, c) {
      document.getElementById('val-total') .textContent = fmtDollar(t);
      document.getElementById('val-market').textContent = fmtDollar(m);
      document.getElementById('val-cash')  .textContent = fmtDollar(c);
    }

    const last = balanceData[balanceData.length - 1];
    updateLegend(last.total_balance, last.market_value, last.cash_balance);

    chart.subscribeCrosshairMove(param => {
      if (!param.time) {
        updateLegend(last.total_balance, last.market_value, last.cash_balance);
        return;
      }
      updateLegend(
        param.seriesData.get(totalSeries) ?.value,
        param.seriesData.get(marketSeries)?.value,
        param.seriesData.get(cashSeries)  ?.value,
      );
    });
  }

  function toggleSeries(key) {
    seriesVisible[key] = !seriesVisible[key];
    const s = { total: totalSeries, market: marketSeries, cash: cashSeries }[key];
    const l = { total: 'leg-total', market: 'leg-market', cash: 'leg-cash' }[key];
    if (s) s.applyOptions({ visible: seriesVisible[key] });
    document.getElementById(l).classList.toggle('off', !seriesVisible[key]);
  }

  function setTf(btn, tf) {
    document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (!chart || !balanceData.length) return;
    if (tf === 'ALL') { chart.timeScale().fitContent(); return; }
    const days = { '5Y': 1825, '1Y': 365, '6M': 183, '3M': 91, '1M': 30, '2W': 14, '1W': 7 }[tf];
    const to   = new Date(balanceData[balanceData.length - 1].date + 'T00:00:00');
    const from = new Date(to);
    from.setDate(from.getDate() - days);
    chart.timeScale().setVisibleRange({
      from: from.toISOString().split('T')[0],
      to:   to  .toISOString().split('T')[0],
    });
  }

  // ── Trade history ─────────────────────────────────────────────────────────
  let filteredTrades = [];
  let sortCol = 'date';
  let sortDir = -1;   // -1 = desc, 1 = asc
  let curPage = 1;

  function initTrades() {
    const years = [...new Set(allTrades.map(t => t.date.slice(0, 4)))].sort().reverse();
    const sel = document.getElementById('f-year');
    years.forEach(y => {
      const o = document.createElement('option');
      o.value = y; o.textContent = y;
      sel.appendChild(o);
    });
    applyFilters();
  }

  function applyFilters() {
    const sym   = document.getElementById('f-symbol').value.trim().toUpperCase();
    const side  = document.getElementById('f-side').value;
    const year  = document.getElementById('f-year').value;
    const month = document.getElementById('f-month').value;

    filteredTrades = allTrades.filter(t => {
      if (sym   && t.symbol !== sym)             return false;
      if (side  && t.side   !== side)            return false;
      if (year  && !t.date.startsWith(year))     return false;
      if (month && t.date.slice(5, 7) !== month) return false;
      return true;
    });

    _applySort();
    curPage = 1;
    renderTrades();
  }

  function sortBy(col) {
    sortDir = (sortCol === col) ? -sortDir : (col === 'date' ? -1 : 1);
    sortCol = col;
    _applySort();
    curPage = 1;
    renderTrades();
  }

  function _applySort() {
    const numeric = new Set(['quantity', 'price', 'value']);
    filteredTrades.sort((a, b) => {
      const av = numeric.has(sortCol) ? +a[sortCol] : a[sortCol];
      const bv = numeric.has(sortCol) ? +b[sortCol] : b[sortCol];
      return av < bv ? -sortDir : av > bv ? sortDir : 0;
    });
    document.querySelectorAll('th[data-col]').forEach(th => {
      th.classList.remove('sort-asc', 'sort-desc');
      const icon = th.querySelector('.sort-icon');
      if (th.dataset.col === sortCol) {
        th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
        icon.textContent = sortDir === 1 ? '↑' : '↓';
      } else {
        icon.textContent = '↕';
      }
    });
  }

  function renderTrades() {
    const total  = filteredTrades.length;
    const pages  = Math.max(1, Math.ceil(total / ROWS_PER_PAGE));
    curPage      = Math.min(curPage, pages);
    const start  = (curPage - 1) * ROWS_PER_PAGE;
    const slice  = filteredTrades.slice(start, start + ROWS_PER_PAGE);

    const countEl = document.getElementById('trade-count');
    if (total === 0) {
      countEl.textContent = 'No trades match the current filters.';
    } else {
      countEl.textContent =
        `Showing ${start + 1}–${Math.min(start + ROWS_PER_PAGE, total)} of ${total.toLocaleString()} trade${total !== 1 ? 's' : ''}`;
    }

    const tbody = document.getElementById('trades-body');
    if (total === 0) {
      tbody.innerHTML = '<tr class="no-rows"><td colspan="6">No trades match the current filters.</td></tr>';
      document.getElementById('pagination').innerHTML = '';
      return;
    }

    tbody.innerHTML = slice.map(t => `
      <tr>
        <td>${t.date}</td>
        <td><b>${t.symbol}</b></td>
        <td class="side-${t.side.toLowerCase()}">${t.side}</td>
        <td>${(+t.quantity).toLocaleString()}</td>
        <td>$${(+t.price).toFixed(2)}</td>
        <td>$${(+t.value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
      </tr>`).join('');

    renderPagination(pages);
  }

  function renderPagination(pages) {
    if (pages <= 1) { document.getElementById('pagination').innerHTML = ''; return; }

    // Build page list: always show first, last, and current ±2; insert ellipsis for gaps
    const show = new Set([1, pages]);
    for (let i = Math.max(1, curPage - 2); i <= Math.min(pages, curPage + 2); i++) show.add(i);
    const sorted = [...show].sort((a, b) => a - b);
    const items = [];
    sorted.forEach((p, idx) => {
      if (idx > 0 && p - sorted[idx - 1] > 1) items.push(null);   // null = ellipsis
      items.push(p);
    });

    document.getElementById('pagination').innerHTML =
      `<button class="page-btn" onclick="goPage(${curPage - 1})" ${curPage === 1 ? 'disabled' : ''}>‹ Prev</button>` +
      items.map(p => p === null
        ? '<span class="page-ellipsis">…</span>'
        : `<button class="page-btn ${p === curPage ? 'active' : ''}" onclick="goPage(${p})">${p}</button>`
      ).join('') +
      `<button class="page-btn" onclick="goPage(${curPage + 1})" ${curPage === pages ? 'disabled' : ''}>Next ›</button>`;
  }

  function goPage(p) {
    curPage = p;
    renderTrades();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  initChart();
  initTrades();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
