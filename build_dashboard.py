"""
build_dashboard.py
--------------------
Gera um dashboard HTML interativo (self-contained) a partir dos
resultados em results/reports/ e results/rankings/.

Uso:
    python build_dashboard.py

Saída:
    results/dashboard.html  -> abra no navegador

Requisitos:
    Apenas pandas (Chart.js é carregado via CDN, requer internet
    para os gráficos renderizarem; os dados ficam embutidos no HTML).
"""

import os
import json
import pandas as pd
import numpy as np

from strategies.documentation import get_strategy_documentation

REPORTS_DIR  = "results/reports"
RANKINGS_DIR = "results/rankings"
OUTPUT_PATH  = "results/dashboard.html"

PROFILES = ["growth", "preservation", "flow", "simplicity", "robustness"]


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[AVISO] Arquivo não encontrado: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _clean_for_json(df: pd.DataFrame) -> list[dict]:
    """Converte DataFrame para lista de dicts, tratando NaN/Inf para JSON válido."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")


def main():
    print("Gerando dashboard...")

    # --- Carregar dados ---
    all_metrics = _safe_read_csv(os.path.join(REPORTS_DIR, "all_metrics.csv"))
    summary     = _safe_read_csv(os.path.join(REPORTS_DIR, "strategy_summary.csv"))

    rankings = {}
    for p in PROFILES:
        df = _safe_read_csv(os.path.join(RANKINGS_DIR, f"ranking_{p}.csv"))
        rankings[p] = df

    if all_metrics.empty:
        print("Nenhum dado encontrado em results/reports/all_metrics.csv. "
              "Execute 'python run_all.py' primeiro.")
        return

    # --- Listas auxiliares ---
    strategies = sorted(all_metrics["strategy"].dropna().astype(str).unique().tolist())
    assets      = sorted(all_metrics["asset"].dropna().astype(str).unique().tolist())
    periods     = sorted(all_metrics["period"].fillna("full").astype(str).unique().tolist())

    # --- Dados para o JS ---
    docs = {name: get_strategy_documentation(name) for name in strategies}

    data_json = {
        "all_metrics": _clean_for_json(all_metrics),
        "summary":     _clean_for_json(summary) if not summary.empty else [],
        "rankings":    {p: _clean_for_json(df) for p, df in rankings.items() if not df.empty},
        "strategies":  strategies,
        "assets":      assets,
        "periods":     periods,
        "strategy_docs": docs,
    }

    html = _build_html(data_json)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard gerado em: {OUTPUT_PATH}")
    print(f"  {len(strategies)} estratégias × {len(assets)} ativos")
    print("Abra o arquivo no navegador para visualizar.")


def _build_html(data: dict) -> str:
    data_json = json.dumps(data, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Backtest — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-light: #334155;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --accent: #3b82f6;
    --green: #22c55e;
    --red: #ef4444;
    --border: #334155;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .subtitle {{ color: var(--text-muted); font-size: 13px; margin-bottom: 24px; }}

  .controls {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 24px;
    padding: 16px;
    background: var(--panel);
    border-radius: 12px;
    border: 1px solid var(--border);
  }}
  .control-group {{ display: flex; flex-direction: column; gap: 4px; }}
  .control-group label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  select {{
    background: var(--panel-light);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    min-width: 180px;
  }}

  .tabs {{ display: flex; gap: 4px; margin-bottom: 16px; border-bottom: 1px solid var(--border); }}
  .tab {{
    padding: 10px 18px;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 13px;
    border-bottom: 2px solid transparent;
    user-select: none;
  }}
  .tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 16px; }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
  }}
  .card h3 {{ margin: 0 0 4px; font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
  .card .value {{ font-size: 26px; font-weight: 700; }}
  .card .value.green {{ color: var(--green); }}
  .card .value.red {{ color: var(--red); }}

  .chart-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
  .chart-card h3 {{ margin: 0 0 12px; font-size: 14px; }}
  .chart-wrap {{ position: relative; height: 360px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
  th {{
    color: var(--text-muted); font-weight: 600; text-transform: uppercase;
    font-size: 10px; letter-spacing: 0.04em; cursor: pointer; position: sticky; top: 0; background: var(--panel);
  }}
  th:hover {{ color: var(--accent); }}
  tr:hover td {{ background: var(--panel-light); }}
  .table-wrap {{ overflow: auto; max-height: 600px; border-radius: 8px; }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--red); }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600;
  }}
  .badge-recorrente, .badge-baixo {{ background: rgba(34,197,94,.15); color: var(--green); }}
  .badge-moderado, .badge-médio {{ background: rgba(234,179,8,.15); color: #eab308; }}
  .badge-esporádico, .badge-alto {{ background: rgba(239,68,68,.15); color: var(--red); }}
  .footer {{ margin-top: 32px; color: var(--text-muted); font-size: 12px; text-align: center; }}
</style>
</head>
<body>

<h1>Trading Backtest Framework — Dashboard</h1>
<div class="subtitle">Análise multi-ativo de estratégias sistemáticas — gerado automaticamente</div>

<div class="tabs">
  <div class="tab active" data-tab="overview">Visão Geral</div>
  <div class="tab" data-tab="explorer">Explorador</div>
  <div class="tab" data-tab="rankings">Rankings por Perfil</div>
  <div class="tab" data-tab="strategy">Estratégia x Ativo</div>
</div>

<!-- ===================== OVERVIEW ===================== -->
<div class="tab-content active" id="tab-overview">
  <div class="grid" id="kpi-cards"></div>

  <div class="chart-card">
    <h3>CAGR médio por Estratégia (todos os ativos)</h3>
    <div class="chart-wrap"><canvas id="chart-cagr-by-strategy"></canvas></div>
  </div>

  <div class="grid">
    <div class="chart-card">
      <h3>Sharpe vs Max Drawdown</h3>
      <div class="chart-wrap"><canvas id="chart-sharpe-dd"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Estratégias lucrativas por ativo</h3>
      <div class="chart-wrap"><canvas id="chart-profitable-by-asset"></canvas></div>
    </div>
  </div>
</div>

<!-- ===================== EXPLORER ===================== -->
<div class="tab-content" id="tab-explorer">
  <div class="controls">
    <div class="control-group">
      <label>Ativo</label>
      <select id="filter-asset"><option value="">Todos</option></select>
    </div>
    <div class="control-group">
      <label>Estratégia</label>
      <select id="filter-strategy"><option value="">Todas</option></select>
    </div>
    <div class="control-group">
      <label>Período</label>
      <select id="filter-period"><option value="">Todos</option></select>
    </div>
    <div class="control-group">
      <label>Ordenar por</label>
      <select id="sort-col">
        <option value="cagr">CAGR</option>
        <option value="sharpe_ratio">Sharpe</option>
        <option value="calmar_ratio">Calmar</option>
        <option value="max_drawdown">Max Drawdown</option>
        <option value="profit_factor">Profit Factor</option>
        <option value="n_trades">Nº Trades</option>
        <option value="win_rate">Win Rate</option>
      </select>
    </div>
  </div>

  <div class="table-wrap card">
    <table id="explorer-table"></table>
  </div>
</div>

<!-- ===================== RANKINGS ===================== -->
<div class="tab-content" id="tab-rankings">
  <div class="controls">
    <div class="control-group">
      <label>Perfil de Investidor</label>
      <select id="profile-select">
        <option value="growth">Crescimento</option>
        <option value="preservation">Preservação de Capital</option>
        <option value="flow">Fluxo</option>
        <option value="simplicity">Simplicidade</option>
        <option value="robustness">Robustez</option>
      </select>
    </div>
  </div>
  <div class="table-wrap card">
    <table id="ranking-table"></table>
  </div>
</div>

<!-- ===================== STRATEGY DETAIL ===================== -->
<div class="tab-content" id="tab-strategy">
  <div class="controls">
    <div class="control-group">
      <label>Estratégia</label>
      <select id="detail-strategy"></select>
    </div>
  </div>

  <div class="grid" id="detail-cards"></div>

  <div class="chart-card" id="strategy-doc-card"></div>

  <div class="chart-card">
    <h3>CAGR por Ativo</h3>
    <div class="chart-wrap"><canvas id="chart-detail-cagr"></canvas></div>
  </div>

  <div class="grid">
    <div class="chart-card">
      <h3>Sharpe por Ativo</h3>
      <div class="chart-wrap"><canvas id="chart-detail-sharpe"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Max Drawdown por Ativo</h3>
      <div class="chart-wrap"><canvas id="chart-detail-dd"></canvas></div>
    </div>
  </div>
</div>

<div class="footer">Trading Backtest Framework — dashboard estático gerado por build_dashboard.py</div>

<script>
const DATA = {data_json};

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------
const COLORS = {{
  blue: '#3b82f6', green: '#22c55e', red: '#ef4444', amber: '#eab308',
  purple: '#a855f7', cyan: '#06b6d4', grid: 'rgba(255,255,255,0.06)',
  text: '#94a3b8'
}};

function fmtPct(v, digits=2) {{
  if (v === null || v === undefined || isNaN(v)) return '—';
  return (v * 100).toFixed(digits) + '%';
}}
function fmtNum(v, digits=2) {{
  if (v === null || v === undefined || isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}}
function fmtInt(v) {{
  if (v === null || v === undefined || isNaN(v)) return '—';
  return Math.round(v).toString();
}}
function colorClass(v) {{
  if (v === null || v === undefined || isNaN(v)) return '';
  return v >= 0 ? 'pos' : 'neg';
}}
function badge(label) {{
  if (!label) return '—';
  const cls = String(label).toLowerCase().replace(/\\s+/g, '-');
  return `<span class="badge badge-${{cls}}">${{label}}</span>`;
}}

const baseChartOpts = {{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{
    legend: {{ labels: {{ color: COLORS.text, font: {{ size: 11 }} }} }},
    tooltip: {{ titleFont: {{size: 11}}, bodyFont: {{size: 11}} }}
  }},
  scales: {{
    x: {{ ticks: {{ color: COLORS.text, font: {{ size: 10 }} }}, grid: {{ color: COLORS.grid }} }},
    y: {{ ticks: {{ color: COLORS.text, font: {{ size: 10 }} }}, grid: {{ color: COLORS.grid }} }}
  }}
}};

// ---------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  }});
}});

// ---------------------------------------------------------------
// Populate filter selects
// ---------------------------------------------------------------
function populateSelect(id, options) {{
  const sel = document.getElementById(id);
  options.forEach(o => {{
    const opt = document.createElement('option');
    opt.value = o; opt.textContent = o;
    sel.appendChild(opt);
  }});
}}
populateSelect('filter-asset', DATA.assets);
populateSelect('filter-strategy', DATA.strategies);
populateSelect('filter-period', DATA.periods || ['full']);
populateSelect('detail-strategy', DATA.strategies);

// ---------------------------------------------------------------
// OVERVIEW
// ---------------------------------------------------------------
function renderOverview() {{
  const m = DATA.all_metrics;
  const nProfitable = m.filter(r => r.cagr > 0).length;
  const avgSharpe = m.reduce((a,r) => a + (r.sharpe_ratio || 0), 0) / m.length;
  const avgCagr = m.reduce((a,r) => a + (r.cagr || 0), 0) / m.length;
  const worstDD = Math.min(...m.map(r => r.max_drawdown ?? 0));

  const cards = [
    {{ label: 'Combinações Testadas', value: m.length, cls: '' }},
    {{ label: '% Lucrativas (CAGR > 0)', value: fmtPct(nProfitable / m.length, 1), cls: 'green' }},
    {{ label: 'Sharpe Médio', value: fmtNum(avgSharpe), cls: avgSharpe >= 0 ? 'green' : 'red' }},
    {{ label: 'CAGR Médio', value: fmtPct(avgCagr), cls: avgCagr >= 0 ? 'green' : 'red' }},
    {{ label: 'Pior Drawdown', value: fmtPct(worstDD), cls: 'red' }},
    {{ label: 'Estratégias', value: DATA.strategies.length, cls: '' }},
    {{ label: 'Ativos', value: DATA.assets.length, cls: '' }},
  ];

  document.getElementById('kpi-cards').innerHTML = cards.map(c => `
    <div class="card">
      <h3>${{c.label}}</h3>
      <div class="value ${{c.cls}}">${{c.value}}</div>
    </div>
  `).join('');

  // CAGR médio por estratégia
  const byStrategy = {{}};
  m.forEach(r => {{
    if (!byStrategy[r.strategy]) byStrategy[r.strategy] = [];
    byStrategy[r.strategy].push(r.cagr || 0);
  }});
  const strategies = Object.keys(byStrategy).sort((a,b) =>
    avg(byStrategy[b]) - avg(byStrategy[a]));
  function avg(arr) {{ return arr.reduce((a,b)=>a+b,0) / arr.length; }}

  new Chart(document.getElementById('chart-cagr-by-strategy'), {{
    type: 'bar',
    data: {{
      labels: strategies,
      datasets: [{{
        label: 'CAGR médio',
        data: strategies.map(s => avg(byStrategy[s]) * 100),
        backgroundColor: strategies.map(s => avg(byStrategy[s]) >= 0 ? COLORS.green : COLORS.red),
      }}]
    }},
    options: {{
      ...baseChartOpts,
      indexAxis: 'y',
      plugins: {{ ...baseChartOpts.plugins, legend: {{ display: false }} }},
      scales: {{
        x: {{ ...baseChartOpts.scales.x, title: {{ display: true, text: 'CAGR (%)', color: COLORS.text }} }},
        y: {{ ...baseChartOpts.scales.y, ticks: {{ ...baseChartOpts.scales.y.ticks, autoSkip: false }} }}
      }}
    }}
  }});

  // Sharpe vs Drawdown scatter
  new Chart(document.getElementById('chart-sharpe-dd'), {{
    type: 'scatter',
    data: {{
      datasets: [{{
        label: 'Combinações',
        data: m.map(r => ({{ x: r.max_drawdown * 100, y: r.sharpe_ratio }})),
        backgroundColor: 'rgba(59,130,246,0.5)',
      }}]
    }},
    options: {{
      ...baseChartOpts,
      plugins: {{ ...baseChartOpts.plugins, legend: {{ display: false }} }},
      scales: {{
        x: {{ ...baseChartOpts.scales.x, title: {{ display: true, text: 'Max Drawdown (%)', color: COLORS.text }} }},
        y: {{ ...baseChartOpts.scales.y, title: {{ display: true, text: 'Sharpe Ratio', color: COLORS.text }} }}
      }}
    }}
  }});

  // Estratégias lucrativas por ativo
  const byAsset = {{}};
  DATA.assets.forEach(a => byAsset[a] = 0);
  m.forEach(r => {{ if (r.cagr > 0) byAsset[r.asset] = (byAsset[r.asset] || 0) + 1; }});

  new Chart(document.getElementById('chart-profitable-by-asset'), {{
    type: 'bar',
    data: {{
      labels: Object.keys(byAsset),
      datasets: [{{
        label: 'Nº estratégias com CAGR > 0',
        data: Object.values(byAsset),
        backgroundColor: COLORS.cyan,
      }}]
    }},
    options: {{ ...baseChartOpts, plugins: {{ ...baseChartOpts.plugins, legend: {{ display: false }} }} }}
  }});
}}

// ---------------------------------------------------------------
// EXPLORER
// ---------------------------------------------------------------
function renderExplorer() {{
  const assetF = document.getElementById('filter-asset').value;
  const stratF = document.getElementById('filter-strategy').value;
  const periodF = document.getElementById('filter-period').value;
  const sortCol = document.getElementById('sort-col').value;

  let rows = DATA.all_metrics.filter(r =>
    (!assetF || r.asset === assetF) &&
    (!stratF || r.strategy === stratF) &&
    (!periodF || (r.period || 'full') === periodF)
  );
  rows = rows.slice().sort((a,b) => {{
    const av = a[sortCol] ?? -Infinity, bv = b[sortCol] ?? -Infinity;
    return bv - av;
  }});

  const cols = [
    {{ key: 'strategy', label: 'Estratégia' }},
    {{ key: 'asset', label: 'Ativo' }},
    {{ key: 'cagr', label: 'CAGR', fmt: fmtPct, cls: colorClass }},
    {{ key: 'sharpe_ratio', label: 'Sharpe', fmt: fmtNum, cls: colorClass }},
    {{ key: 'calmar_ratio', label: 'Calmar', fmt: fmtNum, cls: colorClass }},
    {{ key: 'max_drawdown', label: 'Max DD', fmt: fmtPct, cls: colorClass }},
    {{ key: 'profit_factor', label: 'Profit Factor', fmt: fmtNum }},
    {{ key: 'n_trades', label: 'Nº Trades', fmt: fmtInt }},
    {{ key: 'win_rate', label: 'Win Rate', fmt: fmtPct }},
    {{ key: 'pct_months_positive', label: '% Meses +', fmt: fmtPct }},
    {{ key: 'trades_per_year', label: 'Trades/Ano', fmt: (v)=>fmtNum(v,1) }},
    {{ key: 'flow', label: 'Fluxo', fmt: badge }},
    {{ key: 'ruin_class', label: 'Risco Ruína', fmt: badge }},
    {{ key: 'complexity_label', label: 'Complexidade', fmt: badge }},
  ];

  renderTable('explorer-table', cols, rows);
}}

['filter-asset','filter-strategy','filter-period','sort-col'].forEach(id =>
  document.getElementById(id).addEventListener('change', renderExplorer)
);

// ---------------------------------------------------------------
// RANKINGS
// ---------------------------------------------------------------
function renderRankings() {{
  const profile = document.getElementById('profile-select').value;
  const rows = (DATA.rankings[profile] || []).slice(0, 50);

  const scoreKey = 'score_' + profile;
  const cols = [
    {{ key: 'strategy', label: 'Estratégia' }},
    {{ key: 'asset', label: 'Ativo' }},
    {{ key: scoreKey, label: 'Score', fmt: (v)=>fmtNum(v,3) }},
    {{ key: 'cagr', label: 'CAGR', fmt: fmtPct, cls: colorClass }},
    {{ key: 'sharpe_ratio', label: 'Sharpe', fmt: fmtNum, cls: colorClass }},
    {{ key: 'max_drawdown', label: 'Max DD', fmt: fmtPct, cls: colorClass }},
    {{ key: 'pct_months_positive', label: '% Meses +', fmt: fmtPct }},
    {{ key: 'trades_per_year', label: 'Trades/Ano', fmt: (v)=>fmtNum(v,1) }},
    {{ key: 'complexity_label', label: 'Complexidade', fmt: badge }},
    {{ key: 'ruin_class', label: 'Risco Ruína', fmt: badge }},
  ];

  renderTable('ranking-table', cols, rows, true);
}}
document.getElementById('profile-select').addEventListener('change', renderRankings);

// ---------------------------------------------------------------
// STRATEGY DETAIL
// ---------------------------------------------------------------
let detailCharts = [];
function renderDetail() {{
  const strat = document.getElementById('detail-strategy').value;
  const rows = DATA.all_metrics.filter(r => r.strategy === strat);

  const avg = (key) => {{
    const vals = rows.map(r => r[key]).filter(v => v !== null && !isNaN(v));
    return vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
  }};
  const nProfitable = rows.filter(r => r.cagr > 0).length;

  const cards = [
    {{ label: 'CAGR Médio', value: fmtPct(avg('cagr')), cls: avg('cagr') >= 0 ? 'green':'red' }},
    {{ label: 'Sharpe Médio', value: fmtNum(avg('sharpe_ratio')), cls: avg('sharpe_ratio') >= 0 ? 'green':'red' }},
    {{ label: 'Ativos Lucrativos', value: `${{nProfitable}} / ${{rows.length}}`, cls: '' }},
    {{ label: 'Max DD Médio', value: fmtPct(avg('max_drawdown')), cls: 'red' }},
    {{ label: 'Patrimônio Final (10k)', value: fmtNum(avg('final_equity'), 0), cls: avg('final_equity') >= 10000 ? 'green' : 'red' }},
    {{ label: 'Exposição Média', value: fmtPct(avg('avg_exposure')), cls: '' }},
    {{ label: 'Exposição Máxima', value: fmtPct(avg('max_exposure')), cls: '' }},
    {{ label: 'Trades/Ano (média)', value: fmtNum(avg('trades_per_year'),1), cls: '' }},
    {{ label: 'Win Rate Médio', value: fmtPct(avg('win_rate')), cls: '' }},
  ];
  document.getElementById('detail-cards').innerHTML = cards.map(c => `
    <div class="card"><h3>${{c.label}}</h3><div class="value ${{c.cls}}">${{c.value}}</div></div>
  `).join('');
  const doc = DATA.strategy_docs[strat] || {{}};
  document.getElementById('strategy-doc-card').innerHTML = `
    <h3>Descrição da estratégia</h3>
    <p><strong>${{doc.friendly_name || strat}}</strong> — ${{doc.summary || 'Estratégia do framework de backtest.'}}</p>
    <ul>
      <li><strong>Indicadores:</strong> ${{(doc.indicators || []).join(', ')}}</li>
      <li><strong>Entrada:</strong> ${{doc.entry_trigger || 'Ver lógica da estratégia.'}}</li>
      <li><strong>Saída:</strong> ${{doc.exit_trigger || 'Ver lógica da estratégia.'}}</li>
      <li><strong>Parâmetros principais:</strong> ${{doc.parameters || 'Parâmetros configuráveis.'}}</li>
    </ul>
  `;
  // Destroy previous charts
  detailCharts.forEach(c => c.destroy());
  detailCharts = [];

  const labels = rows.map(r => r.asset);

  detailCharts.push(new Chart(document.getElementById('chart-detail-cagr'), {{
    type: 'bar',
    data: {{ labels, datasets: [{{
      label: 'CAGR',
      data: rows.map(r => (r.cagr||0)*100),
      backgroundColor: rows.map(r => r.cagr >= 0 ? COLORS.green : COLORS.red)
    }}]}},
    options: {{ ...baseChartOpts, plugins: {{...baseChartOpts.plugins, legend:{{display:false}}}} }}
  }}));

  detailCharts.push(new Chart(document.getElementById('chart-detail-sharpe'), {{
    type: 'bar',
    data: {{ labels, datasets: [{{
      label: 'Sharpe',
      data: rows.map(r => r.sharpe_ratio),
      backgroundColor: COLORS.blue
    }}]}},
    options: {{ ...baseChartOpts, plugins: {{...baseChartOpts.plugins, legend:{{display:false}}}} }}
  }}));

  detailCharts.push(new Chart(document.getElementById('chart-detail-dd'), {{
    type: 'bar',
    data: {{ labels, datasets: [{{
      label: 'Max Drawdown',
      data: rows.map(r => (r.max_drawdown||0)*100),
      backgroundColor: COLORS.red
    }}]}},
    options: {{ ...baseChartOpts, plugins: {{...baseChartOpts.plugins, legend:{{display:false}}}} }}
  }}));
}}
document.getElementById('detail-strategy').addEventListener('change', renderDetail);

// ---------------------------------------------------------------
// Generic table renderer
// ---------------------------------------------------------------
function renderTable(tableId, cols, rows, showRank=false) {{
  const table = document.getElementById(tableId);
  let html = '<thead><tr>';
  if (showRank) html += '<th>#</th>';
  cols.forEach(c => html += `<th>${{c.label}}</th>`);
  html += '</tr></thead><tbody>';

  rows.forEach((r, i) => {{
    html += '<tr>';
    if (showRank) html += `<td>${{i+1}}</td>`;
    cols.forEach(c => {{
      const raw = r[c.key];
      const val = c.fmt ? c.fmt(raw) : (raw ?? '—');
      const cls = c.cls ? c.cls(raw) : '';
      html += `<td class="${{cls}}">${{val}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody>';
  table.innerHTML = html;
}}

// ---------------------------------------------------------------
// Init
// ---------------------------------------------------------------
renderOverview();
renderExplorer();
renderRankings();
if (DATA.strategies.length) {{
  document.getElementById('detail-strategy').value = DATA.strategies[0];
  renderDetail();
}}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
