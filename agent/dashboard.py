"""
dashboard.py — Dashboard HTML interativo de performance

Gera outputs/dashboard.html com 5 painéis interativos (Plotly):
  1. Curva de retorno acumulado (portfólio vs S&P500)
  2. Heatmap de retornos mensais
  3. Retornos anuais (barras)
  4. Distribuição de retornos mensais (histograma)
  5. Factor Scores por ativo (radar / barras)
  6. Tabela de posições com scores
"""

import os
import json
from datetime import datetime


MESES = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr",
    5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}


def gerar_dashboard(
    backtest_resultado,
    factor_scores: list,
    posicoes: list[dict],
    ativos_dados: list[dict],
    capital_disponivel: float,
    caminho_saida: str = None,
    oportunidades_screener: list = None,
) -> str:
    """
    Gera dashboard HTML completo e salva em outputs/.
    Retorna o caminho do arquivo gerado.
    """
    os.makedirs("outputs", exist_ok=True)
    if not caminho_saida:
        agora = datetime.now().strftime("%Y%m%d_%H%M")
        caminho_saida = f"outputs/dashboard_{agora}.html"

    bt = backtest_resultado

    # ── Prepara dados para os gráficos ───────────────────────────

    # 1. Curva de retorno
    datas_port = [d for d, _ in bt.curva_portfolio]
    vals_port = [v for _, v in bt.curva_portfolio]
    datas_sp = [d for d, _ in bt.curva_sp500]
    vals_sp = [v for _, v in bt.curva_sp500]

    # Curvas por ativo individual
    curvas_ativos = {}
    for r in bt.resultados_por_ativo:
        if r.curva_retorno:
            base = r.curva_retorno[0][1]
            curvas_ativos[r.ticker] = {
                "datas": [d for d, _ in r.curva_retorno],
                "vals": [round(v / base * 100, 2) for _, v in r.curva_retorno]
            }

    # 2. Heatmap mensal — limita últimos 15 anos para legibilidade
    todos_anos = sorted(bt.retornos_mensais_portfolio.keys())
    anos = todos_anos[-15:] if len(todos_anos) > 15 else todos_anos
    heatmap_z = []
    heatmap_y = [str(a) for a in anos]
    heatmap_x = [MESES[m] for m in range(1, 13)]
    for ano in anos:
        linha = [bt.retornos_mensais_portfolio.get(ano, {}).get(m) for m in range(1, 13)]
        heatmap_z.append(linha)
    # Altura dinâmica baseada no número de anos
    heatmap_altura = max(300, len(anos) * 28)

    # 3. Retornos anuais
    anos_bar = sorted(bt.retornos_anuais_portfolio.keys())
    vals_bar = [bt.retornos_anuais_portfolio[a] for a in anos_bar]
    cores_bar = ["#00c176" if v >= 0 else "#ff4757" for v in vals_bar]

    # 4. Distribuição de retornos mensais (todos os meses de todos os anos)
    todos_retornos = []
    for ano in bt.retornos_mensais_portfolio:
        for mes in bt.retornos_mensais_portfolio[ano]:
            v = bt.retornos_mensais_portfolio[ano][mes]
            if v is not None:
                todos_retornos.append(v)

    # 5. Factor Scores
    tickers_scores = [s.ticker for s in factor_scores]
    scores_mom = [s.score_momentum for s in factor_scores]
    scores_val = [s.score_valuation for s in factor_scores]
    scores_qual = [s.score_qualidade for s in factor_scores]
    scores_vol = [s.score_volatilidade for s in factor_scores]
    scores_total = [s.score_total for s in factor_scores]
    sinais = [s.sinal for s in factor_scores]

    # 6. Tabela de posições
    valor_total = sum(
        a.get("preco_atual", 0) * p.get("quantidade", 0)
        for a, p in zip(ativos_dados, posicoes)
    ) + capital_disponivel

    tabela_rows = []
    for p, a, s in zip(posicoes, ativos_dados, factor_scores):
        preco_atual = a.get("preco_atual", 0)
        preco_medio = p.get("preco_medio", 0)
        qtd = p.get("quantidade", 0)
        val_pos = preco_atual * qtd
        pnl = ((preco_atual - preco_medio) / preco_medio * 100) if preco_medio else 0
        peso = (val_pos / valor_total * 100) if valor_total else 0
        tabela_rows.append({
            "ticker": p["ticker"],
            "qtd": qtd,
            "pm": f"${preco_medio:.2f}",
            "atual": f"${preco_atual:.2f}",
            "pnl": round(pnl, 2),
            "posicao": f"${val_pos:,.0f}",
            "peso": f"{peso:.1f}%",
            "iv": f"{a.get('iv_atual', 'N/A')}%" if a.get("iv_atual") else "N/A",
            "ivr": f"{a.get('iv_rank_1y', 'N/A')}",
            "score": s.score_total,
            "sinal": s.sinal,
            "bt_ret": next(
                (r.retorno_estrategia for r in bt.resultados_por_ativo if r.ticker == p["ticker"]),
                None
            ),
            "bt_bh": next(
                (r.retorno_buy_hold for r in bt.resultados_por_ativo if r.ticker == p["ticker"]),
                None
            ),
        })

    # 7. Screener de oportunidades
    screener_rows = []
    if oportunidades_screener:
        seg_labels = {"large": "S&P500", "mid": "S&P400", "small": "Small Cap",
                      "adr": "ADR BR", "etf": "ETF"}
        for op in oportunidades_screener[:15]:
            screener_rows.append({
                "ticker": op.ticker,
                "nome": op.nome[:30],
                "setor": op.setor[:18],
                "seg": seg_labels.get(op.segmento, op.segmento.upper()),
                "score": op.score_total,
                "sinal": op.sinal,
                "preco": f"${op.preco_atual:.2f}",
                "upside": op.upside_analistas,
                "mom": op.score_momentum,
                "val": op.score_valuation,
                "qual": op.score_qualidade,
                "vol": op.score_volatilidade,
                "destaque": op.motivo_destaque[:50],
            })

    # ── Gera HTML ─────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agente de Gestão de Portfólio — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'SF Mono', 'Fira Code', monospace;
    background: #0a0e1a;
    color: #e0e6f0;
    min-height: 100vh;
  }}
  .header {{
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 24px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .header h1 {{
    font-size: 1.4rem;
    font-weight: 600;
    color: #7eb8f7;
    letter-spacing: 0.05em;
  }}
  .header .meta {{
    font-size: 0.75rem;
    color: #4a6a8a;
  }}
  .kpi-bar {{
    display: flex;
    gap: 1px;
    background: #0d1220;
    border-bottom: 1px solid #1e3a5f;
  }}
  .kpi {{
    flex: 1;
    padding: 16px 24px;
    background: #0d1b2a;
    text-align: center;
  }}
  .kpi-label {{ font-size: 0.65rem; color: #4a6a8a; text-transform: uppercase; letter-spacing: 0.1em; }}
  .kpi-value {{ font-size: 1.4rem; font-weight: 700; margin-top: 4px; }}
  .pos {{ color: #00c176; }}
  .neg {{ color: #ff4757; }}
  .neu {{ color: #7eb8f7; }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: #0d1220;
    padding: 1px;
  }}
  .grid-full {{ grid-column: 1 / -1; }}
  .card {{
    background: #0d1b2a;
    padding: 20px;
  }}
  .card-title {{
    font-size: 0.7rem;
    color: #4a6a8a;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
  }}
  .chart {{ width: 100%; height: 300px; }}
  .chart-tall {{ width: 100%; height: 400px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
  }}
  th {{
    color: #4a6a8a;
    font-weight: 500;
    text-align: right;
    padding: 6px 10px;
    border-bottom: 1px solid #1e3a5f;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  th:first-child {{ text-align: left; }}
  td {{
    padding: 8px 10px;
    text-align: right;
    border-bottom: 1px solid #0f1e30;
    color: #c0cfe0;
  }}
  td:first-child {{
    text-align: left;
    font-weight: 600;
    color: #7eb8f7;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }}
  .badge-fc {{ background: #003d1a; color: #00c176; }}
  .badge-c  {{ background: #002a12; color: #00a86b; }}
  .badge-n  {{ background: #1a2030; color: #7eb8f7; }}
  .badge-v  {{ background: #2a1010; color: #ff6b7a; }}
  .badge-fv {{ background: #3d0000; color: #ff4757; }}
  .score-bar {{
    display: inline-block;
    height: 6px;
    border-radius: 3px;
    vertical-align: middle;
    margin-right: 6px;
  }}
  @media (max-width: 768px) {{
    .grid {{ grid-template-columns: 1fr; }}
    .kpi-bar {{ flex-wrap: wrap; }}
    .kpi {{ min-width: 50%; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>📊 Agente de Gestão de Portfólio</h1>
    <div class="meta">Análise híbrida · Fundamentalista + Técnica + Volatilidade · Powered by Claude</div>
  </div>
  <div class="meta" style="text-align:right">
    Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}<br>
    Período backtest: {bt.periodo_inicio} → {bt.periodo_fim}
  </div>
</div>

<div class="kpi-bar">
  <div class="kpi">
    <div class="kpi-label">Retorno Portfólio</div>
    <div class="kpi-value {'pos' if bt.retorno_portfolio_total >= 0 else 'neg'}">
      {'+'if bt.retorno_portfolio_total >= 0 else ''}{bt.retorno_portfolio_total:.1f}%
    </div>
  </div>
  <div class="kpi">
    <div class="kpi-label">S&P 500 (mesmo período)</div>
    <div class="kpi-value {'pos' if bt.retorno_sp500_periodo >= 0 else 'neg'}">
      {'+'if bt.retorno_sp500_periodo >= 0 else ''}{bt.retorno_sp500_periodo:.1f}%
    </div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Alpha gerado</div>
    <div class="kpi-value {'pos' if bt.retorno_portfolio_total - bt.retorno_sp500_periodo >= 0 else 'neg'}">
      {'+'if bt.retorno_portfolio_total - bt.retorno_sp500_periodo >= 0 else ''}{bt.retorno_portfolio_total - bt.retorno_sp500_periodo:.1f}%
    </div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Sharpe Ratio</div>
    <div class="kpi-value {'pos' if bt.sharpe_portfolio >= 1 else 'neu' if bt.sharpe_portfolio >= 0 else 'neg'}">
      {bt.sharpe_portfolio:.2f}
    </div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Max Drawdown</div>
    <div class="kpi-value neg">{bt.max_drawdown_portfolio:.1f}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Capital Total</div>
    <div class="kpi-value neu">${valor_total:,.0f}</div>
  </div>
</div>

<div class="grid">

  <!-- 1. Curva de retorno acumulado -->
  <div class="card grid-full">
    <div class="card-title">Retorno Acumulado — Portfólio vs S&P 500</div>
    <div id="chart-curva" class="chart-tall"></div>
  </div>

  <!-- 2. Heatmap mensal -->
  <div class="card">
    <div class="card-title">Retornos Mensais (%) — últimos {len(anos)} anos</div>
    <div id="chart-heatmap" style="width:100%;height:{heatmap_altura}px"></div>
  </div>

  <!-- 3. Retornos anuais -->
  <div class="card">
    <div class="card-title">Retornos Anuais (%)</div>
    <div id="chart-anual" class="chart"></div>

    <!-- 4. Distribuição -->
    <div class="card-title" style="margin-top:20px">Distribuição de Retornos Mensais</div>
    <div id="chart-dist" class="chart"></div>
  </div>

  <!-- 5. Factor Scores -->
  <div class="card grid-full">
    <div class="card-title">Factor Scores por Ativo (0–100)</div>
    <div id="chart-factors" class="chart"></div>
  </div>

  <!-- 6. Tabela de posições -->
  <div class="card grid-full">
    <div class="card-title">Posições — Resumo Completo</div>
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Qtd</th>
          <th>Preço Médio</th>
          <th>Atual</th>
          <th>P&L</th>
          <th>Posição</th>
          <th>% Port.</th>
          <th>IV%</th>
          <th>IVR</th>
          <th>Score</th>
          <th>Sinal</th>
          <th>BT Estratégia</th>
          <th>BT Buy&Hold</th>
        </tr>
      </thead>
      <tbody>
        {''.join(_linha_tabela(r) for r in tabela_rows)}
      </tbody>
    </table>
  </div>


  <!-- 7. Screener de Oportunidades -->
  {'<div class="card grid-full"><div class="card-title">🔍 Screener — Oportunidades Identificadas (fora do portfólio)</div>' + _tabela_screener(screener_rows) + '</div>' if screener_rows else ''}

</div>

<script>
const layout_base = {{
  paper_bgcolor: '#0d1b2a',
  plot_bgcolor: '#0a0e1a',
  font: {{ family: 'SF Mono, Fira Code, monospace', color: '#7090b0', size: 11 }},
  margin: {{ t: 10, b: 40, l: 50, r: 20 }},
  xaxis: {{ gridcolor: '#1a2a3a', linecolor: '#1e3a5f', tickfont: {{ size: 10 }} }},
  yaxis: {{ gridcolor: '#1a2a3a', linecolor: '#1e3a5f', tickfont: {{ size: 10 }} }},
  legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 10 }} }},
  hovermode: 'x unified',
}};

// 1. Curva de retorno
const traces_curva = [
  {{
    x: {json.dumps(datas_port)},
    y: {json.dumps(vals_port)},
    name: 'Portfólio (estratégia)',
    type: 'scatter', mode: 'lines',
    line: {{ color: '#7eb8f7', width: 2 }},
    fill: 'tozeroy', fillcolor: 'rgba(126,184,247,0.05)',
  }},
  {{
    x: {json.dumps(datas_sp)},
    y: {json.dumps(vals_sp)},
    name: 'S&P 500',
    type: 'scatter', mode: 'lines',
    line: {{ color: '#ff6b35', width: 1.5, dash: 'dot' }},
  }},
  {_curvas_ativos_js(curvas_ativos)}
];
Plotly.newPlot('chart-curva', traces_curva, {{
  ...layout_base,
  yaxis: {{ ...layout_base.yaxis, ticksuffix: '', title: 'Base 100' }},
  legend: {{ orientation: 'h', y: -0.15 }},
}}, {{responsive: true}});

// 2. Heatmap mensal
Plotly.newPlot('chart-heatmap', [{{
  z: {json.dumps(heatmap_z)},
  x: {json.dumps(heatmap_x)},
  y: {json.dumps(heatmap_y)},
  type: 'heatmap',
  colorscale: [
    [0, '#7b1515'], [0.3, '#b83030'], [0.45, '#3a2a10'],
    [0.5, '#1a2a1a'], [0.6, '#1a5c2a'], [0.8, '#00a85a'],
    [1, '#00e87a']
  ],
  zmid: 0,
  text: {json.dumps([[f"{{v:.1f}}%" if v is not None else "" for v in row] for row in heatmap_z])},
  texttemplate: '%{{text}}',
  textfont: {{ size: 10 }},
  showscale: true,
  colorbar: {{ thickness: 12, tickfont: {{ size: 10 }}, len: 0.8 }},
  xgap: 2,
  ygap: 2,
}}], {{
  ...layout_base,
  margin: {{ t: 10, b: 40, l: 55, r: 70 }},
  yaxis: {{
    ...layout_base.yaxis,
    tickfont: {{ size: 11, color: '#a0c0e0' }},
    autorange: 'reversed',
  }},
  xaxis: {{
    ...layout_base.xaxis,
    tickfont: {{ size: 11, color: '#a0c0e0' }},
    side: 'top',
  }},
}}, {{responsive: true}});

// 3. Retornos anuais
Plotly.newPlot('chart-anual', [{{
  x: {json.dumps([str(a) for a in anos_bar])},
  y: {json.dumps(vals_bar)},
  type: 'bar',
  marker: {{ color: {json.dumps(cores_bar)} }},
  text: {json.dumps([f"{'+'if v>=0 else ''}{v:.1f}%" for v in vals_bar])},
  textposition: 'outside',
  textfont: {{ size: 10 }},
}}], {{
  ...layout_base,
  yaxis: {{ ...layout_base.yaxis, ticksuffix: '%' }},
  margin: {{ t: 20, b: 30, l: 50, r: 10 }},
}}, {{responsive: true}});

// 4. Distribuição
Plotly.newPlot('chart-dist', [{{
  x: {json.dumps(todos_retornos)},
  type: 'histogram',
  nbinsx: 20,
  marker: {{ color: '#2a4a7f', line: {{ color: '#7eb8f7', width: 1 }} }},
  name: 'Freq. Retornos',
}}, {{
  x: {json.dumps(todos_retornos)},
  type: 'violin',
  side: 'positive',
  fillcolor: 'rgba(126,184,247,0.1)',
  line: {{ color: '#7eb8f7', width: 1 }},
  name: 'Distribuição',
  yaxis: 'y2',
}}], {{
  ...layout_base,
  xaxis: {{ ...layout_base.xaxis, ticksuffix: '%', title: 'Retorno Mensal (%)' }},
  yaxis: {{ ...layout_base.yaxis, title: 'Frequência' }},
  yaxis2: {{ overlaying: 'y', side: 'right', showgrid: false }},
  margin: {{ t: 10, b: 40, l: 50, r: 50 }},
  barmode: 'overlay',
}}, {{responsive: true}});

// 5. Factor Scores — barras agrupadas
Plotly.newPlot('chart-factors', [
  {{ x: {json.dumps(tickers_scores)}, y: {json.dumps(scores_mom)},  name: 'Momentum',    type: 'bar', marker: {{ color: '#7eb8f7' }} }},
  {{ x: {json.dumps(tickers_scores)}, y: {json.dumps(scores_val)},  name: 'Valuation',   type: 'bar', marker: {{ color: '#00c176' }} }},
  {{ x: {json.dumps(tickers_scores)}, y: {json.dumps(scores_qual)}, name: 'Qualidade',   type: 'bar', marker: {{ color: '#ffa502' }} }},
  {{ x: {json.dumps(tickers_scores)}, y: {json.dumps(scores_vol)},  name: 'Volatilidade',type: 'bar', marker: {{ color: '#ff6b7a' }} }},
  {{ x: {json.dumps(tickers_scores)}, y: {json.dumps(scores_total)},name: 'Score Total', type: 'scatter', mode: 'markers',
     marker: {{ size: 12, color: '#fff', symbol: 'diamond', line: {{ color: '#7eb8f7', width: 2 }} }} }},
], {{
  ...layout_base,
  barmode: 'group',
  yaxis: {{ ...layout_base.yaxis, range: [0, 105], title: 'Score (0-100)' }},
  shapes: [
    {{ type: 'line', x0: -0.5, x1: {len(tickers_scores)}-0.5, y0: 70, y1: 70, line: {{ color: '#00c176', dash: 'dot', width: 1 }} }},
    {{ type: 'line', x0: -0.5, x1: {len(tickers_scores)}-0.5, y0: 30, y1: 30, line: {{ color: '#ff4757', dash: 'dot', width: 1 }} }},
  ],
  annotations: [
    {{ x: {len(tickers_scores)-0.5}, y: 70, text: 'Compra', showarrow: false, font: {{ color: '#00c176', size: 9 }}, xanchor: 'right' }},
    {{ x: {len(tickers_scores)-0.5}, y: 30, text: 'Venda',  showarrow: false, font: {{ color: '#ff4757', size: 9 }}, xanchor: 'right' }},
  ],
}}, {{responsive: true}});

</script>
</body>
</html>"""

    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write(html)

    return caminho_saida


def _tabela_screener(rows: list) -> str:
    """Gera tabela HTML do screener."""
    if not rows:
        return ""

    badge_map = {
        "FORTE COMPRA": ("badge-fc", "FORTE COMPRA"),
        "COMPRA": ("badge-c", "COMPRA"),
        "NEUTRO": ("badge-n", "NEUTRO"),
        "VENDA": ("badge-v", "VENDA"),
        "FORTE VENDA": ("badge-fv", "FORTE VENDA"),
    }
    seg_cores = {
        "S&P500": "#7eb8f7", "S&P400": "#a29bfe",
        "Small Cap": "#fd79a8", "ADR BR": "#00c176", "ETF": "#ffa502"
    }

    linhas = []
    for r in rows:
        score = r["score"]
        cor_score = "#00c176" if score >= 70 else "#ffa502" if score >= 55 else "#ff6b7a"
        badge_cls, badge_txt = badge_map.get(r["sinal"], ("badge-n", r["sinal"]))
        cor_seg = seg_cores.get(r["seg"], "#7eb8f7")
        upside = r.get("upside", 0) or 0
        cor_up = "#00c176" if upside >= 0 else "#ff4757"
        largura = int(score * 0.55)

        linhas.append(f"""
    <tr>
      <td style="color:#7eb8f7;font-weight:700">{r['ticker']}</td>
      <td>{r['nome']}</td>
      <td style="color:#8090a0">{r['setor']}</td>
      <td><span style="color:{cor_seg};font-size:0.7rem;font-weight:600">{r['seg']}</span></td>
      <td>
        <span style="display:inline-block;width:{largura}px;height:6px;background:{cor_score};border-radius:3px;vertical-align:middle;margin-right:5px"></span>
        <span style="color:{cor_score}">{score:.0f}</span>
      </td>
      <td><span class="badge {badge_cls}">{badge_txt}</span></td>
      <td style="color:#c0d0e0">{r['preco']}</td>
      <td style="color:{cor_up}">{f"+{upside:.1f}%" if upside >= 0 else f"{upside:.1f}%"}</td>
      <td style="color:#7eb8f7">{r['mom']:.0f}</td>
      <td style="color:#00c176">{r['val']:.0f}</td>
      <td style="color:#ffa502">{r['qual']:.0f}</td>
      <td style="color:#ff6b7a">{r['vol']:.0f}</td>
      <td style="color:#8090a0;font-size:0.72rem">{r['destaque']}</td>
    </tr>""")

    header = """<table width="100%" style="border-collapse:collapse;font-size:0.78rem">
  <thead>
    <tr style="color:#4a6a8a;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em">
      <th style="text-align:left;padding:6px 10px;border-bottom:1px solid #1e3a5f">Ticker</th>
      <th style="text-align:left;padding:6px 10px;border-bottom:1px solid #1e3a5f">Nome</th>
      <th style="text-align:left;padding:6px 10px;border-bottom:1px solid #1e3a5f">Setor</th>
      <th style="text-align:left;padding:6px 10px;border-bottom:1px solid #1e3a5f">Seg.</th>
      <th style="text-align:left;padding:6px 10px;border-bottom:1px solid #1e3a5f">Score</th>
      <th style="text-align:left;padding:6px 10px;border-bottom:1px solid #1e3a5f">Sinal</th>
      <th style="text-align:right;padding:6px 10px;border-bottom:1px solid #1e3a5f">Preço</th>
      <th style="text-align:right;padding:6px 10px;border-bottom:1px solid #1e3a5f">Upside</th>
      <th style="text-align:right;padding:6px 10px;border-bottom:1px solid #1e3a5f">Mom</th>
      <th style="text-align:right;padding:6px 10px;border-bottom:1px solid #1e3a5f">Val</th>
      <th style="text-align:right;padding:6px 10px;border-bottom:1px solid #1e3a5f">Qual</th>
      <th style="text-align:right;padding:6px 10px;border-bottom:1px solid #1e3a5f">Vol</th>
      <th style="text-align:left;padding:6px 10px;border-bottom:1px solid #1e3a5f">Destaque</th>
    </tr>
  </thead>
  <tbody>"""

    return header + "".join(linhas) + "\n  </tbody></table>"


def _linha_tabela(r: dict) -> str:
    pnl = r["pnl"]
    cor_pnl = "#00c176" if pnl >= 0 else "#ff4757"
    sinal = r["sinal"]
    badge = {
        "FORTE COMPRA": "badge-fc",
        "COMPRA": "badge-c",
        "NEUTRO": "badge-n",
        "VENDA": "badge-v",
        "FORTE VENDA": "badge-fv",
    }.get(sinal, "badge-n")

    score = r["score"]
    cor_score = "#00c176" if score >= 70 else "#ffa502" if score >= 40 else "#ff4757"
    largura_barra = int(score * 0.6)

    bt_ret = r.get("bt_ret")
    bt_bh = r.get("bt_bh")
    cor_bt = "#00c176" if bt_ret and bt_ret >= 0 else "#ff4757"
    cor_bh = "#00c176" if bt_bh and bt_bh >= 0 else "#ff4757"

    return f"""
    <tr>
      <td>{r['ticker']}</td>
      <td>{r['qtd']}</td>
      <td>{r['pm']}</td>
      <td>{r['atual']}</td>
      <td style="color:{cor_pnl}">{'+'if pnl>=0 else ''}{pnl:.1f}%</td>
      <td>{r['posicao']}</td>
      <td>{r['peso']}</td>
      <td>{r['iv']}</td>
      <td>{r['ivr']}</td>
      <td>
        <span class="score-bar" style="width:{largura_barra}px;background:{cor_score}"></span>
        <span style="color:{cor_score}">{score:.0f}</span>
      </td>
      <td><span class="badge {badge}">{sinal}</span></td>
      <td style="color:{cor_bt}">{'+'if bt_ret and bt_ret>=0 else ''}{bt_ret:.1f}% if bt_ret is not None else 'N/A'</td>
      <td style="color:{cor_bh}">{'+'if bt_bh and bt_bh>=0 else ''}{bt_bh:.1f}% if bt_bh is not None else 'N/A'</td>
    </tr>"""


def _curvas_ativos_js(curvas: dict) -> str:
    """Gera traces JS para cada ativo individualmente."""
    cores = ["#a29bfe", "#fd79a8", "#55efc4", "#fdcb6e", "#e17055", "#74b9ff"]
    traces = []
    for i, (ticker, dados) in enumerate(curvas.items()):
        cor = cores[i % len(cores)]
        traces.append(f"""{{
    x: {json.dumps(dados['datas'])},
    y: {json.dumps(dados['vals'])},
    name: '{ticker}',
    type: 'scatter', mode: 'lines',
    line: {{ color: '{cor}', width: 1, dash: 'dot' }},
    opacity: 0.6,
    visible: 'legendonly',
  }}""")
    return ",\n  ".join(traces)


def adicionar_paineis_mpt(caminho_html: str, mpt_resultado: dict):
    """
    Injeta painéis MPT no dashboard HTML existente:
      - Scatter Monte Carlo (portfólios coloridos por Sharpe + fronteira)
      - Heatmap de correlações
      - Barras de alocação: atual vs ótimo vs risk parity
    """
    mc = mpt_resultado["monte_carlo"]
    ms = mpt_resultado["max_sharpe"]
    mv = mpt_resultado["min_variancia"]
    rp = mpt_resultado["risk_parity"]
    corr = mpt_resultado["corr_matrix"]
    tickers = mpt_resultado["tickers"]

    # Converte para % para visualização
    vols_pct  = [round(v * 100, 2) for v in mc.volatilidades]
    rets_pct  = [round(r * 100, 2) for r in mc.retornos]
    sharpes   = [round(s, 3) for s in mc.sharpes]

    front_vols = [round(v * 100, 2) for v in mpt_resultado["fronteira_vols"]]
    front_rets = [round(r * 100, 2) for r in mpt_resultado["fronteira_rets"]]

    # Portfólios notáveis
    idx_ms = mc.idx_max_sharpe
    idx_mv = mc.idx_min_vol

    corr_z = corr.values.tolist()
    corr_text = [[f"{v:.2f}" for v in row] for row in corr_z]

    # Alocações
    pesos_atual  = [round(p * 100, 1) for p in ms.pesos_atuais]
    pesos_ms     = [round(p * 100, 1) for p in ms.pesos]
    pesos_mv     = [round(p * 100, 1) for p in mv.pesos]
    pesos_rp     = [round(p * 100, 1) for p in rp.pesos]

    script_mpt = f"""
<script>
// ── Monte Carlo + Fronteira Eficiente ────────────────────────────
Plotly.newPlot('chart-monte-carlo', [
  {{
    x: {json.dumps(vols_pct)},
    y: {json.dumps(rets_pct)},
    mode: 'markers',
    type: 'scatter',
    name: 'Portfólios simulados',
    marker: {{
      color: {json.dumps(sharpes)},
      colorscale: [
        [0, '#1a1a4a'], [0.3, '#2255aa'], [0.5, '#00c176'],
        [0.7, '#ffa502'], [1, '#ff4757']
      ],
      size: 3,
      opacity: 0.6,
      colorbar: {{
        title: 'Sharpe',
        thickness: 10,
        tickfont: {{ size: 9, color: '#7090b0' }},
        titlefont: {{ size: 10, color: '#7090b0' }},
      }},
    }},
    hovertemplate: 'Vol: %{{x:.1f}}%<br>Ret: %{{y:.1f}}%<br>Sharpe: %{{marker.color:.2f}}<extra></extra>',
  }},
  {{
    x: {json.dumps(front_vols)},
    y: {json.dumps(front_rets)},
    mode: 'lines',
    name: 'Fronteira Eficiente',
    line: {{ color: '#7eb8f7', width: 2 }},
    hovertemplate: 'Fronteira<br>Vol: %{{x:.1f}}%<br>Ret: %{{y:.1f}}%<extra></extra>',
  }},
  {{
    x: [{round(vols_pct[idx_ms], 2)}],
    y: [{round(rets_pct[idx_ms], 2)}],
    mode: 'markers', name: '⭐ Máx. Sharpe',
    marker: {{ color: '#ff4757', size: 14, symbol: 'star',
               line: {{ color: '#fff', width: 1 }} }},
    hovertemplate: 'Máx. Sharpe<br>Vol: %{{x:.1f}}%<br>Ret: %{{y:.1f}}%<extra></extra>',
  }},
  {{
    x: [{round(vols_pct[idx_mv], 2)}],
    y: [{round(rets_pct[idx_mv], 2)}],
    mode: 'markers', name: '🛡️ Mín. Variância',
    marker: {{ color: '#ffa502', size: 14, symbol: 'star',
               line: {{ color: '#fff', width: 1 }} }},
    hovertemplate: 'Mín. Variância<br>Vol: %{{x:.1f}}%<br>Ret: %{{y:.1f}}%<extra></extra>',
  }},
  {{
    x: [{round(mc.vol_atual * 100, 2)}],
    y: [{round(mc.retorno_atual * 100, 2)}],
    mode: 'markers', name: '📍 Seu portfólio',
    marker: {{ color: '#a29bfe', size: 16, symbol: 'diamond',
               line: {{ color: '#fff', width: 2 }} }},
    hovertemplate: 'Seu portfólio atual<br>Vol: %{{x:.1f}}%<br>Ret: %{{y:.1f}}%<br>Sharpe: {mc.sharpe_atual:.2f}<extra></extra>',
  }},
], {{
  paper_bgcolor: '#0d1b2a', plot_bgcolor: '#0a0e1a',
  font: {{ family: 'SF Mono, Fira Code, monospace', color: '#7090b0', size: 11 }},
  margin: {{ t: 10, b: 50, l: 60, r: 20 }},
  xaxis: {{ title: 'Volatilidade Anual (%)', gridcolor: '#1a2a3a', ticksuffix: '%' }},
  yaxis: {{ title: 'Retorno Anual (%)', gridcolor: '#1a2a3a', ticksuffix: '%' }},
  legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 10 }}, orientation: 'h', y: -0.2 }},
  hovermode: 'closest',
}}, {{responsive: true}});

// ── Heatmap de Correlações ────────────────────────────────────────
Plotly.newPlot('chart-correlacao', [{{
  z: {json.dumps(corr_z)},
  x: {json.dumps(tickers)},
  y: {json.dumps(tickers)},
  type: 'heatmap',
  colorscale: [
    [0, '#7b1515'], [0.25, '#b83030'], [0.5, '#1a2a3a'],
    [0.75, '#1a5c2a'], [1, '#00c176']
  ],
  zmin: -1, zmax: 1, zmid: 0,
  text: {json.dumps(corr_text)},
  texttemplate: '%{{text}}',
  showscale: true,
  colorbar: {{ thickness: 10, tickfont: {{ size: 9 }}, title: 'Correlação' }},
  hovertemplate: '%{{y}} ↔ %{{x}}<br>Correlação: %{{z:.2f}}<extra></extra>',
}}], {{
  paper_bgcolor: '#0d1b2a', plot_bgcolor: '#0a0e1a',
  font: {{ family: 'SF Mono, Fira Code, monospace', color: '#7090b0', size: 11 }},
  margin: {{ t: 10, b: 60, l: 80, r: 60 }},
}}, {{responsive: true}});

// ── Alocação: Atual vs Ótimos ─────────────────────────────────────
Plotly.newPlot('chart-alocacao', [
  {{ x: {json.dumps(tickers)}, y: {json.dumps(pesos_atual)}, name: 'Atual',
     type: 'bar', marker: {{ color: '#a29bfe' }} }},
  {{ x: {json.dumps(tickers)}, y: {json.dumps(pesos_ms)}, name: '⭐ Máx. Sharpe',
     type: 'bar', marker: {{ color: '#ff4757' }} }},
  {{ x: {json.dumps(tickers)}, y: {json.dumps(pesos_mv)}, name: '🛡️ Mín. Variância',
     type: 'bar', marker: {{ color: '#ffa502' }} }},
  {{ x: {json.dumps(tickers)}, y: {json.dumps(pesos_rp)}, name: '⚖️ Risk Parity',
     type: 'bar', marker: {{ color: '#00c176' }} }},
], {{
  paper_bgcolor: '#0d1b2a', plot_bgcolor: '#0a0e1a',
  font: {{ family: 'SF Mono, Fira Code, monospace', color: '#7090b0', size: 11 }},
  barmode: 'group',
  margin: {{ t: 10, b: 40, l: 50, r: 20 }},
  yaxis: {{ ticksuffix: '%', gridcolor: '#1a2a3a', title: 'Peso (%)' }},
  legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 10 }}, orientation: 'h', y: -0.2 }},
}}, {{responsive: true}});
</script>
"""

    paineis_html = f"""
  <!-- Monte Carlo + Fronteira Eficiente -->
  <div class="card grid-full">
    <div class="card-title">
      Fronteira Eficiente &amp; Monte Carlo (10.000 portfólios simulados)
    </div>
    <div id="chart-monte-carlo" class="chart-tall"></div>
  </div>

  <!-- Heatmap Correlações + Alocação -->
  <div class="card">
    <div class="card-title">Matriz de Correlações</div>
    <div id="chart-correlacao" class="chart-tall"></div>
  </div>
  <div class="card">
    <div class="card-title">Alocação: Atual vs Portfólios Ótimos</div>
    <div id="chart-alocacao" class="chart-tall"></div>
  </div>
"""

    # Injeta no HTML antes de </body>
    with open(caminho_html, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace(
        "</div>\n\n<script>",
        paineis_html + "\n</div>\n\n<script>"
    )
    html = html.replace("</body>", script_mpt + "\n</body>")

    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(html)

    return caminho_html

def adicionar_painel_realocacao(
    caminho_html: str,
    posicoes: list[dict],
    ativos_dados: list[dict],
    oportunidades_screener: list,
    mpt_resultado: dict,
):
    """
    Injeta painel visual de comparação:
    Portfólio Atual vs Portfólio Sugerido (com novas alocações do screener).
    """
    if not oportunidades_screener or not mpt_resultado:
        return caminho_html

    valor_total = sum(
        a.get("preco_atual", 0) * p.get("quantidade", 0)
        for a, p in zip(ativos_dados, posicoes)
    )
    if valor_total == 0:
        return caminho_html

    # ── Portfólio ATUAL ───────────────────────────────────────────
    atual_tickers = [p["ticker"] for p in posicoes]
    atual_pesos = [
        round(a.get("preco_atual", 0) * p.get("quantidade", 0) / valor_total * 100, 1)
        for p, a in zip(posicoes, ativos_dados)
    ]
    atual_pnl = [
        round(((a.get("preco_atual", 0) - p.get("preco_medio", 0)) /
               (p.get("preco_medio", 1)) * 100), 1)
        for p, a in zip(posicoes, ativos_dados)
    ]

    # ── Portfólio SUGERIDO ────────────────────────────────────────
    # Usa pesos do Max Sharpe (MPT) como base
    ms = mpt_resultado["max_sharpe"]
    mpt_tickers = ms.tickers
    mpt_pesos = [round(p * 100, 1) for p in ms.pesos]

    # Identifica ativos a reduzir (delta negativo) e a aumentar
    reducoes = []
    for t, pa, po in zip(mpt_tickers, ms.pesos_atuais, ms.pesos):
        delta = (po - pa) * 100
        if delta < -2:
            val_libera = abs(delta / 100) * valor_total
            reducoes.append({"ticker": t, "delta": round(delta, 1),
                            "valor": round(val_libera, 0)})

    # Top oportunidades do screener
    top_screener = oportunidades_screener[:5]
    screener_tickers = [op.ticker for op in top_screener]
    screener_scores = [op.score_total for op in top_screener]
    screener_upsides = [op.upside_analistas or 0 for op in top_screener]
    screener_setores = [op.setor[:20] for op in top_screener]

    # Capital estimado disponível após reduções
    capital_estimado = sum(r["valor"] for r in reducoes)

    # Distribuição sugerida por região
    tickers_br = {"PBR", "VALE", "BBD", "ITUB", "ABEV", "NU", "ELPC",
                  "EMBJ", "CIG", "SID", "GGB", "ELP", "TIMB", "PAGS"}
    pct_br_atual = sum(
        a.get("preco_atual", 0) * p.get("quantidade", 0) / valor_total * 100
        for p, a in zip(posicoes, ativos_dados)
        if p["ticker"] in tickers_br
    )
    pct_eua_atual = 100 - pct_br_atual

    # Após realocação (estimativa)
    pct_br_novo = max(30, pct_br_atual - 20)
    pct_eua_novo = 100 - pct_br_novo

    script = f"""
<script>
// ── Portfólio Atual — Sunburst/Treemap ────────────────────────────
Plotly.newPlot('chart-atual-pizza', [{{
  type: 'pie',
  labels: {json.dumps(atual_tickers)},
  values: {json.dumps(atual_pesos)},
  hole: 0.45,
  textinfo: 'label+percent',
  textfont: {{ size: 11 }},
  marker: {{
    colors: [
      '#7eb8f7','#a29bfe','#fd79a8','#55efc4','#fdcb6e',
      '#e17055','#74b9ff','#00cec9','#6c5ce7','#fab1a0',
      '#81ecec','#dfe6e9'
    ],
  }},
  hovertemplate: '%{{label}}<br>Peso: %{{value:.1f}}%<br>P&L: ' +
    {json.dumps([f"{v:+.1f}%" for v in atual_pnl])}.join(',').split(',')[{json.dumps(list(range(len(atual_tickers))))}.indexOf('%{{label}}')] +
    '<extra></extra>',
}}], {{
  paper_bgcolor: '#0d1b2a',
  font: {{ family: 'SF Mono, Fira Code, monospace', color: '#7090b0', size: 11 }},
  margin: {{ t: 10, b: 10, l: 10, r: 10 }},
  showlegend: true,
  legend: {{ font: {{ size: 9 }}, bgcolor: 'rgba(0,0,0,0)' }},
  annotations: [{{
    text: 'ATUAL<br>${valor_total:,.0f}',
    x: 0.5, y: 0.5, showarrow: false,
    font: {{ size: 11, color: '#7eb8f7' }}
  }}]
}}, {{responsive: true}});

// ── Oportunidades do Screener — Barras ────────────────────────────
Plotly.newPlot('chart-screener-oport', [
  {{
    x: {json.dumps(screener_tickers)},
    y: {json.dumps(screener_scores)},
    name: 'Factor Score',
    type: 'bar',
    marker: {{ color: '#7eb8f7', opacity: 0.85 }},
    hovertemplate: '%{{x}}<br>Score: %{{y:.0f}}/100<extra></extra>',
  }},
  {{
    x: {json.dumps(screener_tickers)},
    y: {json.dumps(screener_upsides)},
    name: 'Upside Analistas (%)',
    type: 'bar',
    marker: {{ color: '#00c176', opacity: 0.85 }},
    hovertemplate: '%{{x}}<br>Upside: %{{y:.1f}}%<extra></extra>',
    yaxis: 'y2',
  }},
], {{
  paper_bgcolor: '#0d1b2a', plot_bgcolor: '#0a0e1a',
  font: {{ family: 'SF Mono, Fira Code, monospace', color: '#7090b0', size: 11 }},
  barmode: 'group',
  margin: {{ t: 10, b: 60, l: 50, r: 60 }},
  xaxis: {{ gridcolor: '#1a2a3a' }},
  yaxis: {{ gridcolor: '#1a2a3a', title: 'Score (0-100)', range: [0, 105] }},
  yaxis2: {{ overlaying: 'y', side: 'right', title: 'Upside (%)',
             ticksuffix: '%', gridcolor: 'transparent' }},
  legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 10 }},
             orientation: 'h', y: -0.25 }},
  shapes: [{{
    type: 'line', x0: -0.5, x1: {len(screener_tickers)}-0.5,
    y0: 70, y1: 70, yref: 'y',
    line: {{ color: '#00c176', dash: 'dot', width: 1 }}
  }}],
}}, {{responsive: true}});

// ── Exposição Geográfica: Antes vs Depois ────────────────────────
Plotly.newPlot('chart-geo-compare', [
  {{
    type: 'bar', name: 'Antes',
    x: ['Brasil', 'EUA / Global'],
    y: [{round(pct_br_atual, 1)}, {round(pct_eua_atual, 1)}],
    marker: {{ color: ['#ff6b7a', '#7eb8f7'] }},
    text: ['{round(pct_br_atual, 1)}%', '{round(pct_eua_atual, 1)}%'],
    textposition: 'outside',
  }},
  {{
    type: 'bar', name: 'Depois (estimado)',
    x: ['Brasil', 'EUA / Global'],
    y: [{round(pct_br_novo, 1)}, {round(pct_eua_novo, 1)}],
    marker: {{ color: ['#ff6b7a', '#00c176'], opacity: 0.7 }},
    text: ['{round(pct_br_novo, 1)}%', '{round(pct_eua_novo, 1)}%'],
    textposition: 'outside',
  }},
], {{
  paper_bgcolor: '#0d1b2a', plot_bgcolor: '#0a0e1a',
  font: {{ family: 'SF Mono, Fira Code, monospace', color: '#7090b0', size: 11 }},
  barmode: 'group',
  margin: {{ t: 10, b: 40, l: 50, r: 20 }},
  yaxis: {{ gridcolor: '#1a2a3a', ticksuffix: '%', range: [0, 110] }},
  legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 10 }} }},
}}, {{responsive: true}});

// ── Comparativo de Alocação: Atual vs Sugerido MPT ───────────────
Plotly.newPlot('chart-delta-alocacao', [
  {{
    x: {json.dumps(mpt_tickers)},
    y: {json.dumps([round(p*100,1) for p in ms.pesos_atuais])},
    name: '📍 Atual', type: 'bar',
    marker: {{ color: '#a29bfe' }},
  }},
  {{
    x: {json.dumps(mpt_tickers)},
    y: {json.dumps(mpt_pesos)},
    name: '⭐ Sugerido (Máx. Sharpe)', type: 'bar',
    marker: {{ color: '#00c176' }},
  }},
], {{
  paper_bgcolor: '#0d1b2a', plot_bgcolor: '#0a0e1a',
  font: {{ family: 'SF Mono, Fira Code, monospace', color: '#7090b0', size: 11 }},
  barmode: 'group',
  margin: {{ t: 10, b: 50, l: 50, r: 20 }},
  yaxis: {{ gridcolor: '#1a2a3a', ticksuffix: '%', title: 'Peso (%)' }},
  legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 10 }},
             orientation: 'h', y: -0.2 }},
}}, {{responsive: true}});
</script>
"""

    painel_html = f"""
  <!-- Painel Realocação -->
  <div class="card grid-full" style="border-top:2px solid #1e3a5f;margin-top:1px">
    <div class="card-title" style="font-size:0.8rem;color:#7eb8f7">
      📊 ANÁLISE DE REALOCAÇÃO — Portfólio Atual vs Sugerido
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1px;background:#0d1220">

      <div style="background:#0d1b2a;padding:16px">
        <div style="font-size:0.65rem;color:#4a6a8a;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:10px">
          Composição Atual
        </div>
        <div id="chart-atual-pizza" style="height:280px"></div>
      </div>

      <div style="background:#0d1b2a;padding:16px">
        <div style="font-size:0.65rem;color:#4a6a8a;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:10px">
          Top Oportunidades do Screener
        </div>
        <div id="chart-screener-oport" style="height:280px"></div>
      </div>

      <div style="background:#0d1b2a;padding:16px">
        <div style="font-size:0.65rem;color:#4a6a8a;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:10px">
          Exposição Geográfica: Antes vs Depois
        </div>
        <div id="chart-geo-compare" style="height:280px"></div>
      </div>

      <div style="background:#0d1b2a;padding:16px">
        <div style="font-size:0.65rem;color:#4a6a8a;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:10px">
          Alocação: Atual vs Sugerido (MPT)
        </div>
        <div id="chart-delta-alocacao" style="height:280px"></div>
      </div>

    </div>

    <!-- KPIs da realocação -->
    <div style="display:flex;gap:1px;background:#0d1220;margin-top:1px">
      <div style="flex:1;background:#0d1b2a;padding:14px;text-align:center">
        <div style="font-size:0.6rem;color:#4a6a8a;text-transform:uppercase">Capital a Liberar</div>
        <div style="font-size:1.3rem;font-weight:700;color:#ff6b7a;margin-top:4px">
          ${capital_estimado:,.0f}
        </div>
      </div>
      <div style="flex:1;background:#0d1b2a;padding:14px;text-align:center">
        <div style="font-size:0.6rem;color:#4a6a8a;text-transform:uppercase">Sharpe Atual</div>
        <div style="font-size:1.3rem;font-weight:700;color:#7eb8f7;margin-top:4px">
          {mpt_resultado["monte_carlo"].sharpe_atual:.2f}
        </div>
      </div>
      <div style="flex:1;background:#0d1b2a;padding:14px;text-align:center">
        <div style="font-size:0.6rem;color:#4a6a8a;text-transform:uppercase">Sharpe Sugerido</div>
        <div style="font-size:1.3rem;font-weight:700;color:#00c176;margin-top:4px">
          {mpt_resultado["max_sharpe"].sharpe:.2f}
        </div>
      </div>
      <div style="flex:1;background:#0d1b2a;padding:14px;text-align:center">
        <div style="font-size:0.6rem;color:#4a6a8a;text-transform:uppercase">Brasil Atual</div>
        <div style="font-size:1.3rem;font-weight:700;color:#ff6b7a;margin-top:4px">
          {round(pct_br_atual, 0):.0f}%
        </div>
      </div>
      <div style="flex:1;background:#0d1b2a;padding:14px;text-align:center">
        <div style="font-size:0.6rem;color:#4a6a8a;text-transform:uppercase">Brasil Sugerido</div>
        <div style="font-size:1.3rem;font-weight:700;color:#00c176;margin-top:4px">
          {round(pct_br_novo, 0):.0f}%
        </div>
      </div>
      <div style="flex:1;background:#0d1b2a;padding:14px;text-align:center">
        <div style="font-size:0.6rem;color:#4a6a8a;text-transform:uppercase">Oportunidades</div>
        <div style="font-size:1.3rem;font-weight:700;color:#ffa502;margin-top:4px">
          {len(oportunidades_screener)}
        </div>
      </div>
    </div>
  </div>
"""

    with open(caminho_html, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("</body>", painel_html + script + "\n</body>")

    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(html)

    return caminho_html


def adicionar_plano_realocacao_html(caminho_html: str, analise_texto: str):
    """
    Extrai o plano de realocação do texto do Claude e renderiza
    como seção HTML formatada no dashboard.
    """
    if not analise_texto or "PLANO DE REALOCAÇÃO" not in analise_texto:
        return caminho_html

    # Extrai seção do plano
    idx = analise_texto.find("PLANO DE REALOCAÇÃO")
    if idx == -1:
        idx = analise_texto.find("Plano de Realocação")
    if idx == -1:
        return caminho_html

    plano_texto = analise_texto[idx:]

    # Converte markdown para HTML
    import re

    def md_to_html(texto: str) -> str:
        linhas = texto.split("\n")
        html_linhas = []
        in_table = False
        in_list = False

        for linha in linhas:
            # Headers
            if linha.startswith("### "):
                if in_table: html_linhas.append("</table>"); in_table = False
                if in_list: html_linhas.append("</ul>"); in_list = False
                txt = linha[4:].strip()
                # Ícone por tipo
                cor = "#ff6b7a" if "VENDER" in txt else "#00c176" if "COMPRAR" in txt else "#ffa502" if "MANTER" in txt else "#7eb8f7"
                html_linhas.append(f'<h3 style="color:{cor};font-size:0.85rem;margin:18px 0 8px;text-transform:uppercase;letter-spacing:0.08em">{txt}</h3>')

            elif linha.startswith("## "):
                if in_table: html_linhas.append("</table>"); in_table = False
                txt = linha[3:].strip()
                html_linhas.append(f'<h2 style="color:#7eb8f7;font-size:0.95rem;margin:20px 0 10px;border-bottom:1px solid #1e3a5f;padding-bottom:6px">{txt}</h2>')

            # Tabela markdown
            elif linha.startswith("|") and "|" in linha[1:]:
                cells = [c.strip() for c in linha.split("|")[1:-1]]
                if all(set(c.replace("-","").replace(":","").strip()) == set() or c.strip().replace("-","").replace(":","") == "" for c in cells):
                    continue  # linha separadora
                if not in_table:
                    html_linhas.append('<table style="width:100%;border-collapse:collapse;font-size:0.78rem;margin:8px 0">')
                    # Verifica se é header
                    is_header = any(c in ["Ticker","Qtd","Preço","Valor","Antes","Depois","Resultante","Setor","Ação"] for c in cells)
                    in_table = True
                    tag = "th" if is_header else "td"
                    style_header = 'style="color:#4a6a8a;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;padding:6px 10px;border-bottom:1px solid #1e3a5f;text-align:left"'
                    style_td = 'style="padding:7px 10px;border-bottom:1px solid #0f1e30;color:#c0cfe0"'
                    row_html = "<tr>" + "".join(
                        f"<{tag} {style_header if tag=='th' else style_td}>{c}</{tag}>"
                        for c in cells
                    ) + "</tr>"
                    html_linhas.append(row_html)
                else:
                    style_td = 'style="padding:7px 10px;border-bottom:1px solid #0f1e30;color:#c0cfe0"'
                    # Colorize primeira coluna (ticker)
                    row_cells = []
                    for i, c in enumerate(cells):
                        if i == 0:
                            row_cells.append(f'<td style="padding:7px 10px;border-bottom:1px solid #0f1e30;color:#7eb8f7;font-weight:700">{c}</td>')
                        elif "%" in c and ("-" in c or c.startswith("~")):
                            row_cells.append(f'<td style="padding:7px 10px;border-bottom:1px solid #0f1e30;color:#ff6b7a">{c}</td>')
                        elif "%" in c:
                            row_cells.append(f'<td style="padding:7px 10px;border-bottom:1px solid #0f1e30;color:#00c176">{c}</td>')
                        else:
                            row_cells.append(f'<td {style_td}>{c}</td>')
                    html_linhas.append("<tr>" + "".join(row_cells) + "</tr>")

            # Bold total/resultado
            elif linha.startswith("**TOTAL") or linha.startswith("**CAIXA"):
                if in_table: html_linhas.append("</table>"); in_table = False
                txt = linha.replace("**", "")
                cor = "#00c176" if "ALOCADO" in txt or "CAIXA" in txt else "#ff6b7a"
                html_linhas.append(f'<p style="color:{cor};font-weight:700;font-size:0.85rem;margin:8px 0 4px">{txt}</p>')

            # Lista com bullet
            elif linha.startswith("- ") or (re.match(r'^\d+\.', linha)):
                if in_table: html_linhas.append("</table>"); in_table = False
                if not in_list:
                    html_linhas.append('<ul style="list-style:none;padding:0;margin:4px 0">')
                    in_list = True
                txt = re.sub(r'^\d+\.\s*', '', linha[2:] if linha.startswith("- ") else linha)
                # Bold inline
                txt = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#e0e6f0">\1</strong>', txt)
                num = re.match(r'^(\d+)\.', linha)
                num_txt = f'<span style="color:#4a6a8a;margin-right:8px">{num.group(1)}.</span>' if num else '<span style="color:#4a6a8a;margin-right:8px">•</span>'
                html_linhas.append(f'<li style="padding:4px 0;color:#a0b0c0;font-size:0.78rem">{num_txt}{txt}</li>')

            # Separador
            elif linha.strip() == "---":
                if in_table: html_linhas.append("</table>"); in_table = False
                if in_list: html_linhas.append("</ul>"); in_list = False
                html_linhas.append('<hr style="border:none;border-top:1px solid #1e3a5f;margin:12px 0">')

            # Parágrafo normal
            elif linha.strip():
                if in_table: html_linhas.append("</table>"); in_table = False
                if in_list: html_linhas.append("</ul>"); in_list = False
                txt = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#e0e6f0">\1</strong>', linha)
                txt = re.sub(r'\*(.+?)\*', r'<em style="color:#a0b8d0">\1</em>', txt)
                html_linhas.append(f'<p style="color:#a0b0c0;font-size:0.8rem;margin:4px 0;line-height:1.5">{txt}</p>')

            else:
                if in_list: html_linhas.append("</ul>"); in_list = False

        if in_table: html_linhas.append("</table>")
        if in_list: html_linhas.append("</ul>")
        return "\n".join(html_linhas)

    plano_html_content = md_to_html(plano_texto)

    painel = f"""
  <!-- Plano de Realocação Detalhado -->
  <div style="background:#0d1b2a;border-top:2px solid #1e3a5f;padding:24px 28px;margin-top:1px">
    <div style="font-size:0.65rem;color:#4a6a8a;text-transform:uppercase;
                letter-spacing:0.12em;margin-bottom:16px;border-bottom:1px solid #1e3a5f;
                padding-bottom:8px">
      📋 Plano de Realocação — Detalhado (gerado pelo Claude)
    </div>
    <div style="max-width:1100px">
      {plano_html_content}
    </div>
  </div>
"""

    with open(caminho_html, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("</body>", painel + "\n</body>")

    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(html)

    return caminho_html
