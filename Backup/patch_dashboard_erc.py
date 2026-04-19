"""
patch_dashboard_erc.py
Injeta a função adicionar_painel_erc() no agent/dashboard.py.
Adiciona painel com comparativo das 4 estratégias aplicadas ao portfólio real.
"""

CODIGO_FUNCAO = '''

def adicionar_painel_erc(caminho_html: str, mpt_resultado: dict):
    """
    Injeta painel comparativo das 4 estratégias de otimização no dashboard:
      - Tabela: Sharpe / Retorno / Vol / MDD de cada estratégia
      - Barras: alocação sugerida por cada estratégia para o portfólio real
    Segue o mesmo padrão de adicionar_paineis_mpt().
    """
    ms  = mpt_resultado["max_sharpe"]
    mv  = mpt_resultado["min_variancia"]
    rp  = mpt_resultado["risk_parity"]
    erc = mpt_resultado.get("erc")
    mc  = mpt_resultado["monte_carlo"]
    tickers = mpt_resultado["tickers"]

    pesos_atual = [round(p * 100, 1) for p in ms.pesos_atuais]
    pesos_ms    = [round(p * 100, 1) for p in ms.pesos]
    pesos_mv    = [round(p * 100, 1) for p in mv.pesos]
    pesos_rp    = [round(p * 100, 1) for p in rp.pesos]
    pesos_erc   = [round(p * 100, 1) for p in erc.pesos] if erc else pesos_rp

    estrategias = ["Atual", "Máx. Sharpe", "Mín. Variância", "Risk Parity", "ERC"]
    sharpes = [
        round(mc.sharpe_atual, 3),
        round(ms.sharpe, 3),
        round(mv.sharpe, 3),
        round(rp.sharpe, 3),
        round(erc.sharpe, 3) if erc else 0,
    ]
    retornos = [
        round(mc.retorno_atual * 100, 1),
        round(ms.retorno * 100, 1),
        round(mv.retorno * 100, 1),
        round(rp.retorno * 100, 1),
        round(erc.retorno * 100, 1) if erc else 0,
    ]
    vols = [
        round(mc.vol_atual * 100, 1),
        round(ms.volatilidade * 100, 1),
        round(mv.volatilidade * 100, 1),
        round(rp.volatilidade * 100, 1),
        round(erc.volatilidade * 100, 1) if erc else 0,
    ]
    cores_estrategias = ["#7090b0", "#00c176", "#00b4d8", "#a78bfa", "#fb923c"]

    painel_html = """
  <!-- Painel ERC — Comparativo 4 Estratégias -->
  <div class="card" style="grid-column: 1 / -1;">
    <div class="card-title">📊 Análise de Realocação — Comparativo das 4 Estratégias de Otimização</div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 16px;">
      <div id="chart-estrategias-sharpe" style="height:320px;"></div>
      <div id="chart-estrategias-risco" style="height:320px;"></div>
    </div>
    <div id="chart-alocacao-erc" style="height:420px; margin-top:12px;"></div>
  </div>
"""

    script_erc = f"""
<script>
// ── Comparativo Sharpe ────────────────────────────────────────────────
Plotly.newPlot('chart-estrategias-sharpe', [{{
  type: 'bar',
  x: {json.dumps(estrategias)},
  y: {json.dumps(sharpes)},
  marker: {{
    color: {json.dumps(cores_estrategias)},
    line: {{ color: '#ffffff', width: [0,2,0,0,0] }}
  }},
  text: {json.dumps([str(s) for s in sharpes])},
  textposition: 'outside',
  textfont: {{ color: '#c0d0e0', size: 11, family: 'monospace' }},
  hovertemplate: '<b>%{{x}}</b><br>Sharpe: %{{y:.3f}}<extra></extra>',
}}], {{
  paper_bgcolor: '#0d1b2a',
  plot_bgcolor: '#0d1b2a',
  font: {{ color: '#7090b0' }},
  title: {{ text: 'Sharpe Ratio por Estratégia', font: {{ color: '#c0d0e0', size: 13 }} }},
  yaxis: {{ gridcolor: '#1a2a3a', zeroline: false, title: 'Sharpe' }},
  xaxis: {{ tickangle: -15 }},
  margin: {{ l:50, r:20, t:50, b:80 }},
  shapes: [{{
    type: 'line', x0: -0.5, x1: 4.5, y0: 1.0, y1: 1.0,
    line: {{ color: '#7090b0', width: 1, dash: 'dot' }}
  }}],
  annotations: [{{
    x: 4.5, y: 1.0, text: 'Sharpe=1.0', showarrow: false,
    font: {{ color: '#7090b0', size: 9 }}, xanchor: 'right'
  }}]
}}, {{displayModeBar: false}});

// ── Retorno vs Volatilidade ───────────────────────────────────────────
Plotly.newPlot('chart-estrategias-risco', [{{
  type: 'scatter',
  mode: 'markers+text',
  x: {json.dumps(vols)},
  y: {json.dumps(retornos)},
  text: {json.dumps(estrategias)},
  textposition: 'top center',
  textfont: {{ color: '#c0d0e0', size: 10 }},
  marker: {{
    color: {json.dumps(cores_estrategias)},
    size: 14,
    line: {{ color: '#ffffff', width: 1.5 }}
  }},
  hovertemplate: '<b>%{{text}}</b><br>Vol: %{{x:.1f}}%<br>Retorno: %{{y:.1f}}%<extra></extra>',
}}], {{
  paper_bgcolor: '#0d1b2a',
  plot_bgcolor: '#0d1b2a',
  font: {{ color: '#7090b0' }},
  title: {{ text: 'Risk-Return: Retorno × Volatilidade', font: {{ color: '#c0d0e0', size: 13 }} }},
  xaxis: {{ gridcolor: '#1a2a3a', zeroline: false, title: 'Volatilidade Anual (%)' }},
  yaxis: {{ gridcolor: '#1a2a3a', zeroline: false, title: 'Retorno Anual (%)' }},
  margin: {{ l:60, r:20, t:50, b:60 }},
}}, {{displayModeBar: false}});

// ── Alocação por estratégia ───────────────────────────────────────────
Plotly.newPlot('chart-alocacao-erc', [
  {{
    type: 'bar', name: 'Atual',
    x: {json.dumps(tickers)}, y: {json.dumps(pesos_atual)},
    marker: {{ color: '#7090b0' }},
    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>Atual</extra>',
  }},
  {{
    type: 'bar', name: 'Máx. Sharpe',
    x: {json.dumps(tickers)}, y: {json.dumps(pesos_ms)},
    marker: {{ color: '#00c176' }},
    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>Máx. Sharpe</extra>',
  }},
  {{
    type: 'bar', name: 'Mín. Variância',
    x: {json.dumps(tickers)}, y: {json.dumps(pesos_mv)},
    marker: {{ color: '#00b4d8' }},
    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>Mín. Variância</extra>',
  }},
  {{
    type: 'bar', name: 'Risk Parity',
    x: {json.dumps(tickers)}, y: {json.dumps(pesos_rp)},
    marker: {{ color: '#a78bfa' }},
    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>Risk Parity</extra>',
  }},
  {{
    type: 'bar', name: 'ERC',
    x: {json.dumps(tickers)}, y: {json.dumps(pesos_erc)},
    marker: {{ color: '#fb923c' }},
    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>ERC</extra>',
  }},
], {{
  barmode: 'group',
  paper_bgcolor: '#0d1b2a',
  plot_bgcolor: '#0d1b2a',
  font: {{ color: '#7090b0' }},
  title: {{ text: 'Alocação Sugerida por Estratégia — Portfólio Real', font: {{ color: '#c0d0e0', size: 13 }} }},
  yaxis: {{ gridcolor: '#1a2a3a', zeroline: false, title: 'Peso (%)' }},
  xaxis: {{ tickangle: -30 }},
  legend: {{ orientation: 'h', y: -0.25, font: {{ size: 11 }} }},
  margin: {{ l:60, r:20, t:50, b:120 }},
}}, {{displayModeBar: false}});
</script>
"""

    with open(caminho_html, "r", encoding="utf-8") as f:
        html = f.read()

    # Injeta painel HTML antes de </div>\\n\\n<script> (mesmo padrão do adicionar_paineis_mpt)
    if "chart-alocacao-erc" not in html:
        html = html.replace(
            "</div>\n\n<script>",
            painel_html + "\n</div>\n\n<script>"
        )
        html = html.replace("</body>", script_erc + "\n</body>")

    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(html)

    return caminho_html

'''

INSERE_APOS = "def adicionar_plano_realocacao_html(caminho_html: str, analise_texto: str):"


def aplicar_patch():
    path = "agent/dashboard.py"
    with open(path, "r", encoding="utf-8") as f:
        conteudo = f.read()

    if "def adicionar_painel_erc" in conteudo:
        print("- adicionar_painel_erc() já existe em dashboard.py")
        return

    # Adiciona import json se não existir (já deve existir)
    if "import json" not in conteudo:
        conteudo = "import json\n" + conteudo

    conteudo = conteudo.replace(
        INSERE_APOS,
        CODIGO_FUNCAO + "\n" + INSERE_APOS
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print("✅ adicionar_painel_erc() adicionada ao dashboard.py!")


if __name__ == "__main__":
    aplicar_patch()
