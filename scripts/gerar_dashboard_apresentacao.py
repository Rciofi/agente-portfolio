"""
Gera um dashboard de apresentacao com auditoria de consistencia.

Objetivo:
- comparar todas as estrategias calculadas pelo MPT;
- deixar claro o que e realizado (backtest) e o que e estimado;
- nunca renderizar graficos vazios: quando faltar dado, mostra um estado explicito.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent.backtest import backtest_portfolio
from agent.collector import carregar_portfolio, coletar_dados_ativo
from agent.factors import calcular_score
from agent.mpt import baixar_retornos, calcular_portfolios_otimos


@dataclass
class StrategyRow:
    nome: str
    origem: str
    retorno: float | None
    volatilidade: float | None
    sharpe: float | None
    pesos: dict[str, float]
    observacoes: list[str]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if _finite(value) else default


def _pct(value: float | None, digits: int = 1, signed: bool = False) -> str:
    if value is None or not _finite(value):
        return "N/D"
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{float(value) * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 3) -> str:
    if value is None or not _finite(value):
        return "N/D"
    return f"{float(value):.{digits}f}"


def _money(value: float | None) -> str:
    if value is None or not _finite(value):
        return "N/D"
    return f"US$ {float(value):,.2f}"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _plotly_tag() -> str:
    cache = ROOT / ".plotly_cache.js"
    if cache.exists():
        try:
            return f"<script>\n{cache.read_text(encoding='utf-8')}\n</script>"
        except Exception:
            pass
    return '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'


def _portfolio_value(posicoes: list[dict], ativos_dados: list[dict], capital: float) -> float:
    total = _as_float(capital)
    for p, a in zip(posicoes, ativos_dados):
        total += _as_float(a.get("preco_atual")) * _as_float(p.get("quantidade"))
    return total


def _pesos_atuais(posicoes: list[dict], ativos_dados: list[dict], capital: float) -> dict[str, float]:
    total = _portfolio_value(posicoes, ativos_dados, capital)
    if total <= 0:
        return {}
    return {
        str(p.get("ticker", "")).upper(): (
            _as_float(a.get("preco_atual")) * _as_float(p.get("quantidade")) / total
        )
        for p, a in zip(posicoes, ativos_dados)
        if str(p.get("ticker", "")).strip()
    }


def _weights_from_port(port: Any) -> dict[str, float]:
    if not port:
        return {}
    if getattr(port, "pesos_dict", None):
        return {str(k): _as_float(v) for k, v in port.pesos_dict.items()}
    tickers = list(getattr(port, "tickers", []) or [])
    pesos = list(getattr(port, "pesos", []) or [])
    return {str(t): _as_float(w) for t, w in zip(tickers, pesos)}


def _strategy_rows(mpt_resultado: dict | None, pesos_atual: dict[str, float]) -> list[StrategyRow]:
    rows: list[StrategyRow] = []
    if not mpt_resultado:
        rows.append(
            StrategyRow(
                nome="Atual",
                origem="Carteira informada",
                retorno=None,
                volatilidade=None,
                sharpe=None,
                pesos=pesos_atual,
                observacoes=["MPT indisponivel: sem retornos historicos suficientes ou falha na coleta."],
            )
        )
        return rows

    configs = [
        ("Atual", "Carteira informada", mpt_resultado.get("portfolio_atual")),
        ("Max Sharpe", "MPT", mpt_resultado.get("max_sharpe")),
        ("Min Variancia", "MPT", mpt_resultado.get("min_variancia")),
        ("Risk Parity", "MPT", mpt_resultado.get("risk_parity")),
        ("ERC", "MPT", mpt_resultado.get("erc")),
        ("DLS", "Deep portfolio proxy", mpt_resultado.get("dls")),
        ("DeepStatArb", "Deep portfolio proxy", mpt_resultado.get("deepstatarb")),
    ]

    for nome, origem, port in configs:
        obs = []
        if not port:
            obs.append("Estrategia nao retornou alocacao valida.")
            rows.append(StrategyRow(nome, origem, None, None, None, {}, obs))
            continue
        rows.append(
            StrategyRow(
                nome=nome,
                origem=origem,
                retorno=getattr(port, "retorno", None),
                volatilidade=getattr(port, "volatilidade", None),
                sharpe=getattr(port, "sharpe", None),
                pesos=_weights_from_port(port),
                observacoes=obs,
            )
        )
    return rows


def _audit(rows: list[StrategyRow], tickers_base: list[str]) -> tuple[list[dict], list[str]]:
    audit_rows = []
    global_notes: list[str] = []
    base_set = set(tickers_base)

    for row in rows:
        notes = list(row.observacoes)
        pesos = row.pesos
        soma = sum(pesos.values()) if pesos else 0.0
        max_peso = max(pesos.values()) if pesos else 0.0
        negativos = [t for t, w in pesos.items() if w < -1e-8]
        fora_base = [t for t in pesos if t not in base_set]

        if pesos and abs(soma - 1.0) > 0.015:
            notes.append(f"Pesos somam {soma * 100:.1f}%, fora da tolerancia de 1,5 p.p.")
        if negativos:
            notes.append("Ha pesos negativos, inconsistente com estrategia long-only.")
        if fora_base:
            notes.append("Ha tickers fora da carteira analisada.")
        if pesos and max_peso > 0.405 and row.nome not in {"Atual", "DeepStatArb"}:
            notes.append("Concentracao acima do limite configurado de 40%.")
        if row.nome == "DeepStatArb" and max_peso > 0.205:
            notes.append("Concentracao acima do limite configurado de 20% para DeepStatArb.")
        for metric_name, metric_value in [
            ("retorno", row.retorno),
            ("volatilidade", row.volatilidade),
            ("sharpe", row.sharpe),
        ]:
            if metric_value is not None and not _finite(metric_value):
                notes.append(f"Metrica invalida: {metric_name}.")

        status = "OK" if not notes and pesos else "ATENCAO"
        if not pesos:
            status = "INDISPONIVEL"
        audit_rows.append(
            {
                "estrategia": row.nome,
                "status": status,
                "soma_pesos": soma,
                "max_peso": max_peso,
                "retorno": row.retorno,
                "volatilidade": row.volatilidade,
                "sharpe": row.sharpe,
                "notas": notes or ["Consistente: pesos, universo e metricas dentro do esperado."],
            }
        )

    if not any(r["status"] == "OK" for r in audit_rows):
        global_notes.append("Nenhuma estrategia otimizada ficou totalmente consistente; use a tabela de auditoria para explicar a limitacao.")
    if any(r["status"] == "INDISPONIVEL" for r in audit_rows):
        global_notes.append("Estrategias indisponiveis foram mantidas no dashboard com motivo explicito, sem grafico vazio.")
    return audit_rows, global_notes


def _chart_container(chart_id: str, title: str, available: bool, reason: str = "") -> str:
    if available:
        return f'<section class="panel"><h2>{_esc(title)}</h2><div id="{_esc(chart_id)}" class="chart"></div></section>'
    return f"""
    <section class="panel">
      <h2>{_esc(title)}</h2>
      <div class="empty-state">
        <strong>Sem dados suficientes</strong>
        <span>{_esc(reason or "A coleta nao retornou observacoes suficientes para este grafico.")}</span>
      </div>
    </section>
    """


def _strategy_table(rows: list[StrategyRow], audit_rows: list[dict]) -> str:
    audit_by_name = {r["estrategia"]: r for r in audit_rows}
    body = []
    for row in rows:
        audit = audit_by_name.get(row.nome, {})
        status = audit.get("status", "ATENCAO")
        body.append(
            "<tr>"
            f"<td><strong>{_esc(row.nome)}</strong><small>{_esc(row.origem)}</small></td>"
            f"<td>{_pct(row.retorno)}</td>"
            f"<td>{_pct(row.volatilidade)}</td>"
            f"<td>{_num(row.sharpe)}</td>"
            f"<td>{_pct(audit.get('soma_pesos'), digits=1)}</td>"
            f"<td>{_pct(audit.get('max_peso'), digits=1)}</td>"
            f'<td><span class="status {status.lower()}">{_esc(status)}</span></td>'
            "</tr>"
        )
    return "\n".join(body)


def _audit_cards(audit_rows: list[dict]) -> str:
    cards = []
    for row in audit_rows:
        notes = "".join(f"<li>{_esc(n)}</li>" for n in row["notas"])
        cards.append(
            f"""
            <article class="audit-card">
              <div>
                <strong>{_esc(row["estrategia"])}</strong>
                <span class="status {row["status"].lower()}">{_esc(row["status"])}</span>
              </div>
              <ul>{notes}</ul>
            </article>
            """
        )
    return "\n".join(cards)


def _positions_table(posicoes: list[dict], ativos_dados: list[dict], scores: list[Any], total: float) -> str:
    rows = []
    for p, a, s in zip(posicoes, ativos_dados, scores):
        preco = _as_float(a.get("preco_atual"))
        qtd = _as_float(p.get("quantidade"))
        pm = _as_float(p.get("preco_medio"))
        valor = preco * qtd
        peso = valor / total if total > 0 else None
        pnl = ((preco - pm) / pm) if pm > 0 else None
        rows.append(
            "<tr>"
            f"<td><strong>{_esc(p.get('ticker'))}</strong><small>{_esc(a.get('nome', ''))}</small></td>"
            f"<td>{qtd:g}</td>"
            f"<td>{_money(pm)}</td>"
            f"<td>{_money(preco)}</td>"
            f"<td class=\"{'pos' if pnl and pnl >= 0 else 'neg' if pnl and pnl < 0 else ''}\">{_pct(pnl, signed=True)}</td>"
            f"<td>{_money(valor)}</td>"
            f"<td>{_pct(peso)}</td>"
            f"<td>{_num(getattr(s, 'score_total', None), 1)}</td>"
            f"<td>{_esc(getattr(s, 'sinal', 'N/D'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _make_html(
    *,
    posicoes: list[dict],
    ativos_dados: list[dict],
    scores: list[Any],
    bt_resultado: Any,
    mpt_resultado: dict | None,
    rows: list[StrategyRow],
    audit_rows: list[dict],
    global_notes: list[str],
    capital: float,
) -> str:
    total = _portfolio_value(posicoes, ativos_dados, capital)
    tickers = [str(p.get("ticker", "")).upper() for p in posicoes]
    best = max([r for r in rows if r.sharpe is not None and _finite(r.sharpe)], key=lambda r: r.sharpe, default=None)

    curva_x = [d for d, _ in getattr(bt_resultado, "curva_portfolio", [])]
    curva_y = [v for _, v in getattr(bt_resultado, "curva_portfolio", [])]
    sp_x = [d for d, _ in getattr(bt_resultado, "curva_sp500", [])]
    sp_y = [v for _, v in getattr(bt_resultado, "curva_sp500", [])]

    chart_scripts = []
    if curva_x and curva_y:
        chart_scripts.append(
            f"""
            Plotly.newPlot('chart-curva', [
              {{x:{_json(curva_x)}, y:{_json(curva_y)}, type:'scatter', mode:'lines', name:'Carteira realizada', line:{{color:'#1769aa', width:3}}}},
              {{x:{_json(sp_x)}, y:{_json(sp_y)}, type:'scatter', mode:'lines', name:'S&P 500', line:{{color:'#d1495b', width:2, dash:'dot'}}}}
            ], layout('Base 100'), config);
            """
        )

    valid_rows = [r for r in rows if r.pesos]
    if valid_rows:
        chart_scripts.append(
            f"""
            Plotly.newPlot('chart-metricas', [
              {{x:{_json([r.nome for r in valid_rows])}, y:{_json([_as_float(r.sharpe, None) for r in valid_rows])}, type:'bar', name:'Sharpe', marker:{{color:'#1769aa'}}}}
            ], layout('Sharpe estimado'), config);
            """
        )
        all_tickers = sorted({t for r in valid_rows for t in r.pesos})
        traces = []
        for r in valid_rows:
            traces.append(
                {
                    "x": all_tickers,
                    "y": [round(r.pesos.get(t, 0) * 100, 2) for t in all_tickers],
                    "type": "bar",
                    "name": r.nome,
                }
            )
        chart_scripts.append(
            f"Plotly.newPlot('chart-pesos', {_json(traces)}, layout('Peso (%)'), config);"
        )

    mc = mpt_resultado.get("monte_carlo") if mpt_resultado else None
    if mc and getattr(mc, "volatilidades", None) and getattr(mc, "retornos", None):
        chart_scripts.append(
            f"""
            Plotly.newPlot('chart-montecarlo', [{{
              x:{_json([round(v * 100, 2) for v in mc.volatilidades])},
              y:{_json([round(v * 100, 2) for v in mc.retornos])},
              mode:'markers',
              type:'scatter',
              marker:{{color:{_json([round(v, 3) for v in mc.sharpes])}, colorscale:'Viridis', size:4, opacity:0.55, showscale:true}},
              name:'Carteiras simuladas',
              hovertemplate:'Vol: %{{x:.1f}}%<br>Ret: %{{y:.1f}}%<br>Sharpe: %{{marker.color:.2f}}<extra></extra>'
            }}], layout('Retorno anual (%)', 'Volatilidade anual (%)'), config);
            """
        )

    score_data = [
        {
            "ticker": getattr(s, "ticker", t),
            "total": _as_float(getattr(s, "score_total", None)),
            "momentum": _as_float(getattr(s, "score_momentum", None)),
            "valuation": _as_float(getattr(s, "score_valuation", None)),
            "qualidade": _as_float(getattr(s, "score_qualidade", None)),
            "volatilidade": _as_float(getattr(s, "score_volatilidade", None)),
        }
        for s, t in zip(scores, tickers)
    ]
    if score_data:
        chart_scripts.append(
            f"""
            Plotly.newPlot('chart-scores', [
              {{x:{_json([x['ticker'] for x in score_data])}, y:{_json([x['momentum'] for x in score_data])}, type:'bar', name:'Momentum'}},
              {{x:{_json([x['ticker'] for x in score_data])}, y:{_json([x['valuation'] for x in score_data])}, type:'bar', name:'Valuation'}},
              {{x:{_json([x['ticker'] for x in score_data])}, y:{_json([x['qualidade'] for x in score_data])}, type:'bar', name:'Qualidade'}},
              {{x:{_json([x['ticker'] for x in score_data])}, y:{_json([x['volatilidade'] for x in score_data])}, type:'bar', name:'Volatilidade'}},
              {{x:{_json([x['ticker'] for x in score_data])}, y:{_json([x['total'] for x in score_data])}, type:'scatter', mode:'markers', name:'Score total', marker:{{size:12, color:'#111', symbol:'diamond'}}}}
            ], Object.assign(layout('Score 0-100'), {{barmode:'group', yaxis:{{range:[0,105], gridcolor:'#d8dee8'}}}}), config);
            """
        )

    notes_html = "".join(f"<li>{_esc(n)}</li>" for n in global_notes) or "<li>As estrategias disponiveis foram auditadas com pesos e metricas coerentes.</li>"
    scripts = "\n".join(chart_scripts)

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard de Consistencia - Portfolio</title>
  {_plotly_tag()}
  <style>
    :root {{ --ink:#17202a; --muted:#607086; --line:#d8dee8; --panel:#ffffff; --bg:#f5f7fb; --blue:#1769aa; --green:#13795b; --red:#b42318; --amber:#a15c00; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, Arial, sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:28px 34px; background:#ffffff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:0 0 14px; font-size:16px; letter-spacing:0; }}
    p {{ margin:0; color:var(--muted); line-height:1.5; }}
    main {{ padding:18px; display:grid; grid-template-columns:repeat(12, 1fr); gap:14px; }}
    .panel {{ grid-column:span 6; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; }}
    .wide {{ grid-column:1 / -1; }}
    .kpis {{ grid-column:1 / -1; display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; }}
    .kpi {{ background:#ffffff; border:1px solid var(--line); border-radius:8px; padding:16px; }}
    .kpi span, small {{ display:block; color:var(--muted); font-size:12px; margin-top:4px; }}
    .kpi strong {{ font-size:22px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ padding:10px 8px; border-bottom:1px solid var(--line); text-align:right; vertical-align:top; }}
    th:first-child, td:first-child {{ text-align:left; }}
    th {{ color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; }}
    .chart {{ width:100%; height:390px; }}
    .empty-state {{ min-height:220px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; gap:8px; background:#f0f3f8; border:1px dashed #b8c2d1; border-radius:8px; color:var(--muted); padding:20px; }}
    .status {{ display:inline-block; border-radius:6px; padding:4px 8px; font-size:11px; font-weight:700; }}
    .ok {{ background:#d9f2e7; color:var(--green); }}
    .atencao {{ background:#fff0d6; color:var(--amber); }}
    .indisponivel {{ background:#f8d7da; color:var(--red); }}
    .audit-grid {{ display:grid; grid-template-columns:repeat(2, 1fr); gap:10px; }}
    .audit-card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fbfcfe; }}
    .audit-card div {{ display:flex; justify-content:space-between; gap:8px; align-items:center; }}
    .audit-card ul, .notes {{ margin:10px 0 0; padding-left:18px; color:var(--muted); line-height:1.45; }}
    .pos {{ color:var(--green); }}
    .neg {{ color:var(--red); }}
    @media (max-width: 900px) {{ main, .kpis, .audit-grid {{ display:block; }} .panel, .kpi {{ margin-bottom:12px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Dashboard de Consistencia do Portfolio</h1>
    <p>Gerado em {datetime.now().strftime("%d/%m/%Y %H:%M")}. Retornos realizados aparecem como backtest; retornos das estrategias otimizadas sao estimativas anualizadas calculadas a partir do historico disponivel.</p>
  </header>
  <main>
    <section class="kpis">
      <div class="kpi"><strong>{_money(total)}</strong><span>Valor total estimado</span></div>
      <div class="kpi"><strong>{len(posicoes)}</strong><span>Ativos analisados</span></div>
      <div class="kpi"><strong>{_pct(getattr(bt_resultado, "retorno_portfolio_total", None) / 100 if bt_resultado else None, signed=True)}</strong><span>Retorno realizado no backtest</span></div>
      <div class="kpi"><strong>{_num(getattr(bt_resultado, "sharpe_portfolio", None), 2)}</strong><span>Sharpe realizado</span></div>
      <div class="kpi"><strong>{_esc(best.nome if best else "N/D")}</strong><span>Melhor Sharpe estimado consistente</span></div>
    </section>

    <section class="panel wide">
      <h2>Resumo das Estrategias</h2>
      <table>
        <thead><tr><th>Estrategia</th><th>Retorno estimado/ano</th><th>Volatilidade</th><th>Sharpe</th><th>Soma pesos</th><th>Maior peso</th><th>Status</th></tr></thead>
        <tbody>{_strategy_table(rows, audit_rows)}</tbody>
      </table>
      <ul class="notes">{notes_html}</ul>
    </section>

    {_chart_container("chart-curva", "Backtest Realizado: Carteira vs S&P 500", bool(curva_x and curva_y), "Backtest sem curva suficiente.")}
    {_chart_container("chart-metricas", "Sharpe Estimado por Estrategia", bool(valid_rows), "Nenhuma estrategia otimizada disponivel.")}
    {_chart_container("chart-pesos", "Alocacao por Estrategia", bool(valid_rows), "Sem pesos validos para comparar.")}
    {_chart_container("chart-montecarlo", "Monte Carlo e Fronteira de Risco-Retorno", bool(mc and getattr(mc, "volatilidades", None)), "MPT/Monte Carlo indisponivel.")}
    {_chart_container("chart-scores", "Factor Scores por Ativo", bool(score_data), "Scores nao foram calculados.")}

    <section class="panel wide">
      <h2>Auditoria de Consistencia</h2>
      <div class="audit-grid">{_audit_cards(audit_rows)}</div>
    </section>

    <section class="panel wide">
      <h2>Posicoes e Scores</h2>
      <table>
        <thead><tr><th>Ticker</th><th>Qtd</th><th>Preco medio</th><th>Preco atual</th><th>P&L</th><th>Valor</th><th>Peso</th><th>Score</th><th>Sinal</th></tr></thead>
        <tbody>{_positions_table(posicoes, ativos_dados, scores, total)}</tbody>
      </table>
    </section>
  </main>
  <script>
    const config = {{responsive:true, displayModeBar:false}};
    function layout(yTitle, xTitle) {{
      return {{
        paper_bgcolor:'#ffffff',
        plot_bgcolor:'#ffffff',
        font:{{family:'Inter, Arial, sans-serif', color:'#17202a', size:12}},
        margin:{{l:55,r:20,t:20,b:60}},
        xaxis:{{title:xTitle || '', gridcolor:'#edf0f5', linecolor:'#d8dee8'}},
        yaxis:{{title:yTitle || '', gridcolor:'#edf0f5', linecolor:'#d8dee8'}},
        legend:{{orientation:'h', y:-0.2}}
      }};
    }}
    {scripts}
  </script>
</body>
</html>"""


def gerar_dashboard(portfolio_csv: Path, saida: Path | None, simulacoes: int) -> Path:
    df, capital = carregar_portfolio(str(portfolio_csv))
    posicoes = df.to_dict("records")
    ativos_dados = [coletar_dados_ativo(p["ticker"]) for p in posicoes]
    scores = [calcular_score(a.get("ticker", p["ticker"]), a) for p, a in zip(posicoes, ativos_dados)]
    bt_resultado = backtest_portfolio(posicoes, ativos_dados)

    pesos_atual = _pesos_atuais(posicoes, ativos_dados, capital)
    retornos_hist = baixar_retornos([p["ticker"] for p in posicoes], periodo="3y")
    mpt_resultado = None
    if not retornos_hist.empty and pesos_atual:
        mpt_resultado = calcular_portfolios_otimos(retornos_hist, pesos_atual, n_simulacoes=simulacoes)

    rows = _strategy_rows(mpt_resultado, pesos_atual)
    audit_rows, global_notes = _audit(rows, [p["ticker"] for p in posicoes])
    html_doc = _make_html(
        posicoes=posicoes,
        ativos_dados=ativos_dados,
        scores=scores,
        bt_resultado=bt_resultado,
        mpt_resultado=mpt_resultado,
        rows=rows,
        audit_rows=audit_rows,
        global_notes=global_notes,
        capital=capital,
    )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    if saida is None:
        saida = out_dir / f"dashboard_apresentacao_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    saida = saida.resolve()
    saida.write_text(html_doc, encoding="utf-8")
    return saida


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dashboard robusto de apresentacao com auditoria.")
    parser.add_argument("--portfolio", default=str(ROOT / "portfolio.csv"), help="Caminho do portfolio CSV.")
    parser.add_argument("--saida", default=None, help="Caminho do HTML de saida.")
    parser.add_argument("--simulacoes", type=int, default=5000, help="Numero de carteiras no Monte Carlo.")
    args = parser.parse_args()

    saida = Path(args.saida) if args.saida else None
    caminho = gerar_dashboard(Path(args.portfolio), saida, args.simulacoes)
    print(f"Dashboard de apresentacao gerado: {caminho}")


if __name__ == "__main__":
    main()
