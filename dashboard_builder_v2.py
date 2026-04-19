"""
dashboard_builder_v2.py
=======================
Estratégia SIMPLES e robusta:
- Copia o dashboard_20260416_2004.html integralmente
- Faz substituições cirúrgicas APENAS nos valores que mudam a cada execução
- Nunca gera HTML do zero via LLM

Valores que mudam por execução (substituição direta):
  - Timestamp "Gerado em:"
  - KPI values (6 números)
  - Curva de retorno (traces JS)
  - Heatmap (zRaw)
  - Retornos anuais (3 arrays pequenos)
  - Distribuição mensal (allRets)
  - Factor scores (5 arrays)
  - Tabela posições (HTML tbody)
  - Screener (HTML tbody)
  - OOS (4 números)
  - Análise AI (bloco HTML final)

Valores fixos (NÃO substituídos — mantidos do template):
  - Monte Carlo (10k portfólios — pesado, recalcular é opcional)
  - Correlações
  - Alocação MPT
  - ERC/Comparativo
  - Pizza, Geo, Delta, Screener gráfico
  → Estes podem ser atualizados por execução via substituição pontual se necessário
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

TEMPLATE_FILENAME = "dashboard_20260416_2004.html"

def _find_template() -> Path:
    """Procura o template em vários lugares possíveis."""
    candidatos = [
        Path(__file__).parent / TEMPLATE_FILENAME,
        Path(sys.argv[0]).parent / TEMPLATE_FILENAME,
        Path.cwd() / TEMPLATE_FILENAME,
        Path(__file__).resolve().parent / TEMPLATE_FILENAME,
    ]
    for p in candidatos:
        if p.exists():
            return p
    return Path.cwd() / TEMPLATE_FILENAME

TEMPLATE_PATH = _find_template()


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def _safe_replace(html: str, old: str, new: str, label: str = "") -> str:
    if old in html:
        return html.replace(old, new, 1)
    print(f"⚠️  Substituição não encontrada: {label or old[:60]}")
    return html


def build_dashboard(dados: dict, output_path: str) -> str:
    """
    Gera dashboard injetando dados no template do dia 16.
    
    Args:
        dados: dict com os dados calculados
        output_path: onde salvar o HTML
    
    Returns:
        caminho do arquivo salvo
    """
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template não encontrado: {TEMPLATE_PATH}")
    
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # ─── 1. Timestamp ─────────────────────────────────────────────────────────
    now = dados.get("gerado_em", datetime.now().strftime("%d/%m/%Y %H:%M"))
    periodo = dados.get("periodo_backtest", "")
    html = _safe_replace(html, "Gerado em: 16/04/2026 20:04", f"Gerado em: {now}", "timestamp")
    html = _safe_replace(html, "2023-11-15 → 2026-04-16", periodo, "periodo_backtest")
    
    # ─── 2. KPIs ──────────────────────────────────────────────────────────────
    kpis = dados.get("kpis", {})
    
    def fmt_pct(v, d=1):
        return f"+{v:.{d}f}%" if v >= 0 else f"{v:.{d}f}%"
    
    def kpi_class(v):
        return "pos" if v > 0 else ("neg" if v < 0 else "neu")
    
    # Retorno Realizado
    ret = kpis.get("retorno_pct", 147.5)
    html = _safe_replace(html,
        '>Retorno Realizado</div>\n    <div class="kpi-value pos">\n      +147.5%\n    </div>',
        f'>Retorno Realizado</div>\n    <div class="kpi-value {kpi_class(ret)}">\n      {fmt_pct(ret)}\n    </div>',
        "KPI retorno")
    
    # S&P 500
    sp = kpis.get("sp500_pct", 96.0)
    html = _safe_replace(html,
        '>S&P 500 (mesmo período)</div>\n    <div class="kpi-value pos">\n      +96.0%\n    </div>',
        f'>S&P 500 (mesmo período)</div>\n    <div class="kpi-value {kpi_class(sp)}">\n      {fmt_pct(sp)}\n    </div>',
        "KPI sp500")
    
    # Alpha
    alpha = kpis.get("alpha_pct", 51.5)
    html = _safe_replace(html,
        '>Alpha vs Benchmark</div>\n    <div class="kpi-value pos">\n      +51.5%\n    </div>',
        f'>Alpha vs Benchmark</div>\n    <div class="kpi-value {kpi_class(alpha)}">\n      {fmt_pct(alpha)}\n    </div>',
        "KPI alpha")
    
    # Sharpe
    sharpe = kpis.get("sharpe", 1.19)
    html = _safe_replace(html,
        '>Sharpe Ratio</div>\n    <div class="kpi-value pos">\n      1.19\n    </div>',
        f'>Sharpe Ratio</div>\n    <div class="kpi-value {kpi_class(sharpe - 0.5)}">\n      {sharpe:.2f}\n    </div>',
        "KPI sharpe")
    
    # Max Drawdown
    dd = kpis.get("max_dd_pct", -21.4)
    html = _safe_replace(html,
        '>Max Drawdown</div>\n    <div class="kpi-value neg">-21.4%</div>',
        f'>Max Drawdown</div>\n    <div class="kpi-value neg">{fmt_pct(dd)}</div>',
        "KPI drawdown")
    
    # Capital
    capital = kpis.get("capital_total", 34690)
    html = _safe_replace(html,
        '>Capital Total</div>\n    <div class="kpi-value neu">$34,690</div>',
        f'>Capital Total</div>\n    <div class="kpi-value neu">${capital:,.0f}</div>',
        "KPI capital")
    
    # ─── 3. Curva de retorno ───────────────────────────────────────────────────
    if "js_curva" in dados:
        # Substitui o array traces_curva completo
        m = re.search(r'const traces_curva = \[[\s\S]*?\];(?=\nPlotly\.newPlot)', html)
        if m:
            html = html[:m.start()] + f"const traces_curva = {dados['js_curva']};" + html[m.end():]
        else:
            print("⚠️  traces_curva não encontrado")
    
    # ─── 4. Heatmap ────────────────────────────────────────────────────────────
    if "heatmap_z" in dados:
        m = re.search(r'const zRaw = \[[\s\S]*?\];', html)
        if m:
            html = html[:m.start()] + f"const zRaw = {_j(dados['heatmap_z'])};" + html[m.end():]
    
    if "heatmap_anos" in dados:
        m = re.search(r'const anos = \[.*?\];', html)
        if m:
            html = html[:m.start()] + f"const anos = {_j(dados['heatmap_anos'])};" + html[m.end():]
    
    # ─── 5. Retornos anuais ────────────────────────────────────────────────────
    anual = dados.get("retornos_anuais", {})
    if anual:
        anos = anual.get("anos", [])
        vals = anual.get("valores", [])
        colors = ["#00c176" if v >= 0 else "#ff4757" for v in vals]
        texts = [fmt_pct(v) for v in vals]
        
        html = _safe_replace(html,
            'x: ["2024", "2025", "2026"],\n  y: [-18.02, 29.76, 68.82],',
            f'x: {_j(anos)},\n  y: {_j(vals)},', "anual xy")
        html = _safe_replace(html,
            'marker: { color: ["#ff4757", "#00c176", "#00c176"] },',
            f'marker: {{ color: {_j(colors)} }},', "anual colors")
        html = _safe_replace(html,
            'text: ["-18.0%", "+29.8%", "+68.8%"],',
            f'text: {_j(texts)},', "anual text")
    
    # ─── 6. Distribuição mensal ────────────────────────────────────────────────
    if "dist_mensais" in dados:
        m = re.search(r'const allRets = \[.*?\];', html)
        if m:
            html = html[:m.start()] + f"const allRets = {_j(dados['dist_mensais'])};" + html[m.end():]
    
    # ─── 7. Factor scores ──────────────────────────────────────────────────────
    fs = dados.get("factor_scores", {})
    if fs:
        tickers_fs = fs.get("tickers", [])
        n_fs = len(tickers_fs)
        
        # Substitui array x (ocorre múltiplas vezes no factor scores)
        old_x = '["PBR", "ELPC", "VALE", "BBD", "EMBJ", "NU", "HIMS", "QQQ", "IVV", "IWM", "SCHD", "VNQ"]'
        new_x = _j(tickers_fs)
        # Substitui todas as ocorrências dentro do bloco factors
        html = html.replace(old_x, new_x)
        
        # Arrays y dos fatores
        factor_pairs = [
            ('[63.3, 55.6, 51.8, 55.9, 40.3, 47.2, 64.0, 55.0, 54.4, 54.4, 55.0, 55.8]', _j(fs.get("momentum", []))),
            ('[30.0, 60.2, 25.5, 32.1, 41.9, 51.1, 42.0, 38.8, 41.4, 44.5, 54.0, 38.4]', _j(fs.get("valuation", []))),
            ('[55.1, 58.0, 31.6, 61.1, 45.7, 85.1, 59.4, 50.0, 50.0, 50.0, 50.0, 50.0]', _j(fs.get("qualidade", []))),
            ('[17.9, 37.8, 26.8, 38.0, 31.0, 34.9, 36.9, 18.8, 18.7, 24.0, 35.9, 19.2]', _j(fs.get("volatilidade", []))),
            ('[41.6, 52.9, 33.9, 46.8, 39.7, 54.6, 50.6, 40.7, 41.1, 43.2, 48.7, 40.8]', _j(fs.get("score_total", []))),
        ]
        for old, new in factor_pairs:
            html = html.replace(old, new, 1)
        
        # Corrige o x1: 12-0.5 → n-0.5
        html = html.replace("x1: 12-0.5", f"x1: {n_fs}-0.5", 1)
    
    # ─── 8. Tabela de posições ─────────────────────────────────────────────────
    if "html_posicoes" in dados:
        # Localiza o tbody da tabela de posições e substitui
        m = re.search(r'<tbody>\s*\n\s*<tr>\s*\n\s*<td>PBR</td>', html)
        if m:
            end_m = html.find('</tbody>\n  </div>', m.start())
            if end_m == -1:
                end_m = html.find('</tbody>\n    </table>', m.start())
            if end_m == -1:
                end_m = html.find('</tbody>', m.start())
            if end_m != -1:
                html = html[:m.start()] + f"<tbody>\n{dados['html_posicoes']}\n      </tbody>" + html[end_m + len('</tbody>'):]
    
    # ─── 9. Screener tabela ────────────────────────────────────────────────────
    if "html_screener" in dados:
        m = re.search(r'<tbody>\s*\n\s*<tr>\s*\n\s*<td style="color:#7eb8f7;font-weight:700">MA</td>', html)
        if m:
            end_m = html.find('</tbody></table></div>', m.start())
            if end_m != -1:
                html = (html[:m.start()] + 
                       f"<tbody>\n{dados['html_screener']}\n  </tbody></table></div>" +
                       html[end_m + len('</tbody></table></div>'):])
    
    # ─── 10. OOS ──────────────────────────────────────────────────────────────
    oos = dados.get("oos", {})
    if oos:
        html = _safe_replace(html,
            '<div class="kpi-value neu">0.95</div>',
            f'<div class="kpi-value neu">{oos.get("sharpe_in", 0.95):.2f}</div>', "oos sharpe_in")
        html = _safe_replace(html,
            '<div class="kpi-value" style="color:#ffa502">0.24</div>',
            f'<div class="kpi-value" style="color:#ffa502">{oos.get("sharpe_out", 0.24):.2f}</div>', "oos sharpe_out")
        html = _safe_replace(html,
            '<div class="kpi-value neu">10.4%</div>',
            f'<div class="kpi-value neu">{oos.get("retorno_pct", 10.4):.1f}%</div>', "oos retorno")
        html = _safe_replace(html,
            '<div class="kpi-value neu">21.1%</div>',
            f'<div class="kpi-value neu">{oos.get("vol_pct", 21.1):.1f}%</div>', "oos vol")
    
    # ─── 11. Análise AI ────────────────────────────────────────────────────────
    if "html_resumo_exec" in dados:
        start_ai = html.find('    <div style="max-width:1100px">\n      <p style="color:#a0b0c0;font-size:0.8rem;margin:4px 0;line-height:1.5">PLANO DE REALOCAÇÃO')
        end_ai = html.find('    </div>\n  </div>\n\n</body>', start_ai if start_ai != -1 else 0)
        
        if start_ai != -1 and end_ai != -1:
            end_tag = '    </div>\n  </div>\n\n</body>'
            html = (html[:start_ai] + 
                   f'    <div style="max-width:1100px">\n{dados["html_resumo_exec"]}\n    </div>\n  </div>\n\n</body>' +
                   html[end_ai + len(end_tag):])
    
    # ─── Salva ────────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size // 1024
    print(f"✅ Dashboard gerado: {out} ({size_kb}KB)")
    return str(out)
