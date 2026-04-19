"""
main.py — Agente de Gestão de Portfólio

Uso:
    python main.py                    # análise completa + backtest + dashboard
    python main.py meu_portfolio.csv  # portfólio customizado
    python main.py --dry-run          # só dados e scores, sem Claude
    python main.py --no-backtest      # pula backtest (mais rápido)
"""

import sys
import os
import webbrowser
import http.server
import threading
import socketserver
import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich import box

from agent.collector import coletar_dados_ativo, carregar_portfolio
from agent.analyzer import analisar_portfolio_com_claude
from agent.factors import calcular_score, formatar_scores_para_prompt
from agent.volatility import salvar_snapshot
from agent.backtest import backtest_portfolio
from agent.screener import (
    coletar_universo_paralelo,
    exibir_oportunidades_terminal,
    formatar_screener_para_prompt,
)
from agent.mpt import (
    baixar_retornos, calcular_portfolios_otimos,
    formatar_mpt_para_prompt, exibir_mpt_terminal
)
from agent.dashboard import gerar_dashboard, adicionar_paineis_mpt, adicionar_painel_realocacao, adicionar_plano_realocacao_html, adicionar_painel_erc, adicionar_resumo_estrategias, adicionar_curva_recomendada

def _garantir_template():
    import shutil
    from pathlib import Path
    nome = "dashboard_20260416_2004.html"
    if not Path(nome).exists():
        for c in [Path("outputs") / nome] + list(Path(".").glob(f"**/{nome}")):
            if c.exists() and c != Path(nome):
                shutil.copy(c, nome)
                return
_garantir_template()
from agent.accuracy import (
    avaliar_recomendacoes_pendentes,
    salvar_recomendacoes,
    exibir_historico_acuracia,
)
from agent.reporter import (
    exibir_header,
    exibir_tabela_portfolio,
    exibir_graficos_ascii,
    exibir_historico_snapshots,
    exibir_analise,
    exportar_relatorio,
)

load_dotenv()
console = Console()




def _md_to_html(texto: str) -> str:
    """Converte markdown para HTML estilizado para o dashboard."""
    if not texto:
        return ""
    if "<div" in texto or "<table" in texto[:200]:
        return texto
    import re
    linhas = texto.split("\n")
    out = []
    em_lista = False
    em_tabela = False
    for linha in linhas:
        s = linha.strip()
        if not s:
            if em_lista: out.append("</ul>"); em_lista = False
            if em_tabela: out.append("</table>"); em_tabela = False
            continue
        if re.match(r"^-{3,}$", s):
            if em_lista: out.append("</ul>"); em_lista = False
            out.append('<hr style="border:none;border-top:1px solid #1e3a5f;margin:12px 0">')
            continue
        m = re.match(r"^(#{1,3}) (.+)$", s)
        if m:
            lvl = len(m.group(1))
            txt = _mdi(m.group(2))
            sz = ["1.05rem","0.95rem","0.85rem"][lvl-1]
            out.append(f'<h{lvl+1} style="color:#7eb8f7;font-size:{sz};margin:16px 0 8px;border-bottom:1px solid #1e3a5f;padding-bottom:4px">{txt}</h{lvl+1}>')
            continue
        if s.startswith("|"):
            if re.match(r"^[|\s\-:]+$", s): continue
            if not em_tabela:
                out.append('<table style="width:100%;border-collapse:collapse;font-size:0.78rem;margin:8px 0">')
                em_tabela = True
            cells = [_mdi(c.strip()) for c in s.strip("|").split("|")]
            tds = "".join(f'<td style="padding:6px 10px;border-bottom:1px solid #0f1e30;color:#c0cfe0">{c}</td>' for c in cells)
            out.append(f"<tr>{tds}</tr>")
            continue
        else:
            if em_tabela: out.append("</table>"); em_tabela = False
        if re.match(r"^[-*✅•] ", s):
            if not em_lista: out.append('<ul style="list-style:none;padding:0;margin:4px 0">'); em_lista = True
            out.append(f'<li style="padding:3px 0;color:#a0b0c0;font-size:0.78rem"><span style="color:#4a6a8a;margin-right:8px">•</span>{_mdi(s[2:])}</li>')
            continue
        m2 = re.match(r"^(\d+)\. (.+)$", s)
        if m2:
            if not em_lista: out.append('<ul style="list-style:none;padding:0;margin:4px 0">'); em_lista = True
            out.append(f'<li style="padding:3px 0;color:#a0b0c0;font-size:0.78rem"><span style="color:#4a6a8a;margin-right:8px">{m2.group(1)}.</span>{_mdi(m2.group(2))}</li>')
            continue
        if em_lista: out.append("</ul>"); em_lista = False
        out.append(f'<p style="color:#a0b0c0;font-size:0.8rem;margin:4px 0;line-height:1.5">{_mdi(s)}</p>')
    if em_lista: out.append("</ul>")
    if em_tabela: out.append("</table>")
    return "\n".join(out)


def _mdi(txt: str) -> str:
    import re
    txt = re.sub(r"\*\*(.+?)\*\*", r'<strong style="color:#e0e6f0">\1</strong>', txt)
    txt = re.sub(r"\*(.+?)\*",       r'<em>\1</em>', txt)
    txt = re.sub(r"`(.+?)`",           r'<code style="color:#7eb8f7">\1</code>', txt)
    return txt


def _abrir_dashboard_servidor(caminho_html: str, porta: int = 8765):
    """
    Serve o dashboard via HTTP local para que o Plotly CDN funcione.
    Evita bloqueio de scripts ao abrir como file://.
    """
    import os
    pasta  = os.path.dirname(os.path.abspath(caminho_html))
    arquivo = os.path.basename(caminho_html)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=pasta, **kwargs)
        def log_message(self, format, *args):
            pass  # silencia logs

    # Tenta porta, se ocupada tenta próximas
    httpd = None
    porta_usada = porta
    for p in range(porta, porta + 10):
        try:
            httpd = socketserver.TCPServer(("", p), Handler)
            httpd.allow_reuse_address = True
            porta_usada = p
            break
        except OSError:
            continue
    if httpd is None:
        webbrowser.open(f"file:///{os.path.abspath(caminho_html)}")
        return caminho_html
    t = threading.Thread(target=httpd.serve_forever, daemon=False)
    t.start()
    url = f"http://localhost:{porta_usada}/{arquivo}"
    webbrowser.open(url)
    console.print(f"\n  🌐 [bold cyan]Dashboard:[/bold cyan] [link]{url}[/link]")
    console.print("  [dim]Pressione Ctrl+C para encerrar...[/dim]\n")
    try:
        t.join()
    except KeyboardInterrupt:
        httpd.shutdown()
        console.print("\n  [dim]Servidor encerrado.[/dim]")
    return url


def exibir_factor_scores(factor_scores: list):
    """Exibe tabela de factor scores no terminal."""
    console.print(Rule("[bold cyan]Factor Scores — Análise Quantitativa[/bold cyan]", style="cyan"))
    console.print()

    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan", show_lines=True)
    table.add_column("Ticker", style="bold white", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Momentum", justify="center")
    table.add_column("Valuation", justify="center")
    table.add_column("Qualidade", justify="center")
    table.add_column("Volatilidade", justify="center")
    table.add_column("Sinal", justify="center")

    def cor_score(v):
        if v >= 70: return "bold green"
        if v >= 55: return "green"
        if v >= 45: return "yellow"
        if v >= 30: return "red"
        return "bold red"

    badge_cores = {
        "FORTE COMPRA": "bold green",
        "COMPRA": "green",
        "NEUTRO": "cyan",
        "VENDA": "red",
        "FORTE VENDA": "bold red",
    }

    for s in factor_scores:
        table.add_row(
            s.ticker,
            f"[{cor_score(s.score_total)}]{s.score_total:.0f}[/{cor_score(s.score_total)}]",
            f"[{cor_score(s.score_momentum)}]{s.score_momentum:.0f}[/{cor_score(s.score_momentum)}]",
            f"[{cor_score(s.score_valuation)}]{s.score_valuation:.0f}[/{cor_score(s.score_valuation)}]",
            f"[{cor_score(s.score_qualidade)}]{s.score_qualidade:.0f}[/{cor_score(s.score_qualidade)}]",
            f"[{cor_score(s.score_volatilidade)}]{s.score_volatilidade:.0f}[/{cor_score(s.score_volatilidade)}]",
            f"[{badge_cores.get(s.sinal, 'white')}]{s.sinal}[/{badge_cores.get(s.sinal, 'white')}]",
        )

    console.print(table)
    console.print("  [dim]Score: 0=ruim · 50=neutro · 100=ótimo  |  Compra>70 · Venda<30[/dim]\n")


def exibir_resumo_backtest(bt_resultado):
    """Exibe resumo do backtest no terminal."""
    console.print(Rule("[bold cyan]Resumo do Backtest[/bold cyan]", style="cyan"))
    console.print()

    table = Table(box=box.SIMPLE, header_style="bold dim", show_lines=False)
    table.add_column("Ticker", style="bold white")
    table.add_column("Período", style="dim")
    table.add_column("Estratégia", justify="right")
    table.add_column("Buy & Hold", justify="right")
    table.add_column("S&P 500", justify="right")
    table.add_column("Alpha", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Trades", justify="right")

    def cor(v): return "green" if v >= 0 else "red"

    for r in bt_resultado.resultados_por_ativo:
        if r.periodo_inicio == "N/A":
            continue
        alpha = r.retorno_estrategia - r.retorno_sp500
        table.add_row(
            r.ticker,
            f"{r.periodo_inicio[:4]}→{r.periodo_fim[:4]}",
            f"[{cor(r.retorno_estrategia)}]{r.retorno_estrategia:+.1f}%[/{cor(r.retorno_estrategia)}]",
            f"[{cor(r.retorno_buy_hold)}]{r.retorno_buy_hold:+.1f}%[/{cor(r.retorno_buy_hold)}]",
            f"[{cor(r.retorno_sp500)}]{r.retorno_sp500:+.1f}%[/{cor(r.retorno_sp500)}]",
            f"[{cor(alpha)}]{alpha:+.1f}%[/{cor(alpha)}]",
            f"{r.sharpe_ratio:.2f}",
            f"[red]{r.max_drawdown:.1f}%[/red]",
            f"{r.win_rate:.0f}%",
            str(r.total_trades),
        )

    console.print(table)
    console.print()

    # Resumo portfolio
    alpha_total = bt_resultado.retorno_portfolio_total - bt_resultado.retorno_sp500_periodo
    console.print(
        f"  📊 [bold]Portfolio:[/bold]  "
        f"Estratégia [{cor(bt_resultado.retorno_portfolio_total)}]{bt_resultado.retorno_portfolio_total:+.1f}%[/{cor(bt_resultado.retorno_portfolio_total)}]  |  "
        f"S&P500 [{cor(bt_resultado.retorno_sp500_periodo)}]{bt_resultado.retorno_sp500_periodo:+.1f}%[/{cor(bt_resultado.retorno_sp500_periodo)}]  |  "
        f"Alpha [{cor(alpha_total)}]{alpha_total:+.1f}%[/{cor(alpha_total)}]  |  "
        f"Sharpe {bt_resultado.sharpe_portfolio:.2f}  |  "
        f"Max DD [red]{bt_resultado.max_drawdown_portfolio:.1f}%[/red]"
    )
    console.print()


def exibir_historico_ordens(ordens: list[dict]):
    if not ordens:
        return
    console.print(Rule("[bold cyan]Histórico de Ordens (últimos 30 dias)[/bold cyan]", style="cyan"))
    console.print()
    table = Table(box=box.SIMPLE, header_style="bold dim", show_lines=False)
    table.add_column("Data", style="dim", min_width=16)
    table.add_column("Ticker", style="bold white")
    table.add_column("Ação")
    table.add_column("Qtd", justify="right")
    table.add_column("Preço", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Fonte", style="dim")
    for o in ordens[:20]:
        acao = o.get("acao", "")
        cor = "green" if "BUY" in acao.upper() else "red"
        moeda = "R$" if o.get("moeda") == "BRL" else "$"
        table.add_row(
            o.get("data", "")[:16], o.get("ticker", ""),
            f"[{cor}]{acao}[/{cor}]",
            f"{o.get('quantidade', 0):.0f}",
            f"{moeda}{o.get('preco', 0):.2f}",
            f"{moeda}{o.get('valor_total', 0):,.0f}",
            o.get("fonte", ""),
        )
    console.print(table)
    console.print()


def _garantir_secoes_completas(caminho_dash: str):
    """
    Verifica se o dashboard tem todas as seções e dados corretos.
    Injeta do template base dia 16: ERC+Pizza+Geo+Delta+Screener e Monte Carlo com dados reais.
    """
    TEMPLATE_FILENAME = "dashboard_20260416_2004.html"
    # Marcadores para detectar seções faltando
    SECOES_CHECK = ['id="chart-erc-sharpe"', 'id="chart-atual-pizza"', 'var estrategias']
    # Bloco completo ERC+Pizza+Geo+Delta+Screener no template
    BLOCO_START  = '<div class="card" style="grid-column: 1 / -1;">\n    <div class="card-title">\U0001f4c8 Comparativo de Estratégias'
    ANALISE_CARD = '<div class="card" style="grid-column: 1 / -1; margin-top: 8px;">'
    # Monte Carlo
    MC_START     = "// ── Monte Carlo + Fronteira Eficiente"
    NEXT_SCRIPT  = "\n\n<script>"

    try:
        html = open(caminho_dash, encoding="utf-8").read()
        falta_erc  = any(s not in html for s in SECOES_CHECK)
        mc_zerado  = 'x: [0],' in html and MC_START in html

        if not falta_erc and not mc_zerado:
            return

        # Localizar template base
        from pathlib import Path
        candidatos = [
            Path(caminho_dash).parent / TEMPLATE_FILENAME,
            Path.cwd() / TEMPLATE_FILENAME,
            Path.cwd() / "outputs" / TEMPLATE_FILENAME,
        ]
        try:
            candidatos += list(Path.cwd().glob(f"**/{TEMPLATE_FILENAME}"))
        except Exception:
            pass

        template_path = next((c for c in candidatos if c.exists()), None)
        if not template_path:
            console.print("  [dim yellow]⚠️  Template base não encontrado.[/dim yellow]")
            return

        html16 = template_path.read_text(encoding="utf-8")
        corrigiu = []

        # 1. Injetar bloco ERC+Pizza+Geo+Delta+Screener
        if falta_erc:
            idx_s = html16.find(BLOCO_START)
            idx_e = html16.find(ANALISE_CARD)
            if idx_s > 0 and idx_e > idx_s:
                bloco = html16[idx_s:idx_e]
                # Inserir antes do card de análise ou antes de </body>
                idx_ins = html.find(ANALISE_CARD)
                if idx_ins == -1:
                    idx_ins = html.rfind("</body>")
                if idx_ins > 0:
                    html = html[:idx_ins] + "\n" + bloco + "\n" + html[idx_ins:]
                    corrigiu.append("ERC/Pizza/Geo")

        # 2. Corrigir Monte Carlo zerado
        if mc_zerado:
            idx_mc16_s = html16.find(MC_START)
            idx_mc16_e = html16.find(NEXT_SCRIPT, idx_mc16_s)
            idx_mc_s   = html.find(MC_START)
            idx_mc_e   = html.find(NEXT_SCRIPT, idx_mc_s)
            if all(i > 0 for i in [idx_mc16_s, idx_mc16_e, idx_mc_s, idx_mc_e]):
                html = html[:idx_mc_s] + html16[idx_mc16_s:idx_mc16_e] + html[idx_mc_e:]
                corrigiu.append("Monte Carlo")

        if corrigiu:
            open(caminho_dash, "w", encoding="utf-8").write(html)
            console.print(f"  [dim]📊 Corrigido: {', '.join(corrigiu)}.[/dim]")

    except Exception as e:
        console.print(f"  [dim yellow]⚠️  _garantir_secoes_completas: {e}[/dim yellow]")


def _embutir_plotly(caminho_html: str):
    """Substitui o CDN do Plotly por script inline, tornando o HTML autônomo."""
    import urllib.request
    PLOTLY_TAG = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
    PLOTLY_URL = "https://cdn.plot.ly/plotly-2.27.0.min.js"
    CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".plotly_cache.js")

    try:
        html = open(caminho_html, encoding="utf-8").read()
        if PLOTLY_TAG not in html:
            return  # já embutido ou tag diferente

        # Tentar cache local primeiro
        js = None
        if os.path.exists(CACHE):
            js = open(CACHE, encoding="utf-8").read()
        else:
            try:
                with urllib.request.urlopen(PLOTLY_URL, timeout=15) as r:
                    js = r.read().decode("utf-8")
                open(CACHE, "w", encoding="utf-8").write(js)
            except Exception:
                return  # sem internet — mantém CDN, usa servidor HTTP

        if js:
            html = html.replace(PLOTLY_TAG, f"<script>\n{js}\n</script>", 1)
            open(caminho_html, "w", encoding="utf-8").write(html)
            console.print("  [dim]📦 Plotly embutido — dashboard funciona offline.[/dim]")
    except Exception:
        pass  # falha silenciosa — não bloqueia o fluxo


def main():
    args = sys.argv[1:]
    retornos_hist = None
    dry_run    = "--dry-run" in args
    no_backtest = "--no-backtest" in args
    csv_forcado = next((a for a in args if not a.startswith("--")), None)

    exibir_header()
    if dry_run:
        console.print("  [yellow]⚡ MODO DRY-RUN — sem análise Claude[/yellow]\n")

    # ── 1. Corretoras ─────────────────────────────────────────────
    from agent.brokers.manager import BrokerManager
    broker_manager = BrokerManager()

    # ── 2. Fonte do portfólio ─────────────────────────────────────
    posicoes = []
    capital_disponivel = 0.0
    fonte_portfolio = "CSV"

    if broker_manager.ativo and not csv_forcado:
        console.print("[bold cyan]🔗 Importando portfólio das corretoras...[/bold cyan]")
        posicoes, capital_disponivel = broker_manager.get_portfolio_automatico()
        fonte_portfolio = " + ".join(broker_manager.brokers_ativos)

    if not posicoes:
        csv_path = csv_forcado or "portfolio.csv"
        if not os.path.exists(csv_path):
            console.print(f"[bold red]Arquivo não encontrado:[/bold red] {csv_path}")
            sys.exit(1)
        df, capital_disponivel = carregar_portfolio(csv_path)
        posicoes = df.to_dict("records")
        fonte_portfolio = f"CSV ({csv_path})"

    console.print(
        f"  [cyan]Portfólio:[/cyan] {len(posicoes)} ativo(s)  |  "
        f"Capital livre: [green]${capital_disponivel:,.2f}[/green]  |  "
        f"[dim]Fonte: {fonte_portfolio}[/dim]\n"
    )

    # ── 3. Histórico de ordens ────────────────────────────────────
    if broker_manager.ativo:
        exibir_historico_ordens(broker_manager.get_historico_ordens(30))

    # ── 4. Coleta dados de mercado ────────────────────────────────
    console.print("[bold cyan]📡 Coletando dados de mercado...[/bold cyan]")
    ativos_dados = []
    for p in posicoes:
        dados = coletar_dados_ativo(p["ticker"])
        ativos_dados.append(dados)
    console.print()

    # ── 5. Factor Scores ──────────────────────────────────────────
    factor_scores = [calcular_score(a.get("ticker", p["ticker"]), a)
                     for p, a in zip(posicoes, ativos_dados)]
    exibir_factor_scores(factor_scores)

    # ── 6. Acurácia histórica ─────────────────────────────────────
    n_aval = avaliar_recomendacoes_pendentes(ativos_dados)
    if n_aval > 0:
        console.print(f"  [dim]📈 {n_aval} recomendação(ões) avaliada(s).[/dim]\n")
    exibir_historico_acuracia()

    # ── 7. Tabela posições ────────────────────────────────────────
    exibir_tabela_portfolio(posicoes, ativos_dados, capital_disponivel)
    exibir_historico_snapshots(posicoes)
    exibir_graficos_ascii(posicoes, ativos_dados)

    # ── 8. Backtest ───────────────────────────────────────────────
    bt_resultado = None
    if not no_backtest:
        bt_resultado = backtest_portfolio(posicoes, ativos_dados)
        exibir_resumo_backtest(bt_resultado)

    # ── 8b. MPT ───────────────────────────────────────────────────
    mpt_resultado = None
    tickers_list = [p["ticker"] for p in posicoes]
    try:
        retornos_hist = baixar_retornos(tickers_list, periodo="3y")
        if not retornos_hist.empty:
            valor_total_mpt = sum(
                a.get("preco_atual", 0) * p.get("quantidade", 0)
                for a, p in zip(ativos_dados, posicoes)
            ) + capital_disponivel
            pesos_atuais_dict = {
                p["ticker"]: (a.get("preco_atual", 0) * p.get("quantidade", 0)) / (valor_total_mpt or 1)
                for p, a in zip(posicoes, ativos_dados)
            }
            mpt_resultado = calcular_portfolios_otimos(retornos_hist, pesos_atuais_dict)
            exibir_mpt_terminal(mpt_resultado)
    except Exception as e:
        console.print(f"  [dim yellow]⚠️  MPT não disponível: {e}[/dim yellow]\n")

    # ── 8c. Screener de oportunidades ────────────────────────────
    oportunidades_screener = []
    if not dry_run:
        try:
            tickers_portfolio = {p["ticker"] for p in posicoes}
            oportunidades_screener = coletar_universo_paralelo(
                max_workers=20,
                score_minimo=58.0,
                top_n=15,
            )
            # Remove ativos já no portfólio
            oportunidades_screener = [
                op for op in oportunidades_screener
                if op.ticker not in tickers_portfolio
            ]
            exibir_oportunidades_terminal(oportunidades_screener)
        except Exception as e:
            console.print(f"  [dim yellow]⚠️  Screener não disponível: {e}[/dim yellow]\n")

    # ── 9. Análise Claude ─────────────────────────────────────────
    if dry_run:
        console.print(Rule("[dim]Dry-run — análise Claude pulada[/dim]", style="dim"))
        if bt_resultado:
            caminho_dash = gerar_dashboard(
                bt_resultado, factor_scores, posicoes, ativos_dados, capital_disponivel,
                oportunidades_screener=oportunidades_screener,
            )
            if mpt_resultado:
                adicionar_paineis_mpt(caminho_dash, mpt_resultado)
            if mpt_resultado and oportunidades_screener:
                adicionar_painel_realocacao(
                    caminho_dash, posicoes, ativos_dados,
                    oportunidades_screener, mpt_resultado
                )
            if mpt_resultado and mpt_resultado.get("erc"):
                adicionar_painel_erc(caminho_dash, mpt_resultado)
            adicionar_resumo_estrategias(caminho_dash, mpt_resultado)
            try:
                adicionar_curva_recomendada(caminho_dash, mpt_resultado, retornos_hist)
            except Exception:
                pass
            console.print(f"\n  ✅ [bold green]Dashboard:[/bold green] [cyan]{caminho_dash}[/cyan]")
            url = _abrir_dashboard_servidor(caminho_dash)
        salvar_snapshot(posicoes, ativos_dados, capital_disponivel)
        broker_manager.desconectar()
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[bold red]ANTHROPIC_API_KEY não encontrada no .env[/bold red]")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    # Adiciona contexto do screener ao resultado
    screener_texto = formatar_screener_para_prompt(oportunidades_screener)

    # Adiciona contexto MPT/ERC/Deep Learning ao prompt do Claude
    mpt_texto = ""
    if mpt_resultado:
        from agent.mpt import formatar_mpt_para_prompt
        mpt_texto = formatar_mpt_para_prompt(mpt_resultado)
        # Adiciona resultados de Deep Learning se disponíveis
        if mpt_resultado.get("dls") or mpt_resultado.get("deepstatarb"):
            from agent.deep_portfolios import formatar_deep_para_prompt
            mpt_texto += formatar_deep_para_prompt(mpt_resultado)

    resultado = analisar_portfolio_com_claude(
        ativos_dados=ativos_dados,
        posicoes=posicoes,
        capital_disponivel=capital_disponivel,
        client=client,
        screener_texto=screener_texto,
        mpt_texto=mpt_texto,
    )

    # ── 10. Output ────────────────────────────────────────────────
    exibir_analise(resultado)

    # ── 11. Dashboard HTML ────────────────────────────────────────
    if bt_resultado:
        scores_final = resultado.get("factor_scores", factor_scores)
        # Gerar dashboard base via agent/dashboard.py (gera Monte Carlo, ERC, etc.)
        caminho_dash = gerar_dashboard(
            bt_resultado, scores_final, posicoes, ativos_dados, capital_disponivel,
            oportunidades_screener=oportunidades_screener,
        )
        if mpt_resultado:
            adicionar_paineis_mpt(caminho_dash, mpt_resultado)
        if mpt_resultado and oportunidades_screener:
            adicionar_painel_realocacao(
                caminho_dash, posicoes, ativos_dados,
                oportunidades_screener, mpt_resultado
            )
        if mpt_resultado and mpt_resultado.get("erc"):
            adicionar_painel_erc(caminho_dash, mpt_resultado)
            adicionar_resumo_estrategias(caminho_dash, mpt_resultado)
        try:
            adicionar_curva_recomendada(caminho_dash, mpt_resultado, retornos_hist)
        except Exception:
            pass
        # Atualiza análise AI (converte markdown → HTML e injeta no card final)
        analise_html = _md_to_html(resultado.get("analise", ""))
        adicionar_plano_realocacao_html(caminho_dash, analise_html)
        # Embutir Plotly inline para o HTML funcionar sem servidor
        _garantir_secoes_completas(caminho_dash)
        _embutir_plotly(caminho_dash)
        console.print(f"\n  ✅ [bold green]Dashboard gerado:[/bold green] [cyan]{caminho_dash}[/cyan]")
        url = _abrir_dashboard_servidor(caminho_dash)

    # ── 12. Salva e exporta ───────────────────────────────────────
    n_salvas = salvar_recomendacoes(resultado["analise"], ativos_dados, posicoes)
    console.print(f"  [dim]📝 {n_salvas} recomendação(ões) salva(s) para avaliação em 7 dias.[/dim]")
    salvar_snapshot(posicoes, ativos_dados, capital_disponivel)
    console.print("  [dim]💾 Snapshot salvo.[/dim]")
    exportar_relatorio(resultado, posicoes, ativos_dados, capital_disponivel)
    broker_manager.desconectar()


if __name__ == "__main__":
    main()
