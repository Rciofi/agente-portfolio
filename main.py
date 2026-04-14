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
from agent.dashboard import gerar_dashboard, adicionar_paineis_mpt, adicionar_painel_realocacao, adicionar_plano_realocacao_html
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


def main():
    args = sys.argv[1:]
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
            console.print(f"\n  ✅ [bold green]Dashboard:[/bold green] [cyan]{caminho_dash}[/cyan]")
            webbrowser.open(f"file:///{os.path.abspath(caminho_dash)}")
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

    resultado = analisar_portfolio_com_claude(
        ativos_dados=ativos_dados,
        posicoes=posicoes,
        capital_disponivel=capital_disponivel,
        client=client,
        screener_texto=screener_texto,
    )

    # ── 10. Output ────────────────────────────────────────────────
    exibir_analise(resultado)

    # ── 11. Dashboard HTML ────────────────────────────────────────
    if bt_resultado:
        scores_final = resultado.get("factor_scores", factor_scores)
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
        adicionar_plano_realocacao_html(caminho_dash, resultado["analise"])
        console.print(f"\n  ✅ [bold green]Dashboard gerado:[/bold green] [cyan]{caminho_dash}[/cyan]")
        webbrowser.open(f"file:///{os.path.abspath(caminho_dash)}")

    # ── 12. Salva e exporta ───────────────────────────────────────
    n_salvas = salvar_recomendacoes(resultado["analise"], ativos_dados, posicoes)
    console.print(f"  [dim]📝 {n_salvas} recomendação(ões) salva(s) para avaliação em 7 dias.[/dim]")
    salvar_snapshot(posicoes, ativos_dados, capital_disponivel)
    console.print("  [dim]💾 Snapshot salvo.[/dim]")
    exportar_relatorio(resultado, posicoes, ativos_dados, capital_disponivel)
    broker_manager.desconectar()


if __name__ == "__main__":
    main()
