"""
reporter.py — Output no terminal (Rich) + gráficos ASCII + exportação TXT
"""

import os
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.rule import Rule

console = Console()


def exibir_header():
    console.print()
    console.print(Panel.fit(
        "[bold white]📊  AGENTE DE GESTÃO DE PORTFÓLIO[/bold white]\n"
        "[dim]Análise híbrida · Fundamentalista + Técnica + Volatilidade · Powered by Claude[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def _cor_resultado(pct: float) -> str:
    if pct >= 10:
        return "bold green"
    elif pct >= 0:
        return "green"
    elif pct >= -10:
        return "red"
    else:
        return "bold red"


def _cor_iv_rank(rank) -> str:
    if rank is None:
        return "dim"
    if rank >= 80:
        return "bold red"
    elif rank >= 50:
        return "yellow"
    elif rank >= 25:
        return "green"
    else:
        return "bold green"


def exibir_tabela_portfolio(posicoes: list, ativos_dados: list, capital_disponivel: float):
    """Tabela resumo de posições com IV integrada."""

    valor_total = sum(
        a.get("preco_atual", 0) * p.get("quantidade", 0)
        for a, p in zip(ativos_dados, posicoes)
    ) + capital_disponivel

    # Tabela principal
    table = Table(
        title="📋 Posições Atuais",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Ticker", style="bold white", justify="center", min_width=7)
    table.add_column("Qtd", justify="right")
    table.add_column("PM", justify="right")
    table.add_column("Atual", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Posição", justify="right")
    table.add_column("% Port.", justify="right")
    table.add_column("IV%", justify="right")
    table.add_column("IVR", justify="right")

    for p, a in zip(posicoes, ativos_dados):
        preco_atual = a.get("preco_atual", 0)
        preco_medio = p.get("preco_medio", 0)
        qtd = p.get("quantidade", 0)
        valor_pos = preco_atual * qtd
        resultado_pct = ((preco_atual - preco_medio) / preco_medio * 100) if preco_medio else 0
        peso = (valor_pos / valor_total * 100) if valor_total else 0

        iv = a.get("iv_atual")
        ivr = a.get("iv_rank_1y")

        cor = _cor_resultado(resultado_pct)
        sinal = "+" if resultado_pct >= 0 else ""

        table.add_row(
            p["ticker"],
            str(qtd),
            f"${preco_medio:.2f}",
            f"${preco_atual:.2f}",
            f"[{cor}]{sinal}{resultado_pct:.1f}%[/{cor}]",
            f"${valor_pos:,.0f}",
            f"{peso:.1f}%",
            f"{iv:.1f}%" if iv else "[dim]N/A[/dim]",
            f"[{_cor_iv_rank(ivr)}]{ivr:.0f}[/{_cor_iv_rank(ivr)}]" if ivr else "[dim]N/A[/dim]",
        )

    console.print(table)
    console.print(
        f"\n  💼 [bold]Total:[/bold] [cyan]${valor_total:,.2f}[/cyan]  |  "
        f"💵 [bold]Capital livre:[/bold] [green]${capital_disponivel:,.2f}[/green]  |  "
        f"[dim]IVR: 0=baixa · 100=alta[/dim]"
    )
    console.print()


def exibir_graficos_ascii(posicoes: list, ativos_dados: list):
    """Exibe gráficos ASCII de preço para cada ativo."""
    from agent.volatility import grafico_ascii_preco

    console.print(Rule("[bold cyan]Histórico de Preço (90 dias)[/bold cyan]", style="cyan"))
    console.print()

    for p, a in zip(posicoes, ativos_dados):
        ticker = p["ticker"]
        preco_medio = p.get("preco_medio", 0)

        console.print(f"  [bold cyan]{ticker}[/bold cyan]  "
                      f"[dim]IV Atual: {a.get('iv_atual', 'N/A')}%  |  "
                      f"IVR: {a.get('iv_rank_1y', 'N/A')}  |  "
                      f"HV21d: {a.get('hv_21d', 'N/A')}%[/dim]")

        grafico = grafico_ascii_preco(ticker, preco_medio)
        console.print(grafico)
        console.print()


def exibir_historico_snapshots(posicoes: list):
    """Exibe evolução histórica do portfólio via snapshots locais."""
    from agent.volatility import carregar_historico_ticker

    tem_historico = False
    for p in posicoes:
        ticker = p["ticker"]
        registros = carregar_historico_ticker(ticker)
        if len(registros) >= 2:
            tem_historico = True
            break

    if not tem_historico:
        return  # Primeira execução — ainda sem histórico

    console.print(Rule("[bold cyan]Evolução do Portfólio (Snapshots)[/bold cyan]", style="cyan"))
    console.print()

    for p in posicoes:
        ticker = p["ticker"]
        registros = carregar_historico_ticker(ticker)
        if len(registros) < 2:
            continue

        table = Table(
            title=f"{ticker} — Histórico de Execuções",
            box=box.SIMPLE,
            header_style="bold dim",
            show_lines=False,
        )
        table.add_column("Data", style="dim")
        table.add_column("Preço", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Posição", justify="right")
        table.add_column("IV%", justify="right")
        table.add_column("IVR", justify="right")

        for r in registros[-10:]:  # últimos 10 snapshots
            pct = r.get("resultado_pct", 0)
            cor = _cor_resultado(pct)
            sinal = "+" if pct >= 0 else ""
            ivr = r.get("iv_rank_1y")

            table.add_row(
                r["data"],
                f"${r.get('preco_atual', 0):.2f}",
                f"[{cor}]{sinal}{pct:.1f}%[/{cor}]",
                f"${r.get('valor_posicao', 0):,.0f}",
                f"{r.get('iv_atual', 0):.1f}%" if r.get("iv_atual") else "N/A",
                f"[{_cor_iv_rank(ivr)}]{ivr:.0f}[/{_cor_iv_rank(ivr)}]" if ivr else "N/A",
            )

        console.print(table)
        console.print()


def exibir_analise(resultado: dict):
    """Exibe a análise do Claude no terminal."""
    console.print(Rule("[bold cyan]Análise e Recomendações — Claude[/bold cyan]", style="cyan"))
    console.print()
    console.print(resultado["analise"])
    console.print()
    console.print(Rule(style="dim"))
    console.print(
        f"[dim]  Tokens: {resultado['tokens_usados']:,} | "
        f"Modelo: claude-opus-4-5 | "
        f"Gerado: {datetime.now().strftime('%d/%m/%Y %H:%M')}[/dim]"
    )
    console.print()


def exportar_relatorio(resultado: dict, posicoes: list, ativos_dados: list, capital_disponivel: float):
    """Salva relatório completo em TXT em outputs/."""
    os.makedirs("outputs", exist_ok=True)
    agora = datetime.now().strftime("%Y%m%d_%H%M")
    caminho = f"outputs/relatorio_{agora}.txt"

    valor_total = resultado["valor_total"]

    linhas = [
        "=" * 72,
        "  AGENTE DE GESTÃO DE PORTFÓLIO",
        f"  Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "=" * 72,
        "",
        "PORTFÓLIO ANALISADO:",
        f"{'TICKER':<8} {'QTD':>5} {'PM':>9} {'ATUAL':>9} {'P&L':>8} {'POSIÇÃO':>12} {'% PORT':>7} {'IV%':>6} {'IVR':>5}",
        "-" * 72,
    ]

    for p, a in zip(posicoes, ativos_dados):
        preco_atual = a.get("preco_atual", 0)
        preco_medio = p.get("preco_medio", 0)
        qtd = p.get("quantidade", 0)
        valor_pos = preco_atual * qtd
        resultado_pct = ((preco_atual - preco_medio) / preco_medio * 100) if preco_medio else 0
        peso = (valor_pos / valor_total * 100) if valor_total else 0
        iv = a.get("iv_atual")
        ivr = a.get("iv_rank_1y")

        sinal = "+" if resultado_pct >= 0 else ""
        linhas.append(
            f"{p['ticker']:<8} {qtd:>5} ${preco_medio:>8.2f} ${preco_atual:>8.2f} "
            f"{sinal}{resultado_pct:>6.1f}% ${valor_pos:>10,.0f} {peso:>6.1f}% "
            f"{iv:>5.1f}% {ivr:>5.0f}" if iv and ivr else
            f"{p['ticker']:<8} {qtd:>5} ${preco_medio:>8.2f} ${preco_atual:>8.2f} "
            f"{sinal}{resultado_pct:>6.1f}% ${valor_pos:>10,.0f} {peso:>6.1f}% {'N/A':>6} {'N/A':>5}"
        )

    linhas += [
        "-" * 72,
        f"Capital disponível: ${capital_disponivel:,.2f}  |  Valor total: ${valor_total:,.2f}",
        "",
        "=" * 72,
        "VOLATILIDADE POR ATIVO:",
        "-" * 72,
    ]

    for p, a in zip(posicoes, ativos_dados):
        linhas.append(
            f"{p['ticker']}: IV={a.get('iv_atual', 'N/A')}% | IVR={a.get('iv_rank_1y', 'N/A')} | "
            f"HV10={a.get('hv_10d', 'N/A')}% | HV21={a.get('hv_21d', 'N/A')}% | HV63={a.get('hv_63d', 'N/A')}%"
        )
        linhas.append(f"  → {a.get('iv_interpretacao', 'N/A')}")

    linhas += [
        "",
        "=" * 72,
        "ANÁLISE E RECOMENDAÇÕES (Claude):",
        "=" * 72,
        "",
        resultado["analise"],
        "",
        "=" * 72,
        f"Tokens utilizados: {resultado['tokens_usados']:,}",
        "=" * 72,
    ]

    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

    console.print(f"\n  ✅ [bold green]Relatório salvo:[/bold green] [cyan]{caminho}[/cyan]\n")
    return caminho
