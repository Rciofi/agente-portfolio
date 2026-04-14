"""
accuracy.py — Histórico e acurácia das recomendações do agente

Fluxo:
  1. Ao gerar recomendação → salva em data/recommendations.json
  2. Na próxima execução → compara preço atual vs preço na data da recomendação
  3. Classifica como CORRETA / INCORRETA / PENDENTE
  4. Exibe score acumulado por ativo e geral no terminal
"""

import os
import json
import re
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

RECOMMENDATIONS_FILE = "data/recommendations.json"

# Mapeamento de recomendação → direção esperada do preço
DIRECAO_ESPERADA = {
    "REALIZAR TUDO":          "baixar",   # esperava que ia cair (ou realizou no topo)
    "REALIZAR PARCIALMENTE":  "baixar",
    "REDUZIR GRADUALMENTE":   "baixar",
    "ADICIONAR":              "subir",
    "MANTER POSIÇÃO":         "neutro",   # avaliado diferente
}

JANELA_AVALIACAO_DIAS = 7   # avalia resultado após 7 dias


def extrair_recomendacoes_do_texto(analise: str, ativos: list[str]) -> list[dict]:
    """
    Faz parse simples do texto do Claude para extrair recomendações por ticker.
    Procura padrões como 'AAPL' seguido de palavras-chave de recomendação.
    """
    recomendacoes = []
    texto = analise.upper()

    keywords = list(DIRECAO_ESPERADA.keys())

    for ticker in ativos:
        ticker_upper = ticker.upper()
        # Encontra ocorrência do ticker no texto e procura recomendação próxima
        idx = texto.find(ticker_upper)
        if idx == -1:
            continue

        # Janela de 300 chars ao redor do ticker para encontrar a recomendação
        trecho = texto[max(0, idx - 50): idx + 300]

        recomendacao_encontrada = None
        for kw in keywords:
            if kw in trecho:
                recomendacao_encontrada = kw
                break

        if recomendacao_encontrada:
            recomendacoes.append({
                "ticker": ticker_upper,
                "recomendacao": recomendacao_encontrada,
            })

    return recomendacoes


def salvar_recomendacoes(analise: str, ativos_dados: list, posicoes: list):
    """Salva recomendações geradas com preço atual para avaliação futura."""
    os.makedirs("data", exist_ok=True)

    tickers = [p["ticker"] for p in posicoes]
    recomendacoes = extrair_recomendacoes_do_texto(analise, tickers)

    # Cria mapa ticker → preço atual
    precos = {a.get("ticker", ""): a.get("preco_atual", 0) for a in ativos_dados}

    registros = []
    agora = datetime.now().isoformat()
    data_avaliacao = (datetime.now() + timedelta(days=JANELA_AVALIACAO_DIAS)).strftime("%Y-%m-%d")

    for rec in recomendacoes:
        ticker = rec["ticker"]
        registros.append({
            "id": f"{ticker}_{agora[:10]}_{agora[11:16].replace(':', '')}",
            "ticker": ticker,
            "recomendacao": rec["recomendacao"],
            "preco_na_recomendacao": precos.get(ticker, 0),
            "data_recomendacao": agora[:10],
            "data_avaliacao_prevista": data_avaliacao,
            "status": "PENDENTE",
            "preco_na_avaliacao": None,
            "variacao_pct": None,
            "resultado": None,   # CORRETA / INCORRETA / NEUTRO
        })

    # Carrega histórico existente
    historico = _carregar_historico()
    historico.extend(registros)

    # Mantém últimos 500 registros
    if len(historico) > 500:
        historico = historico[-500:]

    with open(RECOMMENDATIONS_FILE, "w") as f:
        json.dump(historico, f, indent=2, ensure_ascii=False)

    return len(registros)


def avaliar_recomendacoes_pendentes(ativos_dados: list):
    """
    Verifica recomendações pendentes que já passaram da janela de avaliação.
    Atualiza status com resultado real.
    """
    historico = _carregar_historico()
    if not historico:
        return 0

    precos_atuais = {a.get("ticker", ""): a.get("preco_atual", 0) for a in ativos_dados}
    hoje = datetime.now().strftime("%Y-%m-%d")
    atualizados = 0

    for rec in historico:
        if rec["status"] != "PENDENTE":
            continue

        if rec["data_avaliacao_prevista"] > hoje:
            continue

        ticker = rec["ticker"]
        preco_atual = precos_atuais.get(ticker)
        if not preco_atual:
            continue

        preco_orig = rec["preco_na_recomendacao"]
        if not preco_orig:
            continue

        variacao = ((preco_atual - preco_orig) / preco_orig * 100)
        direcao = DIRECAO_ESPERADA.get(rec["recomendacao"], "neutro")

        if direcao == "neutro":
            resultado = "NEUTRO"
        elif direcao == "subir":
            resultado = "CORRETA" if variacao >= 0 else "INCORRETA"
        else:  # baixar
            resultado = "CORRETA" if variacao <= 0 else "INCORRETA"

        rec["preco_na_avaliacao"] = round(preco_atual, 2)
        rec["variacao_pct"] = round(variacao, 2)
        rec["resultado"] = resultado
        rec["status"] = "AVALIADO"
        atualizados += 1

    if atualizados > 0:
        with open(RECOMMENDATIONS_FILE, "w") as f:
            json.dump(historico, f, indent=2, ensure_ascii=False)

    return atualizados


def exibir_historico_acuracia():
    """Exibe tabela de acurácia no terminal."""
    historico = _carregar_historico()
    avaliados = [r for r in historico if r["status"] == "AVALIADO"]
    pendentes = [r for r in historico if r["status"] == "PENDENTE"]

    if not avaliados and not pendentes:
        return  # Primeira execução — nada ainda

    from rich.rule import Rule
    console.print(Rule("[bold cyan]Histórico de Acurácia das Recomendações[/bold cyan]", style="cyan"))
    console.print()

    # ── Score geral ───────────────────────────────────────────────
    if avaliados:
        corretas = sum(1 for r in avaliados if r["resultado"] == "CORRETA")
        incorretas = sum(1 for r in avaliados if r["resultado"] == "INCORRETA")
        neutras = sum(1 for r in avaliados if r["resultado"] == "NEUTRO")
        total = len(avaliados)
        acuracia = (corretas / (corretas + incorretas) * 100) if (corretas + incorretas) > 0 else 0

        cor_acuracia = "bold green" if acuracia >= 65 else "yellow" if acuracia >= 50 else "bold red"

        console.print(
            f"  📊 Recomendações avaliadas: [bold]{total}[/bold]  |  "
            f"✅ Corretas: [green]{corretas}[/green]  |  "
            f"❌ Incorretas: [red]{incorretas}[/red]  |  "
            f"➖ Neutras: [dim]{neutras}[/dim]  |  "
            f"🎯 Acurácia: [{cor_acuracia}]{acuracia:.1f}%[/{cor_acuracia}]"
        )
        console.print()

    # ── Tabela de recomendações recentes ─────────────────────────
    table = Table(
        title=f"Últimas recomendações (avaliadas + {len(pendentes)} pendente(s))",
        box=box.SIMPLE,
        header_style="bold dim",
        show_lines=False,
    )
    table.add_column("Data", style="dim", min_width=10)
    table.add_column("Ticker", style="bold white", min_width=6)
    table.add_column("Recomendação", min_width=22)
    table.add_column("Preço Rec.", justify="right")
    table.add_column("Preço Aval.", justify="right")
    table.add_column("Variação", justify="right")
    table.add_column("Resultado", justify="center")

    # Mostra últimas 15 (avaliadas + pendentes)
    recentes = sorted(historico, key=lambda x: x["data_recomendacao"], reverse=True)[:15]

    for r in recentes:
        var = r.get("variacao_pct")
        cor_var = "green" if (var or 0) >= 0 else "red"
        sinal = "+" if (var or 0) >= 0 else ""

        resultado = r.get("resultado", "PENDENTE")
        if resultado == "CORRETA":
            resultado_fmt = "[bold green]✅ CORRETA[/bold green]"
        elif resultado == "INCORRETA":
            resultado_fmt = "[bold red]❌ INCORRETA[/bold red]"
        elif resultado == "NEUTRO":
            resultado_fmt = "[dim]➖ NEUTRO[/dim]"
        else:
            resultado_fmt = "[dim yellow]⏳ PENDENTE[/dim yellow]"

        table.add_row(
            r["data_recomendacao"],
            r["ticker"],
            r["recomendacao"],
            f"${r.get('preco_na_recomendacao', 0):.2f}",
            f"${r['preco_na_avaliacao']:.2f}" if r.get("preco_na_avaliacao") else "—",
            f"[{cor_var}]{sinal}{var:.1f}%[/{cor_var}]" if var is not None else "—",
            resultado_fmt,
        )

    console.print(table)
    console.print()

    # ── Acurácia por ticker ───────────────────────────────────────
    if avaliados:
        tickers = list(set(r["ticker"] for r in avaliados))
        acuracia_ticker = []
        for t in sorted(tickers):
            t_avaliados = [r for r in avaliados if r["ticker"] == t]
            t_corretas = sum(1 for r in t_avaliados if r["resultado"] == "CORRETA")
            t_incorretas = sum(1 for r in t_avaliados if r["resultado"] == "INCORRETA")
            total_t = t_corretas + t_incorretas
            if total_t > 0:
                acc = t_corretas / total_t * 100
                acuracia_ticker.append((t, t_corretas, t_incorretas, acc))

        if acuracia_ticker:
            console.print("  [dim]Acurácia por ativo:[/dim]  " + "  |  ".join(
                f"[bold]{t}[/bold] [{('green' if acc >= 60 else 'red')}]{acc:.0f}%[/{'green' if acc >= 60 else 'red'}] ({c}✅/{i}❌)"
                for t, c, i, acc in acuracia_ticker
            ))
            console.print()


def _carregar_historico() -> list:
    if not os.path.exists(RECOMMENDATIONS_FILE):
        return []
    try:
        with open(RECOMMENDATIONS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []
