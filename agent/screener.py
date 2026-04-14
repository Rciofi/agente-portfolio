"""
screener.py — Screener automático de oportunidades de investimento

Varre ~300 ativos (S&P500 large/mid, Russell 2000 small caps, ADRs BR, ETFs)
Calcula factor score de cada um e retorna top oportunidades
filtradas pelo perfil do investidor.

Perfil padrão: crescimento agressivo, evita setores Barsi
"""

import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from agent.factors import calcular_score


# ─────────────────────────────────────────────────────────────────
# UNIVERSO DE ATIVOS
# ─────────────────────────────────────────────────────────────────

# S&P 500 — Large Caps (top 80 por market cap)
SP500_LARGE = [
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B",
    "JPM", "LLY", "V", "MA", "UNH", "XOM", "COST", "HD", "PG", "JNJ",
    "ABBV", "BAC", "CRM", "NFLX", "AMD", "WMT", "MRK", "CVX", "KO", "PEP",
    "ACN", "TMO", "MCD", "CSCO", "ABT", "ADBE", "LIN", "TXN", "DHR", "NKE",
    "NEE", "PM", "ORCL", "IBM", "QCOM", "AMAT", "AMGN", "GE", "RTX", "CAT",
    "SPGI", "INTU", "NOW", "ISRG", "PLD", "BX", "SYK", "MDT", "GILD", "C",
    "AXP", "MMC", "ETN", "BSX", "ADI", "MU", "LRCX", "KLAC", "SNPS", "CDNS",
    "PANW", "CRWD", "FTNT", "ZS", "DDOG", "NET", "SNOW", "PLTR", "UBER", "ABNB",
]

# S&P 400 — Mid Caps (oportunidades menos cobertas)
SP400_MID = [
    "MTCH", "EXAS", "FICO", "RH", "MSTR", "CELH", "WING", "GNRC",
    "TXRH", "PAYC", "MEDP", "SAIA", "CVLT", "MKSI", "AAON", "TREX",
    "MGEE", "NOVT", "CHDN", "ITCI", "LNTH", "ACLS", "FORM", "NOVTA",
    "FN", "AGIO", "CORT", "PRCT", "ARCB", "BCPC", "MGRC", "SFM",
    # Homebuilders
    "DHI", "LEN", "PHM", "NVR", "TOL", "MTH", "KBH", "MDC", "TMHC", "SKY",
    "CSWI", "DXPE", "HWKN", "GTLS", "UFPI", "PATK", "MATX", "KMPR",
    "CINF", "ERIE", "RLI", "AFG", "SIGI", "PLMR", "KNTK", "HCI",
]

# Russell 2000 — Small Caps (maior potencial, mais risco)
RUSSELL_SMALL = [
    "SMPL", "QLYS", "SMAR", "PCVX", "ACAD", "LGND", "PRAX", "RARE",
    "IONS", "FOLD", "RCKT", "IMVT", "NTRA", "HALO", "EXEL", "PTGX",
    "RDNT", "OMCL", "AMSF", "FCFS", "PAYO", "STNE", "NUVL",
    "TGTX", "ARWR", "NTLA", "BEAM", "EDIT", "CRSP", "VERV", "GRFS",
    "BOOT", "KLIC", "AEIS", "ICHR", "ONTO", "ACMR", "UCTT", "AMBA",
    "SLAB", "DIOD", "SMTC", "LFUS", "CTS", "PLXS", "TTMI",
    "BOWL", "GSHD", "RYAN", "BFAM", "LPSN", "MARA", "RIOT", "COIN",
]

# ADRs Brasileiros — NYSE/NASDAQ
ADRS_BR = [
    "PBR", "VALE", "ITUB", "BBD", "ABEV", "CIG", "SID",
    "GGB", "CBD", "VIV", "TIMB", "XPEV", "NU",
    "PAGS", "STNE", "ARCO", "DESP", "VTRU", "MBLY",
]

# ETFs estratégicos (excluindo REITs — perfil Barsi)
ETFS = [
    "QQQ", "SPY", "IVV", "VTI", "VGT", "XLK", "XLV", "XLE", "XLF",
    "XLI", "XLB", "SCHD", "VIG", "DGRO", "NOBL", "VUG", "VBK",
    "IWM", "IJH", "IJR", "SCHA", "VBR", "IWN",
    "GLD", "SLV", "PDBC",
]

# Setores a EXCLUIR (filtro Barsi adaptado)
SETORES_EXCLUIDOS = {
    "airlines", "air freight", "hotel", "motel", "resort", "casino",
    "restaurant", "retail", "apparel", "department store",
    "real estate", "reit", "mortgage", "meat", "poultry", "packaged foods",
    "tobacco", "distillers", "brewers", "gambling", "lodging",
}


# ─────────────────────────────────────────────────────────────────
# ESTRUTURA DE RESULTADO
# ─────────────────────────────────────────────────────────────────

@dataclass
class OportunidadeScreener:
    ticker: str
    nome: str
    setor: str
    industria: str
    score_total: float
    score_momentum: float
    score_valuation: float
    score_qualidade: float
    score_volatilidade: float
    sinal: str
    preco_atual: float
    upside_analistas: float
    market_cap: float
    segmento: str          # large / mid / small / adr / etf
    motivo_destaque: str   # por que apareceu no top


# ─────────────────────────────────────────────────────────────────
# COLETA RÁPIDA (paralela)
# ─────────────────────────────────────────────────────────────────

def _coletar_ativo_screener(ticker: str, segmento: str) -> dict | None:
    """Coleta dados mínimos para o screener. Rápido e resiliente."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        preco = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        if not preco or preco == 0:
            return None

        setor = (info.get("sector") or "").lower()
        industria = (info.get("industry") or "").lower()

        # Verifica se setor está excluído
        texto_setor = f"{setor} {industria}"
        for excluido in SETORES_EXCLUIDOS:
            if excluido in texto_setor:
                return None

        # Histórico para cálculo de scores técnicos
        hist_90 = stock.history(period="3mo")
        media_90 = hist_90["Close"].mean() if not hist_90.empty else preco
        tendencia = "ALTA" if preco > media_90 else "BAIXA"

        hist_1y = stock.history(period="1y")
        high_52 = info.get("fiftyTwoWeekHigh", preco)
        low_52 = info.get("fiftyTwoWeekLow", preco)
        pct_from_high = ((preco - high_52) / high_52 * 100) if high_52 else 0
        pct_from_low = ((preco - low_52) / low_52 * 100) if low_52 else 0

        return {
            "ticker": ticker,
            "segmento": segmento,
            "nome": info.get("longName", ticker),
            "setor": info.get("sector", "N/A"),
            "industria": info.get("industry", "N/A"),
            "preco_atual": round(preco, 2),
            "market_cap": info.get("marketCap", 0),
            # Dados para factor score
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "pb_ratio": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield", 0),
            "roe": info.get("returnOnEquity"),
            "margem_lucro": info.get("profitMargins"),
            "crescimento_receita": info.get("revenueGrowth"),
            "crescimento_lucro": info.get("earningsGrowth"),
            "divida_equity": info.get("debtToEquity"),
            "recomendacao_analistas": info.get("recommendationKey", "N/A"),
            "target_price": info.get("targetMeanPrice"),
            "upside_analysts": round(
                ((info.get("targetMeanPrice", 0) - preco) / preco * 100), 2
            ) if info.get("targetMeanPrice") else None,
            "tendencia_90d": tendencia,
            "media_movel_90d": round(media_90, 2),
            "high_52w": high_52,
            "low_52w": low_52,
            "pct_abaixo_high_52w": round(pct_from_high, 2),
            "pct_acima_low_52w": round(pct_from_low, 2),
            "variacao_dia_pct": 0,
            # Volatilidade básica (sem opções para ser mais rápido)
            "iv_atual": None,
            "iv_rank_1y": None,
            "hv_21d": None,
            "rsi": None,
            "sma50": None,
            "sma200": None,
            "beta": info.get("beta"),
            "atr": None,
        }

    except Exception:
        return None


def coletar_universo_paralelo(
    max_workers: int = 20,
    score_minimo: float = 55.0,
    top_n: int = 15,
    incluir_segmentos: list = None,
) -> list[OportunidadeScreener]:
    """
    Coleta e pontua todos os ativos do universo em paralelo.
    Retorna top N oportunidades ordenadas por score.
    """
    if incluir_segmentos is None:
        incluir_segmentos = ["large", "mid", "small", "adr", "etf"]

    # Monta lista de (ticker, segmento)
    universo = []
    if "large" in incluir_segmentos:
        universo += [(t, "large") for t in SP500_LARGE]
    if "mid" in incluir_segmentos:
        universo += [(t, "mid") for t in SP400_MID]
    if "small" in incluir_segmentos:
        universo += [(t, "small") for t in RUSSELL_SMALL]
    if "adr" in incluir_segmentos:
        universo += [(t, "adr") for t in ADRS_BR]
    if "etf" in incluir_segmentos:
        universo += [(t, "etf") for t in ETFS]

    total = len(universo)
    print(f"\n  🔍 Screener: varrendo {total} ativos em paralelo...")

    dados_coletados = []
    erros = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_coletar_ativo_screener, ticker, seg): (ticker, seg)
            for ticker, seg in universo
        }
        concluidos = 0
        for future in as_completed(futures):
            concluidos += 1
            if concluidos % 50 == 0:
                print(f"  🔍 Screener: {concluidos}/{total} ativos processados...")
            resultado = future.result()
            if resultado:
                dados_coletados.append(resultado)
            else:
                erros += 1

    print(f"  🔍 Screener: {len(dados_coletados)} ativos válidos ({erros} erros/filtrados)")

    # Calcula factor score para cada ativo
    oportunidades = []
    for dados in dados_coletados:
        try:
            fs = calcular_score(dados["ticker"], dados)
            if fs.score_total < score_minimo:
                continue

            motivo = _gerar_motivo(dados, fs)

            oportunidades.append(OportunidadeScreener(
                ticker=dados["ticker"],
                nome=dados["nome"],
                setor=dados["setor"],
                industria=dados["industria"],
                score_total=fs.score_total,
                score_momentum=fs.score_momentum,
                score_valuation=fs.score_valuation,
                score_qualidade=fs.score_qualidade,
                score_volatilidade=fs.score_volatilidade,
                sinal=fs.sinal,
                preco_atual=dados["preco_atual"],
                upside_analistas=dados.get("upside_analysts") or 0,
                market_cap=dados.get("market_cap") or 0,
                segmento=dados["segmento"],
                motivo_destaque=motivo,
            ))
        except Exception:
            continue

    # Ordena por score total (decrescente)
    oportunidades.sort(key=lambda x: x.score_total, reverse=True)

    # Garante diversidade de segmentos no top N
    top = _diversificar_top(oportunidades, top_n)

    print(f"  ✅ Screener: {len(top)} oportunidades encontradas (score ≥ {score_minimo})")
    return top


def _gerar_motivo(dados: dict, fs) -> str:
    """Gera texto explicando por que o ativo se destacou."""
    motivos = []

    if fs.score_qualidade >= 70:
        motivos.append(f"qualidade excepcional ({fs.score_qualidade:.0f}/100)")
    if fs.score_momentum >= 70:
        motivos.append(f"momentum forte ({fs.score_momentum:.0f}/100)")
    if fs.score_valuation >= 70:
        motivos.append("valuation atrativo")

    upside = dados.get("upside_analysts")
    if upside and upside > 20:
        motivos.append(f"upside analistas +{upside:.0f}%")

    cresc = dados.get("crescimento_lucro")
    if cresc and cresc > 0.3:
        motivos.append(f"crescimento lucro +{cresc*100:.0f}%")

    peg = dados.get("peg_ratio")
    if peg and 0 < peg < 1:
        motivos.append(f"PEG {peg:.1f} (muito barato)")

    segmento = dados.get("segmento", "")
    if segmento == "small":
        motivos.append("small cap com potencial")
    elif segmento == "adr":
        motivos.append("ADR brasileiro")

    return " · ".join(motivos) if motivos else "score quantitativo elevado"


def _diversificar_top(
    oportunidades: list[OportunidadeScreener],
    top_n: int
) -> list[OportunidadeScreener]:
    """
    Garante que o top N tenha diversidade de segmentos.
    Máximo: 40% large, 30% mid+small, 20% adr, 10% etf
    """
    limites = {"large": int(top_n * 0.4), "mid": int(top_n * 0.2),
               "small": int(top_n * 0.2), "adr": int(top_n * 0.15),
               "etf": int(top_n * 0.1)}
    contadores = {k: 0 for k in limites}
    top = []

    for op in oportunidades:
        seg = op.segmento
        limite = limites.get(seg, top_n)
        if contadores.get(seg, 0) < limite:
            top.append(op)
            contadores[seg] = contadores.get(seg, 0) + 1
        if len(top) >= top_n:
            break

    # Preenche com os melhores restantes se não chegou ao top_n
    if len(top) < top_n:
        adicionados = {op.ticker for op in top}
        for op in oportunidades:
            if op.ticker not in adicionados:
                top.append(op)
                if len(top) >= top_n:
                    break

    return top


# ─────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────

def exibir_oportunidades_terminal(oportunidades: list[OportunidadeScreener]):
    """Exibe oportunidades do screener no terminal com Rich."""
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule
    from rich import box

    console = Console()
    console.print(Rule("[bold cyan]🔍 Screener — Oportunidades Fora do Portfólio[/bold cyan]", style="cyan"))
    console.print()

    if not oportunidades:
        console.print("  [dim]Nenhuma oportunidade encontrada com os critérios atuais.[/dim]\n")
        return

    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan", show_lines=True)
    table.add_column("Ticker", style="bold white", justify="center", min_width=6)
    table.add_column("Nome", min_width=20)
    table.add_column("Setor", style="dim", min_width=12)
    table.add_column("Seg.", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Sinal", justify="center")
    table.add_column("Preço", justify="right")
    table.add_column("Upside", justify="right")
    table.add_column("Destaque", style="dim")

    seg_cores = {
        "large": "cyan", "mid": "blue", "small": "magenta",
        "adr": "green", "etf": "yellow"
    }
    sinal_cores = {
        "FORTE COMPRA": "bold green", "COMPRA": "green",
        "NEUTRO": "cyan", "VENDA": "red", "FORTE VENDA": "bold red"
    }

    def cor_score(v):
        if v >= 70: return "bold green"
        if v >= 60: return "green"
        return "yellow"

    for op in oportunidades:
        seg = op.segmento
        cor_seg = seg_cores.get(seg, "white")
        cor_sin = sinal_cores.get(op.sinal, "white")
        cor_up = "green" if op.upside_analistas > 0 else "red"
        cap_fmt = f"${op.market_cap/1e9:.1f}B" if op.market_cap > 1e9 else f"${op.market_cap/1e6:.0f}M"

        table.add_row(
            op.ticker,
            op.nome[:28] + "..." if len(op.nome) > 28 else op.nome,
            op.setor[:14] + "..." if len(op.setor) > 14 else op.setor,
            f"[{cor_seg}]{seg.upper()}[/{cor_seg}]",
            f"[{cor_score(op.score_total)}]{op.score_total:.0f}[/{cor_score(op.score_total)}]",
            f"[{cor_sin}]{op.sinal}[/{cor_sin}]",
            f"${op.preco_atual:.2f}",
            f"[{cor_up}]{op.upside_analistas:+.1f}%[/{cor_up}]" if op.upside_analistas else "N/A",
            op.motivo_destaque[:45],
        )

    console.print(table)
    console.print(
        "  [dim]SEG: LARGE=S&P500 · MID=S&P400 · SMALL=Russell2000 · "
        "ADR=Brasil · ETF=Fundos[/dim]\n"
    )


def formatar_screener_para_prompt(oportunidades: list[OportunidadeScreener]) -> str:
    """Formata oportunidades para incluir no prompt do Claude."""
    if not oportunidades:
        return ""

    linhas = ["\n=== OPORTUNIDADES IDENTIFICADAS PELO SCREENER ==="]
    linhas.append("(Ativos fora do portfólio atual com maior potencial)\n")

    for i, op in enumerate(oportunidades[:10], 1):
        linhas.append(
            f"{i}. {op.ticker} ({op.segmento.upper()}) — Score: {op.score_total:.0f}/100 — {op.sinal}\n"
            f"   {op.nome} | {op.setor}\n"
            f"   Preço: ${op.preco_atual:.2f} | Upside analistas: {op.upside_analistas:+.1f}%\n"
            f"   Destaque: {op.motivo_destaque}\n"
            f"   Momentum: {op.score_momentum:.0f} | Valuation: {op.score_valuation:.0f} | "
            f"Qualidade: {op.score_qualidade:.0f} | Volatilidade: {op.score_volatilidade:.0f}"
        )

    return "\n".join(linhas)
