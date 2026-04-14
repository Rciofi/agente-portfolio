"""
volatility.py — Volatilidade implícita + histórico de preços
Fonte: yfinance (cadeia de opções ATM) + histórico OHLCV
Histórico local salvo em data/history/{ticker}.csv para comparação temporal
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


HISTORY_DIR = "data/history"
SNAPSHOT_FILE = "data/snapshots.json"


# ─────────────────────────────────────────────
# VOLATILIDADE IMPLÍCITA
# ─────────────────────────────────────────────

def get_implied_volatility(ticker: str) -> dict:
    """
    Extrai IV da cadeia de opções ATM mais próxima do vencimento atual.
    Retorna IV média de calls + puts ATM, IV percentile e IV rank (IVR).
    """
    result = {
        "iv_atual": None,
        "iv_calls_atm": None,
        "iv_puts_atm": None,
        "iv_percentile_30d": None,   # onde a IV atual está vs últimos 30 dias
        "iv_rank_1y": None,           # IVR: 0-100, quanto % do range anual a IV atual representa
        "iv_interpretacao": "N/A",
        "proximo_vencimento": None,
    }

    try:
        stock = yf.Ticker(ticker)
        preco_atual = stock.info.get("currentPrice") or stock.info.get("regularMarketPrice", 0)

        if not preco_atual:
            return result

        # Pega vencimentos disponíveis
        vencimentos = stock.options
        if not vencimentos:
            return result

        # Escolhe o vencimento mais próximo com pelo menos 7 dias
        hoje = datetime.today()
        venc_validos = [
            v for v in vencimentos
            if (datetime.strptime(v, "%Y-%m-%d") - hoje).days >= 7
        ]
        if not venc_validos:
            venc_validos = list(vencimentos)

        venc_escolhido = venc_validos[0]
        result["proximo_vencimento"] = venc_escolhido

        # Cadeia de opções do vencimento escolhido
        chain = stock.option_chain(venc_escolhido)
        calls = chain.calls
        puts = chain.puts

        if calls.empty or puts.empty:
            return result

        # Filtra opções ATM (strike mais próximo do preço atual)
        calls["dist"] = abs(calls["strike"] - preco_atual)
        puts["dist"] = abs(puts["strike"] - preco_atual)

        # Pega as 3 strikes mais próximas e faz média ponderada por volume
        calls_atm = calls.nsmallest(3, "dist")
        puts_atm = puts.nsmallest(3, "dist")

        # IV média ponderada por volume (evita outliers sem liquidez)
        def iv_ponderada(df):
            df = df[df["impliedVolatility"] > 0].copy()
            if df.empty:
                return None
            vol = df["volume"].fillna(1).clip(lower=1)
            return float(np.average(df["impliedVolatility"], weights=vol))

        iv_calls = iv_ponderada(calls_atm)
        iv_puts = iv_ponderada(puts_atm)

        if iv_calls and iv_puts:
            iv_media = (iv_calls + iv_puts) / 2
        elif iv_calls:
            iv_media = iv_calls
        elif iv_puts:
            iv_media = iv_puts
        else:
            return result

        result["iv_calls_atm"] = round(iv_calls * 100, 2) if iv_calls else None
        result["iv_puts_atm"] = round(iv_puts * 100, 2) if iv_puts else None
        result["iv_atual"] = round(iv_media * 100, 2)

        # ── IV Rank (IVR) via histórico de HV como proxy ──
        # Calcula HV histórica dos últimos 252 dias como referência de range
        hist = stock.history(period="1y")
        if not hist.empty and len(hist) > 30:
            retornos = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

            # HV rolling 30 dias (anualizada) como proxy de IV histórica
            hv_rolling = retornos.rolling(30).std() * np.sqrt(252) * 100
            hv_rolling = hv_rolling.dropna()

            hv_min = float(hv_rolling.min())
            hv_max = float(hv_rolling.max())
            hv_atual = float(hv_rolling.iloc[-1])

            # IV Rank: posição da IV atual dentro do range histórico (0-100)
            if hv_max > hv_min:
                iv_rank = (iv_media * 100 - hv_min) / (hv_max - hv_min) * 100
                result["iv_rank_1y"] = round(float(np.clip(iv_rank, 0, 100)), 1)

            # IV Percentile: % dos dias em que a HV foi menor que a IV atual
            iv_percentile = float((hv_rolling < iv_media * 100).mean() * 100)
            result["iv_percentile_30d"] = round(iv_percentile, 1)

        # Interpretação automática
        iv_rank = result["iv_rank_1y"]
        if iv_rank is not None:
            if iv_rank >= 80:
                result["iv_interpretacao"] = "IV ALTA — opções caras, favorece venda de vol / realizar posição"
            elif iv_rank >= 50:
                result["iv_interpretacao"] = "IV ELEVADA — mercado precificando incerteza acima da média"
            elif iv_rank >= 25:
                result["iv_interpretacao"] = "IV MODERADA — condições normais"
            else:
                result["iv_interpretacao"] = "IV BAIXA — opções baratas, mercado calmo ou possível movimento iminente"

    except Exception as e:
        result["erro_iv"] = str(e)

    return result


def get_historical_volatility(ticker: str, janelas: list = [10, 21, 63]) -> dict:
    """
    Calcula volatilidade histórica realizada em múltiplas janelas.
    janelas: dias de negociação (10=2sem, 21=1mês, 63=3meses)
    """
    result = {}
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            return result

        retornos = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

        for janela in janelas:
            if len(retornos) >= janela:
                hv = retornos.iloc[-janela:].std() * np.sqrt(252) * 100
                result[f"hv_{janela}d"] = round(float(hv), 2)

    except Exception:
        pass

    return result


# ─────────────────────────────────────────────
# HISTÓRICO LOCAL DE SNAPSHOTS
# ─────────────────────────────────────────────

def salvar_snapshot(posicoes: list, ativos_dados: list, capital_disponivel: float):
    """
    Salva snapshot do portfólio em JSON para comparação histórica.
    Cada execução gera um registro com timestamp.
    """
    os.makedirs("data", exist_ok=True)

    snapshot = {
        "data": datetime.now().isoformat(),
        "capital_disponivel": capital_disponivel,
        "ativos": []
    }

    for p, a in zip(posicoes, ativos_dados):
        preco_atual = a.get("preco_atual", 0)
        preco_medio = p.get("preco_medio", 0)
        qtd = p.get("quantidade", 0)
        resultado_pct = ((preco_atual - preco_medio) / preco_medio * 100) if preco_medio else 0

        snapshot["ativos"].append({
            "ticker": p["ticker"],
            "quantidade": qtd,
            "preco_medio": preco_medio,
            "preco_atual": preco_atual,
            "valor_posicao": round(preco_atual * qtd, 2),
            "resultado_pct": round(resultado_pct, 2),
            "iv_atual": a.get("iv_atual"),
            "iv_rank_1y": a.get("iv_rank_1y"),
        })

    # Carrega histórico existente
    historico = []
    if os.path.exists(SNAPSHOT_FILE):
        try:
            with open(SNAPSHOT_FILE, "r") as f:
                historico = json.load(f)
        except Exception:
            historico = []

    historico.append(snapshot)

    # Mantém últimos 180 snapshots (~6 meses diários)
    if len(historico) > 180:
        historico = historico[-180:]

    with open(SNAPSHOT_FILE, "w") as f:
        json.dump(historico, f, indent=2, ensure_ascii=False)


def carregar_historico_ticker(ticker: str) -> list:
    """
    Retorna histórico de snapshots de um ticker específico.
    Útil para ver evolução de preço, resultado e IV ao longo do tempo.
    """
    if not os.path.exists(SNAPSHOT_FILE):
        return []

    try:
        with open(SNAPSHOT_FILE, "r") as f:
            historico = json.load(f)
    except Exception:
        return []

    registros = []
    for snap in historico:
        for ativo in snap.get("ativos", []):
            if ativo["ticker"] == ticker.upper():
                registros.append({
                    "data": snap["data"][:10],
                    "preco_atual": ativo.get("preco_atual", 0),
                    "resultado_pct": ativo.get("resultado_pct", 0),
                    "iv_atual": ativo.get("iv_atual"),
                    "iv_rank_1y": ativo.get("iv_rank_1y"),
                    "valor_posicao": ativo.get("valor_posicao", 0),
                })

    return registros


def get_historico_preco_yfinance(ticker: str, periodo: str = "6mo") -> pd.DataFrame:
    """
    Retorna histórico OHLCV do yfinance para gráficos.
    periodo: 1mo, 3mo, 6mo, 1y, 2y
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=periodo)
        return hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# GRÁFICO ASCII NO TERMINAL
# ─────────────────────────────────────────────

def grafico_ascii_preco(ticker: str, preco_medio: float = None) -> str:
    """
    Gera gráfico ASCII simples de preço nos últimos 90 dias usando plotext.
    Fallback para versão manual se plotext não estiver disponível.
    """
    hist = get_historico_preco_yfinance(ticker, "3mo")
    if hist.empty:
        return f"  [{ticker}] Sem dados históricos disponíveis."

    try:
        import plotext as plt

        precos = hist["Close"].tolist()
        datas = [str(d.date()) for d in hist.index]

        plt.clf()
        plt.plot(precos, color="cyan")
        if preco_medio:
            plt.hline(preco_medio, color="red")
        plt.title(f"{ticker} — Preço Fechamento (90 dias)")
        plt.xlabel("Dias")
        plt.ylabel("USD")
        plt.plotsize(70, 15)
        return plt.build()

    except ImportError:
        # Fallback: mini gráfico manual
        return _grafico_ascii_manual(ticker, hist["Close"].tolist(), preco_medio)


def _grafico_ascii_manual(ticker: str, precos: list, preco_medio: float = None) -> str:
    """Gráfico ASCII simples sem dependências externas."""
    if not precos:
        return ""

    altura = 10
    largura = min(60, len(precos))
    amostra = precos[-largura:]

    minv = min(amostra)
    maxv = max(amostra)
    rng = maxv - minv or 1

    linhas = []
    for row in range(altura, -1, -1):
        threshold = minv + (row / altura) * rng
        linha = ""
        for p in amostra:
            if p >= threshold:
                linha += "█"
            else:
                linha += " "
        label = f"${threshold:7.1f} |" if row % 3 == 0 else "         |"
        linhas.append(label + linha)

    linhas.append("         └" + "─" * largura)
    linhas.insert(0, f"\n  {ticker} — últimos {largura} pregões  (min: ${minv:.2f} | max: ${maxv:.2f})")
    if preco_medio:
        linhas.append(f"  ─── Linha vermelha = preço médio pago: ${preco_medio:.2f}")

    return "\n".join(linhas)
