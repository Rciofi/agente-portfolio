"""
collector.py — Coleta dados fundamentais, técnicos e notícias
Fontes: FinViz (fundamentais + notícias) + yfinance (preço atual + histórico)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

try:
    from finvizfinance.quote import finvizfinance
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False


def get_current_price(ticker: str) -> dict:
    """Preço atual, variação diária e volume via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5d")

        preco_atual = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        preco_anterior = info.get("previousClose", preco_atual)
        variacao_pct = ((preco_atual - preco_anterior) / preco_anterior * 100) if preco_anterior else 0

        # Histórico 90 dias para tendência
        hist_90 = stock.history(period="3mo")
        media_90 = hist_90["Close"].mean() if not hist_90.empty else preco_atual
        tendencia = "ALTA" if preco_atual > media_90 else "BAIXA"

        # Máxima e mínima 52 semanas
        high_52 = info.get("fiftyTwoWeekHigh", 0)
        low_52 = info.get("fiftyTwoWeekLow", 0)
        pct_from_high = ((preco_atual - high_52) / high_52 * 100) if high_52 else 0
        pct_from_low = ((preco_atual - low_52) / low_52 * 100) if low_52 else 0

        return {
            "ticker": ticker,
            "preco_atual": round(preco_atual, 2),
            "variacao_dia_pct": round(variacao_pct, 2),
            "volume": info.get("volume", 0),
            "media_volume": info.get("averageVolume", 0),
            "tendencia_90d": tendencia,
            "media_movel_90d": round(media_90, 2),
            "high_52w": high_52,
            "low_52w": low_52,
            "pct_abaixo_high_52w": round(pct_from_high, 2),
            "pct_acima_low_52w": round(pct_from_low, 2),
        }
    except Exception as e:
        return {"ticker": ticker, "erro": str(e), "preco_atual": 0}


def get_fundamentals(ticker: str) -> dict:
    """Dados fundamentais via yfinance (fallback robusto)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "nome": info.get("longName", ticker),
            "setor": info.get("sector", "N/A"),
            "industria": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "peg_ratio": info.get("pegRatio", None),
            "pb_ratio": info.get("priceToBook", None),
            "dividend_yield": info.get("dividendYield", 0),
            "roe": info.get("returnOnEquity", None),
            "roa": info.get("returnOnAssets", None),
            "margem_lucro": info.get("profitMargins", None),
            "crescimento_receita": info.get("revenueGrowth", None),
            "crescimento_lucro": info.get("earningsGrowth", None),
            "divida_equity": info.get("debtToEquity", None),
            "free_cashflow": info.get("freeCashflow", None),
            "recomendacao_analistas": info.get("recommendationKey", "N/A"),
            "target_price": info.get("targetMeanPrice", None),
            "upside_analysts": round(
                ((info.get("targetMeanPrice", 0) - info.get("currentPrice", 0)) /
                 info.get("currentPrice", 1) * 100), 2
            ) if info.get("targetMeanPrice") and info.get("currentPrice") else None,
        }
    except Exception as e:
        return {"nome": ticker, "erro_fundamentals": str(e)}


def get_finviz_data(ticker: str) -> dict:
    """Dados extras do FinViz se disponível: RSI, insider trading, notícias."""
    result = {"noticias": [], "insider": [], "rsi": None}

    if not FINVIZ_AVAILABLE:
        return result

    try:
        fv = finvizfinance(ticker)

        # Notícias recentes
        try:
            news_df = fv.ticker_news()
            if news_df is not None and not news_df.empty:
                noticias = news_df.head(5)[["Date", "Title"]].to_dict("records")
                result["noticias"] = [
                    {"data": str(n.get("Date", "")), "titulo": n.get("Title", "")}
                    for n in noticias
                ]
        except Exception:
            pass

        # Dados técnicos (RSI, SMA, etc.)
        try:
            fundaments = fv.ticker_fundaments()
            if fundaments:
                result["rsi"] = fundaments.get("RSI (14)", None)
                result["sma20"] = fundaments.get("SMA20", None)
                result["sma50"] = fundaments.get("SMA50", None)
                result["sma200"] = fundaments.get("SMA200", None)
                result["beta"] = fundaments.get("Beta", None)
                result["atr"] = fundaments.get("ATR (14)", None)
        except Exception:
            pass

        # Insider trading
        try:
            insider_df = fv.ticker_inside_trader()
            if insider_df is not None and not insider_df.empty:
                result["insider"] = insider_df.head(3).to_dict("records")
        except Exception:
            pass

    except Exception:
        pass

    return result


def coletar_dados_ativo(ticker: str) -> dict:
    """Agrega todos os dados de um ativo incluindo IV e HV."""
    print(f"  → Coletando {ticker}...")

    preco = get_current_price(ticker)
    fundamentals = get_fundamentals(ticker)
    finviz = get_finviz_data(ticker)

    # Volatilidade implícita e histórica
    from agent.volatility import get_implied_volatility, get_historical_volatility
    print(f"  → Volatilidade implícita {ticker}...")
    iv_data = get_implied_volatility(ticker)
    hv_data = get_historical_volatility(ticker)

    return {
        **preco,
        **fundamentals,
        **finviz,
        **iv_data,
        **hv_data,
    }


def carregar_portfolio(caminho_csv: str) -> pd.DataFrame:
    """Lê o CSV do portfólio e calcula métricas básicas."""
    df = pd.read_csv(caminho_csv)
    df.columns = df.columns.str.strip().str.lower()

    # Remove linha de CASH (não é ativo negociável)
    capital_disponivel = 0
    if "cash" in df["ticker"].str.upper().values:
        cash_row = df[df["ticker"].str.upper() == "CASH"]
        capital_disponivel = float(cash_row["capital_disponivel"].values[0])
        df = df[df["ticker"].str.upper() != "CASH"]

    df = df[df["quantidade"] > 0].copy()
    df["ticker"] = df["ticker"].str.upper()

    # Garante coluna data_compra (pode não existir em CSVs antigos)
    if "data_compra" not in df.columns:
        df["data_compra"] = None
        print("  ⚠️  portfolio.csv sem coluna 'data_compra' — backtest usará histórico completo.")
        print("     Adicione 'data_compra' para backtest preciso desde a data de compra real.")

    return df, capital_disponivel
