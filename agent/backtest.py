import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field

LOOKBACK_PERIOD = "3y"
RISK_FREE_ANNUAL = 0.045

@dataclass
class ResultadoBacktest:
    ticker: str
    periodo_inicio: str
    periodo_fim: str
    retorno_estrategia: float
    retorno_buy_hold: float
    retorno_sp500: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    trades: list = field(default_factory=list)
    retornos_mensais: dict = field(default_factory=dict)
    retornos_anuais: dict = field(default_factory=dict)
    curva_retorno: list = field(default_factory=list)

@dataclass
class ResultadoBacktestPortfolio:
    resultados_por_ativo: list
    retorno_portfolio_total: float
    retorno_sp500_periodo: float
    sharpe_portfolio: float
    max_drawdown_portfolio: float
    periodo_inicio: str
    periodo_fim: str
    curva_portfolio: list
    curva_sp500: list
    retornos_mensais_portfolio: dict
    retornos_anuais_portfolio: dict

def _resultado_portfolio_vazio(resultados=None):
    return ResultadoBacktestPortfolio(
        resultados_por_ativo=resultados or [],
        retorno_portfolio_total=0.0,
        retorno_sp500_periodo=0.0,
        sharpe_portfolio=0.0,
        max_drawdown_portfolio=0.0,
        periodo_inicio="N/A",
        periodo_fim="N/A",
        curva_portfolio=[],
        curva_sp500=[],
        retornos_mensais_portfolio={},
        retornos_anuais_portfolio={}
    )

def _flatten_close(df, ticker=None):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if ticker is not None and ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        elif "Close" in df.columns.get_level_values(0):
            sub = df["Close"]
            s = sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub
        else:
            return pd.Series(dtype=float)
    else:
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()

def _baixar_historico(ticker: str, periodo: str = LOOKBACK_PERIOD) -> pd.Series:
    try:
        df = yf.download(ticker, period=periodo, auto_adjust=True, progress=False, threads=False)
        return _flatten_close(df, ticker)
    except Exception:
        return pd.Series(dtype=float)

def _baixar_historico_desde(ticker: str, start) -> pd.Series:
    try:
        start_ts = pd.to_datetime(start, errors="coerce")
        if pd.isna(start_ts):
            return _baixar_historico(ticker, LOOKBACK_PERIOD)
        df = yf.download(
            ticker,
            start=(start_ts - pd.Timedelta(days=10)).date().isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return _flatten_close(df, ticker)
    except Exception:
        return pd.Series(dtype=float)

def _calcular_sharpe(retornos_diarios: pd.Series, rf_anual: float = RISK_FREE_ANNUAL) -> float:
    retornos_diarios = retornos_diarios.dropna()
    if retornos_diarios.empty or retornos_diarios.std() == 0:
        return 0.0
    rf_diario = rf_anual / 252
    sharpe = (retornos_diarios.mean() - rf_diario) / retornos_diarios.std() * np.sqrt(252)
    return round(float(sharpe), 2)

def _max_drawdown(curva: pd.Series) -> float:
    curva = curva.dropna()
    if curva.empty:
        return 0.0
    roll_max = curva.cummax()
    drawdown = curva / roll_max - 1.0
    return round(float(drawdown.min() * 100), 2)

def _retornos_mensais(curva: pd.Series) -> dict:
    if curva.empty:
        return {}
    monthly = curva.resample("ME").last().pct_change() * 100
    out = {}
    for d, r in monthly.dropna().items():
        out.setdefault(d.year, {})
        out[d.year][d.month] = round(float(r), 2)
    return out

def _retornos_anuais(curva: pd.Series) -> dict:
    if curva.empty:
        return {}
    annual = curva.resample("YE").last().pct_change() * 100
    return {d.year: round(float(r), 2) for d, r in annual.dropna().items()}

def _curva_base_100(precos: pd.Series) -> pd.Series:
    precos = precos.dropna()
    if precos.empty:
        return pd.Series(dtype=float)
    return precos / float(precos.iloc[0]) * 100.0

def _curva_semanal_lista(curva_diaria: pd.Series) -> list:
    if curva_diaria.empty:
        return []
    curva_semanal = curva_diaria.resample("W").last().dropna()
    return [(str(d.date()), round(float(v), 2)) for d, v in curva_semanal.items()]

def _retorno_total(curva: pd.Series) -> float:
    curva = curva.dropna()
    if curva.empty:
        return 0.0
    return round(float((curva.iloc[-1] / curva.iloc[0] - 1.0) * 100), 2)

def _filtrar_desde_data_compra(hist: pd.Series, data_compra):
    if hist.empty or not data_compra:
        return hist
    data = pd.to_datetime(data_compra, errors="coerce")
    if pd.isna(data):
        return hist
    return hist.loc[hist.index >= data]

def backtest_ativo(ticker, preco_medio, dados_atuais, data_compra=None):
    hist = _baixar_historico_desde(ticker, data_compra) if data_compra else _baixar_historico(ticker, LOOKBACK_PERIOD)
    hist = _filtrar_desde_data_compra(hist, data_compra)
    if hist.empty or len(hist) < 30:
        return ResultadoBacktest(
            ticker=ticker, periodo_inicio="N/A", periodo_fim="N/A",
            retorno_estrategia=0, retorno_buy_hold=0, retorno_sp500=0,
            sharpe_ratio=0, max_drawdown=0, win_rate=0, total_trades=0,
        )
    curva = _curva_base_100(hist)
    retornos = hist.pct_change().dropna()
    sharpe = _calcular_sharpe(retornos)
    drawdown = _max_drawdown(curva)
    sp = _baixar_historico("^GSPC", LOOKBACK_PERIOD)
    retorno_sp500 = 0.0
    if not sp.empty:
        sp = sp.reindex(hist.index).ffill().dropna()
        if len(sp) > 1:
            sp_curve = _curva_base_100(sp)
            retorno_sp500 = _retorno_total(sp_curve)
    if preco_medio and float(preco_medio) > 0 and len(hist) > 0:
        retorno = round(float((hist.iloc[-1] / float(preco_medio) - 1.0) * 100), 2)
    else:
        retorno = _retorno_total(curva)
    return ResultadoBacktest(
        ticker=ticker,
        periodo_inicio=str(hist.index[0].date()),
        periodo_fim=str(hist.index[-1].date()),
        retorno_estrategia=retorno,
        retorno_buy_hold=retorno,
        retorno_sp500=round(retorno_sp500, 2),
        sharpe_ratio=sharpe,
        max_drawdown=drawdown,
        win_rate=0.0,
        total_trades=0,
        trades=[],
        retornos_mensais=_retornos_mensais(curva),
        retornos_anuais=_retornos_anuais(curva),
        curva_retorno=_curva_semanal_lista(curva),
    )

def backtest_portfolio(posicoes, ativos_dados):
    resultados_individuais = [
        backtest_ativo(p["ticker"], p.get("preco_medio", 0), {}, p.get("data_compra"))
        for p in posicoes
    ]
    precos = {}
    custos = {}
    datas_compra = {}
    for p, a in zip(posicoes, ativos_dados):
        ticker = p["ticker"]
        data_compra = p.get("data_compra")
        hist = _baixar_historico_desde(ticker, data_compra) if data_compra else _baixar_historico(ticker, LOOKBACK_PERIOD)
        hist = _filtrar_desde_data_compra(hist, data_compra)
        if hist.empty or len(hist) < 2:
            continue
        qtd = float(p.get("quantidade", 0) or 0)
        preco_medio = float(p.get("preco_medio", 0) or 0)
        custo = qtd * preco_medio
        if qtd <= 0 or custo <= 0:
            continue
        precos[ticker] = hist
        custos[ticker] = custo
        datas_compra[ticker] = hist.index[0]
    if not precos:
        return _resultado_portfolio_vazio(resultados_individuais)

    valores = {}
    capital_investido = {}
    for p in posicoes:
        ticker = p["ticker"]
        if ticker not in precos:
            continue
        qtd = float(p.get("quantidade", 0) or 0)
        valores[ticker] = precos[ticker] * qtd
        capital_investido[ticker] = pd.Series(custos[ticker], index=precos[ticker].index)

    df_valores = pd.concat(valores, axis=1).sort_index().ffill()
    df_custos = pd.concat(capital_investido, axis=1).sort_index().ffill()
    valor_total = df_valores.sum(axis=1, min_count=1)
    custo_total = df_custos.sum(axis=1, min_count=1)
    curva_capital = (valor_total / custo_total.replace(0, np.nan) * 100.0).dropna()
    if curva_capital.empty or len(curva_capital) < 30:
        return _resultado_portfolio_vazio(resultados_individuais)
    curva_port = curva_capital
    ret_port_diarios = curva_port.pct_change().dropna()
    retorno_total = round(float(curva_port.iloc[-1] - 100.0), 2)
    sharpe = _calcular_sharpe(ret_port_diarios)
    drawdown = _max_drawdown(curva_port)
    sp = _baixar_historico("^GSPC", LOOKBACK_PERIOD)
    curva_sp500 = []
    retorno_sp = 0.0
    if not sp.empty:
        bench_parts = {}
        bench_costs = {}
        for ticker, custo in custos.items():
            data = datas_compra.get(ticker)
            sp_lote = sp.loc[sp.index >= data]
            if sp_lote.empty:
                continue
            base = float(sp_lote.iloc[0])
            if base <= 0:
                continue
            bench_parts[ticker] = sp_lote / base * custo
            bench_costs[ticker] = pd.Series(custo, index=sp_lote.index)
        if bench_parts:
            df_bench = pd.concat(bench_parts, axis=1).sort_index().ffill()
            df_bench_cost = pd.concat(bench_costs, axis=1).sort_index().ffill()
            valor_bench = df_bench.sum(axis=1, min_count=1)
            custo_bench = df_bench_cost.sum(axis=1, min_count=1)
            sp_curve = (valor_bench / custo_bench.replace(0, np.nan) * 100.0).dropna()
            sp_curve = sp_curve.reindex(curva_port.index).ffill().dropna()
            if len(sp_curve) > 1:
                retorno_sp = round(float(sp_curve.iloc[-1] - 100.0), 2)
                curva_sp500 = _curva_semanal_lista(sp_curve)
    return ResultadoBacktestPortfolio(
        resultados_por_ativo=resultados_individuais,
        retorno_portfolio_total=retorno_total,
        retorno_sp500_periodo=round(retorno_sp, 2),
        sharpe_portfolio=sharpe,
        max_drawdown_portfolio=drawdown,
        periodo_inicio=str(curva_port.index[0].date()),
        periodo_fim=str(curva_port.index[-1].date()),
        curva_portfolio=_curva_semanal_lista(curva_port),
        curva_sp500=curva_sp500,
        retornos_mensais_portfolio=_retornos_mensais(curva_port),
        retornos_anuais_portfolio=_retornos_anuais(curva_port),
    )
