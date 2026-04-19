
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field


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


def _baixar_historico(ticker: str, data_inicio: str = None) -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=data_inicio) if data_inicio else stock.history(period="5y")
        if hist.empty:
            hist = stock.history(period="3y")
        h = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if hasattr(h.index, "tz") and h.index.tz is not None:
            h.index = h.index.tz_localize(None)
        return h
    except Exception:
        return pd.DataFrame()


def _calcular_sharpe(retornos_diarios: pd.Series, rf_anual: float = 0.05) -> float:
    if retornos_diarios.empty or retornos_diarios.std() == 0:
        return 0.0
    rf_diario = rf_anual / 252
    return round(float((retornos_diarios.mean() - rf_diario) / retornos_diarios.std() * np.sqrt(252)), 2)


def _calcular_max_drawdown(curva: pd.Series) -> float:
    if curva.empty:
        return 0.0
    peak = curva.cummax()
    dd = (curva - peak) / peak * 100
    return round(float(dd.min()), 2)


def _retornos_mensais(hist: pd.DataFrame) -> dict:
    if hist.empty:
        return {}
    monthly = hist["Close"].resample("ME").last().pct_change() * 100
    result = {}
    for data, ret in monthly.dropna().items():
        result.setdefault(data.year, {})
        result[data.year][data.month] = round(float(ret), 2)
    return result


def _retornos_anuais(hist: pd.DataFrame) -> dict:
    if hist.empty:
        return {}
    annual = hist["Close"].resample("YE").last().pct_change() * 100
    return {d.year: round(float(r), 2) for d, r in annual.dropna().items()}


def backtest_ativo(ticker: str, preco_medio: float, dados_atuais: dict, data_compra: str = None) -> ResultadoBacktest:
    hist = _baixar_historico(ticker, data_compra)
    if hist.empty or len(hist) < 5:
        return ResultadoBacktest(
            ticker=ticker, periodo_inicio="N/A", periodo_fim="N/A",
            retorno_estrategia=0, retorno_buy_hold=0, retorno_sp500=0,
            sharpe_ratio=0, max_drawdown=0, win_rate=0, total_trades=0,
        )

    data_inicio = hist.index[0].strftime("%Y-%m-%d")
    data_fim = hist.index[-1].strftime("%Y-%m-%d")
    preco_ref = preco_medio if preco_medio and preco_medio > 0 else float(hist["Close"].iloc[0])
    preco_fim_val = float(hist["Close"].iloc[-1])
    retorno_real = (preco_fim_val / preco_ref - 1) * 100

    hist_semanal = hist["Close"].resample("W").last().dropna()
    capital_curva = [(d.strftime("%Y-%m-%d"), round(float(100.0 * v / preco_ref), 2)) for d, v in hist_semanal.items() if preco_ref > 0]

    retornos_diarios = hist["Close"].pct_change().dropna()
    sharpe = _calcular_sharpe(retornos_diarios)
    max_dd = _calcular_max_drawdown(pd.Series([v for _, v in capital_curva]))

    try:
        sp = yf.Ticker("^GSPC").history(start=data_inicio, end=data_fim)["Close"]
        if hasattr(sp.index, "tz") and sp.index.tz is not None:
            sp.index = sp.index.tz_localize(None)
        retorno_sp500 = float((sp.iloc[-1] / sp.iloc[0] - 1) * 100) if len(sp) > 1 else 0.0
    except Exception:
        retorno_sp500 = 0.0

    return ResultadoBacktest(
        ticker=ticker,
        periodo_inicio=data_inicio,
        periodo_fim=data_fim,
        retorno_estrategia=round(retorno_real, 2),
        retorno_buy_hold=round(retorno_real, 2),
        retorno_sp500=round(retorno_sp500, 2),
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=0.0,
        total_trades=0,
        trades=[],
        retornos_mensais=_retornos_mensais(hist),
        retornos_anuais=_retornos_anuais(hist),
        curva_retorno=capital_curva,
    )


def backtest_portfolio(posicoes: list, ativos_dados: list) -> ResultadoBacktestPortfolio:
    resultados = []
    for p, a in zip(posicoes, ativos_dados):
        resultados.append(backtest_ativo(p["ticker"], p.get("preco_medio", 0), a, p.get("data_compra")))

    valores_investidos = {p["ticker"]: p.get("preco_medio", 0) * p.get("quantidade", 0) for p in posicoes}
    total_investido = sum(valores_investidos.values()) or 1.0

    todas_datas = sorted(set(d for r in resultados for d, _ in r.curva_retorno))
    curvas = {}
    for r in resultados:
        if not r.curva_retorno:
            continue
        s = pd.Series({pd.Timestamp(d): v for d, v in r.curva_retorno}).sort_index()
        peso = valores_investidos.get(r.ticker, 0) / total_investido
        curvas[r.ticker] = (s, peso)

    curva_portfolio = []
    for data_str in todas_datas:
        dt = pd.Timestamp(data_str)
        vals_ponderados = []
        for ticker, (s, peso) in curvas.items():
            idx = s.index.get_indexer([dt], method="nearest")[0]
            if idx >= 0:
                vals_ponderados.append(float(s.iloc[idx]) * peso)
        if vals_ponderados:
            curva_portfolio.append((data_str, round(sum(vals_ponderados), 2)))

    curva_sp500 = []
    if todas_datas:
        try:
            sp = yf.Ticker("^GSPC").history(start=todas_datas[0], end=todas_datas[-1])["Close"]
            if hasattr(sp.index, "tz") and sp.index.tz is not None:
                sp.index = sp.index.tz_localize(None)
            if not sp.empty:
                base = float(sp.iloc[0])
                for d, v in zip([x.strftime("%Y-%m-%d") for x in sp.index], sp.values):
                    curva_sp500.append((d, round(v / base * 100, 2)))
        except Exception:
            pass

    rm_port = {}
    ra_port = {}
    for r in resultados:
        peso = valores_investidos.get(r.ticker, 0) / total_investido
        for ano, meses in r.retornos_mensais.items():
            rm_port.setdefault(ano, {})
            for mes, ret in meses.items():
                rm_port[ano].setdefault(mes, [])
                rm_port[ano][mes].append(ret * peso)
        for ano, ret in r.retornos_anuais.items():
            ra_port.setdefault(ano, [])
            ra_port[ano].append(ret * peso)

    retornos_mensais_media = {ano: {mes: round(sum(vals), 2) for mes, vals in meses.items()} for ano, meses in rm_port.items()}
    retornos_anuais_media = {ano: round(sum(vals), 2) for ano, vals in ra_port.items()}

    sp_rets = [r.retorno_sp500 for r in resultados if r.retorno_sp500 != 0]
    sharpes = [r.sharpe_ratio for r in resultados if r.sharpe_ratio != 0]
    dds = [r.max_drawdown for r in resultados if r.max_drawdown != 0]

    ret_ponderado = sum(r.retorno_estrategia * valores_investidos.get(r.ticker, 0) / total_investido for r in resultados if r.periodo_inicio != "N/A")
    periodo_inicio = min((r.periodo_inicio for r in resultados if r.periodo_inicio != "N/A"), default="N/A")
    periodo_fim = max((r.periodo_fim for r in resultados if r.periodo_fim != "N/A"), default="N/A")

    return ResultadoBacktestPortfolio(
        resultados_por_ativo=resultados,
        retorno_portfolio_total=round(ret_ponderado, 2),
        retorno_sp500_periodo=round(float(np.mean(sp_rets)) if sp_rets else 0, 2),
        sharpe_portfolio=round(float(np.mean(sharpes)) if sharpes else 0, 2),
        max_drawdown_portfolio=round(float(min(dds)) if dds else 0, 2),
        periodo_inicio=periodo_inicio,
        periodo_fim=periodo_fim,
        curva_portfolio=curva_portfolio,
        curva_sp500=curva_sp500,
        retornos_mensais_portfolio=retornos_mensais_media,
        retornos_anuais_portfolio=retornos_anuais_media,
    )
