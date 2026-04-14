"""
backtest.py — Engine de backtesting baseado em factor scores

Lógica:
  1. Baixa histórico máximo disponível via yfinance (até 10 anos)
  2. Recalcula factor scores semanalmente com dados disponíveis até aquela data
  3. Simula decisões: ADICIONAR (score>70), REALIZAR (score<30), MANTER
  4. Calcula retornos reais de cada decisão
  5. Compara vs Buy & Hold e vs S&P500

Limitação honesta:
  Backtest com dados de fundamentais históricos sofre de look-ahead bias
  nos dados fundamentais (yfinance retorna dados atuais, não históricos).
  Os dados de PREÇO são corretos historicamente.
  Recomendação: usar principalmente como validação de timing/momentum.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class TradeSimulado:
    data: str
    ticker: str
    acao: str           # BUY / SELL / HOLD
    preco: float
    score: float
    retorno_7d: float = 0.0
    retorno_30d: float = 0.0
    acertou: bool = None


@dataclass
class ResultadoBacktest:
    ticker: str
    periodo_inicio: str
    periodo_fim: str
    retorno_estrategia: float       # % total da estratégia simulada
    retorno_buy_hold: float         # % se tivesse comprado e mantido
    retorno_sp500: float            # % do S&P500 no mesmo período
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float                 # % de trades corretos
    total_trades: int
    trades: list[TradeSimulado] = field(default_factory=list)
    retornos_mensais: dict = field(default_factory=dict)
    retornos_anuais: dict = field(default_factory=dict)
    curva_retorno: list = field(default_factory=list)  # [(data, valor)]


@dataclass
class ResultadoBacktestPortfolio:
    resultados_por_ativo: list[ResultadoBacktest]
    retorno_portfolio_total: float
    retorno_sp500_periodo: float
    sharpe_portfolio: float
    max_drawdown_portfolio: float
    periodo_inicio: str
    periodo_fim: str
    curva_portfolio: list           # [(data, valor normalizado)]
    curva_sp500: list
    retornos_mensais_portfolio: dict
    retornos_anuais_portfolio: dict


def _baixar_historico(ticker: str, periodo: str = "max") -> pd.DataFrame:
    """Baixa histórico OHLCV. Retorna DataFrame vazio se falhar."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=periodo)
        if hist.empty:
            # Fallback para 10 anos
            hist = stock.history(period="10y")
        return hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def _calcular_score_momentum_historico(
    hist: pd.DataFrame,
    data_ref: pd.Timestamp,
    janela: int = 90
) -> float:
    """
    Score de momentum baseado apenas em dados de preço histórico.
    Usado no backtest para evitar look-ahead bias nos fundamentais.
    """
    try:
        dados_ate = hist[hist.index <= data_ref].tail(janela + 50)
        if len(dados_ate) < 20:
            return 50.0

        close = dados_ate["Close"]
        preco_atual = float(close.iloc[-1])

        # RSI 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        # Score RSI
        if rsi < 30:
            s_rsi = 85
        elif rsi < 40:
            s_rsi = 70
        elif rsi <= 60:
            s_rsi = 55
        elif rsi <= 70:
            s_rsi = 35
        else:
            s_rsi = 15

        # SMA 50
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else preco_atual
        dist50 = (preco_atual - sma50) / sma50 * 100
        if -5 <= dist50 <= 10:
            s_sma50 = 70
        elif dist50 > 10:
            s_sma50 = 45
        elif dist50 > -10:
            s_sma50 = 40
        else:
            s_sma50 = 20

        # SMA 200
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else preco_atual
        dist200 = (preco_atual - sma200) / sma200 * 100
        s_sma200 = float(np.clip((dist200 + 30) / 60 * 100, 0, 100))

        # Retorno 3 meses vs início do período
        preco_3m = float(close.iloc[0]) if len(close) >= janela else preco_atual
        ret_3m = (preco_atual - preco_3m) / preco_3m * 100
        s_tendencia = float(np.clip((ret_3m + 20) / 40 * 100, 0, 100))

        return round(float(np.mean([s_rsi, s_sma50, s_sma200, s_tendencia])), 1)

    except Exception:
        return 50.0


def _calcular_score_volatilidade_historico(
    hist: pd.DataFrame,
    data_ref: pd.Timestamp
) -> float:
    """Score de volatilidade baseado em HV histórica."""
    try:
        dados_ate = hist[hist.index <= data_ref].tail(252)
        if len(dados_ate) < 30:
            return 50.0

        close = dados_ate["Close"]
        retornos = np.log(close / close.shift(1)).dropna()

        hv_21 = float(retornos.iloc[-21:].std() * np.sqrt(252) * 100) if len(retornos) >= 21 else 20
        hv_252 = float(retornos.std() * np.sqrt(252) * 100) if len(retornos) >= 60 else hv_21

        # HV baixa vs histórico = melhor momento
        ratio_hv = hv_21 / (hv_252 + 0.01)
        if ratio_hv < 0.7:
            return 75.0
        elif ratio_hv < 1.0:
            return 60.0
        elif ratio_hv < 1.3:
            return 45.0
        else:
            return 25.0

    except Exception:
        return 50.0


def _score_combinado_historico(
    hist: pd.DataFrame,
    data_ref: pd.Timestamp
) -> float:
    """Score combinado usando apenas dados de preço (sem look-ahead bias)."""
    s_mom = _calcular_score_momentum_historico(hist, data_ref)
    s_vol = _calcular_score_volatilidade_historico(hist, data_ref)
    # 60% momentum + 40% volatilidade (dados disponíveis historicamente)
    return round(s_mom * 0.6 + s_vol * 0.4, 1)


def _calcular_sharpe(retornos_diarios: pd.Series, rf_anual: float = 0.04) -> float:
    """Sharpe Ratio anualizado."""
    if retornos_diarios.empty or retornos_diarios.std() == 0:
        return 0.0
    rf_diario = rf_anual / 252
    sharpe = (retornos_diarios.mean() - rf_diario) / retornos_diarios.std() * np.sqrt(252)
    return round(float(sharpe), 2)


def _calcular_max_drawdown(curva: pd.Series) -> float:
    """Máximo drawdown em % da curva de retorno."""
    if curva.empty:
        return 0.0
    peak = curva.cummax()
    drawdown = (curva - peak) / peak * 100
    return round(float(drawdown.min()), 2)


def _retornos_mensais(hist_preco: pd.DataFrame) -> dict:
    """Retornos mensais por ano: {2023: {1: 2.3, 2: -1.1, ...}, ...}"""
    if hist_preco.empty:
        return {}
    monthly = hist_preco["Close"].resample("ME").last().pct_change() * 100
    result = {}
    for data, ret in monthly.dropna().items():
        ano = data.year
        mes = data.month
        if ano not in result:
            result[ano] = {}
        result[ano][mes] = round(float(ret), 2)
    return result


def _retornos_anuais(hist_preco: pd.DataFrame) -> dict:
    """Retornos anuais: {2023: 15.2, 2022: -18.5, ...}"""
    if hist_preco.empty:
        return {}
    annual = hist_preco["Close"].resample("YE").last().pct_change() * 100
    return {data.year: round(float(ret), 2) for data, ret in annual.dropna().items()}


def backtest_ativo(
    ticker: str,
    preco_medio: float,
    dados_atuais: dict,
) -> ResultadoBacktest:
    """
    Roda backtest completo para um ativo com máximo histórico disponível.
    """
    print(f"  → Backtest {ticker}...")

    hist = _baixar_historico(ticker, "max")
    if hist.empty or len(hist) < 60:
        # Retorna resultado vazio se não há dados suficientes
        return ResultadoBacktest(
            ticker=ticker,
            periodo_inicio="N/A", periodo_fim="N/A",
            retorno_estrategia=0, retorno_buy_hold=0, retorno_sp500=0,
            sharpe_ratio=0, max_drawdown=0, win_rate=0, total_trades=0,
        )

    # Periodo real disponível
    data_inicio = hist.index[0].strftime("%Y-%m-%d")
    data_fim = hist.index[-1].strftime("%Y-%m-%d")

    # ── Simulação semanal ─────────────────────────────────────────
    trades = []
    capital = 10000.0        # capital inicial simulado
    capital_curva = []
    posicao_aberta = False
    preco_entrada = 0.0

    # Itera semanalmente
    datas_semanais = hist.resample("W").last().index

    for data_ref in datas_semanais:
        if data_ref not in hist.index:
            # Pega o dia útil mais próximo
            datas_validas = hist.index[hist.index <= data_ref]
            if datas_validas.empty:
                continue
            data_ref = datas_validas[-1]

        preco_atual = float(hist.loc[data_ref, "Close"])
        score = _score_combinado_historico(hist, data_ref)

        # Lógica de decisão baseada no score
        if score >= 68 and not posicao_aberta:
            acao = "BUY"
            posicao_aberta = True
            preco_entrada = preco_atual
            trades.append(TradeSimulado(
                data=data_ref.strftime("%Y-%m-%d"),
                ticker=ticker, acao=acao,
                preco=preco_atual, score=score,
            ))

        elif score <= 32 and posicao_aberta:
            acao = "SELL"
            posicao_aberta = False
            retorno = (preco_atual - preco_entrada) / preco_entrada * 100
            capital *= (1 + retorno / 100)
            if trades:
                trades[-1].retorno_30d = round(retorno, 2)
                trades[-1].acertou = retorno > 0
            trades.append(TradeSimulado(
                data=data_ref.strftime("%Y-%m-%d"),
                ticker=ticker, acao=acao,
                preco=preco_atual, score=score,
            ))
            preco_entrada = 0.0

        capital_curva.append((data_ref.strftime("%Y-%m-%d"), round(capital, 2)))

    # ── Métricas ──────────────────────────────────────────────────
    preco_inicio = float(hist["Close"].iloc[0])
    preco_fim = float(hist["Close"].iloc[-1])

    retorno_buy_hold = (preco_fim - preco_inicio) / preco_inicio * 100
    retorno_estrategia = (capital - 10000) / 10000 * 100

    # S&P500 no mesmo período
    try:
        sp500 = yf.Ticker("^GSPC").history(
            start=hist.index[0], end=hist.index[-1]
        )["Close"]
        retorno_sp500 = float(
            (sp500.iloc[-1] - sp500.iloc[0]) / sp500.iloc[0] * 100
        ) if not sp500.empty else 0.0
    except Exception:
        retorno_sp500 = 0.0

    # Sharpe e Drawdown da curva de capital
    if capital_curva:
        valores = pd.Series([v for _, v in capital_curva])
        retornos_diarios = valores.pct_change().dropna()
        sharpe = _calcular_sharpe(retornos_diarios)
        max_dd = _calcular_max_drawdown(valores)
    else:
        sharpe = 0.0
        max_dd = 0.0

    # Win rate
    trades_com_resultado = [t for t in trades if t.acertou is not None]
    win_rate = (
        sum(1 for t in trades_com_resultado if t.acertou) /
        len(trades_com_resultado) * 100
        if trades_com_resultado else 0.0
    )

    return ResultadoBacktest(
        ticker=ticker,
        periodo_inicio=data_inicio,
        periodo_fim=data_fim,
        retorno_estrategia=round(retorno_estrategia, 2),
        retorno_buy_hold=round(retorno_buy_hold, 2),
        retorno_sp500=round(retorno_sp500, 2),
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=round(win_rate, 1),
        total_trades=len([t for t in trades if t.acao == "BUY"]),
        trades=trades,
        retornos_mensais=_retornos_mensais(hist),
        retornos_anuais=_retornos_anuais(hist),
        curva_retorno=capital_curva,
    )


def backtest_portfolio(
    posicoes: list[dict],
    ativos_dados: list[dict],
) -> ResultadoBacktestPortfolio:
    """
    Roda backtest para todos os ativos e agrega em visão de portfólio.
    """
    print("\n[bold cyan]📈 Rodando backtest (máximo histórico disponível)...[/bold cyan]")

    resultados = []
    for p, a in zip(posicoes, ativos_dados):
        r = backtest_ativo(p["ticker"], p.get("preco_medio", 0), a)
        resultados.append(r)

    # ── Agrega curva do portfólio (média ponderada igual) ─────────
    todas_datas = set()
    for r in resultados:
        for data, _ in r.curva_retorno:
            todas_datas.add(data)
    todas_datas = sorted(todas_datas)

    # Normaliza cada curva para começar em 100
    curvas_normalizadas = {}
    for r in resultados:
        if not r.curva_retorno:
            continue
        base = r.curva_retorno[0][1]
        curvas_normalizadas[r.ticker] = {
            data: val / base * 100
            for data, val in r.curva_retorno
        }

    # Portfólio = média das curvas normalizadas
    curva_portfolio = []
    for data in todas_datas:
        vals = [
            curvas_normalizadas[t][data]
            for t in curvas_normalizadas
            if data in curvas_normalizadas[t]
        ]
        if vals:
            curva_portfolio.append((data, round(float(np.mean(vals)), 2)))

    # S&P500 normalizado para o mesmo período
    curva_sp500 = []
    if todas_datas:
        try:
            sp500 = yf.Ticker("^GSPC").history(
                start=todas_datas[0], end=todas_datas[-1]
            )["Close"]
            if not sp500.empty:
                base_sp = float(sp500.iloc[0])
                for data, val in zip(
                    [d.strftime("%Y-%m-%d") for d in sp500.index],
                    sp500.values
                ):
                    curva_sp500.append((data, round(val / base_sp * 100, 2)))
        except Exception:
            pass

    # Retornos mensais e anuais agregados do portfólio
    retornos_mensais_port = {}
    retornos_anuais_port = {}

    for r in resultados:
        for ano, meses in r.retornos_mensais.items():
            if ano not in retornos_mensais_port:
                retornos_mensais_port[ano] = {}
            for mes, ret in meses.items():
                if mes not in retornos_mensais_port[ano]:
                    retornos_mensais_port[ano][mes] = []
                retornos_mensais_port[ano][mes].append(ret)

        for ano, ret in r.retornos_anuais.items():
            if ano not in retornos_anuais_port:
                retornos_anuais_port[ano] = []
            retornos_anuais_port[ano].append(ret)

    # Médias
    retornos_mensais_media = {
        ano: {mes: round(float(np.mean(vals)), 2) for mes, vals in meses.items()}
        for ano, meses in retornos_mensais_port.items()
    }
    retornos_anuais_media = {
        ano: round(float(np.mean(vals)), 2)
        for ano, vals in retornos_anuais_port.items()
    }

    # Métricas do portfólio
    ret_portfolio = float(np.mean([r.retorno_estrategia for r in resultados if r.total_trades > 0])) if resultados else 0
    ret_sp500 = float(np.mean([r.retorno_sp500 for r in resultados if r.retorno_sp500 != 0])) if resultados else 0
    sharpe_med = float(np.mean([r.sharpe_ratio for r in resultados if r.sharpe_ratio != 0])) if resultados else 0
    max_dd = float(min([r.max_drawdown for r in resultados])) if resultados else 0

    periodo_inicio = min((r.periodo_inicio for r in resultados if r.periodo_inicio != "N/A"), default="N/A")
    periodo_fim = max((r.periodo_fim for r in resultados if r.periodo_fim != "N/A"), default="N/A")

    return ResultadoBacktestPortfolio(
        resultados_por_ativo=resultados,
        retorno_portfolio_total=round(ret_portfolio, 2),
        retorno_sp500_periodo=round(ret_sp500, 2),
        sharpe_portfolio=round(sharpe_med, 2),
        max_drawdown_portfolio=round(max_dd, 2),
        periodo_inicio=periodo_inicio,
        periodo_fim=periodo_fim,
        curva_portfolio=curva_portfolio,
        curva_sp500=curva_sp500,
        retornos_mensais_portfolio=retornos_mensais_media,
        retornos_anuais_portfolio=retornos_anuais_media,
    )
