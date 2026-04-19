"""
Risk Parity Benchmarks
─────────────────────
Implementa duas variantes para comparação no pipeline do agente:

  1. Risk Parity Puro (Naive RP)
     wi ∝ 1/σi  →  cada ativo recebe peso inversamente proporcional
     à sua volatilidade individual. Simples, sem otimização.
     Referência: Qian (2005), "Risk Parity Portfolios"

  2. Equal Risk Contribution (ERC)
     Cada ativo contribui IGUALMENTE para a variância total da carteira.
     Solução via SLSQP; mais robusto que RP puro em universos correlacionados.
     Referência: Roncalli (2013), "Introduction to Risk Parity and Budgeting"

Cada variante pode usar dois estimadores de volatilidade:
  - Rolling 252 dias (janela histórica — padrão Markowitz)
  - EWMA span=50 dias (ex-ante — padrão DLS / Zhang et al. 2021)

Isso gera 4 séries para análise:
  RP_Rolling, RP_EWMA, ERC_Rolling, ERC_EWMA
"""

import numpy as np
import pandas as pd
import logging
from typing import Literal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# ESTIMADORES DE VOLATILIDADE
# ─────────────────────────────────────────────────────────────────────

def rolling_vol(returns: np.ndarray, window: int = 252) -> np.ndarray:
    """
    Volatilidade realizada via janela móvel simples.
    returns: (T, n_assets)
    retorna: (n_assets,) — vol anualizada no final da janela
    """
    tail = returns[-window:]
    vol = tail.std(axis=0) * np.sqrt(252)
    return np.clip(vol, 1e-6, None)


def ewma_vol(returns: np.ndarray, span: int = 50) -> np.ndarray:
    """
    Volatilidade ex-ante via EWMA (mesma metodologia de Zhang et al. 2021).
    Usa apenas os últimos `span * 3` dias para eficiência.
    returns: (T, n_assets)
    retorna: (n_assets,) — vol diária (não anualizada) no final da série
    """
    alpha = 2.0 / (span + 1)
    tail = returns[-span * 3:]  # janela suficiente para convergência
    T, n = tail.shape
    var = tail[:span].var(axis=0)
    for t in range(span, T):
        var = alpha * tail[t] ** 2 + (1 - alpha) * var
    vol_daily = np.sqrt(np.clip(var, 1e-10, None))
    return vol_daily * np.sqrt(252)  # anualiza para consistência com rolling


# ─────────────────────────────────────────────────────────────────────
# PESOS: RISK PARITY PURO (Naive RP)
# ─────────────────────────────────────────────────────────────────────

def naive_rp_weights(vol: np.ndarray) -> np.ndarray:
    """
    wi = (1/σi) / Σ(1/σj)

    Não usa covariância — ignora correlações entre ativos.
    Rápido e interpretável; bom baseline.
    """
    inv_vol = 1.0 / np.clip(vol, 1e-8, None)
    w = inv_vol / inv_vol.sum()
    return w


# ─────────────────────────────────────────────────────────────────────
# PESOS: EQUAL RISK CONTRIBUTION (ERC)
# ─────────────────────────────────────────────────────────────────────

def erc_weights(
    returns: np.ndarray,
    window: int = 252,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Equal Risk Contribution via SLSQP.

    Minimiza: Σ_i Σ_j (RCi - RCj)²
    onde RC_i = wi * (Σw)_i / (w'Σw)  — contribuição marginal ao risco

    Garante que cada ativo contribui igualmente para σ²_p.
    Usa covariância da janela histórica (ledoit-wolf quando disponível).

    returns: (T, n_assets)
    retorna: (n_assets,) — pesos ERC
    """
    from scipy.optimize import minimize

    tail = returns[-window:]
    n = tail.shape[1]

    # Estimativa de covariância com regularização
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(tail)
        cov = lw.covariance_
    except ImportError:
        cov = np.cov(tail.T) + np.eye(n) * 1e-6

    cov = np.array(cov, dtype=np.float64)

    def risk_contributions(w):
        port_var = float(w @ cov @ w)
        if port_var < 1e-12:
            return np.ones(n) / n
        marginal = cov @ w
        rc = w * marginal / port_var
        return rc

    def objective(w):
        rc = risk_contributions(w)
        # Soma dos quadrados das diferenças entre todas as contribuições
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    def grad(w):
        # Gradiente numérico (suficiente para n ≤ 100)
        eps = 1e-6
        g = np.zeros(n)
        f0 = objective(w)
        for i in range(n):
            w_plus = w.copy()
            w_plus[i] += eps
            g[i] = (objective(w_plus) - f0) / eps
        return g

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        objective, w0,
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": max_iter, "disp": False},
    )

    if result.success:
        w = np.clip(result.x, 0, None)
        w /= w.sum()
    else:
        # Fallback: naive RP
        logger.warning("[ERC] Otimização não convergiu — usando Naive RP como fallback")
        vol = np.sqrt(np.diag(cov)) * np.sqrt(252)
        w = naive_rp_weights(vol)

    return w


# ─────────────────────────────────────────────────────────────────────
# BACKTEST GENÉRICO
# ─────────────────────────────────────────────────────────────────────

def _run_rp_backtest(
    prices: pd.DataFrame,
    test_start: str,
    test_end: str,
    strategy: Literal["naive_rp", "erc"],
    vol_estimator: Literal["rolling", "ewma"],
    rolling_window: int = 252,
    ewma_span: int = 50,
    rebalance_freq: int = 5,
    cost_rate: float = 0.0001,
    strategy_label: str = "",
) -> dict:
    """
    Motor de backtest compartilhado por Naive RP e ERC.

    Walk-forward: para cada data de rebalanceamento, calcula pesos
    usando apenas dados históricos disponíveis até aquela data.
    """
    all_returns = prices.pct_change().dropna()
    test_returns = all_returns.loc[test_start:test_end]
    n = prices.shape[1]
    tickers = prices.columns.tolist()

    weights_list = []
    portfolio_returns = []
    current_weights = None

    for i, date in enumerate(test_returns.index):

        if i % rebalance_freq == 0:
            # Histórico disponível até esta data
            hist_ret = all_returns.loc[:date].values.astype(np.float64)

            if len(hist_ret) < max(rolling_window, ewma_span * 3) + 10:
                current_weights = np.ones(n) / n
            else:
                if strategy == "naive_rp":
                    if vol_estimator == "rolling":
                        vol = rolling_vol(hist_ret, window=rolling_window)
                    else:
                        vol = ewma_vol(hist_ret, span=ewma_span)
                    current_weights = naive_rp_weights(vol)

                elif strategy == "erc":
                    current_weights = erc_weights(
                        hist_ret,
                        window=rolling_window if vol_estimator == "rolling" else ewma_span * 3,
                    )

        if current_weights is None:
            current_weights = np.ones(n) / n

        weights_list.append(current_weights.copy())
        day_ret = test_returns.iloc[i].values
        portfolio_returns.append(float((current_weights * day_ret).sum()))

    weights_arr = np.array(weights_list)
    port_ret = np.array(portfolio_returns)

    # Custo de transação (turnover diário)
    turnover = np.abs(np.diff(weights_arr, axis=0, prepend=weights_arr[:1])).sum(axis=1)
    port_ret -= cost_rate * turnover

    # Métricas
    cumulative = np.cumprod(1 + port_ret)
    ann_ret = float(cumulative[-1] ** (252 / len(port_ret)) - 1)
    ann_vol = float(port_ret.std() * np.sqrt(252))
    sharpe = ann_ret / (ann_vol + 1e-8)
    running_max = np.maximum.accumulate(cumulative)
    mdd = float(((cumulative - running_max) / running_max).min())

    logger.info(
        f"[{strategy_label}] Sharpe={sharpe:.3f} | "
        f"Ret={ann_ret:.1%} | Vol={ann_vol:.1%} | MDD={mdd:.1%}"
    )

    weights_df = pd.DataFrame(
        weights_arr, index=test_returns.index, columns=tickers
    )

    return {
        "strategy": strategy_label,
        "metrics": {
            "sharpe_ratio":       round(sharpe, 3),
            "annual_return":      round(ann_ret, 4),
            "annual_volatility":  round(ann_vol, 4),
            "max_drawdown":       round(mdd, 4),
            "cumulative_return":  round(float(cumulative[-1]) - 1, 4),
        },
        "weights_df":        weights_df,
        "returns_series":    pd.Series(port_ret, index=test_returns.index, name=strategy_label),
        "cumulative_series": pd.Series(cumulative, index=test_returns.index, name=strategy_label),
    }


# ─────────────────────────────────────────────────────────────────────
# API PÚBLICA — 4 variantes
# ─────────────────────────────────────────────────────────────────────

def run_naive_rp_rolling(prices, test_start, test_end, **kw) -> dict:
    """Risk Parity Puro · Volatilidade Rolling 252d"""
    return _run_rp_backtest(
        prices, test_start, test_end,
        strategy="naive_rp", vol_estimator="rolling",
        strategy_label="RP Puro (Rolling 252d)", **kw
    )


def run_naive_rp_ewma(prices, test_start, test_end, **kw) -> dict:
    """Risk Parity Puro · Volatilidade EWMA span=50"""
    return _run_rp_backtest(
        prices, test_start, test_end,
        strategy="naive_rp", vol_estimator="ewma",
        strategy_label="RP Puro (EWMA 50d)", **kw
    )


def run_erc_rolling(prices, test_start, test_end, **kw) -> dict:
    """Equal Risk Contribution · Covariância Rolling 252d"""
    return _run_rp_backtest(
        prices, test_start, test_end,
        strategy="erc", vol_estimator="rolling",
        strategy_label="ERC (Rolling 252d)", **kw
    )


def run_erc_ewma(prices, test_start, test_end, **kw) -> dict:
    """Equal Risk Contribution · Covariância EWMA span=50"""
    return _run_rp_backtest(
        prices, test_start, test_end,
        strategy="erc", vol_estimator="ewma",
        strategy_label="ERC (EWMA 50d)", **kw
    )
