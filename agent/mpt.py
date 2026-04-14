"""
mpt.py — Modern Portfolio Theory
Implementa:
  - Matriz de correlação entre ativos
  - Simulação Monte Carlo (10.000 portfólios)
  - Fronteira Eficiente (Markowitz)
  - Portfólio de Máximo Sharpe (tangência) com Ledoit-Wolf
  - Portfólio de Mínima Variância
  - Sugestão de realocação de pesos vs portfólio atual

Referências:
  - Markowitz (1952) — Portfolio Selection
  - Ledoit & Wolf (2004) — Honey, I Shrunk the Sample Covariance Matrix
  - Black & Litterman (1992) — Global Portfolio Optimization
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────
# DADOS HISTÓRICOS
# ─────────────────────────────────────────────────────────────────

def baixar_retornos(tickers: list[str], periodo: str = "3y") -> pd.DataFrame:
    """
    Baixa preços históricos e calcula retornos diários logarítmicos.
    Usa 3 anos por padrão — equilíbrio entre estabilidade e relevância.
    """
    print(f"  → MPT: baixando histórico de {len(tickers)} ativos...")
    try:
        dados = yf.download(
            tickers,
            period=periodo,
            auto_adjust=True,
            progress=False,
        )["Close"]

        # Se só 1 ativo, yfinance retorna Series — converte para DataFrame
        if isinstance(dados, pd.Series):
            dados = dados.to_frame(name=tickers[0])

        # Remove colunas com muitos NaN
        dados = dados.dropna(axis=1, thresh=int(len(dados) * 0.8))
        dados = dados.ffill().dropna()

        retornos = np.log(dados / dados.shift(1)).dropna()
        return retornos

    except Exception as e:
        print(f"  [MPT] Erro ao baixar dados: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# LEDOIT-WOLF SHRINKAGE
# ─────────────────────────────────────────────────────────────────

def covariancia_ledoit_wolf(retornos: pd.DataFrame) -> np.ndarray:
    """
    Estima matriz de covariância usando Ledoit-Wolf shrinkage.
    Mais estável que a covariância amostral quando n_ativos > n_obs/10.

    Fórmula: Σ_LW = (1-α)·Σ_sample + α·μ·I
    onde α é calculado analiticamente para minimizar erro quadrático médio.
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(retornos)
        return lw.covariance_
    except ImportError:
        # Fallback: covariância amostral com regularização simples
        S = retornos.cov().values
        n = S.shape[0]
        mu_target = np.trace(S) / n  # média dos eigenvalores
        alpha = 0.1                   # shrinkage fixo conservador
        return (1 - alpha) * S + alpha * mu_target * np.eye(n)


# ─────────────────────────────────────────────────────────────────
# MÉTRICAS DE PORTFÓLIO
# ─────────────────────────────────────────────────────────────────

def metricas_portfolio(
    pesos: np.ndarray,
    retornos_medios: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = 0.04,
    trading_days: int = 252,
) -> tuple[float, float, float]:
    """
    Retorna (retorno_anual, volatilidade_anual, sharpe_ratio).
    rf = taxa livre de risco anual (default 4% = treasuries 2024)
    """
    ret = float(np.dot(pesos, retornos_medios) * trading_days)
    vol = float(np.sqrt(np.dot(pesos.T, np.dot(cov_matrix * trading_days, pesos))))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


# ─────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────

@dataclass
class ResultadoMonteCarlo:
    retornos: list[float]
    volatilidades: list[float]
    sharpes: list[float]
    pesos_todos: list[list[float]]
    tickers: list[str]
    # Portfólios notáveis
    idx_max_sharpe: int = 0
    idx_min_vol: int = 0
    # Portfólio atual do usuário
    retorno_atual: float = 0.0
    vol_atual: float = 0.0
    sharpe_atual: float = 0.0


def simulacao_monte_carlo(
    retornos: pd.DataFrame,
    cov_matrix: np.ndarray,
    n_simulacoes: int = 10000,
    rf: float = 0.04,
    pesos_atuais: np.ndarray = None,
) -> ResultadoMonteCarlo:
    """
    Simula n_simulacoes portfólios com pesos aleatórios.
    Cada portfólio tem pesos que somam 1 (sem short selling).
    """
    tickers = list(retornos.columns)
    n_ativos = len(tickers)
    ret_medios = retornos.mean().values

    rets, vols, sharpes, todos_pesos = [], [], [], []

    np.random.seed(42)  # reprodutibilidade
    for _ in range(n_simulacoes):
        # Pesos aleatórios com distribuição Dirichlet (mais uniforme que Uniform)
        pesos = np.random.dirichlet(np.ones(n_ativos))
        ret, vol, sharpe = metricas_portfolio(pesos, ret_medios, cov_matrix, rf)
        rets.append(round(ret, 6))
        vols.append(round(vol, 6))
        sharpes.append(round(sharpe, 6))
        todos_pesos.append(pesos.tolist())

    resultado = ResultadoMonteCarlo(
        retornos=rets,
        volatilidades=vols,
        sharpes=sharpes,
        pesos_todos=todos_pesos,
        tickers=tickers,
        idx_max_sharpe=int(np.argmax(sharpes)),
        idx_min_vol=int(np.argmin(vols)),
    )

    # Métricas do portfólio atual (se fornecido)
    if pesos_atuais is not None:
        try:
            r, v, s = metricas_portfolio(pesos_atuais, ret_medios, cov_matrix, rf)
            resultado.retorno_atual = round(r, 6)
            resultado.vol_atual = round(v, 6)
            resultado.sharpe_atual = round(s, 6)
        except Exception:
            pass

    return resultado


# ─────────────────────────────────────────────────────────────────
# OTIMIZAÇÃO — FRONTEIRA EFICIENTE
# ─────────────────────────────────────────────────────────────────

@dataclass
class PortfolioOtimo:
    tipo: str           # "max_sharpe" | "min_variancia" | "risk_parity"
    tickers: list[str]
    pesos: list[float]
    retorno: float
    volatilidade: float
    sharpe: float
    pesos_atuais: list[float] = field(default_factory=list)
    delta_pesos: list[float] = field(default_factory=list)   # pesos_otimo - pesos_atuais


def _portfolio_max_sharpe(
    ret_medios: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float = 0.04,
    n_ativos: int = None,
) -> np.ndarray:
    """Maximiza Sharpe Ratio via scipy.optimize (SLSQP)."""
    n = n_ativos or len(ret_medios)
    pesos_init = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.02, 0.60)] * n  # mínimo 2%, máximo 60% por ativo

    def neg_sharpe(w):
        r, v, s = metricas_portfolio(w, ret_medios, cov_matrix, rf)
        return -s

    resultado = minimize(
        neg_sharpe, pesos_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    return resultado.x if resultado.success else pesos_init


def _portfolio_min_variancia(
    cov_matrix: np.ndarray,
    n_ativos: int,
) -> np.ndarray:
    """Minimiza variância do portfólio."""
    pesos_init = np.ones(n_ativos) / n_ativos
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.02, 0.60)] * n_ativos

    def variancia(w):
        return float(np.dot(w.T, np.dot(cov_matrix, w)))

    resultado = minimize(
        variancia, pesos_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )
    return resultado.x if resultado.success else pesos_init


def _portfolio_risk_parity(
    cov_matrix: np.ndarray,
    n_ativos: int,
) -> np.ndarray:
    """
    Risk Parity: cada ativo contribui igualmente para o risco total.
    Pesos inversamente proporcionais à volatilidade individual.
    Aproximação simples (1/vol normalizado).
    """
    vols = np.sqrt(np.diag(cov_matrix))
    pesos = 1 / (vols + 1e-8)
    return pesos / pesos.sum()


def _fronteira_eficiente_pontos(
    ret_medios: np.ndarray,
    cov_matrix: np.ndarray,
    n_pontos: int = 50,
    rf: float = 0.04,
) -> tuple[list, list]:
    """
    Traça a fronteira eficiente calculando portfólio de mínima variância
    para diferentes alvos de retorno.
    """
    n = len(ret_medios)
    ret_min = float(ret_medios.min()) * 252
    ret_max = float(ret_medios.max()) * 252
    alvos = np.linspace(ret_min, ret_max, n_pontos)

    vols_front, rets_front = [], []

    for alvo in alvos:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, a=alvo: np.dot(w, ret_medios) * 252 - a},
        ]
        bounds = [(0.0, 1.0)] * n

        def variancia(w):
            return float(np.dot(w.T, np.dot(cov_matrix * 252, w)))

        resultado = minimize(
            variancia,
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )
        if resultado.success:
            vol = float(np.sqrt(resultado.fun))
            vols_front.append(round(vol, 6))
            rets_front.append(round(alvo, 6))

    return vols_front, rets_front


def calcular_portfolios_otimos(
    retornos: pd.DataFrame,
    pesos_atuais_dict: dict,
    rf: float = 0.04,
) -> dict:
    """
    Calcula os 3 portfólios ótimos + fronteira eficiente.

    Args:
        retornos: DataFrame de retornos diários
        pesos_atuais_dict: {ticker: peso_atual} do portfólio real
        rf: taxa livre de risco anual

    Returns:
        dict com max_sharpe, min_variancia, risk_parity, fronteira
    """
    tickers = list(retornos.columns)
    n = len(tickers)
    ret_medios = retornos.mean().values
    cov_matrix = covariancia_ledoit_wolf(retornos)

    # Pesos atuais alinhados com os tickers disponíveis
    pesos_atuais = np.array([
        pesos_atuais_dict.get(t, 1/n) for t in tickers
    ])
    pesos_atuais = pesos_atuais / pesos_atuais.sum()  # normaliza

    print("  → MPT: otimizando portfólios...")

    # ── Portfólio Max Sharpe ──────────────────────────────────────
    w_ms = _portfolio_max_sharpe(ret_medios, cov_matrix, rf, n)
    r_ms, v_ms, s_ms = metricas_portfolio(w_ms, ret_medios, cov_matrix, rf)
    max_sharpe = PortfolioOtimo(
        tipo="max_sharpe",
        tickers=tickers,
        pesos=[round(float(w), 4) for w in w_ms],
        retorno=round(r_ms, 4),
        volatilidade=round(v_ms, 4),
        sharpe=round(s_ms, 4),
        pesos_atuais=[round(float(p), 4) for p in pesos_atuais],
        delta_pesos=[round(float(w-p), 4) for w, p in zip(w_ms, pesos_atuais)],
    )

    # ── Portfólio Mínima Variância ────────────────────────────────
    w_mv = _portfolio_min_variancia(cov_matrix * 252, n)
    r_mv, v_mv, s_mv = metricas_portfolio(w_mv, ret_medios, cov_matrix, rf)
    min_variancia = PortfolioOtimo(
        tipo="min_variancia",
        tickers=tickers,
        pesos=[round(float(w), 4) for w in w_mv],
        retorno=round(r_mv, 4),
        volatilidade=round(v_mv, 4),
        sharpe=round(s_mv, 4),
        pesos_atuais=[round(float(p), 4) for p in pesos_atuais],
        delta_pesos=[round(float(w-p), 4) for w, p in zip(w_mv, pesos_atuais)],
    )

    # ── Risk Parity ───────────────────────────────────────────────
    w_rp = _portfolio_risk_parity(cov_matrix, n)
    r_rp, v_rp, s_rp = metricas_portfolio(w_rp, ret_medios, cov_matrix, rf)
    risk_parity = PortfolioOtimo(
        tipo="risk_parity",
        tickers=tickers,
        pesos=[round(float(w), 4) for w in w_rp],
        retorno=round(r_rp, 4),
        volatilidade=round(v_rp, 4),
        sharpe=round(s_rp, 4),
        pesos_atuais=[round(float(p), 4) for p in pesos_atuais],
        delta_pesos=[round(float(w-p), 4) for w, p in zip(w_rp, pesos_atuais)],
    )

    # ── Fronteira Eficiente ───────────────────────────────────────
    print("  → MPT: calculando fronteira eficiente...")
    vols_front, rets_front = _fronteira_eficiente_pontos(ret_medios, cov_matrix, rf=rf)

    # ── Monte Carlo ───────────────────────────────────────────────
    print("  → MPT: simulação Monte Carlo (10.000 portfólios)...")
    monte_carlo = simulacao_monte_carlo(
        retornos, cov_matrix, n_simulacoes=10000, rf=rf,
        pesos_atuais=pesos_atuais,
    )

    # ── Matriz de correlação ──────────────────────────────────────
    corr_matrix = retornos.corr()

    return {
        "max_sharpe": max_sharpe,
        "min_variancia": min_variancia,
        "risk_parity": risk_parity,
        "fronteira_vols": vols_front,
        "fronteira_rets": rets_front,
        "monte_carlo": monte_carlo,
        "corr_matrix": corr_matrix,
        "tickers": tickers,
        "rf": rf,
    }


# ─────────────────────────────────────────────────────────────────
# TEXTO PARA O CLAUDE
# ─────────────────────────────────────────────────────────────────

def formatar_mpt_para_prompt(mpt_resultado: dict) -> str:
    """Formata insights MPT para incluir no prompt do Claude."""
    ms = mpt_resultado["max_sharpe"]
    mv = mpt_resultado["min_variancia"]
    mc = mpt_resultado["monte_carlo"]

    linhas = ["=== MODERN PORTFOLIO THEORY (MPT) ==="]

    # Portfólio atual vs ótimo
    linhas.append(f"\nPortfólio atual:")
    linhas.append(
        f"  Retorno: {mc.retorno_atual*100:.1f}% | "
        f"Volatilidade: {mc.vol_atual*100:.1f}% | "
        f"Sharpe: {mc.sharpe_atual:.2f}"
    )

    linhas.append(f"\nPortfólio ótimo (Máx. Sharpe — Tangência):")
    linhas.append(
        f"  Retorno: {ms.retorno*100:.1f}% | "
        f"Volatilidade: {ms.volatilidade*100:.1f}% | "
        f"Sharpe: {ms.sharpe:.2f}"
    )

    # Realocação sugerida
    linhas.append("\nRealocação sugerida (portfólio atual → ótimo):")
    for t, p_atual, p_otimo, delta in zip(
        ms.tickers, ms.pesos_atuais, ms.pesos, ms.delta_pesos
    ):
        sinal = "▲" if delta > 0.02 else "▼" if delta < -0.02 else "─"
        linhas.append(
            f"  {sinal} {t}: {p_atual*100:.1f}% → {p_otimo*100:.1f}% "
            f"({delta*100:+.1f}%)"
        )

    # Correlações altas (risco de concentração)
    corr = mpt_resultado["corr_matrix"]
    pares_correlacionados = []
    tickers = mpt_resultado["tickers"]
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            c = float(corr.iloc[i, j])
            if abs(c) > 0.65:
                pares_correlacionados.append((tickers[i], tickers[j], c))

    if pares_correlacionados:
        linhas.append("\nPares altamente correlacionados (risco de concentração):")
        for t1, t2, c in sorted(pares_correlacionados, key=lambda x: abs(x[2]), reverse=True):
            linhas.append(f"  {t1} ↔ {t2}: {c:.2f}")
    else:
        linhas.append("\nDiversificação: nenhum par com correlação > 0.65 ✓")

    return "\n".join(linhas)


# ─────────────────────────────────────────────────────────────────
# DISPLAY TERMINAL
# ─────────────────────────────────────────────────────────────────

def exibir_mpt_terminal(mpt_resultado: dict):
    """Exibe resumo MPT no terminal com Rich."""
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule
    from rich import box

    console = Console()
    console.print(Rule("[bold cyan]Modern Portfolio Theory — Otimização[/bold cyan]", style="cyan"))
    console.print()

    ms = mpt_resultado["max_sharpe"]
    mv = mpt_resultado["min_variancia"]
    rp = mpt_resultado["risk_parity"]
    mc = mpt_resultado["monte_carlo"]

    # ── Comparativo de portfólios ─────────────────────────────────
    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    table.add_column("Portfólio", style="bold white")
    table.add_column("Retorno/ano", justify="right")
    table.add_column("Volatilidade", justify="right")
    table.add_column("Sharpe", justify="right")

    def cor(v, limiar=0): return "green" if v >= limiar else "red"

    table.add_row(
        "📍 Atual",
        f"[{cor(mc.retorno_atual)}]{mc.retorno_atual*100:+.1f}%[/{cor(mc.retorno_atual)}]",
        f"{mc.vol_atual*100:.1f}%",
        f"[{cor(mc.sharpe_atual, 0.5)}]{mc.sharpe_atual:.2f}[/{cor(mc.sharpe_atual, 0.5)}]",
    )
    table.add_row(
        "⭐ Máx. Sharpe",
        f"[green]{ms.retorno*100:+.1f}%[/green]",
        f"{ms.volatilidade*100:.1f}%",
        f"[bold green]{ms.sharpe:.2f}[/bold green]",
    )
    table.add_row(
        "🛡️  Mín. Variância",
        f"[{cor(mv.retorno)}]{mv.retorno*100:+.1f}%[/{cor(mv.retorno)}]",
        f"[green]{mv.volatilidade*100:.1f}%[/green]",
        f"{mv.sharpe:.2f}",
    )
    table.add_row(
        "⚖️  Risk Parity",
        f"[{cor(rp.retorno)}]{rp.retorno*100:+.1f}%[/{cor(rp.retorno)}]",
        f"{rp.volatilidade*100:.1f}%",
        f"{rp.sharpe:.2f}",
    )
    console.print(table)
    console.print()

    # ── Realocação sugerida ───────────────────────────────────────
    console.print("  [bold]Realocação sugerida → Máx. Sharpe:[/bold]")
    for t, p_atual, p_otimo, delta in zip(
        ms.tickers, ms.pesos_atuais, ms.pesos, ms.delta_pesos
    ):
        if abs(delta) < 0.01:
            continue
        sinal = "▲" if delta > 0 else "▼"
        cor_s = "green" if delta > 0 else "red"
        console.print(
            f"    [{cor_s}]{sinal}[/{cor_s}] {t}: "
            f"{p_atual*100:.1f}% → {p_otimo*100:.1f}% "
            f"[{cor_s}]({delta*100:+.1f}%)[/{cor_s}]"
        )
    console.print()

    # ── Correlações ───────────────────────────────────────────────
    corr = mpt_resultado["corr_matrix"]
    tickers = mpt_resultado["tickers"]
    alertas = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            c = float(corr.iloc[i, j])
            if abs(c) > 0.65:
                alertas.append((tickers[i], tickers[j], c))

    if alertas:
        console.print("  [bold yellow]⚠️  Correlações elevadas (risco de concentração):[/bold yellow]")
        for t1, t2, c in sorted(alertas, key=lambda x: abs(x[2]), reverse=True):
            cor_c = "bold red" if abs(c) > 0.85 else "yellow"
            console.print(f"    [{cor_c}]{t1} ↔ {t2}: {c:.2f}[/{cor_c}]")
        console.print()
    else:
        console.print("  [green]✓ Diversificação saudável — nenhum par com correlação > 0.65[/green]\n")
