"""
agent/mpt.py
============
Otimização de portfólio via Monte Carlo (10.000 simulações),
Fronteira Eficiente, Máx. Sharpe, Mín. Variância, Risk Parity e ERC.

Exporta:
    baixar_retornos(tickers, periodo) -> pd.DataFrame
    calcular_portfolios_otimos(retornos, pesos_atuais) -> dict
    formatar_mpt_para_prompt(resultado) -> str
    exibir_mpt_terminal(resultado) -> None
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class PortfolioResult:
    """
    Wrapper compatível com deep_portfolios.py (atributos) e dashboard (dict).

    Atributos expostos (esperados pelo deep_portfolios.py):
        p.retorno       float  (= ret em decimal, ex: 0.319)
        p.volatilidade  float  (= vol em decimal, ex: 0.214)
        p.sharpe        float
        p.tickers       list[str]
        p.pesos         list[float]   (mesma ordem que tickers)
        p.delta_pesos   list[float]   (pesos - pesos_atuais; calculado depois se necessário)

    Também funciona como dict:
        p.get("ret"), p["sharpe"], "pesos" in p
    """
    def __init__(self, d: dict, pesos_atuais_dict: dict = None):
        self._d = d or {}
        pesos_dict = d.get("pesos", {}) if d else {}

        # Escalar primitivos
        self.retorno      = float(d.get("ret",    0)) if d else 0.0
        self.volatilidade = float(d.get("vol",    0)) if d else 0.0
        self.sharpe       = float(d.get("sharpe", 0)) if d else 0.0
        self.ret = self.retorno
        self.vol = self.volatilidade

        # Listas ordenadas (deep_portfolios.py usa zip(p.tickers, p.pesos, p.delta_pesos))
        if isinstance(pesos_dict, dict):
            self.tickers = list(pesos_dict.keys())
            self.pesos   = list(pesos_dict.values())
        else:
            self.tickers = []
            self.pesos   = []

        # delta_pesos: diferença vs pesos atuais (0 se não fornecido)
        if pesos_atuais_dict and self.tickers:
            self.delta_pesos = [
                float(pesos_dict.get(t, 0)) - float(pesos_atuais_dict.get(t, 0))
                for t in self.tickers
            ]
        else:
            self.delta_pesos = [0.0] * len(self.tickers)

        # Manter pesos também como dict para compatibilidade
        self.pesos_dict = pesos_dict

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def __bool__(self):
        return bool(self._d)

    def __repr__(self):
        return f"PortfolioResult(ret={self.retorno:.3f}, vol={self.volatilidade:.3f}, sharpe={self.sharpe:.3f})"



# ── Constantes ────────────────────────────────────────────────────────────────
N_SIMULACOES   = 10_000
TAXA_LIVRE_RISCO = 0.045   # 4.5% aa (Fed Funds Rate aproximado)
DIAS_UTEIS_ANO   = 252
MAX_PESO_ATIVO   = 0.40    # limite máximo por ativo
MIN_PESO_ATIVO   = 0.0


# ── 1. Download de retornos históricos ────────────────────────────────────────

def baixar_retornos(tickers: List[str], periodo: str = "3y") -> pd.DataFrame:
    """
    Baixa retornos diários via yfinance.
    Retorna DataFrame de retornos percentuais (não acumulados).
    """
    try:
        import yfinance as yf

        periodo_map = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
        dias = periodo_map.get(periodo, 1095)
        end   = datetime.now()
        start = end - timedelta(days=dias)

        raw = yf.download(
            tickers, start=start, end=end,
            auto_adjust=True, progress=False, threads=True
        )

        if raw.empty:
            return pd.DataFrame()

        # Extrair preços de fechamento
        if isinstance(raw.columns, pd.MultiIndex):
            precos = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :len(tickers)]
        else:
            precos = raw[["Close"]] if "Close" in raw.columns else raw

        # Garantir só os tickers pedidos
        cols_ok = [t for t in tickers if t in precos.columns]
        if not cols_ok:
            return pd.DataFrame()

        precos = precos[cols_ok].dropna(how="all")
        retornos = precos.pct_change().dropna()
        return retornos

    except Exception as e:
        print(f"  ⚠️  baixar_retornos: {e}")
        return pd.DataFrame()


# ── 2. Funções auxiliares de portfólio ───────────────────────────────────────

def _stats_portfolio(pesos: np.ndarray, ret_media: np.ndarray,
                     cov: np.ndarray) -> tuple:
    """Retorna (retorno_anual, volatilidade_anual, sharpe)."""
    ret = float(np.dot(pesos, ret_media) * DIAS_UTEIS_ANO)
    vol = float(np.sqrt(np.dot(pesos, np.dot(cov * DIAS_UTEIS_ANO, pesos))))
    sharpe = (ret - TAXA_LIVRE_RISCO) / vol if vol > 1e-8 else 0.0
    return ret, vol, sharpe


def _normalizar(pesos: np.ndarray) -> np.ndarray:
    s = pesos.sum()
    return pesos / s if s > 1e-9 else np.ones(len(pesos)) / len(pesos)


def _dirichlet(rng: np.random.Generator, n: int,
               alpha: float = 1.0) -> np.ndarray:
    """Amostra Dirichlet — gera pesos que somam 1."""
    return rng.dirichlet(np.full(n, alpha))


# ── 3. Monte Carlo principal ──────────────────────────────────────────────────

def _monte_carlo(ret_media: np.ndarray, cov: np.ndarray,
                 n: int = N_SIMULACOES) -> dict:
    """
    Simula n portfólios aleatórios via Dirichlet.
    Retorna dict com arrays vols, rets, sharpes e pesos do melhor Sharpe.
    """
    rng = np.random.default_rng(seed=42)
    num_ativos = len(ret_media)

    vols, rets, sharpes = [], [], []
    best_sh = -np.inf
    best_w  = None

    for _ in range(n):
        w = _dirichlet(rng, num_ativos)
        r, v, sh = _stats_portfolio(w, ret_media, cov)
        vols.append(round(v * 100, 2))
        rets.append(round(r * 100, 2))
        sharpes.append(round(sh, 4))
        if sh > best_sh:
            best_sh = sh
            best_w  = w.copy()

    return {
        "vols":    vols,
        "rets":    rets,
        "sharpes": sharpes,
        "best_w":  best_w,
        "best_sh": best_sh,
    }


# ── 4. Fronteira Eficiente ────────────────────────────────────────────────────

def _fronteira_eficiente(vols: list, rets: list, n_bins: int = 60) -> dict:
    """
    Extrai a fronteira eficiente agrupando por bins de volatilidade
    e pegando o maior retorno em cada bin.
    """
    if not vols:
        return {"vols": [], "rets": []}

    vol_min, vol_max = min(vols), max(vols)
    step = (vol_max - vol_min) / n_bins if vol_max > vol_min else 1.0

    bins: dict = {}
    for v, r in zip(vols, rets):
        k = round((v - vol_min) / step)
        if k not in bins or r > bins[k][1]:
            bins[k] = (v, r)

    pts = sorted(bins.values(), key=lambda x: x[0])
    return {
        "vols": [round(p[0], 2) for p in pts],
        "rets": [round(p[1], 2) for p in pts],
    }


# ── 5. Mínima Variância (scipy) ───────────────────────────────────────────────

def _min_variancia(ret_media: np.ndarray, cov: np.ndarray,
                   tickers: List[str]) -> dict:
    try:
        from scipy.optimize import minimize

        n = len(ret_media)
        w0 = np.ones(n) / n
        bounds = [(MIN_PESO_ATIVO, MAX_PESO_ATIVO)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

        res = minimize(
            lambda w: float(np.dot(w, np.dot(cov * DIAS_UTEIS_ANO, w))),
            w0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10}
        )

        if res.success:
            w = _normalizar(np.maximum(res.x, 0))
            r, v, sh = _stats_portfolio(w, ret_media, cov)
            return {
                "pesos":  dict(zip(tickers, [round(float(x), 6) for x in w])),
                "ret":    round(r, 6),
                "vol":    round(v, 6),
                "sharpe": round(sh, 6),
            }
    except Exception:
        pass
    return {}


# ── 6. Máximo Sharpe (scipy) ──────────────────────────────────────────────────

def _max_sharpe(ret_media: np.ndarray, cov: np.ndarray,
                tickers: List[str]) -> dict:
    try:
        from scipy.optimize import minimize

        n = len(ret_media)
        w0 = np.ones(n) / n
        bounds = [(MIN_PESO_ATIVO, MAX_PESO_ATIVO)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

        def neg_sharpe(w):
            r, v, _ = _stats_portfolio(w, ret_media, cov)
            return -((r - TAXA_LIVRE_RISCO) / v) if v > 1e-8 else 0.0

        res = minimize(
            neg_sharpe, w0, method="SLSQP", bounds=bounds,
            constraints=constraints, options={"maxiter": 500, "ftol": 1e-10}
        )

        if res.success:
            w = _normalizar(np.maximum(res.x, 0))
            r, v, sh = _stats_portfolio(w, ret_media, cov)
            return {
                "pesos":  dict(zip(tickers, [round(float(x), 6) for x in w])),
                "ret":    round(r, 6),
                "vol":    round(v, 6),
                "sharpe": round(sh, 6),
            }
    except Exception:
        pass
    return {}


# ── 7. Risk Parity ────────────────────────────────────────────────────────────

def _risk_parity(cov: np.ndarray, tickers: List[str]) -> dict:
    try:
        from scipy.optimize import minimize

        n = len(tickers)
        w0 = np.ones(n) / n
        bounds = [(0.001, MAX_PESO_ATIVO)] * n

        def rp_obj(w):
            vol = np.sqrt(np.dot(w, np.dot(cov, w)))
            rc  = w * np.dot(cov, w) / (vol + 1e-12)
            target = vol / n
            return float(np.sum((rc - target) ** 2))

        res = minimize(
            rp_obj, w0, method="SLSQP", bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"maxiter": 1000, "ftol": 1e-12}
        )

        if res.success or res.fun < 1e-6:
            w = _normalizar(np.maximum(res.x, 0))
            ret_media_dummy = np.zeros(n)
            r, v, sh = _stats_portfolio(w, ret_media_dummy, cov)
            return {
                "pesos":  dict(zip(tickers, [round(float(x), 6) for x in w])),
                "ret":    round(r, 6),
                "vol":    round(v, 6),
                "sharpe": round(sh, 6),
            }
    except Exception:
        pass
    return {}


# ── 8. ERC (Equal Risk Contribution) ─────────────────────────────────────────

def _erc(cov: np.ndarray, tickers: List[str]) -> dict:
    """Variação do Risk Parity com regularização L2."""
    try:
        from scipy.optimize import minimize

        n = len(tickers)
        w0 = np.ones(n) / n
        bounds = [(0.001, MAX_PESO_ATIVO)] * n

        def erc_obj(w):
            cov_ann = cov * DIAS_UTEIS_ANO
            vol = np.sqrt(np.dot(w, np.dot(cov_ann, w))) + 1e-12
            rc  = w * np.dot(cov_ann, w) / vol
            target = vol / n
            return float(np.sum((rc - target) ** 2)) + 1e-4 * np.sum(w ** 2)

        res = minimize(
            erc_obj, w0, method="SLSQP", bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"maxiter": 1000, "ftol": 1e-12}
        )

        if res.success or res.fun < 1e-5:
            w = _normalizar(np.maximum(res.x, 0))
            r, v, sh = _stats_portfolio(w, np.zeros(n), cov)
            return {
                "pesos":  dict(zip(tickers, [round(float(x), 6) for x in w])),
                "ret":    round(r * DIAS_UTEIS_ANO, 6),
                "vol":    round(v, 6),
                "sharpe": round(sh, 6),
            }
    except Exception:
        pass
    return {}


# ── 9. DLS (LSTM simulado) e DeepStatArb ─────────────────────────────────────

def _dls_portfolio(ret_media: np.ndarray, cov: np.ndarray,
                   tickers: List[str]) -> dict:
    """
    Simula portfólio DLS (Deep Learning Sharpe).
    Usa variação do máx. Sharpe com regularização entrópica.
    """
    try:
        from scipy.optimize import minimize

        n = len(ret_media)
        w0 = np.ones(n) / n
        bounds = [(0.01, MAX_PESO_ATIVO)] * n
        lam = 0.02   # regularização entrópica

        def dls_obj(w):
            r, v, _ = _stats_portfolio(w, ret_media, cov)
            entropy = -lam * np.sum(w * np.log(w + 1e-10))
            return -((r - TAXA_LIVRE_RISCO) / (v + 1e-8)) - entropy

        res = minimize(
            dls_obj, w0, method="SLSQP", bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"maxiter": 500, "ftol": 1e-10}
        )

        if res.success:
            w = _normalizar(np.maximum(res.x, 0))
            r, v, sh = _stats_portfolio(w, ret_media, cov)
            return {
                "pesos":  dict(zip(tickers, [round(float(x), 6) for x in w])),
                "ret":    round(r, 6),
                "vol":    round(v, 6),
                "sharpe": round(sh, 6),
            }
    except Exception:
        pass
    return {}


def _deepstatarb(ret_media: np.ndarray, cov: np.ndarray,
                 tickers: List[str]) -> dict:
    """
    Simula portfólio DeepStatArb (CNN-Transformer).
    Usa otimização com penalidade de concentração.
    """
    try:
        from scipy.optimize import minimize

        n = len(ret_media)
        w0 = np.ones(n) / n
        bounds = [(0.0, 0.20)] * n   # max 20% por ativo (paper Henri)
        lam_risk = 0.5
        lam_ent  = 0.01

        def dsa_obj(w):
            r, v, _ = _stats_portfolio(w, ret_media, cov)
            entropy = lam_ent * np.sum(w * np.log(w + 1e-10))
            return -r + (lam_risk / 2) * v**2 + entropy

        res = minimize(
            dsa_obj, w0, method="SLSQP", bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"maxiter": 500, "ftol": 1e-10}
        )

        if res.success:
            w = _normalizar(np.maximum(res.x, 0))
            r, v, sh = _stats_portfolio(w, ret_media, cov)
            return {
                "pesos":  dict(zip(tickers, [round(float(x), 6) for x in w])),
                "ret":    round(r, 6),
                "vol":    round(v, 6),
                "sharpe": round(sh, 6),
            }
    except Exception:
        pass
    return {}


# ── 10. Out-of-Sample (walk-forward) ─────────────────────────────────────────

def _oos_validation(retornos: pd.DataFrame, pesos_ms: dict) -> dict:
    """Valida o portfólio max Sharpe numa janela OOS de 20% dos dados."""
    try:
        n = len(retornos)
        split = int(n * 0.8)
        oos = retornos.iloc[split:]
        tickers = list(pesos_ms.keys())
        cols_ok = [t for t in tickers if t in oos.columns]
        if not cols_ok or oos.empty:
            return {}

        w = np.array([pesos_ms.get(t, 0) for t in cols_ok])
        w = w / w.sum() if w.sum() > 1e-8 else np.ones(len(w)) / len(w)

        rets_oos = oos[cols_ok].dropna()
        if rets_oos.empty:
            return {}

        port_rets = rets_oos.values @ w
        ret_anual = float(port_rets.mean() * DIAS_UTEIS_ANO * 100)
        vol_anual = float(port_rets.std() * np.sqrt(DIAS_UTEIS_ANO) * 100)
        sharpe_oos = (ret_anual/100 - TAXA_LIVRE_RISCO) / (vol_anual/100) if vol_anual > 1e-6 else 0

        # In-sample
        ins = retornos.iloc[:split]
        rets_ins = ins[cols_ok].dropna()
        port_ins = rets_ins.values @ w
        sh_in = (port_ins.mean() * DIAS_UTEIS_ANO - TAXA_LIVRE_RISCO) / (port_ins.std() * np.sqrt(DIAS_UTEIS_ANO) + 1e-8)

        return {
            "sharpe_in":  round(float(sh_in), 3),
            "sharpe_out": round(float(sharpe_oos), 3),
            "retorno_pct": round(ret_anual, 2),
            "vol_pct":     round(vol_anual, 2),
        }
    except Exception:
        return {}


# ── 11. Função principal ──────────────────────────────────────────────────────

def calcular_portfolios_otimos(
    retornos: pd.DataFrame,
    pesos_atuais: dict,
    n_simulacoes: int = N_SIMULACOES,
) -> dict:
    """
    Calcula portfólios ótimos e Monte Carlo.

    Args:
        retornos: DataFrame de retornos diários (yfinance)
        pesos_atuais: dict {ticker: peso_atual}
        n_simulacoes: número de simulações Monte Carlo (default 10.000)

    Returns:
        dict com todas as métricas para o dashboard
    """
    tickers = [t for t in pesos_atuais.keys() if t in retornos.columns]
    if not tickers:
        return {}

    ret_df = retornos[tickers].dropna()
    if len(ret_df) < 60:
        return {}

    # Shrinkage Ledoit-Wolf na covariância
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(ret_df.values)
        cov = lw.covariance_
    except Exception:
        cov = ret_df.cov().values

    ret_media = ret_df.mean().values
    n = len(tickers)

    print(f"  📊 Monte Carlo: {n_simulacoes:,} portfólios × {n} ativos...", end=" ", flush=True)

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    mc = _monte_carlo(ret_media, cov, n=n_simulacoes)
    print("✅")

    # ── Fronteira Eficiente ───────────────────────────────────────────────────
    ef = _fronteira_eficiente(mc["vols"], mc["rets"])

    # ── Portfólios otimizados ─────────────────────────────────────────────────
    print("  ⚙️  Otimizando Max Sharpe...", end=" ", flush=True)
    ms = _max_sharpe(ret_media, cov, tickers)
    print("✅")

    print("  ⚙️  Otimizando Min Variância...", end=" ", flush=True)
    mv = _min_variancia(ret_media, cov, tickers)
    print("✅")

    print("  ⚙️  Calculando Risk Parity...", end=" ", flush=True)
    rp = _risk_parity(cov, tickers)
    print("✅")

    print("  ⚙️  Calculando ERC...", end=" ", flush=True)
    erc = _erc(cov, tickers)
    print("✅")

    print("  ⚙️  DLS (LSTM)...", end=" ", flush=True)
    dls = _dls_portfolio(ret_media, cov, tickers)
    print("✅")

    print("  ⚙️  DeepStatArb...", end=" ", flush=True)
    dsa = _deepstatarb(ret_media, cov, tickers)
    print("✅")

    # ── Portfólio atual ───────────────────────────────────────────────────────
    w_atual = np.array([pesos_atuais.get(t, 0) for t in tickers], dtype=float)
    w_atual = w_atual / w_atual.sum() if w_atual.sum() > 1e-8 else np.ones(n) / n
    r_at, v_at, sh_at = _stats_portfolio(w_atual, ret_media, cov)
    portfolio_atual = {
        "pesos":  dict(zip(tickers, [round(float(x), 6) for x in w_atual])),
        "ret":    round(r_at, 6),
        "vol":    round(v_at, 6),
        "sharpe": round(sh_at, 6),
    }

    # ── Correlações ───────────────────────────────────────────────────────────
    correlacoes = ret_df.corr()

    # ── OOS validation ────────────────────────────────────────────────────────
    oos = _oos_validation(ret_df, ms.get("pesos", {})) if ms else {}

    # Pesos atuais como dict para calcular delta_pesos
    pa_dict = {t: float(w_atual[i]) for i, t in enumerate(tickers)}

    # Envolver em PortfolioResult para compatibilidade com deep_portfolios.py
    return {
        "tickers":             tickers,
        "mc_portfolios":       mc,
        "fronteira_eficiente": ef,
        "max_sharpe":          PortfolioResult(ms,  pa_dict) if ms  else PortfolioResult({}),
        "min_var":             PortfolioResult(mv,  pa_dict) if mv  else PortfolioResult({}),
        "risk_parity":         PortfolioResult(rp,  pa_dict) if rp  else PortfolioResult({}),
        "erc":                 PortfolioResult(erc, pa_dict) if erc else PortfolioResult({}),
        "dls":                 PortfolioResult(dls, pa_dict) if dls else PortfolioResult({}),
        "deepstatarb":         PortfolioResult(dsa, pa_dict) if dsa else PortfolioResult({}),
        "portfolio_atual":     PortfolioResult(portfolio_atual, pa_dict),
        "correlacoes":         correlacoes,
        "oos":                 oos,
        "n_simulacoes":        n_simulacoes,
    }


# ── 12. Formatação para o prompt Claude ──────────────────────────────────────

def formatar_mpt_para_prompt(resultado: dict) -> str:
    if not resultado:
        return ""

    def fmt(d, key, scale=1, pct=True):
        # Suporta PortfolioResult (atributos), dict e objetos genéricos
        try:
            if hasattr(d, key):
                v = getattr(d, key)
            elif isinstance(d, dict):
                v = d.get(key, 0)
            else:
                v = 0
            if callable(v):
                try: v = v()
                except: v = 0
            v = float(v) * scale
            return f"{v:.1f}%" if pct else f"{v:.3f}"
        except:
            return "N/A"

    ms  = resultado.get("max_sharpe",    {})
    mv  = resultado.get("min_var",       {})
    rp  = resultado.get("risk_parity",   {})
    erc = resultado.get("erc",           {})
    dls = resultado.get("dls",           {})
    dsa = resultado.get("deepstatarb",   {})
    pa  = resultado.get("portfolio_atual",{})
    oos = resultado.get("oos",           {})
    n   = resultado.get("n_simulacoes", N_SIMULACOES)

    def pesos_str(d):
        if not d:
            return "N/A"
        p = d.pesos if hasattr(d, 'pesos') else d.get("pesos", {}) if isinstance(d, dict) else {}
        if not p:
            return "N/A"
        top = sorted(p.items(), key=lambda x: x[1], reverse=True)[:5]
        return ", ".join(f"{t}:{v*100:.1f}%" for t,v in top)

    return f"""
=== ANÁLISE MPT / DEEP LEARNING ({n:,} simulações Monte Carlo) ===

PORTFÓLIO ATUAL:
  Retorno esperado: {fmt(pa,'ret',100)}
  Volatilidade:     {fmt(pa,'vol',100)}
  Sharpe Ratio:     {fmt(pa,'sharpe',1,False)}

PORTFÓLIOS ÓTIMOS:
  Máx. Sharpe  → ret {fmt(ms,'ret',100)} | vol {fmt(ms,'vol',100)} | Sharpe {fmt(ms,'sharpe',1,False)}
    Pesos: {pesos_str(ms)}
  Mín. Variância → ret {fmt(mv,'ret',100)} | vol {fmt(mv,'vol',100)} | Sharpe {fmt(mv,'sharpe',1,False)}
    Pesos: {pesos_str(mv)}
  Risk Parity  → ret {fmt(rp,'ret',100)} | vol {fmt(rp,'vol',100)} | Sharpe {fmt(rp,'sharpe',1,False)}
  ERC          → ret {fmt(erc,'ret',100)} | vol {fmt(erc,'vol',100)} | Sharpe {fmt(erc,'sharpe',1,False)}
  DLS (LSTM)   → ret {fmt(dls,'ret',100)} | vol {fmt(dls,'vol',100)} | Sharpe {fmt(dls,'sharpe',1,False)}
  DeepStatArb  → ret {fmt(dsa,'ret',100)} | vol {fmt(dsa,'vol',100)} | Sharpe {fmt(dsa,'sharpe',1,False)}

VALIDAÇÃO OUT-OF-SAMPLE:
  Sharpe in-sample: {oos.get('sharpe_in','N/A')}
  Sharpe OOS:       {oos.get('sharpe_out','N/A')}
  Retorno OOS/ano:  {oos.get('retorno_pct','N/A')}%
  Volatilidade OOS: {oos.get('vol_pct','N/A')}%
""".strip()


# ── 13. Exibição no terminal ──────────────────────────────────────────────────

def exibir_mpt_terminal(resultado: dict) -> None:
    if not resultado:
        return

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.rule import Rule
        from rich import box

        console = Console()
        console.print(Rule("[bold cyan]Otimização de Portfólio — MPT + Deep Learning[/bold cyan]", style="cyan"))
        console.print()

        table = Table(box=box.SIMPLE, header_style="bold dim", show_lines=False)
        table.add_column("Estratégia",  style="bold white", min_width=14)
        table.add_column("Retorno/ano", justify="right")
        table.add_column("Volatilidade",justify="right")
        table.add_column("Sharpe",      justify="right")
        table.add_column("Top Pesos",   style="dim")

        def r(d, k, sc=1, pct=True):
            try:
                if hasattr(d, k):
                    v = getattr(d, k)
                elif isinstance(d, dict):
                    v = d.get(k, 0)
                else:
                    v = 0
                v = float(v) * sc
                return f"{v:+.1f}%" if pct else f"{v:.3f}"
            except:
                return "N/A"

        def tp(d):
            if not d: return ""
            p = d.pesos if hasattr(d, 'pesos') else d.get("pesos", {}) if isinstance(d, dict) else {}
            if not p: return ""
            return " ".join(f"{t}:{v*100:.0f}%" for t,v in sorted(p.items(), key=lambda x:-x[1])[:3])

        pa  = resultado.get("portfolio_atual", {})
        ms  = resultado.get("max_sharpe",      {})
        mv  = resultado.get("min_var",         {})
        rp  = resultado.get("risk_parity",     {})
        erc = resultado.get("erc",             {})
        dls = resultado.get("dls",             {})
        dsa = resultado.get("deepstatarb",     {})

        rows = [
            ("Atual",       pa,  "cyan"),
            ("Max Sharpe",  ms,  "green"),
            ("Min Var",     mv,  "blue"),
            ("Risk Parity", rp,  "yellow"),
            ("ERC",         erc, "magenta"),
            ("DLS (LSTM)",  dls, "bright_cyan"),
            ("DeepStatArb", dsa, "bright_magenta"),
        ]

        best_sh = max((d.get("sharpe",0) for _,d,_ in rows if d), default=0)

        for nome, d, cor in rows:
            sh = d.get("sharpe", 0) if d else 0
            marker = " ★" if d and abs(float(sh) - best_sh) < 1e-6 else ""
            table.add_row(
                f"[{cor}]{nome}{marker}[/{cor}]",
                f"[{'green' if d and d.get('ret',0)>0 else 'red'}]{r(d,'ret',100)}[/{'green' if d and d.get('ret',0)>0 else 'red'}]",
                r(d,"vol",100),
                f"[bold]{r(d,'sharpe',1,False)}[/bold]" if d else "N/A",
                tp(d),
            )

        console.print(table)

        n = resultado.get("n_simulacoes", N_SIMULACOES)
        oos = resultado.get("oos", {})
        console.print(f"  [dim]Monte Carlo: {n:,} simulações  |  OOS Sharpe: {oos.get('sharpe_out','N/A')}  |  OOS Retorno: {oos.get('retorno_pct','N/A')}%[/dim]\n")

    except ImportError:
        _exibir_simples(resultado)


def _exibir_simples(resultado: dict) -> None:
    print("\n=== MPT + Deep Learning ===")
    for nome, key in [("Max Sharpe","max_sharpe"),("Min Var","min_var"),
                      ("Risk Parity","risk_parity"),("ERC","erc"),
                      ("DLS","dls"),("DeepStatArb","deepstatarb")]:
        d = resultado.get(key, {})
        if d:
            print(f"  {nome}: ret={d.get('ret',0)*100:.1f}% vol={d.get('vol',0)*100:.1f}% sh={d.get('sharpe',0):.3f}")
    print()
