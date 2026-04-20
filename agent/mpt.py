"""
agent/mpt.py — versão definitiva
Estrutura de retorno compatível com agent/dashboard.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ── Constantes ─────────────────────────────────────────────────────────────────
N_SIMULACOES     = 10_000
TAXA_LIVRE_RISCO = 0.045
DIAS_UTEIS_ANO   = 252
MAX_PESO_ATIVO   = 0.40
MIN_PESO_ATIVO   = 0.0
RETORNO_MERCADO_LONGO_PRAZO = 0.09
RETORNO_ESPERADO_MIN = 0.02
RETORNO_ESPERADO_MAX = 0.18
ALPHA_ESPERADO_MAX = 0.03


# ── Classe MonteCarloResult ────────────────────────────────────────────────────
class MonteCarloResult:
    """
    Objeto retornado em mpt_resultado["monte_carlo"].
    Atributos exigidos pelo dashboard.py:
        .volatilidades   list[float]  valores em decimal (ex: 0.18 = 18%)
        .retornos        list[float]  valores em decimal
        .sharpes         list[float]
        .idx_max_sharpe  int
        .idx_min_vol     int
        .vol_atual       float        decimal
        .retorno_atual   float        decimal
        .sharpe_atual    float
    """
    def __init__(self, vols, rets, sharpes, vol_at, ret_at, sh_at):
        self.volatilidades  = vols
        self.retornos       = rets
        self.sharpes        = sharpes
        self.idx_max_sharpe = int(np.argmax(sharpes))
        self.idx_min_vol    = int(np.argmin(vols))
        self.vol_atual      = vol_at
        self.retorno_atual  = ret_at
        self.sharpe_atual   = sh_at

    # Para compatibilidade com código que usa mc["vols"] etc.
    def get(self, key, default=None):
        mapping = {
            "vols":    self.volatilidades,
            "rets":    self.retornos,
            "sharpes": self.sharpes,
            "volatilidades": self.volatilidades,
            "retornos":      self.retornos,
        }
        return mapping.get(key, default)

    def __getitem__(self, key):
        v = self.get(key)
        return v if v is not None else {}

    def __bool__(self):
        return bool(self.volatilidades)


# ── Classe PortfolioResult ─────────────────────────────────────────────────────
class PortfolioResult:
    """
    Wrapper para portfólios ótimos.
    Atributos exigidos pelo dashboard.py:
        .pesos          list[float]   pesos do portfólio ótimo (decimal)
        .pesos_atuais   list[float]   pesos do portfólio atual (mesma ordem)
        .retorno        float         decimal
        .volatilidade   float         decimal
        .sharpe         float
        .tickers        list[str]
        .delta_pesos    list[float]
    """
    def __init__(self, d: dict, pesos_atuais_dict: dict = None):
        self._d = d or {}
        pesos_dict = d.get("pesos", {}) if d else {}

        self.retorno      = float(d.get("ret",    0)) if d else 0.0
        self.volatilidade = float(d.get("vol",    0)) if d else 0.0
        self.sharpe       = float(d.get("sharpe", 0)) if d else 0.0
        self.ret = self.retorno
        self.vol = self.volatilidade

        if isinstance(pesos_dict, dict) and pesos_dict:
            self.tickers   = list(pesos_dict.keys())
            self.pesos     = list(pesos_dict.values())       # decimal
            self.pesos_dict = pesos_dict
        else:
            self.tickers   = []
            self.pesos     = []
            self.pesos_dict = {}

        # pesos_atuais na mesma ordem que tickers (exigido pelo dashboard.py)
        if pesos_atuais_dict and self.tickers:
            self.pesos_atuais = [float(pesos_atuais_dict.get(t, 0)) for t in self.tickers]
        else:
            self.pesos_atuais = [0.0] * len(self.tickers)

        self.delta_pesos = [
            p - pa for p, pa in zip(self.pesos, self.pesos_atuais)
        ]

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __getitem__(self, key):
        return self._d.get(key)

    def __contains__(self, key):
        return key in self._d

    def __bool__(self):
        return bool(self._d)

    def __repr__(self):
        return f"PortfolioResult(ret={self.retorno:.3f}, vol={self.volatilidade:.3f}, sharpe={self.sharpe:.3f})"


# ── ResultadoMPT — nunca levanta KeyError ──────────────────────────────────────
class ResultadoMPT(dict):
    """Dict que retorna None para qualquer chave desconhecida."""
    def __missing__(self, key):
        return None

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return None


# ── Download de retornos ───────────────────────────────────────────────────────
def baixar_retornos(tickers: List[str], periodo: str = "3y") -> pd.DataFrame:
    try:
        import yfinance as yf
        dias_map = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
        dias  = dias_map.get(periodo, 1095)
        end   = datetime.now()
        start = end - timedelta(days=dias)
        raw   = yf.download(tickers, start=start, end=end,
                            auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            precos = raw["Close"]
        else:
            precos = raw
        cols_ok = [t for t in tickers if t in precos.columns]
        if not cols_ok:
            return pd.DataFrame()
        return precos[cols_ok].dropna(how="all").pct_change().dropna()
    except Exception as e:
        print(f"  ⚠️  baixar_retornos: {e}")
        return pd.DataFrame()


# ── Funções de portfólio ───────────────────────────────────────────────────────
def _stats(w, mu, cov):
    r  = float(np.dot(w, mu) * DIAS_UTEIS_ANO)
    v  = float(np.sqrt(np.dot(w, np.dot(cov * DIAS_UTEIS_ANO, w))))
    sh = (r - TAXA_LIVRE_RISCO) / v if v > 1e-8 else 0.0
    return r, v, sh

def _norm(w):
    s = w.sum()
    return w / s if s > 1e-9 else np.ones(len(w)) / len(w)

def _dirichlet(rng, n, alpha=1.0):
    return rng.dirichlet(np.full(n, alpha))

def _retorno_geometrico_anual(retornos: pd.Series) -> float:
    retornos = pd.to_numeric(retornos, errors="coerce").dropna()
    if len(retornos) < 2:
        return RETORNO_MERCADO_LONGO_PRAZO
    total = float((1.0 + retornos).prod() - 1.0)
    anos = len(retornos) / DIAS_UTEIS_ANO
    if anos <= 0 or total <= -0.99:
        return RETORNO_MERCADO_LONGO_PRAZO
    return float((1.0 + total) ** (1.0 / anos) - 1.0)

def _estimativa_retornos_conservadora(retornos: pd.DataFrame, tickers: list[str]) -> tuple[np.ndarray, dict]:
    """
    Estima retornos esperados sem extrapolar vencedores recentes.

    O retorno esperado fica ancorado em uma premissa de mercado de longo prazo
    e recebe apenas uma parcela pequena do alpha historico recente. Isso evita
    que Max Sharpe/DLS virem uma aposta mecanica nos ativos que mais subiram.
    """
    candidatos_benchmark = [t for t in ["IVV", "SPY", "VOO", "VT", "ACWI"] if t in retornos.columns]
    if candidatos_benchmark:
        bench_col = candidatos_benchmark[0]
        benchmark_realizado = _retorno_geometrico_anual(retornos[bench_col])
    else:
        bench_col = "Equal Weight"
        benchmark_realizado = _retorno_geometrico_anual(retornos[tickers].mean(axis=1))

    benchmark_realizado = float(np.clip(benchmark_realizado, -0.05, 0.22))
    anchor = 0.80 * RETORNO_MERCADO_LONGO_PRAZO + 0.20 * benchmark_realizado

    mu_anuais = []
    hist_anuais = {}
    for ticker in tickers:
        hist = _retorno_geometrico_anual(retornos[ticker])
        hist_anuais[ticker] = hist
        alpha_hist = float(np.clip(hist - benchmark_realizado, -0.20, 0.20))
        alpha_esperado = float(np.clip(0.15 * alpha_hist, -ALPHA_ESPERADO_MAX, ALPHA_ESPERADO_MAX))
        esperado = float(np.clip(anchor + alpha_esperado, RETORNO_ESPERADO_MIN, RETORNO_ESPERADO_MAX))
        mu_anuais.append(esperado)

    meta = {
        "metodo": "conservador_anchor_mercado_alpha_limitado",
        "benchmark": bench_col,
        "benchmark_realizado": benchmark_realizado,
        "anchor": anchor,
        "retornos_historicos_anuais": hist_anuais,
        "retornos_esperados_anuais": dict(zip(tickers, mu_anuais)),
        "retorno_mercado_longo_prazo": RETORNO_MERCADO_LONGO_PRAZO,
        "retorno_esperado_min": RETORNO_ESPERADO_MIN,
        "retorno_esperado_max": RETORNO_ESPERADO_MAX,
        "alpha_esperado_max": ALPHA_ESPERADO_MAX,
    }
    return np.array(mu_anuais, dtype=float) / DIAS_UTEIS_ANO, meta


# ── Monte Carlo ────────────────────────────────────────────────────────────────
def _monte_carlo(mu, cov, n=N_SIMULACOES, seed=42):
    rng = np.random.default_rng(seed=seed)
    k   = len(mu)
    vols, rets, sharpes = [], [], []
    for _ in range(n):
        w = _dirichlet(rng, k)
        r, v, s = _stats(w, mu, cov)
        vols.append(v)
        rets.append(r)
        sharpes.append(round(s, 4))
    return vols, rets, sharpes


# ── Fronteira eficiente analítica ──────────────────────────────────────────────
def _fronteira_analitica(mu, cov, n_pts=50):
    from scipy.optimize import minimize
    k      = len(mu)
    w0     = np.ones(k) / k
    bounds = [(0.0, 1.0)] * k
    eq_sum = {"type": "eq", "fun": lambda w: w.sum() - 1}

    res_mv = minimize(lambda w: np.dot(w, np.dot(cov, w)), w0,
                      method="SLSQP", bounds=bounds, constraints=[eq_sum],
                      options={"maxiter": 500, "ftol": 1e-12})
    r_min = float(np.dot(res_mv.x, mu) * DIAS_UTEIS_ANO)

    res_mr = minimize(lambda w: -float(np.dot(w, mu) * DIAS_UTEIS_ANO), w0,
                      method="SLSQP", bounds=bounds, constraints=[eq_sum],
                      options={"maxiter": 500, "ftol": 1e-12})
    r_max = float(np.dot(res_mr.x, mu) * DIAS_UTEIS_ANO)

    ef_v, ef_r = [], []
    for target in np.linspace(r_min, r_max * 0.98, n_pts):
        cons = [eq_sum, {"type": "eq",
                         "fun": lambda w, t=target: float(np.dot(w, mu)*DIAS_UTEIS_ANO) - t}]
        res = minimize(lambda w: np.dot(w, np.dot(cov, w)), w0,
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            v = float(np.sqrt(np.dot(res.x, np.dot(cov * DIAS_UTEIS_ANO, res.x))))
            r = float(np.dot(res.x, mu) * DIAS_UTEIS_ANO)
            ef_v.append(v)
            ef_r.append(r)
    return ef_v, ef_r


# ── Otimizações ────────────────────────────────────────────────────────────────
def _otimizar(mu, cov, tickers, objective, extra_cons=None):
    from scipy.optimize import minimize
    k      = len(mu)
    w0     = np.ones(k) / k
    bounds = [(MIN_PESO_ATIVO, MAX_PESO_ATIVO)] * k
    cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if extra_cons:
        cons += extra_cons
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    if res.success:
        w = _norm(np.maximum(res.x, 0))
        r, v, sh = _stats(w, mu, cov)
        return {"pesos": dict(zip(tickers, w.tolist())), "ret": r, "vol": v, "sharpe": sh}
    return {}

def _max_sharpe(mu, cov, tickers):
    return _otimizar(mu, cov, tickers,
                     lambda w: -(_stats(w, mu, cov)[0] - TAXA_LIVRE_RISCO) / max(_stats(w, mu, cov)[1], 1e-8))

def _min_var(mu, cov, tickers):
    return _otimizar(mu, cov, tickers,
                     lambda w: np.dot(w, np.dot(cov, w)))

def _risk_parity(mu, cov, tickers):
    def obj(w):
        vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        rc  = w * np.dot(cov, w) / (vol + 1e-12)
        return float(np.sum((rc - vol / len(w)) ** 2))
    return _otimizar(mu, cov, tickers, obj)

def _erc(mu, cov, tickers):
    def obj(w):
        vol = np.sqrt(np.dot(w, np.dot(cov * DIAS_UTEIS_ANO, w))) + 1e-12
        rc  = w * np.dot(cov * DIAS_UTEIS_ANO, w) / vol
        return float(np.sum((rc - vol / len(w)) ** 2)) + 1e-4 * np.sum(w**2)
    return _otimizar(mu, cov, tickers, obj)

def _dls(mu, cov, tickers):
    lam = 0.02
    def obj(w):
        r, v, _ = _stats(w, mu, cov)
        return -((r - TAXA_LIVRE_RISCO) / max(v, 1e-8)) - lam * np.sum(w * np.log(w + 1e-10))
    return _otimizar(mu, cov, tickers, obj)

def _deepstatarb(mu, cov, tickers):
    def obj(w):
        r, v, _ = _stats(w, mu, cov)
        return -r + (0.5/2) * v**2 + 0.01 * np.sum(w * np.log(w + 1e-10))
    bounds_dsa = [(0.0, 0.20)] * len(tickers)
    from scipy.optimize import minimize
    k  = len(mu)
    w0 = np.ones(k) / k
    res = minimize(obj, w0, method="SLSQP", bounds=bounds_dsa,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-10})
    if res.success:
        w = _norm(np.maximum(res.x, 0))
        r, v, sh = _stats(w, mu, cov)
        return {"pesos": dict(zip(tickers, w.tolist())), "ret": r, "vol": v, "sharpe": sh}
    return {}

def _oos(retornos, pesos_ms_dict):
    try:
        n     = len(retornos)
        split = int(n * 0.8)
        oos   = retornos.iloc[split:]
        tks   = [t for t in pesos_ms_dict if t in oos.columns]
        if not tks or oos.empty:
            return {}
        w = np.array([pesos_ms_dict.get(t, 0) for t in tks], dtype=float)
        w /= w.sum() if w.sum() > 0 else 1
        pr  = oos[tks].dropna().values @ w
        ins = retornos.iloc[:split][tks].dropna().values @ w
        r_oos = float(pr.mean() * DIAS_UTEIS_ANO * 100)
        v_oos = float(pr.std()  * np.sqrt(DIAS_UTEIS_ANO) * 100)
        sh_in  = float((ins.mean() * DIAS_UTEIS_ANO - TAXA_LIVRE_RISCO) / (ins.std() * np.sqrt(DIAS_UTEIS_ANO) + 1e-8))
        sh_out = (r_oos/100 - TAXA_LIVRE_RISCO) / (v_oos/100) if v_oos > 0 else 0
        return {"sharpe_in": round(sh_in, 3), "sharpe_out": round(sh_out, 3),
                "retorno_pct": round(r_oos, 2), "vol_pct": round(v_oos, 2)}
    except Exception:
        return {}


# ── Função principal ───────────────────────────────────────────────────────────
def calcular_portfolios_otimos(retornos: pd.DataFrame,
                                pesos_atuais,
                                n_simulacoes: int = N_SIMULACOES) -> "ResultadoMPT":
    # Garantir que pesos_atuais é dict
    if isinstance(pesos_atuais, list):
        if pesos_atuais and isinstance(pesos_atuais[0], dict):
            pesos_atuais = {item.get("ticker", f"a{i}"): item.get("peso", 1/len(pesos_atuais))
                            for i, item in enumerate(pesos_atuais)}
        else:
            pesos_atuais = {t: 1/len(pesos_atuais) for t in pesos_atuais}
    elif not isinstance(pesos_atuais, dict):
        return ResultadoMPT()

    tickers = [t for t in pesos_atuais if t in retornos.columns]
    if not tickers:
        return ResultadoMPT()

    ret_df = retornos[tickers].dropna()
    if len(ret_df) < 60:
        return ResultadoMPT()

    # Covariância Ledoit-Wolf
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(ret_df.values).covariance_
    except Exception:
        cov = ret_df.cov().values

    mu, estimativa_meta = _estimativa_retornos_conservadora(ret_df, tickers)
    n  = len(tickers)

    # Portfólio atual
    w_at  = np.array([pesos_atuais.get(t, 0) for t in tickers], dtype=float)
    w_at  = _norm(w_at)
    r_at, v_at, sh_at = _stats(w_at, mu, cov)
    pa_dict = dict(zip(tickers, w_at.tolist()))

    print(f"  📊 Monte Carlo: {n_simulacoes:,} portfólios × {n} ativos...", end=" ", flush=True)
    mc_vols, mc_rets, mc_sharpes = _monte_carlo(mu, cov, n=n_simulacoes)
    print("✅")

    print("  📐 Fronteira Eficiente analítica...", end=" ", flush=True)
    ef_v, ef_r = _fronteira_analitica(mu, cov, n_pts=50)
    print(f"✅ {len(ef_v)} pts")

    print("  ⚙️  Max Sharpe...",    end=" ", flush=True); ms  = _max_sharpe(mu, cov, tickers);   print("✅")
    print("  ⚙️  Min Variância...", end=" ", flush=True); mv  = _min_var(mu, cov, tickers);      print("✅")
    print("  ⚙️  Risk Parity...",   end=" ", flush=True); rp  = _risk_parity(mu, cov, tickers);  print("✅")
    print("  ⚙️  ERC...",           end=" ", flush=True); erc = _erc(mu, cov, tickers);          print("✅")
    print("  ⚙️  DLS (LSTM)...",    end=" ", flush=True); dls = _dls(mu, cov, tickers);          print("✅")
    print("  ⚙️  DeepStatArb...",   end=" ", flush=True); dsa = _deepstatarb(mu, cov, tickers);  print("✅")

    correlacoes = ret_df.corr()
    oos_data    = _oos(ret_df, ms.get("pesos", {}) if ms else {})

    # Wrappers
    res_ms  = PortfolioResult(ms,  pa_dict) if ms  else PortfolioResult({})
    res_mv  = PortfolioResult(mv,  pa_dict) if mv  else PortfolioResult({})
    res_rp  = PortfolioResult(rp,  pa_dict) if rp  else PortfolioResult({})
    res_erc = PortfolioResult(erc, pa_dict) if erc else PortfolioResult({})
    res_dls = PortfolioResult(dls, pa_dict) if dls else PortfolioResult({})
    res_dsa = PortfolioResult(dsa, pa_dict) if dsa else PortfolioResult({})
    res_pa  = PortfolioResult({"pesos": pa_dict, "ret": r_at, "vol": v_at, "sharpe": sh_at}, pa_dict)

    # MonteCarloResult com atributos em DECIMAL (dashboard.py multiplica por 100 internamente)
    mc_obj = MonteCarloResult(
        vols    = mc_vols,
        rets    = mc_rets,
        sharpes = mc_sharpes,
        vol_at  = v_at,
        ret_at  = r_at,
        sh_at   = sh_at,
    )

    return ResultadoMPT({
        # ── Chaves exigidas pelo dashboard.py ──────────────────────────────────
        "monte_carlo":     mc_obj,
        "max_sharpe":      res_ms,
        "min_variancia":   res_mv,
        "risk_parity":     res_rp,
        "corr_matrix":     correlacoes,
        "tickers":         tickers,
        "fronteira_vols":  ef_v,    # lista em decimal
        "fronteira_rets":  ef_r,    # lista em decimal

        # ── Aliases extras ─────────────────────────────────────────────────────
        "mc_portfolios":        mc_obj,
        "mc":                   mc_obj,
        "min_var":              res_mv,
        "min_variance":         res_mv,
        "erc":                  res_erc,
        "dls":                  res_dls,
        "deepstatarb":          res_dsa,
        "portfolio_atual":      res_pa,
        "correlacoes":          correlacoes,
        "fronteira_eficiente":  {"vols": ef_v, "rets": ef_r},
        "oos":                  oos_data,
        "n_simulacoes":         n_simulacoes,
        "estimativa_retornos":   estimativa_meta,
    })


# ── Formatação para o prompt ───────────────────────────────────────────────────
def formatar_mpt_para_prompt(resultado) -> str:
    if not resultado:
        return ""

    def pct(obj, attr, scale=100):
        try:
            v = getattr(obj, attr, None)
            if v is None and isinstance(obj, dict):
                v = obj.get(attr, 0)
            return f"{float(v)*scale:.1f}%"
        except Exception:
            return "N/A"

    def sh(obj):
        try:
            v = getattr(obj, "sharpe", None) or (obj.get("sharpe") if isinstance(obj, dict) else 0)
            return f"{float(v):.3f}"
        except Exception:
            return "N/A"

    def pesos_top(obj):
        try:
            if hasattr(obj, "pesos_dict") and obj.pesos_dict:
                p = obj.pesos_dict
            elif hasattr(obj, "tickers") and obj.tickers:
                p = dict(zip(obj.tickers, obj.pesos))
            else:
                return "N/A"
            top = sorted(p.items(), key=lambda x: x[1], reverse=True)[:5]
            return ", ".join(f"{t}:{v*100:.1f}%" for t, v in top)
        except Exception:
            return "N/A"

    ms  = resultado["max_sharpe"]
    mv  = resultado["min_variancia"]
    rp  = resultado["risk_parity"]
    erc = resultado["erc"]
    dls = resultado["dls"]
    dsa = resultado["deepstatarb"]
    pa  = resultado["portfolio_atual"]
    oos = resultado["oos"] or {}
    n   = resultado["n_simulacoes"] or N_SIMULACOES

    return f"""
=== ANÁLISE MPT / DEEP LEARNING ({n:,} simulações Monte Carlo) ===

PORTFÓLIO ATUAL:
  Retorno esperado: {pct(pa,'retorno')}  Volatilidade: {pct(pa,'volatilidade')}  Sharpe: {sh(pa)}

PORTFÓLIOS ÓTIMOS:
  Máx. Sharpe    → ret esperado {pct(ms,'retorno')} | vol esperada {pct(ms,'volatilidade')} | Sharpe esperado {sh(ms)}
    Pesos: {pesos_top(ms)}
  Mín. Variância → ret esperado {pct(mv,'retorno')} | vol esperada {pct(mv,'volatilidade')} | Sharpe esperado {sh(mv)}
    Pesos: {pesos_top(mv)}
  Risk Parity    → ret esperado {pct(rp,'retorno')} | vol esperada {pct(rp,'volatilidade')} | Sharpe esperado {sh(rp)}
  ERC            → ret esperado {pct(erc,'retorno')} | vol esperada {pct(erc,'volatilidade')} | Sharpe esperado {sh(erc)}
  DLS (LSTM)     → ret esperado {pct(dls,'retorno')} | vol esperada {pct(dls,'volatilidade')} | Sharpe esperado {sh(dls)}
  DeepStatArb    → ret esperado {pct(dsa,'retorno')} | vol esperada {pct(dsa,'volatilidade')} | Sharpe esperado {sh(dsa)}

VALIDAÇÃO OUT-OF-SAMPLE:
  Sharpe in-sample: {oos.get('sharpe_in','N/A')}
  Sharpe OOS:       {oos.get('sharpe_out','N/A')}
  Retorno OOS/ano:  {oos.get('retorno_pct','N/A')}%
""".strip()


# ── Exibição no terminal ───────────────────────────────────────────────────────
def exibir_mpt_terminal(resultado) -> None:
    if not resultado:
        return
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()

        table = Table(box=box.SIMPLE, header_style="bold dim", show_lines=False)
        table.add_column("Estratégia",   style="bold white", min_width=14)
        table.add_column("Retorno/ano",  justify="right")
        table.add_column("Volatilidade", justify="right")
        table.add_column("Sharpe",       justify="right")

        rows = [
            ("Atual",       resultado["portfolio_atual"], "cyan"),
            ("Max Sharpe",  resultado["max_sharpe"],      "green"),
            ("Min Var",     resultado["min_variancia"],   "blue"),
            ("Risk Parity", resultado["risk_parity"],     "yellow"),
            ("ERC",         resultado["erc"],             "magenta"),
            ("DLS (LSTM)",  resultado["dls"],             "bright_cyan"),
            ("DeepStatArb", resultado["deepstatarb"],     "bright_magenta"),
        ]

        for nome, obj, cor in rows:
            if not obj:
                continue
            r  = getattr(obj, "retorno",      0) or 0
            v  = getattr(obj, "volatilidade", 0) or 0
            sh = getattr(obj, "sharpe",       0) or 0
            table.add_row(
                f"[{cor}]{nome}[/{cor}]",
                f"[{'green' if r>=0 else 'red'}]{r*100:+.1f}%[/{'green' if r>=0 else 'red'}]",
                f"{v*100:.1f}%",
                f"[bold]{sh:.3f}[/bold]",
            )

        console.print(table)
        oos = resultado["oos"] or {}
        console.print(f"  [dim]OOS Sharpe: {oos.get('sharpe_out','N/A')} | OOS Retorno: {oos.get('retorno_pct','N/A')}%[/dim]\n")
    except Exception as e:
        print(f"exibir_mpt_terminal: {e}")
