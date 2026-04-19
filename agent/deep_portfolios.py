"""
deep_portfolios.py — Portfólios de Deep Learning aplicados ao portfólio real
Integra DLS (Zhang et al., 2021) e DeepStatArb (CNN-Transformer) ao pipeline
do agente, usando os mesmos ativos e dados históricos do MPT.

Ambos retornam um PortfolioOtimo compatível com calcular_portfolios_otimos(),
podendo ser exibidos no dashboard e enviados ao Claude para recomendação.
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# UTILITÁRIOS COMUNS
# ─────────────────────────────────────────────────────────────────────

def _metricas(pesos: np.ndarray, retornos: pd.DataFrame, rf: float = 0.04):
    """Retorna (retorno_anual, vol_anual, sharpe) para um vetor de pesos."""
    ret_medios = retornos.mean().values
    cov = retornos.cov().values
    ret = float(np.dot(pesos, ret_medios) * 252)
    vol = float(np.sqrt(pesos @ cov * 252 @ pesos))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return round(ret, 4), round(vol, 4), round(sharpe, 4)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────
# DLS — Deep Learning Sharpe (Zhang et al., 2021)
# Otimização direta do Sharpe via LSTM end-to-end
# ─────────────────────────────────────────────────────────────────────

def _treinar_dls(
    retornos: pd.DataFrame,
    lookback: int = 50,
    epochs: int = 80,
    lr: float = 0.001,
    lstm_units: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """
    Treina LSTM para maximizar Sharpe diretamente (sem previsão de retornos).
    Retorna vetor de pesos ótimos (n_ativos,).
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        logger.warning("[DLS] PyTorch não disponível — usando fallback Risk Parity")
        vols = retornos.std().values * np.sqrt(252)
        w = 1 / (vols + 1e-8)
        return w / w.sum()

    torch.manual_seed(seed)
    np.random.seed(seed)

    T, n = retornos.shape
    rets = retornos.values.astype(np.float32)

    # Prepara features: preços normalizados + retornos (janela de lookback)
    prices = np.cumprod(1 + rets, axis=0)
    X_list, y_list = [], []
    for t in range(lookback, T - 1):
        p_win = prices[t-lookback:t]
        r_win = rets[t-lookback:t]
        p_norm = p_win / (p_win[0] + 1e-8)
        feat = np.concatenate([p_norm, r_win], axis=1)  # (lookback, n*2)
        X_list.append(feat)
        y_list.append(rets[t+1])

    if len(X_list) < 20:
        logger.warning("[DLS] Dados insuficientes — usando fallback")
        vols = retornos.std().values * np.sqrt(252)
        w = 1 / (vols + 1e-8)
        return w / w.sum()

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)

    # Arquitetura LSTM simples
    class DLSNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n * 2, lstm_units, batch_first=True)
            self.fc = nn.Linear(lstm_units, n)
            self.drop = nn.Dropout(0.1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.softmax(self.fc(self.drop(out[:, -1])), dim=-1)

    model = DLSNet()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_sharpe = -999
    best_weights = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        w = model(X)
        port_ret = (w * y).sum(dim=1)
        mean_r = port_ret.mean()
        std_r = port_ret.std() + 1e-8
        loss = -(mean_r / std_r)  # maximiza Sharpe
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        sharpe_val = float(-loss.item())
        if sharpe_val > best_sharpe:
            best_sharpe = sharpe_val
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)

    # Pesos finais: média das últimas previsões
    model.eval()
    with torch.no_grad():
        w_final = model(X[-20:]).mean(dim=0).numpy()

    logger.info(f"[DLS] Treinado — Sharpe in-sample: {best_sharpe:.3f}")
    return w_final


# ─────────────────────────────────────────────────────────────────────
# DeepStatArb — CNN-Transformer (arquitetura do pré-projeto Henri)
# Extração de features via CNN + atenção temporal via Transformer
# ─────────────────────────────────────────────────────────────────────

def _treinar_deepstatarb(
    retornos: pd.DataFrame,
    lookback: int = 60,
    epochs: int = 60,
    lr: float = 0.001,
    seed: int = 42,
) -> np.ndarray:
    """
    CNN-Transformer com função de perda média-variância + regularização entrópica.
    Retorna vetor de pesos ótimos (n_ativos,).
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        logger.warning("[DeepStatArb] PyTorch não disponível — usando fallback")
        vols = retornos.std().values * np.sqrt(252)
        w = 1 / (vols + 1e-8)
        return w / w.sum()

    torch.manual_seed(seed)
    np.random.seed(seed)

    T, n = retornos.shape
    rets = retornos.values.astype(np.float32)

    # Features: retornos z-scored (resíduos simplificados sem Fama-French
    # para compatibilidade com qualquer universo de ativos)
    mean_r = rets.mean(axis=0)
    std_r = rets.std(axis=0) + 1e-8
    rets_norm = (rets - mean_r) / std_r

    X_list, y_list = [], []
    for t in range(lookback, T - 1):
        X_list.append(rets_norm[t-lookback:t])   # (lookback, n)
        y_list.append(rets[t+1])

    if len(X_list) < 20:
        logger.warning("[DeepStatArb] Dados insuficientes — usando fallback")
        vols = retornos.std().values * np.sqrt(252)
        w = 1 / (vols + 1e-8)
        return w / w.sum()

    X = torch.tensor(np.array(X_list), dtype=torch.float32)  # (T, lookback, n)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)

    # Arquitetura CNN-Transformer
    class DeepStatArbNet(nn.Module):
        def __init__(self):
            super().__init__()
            # CNN 1D extrai padrões locais
            self.conv1 = nn.Conv1d(n, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
            self.pool  = nn.AdaptiveAvgPool1d(32)
            self.relu  = nn.ReLU()
            self.drop  = nn.Dropout(0.1)
            # Transformer Encoder captura dependências temporais
            enc_layer = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64,
                dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
            # Saída: pesos via softmax
            self.fc = nn.Linear(32, n)

        def forward(self, x):
            # x: (batch, lookback, n) → CNN espera (batch, n, lookback)
            x = x.permute(0, 2, 1)
            x = self.drop(self.relu(self.conv1(x)))
            x = self.drop(self.relu(self.conv2(x)))
            x = self.pool(x)           # (batch, 32, 32)
            x = x.permute(0, 2, 1)    # (batch, 32, 32) para Transformer
            x = self.transformer(x)
            x = x.mean(dim=1)         # pooling temporal
            return torch.softmax(self.fc(x), dim=-1)

    model = DeepStatArbNet()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    best_loss = float("inf")
    best_weights = None
    lam = 0.5   # aversão ao risco
    alpha = 0.01  # regularização entrópica

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        w = model(X)
        port_ret = (w * y).sum(dim=1)
        mu_p  = port_ret.mean()
        var_p = port_ret.var() + 1e-8
        # L = -μ + (λ/2)σ² + α·Σwᵢlog(wᵢ)  (média-variância + entropia)
        entropy = (w * torch.log(w + 1e-8)).sum(dim=1).mean()
        loss = -mu_p + (lam / 2) * var_p + alpha * entropy
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        w_final = model(X[-20:]).mean(dim=0).numpy()

    # Calcula Sharpe aproximado para log
    port_rets = (rets[-len(y_list):] * w_final).sum(axis=1)
    sharpe_approx = (port_rets.mean() / (port_rets.std() + 1e-8)) * np.sqrt(252)
    logger.info(f"[DeepStatArb] Treinado — Sharpe in-sample aprox: {sharpe_approx:.3f}")

    return w_final


# ─────────────────────────────────────────────────────────────────────
# API PÚBLICA — integra com calcular_portfolios_otimos()
# ─────────────────────────────────────────────────────────────────────

def calcular_portfolios_deep(
    retornos: pd.DataFrame,
    pesos_atuais: np.ndarray,
    rf: float = 0.04,
) -> dict:
    """
    Calcula portfólios DLS e DeepStatArb para os ativos reais do portfólio.

    Args:
        retornos:      DataFrame de retornos diários (mesmo usado pelo MPT)
        pesos_atuais:  array com pesos atuais alinhados com retornos.columns
        rf:            taxa livre de risco

    Returns:
        dict com "dls" e "deepstatarb" — objetos PortfolioOtimo compatíveis
    """
    from agent.mpt import PortfolioOtimo, metricas_portfolio, covariancia_ledoit_wolf

    tickers = list(retornos.columns)
    n = len(tickers)
    ret_medios = retornos.mean().values
    cov_matrix = covariancia_ledoit_wolf(retornos)

    resultados = {}

    # ── DLS ──────────────────────────────────────────────────────────
    logger.info("[Deep] Treinando DLS nos ativos do portfólio...")
    try:
        w_dls = _treinar_dls(retornos)
        w_dls = np.clip(w_dls, 0, None)
        w_dls = w_dls / (w_dls.sum() + 1e-8)
        r_dls, v_dls, s_dls = metricas_portfolio(w_dls, ret_medios, cov_matrix, rf)

        resultados["dls"] = PortfolioOtimo(
            tipo="dls",
            tickers=tickers,
            pesos=[round(float(w), 4) for w in w_dls],
            retorno=round(r_dls, 4),
            volatilidade=round(v_dls, 4),
            sharpe=round(s_dls, 4),
            pesos_atuais=[round(float(p), 4) for p in pesos_atuais],
            delta_pesos=[round(float(w-p), 4) for w, p in zip(w_dls, pesos_atuais)],
        )
        logger.info(f"[DLS] Sharpe={s_dls:.3f} | Ret={r_dls:.1%} | Vol={v_dls:.1%}")

    except Exception as e:
        logger.error(f"[DLS] Falha: {e}")
        resultados["dls"] = None

    # ── DeepStatArb ───────────────────────────────────────────────────
    logger.info("[Deep] Treinando DeepStatArb nos ativos do portfólio...")
    try:
        w_dsa = _treinar_deepstatarb(retornos)
        w_dsa = np.clip(w_dsa, 0, None)
        w_dsa = w_dsa / (w_dsa.sum() + 1e-8)
        r_dsa, v_dsa, s_dsa = metricas_portfolio(w_dsa, ret_medios, cov_matrix, rf)

        resultados["deepstatarb"] = PortfolioOtimo(
            tipo="deepstatarb",
            tickers=tickers,
            pesos=[round(float(w), 4) for w in w_dsa],
            retorno=round(r_dsa, 4),
            volatilidade=round(v_dsa, 4),
            sharpe=round(s_dsa, 4),
            pesos_atuais=[round(float(p), 4) for p in pesos_atuais],
            delta_pesos=[round(float(w-p), 4) for w, p in zip(w_dsa, pesos_atuais)],
        )
        logger.info(f"[DeepStatArb] Sharpe={s_dsa:.3f} | Ret={r_dsa:.1%} | Vol={v_dsa:.1%}")

    except Exception as e:
        logger.error(f"[DeepStatArb] Falha: {e}")
        resultados["deepstatarb"] = None

    return resultados


def formatar_deep_para_prompt(deep_resultado: dict) -> str:
    """Formata resultados Deep Learning para o prompt do Claude."""
    linhas = ["\n=== DEEP LEARNING PORTFOLIO OPTIMIZATION ==="]

    for key, label in [("dls", "DLS — LSTM (Zhang et al., 2021)"),
                        ("deepstatarb", "DeepStatArb — CNN-Transformer")]:
        p = deep_resultado.get(key)
        if not p:
            continue
        linhas.append(f"\n{label}:")
        linhas.append(f"  Retorno: {p.retorno*100:.1f}% | Vol: {p.volatilidade*100:.1f}% | Sharpe: {p.sharpe:.3f}")
        linhas.append("  Alocação sugerida:")
        for t, w, delta in zip(p.tickers, p.pesos, p.delta_pesos):
            if abs(delta) > 0.02:
                sinal = "▲" if delta > 0 else "▼"
                linhas.append(f"    {sinal} {t}: {(w)*100:.1f}% ({delta*100:+.1f}%)")

    return "\n".join(linhas)
