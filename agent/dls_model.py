"""
DLS - Deep Learning Sharpe Model
Baseado em: Zhang, Zohren & Roberts (2021)
"Deep Learning for Portfolio Optimization"
Oxford-Man Institute of Quantitative Finance

Implementa:
- Arquitetura LSTM end-to-end com otimização direta do Sharpe ratio
- Volatility scaling (σtgt) para controle de risco
- Sensitivity analysis das features de input
- Long-only portfolio via softmax output
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. ARQUITECTURA LSTM
# ─────────────────────────────────────────────

class DLSNetwork(nn.Module):
    """
    Rede LSTM para otimização direta do Sharpe ratio.

    Input:  (batch, lookback, n_assets * n_features)
    Output: (batch, n_assets)  →  pesos via softmax (long-only, soma = 1)

    Features por ativo: preço normalizado + retorno diário
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int = 2,
        lookback: int = 50,
        lstm_units: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.n_features = n_features
        self.lookback = lookback

        input_size = n_assets * n_features

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_units, n_assets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, lookback, n_assets * n_features)
        retorna pesos: (batch, n_assets)
        """
        lstm_out, _ = self.lstm(x)          # (batch, lookback, lstm_units)
        last = lstm_out[:, -1, :]           # pega o último timestep
        last = self.dropout(last)
        raw = self.fc(last)                 # (batch, n_assets)
        weights = torch.softmax(raw, dim=-1)
        return weights


# ─────────────────────────────────────────────
# 2. FUNÇÃO DE PERDA: Sharpe direto
# ─────────────────────────────────────────────

def sharpe_loss(
    weights: torch.Tensor,
    returns: torch.Tensor,
    cost_rate: float = 0.0001,
    prev_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Maximiza o Sharpe ratio diretamente (gradient ascent → loss negativa).

    weights:      (batch, n_assets)
    returns:      (batch, n_assets)  — retornos do período seguinte
    cost_rate:    taxa de transação (padrão 1bp)
    prev_weights: pesos anteriores para cálculo de custo de transação
    """
    portfolio_returns = (weights * returns).sum(dim=1)  # (batch,)

    # Custos de transação
    if prev_weights is not None:
        turnover = (weights - prev_weights).abs().sum(dim=1)
        portfolio_returns = portfolio_returns - cost_rate * turnover

    mean_ret = portfolio_returns.mean()
    std_ret = portfolio_returns.std() + 1e-8

    sharpe = mean_ret / std_ret
    return -sharpe  # negativo para minimizar (= maximizar Sharpe)


# ─────────────────────────────────────────────
# 3. VOLATILITY SCALING
# ─────────────────────────────────────────────

def compute_ewm_volatility(
    returns: np.ndarray,
    span: int = 50,
) -> np.ndarray:
    """
    Volatilidade ex-ante via EWMA (exponentially weighted moving std).
    Mesma metodologia do paper de Oxford (span=50 dias).

    returns: (T, n_assets)
    retorna: (T, n_assets)
    """
    T, n = returns.shape
    vol = np.zeros_like(returns)
    alpha = 2.0 / (span + 1)

    # Variância EWMA
    ewm_var = np.var(returns[:span], axis=0)
    for t in range(T):
        ewm_var = alpha * returns[t] ** 2 + (1 - alpha) * ewm_var
        vol[t] = np.sqrt(ewm_var) + 1e-8

    return vol


def apply_volatility_scaling(
    weights: np.ndarray,
    returns: np.ndarray,
    sigma_target: float = 0.10,
    span: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica volatility scaling nos pesos conforme Zhang et al. (2021) Eq. 7.

    Rp_scaled = Σ (σtgt / σi,t-1) * wi,t * ri,t

    weights:      (T, n_assets)
    returns:      (T, n_assets)
    sigma_target: volatilidade alvo anualizada (default 10%)
    retorna:      (scaled_weights, ewm_vol)
    """
    sigma_daily = sigma_target / np.sqrt(252)
    ewm_vol = compute_ewm_volatility(returns, span=span)

    # Scaling factor por ativo e período
    scaling = sigma_daily / (ewm_vol + 1e-8)

    # Aplica scaling e renormaliza para soma = 1 (long-only)
    scaled = weights * scaling
    row_sums = scaled.sum(axis=1, keepdims=True) + 1e-8
    scaled_weights = scaled / row_sums

    return scaled_weights, ewm_vol


# ─────────────────────────────────────────────
# 4. SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────

def compute_sensitivity(
    model: DLSNetwork,
    x: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Sensitivity analysis normalizada conforme Zhang et al. (2021) Eq. 8.

    Si = |dL/dxi| / max_j(|dL/dxj|)

    x:       (1, lookback, n_assets * n_features)
    retorna: (lookback, n_assets * n_features)  — importância de cada feature
    """
    model.eval()
    x = x.to(device).requires_grad_(True)

    weights = model(x)

    # Usa a soma dos pesos como proxy da função objetivo
    objective = weights.sum()
    objective.backward()

    if x.grad is None:
        return np.zeros((x.shape[1], x.shape[2]))

    grad = x.grad.abs().squeeze(0)  # (lookback, n_features)
    max_val = grad.max() + 1e-8
    sensitivity = (grad / max_val).detach().cpu().numpy()

    return sensitivity


# ─────────────────────────────────────────────
# 5. PREPARAÇÃO DE DADOS
# ─────────────────────────────────────────────

def prepare_dls_features(
    prices: np.ndarray,
    lookback: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepara features no formato do paper: preços normalizados + retornos.

    prices: (T, n_assets)
    retorna:
        X: (T - lookback, lookback, n_assets * 2)
        y: (T - lookback, n_assets)  — retornos do próximo período
    """
    T, n = prices.shape

    # Retornos diários
    rets = np.zeros_like(prices)
    rets[1:] = prices[1:] / (prices[:-1] + 1e-8) - 1

    X_list, y_list = [], []

    for t in range(lookback, T - 1):
        price_window = prices[t - lookback:t]  # (lookback, n)
        ret_window = rets[t - lookback:t]       # (lookback, n)

        # Normaliza preços pela janela
        price_norm = price_window / (price_window[0] + 1e-8)

        # Concatena features: [preço_norm, retorno]  → (lookback, n*2)
        features = np.concatenate([price_norm, ret_window], axis=1)
        X_list.append(features)

        # Target: retorno do próximo dia
        y_list.append(rets[t + 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


# ─────────────────────────────────────────────
# 6. TREINAMENTO
# ─────────────────────────────────────────────

def train_dls(
    prices_train: np.ndarray,
    lookback: int = 50,
    lstm_units: int = 64,
    lr: float = 0.001,
    epochs: int = 100,
    batch_size: int = 64,
    cost_rate: float = 0.0001,
    val_split: float = 0.10,
    patience: int = 15,
    device: str = "cpu",
    seed: int = 42,
) -> DLSNetwork:
    """
    Treina o modelo DLS com otimização direta do Sharpe ratio.

    prices_train: (T, n_assets) — preços ajustados do período de treino
    retorna: modelo treinado
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    T, n_assets = prices_train.shape
    logger.info(f"[DLS] Treinando com {T} dias, {n_assets} ativos, lookback={lookback}")

    X, y = prepare_dls_features(prices_train, lookback=lookback)

    # Split treino/validação temporal (sem data leakage)
    split = int(len(X) * (1 - val_split))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DLSNetwork(
        n_assets=n_assets,
        n_features=2,
        lookback=lookback,
        lstm_units=lstm_units,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=False
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        prev_w = None
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            w = model(X_batch)
            loss = sharpe_loss(w, y_batch, cost_rate=cost_rate, prev_weights=prev_w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            prev_w = w.detach()

        # Validação
        model.eval()
        with torch.no_grad():
            w_val = model(X_val_t.to(device))
            val_loss = sharpe_loss(w_val, y_val_t.to(device), cost_rate=cost_rate)

        scheduler.step(val_loss)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"[DLS] Early stopping na época {epoch + 1}")
            break

        if (epoch + 1) % 20 == 0:
            mean_tl = np.mean(train_losses)
            logger.info(
                f"[DLS] Época {epoch + 1:3d} | "
                f"Train Sharpe: {-mean_tl:.3f} | "
                f"Val Sharpe: {-val_loss.item():.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(f"[DLS] Treinamento concluído. Melhor val Sharpe: {-best_val_loss:.3f}")
    return model


# ─────────────────────────────────────────────
# 7. INFERÊNCIA / BACKTEST
# ─────────────────────────────────────────────

def run_dls_backtest(
    model: DLSNetwork,
    prices_test: np.ndarray,
    lookback: int = 50,
    sigma_target: float = 0.10,
    cost_rate: float = 0.0001,
    rebalance_freq: int = 5,
    apply_vol_scaling: bool = True,
    device: str = "cpu",
) -> dict:
    """
    Executa backtest fora da amostra com o modelo DLS treinado.

    prices_test: (T, n_assets)
    retorna: dict com métricas e séries temporais
    """
    model.eval()
    T, n_assets = prices_test.shape

    X, y_true = prepare_dls_features(prices_test, lookback=lookback)

    raw_weights = []
    with torch.no_grad():
        for i in range(0, len(X), rebalance_freq):
            x_t = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
            w = model(x_t).cpu().numpy().squeeze()
            # Mantém pesos fixos até próximo rebalanceamento
            for _ in range(min(rebalance_freq, len(X) - i)):
                raw_weights.append(w)

    raw_weights = np.array(raw_weights[:len(X)])

    if apply_vol_scaling:
        rets_for_vol = y_true
        scaled_weights, ewm_vol = apply_volatility_scaling(
            raw_weights, rets_for_vol, sigma_target=sigma_target
        )
        weights_used = scaled_weights
    else:
        weights_used = raw_weights
        ewm_vol = None

    # Retornos da carteira
    portfolio_returns = (weights_used * y_true).sum(axis=1)

    # Custos de transação
    turnover = np.abs(np.diff(weights_used, axis=0, prepend=weights_used[:1])).sum(axis=1)
    portfolio_returns -= cost_rate * turnover

    # Métricas
    ann_factor = 252
    cumulative = np.cumprod(1 + portfolio_returns)
    ann_return = cumulative[-1] ** (ann_factor / len(portfolio_returns)) - 1
    ann_vol = portfolio_returns.std() * np.sqrt(ann_factor)
    sharpe = ann_return / (ann_vol + 1e-8)

    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / (running_max + 1e-8)
    max_drawdown = drawdown.min()

    logger.info(
        f"[DLS Backtest] Sharpe={sharpe:.3f} | "
        f"Ret={ann_return:.1%} | Vol={ann_vol:.1%} | MDD={max_drawdown:.1%}"
    )

    return {
        "weights": weights_used,
        "raw_weights": raw_weights,
        "portfolio_returns": portfolio_returns,
        "cumulative_returns": cumulative,
        "ewm_vol": ewm_vol,
        "metrics": {
            "sharpe_ratio": round(sharpe, 3),
            "annual_return": round(ann_return, 4),
            "annual_volatility": round(ann_vol, 4),
            "max_drawdown": round(max_drawdown, 4),
            "cumulative_return": round(float(cumulative[-1]) - 1, 4),
        },
    }
