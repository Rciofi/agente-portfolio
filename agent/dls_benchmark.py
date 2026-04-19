"""
DLS Benchmark Runner
Integra o modelo DLS (Zhang et al., 2021) ao pipeline de backtesting
existente do Agente de Gestão de Portfólio como 4º benchmark.

Uso:
    from benchmarks.dls_benchmark import DLSBenchmark

    dls = DLSBenchmark(tickers=tickers, prices_df=prices_df)
    dls.fit(train_end="2014-12-31")
    results = dls.run(test_start="2015-01-01", test_end="2017-12-31")
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DLSConfig:
    """Configurações do modelo DLS (Zhang et al., 2021)."""

    # Arquitetura
    lookback: int = 50          # janela de observações passadas (dias)
    lstm_units: int = 64        # unidades LSTM (paper usa 64)
    dropout: float = 0.1

    # Treinamento
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    val_split: float = 0.10
    patience: int = 15
    seed: int = 42

    # Portfólio
    cost_rate: float = 0.0001   # 1bp (paper: C = 0.01%)
    rebalance_freq: int = 5     # dias úteis (semanal)

    # Volatility scaling
    apply_vol_scaling: bool = True
    sigma_target: float = 0.10  # 10% a.a.
    ewm_span: int = 50

    # Device
    device: str = "cpu"


class DLSBenchmark:
    """
    Benchmark DLS para comparação com DeepStatArb, Markowitz e Buy&Hold.

    Implementa a metodologia de Zhang, Zohren & Roberts (2021):
    - LSTM end-to-end com otimização direta do Sharpe
    - Volatility scaling por ativo
    - Custo de transação explícito
    """

    def __init__(
        self,
        tickers: list[str],
        prices_df: pd.DataFrame,
        config: Optional[DLSConfig] = None,
    ):
        """
        tickers:   lista de tickers no universo
        prices_df: DataFrame (index=datas, columns=tickers) com preços ajustados
        config:    configurações do modelo (usa padrões do paper se None)
        """
        self.tickers = tickers
        self.prices_df = prices_df[tickers].copy()
        self.config = config or DLSConfig()
        self.model = None
        self._is_fitted = False

        logger.info(
            f"[DLSBenchmark] Inicializado: {len(tickers)} ativos | "
            f"vol_scaling={self.config.apply_vol_scaling} | "
            f"σtgt={self.config.sigma_target:.0%}"
        )

    # ─────────────────────────────────────────
    # TREINO
    # ─────────────────────────────────────────

    def fit(self, train_end: str) -> "DLSBenchmark":
        """
        Treina o modelo com dados até train_end (exclusive).

        train_end: "YYYY-MM-DD"
        """
        try:
            from models.dls_model import train_dls
        except ImportError:
            logger.error("[DLSBenchmark] Não foi possível importar dls_model. Verifique o PYTHONPATH.")
            raise

        prices_train = self.prices_df.loc[:train_end].values.astype(np.float32)

        logger.info(
            f"[DLSBenchmark] Treinando até {train_end} "
            f"({len(prices_train)} dias de dados)"
        )

        self.model = train_dls(
            prices_train=prices_train,
            lookback=self.config.lookback,
            lstm_units=self.config.lstm_units,
            lr=self.config.learning_rate,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            cost_rate=self.config.cost_rate,
            val_split=self.config.val_split,
            patience=self.config.patience,
            device=self.config.device,
            seed=self.config.seed,
        )
        self._is_fitted = True
        return self

    # ─────────────────────────────────────────
    # BACKTEST
    # ─────────────────────────────────────────

    def run(
        self,
        test_start: str,
        test_end: str,
    ) -> dict:
        """
        Executa backtest fora da amostra e retorna resultados estruturados
        no mesmo formato dos demais benchmarks do agente.

        retorna: dict compatível com o pipeline de comparação existente
        """
        if not self._is_fitted:
            raise RuntimeError("Chame .fit() antes de .run()")

        try:
            from models.dls_model import run_dls_backtest
        except ImportError:
            raise

        prices_test = self.prices_df.loc[test_start:test_end].values.astype(np.float32)
        test_dates = self.prices_df.loc[test_start:test_end].index

        logger.info(
            f"[DLSBenchmark] Backtest: {test_start} → {test_end} "
            f"({len(prices_test)} dias)"
        )

        raw = run_dls_backtest(
            model=self.model,
            prices_test=prices_test,
            lookback=self.config.lookback,
            sigma_target=self.config.sigma_target,
            cost_rate=self.config.cost_rate,
            rebalance_freq=self.config.rebalance_freq,
            apply_vol_scaling=self.config.apply_vol_scaling,
            device=self.config.device,
        )

        # Alinha datas (lookback reduz o número de observações)
        n_obs = len(raw["portfolio_returns"])
        aligned_dates = test_dates[-n_obs:]

        # Monta DataFrame de pesos por ativo (para o dashboard)
        weights_df = pd.DataFrame(
            raw["weights"],
            index=aligned_dates,
            columns=self.tickers,
        )

        returns_series = pd.Series(
            raw["portfolio_returns"],
            index=aligned_dates,
            name="DLS",
        )

        cumulative_series = pd.Series(
            raw["cumulative_returns"],
            index=aligned_dates,
            name="DLS",
        )

        return {
            "strategy": "DLS (Zhang et al., 2021)",
            "metrics": raw["metrics"],
            "weights_df": weights_df,
            "returns_series": returns_series,
            "cumulative_series": cumulative_series,
            "ewm_vol": raw.get("ewm_vol"),
            "raw_weights": raw.get("raw_weights"),
        }

    # ─────────────────────────────────────────
    # SENSITIVITY ANALYSIS
    # ─────────────────────────────────────────

    def sensitivity_analysis(
        self,
        prices_sample: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula sensitivity analysis das features (Zhang et al. Eq. 8).

        prices_sample: (lookback + 1, n_assets) — janela recente
        retorna: (lookback, n_assets * 2) — importância normalizada por feature
        """
        if not self._is_fitted:
            raise RuntimeError("Chame .fit() antes de sensitivity_analysis()")

        import torch
        from models.dls_model import prepare_dls_features, compute_sensitivity

        X, _ = prepare_dls_features(prices_sample, lookback=self.config.lookback)
        if len(X) == 0:
            logger.warning("[DLSBenchmark] Amostra insuficiente para sensitivity analysis")
            return np.array([])

        x_tensor = torch.tensor(X[-1:], dtype=torch.float32)
        return compute_sensitivity(self.model, x_tensor, device=self.config.device)

    # ─────────────────────────────────────────
    # WALK-FORWARD RETRAINING
    # ─────────────────────────────────────────

    def walk_forward_fit(
        self,
        retrain_every_years: int = 2,
        train_start: str = "2010-01-01",
        test_end: str = "2017-12-31",
    ) -> "DLSBenchmark":
        """
        Retreinamento periódico conforme Zhang et al. (2021):
        'We retrain our model at every 2 years and use all data available
         up to that point to update parameters.'

        Armazena pontos de retreinamento para logging.
        """
        retrain_dates = pd.date_range(
            start=train_start,
            end=test_end,
            freq=f"{retrain_every_years}YE",
        )

        self._retrain_checkpoints = []
        for dt in retrain_dates:
            logger.info(f"[DLSBenchmark] Walk-forward retrain até {dt.date()}")
            self.fit(train_end=str(dt.date()))
            self._retrain_checkpoints.append(str(dt.date()))

        return self
