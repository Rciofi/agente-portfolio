"""
brokers/manager.py — Orquestra corretoras e monta portfólio automaticamente

Prioridade de input:
  1. IB/Avenue (se BROKER_IB_ENABLED=true)  → mercado EUA
  2. BTG       (se BROKER_BTG_ENABLED=true)  → mercado BR
  3. portfolio.csv                            → fallback sempre disponível

O portfólio resultante é sempre a união das fontes ativas.
"""

import os
import pandas as pd
from typing import Optional
from .base import BrokerBase, Ordem, ResultadoOrdem, Posicao


class BrokerManager:

    def __init__(self):
        self._ib: Optional[BrokerBase] = None
        self._btg: Optional[BrokerBase] = None
        self._inicializar()

    def _inicializar(self):
        ib_enabled = os.getenv("BROKER_IB_ENABLED", "false").lower() == "true"
        btg_enabled = os.getenv("BROKER_BTG_ENABLED", "false").lower() == "true"

        if ib_enabled:
            from .interactive_brokers import InteractiveBrokersBroker
            broker = InteractiveBrokersBroker()
            if broker.conectar():
                self._ib = broker
            else:
                print("  ⚠️  [IB] Conexão falhou — usando fallback CSV para EUA.")

        if btg_enabled:
            from .btg import BTGBroker
            broker = BTGBroker()
            if broker.conectar():
                self._btg = broker
            else:
                print("  ⚠️  [BTG] Conexão falhou — usando fallback CSV para BR.")

    @property
    def ativo(self) -> bool:
        return self._ib is not None or self._btg is not None

    @property
    def brokers_ativos(self) -> list[str]:
        ativos = []
        if self._ib:
            ativos.append("Interactive Brokers / Avenue")
        if self._btg:
            ativos.append("BTG Pactual")
        return ativos

    # ─────────────────────────────────────────────────────────────
    # PORTFÓLIO AUTOMÁTICO
    # ─────────────────────────────────────────────────────────────

    def get_portfolio_automatico(self) -> tuple[list[dict], float]:
        """
        Busca posições de todas as corretoras ativas e retorna
        no mesmo formato que carregar_portfolio() usa do CSV.

        Returns:
            (posicoes, capital_disponivel)
            posicoes: lista de dicts com ticker, quantidade, preco_medio
        """
        posicoes = []
        capital_total = 0.0

        # IB / Avenue — EUA
        if self._ib:
            ib_posicoes = self._ib.get_posicoes()
            capital_total += self._ib.get_saldo_disponivel()
            for p in ib_posicoes:
                posicoes.append({
                    "ticker": p.ticker,
                    "quantidade": p.quantidade,
                    "preco_medio": p.preco_medio,
                    "mercado": "US",
                    "fonte": "IB/Avenue",
                })
            print(f"  [IB] {len(ib_posicoes)} posição(ões) importada(s)")

        # BTG — Brasil
        if self._btg:
            btg_posicoes = self._btg.get_posicoes()
            capital_total += self._btg.get_saldo_disponivel()
            for p in btg_posicoes:
                posicoes.append({
                    "ticker": p.ticker,
                    "quantidade": p.quantidade,
                    "preco_medio": p.preco_medio,
                    "mercado": "BR",
                    "fonte": "BTG",
                })
            print(f"  [BTG] {len(btg_posicoes)} posição(ões) importada(s)")

        return posicoes, capital_total

    def get_historico_ordens(self, dias: int = 30) -> list[dict]:
        """
        Agrega histórico de ordens executadas de todas as corretoras ativas.
        Ordena por data decrescente.
        """
        ordens = []

        if self._ib:
            ordens_ib = self._ib.get_historico_ordens(dias)
            for o in ordens_ib:
                o["fonte"] = "IB/Avenue"
            ordens.extend(ordens_ib)

        if self._btg:
            ordens_btg = self._btg.get_historico_ordens(dias)
            for o in ordens_btg:
                o["fonte"] = "BTG"
            ordens.extend(ordens_btg)

        return sorted(ordens, key=lambda x: x.get("data", ""), reverse=True)

    # ─────────────────────────────────────────────────────────────
    # SALDO E ORDENS
    # ─────────────────────────────────────────────────────────────

    def get_saldo_disponivel(self, mercado: str = "us") -> float:
        if mercado == "us" and self._ib:
            return self._ib.get_saldo_disponivel()
        if mercado == "br" and self._btg:
            return self._btg.get_saldo_disponivel()
        return 0.0

    def enviar_ordem(self, ordem: Ordem, mercado: str = "us") -> ResultadoOrdem:
        broker = self._ib if mercado == "us" else self._btg
        if not broker:
            return ResultadoOrdem(
                sucesso=False, ordem_id=None,
                ticker=ordem.ticker, acao=ordem.acao.value,
                quantidade=ordem.quantidade, preco_executado=None,
                mensagem=f"Nenhuma corretora ativa para mercado '{mercado}'"
            )
        return broker.enviar_ordem(ordem)

    def desconectar(self):
        if self._ib:
            try:
                self._ib.desconectar()
            except Exception:
                pass
