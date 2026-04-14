"""
brokers/base.py — Interface abstrata para integração com corretoras
Todas as corretoras devem implementar estes métodos.

Para ativar uma corretora, mude no .env:
  BROKER_IB_ENABLED=true     → Interactive Brokers / Avenue
  BROKER_BTG_ENABLED=true    → BTG Pactual
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class OrdemTipo(Enum):
    MARKET = "market"       # executa ao preço atual
    LIMIT = "limit"         # executa só no preço definido
    STOP = "stop"           # stop loss


class OrdemAcao(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Ordem:
    ticker: str
    acao: OrdemAcao
    quantidade: int
    tipo: OrdemTipo = OrdemTipo.MARKET
    preco_limite: Optional[float] = None   # só para LIMIT e STOP
    motivo: str = ""                        # justificativa do agente


@dataclass
class ResultadoOrdem:
    sucesso: bool
    ordem_id: Optional[str]
    ticker: str
    acao: str
    quantidade: int
    preco_executado: Optional[float]
    mensagem: str


@dataclass
class Posicao:
    ticker: str
    quantidade: int
    preco_medio: float
    preco_atual: float
    valor_total: float
    resultado_pct: float


class BrokerBase(ABC):
    """Interface que toda corretora deve implementar."""

    ENABLED: bool = False   # ← flag de segurança — sempre False por padrão

    @abstractmethod
    def conectar(self) -> bool:
        """Estabelece conexão com a corretora. Retorna True se conectado."""
        pass

    @abstractmethod
    def get_posicoes(self) -> list[Posicao]:
        """Retorna posições atuais da conta na corretora."""
        pass

    @abstractmethod
    def get_saldo_disponivel(self) -> float:
        """Retorna capital disponível em USD (ou BRL para BTG)."""
        pass

    @abstractmethod
    def enviar_ordem(self, ordem: Ordem) -> ResultadoOrdem:
        """Envia uma ordem. Só executa se ENABLED=True."""
        pass

    @abstractmethod
    def cancelar_ordem(self, ordem_id: str) -> bool:
        """Cancela ordem pendente pelo ID."""
        pass

    def validar_ordem(self, ordem: Ordem) -> tuple[bool, str]:
        """
        Validações de segurança antes de qualquer envio.
        Retorna (pode_executar, motivo).
        """
        if not self.ENABLED:
            return False, "Corretora desativada — ENABLED=False no .env"

        if ordem.quantidade <= 0:
            return False, "Quantidade inválida"

        if ordem.tipo == OrdemTipo.LIMIT and not ordem.preco_limite:
            return False, "Ordem LIMIT requer preco_limite"

        saldo = self.get_saldo_disponivel()
        if ordem.acao == OrdemAcao.BUY and saldo <= 0:
            return False, f"Saldo insuficiente: ${saldo:.2f}"

        return True, "OK"
