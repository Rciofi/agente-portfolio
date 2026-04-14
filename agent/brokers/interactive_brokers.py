"""
brokers/interactive_brokers.py — Interactive Brokers / Avenue
Suporta duas formas de conexão:

MODO 1 — IBKR Client Portal API (recomendado, sem app local)
  - REST API oficial da IB
  - Requer: Client Portal Gateway rodando (jar leve, sem interface gráfica)
  - Docs: https://www.interactivebrokers.com/api/doc.html
  - Setup: https://github.com/InteractiveBrokers/cpwebapi

MODO 2 — TWS / IB Gateway via ib_insync (alternativo)
  - Requer TWS ou IB Gateway aberto localmente
  - Docs: https://ib-insync.readthedocs.io

ATIVAR:
  No .env: BROKER_IB_ENABLED=true
  Escolha o modo: IB_MODE=clientportal  (ou tws)

AVENUE:
  Conta Avenue usa custódia IB. Se você tiver conta IB associada,
  use as credenciais IB normalmente. Sem conta IB direta, use portfolio.csv.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Optional
from .base import BrokerBase, Ordem, OrdemAcao, OrdemTipo, ResultadoOrdem, Posicao


class InteractiveBrokersBroker(BrokerBase):

    ENABLED: bool = os.getenv("BROKER_IB_ENABLED", "false").lower() == "true"
    MODO: str = os.getenv("IB_MODE", "clientportal")  # clientportal | tws

    # Client Portal API
    CP_BASE_URL: str = os.getenv("IB_CP_URL", "https://localhost:5000/v1/api")

    # TWS / IB Gateway
    IB_HOST: str = os.getenv("IB_HOST", "127.0.0.1")
    IB_PORT: int = int(os.getenv("IB_PORT", "7497"))
    IB_CLIENT_ID: int = int(os.getenv("IB_CLIENT_ID", "1"))

    def __init__(self):
        self._ib = None          # ib_insync instance (modo tws)
        self._session = None     # requests.Session (modo clientportal)
        self._conectado = False
        self._account_id: Optional[str] = None

    # ─────────────────────────────────────────────────────────────
    # CONEXÃO
    # ─────────────────────────────────────────────────────────────

    def conectar(self) -> bool:
        if not self.ENABLED:
            print("  [IB] Desativado — BROKER_IB_ENABLED=false")
            return False

        if self.MODO == "clientportal":
            return self._conectar_clientportal()
        else:
            return self._conectar_tws()

    def _conectar_clientportal(self) -> bool:
        """
        Client Portal Gateway — REST sem app gráfico.
        Inicie o gateway antes: ./bin/run.sh root/conf.yaml
        """
        try:
            # CP Gateway usa certificado self-signed — desabilita verificação SSL local
            self._session = requests.Session()
            self._session.verify = False

            # Suprimir warnings de SSL em desenvolvimento
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            # Verifica autenticação
            resp = self._session.get(
                f"{self.CP_BASE_URL}/iserver/auth/status",
                timeout=5
            )
            data = resp.json()

            if data.get("authenticated"):
                # Descobre account ID
                acc_resp = self._session.get(
                    f"{self.CP_BASE_URL}/portfolio/accounts",
                    timeout=5
                )
                accounts = acc_resp.json()
                if accounts:
                    self._account_id = accounts[0].get("accountId")

                self._conectado = True
                print(f"  [IB] Client Portal conectado → conta {self._account_id}")
                return True
            else:
                print("  [IB] Client Portal não autenticado.")
                print("       Acesse https://localhost:5000 no browser e faça login.")
                return False

        except requests.exceptions.ConnectionError:
            print("  [IB] Client Portal Gateway não está rodando.")
            print("       Inicie com: ./bin/run.sh root/conf.yaml")
            return False
        except Exception as e:
            print(f"  [IB] Erro Client Portal: {e}")
            return False

    def _conectar_tws(self) -> bool:
        """TWS / IB Gateway via ib_insync."""
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self.IB_HOST, self.IB_PORT, clientId=self.IB_CLIENT_ID)
            self._conectado = self._ib.isConnected()
            if self._conectado:
                print(f"  [IB] TWS conectado → {self.IB_HOST}:{self.IB_PORT}")
            return self._conectado
        except ImportError:
            print("  [IB] ib_insync não instalado. Execute: pip install ib_insync")
            return False
        except Exception as e:
            print(f"  [IB] Erro TWS: {e}")
            return False

    # ─────────────────────────────────────────────────────────────
    # POSIÇÕES
    # ─────────────────────────────────────────────────────────────

    def get_posicoes(self) -> list[Posicao]:
        if not self.ENABLED or not self._conectado:
            return []

        if self.MODO == "clientportal":
            return self._posicoes_clientportal()
        else:
            return self._posicoes_tws()

    def _posicoes_clientportal(self) -> list[Posicao]:
        try:
            resp = self._session.get(
                f"{self.CP_BASE_URL}/portfolio/{self._account_id}/positions/0",
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            posicoes = []
            for item in data:
                preco_medio = float(item.get("avgCost", 0))
                preco_atual = float(item.get("mktPrice", 0))
                qtd = int(item.get("position", 0))

                if qtd == 0:
                    continue

                resultado_pct = (
                    (preco_atual - preco_medio) / preco_medio * 100
                    if preco_medio else 0
                )

                posicoes.append(Posicao(
                    ticker=item.get("ticker", item.get("contractDesc", "")),
                    quantidade=abs(qtd),
                    preco_medio=round(preco_medio, 4),
                    preco_atual=round(preco_atual, 4),
                    valor_total=round(preco_atual * abs(qtd), 2),
                    resultado_pct=round(resultado_pct, 2),
                ))

            return posicoes

        except Exception as e:
            print(f"  [IB] Erro ao buscar posições (CP): {e}")
            return []

    def _posicoes_tws(self) -> list[Posicao]:
        try:
            posicoes = []
            for item in self._ib.portfolio():
                preco_medio = item.averageCost
                preco_atual = item.marketPrice
                qtd = int(item.position)
                resultado_pct = (
                    (preco_atual - preco_medio) / preco_medio * 100
                    if preco_medio else 0
                )
                posicoes.append(Posicao(
                    ticker=item.contract.symbol,
                    quantidade=abs(qtd),
                    preco_medio=round(preco_medio, 4),
                    preco_atual=round(preco_atual, 4),
                    valor_total=round(item.marketValue, 2),
                    resultado_pct=round(resultado_pct, 2),
                ))
            return posicoes
        except Exception as e:
            print(f"  [IB] Erro ao buscar posições (TWS): {e}")
            return []

    # ─────────────────────────────────────────────────────────────
    # SALDO
    # ─────────────────────────────────────────────────────────────

    def get_saldo_disponivel(self) -> float:
        if not self.ENABLED or not self._conectado:
            return 0.0

        if self.MODO == "clientportal":
            return self._saldo_clientportal()
        else:
            return self._saldo_tws()

    def _saldo_clientportal(self) -> float:
        try:
            resp = self._session.get(
                f"{self.CP_BASE_URL}/portfolio/{self._account_id}/summary",
                timeout=10
            )
            data = resp.json()
            cash = data.get("availablefunds", {})
            return float(cash.get("amount", 0))
        except Exception:
            return 0.0

    def _saldo_tws(self) -> float:
        try:
            for v in self._ib.accountValues():
                if v.tag == "AvailableFunds" and v.currency == "USD":
                    return float(v.value)
            return 0.0
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────
    # HISTÓRICO DE ORDENS
    # ─────────────────────────────────────────────────────────────

    def get_historico_ordens(self, dias: int = 30) -> list[dict]:
        """
        Retorna ordens executadas nos últimos N dias.
        Campos: ticker, acao, quantidade, preco, data, status
        """
        if not self.ENABLED or not self._conectado:
            return []

        if self.MODO == "clientportal":
            return self._historico_clientportal(dias)
        else:
            return self._historico_tws()

    def _historico_clientportal(self, dias: int) -> list[dict]:
        try:
            resp = self._session.get(
                f"{self.CP_BASE_URL}/iserver/account/trades",
                timeout=10
            )
            resp.raise_for_status()
            trades = resp.json()

            corte = datetime.now() - timedelta(days=dias)
            ordens = []

            for t in trades:
                # Parse de data (formato IB: "YYYYMMDD-HH:MM:SS")
                raw_date = t.get("trade_time", t.get("tradeTime", ""))
                try:
                    if "-" in raw_date:
                        data = datetime.strptime(raw_date, "%Y%m%d-%H:%M:%S")
                    else:
                        data = datetime.strptime(raw_date[:8], "%Y%m%d")
                except Exception:
                    data = datetime.now()

                if data < corte:
                    continue

                ordens.append({
                    "ticker": t.get("symbol", ""),
                    "acao": t.get("side", ""),
                    "quantidade": abs(float(t.get("size", 0))),
                    "preco": float(t.get("price", 0)),
                    "valor_total": abs(float(t.get("size", 0))) * float(t.get("price", 0)),
                    "data": data.strftime("%Y-%m-%d %H:%M"),
                    "status": "EXECUTADA",
                    "exchange": t.get("exchange", ""),
                })

            return sorted(ordens, key=lambda x: x["data"], reverse=True)

        except Exception as e:
            print(f"  [IB] Erro ao buscar histórico (CP): {e}")
            return []

    def _historico_tws(self) -> list[dict]:
        try:
            execucoes = self._ib.reqExecutions()
            ordens = []
            for ex in execucoes:
                ordens.append({
                    "ticker": ex.contract.symbol,
                    "acao": ex.execution.side,
                    "quantidade": ex.execution.shares,
                    "preco": ex.execution.price,
                    "valor_total": ex.execution.shares * ex.execution.price,
                    "data": ex.execution.time,
                    "status": "EXECUTADA",
                    "exchange": ex.execution.exchange,
                })
            return sorted(ordens, key=lambda x: x["data"], reverse=True)
        except Exception as e:
            print(f"  [IB] Erro ao buscar histórico (TWS): {e}")
            return []

    # ─────────────────────────────────────────────────────────────
    # ORDENS
    # ─────────────────────────────────────────────────────────────

    def enviar_ordem(self, ordem: Ordem) -> ResultadoOrdem:
        pode, motivo = self.validar_ordem(ordem)
        if not pode:
            return ResultadoOrdem(
                sucesso=False, ordem_id=None,
                ticker=ordem.ticker, acao=ordem.acao.value,
                quantidade=ordem.quantidade, preco_executado=None,
                mensagem=f"Bloqueado: {motivo}"
            )

        if self.MODO == "clientportal":
            return self._enviar_clientportal(ordem)
        else:
            return self._enviar_tws(ordem)

    def _enviar_clientportal(self, ordem: Ordem) -> ResultadoOrdem:
        try:
            # Primeiro busca conid (contract ID) do ticker
            search = self._session.get(
                f"{self.CP_BASE_URL}/iserver/secdef/search",
                params={"symbol": ordem.ticker, "secType": "STK"},
                timeout=10
            )
            contracts = search.json()
            if not contracts:
                raise ValueError(f"Ticker {ordem.ticker} não encontrado")

            conid = contracts[0]["conid"]

            payload = {
                "acctId": self._account_id,
                "conid": conid,
                "secType": f"{conid}:STK",
                "orderType": ordem.tipo.value.upper(),
                "side": ordem.acao.value,
                "quantity": ordem.quantidade,
                "tif": "DAY",
            }
            if ordem.preco_limite:
                payload["price"] = ordem.preco_limite

            resp = self._session.post(
                f"{self.CP_BASE_URL}/iserver/account/{self._account_id}/orders",
                json={"orders": [payload]},
                timeout=10
            )
            data = resp.json()

            ordem_id = str(data[0].get("order_id", "")) if data else None
            return ResultadoOrdem(
                sucesso=True, ordem_id=ordem_id,
                ticker=ordem.ticker, acao=ordem.acao.value,
                quantidade=ordem.quantidade, preco_executado=None,
                mensagem=f"Ordem enviada via CP — ID: {ordem_id} | {ordem.motivo}"
            )

        except Exception as e:
            return ResultadoOrdem(
                sucesso=False, ordem_id=None,
                ticker=ordem.ticker, acao=ordem.acao.value,
                quantidade=ordem.quantidade, preco_executado=None,
                mensagem=f"Erro CP: {e}"
            )

    def _enviar_tws(self, ordem: Ordem) -> ResultadoOrdem:
        try:
            from ib_insync import Stock, MarketOrder, LimitOrder, StopOrder
            contrato = Stock(ordem.ticker, "SMART", "USD")

            if ordem.tipo == OrdemTipo.MARKET:
                ib_ordem = MarketOrder(ordem.acao.value, ordem.quantidade)
            elif ordem.tipo == OrdemTipo.LIMIT:
                ib_ordem = LimitOrder(ordem.acao.value, ordem.quantidade, ordem.preco_limite)
            else:
                ib_ordem = StopOrder(ordem.acao.value, ordem.quantidade, ordem.preco_limite)

            trade = self._ib.placeOrder(contrato, ib_ordem)
            self._ib.sleep(1)

            return ResultadoOrdem(
                sucesso=True,
                ordem_id=str(trade.order.orderId),
                ticker=ordem.ticker, acao=ordem.acao.value,
                quantidade=ordem.quantidade,
                preco_executado=trade.orderStatus.avgFillPrice or None,
                mensagem=f"Ordem TWS — ID: {trade.order.orderId} | {ordem.motivo}"
            )
        except Exception as e:
            return ResultadoOrdem(
                sucesso=False, ordem_id=None,
                ticker=ordem.ticker, acao=ordem.acao.value,
                quantidade=ordem.quantidade, preco_executado=None,
                mensagem=f"Erro TWS: {e}"
            )

    def cancelar_ordem(self, ordem_id: str) -> bool:
        if not self.ENABLED or not self._conectado:
            return False
        if self.MODO == "clientportal":
            try:
                resp = self._session.delete(
                    f"{self.CP_BASE_URL}/iserver/account/{self._account_id}/order/{ordem_id}",
                    timeout=10
                )
                return resp.status_code == 200
            except Exception:
                return False
        else:
            try:
                for o in self._ib.openOrders():
                    if str(o.orderId) == ordem_id:
                        self._ib.cancelOrder(o)
                        return True
                return False
            except Exception:
                return False

    def desconectar(self):
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass


AvenuebrokerBroker = InteractiveBrokersBroker
