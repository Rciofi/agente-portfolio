"""
brokers/btg.py — BTG Pactual Digital
API REST com OAuth 2.0 (client_credentials)

ATIVAR:
  1. No .env: BROKER_BTG_ENABLED=true
  2. Solicite acesso em: https://www.btgpactualdigital.com/plataforma/developer
  3. Preencha BTG_CLIENT_ID, BTG_CLIENT_SECRET e BTG_ACCOUNT_ID no .env

NOTAS:
  - Ordens em BRL para ativos B3
  - Token OAuth expira em 1h — renovação automática implementada
  - Para ADRs via BTG, confirme disponibilidade na sua conta
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Optional
from .base import BrokerBase, Ordem, OrdemAcao, OrdemTipo, ResultadoOrdem, Posicao


class BTGBroker(BrokerBase):

    ENABLED: bool = os.getenv("BROKER_BTG_ENABLED", "false").lower() == "true"

    BASE_URL = "https://api.btgpactual.com"
    AUTH_URL = "https://auth.btgpactual.com/oauth/token"

    def __init__(self):
        self._client_id = os.getenv("BTG_CLIENT_ID", "")
        self._client_secret = os.getenv("BTG_CLIENT_SECRET", "")
        self._account_id = os.getenv("BTG_ACCOUNT_ID", "")
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._conectado = False

    # ─────────────────────────────────────────────────────────────
    # AUTENTICAÇÃO
    # ─────────────────────────────────────────────────────────────

    def conectar(self) -> bool:
        if not self.ENABLED:
            print("  [BTG] Desativado — BROKER_BTG_ENABLED=false")
            return False

        if not self._client_id or not self._client_secret:
            print("  [BTG] BTG_CLIENT_ID / BTG_CLIENT_SECRET não configurados.")
            return False

        try:
            resp = requests.post(self.AUTH_URL, data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            }, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            self._token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
            self._conectado = True
            print(f"  [BTG] Autenticado — conta {self._account_id}")
            return True

        except requests.exceptions.HTTPError as e:
            print(f"  [BTG] Falha de autenticação: {e.response.status_code} — verifique credenciais.")
            return False
        except Exception as e:
            print(f"  [BTG] Erro: {e}")
            return False

    def _headers(self) -> dict:
        """Renova token automaticamente se expirado."""
        if not self._token or datetime.now() >= (self._token_expiry or datetime.min):
            self.conectar()
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "x-account-id": self._account_id,
        }

    # ─────────────────────────────────────────────────────────────
    # POSIÇÕES
    # ─────────────────────────────────────────────────────────────

    def get_posicoes(self) -> list[Posicao]:
        if not self.ENABLED or not self._conectado:
            return []

        try:
            resp = requests.get(
                f"{self.BASE_URL}/v1/portfolio/positions",
                headers=self._headers(), timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            posicoes = []
            for item in data.get("positions", []):
                preco_medio = float(item.get("averagePrice", 0))
                preco_atual = float(item.get("currentPrice", 0))
                qtd = int(item.get("quantity", 0))

                if qtd == 0:
                    continue

                resultado_pct = (
                    (preco_atual - preco_medio) / preco_medio * 100
                    if preco_medio else 0
                )

                posicoes.append(Posicao(
                    ticker=item.get("ticker", ""),
                    quantidade=abs(qtd),
                    preco_medio=round(preco_medio, 4),
                    preco_atual=round(preco_atual, 4),
                    valor_total=round(preco_atual * abs(qtd), 2),
                    resultado_pct=round(resultado_pct, 2),
                ))

            return posicoes

        except Exception as e:
            print(f"  [BTG] Erro ao buscar posições: {e}")
            return []

    # ─────────────────────────────────────────────────────────────
    # SALDO
    # ─────────────────────────────────────────────────────────────

    def get_saldo_disponivel(self) -> float:
        if not self.ENABLED or not self._conectado:
            return 0.0
        try:
            resp = requests.get(
                f"{self.BASE_URL}/v1/account/balance",
                headers=self._headers(), timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("availableBalance", 0))
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────
    # HISTÓRICO DE ORDENS
    # ─────────────────────────────────────────────────────────────

    def get_historico_ordens(self, dias: int = 30) -> list[dict]:
        """
        Retorna ordens executadas nos últimos N dias via BTG.
        """
        if not self.ENABLED or not self._conectado:
            return []

        try:
            data_inicio = (datetime.now() - timedelta(days=dias)).strftime("%Y-%m-%d")
            data_fim = datetime.now().strftime("%Y-%m-%d")

            resp = requests.get(
                f"{self.BASE_URL}/v1/orders",
                headers=self._headers(),
                params={
                    "startDate": data_inicio,
                    "endDate": data_fim,
                    "status": "FILLED",   # só executadas
                },
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            ordens = []
            for item in data.get("orders", []):
                preco = float(item.get("executedPrice", item.get("price", 0)))
                qtd = float(item.get("executedQuantity", item.get("quantity", 0)))

                ordens.append({
                    "ticker": item.get("ticker", ""),
                    "acao": item.get("side", ""),
                    "quantidade": qtd,
                    "preco": preco,
                    "valor_total": round(qtd * preco, 2),
                    "data": item.get("executedAt", item.get("createdAt", "")),
                    "status": "EXECUTADA",
                    "exchange": "B3",
                    "moeda": "BRL",
                })

            return sorted(ordens, key=lambda x: x["data"], reverse=True)

        except Exception as e:
            print(f"  [BTG] Erro ao buscar histórico: {e}")
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

        try:
            payload = {
                "ticker": ordem.ticker,
                "side": ordem.acao.value,
                "quantity": ordem.quantidade,
                "orderType": ordem.tipo.value.upper(),
                "notes": ordem.motivo,
            }
            if ordem.preco_limite:
                payload["price"] = ordem.preco_limite

            resp = requests.post(
                f"{self.BASE_URL}/v1/orders",
                json=payload,
                headers=self._headers(),
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            return ResultadoOrdem(
                sucesso=True,
                ordem_id=data.get("orderId"),
                ticker=ordem.ticker, acao=ordem.acao.value,
                quantidade=ordem.quantidade,
                preco_executado=data.get("executedPrice"),
                mensagem=f"Ordem BTG — ID: {data.get('orderId')} | {ordem.motivo}"
            )

        except Exception as e:
            return ResultadoOrdem(
                sucesso=False, ordem_id=None,
                ticker=ordem.ticker, acao=ordem.acao.value,
                quantidade=ordem.quantidade, preco_executado=None,
                mensagem=f"Erro BTG: {e}"
            )

    def cancelar_ordem(self, ordem_id: str) -> bool:
        if not self.ENABLED or not self._conectado:
            return False
        try:
            resp = requests.delete(
                f"{self.BASE_URL}/v1/orders/{ordem_id}",
                headers=self._headers(), timeout=10
            )
            return resp.status_code == 200
        except Exception:
            return False
