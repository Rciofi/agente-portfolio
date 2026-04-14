"""
factors.py — Score multi-fator quantitativo (0-100) por ativo

Modelo de 4 fatores com pesos iguais:
  25% Momentum     — RSI, tendência, distância das médias móveis
  25% Valuation    — P/L, PEG, P/VP, upside analistas
  25% Qualidade    — ROE, margens, crescimento, endividamento
  25% Volatilidade — IVR, HV vs IV, beta, ATR relativo

O score é enviado ao Claude como contexto adicional,
tornando as recomendações mais direcionadas e consistentes.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class FactorScore:
    ticker: str
    score_total: float          # 0-100
    score_momentum: float       # 0-100
    score_valuation: float      # 0-100
    score_qualidade: float      # 0-100
    score_volatilidade: float   # 0-100
    sinal: str                  # FORTE COMPRA / COMPRA / NEUTRO / VENDA / FORTE VENDA
    detalhes: dict              # breakdown de cada sub-fator


def _normalizar(valor, minv, maxv, inverso=False) -> float:
    """Normaliza um valor para 0-100. inverso=True quando menor é melhor."""
    if valor is None or minv == maxv:
        return 50.0
    try:
        valor = float(valor)
        score = (valor - minv) / (maxv - minv) * 100
        score = float(np.clip(score, 0, 100))
        return 100 - score if inverso else score
    except Exception:
        return 50.0


# ─────────────────────────────────────────────────────────────────
# FATOR 1: MOMENTUM
# ─────────────────────────────────────────────────────────────────

def calcular_momentum(dados: dict) -> tuple[float, dict]:
    """
    Componentes:
    - RSI (14): sobrevendido=bom para compra, sobrecomprado=sinal de venda
    - Distância da SMA50 (%): acima = momentum positivo
    - Distância da SMA200 (%): tendência de longo prazo
    - Tendência 90d: ALTA=bom, BAIXA=ruim
    - Variação do dia: sinal de curto prazo
    """
    scores = {}

    # RSI — zona ideal entre 40-60, penaliza extremos
    rsi = dados.get("rsi")
    if rsi:
        try:
            rsi = float(str(rsi).replace("%", ""))
            if rsi < 30:
                scores["rsi"] = 85   # sobrevendido — oportunidade de compra
            elif rsi < 40:
                scores["rsi"] = 70
            elif rsi <= 60:
                scores["rsi"] = 55   # zona neutra saudável
            elif rsi <= 70:
                scores["rsi"] = 35
            else:
                scores["rsi"] = 15   # sobrecomprado — risco de realização
        except Exception:
            scores["rsi"] = 50
    else:
        scores["rsi"] = 50

    # Distância SMA50 — acima da média = momentum positivo
    preco = dados.get("preco_atual", 0)
    sma50 = dados.get("sma50")
    if sma50 and preco:
        try:
            sma50 = float(str(sma50).replace("%", ""))
            dist = (preco - sma50) / sma50 * 100
            # Leve acima da SMA50 é bom; muito acima pode ser esticado
            if -5 <= dist <= 10:
                scores["sma50"] = 70
            elif dist > 10:
                scores["sma50"] = 45   # esticado
            elif dist > -10:
                scores["sma50"] = 40
            else:
                scores["sma50"] = 20   # abaixo da média — tendência negativa
        except Exception:
            scores["sma50"] = 50
    else:
        scores["sma50"] = 50

    # Distância SMA200 — tendência de longo prazo
    sma200 = dados.get("sma200")
    if sma200 and preco:
        try:
            sma200 = float(str(sma200).replace("%", ""))
            dist200 = (preco - sma200) / sma200 * 100
            scores["sma200"] = _normalizar(dist200, -30, 30)
        except Exception:
            scores["sma200"] = 50
    else:
        scores["sma200"] = 50

    # Tendência 90 dias
    tendencia = dados.get("tendencia_90d", "")
    scores["tendencia"] = 70 if tendencia == "ALTA" else 30

    # Variação do dia (sinal de momentum recente)
    var_dia = dados.get("variacao_dia_pct", 0) or 0
    scores["var_dia"] = _normalizar(var_dia, -5, 5)

    score_final = np.mean(list(scores.values()))
    return round(float(score_final), 1), scores


# ─────────────────────────────────────────────────────────────────
# FATOR 2: VALUATION
# ─────────────────────────────────────────────────────────────────

def calcular_valuation(dados: dict) -> tuple[float, dict]:
    """
    Componentes:
    - P/L trailing: menor é melhor (até certo ponto)
    - Forward P/L: expectativa futura
    - PEG Ratio: < 1 é barato, > 2 é caro
    - P/VP: menor é melhor para valor
    - Upside analistas: maior upside = mais barato vs consenso
    """
    scores = {}

    # P/L Trailing — range típico 10-35 para ações de qualidade
    pe = dados.get("pe_ratio")
    if pe:
        try:
            pe = float(pe)
            if pe <= 0:
                scores["pe"] = 20      # negativo = prejuízo
            elif pe < 10:
                scores["pe"] = 80      # barato (ou setor diferente)
            elif pe < 20:
                scores["pe"] = 70
            elif pe < 30:
                scores["pe"] = 50
            elif pe < 40:
                scores["pe"] = 35
            else:
                scores["pe"] = 15      # caro
        except Exception:
            scores["pe"] = 50
    else:
        scores["pe"] = 50

    # Forward P/L — expectativa futura
    fpe = dados.get("forward_pe")
    if fpe:
        try:
            fpe = float(fpe)
            scores["forward_pe"] = _normalizar(fpe, 40, 10, inverso=True)
        except Exception:
            scores["forward_pe"] = 50
    else:
        scores["forward_pe"] = 50

    # PEG Ratio — < 1 ótimo, 1-2 ok, > 2 caro
    peg = dados.get("peg_ratio")
    if peg:
        try:
            peg = float(peg)
            if peg <= 0:
                scores["peg"] = 50
            elif peg < 0.5:
                scores["peg"] = 90
            elif peg < 1.0:
                scores["peg"] = 75
            elif peg < 1.5:
                scores["peg"] = 55
            elif peg < 2.0:
                scores["peg"] = 35
            else:
                scores["peg"] = 15
        except Exception:
            scores["peg"] = 50
    else:
        scores["peg"] = 50

    # P/VP — menor é melhor para valor
    pb = dados.get("pb_ratio")
    if pb:
        try:
            pb = float(pb)
            scores["pb"] = _normalizar(pb, 10, 1, inverso=True)
        except Exception:
            scores["pb"] = 50
    else:
        scores["pb"] = 50

    # Upside analistas — maior upside = mais valor não precificado
    upside = dados.get("upside_analysts")
    if upside is not None:
        try:
            scores["upside"] = _normalizar(float(upside), -20, 40)
        except Exception:
            scores["upside"] = 50
    else:
        scores["upside"] = 50

    score_final = np.mean(list(scores.values()))
    return round(float(score_final), 1), scores


# ─────────────────────────────────────────────────────────────────
# FATOR 3: QUALIDADE
# ─────────────────────────────────────────────────────────────────

def calcular_qualidade(dados: dict) -> tuple[float, dict]:
    """
    Componentes:
    - ROE: retorno sobre patrimônio (maior = melhor)
    - Margem líquida: eficiência operacional
    - Crescimento de receita: expansão do negócio
    - Crescimento de lucro: qualidade do crescimento
    - Dívida/Equity: alavancagem (menor é mais seguro)
    """
    scores = {}

    # ROE — acima de 15% é bom, acima de 25% é excelente
    roe = dados.get("roe")
    if roe:
        try:
            roe = float(roe) * 100
            scores["roe"] = _normalizar(roe, 0, 40)
        except Exception:
            scores["roe"] = 50
    else:
        scores["roe"] = 50

    # Margem líquida
    margem = dados.get("margem_lucro")
    if margem:
        try:
            margem = float(margem) * 100
            scores["margem"] = _normalizar(margem, -10, 35)
        except Exception:
            scores["margem"] = 50
    else:
        scores["margem"] = 50

    # Crescimento de receita
    cresc_rec = dados.get("crescimento_receita")
    if cresc_rec:
        try:
            cresc_rec = float(cresc_rec) * 100
            scores["cresc_receita"] = _normalizar(cresc_rec, -10, 30)
        except Exception:
            scores["cresc_receita"] = 50
    else:
        scores["cresc_receita"] = 50

    # Crescimento de lucro
    cresc_luc = dados.get("crescimento_lucro")
    if cresc_luc:
        try:
            cresc_luc = float(cresc_luc) * 100
            scores["cresc_lucro"] = _normalizar(cresc_luc, -20, 50)
        except Exception:
            scores["cresc_lucro"] = 50
    else:
        scores["cresc_lucro"] = 50

    # Dívida/Equity — menor é mais seguro (inverso)
    de = dados.get("divida_equity")
    if de:
        try:
            de = float(de)
            scores["divida_equity"] = _normalizar(de, 200, 0, inverso=True)
        except Exception:
            scores["divida_equity"] = 50
    else:
        scores["divida_equity"] = 50

    score_final = np.mean(list(scores.values()))
    return round(float(score_final), 1), scores


# ─────────────────────────────────────────────────────────────────
# FATOR 4: VOLATILIDADE
# ─────────────────────────────────────────────────────────────────

def calcular_volatilidade(dados: dict) -> tuple[float, dict]:
    """
    Componentes:
    - IVR (IV Rank): alta IV = risco de movimento brusco
    - Spread IV-HV: IV muito acima da HV = opções caras, risco elevado
    - Beta: sensibilidade ao mercado
    - Posição na faixa 52 semanas: próximo à mínima = oportunidade

    Nota: score alto = volatilidade FAVORÁVEL para entrar/manter
          score baixo = volatilidade indica risco ou momento ruim
    """
    scores = {}

    # IVR — IV baixa (0-30) é melhor para comprar/manter
    ivr = dados.get("iv_rank_1y")
    if ivr is not None:
        try:
            ivr = float(ivr)
            # IVR baixo = bom momento (vol barata), IVR alto = risco
            scores["ivr"] = _normalizar(ivr, 100, 0, inverso=False)
        except Exception:
            scores["ivr"] = 50
    else:
        scores["ivr"] = 50

    # Spread IV - HV21: IV muito acima da HV sugere evento de risco iminente
    iv = dados.get("iv_atual")
    hv21 = dados.get("hv_21d")
    if iv and hv21:
        try:
            spread = float(iv) - float(hv21)
            # Spread negativo ou zero é saudável
            scores["iv_hv_spread"] = _normalizar(spread, 20, -5, inverso=True)
        except Exception:
            scores["iv_hv_spread"] = 50
    else:
        scores["iv_hv_spread"] = 50

    # Beta — beta < 1 é menos volátil que o mercado
    beta = dados.get("beta")
    if beta:
        try:
            beta = float(beta)
            if beta < 0:
                scores["beta"] = 40    # beta negativo = descorrelacionado
            elif beta < 0.8:
                scores["beta"] = 75    # defensivo
            elif beta < 1.2:
                scores["beta"] = 60    # em linha com mercado
            elif beta < 1.8:
                scores["beta"] = 40    # mais volátil
            else:
                scores["beta"] = 20    # muito volátil
        except Exception:
            scores["beta"] = 50
    else:
        scores["beta"] = 50

    # Posição na faixa 52 semanas
    # Próximo à mínima = possível oportunidade
    pct_low = dados.get("pct_acima_low_52w", 0) or 0
    pct_high = dados.get("pct_abaixo_high_52w", 0) or 0
    try:
        # Ativo próximo da mínima recebe score maior (oportunidade)
        # Ativo próximo da máxima recebe score menor (risco de realização)
        posicao_range = abs(float(pct_high)) / (abs(float(pct_low)) + abs(float(pct_high)) + 0.01)
        scores["range_52w"] = round(posicao_range * 100, 1)
    except Exception:
        scores["range_52w"] = 50

    score_final = np.mean(list(scores.values()))
    return round(float(score_final), 1), scores


# ─────────────────────────────────────────────────────────────────
# SCORE FINAL
# ─────────────────────────────────────────────────────────────────

PESOS = {
    "momentum":    0.25,
    "valuation":   0.25,
    "qualidade":   0.25,
    "volatilidade": 0.25,
}


def calcular_score(ticker: str, dados: dict) -> FactorScore:
    """Calcula score multi-fator completo para um ativo."""

    s_mom, det_mom = calcular_momentum(dados)
    s_val, det_val = calcular_valuation(dados)
    s_qual, det_qual = calcular_qualidade(dados)
    s_vol, det_vol = calcular_volatilidade(dados)

    score_total = (
        s_mom  * PESOS["momentum"] +
        s_val  * PESOS["valuation"] +
        s_qual * PESOS["qualidade"] +
        s_vol  * PESOS["volatilidade"]
    )
    score_total = round(float(score_total), 1)

    # Sinal baseado no score total
    if score_total >= 72:
        sinal = "FORTE COMPRA"
    elif score_total >= 58:
        sinal = "COMPRA"
    elif score_total >= 42:
        sinal = "NEUTRO"
    elif score_total >= 28:
        sinal = "VENDA"
    else:
        sinal = "FORTE VENDA"

    return FactorScore(
        ticker=ticker,
        score_total=score_total,
        score_momentum=s_mom,
        score_valuation=s_val,
        score_qualidade=s_qual,
        score_volatilidade=s_vol,
        sinal=sinal,
        detalhes={
            "momentum": det_mom,
            "valuation": det_val,
            "qualidade": det_qual,
            "volatilidade": det_vol,
        }
    )


def formatar_scores_para_prompt(scores: list[FactorScore]) -> str:
    """Formata scores para incluir no prompt do Claude."""
    linhas = ["=== FACTOR SCORES (0-100) ==="]
    for s in scores:
        barra = "█" * int(s.score_total / 5) + "░" * (20 - int(s.score_total / 5))
        linhas.append(
            f"\n{s.ticker}: [{barra}] {s.score_total}/100 — {s.sinal}\n"
            f"  Momentum: {s.score_momentum:.0f}  |  "
            f"Valuation: {s.score_valuation:.0f}  |  "
            f"Qualidade: {s.score_qualidade:.0f}  |  "
            f"Volatilidade: {s.score_volatilidade:.0f}"
        )
    return "\n".join(linhas)
