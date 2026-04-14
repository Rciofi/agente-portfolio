"""
analyzer.py — Motor de análise híbrida usando Claude API
Cruza fundamentais + técnico + notícias + contexto do portfólio
"""

import anthropic
import json
import os
from agent.factors import calcular_score, formatar_scores_para_prompt


def formatar_dados_para_prompt(ativo: dict, posicao: dict, contexto_portfolio: dict) -> str:
    """Prepara o contexto estruturado de um ativo para o Claude."""

    preco_atual = ativo.get("preco_atual", 0)
    preco_medio = posicao.get("preco_medio", 0)
    quantidade = posicao.get("quantidade", 0)
    valor_posicao = preco_atual * quantidade
    resultado_pct = ((preco_atual - preco_medio) / preco_medio * 100) if preco_medio else 0
    resultado_usd = (preco_atual - preco_medio) * quantidade

    noticias_txt = "\n".join([
        f"  - [{n.get('data', '')}] {n.get('titulo', '')}"
        for n in ativo.get("noticias", [])[:5]
    ]) or "  Nenhuma notícia recente disponível."

    insider_txt = str(ativo.get("insider", [])[:3]) if ativo.get("insider") else "Nenhum dado"

    return f"""
=== {ativo.get('ticker', 'N/A')} — {ativo.get('nome', 'N/A')} ===
Setor: {ativo.get('setor', 'N/A')} | Indústria: {ativo.get('industria', 'N/A')}

POSIÇÃO DO INVESTIDOR:
- Quantidade: {quantidade} ações
- Preço médio pago: ${preco_medio:.2f}
- Preço atual: ${preco_atual:.2f}
- Valor total da posição: ${valor_posicao:,.2f}
- Resultado: {resultado_pct:+.2f}% (${resultado_usd:+,.2f})
- % do portfólio total: {posicao.get('peso_portfolio', 0):.1f}%

DADOS TÉCNICOS:
- Variação hoje: {ativo.get('variacao_dia_pct', 0):+.2f}%
- Tendência 90 dias: {ativo.get('tendencia_90d', 'N/A')}
- Média móvel 90d: ${ativo.get('media_movel_90d', 0):.2f}
- RSI (14): {ativo.get('rsi', 'N/A')}
- SMA20: {ativo.get('sma20', 'N/A')} | SMA50: {ativo.get('sma50', 'N/A')} | SMA200: {ativo.get('sma200', 'N/A')}
- Beta: {ativo.get('beta', 'N/A')} | ATR: {ativo.get('atr', 'N/A')}
- 52 semanas: High ${ativo.get('high_52w', 0):.2f} ({ativo.get('pct_abaixo_high_52w', 0):.1f}% abaixo) | Low ${ativo.get('low_52w', 0):.2f} ({ativo.get('pct_acima_low_52w', 0):.1f}% acima)

DADOS FUNDAMENTAIS:
- P/L (trailing): {ativo.get('pe_ratio', 'N/A')} | P/L (forward): {ativo.get('forward_pe', 'N/A')}
- PEG Ratio: {ativo.get('peg_ratio', 'N/A')} | P/VP: {ativo.get('pb_ratio', 'N/A')}
- Dividend Yield: {(ativo.get('dividend_yield') or 0) * 100:.2f}%
- ROE: {(ativo.get('roe') or 0) * 100:.1f}% | ROA: {(ativo.get('roa') or 0) * 100:.1f}%
- Margem Líquida: {(ativo.get('margem_lucro') or 0) * 100:.1f}%
- Crescimento Receita: {(ativo.get('crescimento_receita') or 0) * 100:.1f}%
- Crescimento Lucro: {(ativo.get('crescimento_lucro') or 0) * 100:.1f}%
- Dívida/Equity: {ativo.get('divida_equity', 'N/A')}
- Recomendação analistas: {ativo.get('recomendacao_analistas', 'N/A')}
- Preço alvo analistas: ${ativo.get('target_price', 0) or 0:.2f} (upside: {ativo.get('upside_analysts', 'N/A')}%)

VOLATILIDADE:
- IV Atual (ATM): {ativo.get('iv_atual', 'N/A')}% | Calls: {ativo.get('iv_calls_atm', 'N/A')}% | Puts: {ativo.get('iv_puts_atm', 'N/A')}%
- IV Rank (1 ano): {ativo.get('iv_rank_1y', 'N/A')}/100 — {ativo.get('iv_interpretacao', 'N/A')}
- IV Percentile: {ativo.get('iv_percentile_30d', 'N/A')}%
- HV 10d: {ativo.get('hv_10d', 'N/A')}% | HV 21d: {ativo.get('hv_21d', 'N/A')}% | HV 63d: {ativo.get('hv_63d', 'N/A')}%
- Próximo vencimento de opções: {ativo.get('proximo_vencimento', 'N/A')}

NOTÍCIAS RECENTES:
{noticias_txt}

INSIDER TRADING (últimos):
{insider_txt}
"""


def analisar_portfolio_com_claude(
    ativos_dados: list,
    posicoes: list,
    capital_disponivel: float,
    client: anthropic.Anthropic,
    screener_texto: str = "",
) -> dict:
    """Envia o portfólio completo para o Claude e recebe análise estruturada."""

    # Calcula factor scores
    factor_scores = [calcular_score(a.get("ticker", ""), a) for a in ativos_dados]
    scores_texto = formatar_scores_para_prompt(factor_scores)

    # Calcula valor total do portfólio
    valor_total = sum(
        a.get("preco_atual", 0) * p.get("quantidade", 0)
        for a, p in zip(ativos_dados, posicoes)
    ) + capital_disponivel

    # Adiciona peso de cada posição
    for a, p in zip(ativos_dados, posicoes):
        valor_pos = a.get("preco_atual", 0) * p.get("quantidade", 0)
        p["peso_portfolio"] = (valor_pos / valor_total * 100) if valor_total else 0

    # Monta contexto completo
    contexto_portfolio = {
        "valor_total_portfolio": valor_total,
        "capital_disponivel": capital_disponivel,
        "num_ativos": len(ativos_dados),
    }

    ativos_texto = "\n\n".join([
        formatar_dados_para_prompt(a, p, contexto_portfolio)
        for a, p in zip(ativos_dados, posicoes)
    ])

    system_prompt = """Você é um analista de investimentos sênior especializado em ações americanas e ADRs brasileiros negociados em NYSE/NASDAQ.

Seu papel é analisar portfólios de forma objetiva e prática, fornecendo recomendações acionáveis baseadas em:
1. Análise fundamentalista (valuação, crescimento, qualidade do negócio)
2. Análise técnica (tendência, momentum, RSI, médias móveis)
3. Contexto de notícias e sentimento
4. Gestão de posição (tamanho, concentração, risco)

Você sempre responde em português brasileiro.
Você é direto e prático — não enrola.
Você considera o custo de oportunidade: manter um ativo ruim significa deixar de alocar em algo melhor.
"""

    user_prompt = f"""Analise meu portfólio completo abaixo e forneça recomendações de gestão de posição.

CONTEXTO DO PORTFÓLIO:
- Valor total: ${valor_total:,.2f}
- Capital disponível para novo investimento: ${capital_disponivel:,.2f}
- Número de ativos: {len(ativos_dados)}

FACTOR SCORES (modelo quantitativo 0-100):
{scores_texto}

---

DADOS COMPLETOS DOS ATIVOS:
{ativos_texto}

---

Para cada ativo, forneça:

1. **RECOMENDAÇÃO** (escolha exatamente uma):
   - MANTER POSIÇÃO — tese intacta, sem ação necessária
   - ADICIONAR — bom momento para aumentar posição (indique % ou valor sugerido)
   - REALIZAR PARCIALMENTE — realize X% da posição (explique quanto e por quê)
   - REALIZAR TUDO — sair completamente da posição
   - REDUZIR GRADUALMENTE — venda em parcelas nos próximos dias/semanas

2. **JUSTIFICATIVA** (3-5 linhas objetivas cruzando técnico + fundamentalista)

3. **NÍVEL DE CONVICÇÃO** (Alta / Média / Baixa) e principal risco da recomendação

Ao final, forneça:
- **VISÃO GERAL DO PORTFÓLIO**: concentração, setores, riscos sistêmicos
- **TOP PRIORIDADE**: qual é a ação mais urgente no portfólio hoje?

## 📋 PLANO DE REALOCAÇÃO — AÇÕES CONCRETAS

Consolide TUDO em um plano executável com números exatos.

VENDER (libera capital):
- Ticker | qtd exata | preço atual | valor liberado | % portfólio antes e depois
- TOTAL LIBERADO: $X

COMPRAR (com capital liberado + disponível):
- Ticker | qtd exata | preço atual | valor alocado | % portfólio resultante
- TOTAL ALOCADO: $X

MANTER SEM ALTERAÇÃO:
- Liste ativos sem ação

RESULTADO ESPERADO:
- Distribuição: Brasil X% vs EUA Y%
- Sharpe estimado após realocação
- Número de ativos antes e depois

ORDEM DE EXECUÇÃO:
Numere as operações em sequência lógica.
"""

    print("\n🤖 Claude analisando portfólio...\n")

    # Adiciona screener ao prompt se disponível
    if screener_texto:
        user_prompt = user_prompt + "\n\n" + screener_texto + """

---

Ao final da sua análise, adicione OBRIGATORIAMENTE duas seções:

## 🔍 OPORTUNIDADES DO SCREENER
Analise as oportunidades identificadas pelo screener acima e recomende as TOP 3-5
que melhor complementam o portfólio atual, considerando:
- Diversificação setorial (portfólio é 93% Brasil — priorize diversificação EUA)
- Complementaridade com posições existentes
- Potencial de retorno ajustado ao risco
Para cada sugestão: ticker, por que faz sentido, quanto alocar em USD, nível de convicção.

## 📋 PLANO DE REALOCAÇÃO — AÇÕES CONCRETAS

Consolide TUDO em um plano executável com números exatos.

VENDER (libera capital):
- Ticker | qtd exata | preço atual | valor liberado | % portfólio antes e depois
- TOTAL LIBERADO: $X

COMPRAR (com capital liberado + disponível):
- Ticker | qtd exata | preço atual | valor alocado | % portfólio resultante
- TOTAL ALOCADO: $X (deve ser menor ou igual ao total liberado + disponível)

MANTER SEM ALTERAÇÃO:
- Liste ativos sem ação

RESULTADO ESPERADO:
- Distribuição por setor: Brasil X% vs EUA Y%
- Sharpe estimado após realocação
- Ativos: antes N, depois M

ORDEM DE EXECUÇÃO:
Numere as operações em sequência lógica. O que fazer primeiro.
"""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=6000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    analise_texto = response.content[0].text

    return {
        "factor_scores": factor_scores,
        "analise": analise_texto,
        "valor_total": valor_total,
        "capital_disponivel": capital_disponivel,
        "tokens_usados": response.usage.input_tokens + response.usage.output_tokens,
    }
