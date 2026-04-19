"""
patch_mpt_erc.py
Aplica as modificações necessárias no mpt.py:
  1. Adiciona _portfolio_erc() — Equal Risk Contribution via SLSQP
  2. Adiciona "erc" no retorno de calcular_portfolios_otimos()
  3. Atualiza formatar_mpt_para_prompt() para incluir ERC e pedir recomendação
  4. Atualiza exibir_mpt_terminal() para mostrar ERC
"""

INSERE_ERC_APOS = "    return pesos / pesos.sum()\n"

CODIGO_ERC = '''

def _portfolio_erc(
    cov_matrix: np.ndarray,
    n_ativos: int,
) -> np.ndarray:
    """
    Equal Risk Contribution (ERC) — Roncalli (2013).
    Cada ativo contribui IGUALMENTE para a variância total da carteira.
    Minimiza: sum_i sum_j (RC_i - RC_j)^2
    onde RC_i = w_i * (Cov @ w)_i / (w' Cov w)
    """
    pesos_init = np.ones(n_ativos) / n_ativos
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 0.99)] * n_ativos

    def risk_contributions(w):
        port_var = float(w @ cov_matrix @ w)
        if port_var < 1e-12:
            return np.ones(n_ativos) / n_ativos
        return w * (cov_matrix @ w) / port_var

    def objective(w):
        rc = risk_contributions(w)
        target = 1.0 / n_ativos
        return float(np.sum((rc - target) ** 2))

    resultado = minimize(
        objective, pesos_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
    )

    if resultado.success:
        w = np.clip(resultado.x, 0, None)
        return w / w.sum()
    else:
        # Fallback: Risk Parity simples
        vols = np.sqrt(np.diag(cov_matrix))
        w = 1 / (vols + 1e-8)
        return w / w.sum()

'''

# Onde adicionar ERC no retorno de calcular_portfolios_otimos
INSERE_ERC_CALCULO_APOS = '    w_rp = _portfolio_risk_parity(cov_matrix, n)\n'

CODIGO_ERC_CALCULO = '''    # ── ERC ───────────────────────────────────────────────────────────────
    w_erc = _portfolio_erc(cov_matrix, n)
    r_erc, v_erc, s_erc = metricas_portfolio(w_erc, ret_medios, cov_matrix, rf)
    erc = PortfolioOtimo(
        tipo="erc",
        tickers=tickers,
        pesos=[round(float(w), 4) for w in w_erc],
        retorno=round(r_erc, 4),
        volatilidade=round(v_erc, 4),
        sharpe=round(s_erc, 4),
        pesos_atuais=[round(float(p), 4) for p in pesos_atuais],
        delta_pesos=[round(float(w-p), 4) for w, p in zip(w_erc, pesos_atuais)],
    )

'''

# Troca o return de calcular_portfolios_otimos para incluir erc
RETURN_ANTIGO = '''    return {
        "max_sharpe": max_sharpe,
        "min_variancia": min_variancia,
        "risk_parity": risk_parity,'''

RETURN_NOVO = '''    return {
        "max_sharpe": max_sharpe,
        "min_variancia": min_variancia,
        "risk_parity": risk_parity,
        "erc": erc,'''

# Novo formatar_mpt_para_prompt com ERC e instrução pro Claude
FORMATAR_ANTIGO = 'def formatar_mpt_para_prompt(mpt_resultado: dict) -> str:\n    """Formata insights MPT para incluir no prompt do Claude."""\n    ms = mpt_resultado["max_sharpe"]\n    mv = mpt_resultado["min_variancia"]\n    mc = mpt_resultado["monte_carlo"]'

FORMATAR_NOVO = '''def formatar_mpt_para_prompt(mpt_resultado: dict) -> str:
    """Formata insights MPT para incluir no prompt do Claude."""
    ms = mpt_resultado["max_sharpe"]
    mv = mpt_resultado["min_variancia"]
    mc = mpt_resultado["monte_carlo"]
    rp = mpt_resultado.get("risk_parity")
    erc = mpt_resultado.get("erc")'''

# Adiciona ERC e instrução no texto do prompt
PROMPT_ANTIGO = '    if pares_correlacionados:\n        linhas.append("\\nPares altamente correlacionados (risco de concentração):")'

PROMPT_NOVO = '''    # ERC
    if erc:
        linhas.append(f"\\nPortfólio ERC (Equal Risk Contribution — Roncalli 2013):")
        linhas.append(
            f"  Retorno: {erc.retorno*100:.1f}% | "
            f"Volatilidade: {erc.volatilidade*100:.1f}% | "
            f"Sharpe: {erc.sharpe:.2f}"
        )
        linhas.append("  Realocação sugerida (ERC):")
        for t, p_atual, p_otimo, delta in zip(
            erc.tickers, erc.pesos_atuais, erc.pesos, erc.delta_pesos
        ):
            sinal = "▲" if delta > 0.02 else "▼" if delta < -0.02 else "─"
            linhas.append(
                f"    {sinal} {t}: {p_atual*100:.1f}% → {p_otimo*100:.1f}% "
                f"({delta*100:+.1f}%)"
            )

    # Risk Parity
    if rp:
        linhas.append(f"\\nPortfólio Risk Parity (Qian 2005):")
        linhas.append(
            f"  Retorno: {rp.retorno*100:.1f}% | "
            f"Volatilidade: {rp.volatilidade*100:.1f}% | "
            f"Sharpe: {rp.sharpe:.2f}"
        )

    linhas.append("""
=== INSTRUÇÃO PARA O CLAUDE ===
Com base nas 4 estratégias de otimização calculadas (Máx. Sharpe, Mín. Variância,
Risk Parity e ERC), indique QUAL estratégia melhor se adapta ao perfil e situação
atual deste portfólio, justificando com base em:
1. Nível de concentração atual e necessidade de diversificação
2. Tolerância ao risco implícita nas posições existentes
3. Horizonte de investimento observado
4. Qual estratégia entregaria a melhor relação risco-retorno dado o contexto
Seja direto: recomende UMA estratégia principal e explique por que as outras são
menos adequadas neste momento específico.
""")

    if pares_correlacionados:
        linhas.append("\\nPares altamente correlacionados (risco de concentração):")'''

# Adiciona ERC no terminal display
TERMINAL_ANTIGO = '''    table.add_row(
        "⚖️  Risk Parity",
        f"[{cor(rp.retorno)}]{rp.retorno*100:+.1f}%[/{cor(rp.retorno)}]",
        f"{rp.volatilidade*100:.1f}%",
        f"{rp.sharpe:.2f}",
    )'''

TERMINAL_NOVO = '''    table.add_row(
        "⚖️  Risk Parity",
        f"[{cor(rp.retorno)}]{rp.retorno*100:+.1f}%[/{cor(rp.retorno)}]",
        f"{rp.volatilidade*100:.1f}%",
        f"{rp.sharpe:.2f}",
    )
    erc = mpt_resultado.get("erc")
    if erc:
        table.add_row(
            "🎯 ERC",
            f"[{cor(erc.retorno)}]{erc.retorno*100:+.1f}%[/{cor(erc.retorno)}]",
            f"{erc.volatilidade*100:.1f}%",
            f"{erc.sharpe:.2f}",
        )'''


def aplicar_patches():
    path = "agent/mpt.py"
    with open(path, "r", encoding="utf-8") as f:
        conteudo = f.read()

    # 1. Adiciona função _portfolio_erc após _portfolio_risk_parity
    if "_portfolio_erc" not in conteudo:
        conteudo = conteudo.replace(INSERE_ERC_APOS, INSERE_ERC_APOS + CODIGO_ERC)
        print("✓ _portfolio_erc() adicionada")
    else:
        print("- _portfolio_erc() já existe")

    # 2. Adiciona cálculo de ERC em calcular_portfolios_otimos
    if '"erc": erc,' not in conteudo:
        conteudo = conteudo.replace(
            INSERE_ERC_CALCULO_APOS,
            INSERE_ERC_CALCULO_APOS + CODIGO_ERC_CALCULO
        )
        conteudo = conteudo.replace(RETURN_ANTIGO, RETURN_NOVO)
        print("✓ ERC adicionado ao calcular_portfolios_otimos()")
    else:
        print("- ERC já está em calcular_portfolios_otimos()")

    # 3. Atualiza formatar_mpt_para_prompt
    if 'erc = mpt_resultado.get("erc")' not in conteudo:
        conteudo = conteudo.replace(FORMATAR_ANTIGO, FORMATAR_NOVO)
        conteudo = conteudo.replace(PROMPT_ANTIGO, PROMPT_NOVO)
        print("✓ formatar_mpt_para_prompt() atualizado com ERC e instrução Claude")
    else:
        print("- formatar_mpt_para_prompt() já atualizado")

    # 4. Atualiza exibir_mpt_terminal
    if '"erc")' not in conteudo or "🎯 ERC" not in conteudo:
        conteudo = conteudo.replace(TERMINAL_ANTIGO, TERMINAL_NOVO)
        print("✓ exibir_mpt_terminal() atualizado com ERC")
    else:
        print("- exibir_mpt_terminal() já atualizado")

    with open(path, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(f"\n✅ {path} atualizado com sucesso!")


if __name__ == "__main__":
    aplicar_patches()
