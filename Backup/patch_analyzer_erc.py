"""
patch_analyzer_erc.py
Modifica agent/analyzer.py para:
  1. Aceitar mpt_texto como parâmetro em analisar_portfolio_com_claude()
  2. Incluir o contexto MPT/ERC no prompt do Claude
  3. Instruir Claude a recomendar qual estratégia seguir
"""

# Assinatura antiga
ASSINATURA_ANTIGA = """def analisar_portfolio_com_claude(
    ativos_dados: list,
    posicoes: list,
    capital_disponivel: float,
    client: anthropic.Anthropic,
    screener_texto: str = "",
) -> dict:"""

ASSINATURA_NOVA = """def analisar_portfolio_com_claude(
    ativos_dados: list,
    posicoes: list,
    capital_disponivel: float,
    client: anthropic.Anthropic,
    screener_texto: str = "",
    mpt_texto: str = "",
) -> dict:"""

# Onde o user_prompt é finalizado — adiciona seção MPT/ERC
USER_PROMPT_ANTIGO = '    user_prompt = f"""Analise meu portfólio completo abaixo e forneça recomendações de gestão de posição.\n\nCONTEXTO DO PORTFÓLIO:\n- Valor total: ${valor_total:,.2f}\n- Capital disponível para novo investimento: ${capital_disponivel:,.2f}'

USER_PROMPT_NOVO = '    mpt_secao = f"\\n\\n{mpt_texto}" if mpt_texto else ""\n\n    user_prompt = f"""Analise meu portfólio completo abaixo e forneça recomendações de gestão de posição.\n\nCONTEXTO DO PORTFÓLIO:\n- Valor total: ${valor_total:,.2f}\n- Capital disponível para novo investimento: ${capital_disponivel:,.2f}'


def encontrar_fim_user_prompt(conteudo: str) -> tuple[str, str]:
    """Encontra onde o user_prompt é fechado e adiciona mpt_secao."""
    # Procura pela última ocorrência do fechamento do f-string do user_prompt
    # que termina antes da chamada ao client.messages.create
    marcador = 'client.messages.create'
    idx = conteudo.find(marcador)
    if idx == -1:
        return conteudo, False

    # Encontra o trecho do user_prompt antes do client.messages.create
    trecho_antes = conteudo[:idx]

    # O user_prompt termina com """ antes de ser usado
    # Adiciona {mpt_secao} antes do fechamento do f-string
    if '{mpt_secao}' not in trecho_antes:
        # Encontra o padrão de fechamento do prompt de usuário
        fechamento = '"""\n\n    resposta'
        if fechamento in trecho_antes:
            novo = '\\n{mpt_secao}"""\n\n    resposta'
            conteudo = conteudo.replace(fechamento, novo, 1)
            return conteudo, True

        # Tenta outro padrão comum
        fechamento2 = '"""\n    resposta'
        if fechamento2 in trecho_antes:
            novo2 = '\\n{mpt_secao}"""\n    resposta'
            conteudo = conteudo.replace(fechamento2, novo2, 1)
            return conteudo, True

    return conteudo, False


def aplicar_patches():
    path = "agent/analyzer.py"
    with open(path, "r", encoding="utf-8") as f:
        conteudo = f.read()

    # 1. Adiciona mpt_texto como parâmetro
    if "mpt_texto" not in conteudo:
        conteudo = conteudo.replace(ASSINATURA_ANTIGA, ASSINATURA_NOVA)
        print("✓ mpt_texto adicionado como parâmetro")
    else:
        print("- mpt_texto já é parâmetro")

    # 2. Adiciona mpt_secao no user_prompt
    if "mpt_secao" not in conteudo:
        conteudo = conteudo.replace(USER_PROMPT_ANTIGO, USER_PROMPT_NOVO)
        conteudo, ok = encontrar_fim_user_prompt(conteudo)
        if ok:
            print("✓ mpt_secao injetado no user_prompt")
        else:
            print("⚠ Não foi possível injetar mpt_secao automaticamente")
            print("  Adicione manualmente {mpt_secao} no final do user_prompt")
    else:
        print("- mpt_secao já existe")

    with open(path, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(f"\n✅ {path} atualizado com sucesso!")


if __name__ == "__main__":
    aplicar_patches()
