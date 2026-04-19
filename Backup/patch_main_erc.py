"""
patch_main_erc.py
Modifica main.py para:
  1. Importar adicionar_painel_erc do dashboard
  2. Chamar adicionar_painel_erc() após adicionar_paineis_mpt()
  3. Passar mpt_resultado ao prompt do Claude via formatar_mpt_para_prompt()
"""

IMPORT_ANTIGO = "from agent.dashboard import gerar_dashboard, adicionar_paineis_mpt, adicionar_painel_realocacao, adicionar_plano_realocacao_html"

IMPORT_NOVO = "from agent.dashboard import gerar_dashboard, adicionar_paineis_mpt, adicionar_painel_realocacao, adicionar_plano_realocacao_html, adicionar_painel_erc"

# Adiciona mpt_texto ao prompt do Claude
CLAUDE_ANTIGO = """    resultado = analisar_portfolio_com_claude(
        ativos_dados=ativos_dados,
        posicoes=posicoes,
        capital_disponivel=capital_disponivel,
        client=client,
        screener_texto=screener_texto,
    )"""

CLAUDE_NOVO = """    # Adiciona contexto MPT/ERC ao prompt do Claude
    mpt_texto = ""
    if mpt_resultado:
        from agent.mpt import formatar_mpt_para_prompt
        mpt_texto = formatar_mpt_para_prompt(mpt_resultado)

    resultado = analisar_portfolio_com_claude(
        ativos_dados=ativos_dados,
        posicoes=posicoes,
        capital_disponivel=capital_disponivel,
        client=client,
        screener_texto=screener_texto,
        mpt_texto=mpt_texto,
    )"""

# Adiciona chamada ao painel ERC no dashboard
DASH_ANTIGO = """        if mpt_resultado and oportunidades_screener:
            adicionar_painel_realocacao(
                caminho_dash, posicoes, ativos_dados,
                oportunidades_screener, mpt_resultado
            )
        adicionar_plano_realocacao_html(caminho_dash, resultado["analise"])"""

DASH_NOVO = """        if mpt_resultado and oportunidades_screener:
            adicionar_painel_realocacao(
                caminho_dash, posicoes, ativos_dados,
                oportunidades_screener, mpt_resultado
            )
        if mpt_resultado and mpt_resultado.get("erc"):
            adicionar_painel_erc(caminho_dash, mpt_resultado)
        adicionar_plano_realocacao_html(caminho_dash, resultado["analise"])"""

# Mesmo para o bloco dry_run
DASH_ANTIGO_DRY = """            if mpt_resultado and oportunidades_screener:
                adicionar_painel_realocacao(
                    caminho_dash, posicoes, ativos_dados,
                    oportunidades_screener, mpt_resultado
                )
            console.print(f"""

DASH_NOVO_DRY = """            if mpt_resultado and oportunidades_screener:
                adicionar_painel_realocacao(
                    caminho_dash, posicoes, ativos_dados,
                    oportunidades_screener, mpt_resultado
                )
            if mpt_resultado and mpt_resultado.get("erc"):
                adicionar_painel_erc(caminho_dash, mpt_resultado)
            console.print(f"""


def aplicar_patches():
    path = "main.py"
    with open(path, "r", encoding="utf-8") as f:
        conteudo = f.read()

    # 1. Import
    if "adicionar_painel_erc" not in conteudo:
        conteudo = conteudo.replace(IMPORT_ANTIGO, IMPORT_NOVO)
        print("✓ import adicionar_painel_erc adicionado")
    else:
        print("- import já existe")

    # 2. Passa mpt_texto ao Claude
    if "mpt_texto" not in conteudo:
        conteudo = conteudo.replace(CLAUDE_ANTIGO, CLAUDE_NOVO)
        print("✓ mpt_texto adicionado ao prompt do Claude")
    else:
        print("- mpt_texto já existe")

    # 3. Chamada ao painel ERC (bloco normal)
    if "adicionar_painel_erc(caminho_dash" not in conteudo:
        conteudo = conteudo.replace(DASH_ANTIGO, DASH_NOVO)
        conteudo = conteudo.replace(DASH_ANTIGO_DRY, DASH_NOVO_DRY)
        print("✓ adicionar_painel_erc() chamado no main()")
    else:
        print("- adicionar_painel_erc() já chamado")

    with open(path, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(f"\n✅ {path} atualizado com sucesso!")


if __name__ == "__main__":
    aplicar_patches()
