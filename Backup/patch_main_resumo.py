"""Adiciona chamada de adicionar_resumo_estrategias ao main.py"""

path = "main.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

if "adicionar_resumo_estrategias" in content:
    print("- main.py já tem adicionar_resumo_estrategias")
else:
    # Adiciona ao import do dashboard
    old_import = "from agent.dashboard import gerar_dashboard, adicionar_paineis_mpt, adicionar_painel_realocacao, adicionar_plano_realocacao_html, adicionar_painel_erc"
    new_import = "from agent.dashboard import gerar_dashboard, adicionar_paineis_mpt, adicionar_painel_realocacao, adicionar_plano_realocacao_html, adicionar_painel_erc, adicionar_resumo_estrategias"
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        print("✓ Import atualizado")
    
    # Adiciona chamada após adicionar_painel_erc
    old_call = "        adicionar_painel_erc(caminho_html, mpt_resultado)"
    new_call = """        adicionar_painel_erc(caminho_html, mpt_resultado)
        adicionar_resumo_estrategias(caminho_html, mpt_resultado)"""
    
    if old_call in content:
        content = content.replace(old_call, new_call)
        print("✓ Chamada adicionar_resumo_estrategias adicionada")
    else:
        print("⚠ Padrão adicionar_painel_erc não encontrado")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

import py_compile
try:
    py_compile.compile(path, doraise=True)
    print("✅ main.py sintaxe OK")
except Exception as e:
    print(f"❌ {e}")
