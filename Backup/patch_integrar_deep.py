"""
patch_integrar_deep.py
Integra DLS e DeepStatArb ao pipeline real do agente:
  1. mpt.py  → calcular_portfolios_otimos() passa a incluir dls e deepstatarb
  2. dashboard.py → painel ERC atualizado para mostrar 6 estratégias
  3. main.py → passa .env flag DEEP_LEARNING_ENABLED para controlar treinamento
"""

import os
import re


# ─────────────────────────────────────────────────────────────────────
# 1. mpt.py — adiciona deep portfolios ao retorno
# ─────────────────────────────────────────────────────────────────────

def patch_mpt():
    path = "agent/mpt.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "deep_portfolios" in content:
        print("- mpt.py já tem deep_portfolios")
        return

    # Adiciona import no topo
    old_import = "from scipy.optimize import minimize"
    new_import = "from scipy.optimize import minimize\n\ntry:\n    from agent.deep_portfolios import calcular_portfolios_deep, formatar_deep_para_prompt\n    DEEP_AVAILABLE = True\nexcept ImportError:\n    DEEP_AVAILABLE = False"

    content = content.replace(old_import, new_import, 1)

    # Adiciona deep ao retorno de calcular_portfolios_otimos
    old_return = '    return {\n        "max_sharpe": max_sharpe,\n        "min_variancia": min_variancia,\n        "risk_parity": risk_parity,\n        "erc": erc,'
    new_return = '''    # ── Deep Learning Portfolios ─────────────────────────────────────
    deep_resultado = {}
    deep_enabled = os.environ.get("DEEP_LEARNING_ENABLED", "true").lower() == "true"
    if DEEP_AVAILABLE and deep_enabled and len(tickers) >= 3:
        try:
            deep_resultado = calcular_portfolios_deep(retornos, pesos_atuais, rf)
        except Exception as e:
            print(f"  ⚠ Deep Learning: {e}")

    return {
        "max_sharpe": max_sharpe,
        "min_variancia": min_variancia,
        "risk_parity": risk_parity,
        "erc": erc,
        "dls": deep_resultado.get("dls"),
        "deepstatarb": deep_resultado.get("deepstatarb"),'''

    content = content.replace(old_return, new_return, 1)

    # Adiciona import os se não existir
    if "import os" not in content:
        content = "import os\n" + content

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✓ mpt.py atualizado com deep portfolios")


# ─────────────────────────────────────────────────────────────────────
# 2. dashboard.py — expande painel ERC para 6 estratégias
# ─────────────────────────────────────────────────────────────────────

def patch_dashboard():
    path = "agent/dashboard.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "deepstatarb" in content:
        print("- dashboard.py já tem deepstatarb")
        return

    # Atualiza adicionar_painel_erc para incluir DLS e DeepStatArb
    old_pesos = '''    pesos_atual = [round(p * 100, 1) for p in ms.pesos_atuais]
    pesos_ms    = [round(p * 100, 1) for p in ms.pesos]
    pesos_mv    = [round(p * 100, 1) for p in mv.pesos]
    pesos_rp    = [round(p * 100, 1) for p in rp.pesos]
    pesos_erc   = [round(p * 100, 1) for p in erc.pesos] if erc else pesos_rp

    estrategias = ["Atual", "Máx. Sharpe", "Mín. Variância", "Risk Parity", "ERC"]
    sharpes = [
        round(mc.sharpe_atual, 3),
        round(ms.sharpe, 3),
        round(mv.sharpe, 3),
        round(rp.sharpe, 3),
        round(erc.sharpe, 3) if erc else 0,
    ]
    retornos = [
        round(mc.retorno_atual * 100, 1),
        round(ms.retorno * 100, 1),
        round(mv.retorno * 100, 1),
        round(rp.retorno * 100, 1),
        round(erc.retorno * 100, 1) if erc else 0,
    ]
    vols = [
        round(mc.vol_atual * 100, 1),
        round(ms.volatilidade * 100, 1),
        round(mv.volatilidade * 100, 1),
        round(rp.volatilidade * 100, 1),
        round(erc.volatilidade * 100, 1) if erc else 0,
    ]
    cores_estrategias = ["#7090b0", "#00c176", "#00b4d8", "#a78bfa", "#fb923c"]'''

    new_pesos = '''    dls = mpt_resultado.get("dls")
    dsa = mpt_resultado.get("deepstatarb")

    pesos_atual = [round(p * 100, 1) for p in ms.pesos_atuais]
    pesos_ms    = [round(p * 100, 1) for p in ms.pesos]
    pesos_mv    = [round(p * 100, 1) for p in mv.pesos]
    pesos_rp    = [round(p * 100, 1) for p in rp.pesos]
    pesos_erc   = [round(p * 100, 1) for p in erc.pesos] if erc else pesos_rp
    pesos_dls   = [round(p * 100, 1) for p in dls.pesos] if dls else pesos_rp
    pesos_dsa   = [round(p * 100, 1) for p in dsa.pesos] if dsa else pesos_rp

    estrategias = ["Atual", "Máx. Sharpe", "Mín. Variância", "Risk Parity", "ERC", "DLS", "DeepStatArb"]
    sharpes = [
        round(mc.sharpe_atual, 3),
        round(ms.sharpe, 3),
        round(mv.sharpe, 3),
        round(rp.sharpe, 3),
        round(erc.sharpe, 3) if erc else 0,
        round(dls.sharpe, 3) if dls else 0,
        round(dsa.sharpe, 3) if dsa else 0,
    ]
    retornos = [
        round(mc.retorno_atual * 100, 1),
        round(ms.retorno * 100, 1),
        round(mv.retorno * 100, 1),
        round(rp.retorno * 100, 1),
        round(erc.retorno * 100, 1) if erc else 0,
        round(dls.retorno * 100, 1) if dls else 0,
        round(dsa.retorno * 100, 1) if dsa else 0,
    ]
    vols = [
        round(mc.vol_atual * 100, 1),
        round(ms.volatilidade * 100, 1),
        round(mv.volatilidade * 100, 1),
        round(rp.volatilidade * 100, 1),
        round(erc.volatilidade * 100, 1) if erc else 0,
        round(dls.volatilidade * 100, 1) if dls else 0,
        round(dsa.volatilidade * 100, 1) if dsa else 0,
    ]
    cores_estrategias = ["#7090b0", "#00c176", "#00b4d8", "#a78bfa", "#fb923c", "#38bdf8", "#f472b6"]'''

    if old_pesos in content:
        content = content.replace(old_pesos, new_pesos)
        print("✓ dashboard.py atualizado com DLS e DeepStatArb")
    else:
        print("⚠ dashboard.py: padrão não encontrado, verifique manualmente")

    # Atualiza o gráfico de alocação para incluir DLS e DeepStatArb
    old_bars = '''  {{
    type: 'bar', name: 'ERC',
    x: {json.dumps(tickers)}, y: {json.dumps(pesos_erc)},
    marker: {{ color: '#fb923c' }},
    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>ERC</extra>',
  }},
], {{'''

    new_bars = '''  {{
    type: 'bar', name: 'ERC',
    x: {json.dumps(tickers)}, y: {json.dumps(pesos_erc)},
    marker: {{ color: '#fb923c' }},
    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>ERC</extra>',
  }},
  {{
    type: 'bar', name: 'DLS (LSTM)',
    x: {json.dumps(tickers)}, y: {json.dumps(pesos_dls)},
    marker: {{ color: '#38bdf8' }},
    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>DLS</extra>',
  }},
  {{
    type: 'bar', name: 'DeepStatArb',
    x: {json.dumps(tickers)}, y: {json.dumps(pesos_dsa)},
    marker: {{ color: '#f472b6' }},
    hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>DeepStatArb</extra>',
  }},
], {{'''

    if old_bars in content:
        content = content.replace(old_bars, new_bars)
        print("✓ Barras DLS/DeepStatArb adicionadas ao gráfico de alocação")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ─────────────────────────────────────────────────────────────────────
# 3. main.py — passa deep results ao prompt do Claude
# ─────────────────────────────────────────────────────────────────────

def patch_main():
    path = "main.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "formatar_deep_para_prompt" in content:
        print("- main.py já tem deep prompt")
        return

    old_mpt_texto = '''    # Adiciona contexto MPT/ERC ao prompt do Claude
    mpt_texto = ""
    if mpt_resultado:
        from agent.mpt import formatar_mpt_para_prompt
        mpt_texto = formatar_mpt_para_prompt(mpt_resultado)'''

    new_mpt_texto = '''    # Adiciona contexto MPT/ERC/Deep Learning ao prompt do Claude
    mpt_texto = ""
    if mpt_resultado:
        from agent.mpt import formatar_mpt_para_prompt
        mpt_texto = formatar_mpt_para_prompt(mpt_resultado)
        # Adiciona resultados de Deep Learning se disponíveis
        if mpt_resultado.get("dls") or mpt_resultado.get("deepstatarb"):
            from agent.deep_portfolios import formatar_deep_para_prompt
            mpt_texto += formatar_deep_para_prompt(mpt_resultado)'''

    if old_mpt_texto in content:
        content = content.replace(old_mpt_texto, new_mpt_texto)
        print("✓ main.py atualizado com deep prompt")
    else:
        print("⚠ main.py: padrão não encontrado")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ─────────────────────────────────────────────────────────────────────
# EXECUTA
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Aplicando patches de integração Deep Learning...\n")
    patch_mpt()
    patch_dashboard()
    patch_main()

    # Verifica sintaxe dos 3 arquivos
    import py_compile, sys
    erros = []
    for f in ["agent/mpt.py", "agent/dashboard.py", "main.py"]:
        try:
            py_compile.compile(f, doraise=True)
            print(f"✅ {f} — sintaxe OK")
        except py_compile.PyCompileError as e:
            print(f"❌ {f} — {e}")
            erros.append(f)

    if not erros:
        print("\n✅ Integração concluída! Rode: python main.py")
    else:
        print(f"\n❌ {len(erros)} arquivo(s) com erro de sintaxe")
        sys.exit(1)
