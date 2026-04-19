"""
embutir_plotly.py
=================
Baixa o Plotly e emite inline em todos os dashboards da pasta outputs/.
Depois disso, os dashboards abrem direto no browser sem servidor HTTP.

Uso:
    python embutir_plotly.py                    # processa todos em outputs/
    python embutir_plotly.py dashboard_XXX.html # processa um específico
"""

import os
import sys
import glob
import urllib.request
from pathlib import Path

PLOTLY_URL = "https://cdn.plot.ly/plotly-2.27.0.min.js"
PLOTLY_TAG = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
CACHE_FILE = Path(__file__).parent / ".plotly_cache.js"


def baixar_plotly() -> str:
    """Baixa o Plotly.js e guarda em cache local."""
    if CACHE_FILE.exists():
        print(f"  📦 Usando cache: {CACHE_FILE}")
        return CACHE_FILE.read_text(encoding="utf-8")

    print(f"  ⬇️  Baixando Plotly de {PLOTLY_URL}...")
    try:
        with urllib.request.urlopen(PLOTLY_URL, timeout=30) as r:
            js = r.read().decode("utf-8")
        CACHE_FILE.write_text(js, encoding="utf-8")
        print(f"  ✅ Plotly baixado ({len(js)//1024}KB) e salvo em cache.")
        return js
    except Exception as e:
        print(f"  ❌ Erro ao baixar Plotly: {e}")
        print(f"     Baixe manualmente: {PLOTLY_URL}")
        print(f"     Salve como: {CACHE_FILE}")
        sys.exit(1)


def embutir(caminho: str, js_plotly: str) -> bool:
    """Substitui o <script src=CDN> pelo script inline no arquivo HTML."""
    html = Path(caminho).read_text(encoding="utf-8")

    if PLOTLY_TAG not in html:
        # Verificar se já foi embutido
        if "plotly-2.27.0" in html and "src=" not in html[:html.find("Plotly")]:
            print(f"  ⏭️  Já embutido: {caminho}")
            return False
        print(f"  ⚠️  Tag CDN não encontrada em: {caminho}")
        return False

    novo_tag = f"<script>\n{js_plotly}\n</script>"
    html_novo = html.replace(PLOTLY_TAG, novo_tag, 1)

    Path(caminho).write_text(html_novo, encoding="utf-8")
    tamanho = Path(caminho).stat().st_size // 1024
    print(f"  ✅ {Path(caminho).name} — {tamanho}KB (agora autônomo)")
    return True


def main():
    args = sys.argv[1:]

    # Encontrar arquivos para processar
    if args and not args[0].startswith("--"):
        arquivos = []
        for a in args:
            for candidate in [a, f"outputs/{a}"]:
                if os.path.exists(candidate):
                    arquivos.append(candidate)
                    break
            else:
                print(f"  ❌ Não encontrado: {a}")
    else:
        # Todos os dashboards em outputs/ e na raiz
        arquivos = (
            glob.glob("outputs/dashboard_*.html") +
            glob.glob("dashboard_*.html")
        )
        # Remove duplicatas mantendo outputs/ primeiro
        vistos = set()
        unicos = []
        for f in arquivos:
            nome = Path(f).name
            if nome not in vistos:
                vistos.add(nome)
                unicos.append(f)
        arquivos = sorted(unicos, key=os.path.getmtime, reverse=True)

    if not arquivos:
        print("❌ Nenhum dashboard encontrado.")
        return

    print(f"\n📊 {len(arquivos)} dashboard(s) encontrado(s).\n")

    # Baixar Plotly (uma vez)
    js_plotly = baixar_plotly()
    print()

    # Processar cada arquivo
    processados = 0
    for arq in arquivos:
        if embutir(arq, js_plotly):
            processados += 1

    print(f"\n✅ {processados} arquivo(s) atualizado(s).")
    print("   Agora você pode abrir os dashboards direto no browser (duplo clique no arquivo).\n")


if __name__ == "__main__":
    main()
