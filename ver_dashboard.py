"""
ver_dashboard.py
================
Abre o dashboard mais recente (ou um específico) via servidor HTTP local.

Uso:
    python ver_dashboard.py                    # abre o mais recente
    python ver_dashboard.py dashboard_XXX.html # abre um específico
    python ver_dashboard.py --porta 9000       # porta customizada
"""

import os
import sys
import glob
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path


def encontrar_dashboard(nome_especifico=None):
    """Encontra o dashboard mais recente ou o especificado."""
    # Procura em outputs/ e na raiz
    padroes = [
        "outputs/dashboard_*.html",
        "dashboard_*.html",
        "outputs/*.html",
    ]

    if nome_especifico:
        # Tenta direto, depois em outputs/
        for caminho in [nome_especifico, f"outputs/{nome_especifico}"]:
            if os.path.exists(caminho):
                return os.path.abspath(caminho)
        print(f"❌ Arquivo não encontrado: {nome_especifico}")
        sys.exit(1)

    # Encontra o mais recente
    candidatos = []
    for padrao in padroes:
        candidatos.extend(glob.glob(padrao))

    # Filtra arquivos que contêm "dashboard_2026" ou "dashboard_2025"
    dashboards = [f for f in candidatos if "dashboard_" in f and f.endswith(".html")]

    if not dashboards:
        print("❌ Nenhum dashboard encontrado.")
        print("   Procurei em: outputs/dashboard_*.html e dashboard_*.html")
        sys.exit(1)

    # Ordena por data de modificação (mais recente primeiro)
    dashboards.sort(key=os.path.getmtime, reverse=True)

    print(f"📊 Dashboards encontrados ({len(dashboards)}):")
    for i, d in enumerate(dashboards[:5]):
        mtime = os.path.getmtime(d)
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime).strftime("%d/%m %H:%M")
        marker = "→ " if i == 0 else "  "
        print(f"   {marker}{d} ({dt})")
    if len(dashboards) > 5:
        print(f"   ... e mais {len(dashboards)-5}")

    return os.path.abspath(dashboards[0])


def servir(caminho_html, porta=8765):
    """Sobe servidor HTTP e abre no browser."""
    pasta   = os.path.dirname(caminho_html)
    arquivo = os.path.basename(caminho_html)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=pasta, **kwargs)
        def log_message(self, fmt, *args):
            pass  # silencia logs

    # Tenta porta, se ocupada tenta próximas
    httpd = None
    porta_usada = porta
    for p in range(porta, porta + 20):
        try:
            httpd = socketserver.TCPServer(("", p), Handler)
            httpd.allow_reuse_address = True
            porta_usada = p
            break
        except OSError:
            continue

    if httpd is None:
        print(f"❌ Nenhuma porta disponível entre {porta} e {porta+19}")
        print(f"   Tentando abrir como file://")
        webbrowser.open(f"file:///{caminho_html}")
        return

    # Thread não-daemon: mantém processo vivo
    t = threading.Thread(target=httpd.serve_forever, daemon=False)
    t.start()

    url = f"http://localhost:{porta_usada}/{arquivo}"
    webbrowser.open(url)

    print(f"\n  ✅ Servidor rodando em: http://localhost:{porta_usada}/")
    print(f"  🌐 Dashboard: {url}")
    print(f"\n  Pressione Ctrl+C para encerrar...\n")

    try:
        t.join()
    except KeyboardInterrupt:
        httpd.shutdown()
        print("\n  Servidor encerrado.")


def main():
    args = sys.argv[1:]

    # Porta customizada
    porta = 8765
    if "--porta" in args:
        idx = args.index("--porta")
        try:
            porta = int(args[idx + 1])
            args = [a for i, a in enumerate(args) if i != idx and i != idx + 1]
        except (IndexError, ValueError):
            pass

    # Nome específico ou mais recente
    nome = args[0] if args else None
    caminho = encontrar_dashboard(nome)

    print(f"\n  📂 Abrindo: {caminho}")
    servir(caminho, porta)


if __name__ == "__main__":
    main()
