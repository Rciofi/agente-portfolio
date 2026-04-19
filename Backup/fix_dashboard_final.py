path = "agent/dashboard.py"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Remove linhas de lixo após o return da função adicionar_painel_erc
cleaned = []
skip_until_def = False

for i, line in enumerate(lines):
    if skip_until_def:
        if line.startswith("def ") or line.startswith("class "):
            skip_until_def = False
            cleaned.append(line)
        continue
    
    # Detecta lixo: linha com <script>" após o return caminho_html
    if cleaned and cleaned[-1].strip() == "return caminho_html":
        stripped = line.strip()
        if stripped.startswith('<') or (stripped.startswith('"') and '<' in stripped):
            skip_until_def = True
            continue
    
    cleaned.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.writelines(cleaned)

# Verifica sintaxe
try:
    compile(open(path).read(), path, "exec")
    print("OK - sintaxe correta!")
except SyntaxError as e:
    print(f"Erro na linha {e.lineno}: {e.text}")