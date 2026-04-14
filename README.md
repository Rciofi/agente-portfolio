# 📊 Agente de Gestão de Portfólio

> Agente inteligente de análise híbrida para gestão ativa de portfólio de ações
> NYSE · NASDAQ · ADRs brasileiros · Powered by Claude API

---

## O que faz

Para cada ativo no portfólio, o agente:

- Coleta fundamentos, dados técnicos e notícias (FinViz + yfinance)
- Calcula **Volatilidade Implícita (IV)** da cadeia de opções ATM
- Calcula **IV Rank (IVR)** e **IV Percentile** vs histórico de 1 ano
- Calcula **Volatilidade Histórica Realizada** (HV 10d / 21d / 63d)
- Exibe gráfico ASCII de preço dos últimos 90 dias no terminal
- Mantém **histórico de snapshots** para comparação entre execuções
- Avalia **acurácia das recomendações anteriores** (janela de 7 dias)
- Envia tudo ao **Claude** para análise híbrida (fundamentalista + técnica + vol)
- Gera recomendações de gestão de posição com justificativa
- Exporta relatório completo em `outputs/`

---

## Estrutura

```
agente_portfolio/
├── main.py
├── portfolio.csv
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── agent/
│   ├── collector.py
│   ├── volatility.py
│   ├── analyzer.py
│   ├── accuracy.py
│   ├── reporter.py
│   └── brokers/
│       ├── base.py
│       ├── interactive_brokers.py   ← IB / Avenue (flag: BROKER_IB_ENABLED)
│       ├── btg.py                   ← BTG Pactual (flag: BROKER_BTG_ENABLED)
│       └── manager.py
├── data/                            ← auto-gerado
└── outputs/                         ← auto-gerado
```

---

## Setup

```bash
git clone https://github.com/seu-usuario/agente-portfolio.git
cd agente-portfolio
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # adicione sua ANTHROPIC_API_KEY
python main.py
```

---

## Modos de uso

```bash
python main.py                    # análise completa
python main.py meu_portfolio.csv  # portfólio customizado
python main.py --dry-run          # só dados, sem Claude
```

---

## Ativar corretoras (desativadas por padrão)

No `.env`:

```env
BROKER_IB_ENABLED=true   # Interactive Brokers / Avenue
BROKER_BTG_ENABLED=true  # BTG Pactual
```

> Ative somente após validar acurácia. Comece em paper trading (IB_PORT=7497).

---

## Aviso legal

Para fins educacionais. Não constitui recomendação de investimento.

## Licença

MIT
