# Arquivos corrigidos da auditoria

Substitua estes arquivos na pasta `agent/` do seu projeto:

- `agent/backtest.py`
- `agent/mpt.py`
- `agent/dashboard.py`

## O que foi corrigido

### backtest.py
- Portfólio agregado por curva histórica real de valor
- Benchmark no S&P 500 com os mesmos aportes e datas
- Sharpe / drawdown do portfólio via curva agregada
- Retornos mensais e anuais calculados da curva real do portfólio

### mpt.py
- Retornos simples diários
- Expected returns com shrinkage + clipping conservador
- Penalidade de turnover e concentração
- Separação mais clara entre métrica esperada e realizada
- Validação OOS mantida e incorporada ao resultado

### dashboard.py
- Rótulos mais corretos para não confundir retorno esperado com realizado
- Bloco OOS no dashboard quando existir
- Ajuste de cores em células com retorno zero
- Benchmark renomeado para refletir aportes equivalentes
