# Trading Backtest Framework

Framework quantitativo para estudo e avaliação sistemática de estratégias de trading em múltiplas classes de ativos.

## Filosofia

O objetivo **não** é encontrar a estratégia com maior retorno histórico.  
O objetivo é identificar **princípios robustos** que sobrevivem a diferentes mercados, regimes e períodos.

## Universo de Ativos

| Ativo     | Descrição               |
|-----------|-------------------------|
| BTCUSD    | Bitcoin / Dólar         |
| BOVA11    | ETF Ibovespa            |
| IVVB11    | ETF S&P 500 (BRL)       |
| USDBRL    | Dólar / Real            |
| XAUUSD    | Ouro / Dólar            |
| WTIUSD    | Petróleo WTI / Dólar    |
| DOLFUT    | Futuro de Dólar         |
| COPPERFUT | Futuro de Cobre         |
| SOYFUT    | Futuro de Soja          |

## Famílias de Estratégias

- **Trend Following**: Médias móveis, Donchian Breakout, Momentum
- **Mean Reversion**: RSI, Bollinger Bands, Z-Score, Drawdown percentual
- **Híbridas**: Combinações de tendência + reversão + filtros de regime

## Estrutura do Projeto

```
trading_backtest/
├── data/
│   ├── raw/               # CSVs originais (um por ativo)
│   └── processed/         # Dados normalizados e prontos para uso
├── strategies/
│   ├── trend_following/   # Estratégias de tendência
│   ├── mean_reversion/    # Estratégias de reversão
│   └── hybrid/            # Estratégias híbridas
├── backtest/              # Engine de backtest com entradas/saídas graduais
├── metrics/               # Cálculo de todas as métricas
├── regimes/               # Detecção e classificação de regimes de mercado
├── reporting/             # Geração de relatórios e rankings por perfil
├── utils/                 # Funções utilitárias
├── notebooks/             # Análises exploratórias
├── results/               # Outputs gerados
│   ├── reports/
│   ├── charts/
│   └── rankings/
└── tests/                 # Testes unitários
```

## Instalação

```bash
git clone https://github.com/marcelo-f-rodrigues/trading_backtest.git
cd trading_backtest
pip install -r requirements.txt
```

## Uso Rápido

```python
# 1. Carregar dados
from data.loader import DataLoader
loader = DataLoader(data_dir="data/raw")
df = loader.load("BTCUSD")

# 2. Executar backtest de uma estratégia
from strategies.trend_following.moving_average import MovingAverageCrossover
from backtest.engine import BacktestEngine
from reporting.export_raw import export_backtest_raw

strategy = MovingAverageCrossover(fast=20, slow=200)
engine = BacktestEngine(df, strategy, gradual_entry=True)
result = engine.run()

export_backtest_raw(
        result=result,
        price_df=df,
        strategy_name=result.strategy_name,
        asset=result.asset,
        period="full",
        output_dir="results",
    )

# 3. Calcular métricas completas
from metrics.calculator import MetricsCalculator
calc = MetricsCalculator(result)
report = calc.full_report()
print(report)
```

## Execução Completa (todos os ativos e estratégias)

```bash
python run_all.py
```

Os resultados são salvos em `results/`.

## Perfis de Avaliação

| Perfil            | Prioridade principal                          |
|-------------------|-----------------------------------------------|
| Crescimento       | CAGR, eficiência do capital                   |
| Preservação       | Drawdown, risco de ruína, recuperação         |
| Fluxo             | Regularidade, frequência de oportunidades     |
| Simplicidade      | Baixa manutenção, poucas decisões             |
| Robustez          | Cross-asset, temporal, paramétrica            |

## Princípios Anti-Overfitting

- Entradas e saídas graduais (sem execução instantânea total)
- Avaliação em múltiplos ativos e períodos
- Análise de sensibilidade paramétrica
- Nenhuma estratégia é otimizada para um único ativo
