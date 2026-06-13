"""
analyze_single.py
------------------
Análise aprofundada de uma estratégia em um único ativo.
Gera equity curve, distribuição de trades, heatmap mensal e radar de perfis.

Uso:
    python analyze_single.py --asset BTCUSD --strategy MA_Cross_SMA_20_200
    python analyze_single.py --asset XAUUSD --strategy Donchian_55_20 --show
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.loader import DataLoader
from run_all import get_strategies
from backtest.engine import BacktestEngine
from metrics.calculator import MetricsCalculator
from reporting.charts import (
    plot_equity_curve,
    plot_trade_distribution,
    plot_monthly_returns_heatmap,
    plot_profile_radar,
)
from reporting.ranker import build_comparison_table, rank_by_profile
from reporting.export_raw import export_backtest_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset",    required=True, help="Ex: BTCUSD")
    parser.add_argument("--strategy", required=True, help="Nome da estratégia (parcial aceito)")
    parser.add_argument("--show", action="store_true", help="Exibir gráficos na tela")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # Encontrar estratégia
    all_strats = get_strategies()
    matches = [s for s in all_strats if args.strategy.lower() in s.name.lower()]
    if not matches:
        print(f"Estratégia '{args.strategy}' não encontrada.")
        print("Estratégias disponíveis:")
        for s in all_strats:
            print(f"  {s.name}")
        return

    strategy = matches[0]
    print(f"Estratégia: {strategy.name}")
    print(f"Ativo:      {args.asset}")

    # Carregar dados
    loader = DataLoader(config_path=args.config)
    df = loader.load(args.asset)

    # Backtest
    engine = BacktestEngine(df, strategy, asset=args.asset, config_path=args.config)
    result = engine.run()

    export_backtest_raw(
        result=result,
        price_df=df,
        strategy_name=result.strategy_name,
        asset=result.asset,
        period="full",
        output_dir="results",
    )

    # Métricas
    calc = MetricsCalculator(result, n_parameters=strategy.n_parameters)
    m    = calc.compute()

    # Imprimir sumário
    print("\n" + "=" * 60)
    print(f"SUMÁRIO DE MÉTRICAS — {strategy.name} | {args.asset}")
    print("=" * 60)
    print(f"Retorno Total:     {m.total_return:.2%}")
    print(f"CAGR:              {m.cagr:.2%}")
    print(f"Sharpe Ratio:      {m.sharpe_ratio:.3f}")
    print(f"Sortino Ratio:     {m.sortino_ratio:.3f}")
    print(f"Calmar Ratio:      {m.calmar_ratio:.3f}")
    print(f"Max Drawdown:      {m.max_drawdown:.2%}")
    print(f"Profit Factor:     {m.profit_factor:.3f}")
    print(f"Nº Trades:         {m.n_trades}")
    print(f"Win Rate:          {m.win_rate:.2%}")
    print(f"Expectancy:        {m.expectancy:.4f}")
    print(f"Payoff Ratio:      {m.payoff_ratio:.3f}")
    print(f"% Meses Positivos: {m.pct_months_positive:.2%}")
    print(f"Trades/Ano:        {m.trades_per_year:.1f}")
    print(f"Fluxo:             {m.flow_classification}")
    print(f"Risco de Ruína:    {m.ruin_classification} ({m.ruin_risk:.2%})")
    print(f"Complexidade:      {m.complexity_label} ({m.complexity_score:.2f})")
    print(f"Pain Index:        {m.pain_index:.2f}")
    print(f"Top 5 trades:      {m.pct_profit_top5:.2%} do lucro total")
    print(f"Gini dos trades:   {m.gini_trades:.3f}")

    show = args.show
    base = f"results/charts/{args.asset}_{strategy.name}"

    # Gráficos
    plot_equity_curve(result, save_path=f"{base}_equity.png", show=show)
    plot_trade_distribution(result, save_path=f"{base}_trades.png", show=show)
    plot_monthly_returns_heatmap(result, save_path=f"{base}_heatmap.png", show=show)

    # Radar por perfil
    df_m = build_comparison_table([m])
    scores = {}
    for p in ["growth", "preservation", "flow", "simplicity", "robustness"]:
        ranked = rank_by_profile(df_m, p)
        scores[p] = float(ranked[f"score_{p}"].iloc[0])
    plot_profile_radar(scores, strategy_name=strategy.name,
                       save_path=f"{base}_radar.png", show=show)

    print(f"\nGráficos salvos em results/charts/")


if __name__ == "__main__":
    main()
