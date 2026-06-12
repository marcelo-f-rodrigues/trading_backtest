"""
run_all.py
-----------
Script principal: executa todas as estratégias em todos os ativos
e gera o relatório completo de rankings por perfil de investidor.

Uso:
    python run_all.py
    python run_all.py --assets BTCUSD XAUUSD
    python run_all.py --no-charts
"""

import argparse
import os
import yaml
import pandas as pd
from tqdm import tqdm

from data.loader import DataLoader
from strategies.trend_following.moving_average import (
    MovingAverageCrossover, MovingAverageFilter, TripleMovingAverage
)
from strategies.trend_following.donchian import DonchianBreakout
from strategies.trend_following.momentum import PriceReturnMomentum, RateOfChange
from strategies.mean_reversion.rsi_strategy import RSIReversion, RSIBands
from strategies.mean_reversion.bollinger import BollingerReversion, ZScoreReversion
from strategies.hybrid.trend_reversion import TrendFilteredReversion, MomentumReversion
from backtest.engine import BacktestEngine
from metrics.calculator import MetricsCalculator
from benchmark_dca import simulate_dca_benchmark
from reporting.ranker import build_comparison_table, full_ranking, export_report


# ---------------------------------------------------------------------------
# Registro de estratégias
# ---------------------------------------------------------------------------

def get_strategies():
    """Retorna a lista completa de estratégias a serem testadas."""
    return [
        # Trend Following
        MovingAverageCrossover(fast=20, slow=200, ma_type="sma"),
        MovingAverageCrossover(fast=50, slow=200, ma_type="sma"),
        MovingAverageCrossover(fast=20, slow=200, ma_type="ema"),
        MovingAverageFilter(period=200, ma_type="sma"),
        MovingAverageFilter(period=50,  ma_type="ema"),
        TripleMovingAverage(short=10, mid=50, long=200),
        DonchianBreakout(entry_period=20, exit_period=10),
        DonchianBreakout(entry_period=55, exit_period=20),
        PriceReturnMomentum(lookback=252, threshold=0.0),
        PriceReturnMomentum(lookback=126, threshold=0.05),
        RateOfChange(period=20, threshold=0.0),

        # Mean Reversion
        RSIReversion(period=14, oversold=30, overbought=70),
        RSIReversion(period=14, oversold=25, overbought=75),
        RSIBands(period=14, buy_level=20, exit_level=50),
        BollingerReversion(period=20, std_dev=2.0, exit_at="mean"),
        BollingerReversion(period=20, std_dev=2.5, exit_at="upper"),
        ZScoreReversion(period=60, entry_z=2.0, exit_z=0.0),

        # Híbridas
        TrendFilteredReversion(trend_period=200, rsi_period=14, oversold=35, overbought=65),
        MomentumReversion(momentum_period=252, reversion_period=21, reversion_threshold=-0.05),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_analysis_periods(config_path: str = "config.yaml"):
    """Lê as janelas temporais configuradas para análise comparativa."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    windows = cfg.get("temporal_windows", [])
    if not windows:
        return [("full", None, None)]
    return [(w.get("name", "full"), w.get("start"), w.get("end")) for w in windows]


def main():
    parser = argparse.ArgumentParser(description="Trading Backtest Framework — Run All")
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Ativos específicos (padrão: todos disponíveis)")
    parser.add_argument("--no-charts", action="store_true",
                        help="Não gerar gráficos individuais")
    parser.add_argument("--config", default="config.yaml", help="Caminho do config")
    args = parser.parse_args()

    print("=" * 70)
    print("  TRADING BACKTEST FRAMEWORK")
    print("=" * 70)

    # Carregar dados
    loader = DataLoader(config_path=args.config)
    periods = get_analysis_periods(args.config)
    strategies = get_strategies()
    all_metrics = []

    if not periods:
        periods = [("full", None, None)]

    total = len(periods) * len(strategies) * len(loader.config.get("assets", []))
    with tqdm(total=total, desc="Backtests") as pbar:
        for period_name, start, end in periods:
            asset_data = loader.load_all(assets=args.assets, start=start, end=end)
            if not asset_data:
                print(f"Nenhum dado carregado para o período {period_name}. Verifique a pasta data/raw/")
                continue

            for strategy in strategies:
                for asset, df in asset_data.items():
                    pbar.set_description(f"{period_name} | {strategy.name} | {asset}")

                    try:
                        engine = BacktestEngine(df, strategy, asset=asset, config_path=args.config)
                        result = engine.run()
                        calc   = MetricsCalculator(result, n_parameters=strategy.n_parameters)
                        m      = calc.compute()
                        m.strategy_name = strategy.name
                        m.period = period_name
                        all_metrics.append(m)

                        if not args.no_charts:
                            from reporting.charts import plot_equity_curve
                            chart_path = f"results/charts/{period_name}/{asset}/{strategy.name}.png"
                            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
                            plot_equity_curve(result, save_path=chart_path, show=False)
                            plt.close("all")

                    except Exception as e:
                        print(f"\n[ERRO] {period_name} | {strategy.name} | {asset}: {e}")

                    pbar.update(1)

            # Benchmarks DCA adicionais por período
            for frequency, name in (("W", "DCA_Weekly"), ("M", "DCA_Monthly")):
                for asset, df in asset_data.items():
                    try:
                        result = simulate_dca_benchmark(df, asset=asset, frequency=frequency, initial_capital=10_000.0)
                        calc = MetricsCalculator(result, n_parameters=2)
                        m = calc.compute()
                        m.strategy_name = name
                        m.period = period_name
                        all_metrics.append(m)
                    except Exception as exc:
                        print(f"[ERRO] Benchmark {period_name} | {name} | {asset}: {exc}")
                    pbar.update(1)

    if not all_metrics:
        print("Nenhuma métrica calculada.")
        return

    # Tabela comparativa
    df_metrics = build_comparison_table(all_metrics)

    # Rankings por perfil
    rankings = full_ranking(df_metrics, save_dir="results/rankings")

    # Exportar relatórios
    export_report(df_metrics, rankings, output_dir="results/reports")

    print("\n" + "=" * 70)
    print("  Análise concluída. Resultados em results/")
    print("=" * 70)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")   # Sem GUI em modo batch
    import matplotlib.pyplot as plt
    main()
