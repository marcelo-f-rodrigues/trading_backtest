"""
analyze_robustness.py
----------------------
Executa análise completa de robustez para uma estratégia:
  - Cross-asset
  - Temporal (janelas de tempo)
  - Paramétrica (grid search)

Uso:
    python analyze_robustness.py --strategy MA_Cross --asset BTCUSD
"""

import argparse
import matplotlib
matplotlib.use("Agg")

from data.loader import DataLoader
from run_all import get_strategies
from strategies.trend_following.moving_average import MovingAverageCrossover
from strategies.trend_following.donchian import DonchianBreakout
from metrics.robustness import (
    cross_asset_robustness,
    temporal_robustness,
    parametric_robustness,
)


PARAM_GRIDS = {
    "MA_Cross": {
        "class": MovingAverageCrossover,
        "grid": {"fast": [10, 20, 50], "slow": [100, 150, 200]},
    },
    "Donchian": {
        "class": DonchianBreakout,
        "grid": {"entry_period": [20, 55, 100], "exit_period": [10, 20, 40]},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset",    default="BTCUSD", help="Ativo para análise temporal/paramétrica")
    parser.add_argument("--strategy", default="MA_Cross", help="Prefixo da estratégia no PARAM_GRIDS")
    parser.add_argument("--config",   default="config.yaml")
    args = parser.parse_args()

    loader     = DataLoader(config_path=args.config)
    all_assets = loader.load_all()

    # --- 1. Cross-Asset ---
    print("\n[1/3] ROBUSTEZ CROSS-ASSET")
    strat_cfg = PARAM_GRIDS.get(args.strategy, PARAM_GRIDS["MA_Cross"])
    strategy_class = strat_cfg["class"]

    # Usar parâmetros padrão para cross-asset
    factory = lambda: strategy_class()
    df_cross = cross_asset_robustness(factory, all_assets)
    df_cross.to_csv(f"results/reports/robustness_cross_asset_{args.strategy}.csv")

    # --- 2. Temporal ---
    print("\n[2/3] ROBUSTEZ TEMPORAL")
    df_single = loader.load(args.asset)
    df_temporal = temporal_robustness(factory, df_single, asset=args.asset)
    df_temporal.to_csv(f"results/reports/robustness_temporal_{args.strategy}_{args.asset}.csv")

    # --- 3. Paramétrica ---
    print("\n[3/3] ROBUSTEZ PARAMÉTRICA")
    df_param = parametric_robustness(
        strategy_class=strategy_class,
        param_grid=strat_cfg["grid"],
        df=df_single,
        asset=args.asset,
    )
    df_param.to_csv(f"results/reports/robustness_parametric_{args.strategy}_{args.asset}.csv")

    print("\nAnálise de robustez concluída. Resultados em results/reports/")


if __name__ == "__main__":
    main()
