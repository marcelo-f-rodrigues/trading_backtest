"""
metrics/robustness.py
----------------------
Análise de robustez em três dimensões:

  1. Cross-Asset    : estratégia funciona em múltiplos ativos?
  2. Temporal       : estratégia é estável em diferentes janelas de tempo?
  3. Paramétrica    : resultados são sensíveis a pequenas mudanças nos parâmetros?
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import Callable, Any
from tqdm import tqdm

from backtest.engine import BacktestEngine
from metrics.calculator import MetricsCalculator
from reporting.export_raw import export_backtest_raw


# ---------------------------------------------------------------------------
# 1. Robustez Cross-Asset
# ---------------------------------------------------------------------------

def cross_asset_robustness(
    strategy_factory: Callable,
    asset_data: dict[str, pd.DataFrame],
    **engine_kwargs,
) -> pd.DataFrame:
    """
    Executa a estratégia em todos os ativos disponíveis.

    Parameters
    ----------
    strategy_factory : callable
        Função que retorna uma instância da estratégia (sem argumentos).
        Ex: lambda: MovingAverageCrossover(20, 200)
    asset_data : dict
        {asset_name: DataFrame OHLCV}
    engine_kwargs : dict
        Argumentos extras para BacktestEngine (commission, slippage, etc.)

    Returns
    -------
    pd.DataFrame com métricas principais por ativo + score de robustez.
    """
    rows = []
    for asset, df in asset_data.items():
        strategy = strategy_factory()
        engine   = BacktestEngine(df, strategy, asset=asset, **engine_kwargs)
        result   = engine.run()
        export_backtest_raw(
            result=result,
            price_df=df,
            strategy_name=result.strategy_name,
            asset=result.asset,
            period="full",
            output_dir="results",
        )
        calc     = MetricsCalculator(result, n_parameters=strategy.n_parameters)
        m        = calc.compute()

        rows.append({
            "asset":           asset,
            "cagr":            m.cagr,
            "sharpe":          m.sharpe_ratio,
            "max_drawdown":    m.max_drawdown,
            "calmar":          m.calmar_ratio,
            "profit_factor":   m.profit_factor,
            "n_trades":        m.n_trades,
            "win_rate":        m.win_rate,
            "profitable":      m.cagr > 0,
            "beats_bh":        _beats_buyhold(result),
            "sharpe_positive": m.sharpe_ratio > 0,
        })

    df_result = pd.DataFrame(rows).set_index("asset")

    # Score de robustez cross-asset
    df_result["cross_asset_score"] = (
        df_result["profitable"].astype(float) * 0.4
        + df_result["sharpe_positive"].astype(float) * 0.3
        + df_result["beats_bh"].astype(float) * 0.3
    )

    # Sumário
    print(f"\n{'='*60}")
    print(f"Robustez Cross-Asset: {df_result['profitable'].sum()}/{len(df_result)} ativos lucrativos")
    print(f"Score médio: {df_result['cross_asset_score'].mean():.2f}")
    print(df_result[["cagr", "sharpe", "max_drawdown", "cross_asset_score"]].round(3))

    return df_result


def _beats_buyhold(result) -> bool:
    if result.equity.empty or result.bh_equity.empty:
        return False
    return float(result.equity.iloc[-1]) > float(result.bh_equity.iloc[-1])


# ---------------------------------------------------------------------------
# 2. Robustez Temporal
# ---------------------------------------------------------------------------

def temporal_robustness(
    strategy_factory: Callable,
    df: pd.DataFrame,
    asset: str = "UNKNOWN",
    windows: list[dict] | None = None,
    **engine_kwargs,
) -> pd.DataFrame:
    """
    Executa a estratégia em diferentes janelas de tempo.

    Parameters
    ----------
    windows : list de dicts com chaves: name, start, end
    """
    if windows is None:
        windows = [
            {"name": "2010-2015", "start": "2010-01-01", "end": "2015-12-31"},
            {"name": "2015-2020", "start": "2015-01-01", "end": "2020-12-31"},
            {"name": "2020-2026", "start": "2020-01-01", "end": "2026-12-31"},
            {"name": "full",      "start": None,          "end": None},
        ]

    rows = []
    for w in windows:
        df_w = df.copy()
        if w["start"]:
            df_w = df_w[df_w.index >= pd.to_datetime(w["start"])]
        if w["end"]:
            df_w = df_w[df_w.index <= pd.to_datetime(w["end"])]

        if len(df_w) < 50:
            continue

        strategy = strategy_factory()
        engine   = BacktestEngine(df_w, strategy, asset=asset, **engine_kwargs)
        result   = engine.run()
        export_backtest_raw(
            result=result,
            price_df=df,
            strategy_name=result.strategy_name,
            asset=result.asset,
            period="full",
            output_dir="results",
        )
        calc     = MetricsCalculator(result, n_parameters=strategy.n_parameters)
        m        = calc.compute()

        rows.append({
            "window":        w["name"],
            "cagr":          m.cagr,
            "sharpe":        m.sharpe_ratio,
            "max_drawdown":  m.max_drawdown,
            "profit_factor": m.profit_factor,
            "n_trades":      m.n_trades,
        })

    df_result = pd.DataFrame(rows).set_index("window")

    # Variação entre janelas (menor variação = mais robusto)
    cagr_std = df_result["cagr"].std()
    sharpe_std = df_result["sharpe"].std()
    df_result.attrs["temporal_stability_cagr"]   = cagr_std
    df_result.attrs["temporal_stability_sharpe"]  = sharpe_std

    print(f"\n{'='*60}")
    print(f"Robustez Temporal — {asset}")
    print(df_result.round(3))
    print(f"Estabilidade CAGR (std): {cagr_std:.4f}  |  Sharpe (std): {sharpe_std:.4f}")

    return df_result


# ---------------------------------------------------------------------------
# 3. Robustez Paramétrica
# ---------------------------------------------------------------------------

def parametric_robustness(
    strategy_class,
    param_grid: dict[str, list],
    df: pd.DataFrame,
    asset: str = "UNKNOWN",
    **engine_kwargs,
) -> pd.DataFrame:
    """
    Varre o espaço de parâmetros e avalia a sensibilidade dos resultados.

    Parameters
    ----------
    strategy_class : class
        Classe da estratégia (não instanciada).
    param_grid : dict
        Ex: {"fast": [10, 20, 50], "slow": [100, 150, 200]}
    df : pd.DataFrame
        Dados OHLCV.

    Returns
    -------
    pd.DataFrame com métricas para cada combinação de parâmetros.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    rows = []
    for combo in tqdm(combinations, desc=f"Param sweep — {asset}"):
        params = dict(zip(keys, combo))
        try:
            strategy = strategy_class(**params)
            engine   = BacktestEngine(df, strategy, asset=asset, **engine_kwargs)
            result   = engine.run()
            export_backtest_raw(
                result=result,
                price_df=df,
                strategy_name=result.strategy_name,
                asset=result.asset,
                period="full",
                output_dir="results",
            )
            calc     = MetricsCalculator(result, n_parameters=len(params))
            m        = calc.compute()

            row = {**params, "cagr": m.cagr, "sharpe": m.sharpe_ratio,
                   "max_drawdown": m.max_drawdown, "n_trades": m.n_trades}
            rows.append(row)
        except Exception as e:
            pass  # Combinação inválida (ex: fast >= slow)

    df_result = pd.DataFrame(rows)

    # Sensibilidade: CV (coeficiente de variação) do Sharpe
    if len(df_result) > 1:
        sharpe_cv = df_result["sharpe"].std() / abs(df_result["sharpe"].mean())
        df_result.attrs["parametric_sensitivity"] = sharpe_cv
        print(f"\nRobustez Paramétrica — {asset}")
        print(f"Combinações testadas: {len(df_result)}")
        print(f"Sharpe médio: {df_result['sharpe'].mean():.3f} ± {df_result['sharpe'].std():.3f}")
        print(f"Sensibilidade (CV): {sharpe_cv:.3f} — {'baixa' if sharpe_cv < 0.5 else 'alta'}")

    return df_result
