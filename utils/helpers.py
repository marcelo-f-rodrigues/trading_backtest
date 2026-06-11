"""
utils/helpers.py
-----------------
Funções utilitárias gerais do framework.
"""

import pandas as pd
import numpy as np
from typing import Optional


def annualized_return(equity: pd.Series, trading_days: int = 252) -> float:
    """Calcula CAGR a partir de uma equity curve normalizada."""
    n = len(equity)
    if n < 2 or equity.iloc[0] == 0:
        return np.nan
    n_years = n / trading_days
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)


def max_drawdown(equity: pd.Series) -> float:
    """Retorna o máximo drawdown (negativo) de uma equity curve."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def sharpe(returns: pd.Series, trading_days: int = 252, risk_free: float = 0.0) -> float:
    """Sharpe Ratio anualizado."""
    excess = returns - risk_free / trading_days
    if excess.std() == 0:
        return np.nan
    return float(excess.mean() / excess.std() * np.sqrt(trading_days))


def resample_ohlcv(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Reamostra um DataFrame OHLCV para frequência maior.

    Parameters
    ----------
    freq : str
        'W' = semanal, 'ME' = mensal, 'QE' = trimestral
    """
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(freq).agg(agg).dropna(subset=["close"])


def print_metrics_table(metrics_dict: dict) -> None:
    """Imprime métricas formatadas em tabela simples."""
    print(f"\n{'Métrica':<35} {'Valor':>12}")
    print("-" * 50)
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            if abs(v) < 10:
                print(f"{k:<35} {v:>12.4f}")
            else:
                print(f"{k:<35} {v:>12.2f}")
        elif isinstance(v, int):
            print(f"{k:<35} {v:>12d}")
        else:
            print(f"{k:<35} {str(v):>12}")
