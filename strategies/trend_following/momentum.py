"""
strategies/trend_following/momentum.py
---------------------------------------
Estratégias de momentum baseadas em retorno passado.

Implementa:
  - PriceReturnMomentum : compra se retorno dos últimos N dias > threshold
  - RateOfChange        : compra se ROC > threshold
"""

import pandas as pd
from strategies.base import BaseStrategy


class PriceReturnMomentum(BaseStrategy):
    """
    Comprado se o retorno dos últimos `lookback` dias for positivo
    e superar o `threshold` mínimo.

    Parameters
    ----------
    lookback : int
        Janela de retorno (dias).
    threshold : float
        Retorno mínimo para acionar o sinal (ex: 0.05 = 5%).
    """

    def __init__(self, lookback: int = 252, threshold: float = 0.0):
        self.lookback = lookback
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"Momentum_{self.lookback}d_th{int(self.threshold * 100)}pct"

    @property
    def parameters(self) -> dict:
        return {"lookback": self.lookback, "threshold": self.threshold}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        close = df["close"]
        ret = close.pct_change(self.lookback)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[ret > self.threshold] = 1
        signal.iloc[: self.lookback] = 0
        return signal


class RateOfChange(BaseStrategy):
    """
    Rate of Change (ROC): variação percentual em N períodos.
    Comprado se ROC > threshold.

    Parameters
    ----------
    period : int
    threshold : float
    """

    def __init__(self, period: int = 20, threshold: float = 0.0):
        self.period = period
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"ROC_{self.period}_th{int(self.threshold * 100)}"

    @property
    def parameters(self) -> dict:
        return {"period": self.period, "threshold": self.threshold}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        close = df["close"]
        roc = (close - close.shift(self.period)) / close.shift(self.period)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[roc > self.threshold] = 1
        signal.iloc[: self.period] = 0
        return signal
