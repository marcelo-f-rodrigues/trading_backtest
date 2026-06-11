"""
strategies/mean_reversion/rsi_strategy.py
------------------------------------------
Estratégias baseadas em RSI (Relative Strength Index).

Implementa:
  - RSIReversion  : compra em sobrevenda, sai em sobrecompra
  - RSIBands      : variante com bandas assimétricas
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calcula o RSI padrão de Wilder."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class RSIReversion(BaseStrategy):
    """
    Estratégia de reversão à média via RSI.

    Compra quando RSI < oversold.
    Sai quando RSI > overbought.

    Parameters
    ----------
    period : int
        Período do RSI.
    oversold : float
        Nível de sobrevenda (ex: 30).
    overbought : float
        Nível de sobrecompra (ex: 70).
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return f"RSI_Rev_{self.period}_{int(self.oversold)}_{int(self.overbought)}"

    @property
    def parameters(self) -> dict:
        return {
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        rsi = compute_rsi(df["close"], self.period)

        signal = pd.Series(0, index=df.index, dtype=int)
        position = 0

        for i in range(self.period, len(df)):
            if position == 0 and rsi.iloc[i] < self.oversold:
                position = 1
            elif position == 1 and rsi.iloc[i] > self.overbought:
                position = 0
            signal.iloc[i] = position

        return signal


class RSIBands(BaseStrategy):
    """
    Variante com bandas assimétricas: compra em oversold extremo,
    sai em nível neutro.

    Parameters
    ----------
    period : int
    buy_level : float
        RSI de compra (ex: 20).
    exit_level : float
        RSI de saída (ex: 50).
    """

    def __init__(self, period: int = 14, buy_level: float = 20, exit_level: float = 50):
        self.period = period
        self.buy_level = buy_level
        self.exit_level = exit_level

    @property
    def name(self) -> str:
        return f"RSI_Bands_{self.period}_{int(self.buy_level)}_{int(self.exit_level)}"

    @property
    def parameters(self) -> dict:
        return {
            "period": self.period,
            "buy_level": self.buy_level,
            "exit_level": self.exit_level,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        rsi = compute_rsi(df["close"], self.period)

        signal = pd.Series(0, index=df.index, dtype=int)
        position = 0

        for i in range(self.period, len(df)):
            if position == 0 and rsi.iloc[i] < self.buy_level:
                position = 1
            elif position == 1 and rsi.iloc[i] > self.exit_level:
                position = 0
            signal.iloc[i] = position

        return signal
