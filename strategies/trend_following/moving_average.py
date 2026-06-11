"""
strategies/trend_following/moving_average.py
--------------------------------------------
Estratégias baseadas em médias móveis.

Implementa:
  - MovingAverageCrossover  : cruzamento de duas médias (fast/slow)
  - MovingAverageFilter     : filtro simples de preço vs média
  - TripleMovingAverage     : 3 médias (short/mid/long)
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class MovingAverageCrossover(BaseStrategy):
    """
    Sinal de compra quando a média rápida cruza acima da média lenta.
    Sinal de venda (flat) quando a média rápida cruza abaixo da média lenta.

    Parameters
    ----------
    fast : int
        Período da média rápida.
    slow : int
        Período da média lenta.
    ma_type : str
        Tipo de média: 'sma' ou 'ema'.
    """

    def __init__(self, fast: int = 20, slow: int = 200, ma_type: str = "sma"):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) deve ser menor que slow ({slow}).")
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type.lower()

    @property
    def name(self) -> str:
        return f"MA_Cross_{self.ma_type.upper()}_{self.fast}_{self.slow}"

    @property
    def parameters(self) -> dict:
        return {"fast": self.fast, "slow": self.slow, "ma_type": self.ma_type}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        close = df["close"]

        ma_fast = self._ma(close, self.fast)
        ma_slow = self._ma(close, self.slow)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[ma_fast > ma_slow] = 1

        # Remover sinal antes da média lenta ter dados suficientes
        signal.iloc[: self.slow] = 0
        return signal

    def _ma(self, series: pd.Series, period: int) -> pd.Series:
        if self.ma_type == "ema":
            return series.ewm(span=period, adjust=False).mean()
        return series.rolling(period).mean()


class MovingAverageFilter(BaseStrategy):
    """
    Comprado quando preço está acima da média. Flat caso contrário.

    Parameters
    ----------
    period : int
        Período da média.
    ma_type : str
        'sma' ou 'ema'.
    """

    def __init__(self, period: int = 200, ma_type: str = "sma"):
        self.period = period
        self.ma_type = ma_type.lower()

    @property
    def name(self) -> str:
        return f"MA_Filter_{self.ma_type.upper()}_{self.period}"

    @property
    def parameters(self) -> dict:
        return {"period": self.period, "ma_type": self.ma_type}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        close = df["close"]

        if self.ma_type == "ema":
            ma = close.ewm(span=self.period, adjust=False).mean()
        else:
            ma = close.rolling(self.period).mean()

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[close > ma] = 1
        signal.iloc[: self.period] = 0
        return signal


class TripleMovingAverage(BaseStrategy):
    """
    Comprado quando short > mid > long.
    Flat em qualquer outra configuração.

    Parameters
    ----------
    short : int
    mid   : int
    long  : int
    ma_type : str
    """

    def __init__(self, short: int = 10, mid: int = 50, long: int = 200, ma_type: str = "sma"):
        if not (short < mid < long):
            raise ValueError("short < mid < long obrigatório.")
        self.short = short
        self.mid = mid
        self.long = long
        self.ma_type = ma_type.lower()

    @property
    def name(self) -> str:
        return f"Triple_MA_{self.ma_type.upper()}_{self.short}_{self.mid}_{self.long}"

    @property
    def parameters(self) -> dict:
        return {
            "short": self.short,
            "mid": self.mid,
            "long": self.long,
            "ma_type": self.ma_type,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        close = df["close"]

        def ma(s, p):
            if self.ma_type == "ema":
                return s.ewm(span=p, adjust=False).mean()
            return s.rolling(p).mean()

        ma_s = ma(close, self.short)
        ma_m = ma(close, self.mid)
        ma_l = ma(close, self.long)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[(ma_s > ma_m) & (ma_m > ma_l)] = 1
        signal.iloc[: self.long] = 0
        return signal
