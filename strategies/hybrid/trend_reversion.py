"""
strategies/hybrid/trend_reversion.py
--------------------------------------
Estratégias híbridas que combinam tendência e reversão.

Implementa:
  - TrendFilteredReversion : reversão ativada apenas em regime de tendência
  - MomentumReversion      : momentum de médio prazo + reversão de curto prazo
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy
from strategies.mean_reversion.rsi_strategy import compute_rsi


class TrendFilteredReversion(BaseStrategy):
    """
    Estratégia híbrida: só opera reversão quando há tendência de fundo.

    Lógica:
      - Filtro de tendência: preço acima da MA longa (ex: 200d)
      - Sinal de entrada: RSI < oversold (reversão dentro da tendência)
      - Sinal de saída: RSI > overbought ou preço abaixo da MA longa

    Parameters
    ----------
    trend_period : int
        Período da média de tendência.
    rsi_period : int
    oversold : float
    overbought : float
    """

    def __init__(
        self,
        trend_period: int = 200,
        rsi_period: int = 14,
        oversold: float = 35,
        overbought: float = 65,
    ):
        self.trend_period = trend_period
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return (
            f"TrendRev_MA{self.trend_period}_RSI{self.rsi_period}"
            f"_{int(self.oversold)}_{int(self.overbought)}"
        )

    @property
    def parameters(self) -> dict:
        return {
            "trend_period": self.trend_period,
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        close = df["close"]

        ma_trend = close.rolling(self.trend_period).mean()
        rsi      = compute_rsi(close, self.rsi_period)
        in_trend = close > ma_trend

        signal = pd.Series(0, index=df.index, dtype=int)
        position = 0

        warmup = max(self.trend_period, self.rsi_period)
        for i in range(warmup, len(df)):
            if position == 0 and in_trend.iloc[i] and rsi.iloc[i] < self.oversold:
                position = 1
            elif position == 1 and (rsi.iloc[i] > self.overbought or not in_trend.iloc[i]):
                position = 0
            signal.iloc[i] = position

        return signal


class MomentumReversion(BaseStrategy):
    """
    Momentum de médio prazo + Reversão de curto prazo.

    Lógica:
      - Momentum positivo (retorno 12m > threshold_mom)
      - E preço em queda de curto prazo (retorno 1m < threshold_rev)
      → Compra (pullback em tendência de alta)

    Parameters
    ----------
    momentum_period : int
        Janela de momentum (dias).
    reversion_period : int
        Janela de reversão (dias).
    momentum_threshold : float
        Retorno mínimo de momentum.
    reversion_threshold : float
        Retorno máximo de curto prazo para acionar entrada.
    """

    def __init__(
        self,
        momentum_period: int = 252,
        reversion_period: int = 21,
        momentum_threshold: float = 0.0,
        reversion_threshold: float = -0.05,
    ):
        self.momentum_period = momentum_period
        self.reversion_period = reversion_period
        self.momentum_threshold = momentum_threshold
        self.reversion_threshold = reversion_threshold

    @property
    def name(self) -> str:
        return (
            f"MomRev_{self.momentum_period}m_{self.reversion_period}r"
            f"_th{int(self.reversion_threshold * 100)}"
        )

    @property
    def parameters(self) -> dict:
        return {
            "momentum_period": self.momentum_period,
            "reversion_period": self.reversion_period,
            "momentum_threshold": self.momentum_threshold,
            "reversion_threshold": self.reversion_threshold,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        close = df["close"]

        ret_mom = close.pct_change(self.momentum_period)
        ret_rev = close.pct_change(self.reversion_period)

        signal = pd.Series(0, index=df.index, dtype=int)
        position = 0

        warmup = self.momentum_period
        for i in range(warmup, len(df)):
            strong_mom = ret_mom.iloc[i] > self.momentum_threshold
            pullback   = ret_rev.iloc[i] < self.reversion_threshold

            if position == 0 and strong_mom and pullback:
                position = 1
            elif position == 1 and ret_rev.iloc[i] > 0:
                # Sai quando o curto prazo voltou a ficar positivo
                position = 0
            signal.iloc[i] = position

        return signal
