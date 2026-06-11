"""
strategies/trend_following/donchian.py
--------------------------------------
Donchian Channel Breakout — clássico sistema turtle.

Compra quando o preço fecha acima da máxima dos últimos N dias.
Sai quando o preço fecha abaixo da mínima dos últimos M dias.
"""

import pandas as pd
from strategies.base import BaseStrategy


class DonchianBreakout(BaseStrategy):
    """
    Sistema de breakout de canal de Donchian.

    Parameters
    ----------
    entry_period : int
        Períodos para o canal de entrada (breakout de máxima).
    exit_period : int
        Períodos para o canal de saída (breakout de mínima).
        Geralmente < entry_period.
    """

    def __init__(self, entry_period: int = 20, exit_period: int = 10):
        if exit_period >= entry_period:
            raise ValueError("exit_period deve ser < entry_period.")
        self.entry_period = entry_period
        self.exit_period = exit_period

    @property
    def name(self) -> str:
        return f"Donchian_{self.entry_period}_{self.exit_period}"

    @property
    def parameters(self) -> dict:
        return {"entry_period": self.entry_period, "exit_period": self.exit_period}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)

        high_col = df["high"] if "high" in df.columns else df["close"]
        low_col  = df["low"]  if "low"  in df.columns else df["close"]
        close     = df["close"]

        upper = high_col.rolling(self.entry_period).max()
        lower = low_col.rolling(self.exit_period).min()

        signal = pd.Series(0, index=df.index, dtype=int)
        position = 0

        for i in range(self.entry_period, len(df)):
            if position == 0 and close.iloc[i] >= upper.iloc[i - 1]:
                position = 1
            elif position == 1 and close.iloc[i] <= lower.iloc[i - 1]:
                position = 0
            signal.iloc[i] = position

        return signal
