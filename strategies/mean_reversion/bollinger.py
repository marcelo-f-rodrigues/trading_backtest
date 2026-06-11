"""
strategies/mean_reversion/bollinger.py
---------------------------------------
Estratégias baseadas em Bollinger Bands.

Implementa:
  - BollingerReversion : compra abaixo da banda inferior, sai na média
  - BollingerSqueeze   : detecta contração de volatilidade (setup)
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class BollingerReversion(BaseStrategy):
    """
    Reversão à média via Bollinger Bands.

    Compra quando o preço fecha abaixo da banda inferior.
    Sai quando o preço fecha acima da média (ou banda superior).

    Parameters
    ----------
    period : int
        Período da média móvel.
    std_dev : float
        Multiplicador do desvio padrão para as bandas.
    exit_at : str
        'mean' → sai na média móvel | 'upper' → sai na banda superior.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0, exit_at: str = "mean"):
        self.period = period
        self.std_dev = std_dev
        self.exit_at = exit_at

    @property
    def name(self) -> str:
        return f"BB_Rev_{self.period}_{self.std_dev}std_{self.exit_at}"

    @property
    def parameters(self) -> dict:
        return {"period": self.period, "std_dev": self.std_dev, "exit_at": self.exit_at}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        close = df["close"]

        mean = close.rolling(self.period).mean()
        std  = close.rolling(self.period).std()
        lower = mean - self.std_dev * std
        upper = mean + self.std_dev * std

        exit_line = upper if self.exit_at == "upper" else mean

        signal = pd.Series(0, index=df.index, dtype=int)
        position = 0

        for i in range(self.period, len(df)):
            if position == 0 and close.iloc[i] < lower.iloc[i]:
                position = 1
            elif position == 1 and close.iloc[i] > exit_line.iloc[i]:
                position = 0
            signal.iloc[i] = position

        return signal


class ZScoreReversion(BaseStrategy):
    """
    Reversão via Z-Score do preço em relação à média histórica.

    Compra quando Z-Score < -entry_z (preço muito abaixo da média).
    Sai quando Z-Score > -exit_z (preço retorna à média).

    Parameters
    ----------
    period : int
        Janela para calcular média e desvio.
    entry_z : float
        Z-Score de entrada (ex: -2.0 → preço 2 desvios abaixo).
    exit_z : float
        Z-Score de saída (ex: 0.0 → retorno à média).
    """

    def __init__(self, period: int = 60, entry_z: float = 2.0, exit_z: float = 0.0):
        self.period = period
        self.entry_z = entry_z
        self.exit_z = exit_z

    @property
    def name(self) -> str:
        return f"ZScore_{self.period}_entry{self.entry_z}_exit{self.exit_z}"

    @property
    def parameters(self) -> dict:
        return {"period": self.period, "entry_z": self.entry_z, "exit_z": self.exit_z}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        self.validate(df)
        close = df["close"]

        mean = close.rolling(self.period).mean()
        std  = close.rolling(self.period).std().replace(0, np.nan)
        zscore = (close - mean) / std

        signal = pd.Series(0, index=df.index, dtype=int)
        position = 0

        for i in range(self.period, len(df)):
            if position == 0 and zscore.iloc[i] < -self.entry_z:
                position = 1
            elif position == 1 and zscore.iloc[i] > -self.exit_z:
                position = 0
            signal.iloc[i] = position

        return signal
