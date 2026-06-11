"""
strategies/base.py
------------------
Classe base abstrata para todas as estratégias.
Toda estratégia deve herdar desta classe e implementar `generate_signals`.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class BaseStrategy(ABC):
    """
    Interface padrão para todas as estratégias do framework.

    Subclasses devem implementar:
        - generate_signals(df) → pd.Series com valores em {-1, 0, 1}
        - name (property)
        - parameters (property)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome único da estratégia (ex: 'MA_Crossover_20_200')."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Dicionário com todos os parâmetros e seus valores atuais."""
        ...

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Gera sinais de posição a partir de um DataFrame OHLCV.

        Parameters
        ----------
        df : pd.DataFrame
            Colunas: open, high, low, close, volume. Index: DatetimeIndex.

        Returns
        -------
        pd.Series
            Index igual ao df.index.
            Valores:
              +1 → comprado (long)
               0 → sem posição (cash)
              -1 → vendido (short) — reservado para fase 2
        """
        ...

    def validate(self, df: pd.DataFrame) -> None:
        """Valida que o DataFrame tem as colunas mínimas necessárias."""
        required = {"close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame faltando colunas: {missing}")
        if len(df) < 2:
            raise ValueError("DataFrame deve ter ao menos 2 linhas.")

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.__class__.__name__}({params})"
