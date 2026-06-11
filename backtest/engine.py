"""
backtest/engine.py
-------------------
Engine central de backtest com suporte a entradas e saídas graduais.

Filosofia:
  - Nunca monta/desmonta a posição inteira de uma vez.
  - Divide a entrada/saída em N tranches ao longo de N dias.
  - Aplica comissão e slippage em cada tranche.
  - Retorna um objeto BacktestResult com equity curve, trades e estatísticas básicas.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import yaml

from strategies.base import BaseStrategy


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date:  Optional[pd.Timestamp]
    entry_price: float
    exit_price:  Optional[float]
    size:        float          # Fração do capital alocado neste trade (0–1)
    pnl_pct:     float = 0.0   # Retorno percentual do trade
    pnl_abs:     float = 0.0   # P&L em unidades monetárias
    duration:    int   = 0     # Dias em posição


@dataclass
class BacktestResult:
    strategy_name:  str
    asset:          str
    equity:         pd.Series                 # Curva de capital (normalizada, início = 1.0)
    returns:        pd.Series                 # Retornos diários da estratégia
    bh_equity:      pd.Series                 # Buy & Hold para comparação
    trades:         list[Trade] = field(default_factory=list)
    positions:      pd.Series  = field(default_factory=pd.Series)  # Exposição diária (0–1)
    signals:        pd.Series  = field(default_factory=pd.Series)  # Sinais brutos da estratégia


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Executa o backtest de uma estratégia em um DataFrame OHLCV.

    Parameters
    ----------
    df : pd.DataFrame
        Dados OHLCV com DatetimeIndex.
    strategy : BaseStrategy
        Instância de estratégia com método generate_signals().
    asset : str
        Nome do ativo (usado no resultado).
    initial_capital : float
    commission : float
        Comissão por tranche como fração do valor negociado.
    slippage : float
        Slippage por tranche como fração do preço.
    gradual_entry : bool
        Se True, usa entradas/saídas em tranches.
    n_tranches : int
        Número de tranches para montar/desmontar posição.
    tranche_interval_days : int
        Intervalo em dias entre cada tranche.
    config_path : str
        Caminho para config.yaml (fallback de parâmetros).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        asset: str = "UNKNOWN",
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.001,
        gradual_entry: bool = True,
        n_tranches: int = 3,
        tranche_interval_days: int = 3,
        config_path: str = "config.yaml",
    ):
        # Tenta carregar do config se não especificado explicitamente
        try:
            cfg = load_config(config_path)
            bt_cfg = cfg.get("backtest", {})
            ge_cfg = bt_cfg.get("gradual_entry", {})
            self.initial_capital = bt_cfg.get("initial_capital", initial_capital)
            self.commission       = bt_cfg.get("commission", commission)
            self.slippage         = bt_cfg.get("slippage", slippage)
            self.gradual_entry    = ge_cfg.get("enabled", gradual_entry)
            self.n_tranches       = ge_cfg.get("n_tranches", n_tranches)
            self.tranche_interval = ge_cfg.get("tranche_interval_days", tranche_interval_days)
        except FileNotFoundError:
            self.initial_capital = initial_capital
            self.commission       = commission
            self.slippage         = slippage
            self.gradual_entry    = gradual_entry
            self.n_tranches       = n_tranches
            self.tranche_interval = tranche_interval_days

        self.df       = df.copy()
        self.strategy = strategy
        self.asset    = asset

    def run(self) -> BacktestResult:
        """Executa o backtest e retorna BacktestResult."""
        signals = self.strategy.generate_signals(self.df)

        if self.gradual_entry:
            positions, trades = self._simulate_gradual(signals)
        else:
            positions, trades = self._simulate_instant(signals)

        # Equity curve: retorno diário = posição * retorno do preço - custos
        price_returns = self.df["close"].pct_change().fillna(0)
        strat_returns = positions.shift(1).fillna(0) * price_returns

        equity    = (1 + strat_returns).cumprod()
        bh_equity = (1 + price_returns).cumprod()

        return BacktestResult(
            strategy_name=self.strategy.name,
            asset=self.asset,
            equity=equity,
            returns=strat_returns,
            bh_equity=bh_equity,
            trades=trades,
            positions=positions,
            signals=signals,
        )

    # -----------------------------------------------------------------------
    # Simulação: entrada/saída gradual
    # -----------------------------------------------------------------------

    def _simulate_gradual(self, signals: pd.Series):
        """
        Simula entradas e saídas graduais em N tranches.

        A posição é aumentada/reduzida em 1/N a cada `tranche_interval` dias
        após a mudança de sinal.
        """
        n = len(self.df)
        close = self.df["close"].values
        dates = self.df.index

        positions = np.zeros(n)        # Exposição atual (0.0 a 1.0)
        target    = np.zeros(n)        # Target de exposição pelo sinal
        trades: list[Trade] = []

        tranche_size = 1.0 / self.n_tranches
        transaction_cost = self.commission + self.slippage

        # Estado da construção de posição
        tranches_done   = 0
        tranches_target = 0
        next_tranche_i  = None
        current_entry_price = None
        current_entry_date  = None

        for i in range(n):
            sig = signals.iloc[i]
            new_target = float(sig)  # 0.0 ou 1.0

            # Mudança de sinal detectada
            if i > 0 and new_target != target[i - 1]:
                if new_target > target[i - 1]:
                    # Iniciando entrada
                    tranches_target = self.n_tranches
                    tranches_done   = 0
                    next_tranche_i  = i
                    current_entry_date  = dates[i]
                    current_entry_price = close[i] * (1 + self.slippage)
                else:
                    # Iniciando saída
                    tranches_target = 0
                    tranches_done   = int(positions[i - 1] / tranche_size)
                    next_tranche_i  = i

            target[i] = new_target

            # Executar tranche se for o momento
            if next_tranche_i is not None and i >= next_tranche_i:
                current_pos = positions[i - 1] if i > 0 else 0.0

                if tranches_target > tranches_done:
                    # Adicionando tranche de compra
                    positions[i] = min(current_pos + tranche_size, 1.0)
                    tranches_done += 1
                    if tranches_done < self.n_tranches:
                        next_tranche_i = i + self.tranche_interval
                    else:
                        next_tranche_i = None

                elif tranches_target < tranches_done and current_pos > 0:
                    # Removendo tranche de venda
                    positions[i] = max(current_pos - tranche_size, 0.0)
                    tranches_done -= 1

                    if positions[i] == 0.0:
                        # Posição completamente fechada → registrar trade
                        if current_entry_price and current_entry_date:
                            exit_price = close[i] * (1 - self.slippage)
                            gross_ret  = (exit_price / current_entry_price) - 1
                            net_ret    = gross_ret - transaction_cost * 2  # entrada + saída
                            trades.append(Trade(
                                entry_date=current_entry_date,
                                exit_date=dates[i],
                                entry_price=current_entry_price,
                                exit_price=exit_price,
                                size=1.0,
                                pnl_pct=net_ret,
                                pnl_abs=net_ret * self.initial_capital,
                                duration=(dates[i] - current_entry_date).days,
                            ))
                        next_tranche_i = None

                    elif tranches_done > tranches_target:
                        next_tranche_i = i + self.tranche_interval
                    else:
                        next_tranche_i = None
                else:
                    positions[i] = positions[i - 1] if i > 0 else 0.0
            else:
                positions[i] = positions[i - 1] if i > 0 else 0.0

        positions_series = pd.Series(positions, index=self.df.index, name="position")
        return positions_series, trades

    # -----------------------------------------------------------------------
    # Simulação: entrada/saída instantânea (modo simples)
    # -----------------------------------------------------------------------

    def _simulate_instant(self, signals: pd.Series):
        """Versão simplificada: executa toda a posição de uma vez."""
        positions = signals.astype(float)
        close = self.df["close"]
        dates = self.df.index
        trades = []

        in_trade = False
        entry_date = None
        entry_price = None

        for i in range(len(signals)):
            sig = signals.iloc[i]

            if not in_trade and sig == 1:
                in_trade    = True
                entry_date  = dates[i]
                entry_price = close.iloc[i] * (1 + self.slippage)

            elif in_trade and sig == 0:
                exit_price = close.iloc[i] * (1 - self.slippage)
                gross_ret  = (exit_price / entry_price) - 1
                net_ret    = gross_ret - (self.commission + self.slippage) * 2
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=1.0,
                    pnl_pct=net_ret,
                    pnl_abs=net_ret * self.initial_capital,
                    duration=(dates[i] - entry_date).days,
                ))
                in_trade    = False
                entry_date  = None
                entry_price = None

        return positions, trades
