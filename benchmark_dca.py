"""Benchmark DCA (Dollar Cost Averaging) para comparação com estratégias."""

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, Trade


def simulate_dca_benchmark(
    df: pd.DataFrame,
    asset: str,
    frequency: str = "W",
    initial_capital: float = 10_000.0,
    contribution: float | None = None,
) -> BacktestResult:
    """Simula um benchmark DCA semanal ou mensal.

    O benchmark compra um valor fixo em cada período programado, mantendo
    o restante em caixa. O patrimônio total é recalculado diariamente.
    """
    close = df["close"].astype(float)
    dates = df.index

    if contribution is None:
        # Aproximação simples: dividir o capital inicial por número de períodos
        # de compra esperados no histórico. Isso mantém o benchmark comparável.
        if frequency.upper().startswith("W"):
            periods = max(1, int((dates[-1] - dates[0]).days / 7) + 1)
        else:
            periods = max(1, (dates[-1].year - dates[0].year) * 12 + (dates[-1].month - dates[0].month) + 1)
        contribution = initial_capital / periods

    cash = float(initial_capital)
    shares = 0.0
    equity = []
    trades = []
    positions = []
    entry_dates = []

    if frequency.upper().startswith("W"):
        resampler = df.resample("W-MON", label="right", closed="right")
    else:
        resampler = df.resample("ME", label="right", closed="right")

    for dt, row in df.iterrows():
        price = float(row["close"])
        if dt in resampler.last().index:
            # Compra programada no fim do período (ou no último dia do período)
            if cash >= contribution:
                qty = contribution / price
                shares += qty
                cash -= contribution
                trades.append(Trade(
                    entry_date=pd.Timestamp(dt),
                    exit_date=None,
                    entry_price=price,
                    exit_price=None,
                    size=contribution / initial_capital,
                    pnl_pct=0.0,
                    pnl_abs=0.0,
                    duration=0,
                ))
                entry_dates.append(pd.Timestamp(dt))

        total_value = cash + shares * price
        equity.append(total_value)
        positions.append((shares * price) / total_value if total_value > 0 else 0.0)

    equity_series = pd.Series(equity, index=dates, name="equity")
    eq_norm = equity_series / initial_capital
    returns = eq_norm.pct_change().fillna(0.0)
    bh_equity = (1 + close.pct_change().fillna(0.0)).cumprod()

    return BacktestResult(
        strategy_name="DCA_Weekly" if frequency.upper().startswith("W") else "DCA_Monthly",
        asset=asset,
        equity=eq_norm,
        returns=returns,
        bh_equity=bh_equity,
        trades=trades,
        positions=pd.Series(positions, index=dates, name="position"),
        signals=pd.Series(1, index=dates, name="signal", dtype=int),
    )
