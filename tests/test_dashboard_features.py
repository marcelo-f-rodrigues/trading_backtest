import pandas as pd

from backtest.engine import BacktestEngine
from benchmark_dca import simulate_dca_benchmark
from metrics.calculator import MetricsCalculator
from strategies.base import BaseStrategy


class AlwaysLongStrategy(BaseStrategy):
    @property
    def name(self):
        return "AlwaysLong"

    @property
    def parameters(self):
        return {"mode": "always_long"}

    def generate_signals(self, df):
        self.validate(df)
        return pd.Series(1, index=df.index, dtype=int, name="signal")


def _sample_df():
    return pd.DataFrame(
        {
            "open": [10.0, 10.5, 11.0, 11.5, 12.0],
            "high": [10.2, 10.8, 11.4, 11.8, 12.2],
            "low": [9.8, 10.2, 10.7, 11.1, 11.7],
            "close": [10.0, 10.5, 11.0, 11.5, 12.0],
            "volume": [100, 100, 100, 100, 100],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )


def test_dca_benchmark_produces_equity_curve():
    df = _sample_df()

    result = simulate_dca_benchmark(df, asset="TEST", frequency="W", initial_capital=10_000.0)

    assert result.equity is not None
    assert len(result.equity) == len(df)
    assert result.equity.iloc[-1] > 0


def test_metrics_capture_signals_and_exposure():
    df = _sample_df()
    strategy = AlwaysLongStrategy()
    result = BacktestEngine(df, strategy, asset="TEST", initial_capital=10_000.0, gradual_entry=True).run()
    metrics = MetricsCalculator(result, n_parameters=strategy.n_parameters).compute()

    assert metrics.n_signals >= 0
    assert metrics.n_entries >= 0
    assert metrics.max_exposure >= 0
    assert metrics.final_equity > 0
