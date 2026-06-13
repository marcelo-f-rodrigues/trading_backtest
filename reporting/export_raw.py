from pathlib import Path
import pandas as pd


def export_backtest_raw(
    result,
    price_df,
    strategy_name,
    asset,
    period="full",
    output_dir="results",
):
    """
    Exporta arquivos CSV para consumo externo (Lovable, dashboards etc.)

    Gera:

    results/equity/
    results/trades/
    results/prices/
    results/signals/
    """

    base_name = f"{strategy_name}__{asset}__{period}.csv"

    output_dir = Path(output_dir)

    # =====================================================
    # EQUITY
    # =====================================================

    equity_dir = output_dir / "equity"
    equity_dir.mkdir(parents=True, exist_ok=True)

    equity_df = pd.DataFrame({
        "date": result.equity.index,
        "equity": result.equity.values,
        "bh_equity": result.bh_equity.reindex(result.equity.index).values,
        "position": result.positions.reindex(result.equity.index).fillna(0).values,
    })

    equity_df.to_csv(
        equity_dir / base_name,
        index=False
    )

    # =====================================================
    # TRADES
    # =====================================================

    trades_dir = output_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    trade_rows = []

    for trade in result.trades:

        trade_rows.append({
            "entry_date": trade.entry_date,
            "entry_price": trade.entry_price,
            "exit_date": trade.exit_date,
            "exit_price": trade.exit_price,
            "return_pct": trade.pnl_pct,
            "pnl": trade.pnl_abs,
            "bars_held": trade.duration,

            # placeholder
            "trigger": strategy_name,
        })

    pd.DataFrame(trade_rows).to_csv(
        trades_dir / base_name,
        index=False
    )

    # =====================================================
    # PRICES
    # =====================================================

    prices_dir = output_dir / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    price_export = price_df.copy()

    price_export = price_export.reset_index()

    if "index" in price_export.columns:
        price_export.rename(columns={"index": "date"}, inplace=True)

    price_export.to_csv(
        prices_dir / base_name,
        index=False
    )

    # =====================================================
    # SIGNALS
    # =====================================================

    signals_dir = output_dir / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    signals_df = pd.DataFrame({
        "date": result.signals.index,
        "signal": result.signals.values,
    })

    signals_df.to_csv(
        signals_dir / base_name,
        index=False
    )