import pandas as pd
from pathlib import Path


def build_investor_report(
    all_results,
    output_dir="results/reports"
):

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )


    rows = []


    for r in all_results:

        final_equity = r.equity.iloc[-1]
        bh_final = r.bh_equity.iloc[-1]

        total_return = final_equity - 1


        rows.append({

            "strategy": r.strategy_name,
            "asset": r.asset,

            "return": total_return,

            "beats_buy_hold":
                final_equity > bh_final,

            "trades":
                len(r.trades),

            "avg_trade":
                (
                    sum(
                        t.pnl_pct 
                        for t in r.trades
                    )
                    /
                    len(r.trades)
                )
                if r.trades else 0,


            "max_drawdown":

                (
                    (r.equity /
                     r.equity.cummax())
                    -1
                ).min()

        })


    df = pd.DataFrame(rows)


    # consistência por estratégia
    summary = (

        df.groupby("strategy")
        .agg({

            "return":"mean",

            "beats_buy_hold":"mean",

            "trades":"mean",

            "max_drawdown":"mean"

        })

        .reset_index()

    )


    # score simples
    summary["robust_score"] = (

        summary["return"] * 0.4

        +

        summary["beats_buy_hold"] * 0.3

        -

        abs(summary["max_drawdown"]) * 0.3

    )


    summary = (

        summary
        .sort_values(
            "robust_score",
            ascending=False
        )

    )


    summary.to_csv(
        Path(output_dir)
        /
        "investor_strategy_ranking.csv",

        index=False

    )


    return summary