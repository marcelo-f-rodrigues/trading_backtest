"""
Exporta resultados consolidados para análise posterior.
"""

from pathlib import Path
import pandas as pd
import json


def export_dataframe(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        path,
        index=False,
        encoding="utf-8"
    )


def export_json(data, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            default=str,
            ensure_ascii=False
        )


def export_strategy_analysis(
    ranking,
    investor_report,
    output_dir="results/reports"
):

    output = Path(output_dir)

    # ranking geral
    if isinstance(ranking, pd.DataFrame):
        export_dataframe(
            ranking,
            output / "strategy_ranking.csv"
        )


    # visão por perfil
    if isinstance(investor_report, pd.DataFrame):

        export_dataframe(
            investor_report,
            output / "investor_report.csv"
        )


    print("Relatórios estratégicos exportados.")