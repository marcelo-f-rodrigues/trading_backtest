"""
reporting/strategy_profiler.py

Transforma resultados de backtest em análise de estratégia
com linguagem orientada ao investidor.

Objetivo:
- encontrar estratégias robustas em múltiplos ativos
- separar estratégias de crescimento e geração de fluxo
- gerar relatórios fáceis de interpretar
"""


from pathlib import Path
import pandas as pd
import numpy as np



# ---------------------------------------------------------
# Métricas básicas
# ---------------------------------------------------------

def total_return(equity):
    return equity.iloc[-1] - 1


def max_drawdown(equity):
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()


def win_rate(trades):

    if not trades:
        return 0

    wins = [
        t for t in trades
        if t.pnl_pct > 0
    ]

    return len(wins) / len(trades)



def trades_per_year(trades):

    if not trades:
        return 0

    days = (
        trades[-1].exit_date -
        trades[0].entry_date
    ).days

    if days <= 0:
        return 0

    return len(trades) / (days / 365)



# ---------------------------------------------------------
# Perfil da estratégia
# ---------------------------------------------------------

def classify_strategy(
    growth_score,
    cashflow_score
):

    if growth_score >= cashflow_score:

        return {
            "perfil": "Crescimento de patrimônio",
            "descricao":
            "Estratégia focada em capturar grandes movimentos de alta. "
            "Aceita ficar períodos fora do mercado para buscar tendências longas."
        }


    else:

        return {
            "perfil": "Geração de fluxo",
            "descricao":
            "Estratégia que busca realizar lucros com maior frequência, "
            "vendendo momentos de alta e recomprando oportunidades."
        }



# ---------------------------------------------------------
# Avaliação individual
# ---------------------------------------------------------

def evaluate_result(result):

    ret = total_return(result.equity)

    dd = abs(max_drawdown(result.equity))


    trades = len(result.trades)

    win = win_rate(result.trades)

    frequency = trades_per_year(result.trades)



    # --------------------------
    # Score crescimento
    # --------------------------

    growth_score = (
        min(ret * 50, 50)
        +
        min((1-dd) * 30, 30)
        +
        min(win * 20,20)
    )


    # --------------------------
    # Score fluxo
    # --------------------------

    cashflow_score = (
        min(frequency * 10,40)
        +
        min(win * 40,40)
        +
        min(trades,20)
    )


    profile = classify_strategy(
        growth_score,
        cashflow_score
    )


    return {

        "estrategia":
            result.strategy_name,

        "ativo":
            result.asset,


        "retorno_total_%":
            round(ret*100,2),


        "drawdown_max_%":
            round(dd*100,2),


        "numero_trades":
            trades,


        "trades_por_ano":
            round(frequency,2),


        "taxa_acerto_%":
            round(win*100,2),


        "score_crescimento":
            round(growth_score,1),


        "score_fluxo":
            round(cashflow_score,1),


        "perfil":
            profile["perfil"],


        "explicacao":
            profile["descricao"]

    }



# ---------------------------------------------------------
# Ranking entre ativos
# ---------------------------------------------------------

def build_strategy_ranking(results, output="results/reports"):


    rows = []


    for r in results:

        rows.append(
            evaluate_result(r)
        )


    df = pd.DataFrame(rows)


    Path(output).mkdir(
        parents=True,
        exist_ok=True
    )


    # ranking geral
    df["nota_final"] = (
        df["score_crescimento"]
        +
        df["score_fluxo"]
    ) / 2


    df = df.sort_values(
        "nota_final",
        ascending=False
    )


    df.to_csv(
        Path(output) /
        "strategy_ranking.csv",
        index=False
    )


    return df




def export_strategy_ranking(ranking, output_dir="results/reports"):
    """
    Exporta ranking consolidado das estratégias.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # dataframe
    df = pd.DataFrame(ranking)

    csv_path = output / "strategy_ranking.csv"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8"
    )

    print(f"Ranking exportado: {csv_path}")


    # resumo humano
    txt_path = output / "strategy_ranking_summary.txt"

    with open(txt_path, "w", encoding="utf-8") as f:

        f.write("="*70 + "\n")
        f.write("RANKING DE ESTRATÉGIAS\n")
        f.write("="*70 + "\n\n")


        for _, row in df.iterrows():

            f.write(
                f"""
Estratégia: {row.get('strategy','')}
Perfil: {row.get('profile','')}

Score:
{row}

------------------------------

"""
            )


    print(f"Resumo exportado: {txt_path}")