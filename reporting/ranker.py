"""
reporting/ranker.py
--------------------
Geração de rankings por perfil de investidor e relatórios comparativos.

Perfis disponíveis:
  - growth       : crescimento (CAGR, eficiência)
  - preservation : preservação de capital (drawdown, ruína)
  - flow         : fluxo recorrente (frequência, regularidade)
  - simplicity   : simplicidade operacional
  - robustness   : robustez (cross-asset, temporal, paramétrica)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional

from metrics.calculator import FullMetrics


# ---------------------------------------------------------------------------
# Normalização de métricas (min-max dentro do universo de estratégias)
# ---------------------------------------------------------------------------

def _normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Normaliza para [0, 1]. NaN → 0."""
    s = series.fillna(0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    norm = (s - mn) / (mx - mn)
    return norm if higher_is_better else (1 - norm)


# ---------------------------------------------------------------------------
# Construção da tabela comparativa
# ---------------------------------------------------------------------------

def build_comparison_table(metrics_list: list[FullMetrics]) -> pd.DataFrame:
    """
    Constrói DataFrame comparativo com todas as métricas relevantes.

    Parameters
    ----------
    metrics_list : list[FullMetrics]

    Returns
    -------
    pd.DataFrame indexado por strategy_name + asset
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "strategy":               m.strategy_name,
            "asset":                  m.asset,
            "period":                 getattr(m, "period", "full"),
            # Performance
            "cagr":                   m.cagr,
            "final_equity":           m.final_equity,
            "total_return":           m.total_return,
            "sharpe_ratio":           m.sharpe_ratio,
            "sortino_ratio":          m.sortino_ratio,
            "calmar_ratio":           m.calmar_ratio,
            "profit_factor":          m.profit_factor,
            "volatility_annual":      m.volatility_annual,
            "annualized_volatility":  m.annualized_volatility,
            # Drawdown
            "max_drawdown":           m.max_drawdown,
            "avg_drawdown":           m.avg_drawdown,
            "max_recovery_days":      m.max_drawdown_recovery_days,
            "longest_flat_days":      m.longest_flat_period_days,
            # Trades
            "n_trades":               m.n_trades,
            "n_signals":              m.n_signals,
            "n_entries":              m.n_entries,
            "n_completed_trades":     m.n_completed_trades,
            "win_rate":               m.win_rate,
            "expectancy":             m.expectancy,
            "payoff_ratio":           m.payoff_ratio,
            # Fluxo
            "trades_per_year":        m.trades_per_year,
            "pct_months_positive":    m.pct_months_positive,
            "pct_years_positive":     m.pct_years_positive,
            "monthly_return_std":     m.monthly_return_std,
            "flow":                   m.flow_classification,
            # Eficiência
            "pct_time_invested":      m.pct_time_invested,
            "avg_exposure":           m.avg_exposure,
            "max_exposure":           m.max_exposure,
            "avg_trade_duration":     m.avg_trade_duration,
            # Raros
            "pct_profit_top5":        m.pct_profit_top5,
            "pct_profit_top20pct":    m.pct_profit_top20pct,
            "gini_trades":            m.gini_trades,
            # Risco
            "ruin_risk":              m.ruin_risk,
            "ruin_class":             m.ruin_classification,
            "worst_loss_streak":      m.worst_loss_streak,
            # Operacional
            "complexity_score":       m.complexity_score,
            "complexity_label":       m.complexity_label,
            "n_parameters":           m.n_parameters,
            "decisions_per_year":     m.decisions_per_year,
            # Psicológico
            "pain_index":             m.pain_index,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rankings por Perfil
# ---------------------------------------------------------------------------

def rank_by_profile(
    df: pd.DataFrame,
    profile: str = "growth",
) -> pd.DataFrame:
    """
    Gera ranking das estratégias para um perfil específico.

    Parameters
    ----------
    df : pd.DataFrame
        Saída de build_comparison_table().
    profile : str
        Um de: 'growth', 'preservation', 'flow', 'simplicity', 'robustness'

    Returns
    -------
    pd.DataFrame ordenado por score do perfil.
    """
    df = df.copy()

    PROFILES = {
        "growth": {
            "metrics": ["cagr", "total_return", "pct_time_invested"],
            "weights": [0.5, 0.3, 0.2],
            "higher":  [True, True, True],
        },
        "preservation": {
            "metrics": ["max_drawdown", "ruin_risk", "max_recovery_days"],
            "weights": [0.4, 0.4, 0.2],
            "higher":  [False, False, False],  # menor drawdown = melhor
        },
        "flow": {
            "metrics": ["pct_months_positive", "trades_per_year", "monthly_return_std"],
            "weights": [0.4, 0.3, 0.3],
            "higher":  [True, True, False],
        },
        "simplicity": {
            "metrics": ["complexity_score", "decisions_per_year", "n_parameters"],
            "weights": [0.5, 0.3, 0.2],
            "higher":  [False, False, False],
        },
        "robustness": {
            "metrics": ["sharpe_ratio", "calmar_ratio", "profit_factor"],
            "weights": [0.4, 0.3, 0.3],
            "higher":  [True, True, True],
        },
    }

    if profile not in PROFILES:
        raise ValueError(f"Perfil desconhecido: {profile}. Opções: {list(PROFILES.keys())}")

    cfg = PROFILES[profile]
    score = pd.Series(0.0, index=df.index)

    for metric, weight, higher in zip(cfg["metrics"], cfg["weights"], cfg["higher"]):
        if metric in df.columns:
            norm = _normalize(df[metric], higher_is_better=higher)
            score += norm * weight

    df[f"score_{profile}"] = score
    ranked = df.sort_values(f"score_{profile}", ascending=False).reset_index(drop=True)
    ranked.index += 1  # Ranking começa em 1
    ranked.index.name = "rank"

    return ranked


def full_ranking(df: pd.DataFrame, save_dir: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """
    Gera rankings para todos os perfis.

    Returns
    -------
    dict : {profile_name: ranked_DataFrame}
    """
    profiles = ["growth", "preservation", "flow", "simplicity", "robustness"]
    rankings = {}

    for p in profiles:
        ranked = rank_by_profile(df, p)
        rankings[p] = ranked
        print(f"\n{'='*60}")
        print(f"RANKING — Perfil: {p.upper()}")
        cols = ["strategy", "asset", f"score_{p}", "cagr", "sharpe_ratio", "max_drawdown"]
        cols = [c for c in cols if c in ranked.columns]
        print(ranked[cols].head(10).round(3).to_string())

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ranked.to_csv(os.path.join(save_dir, f"ranking_{p}.csv"))

    return rankings


# ---------------------------------------------------------------------------
# Exportação de Relatório
# ---------------------------------------------------------------------------

def export_report(
    df: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    output_dir: str = "results/reports",
) -> None:
    """
    Exporta CSVs e um sumário consolidado.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Tabela completa
    df.to_csv(os.path.join(output_dir, "all_metrics.csv"), index=False)

    # Rankings
    for profile, ranked in rankings.items():
        ranked.to_csv(os.path.join(output_dir, f"ranking_{profile}.csv"))

    # Sumário por estratégia (média cross-asset)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df.groupby("strategy")[numeric_cols].mean().round(4)
    summary.to_csv(os.path.join(output_dir, "strategy_summary.csv"))

    print(f"\nRelatórios salvos em: {output_dir}")
