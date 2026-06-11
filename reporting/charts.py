"""
reporting/charts.py
--------------------
Geração de gráficos para análise e apresentação dos resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from typing import Optional

from backtest.engine import BacktestResult


STYLE = {
    "strategy": "#2563EB",   # azul
    "bh":       "#64748B",   # cinza
    "drawdown": "#DC2626",   # vermelho
    "positive": "#16A34A",   # verde
}


def plot_equity_curve(
    result: BacktestResult,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plota a equity curve da estratégia vs Buy & Hold com drawdown.
    """
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    eq = result.equity
    bh = result.bh_equity
    pos = result.positions

    # --- Equity ---
    ax1.plot(eq.index, eq.values, color=STYLE["strategy"], lw=1.5,
             label=f"Estratégia: {result.strategy_name}")
    ax1.plot(bh.index, bh.values, color=STYLE["bh"], lw=1, ls="--",
             label="Buy & Hold", alpha=0.7)
    ax1.set_ylabel("Capital (base 1.0)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(f"{result.strategy_name}  |  {result.asset}", fontsize=10, fontweight="bold")
    ax1.grid(alpha=0.3)

    # --- Drawdown ---
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    ax2.fill_between(dd.index, dd.values, 0, color=STYLE["drawdown"], alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)

    # --- Posição ---
    ax3.fill_between(pos.index, pos.values, 0, color=STYLE["strategy"], alpha=0.5)
    ax3.set_ylabel("Exposição")
    ax3.set_ylim(-0.05, 1.15)
    ax3.grid(alpha=0.3)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Gráfico salvo: {save_path}")

    if show:
        plt.show()

    return fig


def plot_trade_distribution(
    result: BacktestResult,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plota a distribuição dos retornos de trade (histograma + curva de Lorenz).
    """
    if not result.trades:
        print("Nenhum trade para plotar.")
        return None

    pnls = np.array([t.pnl_pct * 100 for t in result.trades])  # em %

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Distribuição de Trades — {result.strategy_name} | {result.asset}", fontsize=10)

    # Histograma
    ax = axes[0]
    colors = [STYLE["positive"] if p > 0 else STYLE["drawdown"] for p in pnls]
    ax.bar(range(len(pnls)), sorted(pnls), color=sorted(colors), alpha=0.8, width=1.0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Trade (ordenado por retorno)")
    ax.set_ylabel("Retorno (%)")
    ax.set_title("Retornos por Trade")
    ax.grid(alpha=0.3)

    # Curva de Lorenz
    ax = axes[1]
    sorted_pnls = np.sort(pnls)
    cumulative  = np.cumsum(sorted_pnls)
    if cumulative[-1] != 0:
        cumulative = cumulative / cumulative[-1]
    x = np.linspace(0, 1, len(cumulative))
    ax.plot(x, cumulative, color=STYLE["strategy"], lw=2, label="Lorenz")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Igualdade perfeita")
    ax.fill_between(x, cumulative, x, alpha=0.2, color=STYLE["strategy"])
    ax.set_xlabel("Fração de trades")
    ax.set_ylabel("Fração do lucro acumulado")
    ax.set_title("Curva de Lorenz dos Trades")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_monthly_returns_heatmap(
    result: BacktestResult,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Heatmap de retornos mensais (anos × meses).
    """
    monthly = result.returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    table   = monthly.groupby([monthly.index.year, monthly.index.month]).sum().unstack()
    table.columns = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(table) * 0.5 + 1)))
    vmax = max(abs(table.values[~np.isnan(table.values)]).max(), 1)

    im = ax.imshow(table.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Retorno (%)")

    ax.set_xticks(range(12))
    ax.set_xticklabels(table.columns, fontsize=8)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table.index, fontsize=8)

    # Anotar valores
    for i in range(len(table)):
        for j in range(12):
            val = table.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=6, color="black")

    ax.set_title(f"Retornos Mensais (%) — {result.strategy_name} | {result.asset}", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_profile_radar(
    scores: dict[str, float],
    strategy_name: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Gráfico de radar com scores por perfil de investidor.

    Parameters
    ----------
    scores : dict
        Ex: {"growth": 0.7, "preservation": 0.5, "flow": 0.3, ...}
    """
    labels = list(scores.keys())
    values = list(scores.values())
    n = len(labels)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(angles, values, color=STYLE["strategy"], lw=2)
    ax.fill(angles, values, color=STYLE["strategy"], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.upper() for l in labels], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(f"Perfis — {strategy_name}", fontsize=11, pad=15)
    ax.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()

    return fig
