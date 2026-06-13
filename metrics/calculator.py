"""
metrics/calculator.py
----------------------
Calcula todas as métricas do framework a partir de um BacktestResult.

Categorias de métricas:
  1. Performance clássica (CAGR, Sharpe, Sortino, Calmar...)
  2. Distribuição de trades (taxa de acerto, payoff, expectancy...)
  3. Dependência de eventos raros (Lorenz, Gini, concentração...)
  4. Eficiência do capital (tempo investido, retorno por dia...)
  5. Métricas de fluxo (frequência, regularidade mensal...)
  6. Métricas temporais (tempo de recuperação, flat periods...)
  7. Métricas psicológicas (Pain Index)
  8. Métricas operacionais (Complexity Score)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from backtest.engine import BacktestResult, Trade


TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Dataclass de resultado completo
# ---------------------------------------------------------------------------

@dataclass
class FullMetrics:
    # Identificação
    strategy_name: str = ""
    asset: str = ""

    # --- 1. Performance Clássica ---
    total_return: float = np.nan
    final_equity: float = np.nan
    cagr: float = np.nan
    annual_return_mean: float = np.nan
    annual_return_median: float = np.nan
    sharpe_ratio: float = np.nan
    sortino_ratio: float = np.nan
    calmar_ratio: float = np.nan
    profit_factor: float = np.nan
    recovery_factor: float = np.nan
    ulcer_index: float = np.nan
    volatility_annual: float = np.nan
    annualized_volatility: float = np.nan
    max_drawdown: float = np.nan
    avg_drawdown: float = np.nan

    # --- 2. Distribuição de Trades ---
    n_trades: int = 0
    win_rate: float = np.nan
    expectancy: float = np.nan
    payoff_ratio: float = np.nan
    avg_win: float = np.nan
    avg_loss: float = np.nan
    median_trade: float = np.nan
    best_trade: float = np.nan
    worst_trade: float = np.nan

    # --- 3. Dependência de Raros ---
    pct_profit_top5: float = np.nan
    pct_profit_top10: float = np.nan
    pct_profit_top20pct: float = np.nan
    gini_trades: float = np.nan

    # --- 4. Eficiência do Capital ---
    avg_exposure: float = np.nan
    max_exposure: float = np.nan
    pct_time_invested: float = np.nan
    pct_time_cash: float = np.nan
    avg_trade_duration: float = np.nan
    max_trade_duration: float = np.nan
    return_per_day_invested: float = np.nan

    # --- 5. Métricas de Fluxo ---
    n_signals: int = 0
    n_entries: int = 0
    n_completed_trades: int = 0
    trades_per_year: float = np.nan
    avg_interval_between_trades: float = np.nan
    max_interval_between_trades: float = np.nan
    monthly_return_mean: float = np.nan
    monthly_return_median: float = np.nan
    monthly_return_std: float = np.nan
    pct_months_positive: float = np.nan
    pct_quarters_positive: float = np.nan
    pct_years_positive: float = np.nan
    max_trades_drought: float = np.nan      # Dias sem operação
    max_profit_drought: float = np.nan      # Dias sem lucro
    flow_classification: str = ""

    # --- 6. Métricas Temporais ---
    avg_drawdown_recovery_days: float = np.nan
    max_drawdown_recovery_days: float = np.nan
    longest_flat_period_days: float = np.nan

    # --- 7. Psicológicas ---
    pain_index: float = np.nan

    # --- 8. Operacionais ---
    n_parameters: int = 0
    decisions_per_year: float = np.nan
    complexity_score: float = np.nan
    complexity_label: str = ""

    # --- Risco de Ruína ---
    worst_loss_streak: float = np.nan
    max_consecutive_losses: int = 0
    ruin_risk: float = np.nan
    ruin_classification: str = ""

    # --- Scores de Objetivo ---
    growth_score: float = np.nan
    income_score: float = np.nan
    safety_score: float = np.nan


# ---------------------------------------------------------------------------
# Calculadora Principal
# ---------------------------------------------------------------------------

class MetricsCalculator:
    """
    Calcula todas as métricas do framework a partir de um BacktestResult.

    Usage:
        calc = MetricsCalculator(result)
        m = calc.compute()
        print(m.cagr, m.sharpe_ratio)
    """

    def __init__(self, result: BacktestResult, n_parameters: int = 2):
        self.result = result
        self.n_parameters = n_parameters
        self.equity   = result.equity
        self.returns  = result.returns
        self.trades   = result.trades
        self.positions = result.positions

    def compute(self) -> FullMetrics:
        m = FullMetrics(
            strategy_name=self.result.strategy_name,
            asset=self.result.asset,
            n_parameters=self.n_parameters,
        )

        self._performance(m)
        self._trade_distribution(m)
        self._rare_events(m)
        self._capital_efficiency(m)
        self._flow(m)
        self._temporal(m)
        self._psychological(m)
        self._operational(m)
        self._ruin_risk(m)
        self._objective_scores(m)

        return m

    # -----------------------------------------------------------------------
    # 1. Performance Clássica
    # -----------------------------------------------------------------------

    def _performance(self, m: FullMetrics):
        eq  = self.equity.dropna()
        ret = self.returns.dropna()

        if len(eq) < 2:
            return

        n_years = len(eq) / TRADING_DAYS

        m.total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        m.final_equity = float(eq.iloc[-1] * 10000.0)
        m.cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else np.nan

        # Retornos anuais
        annual = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        m.annual_return_mean   = float(annual.mean())
        m.annual_return_median = float(annual.median())

        # Volatilidade
        m.volatility_annual = float(ret.std() * np.sqrt(TRADING_DAYS))
        m.annualized_volatility = float(ret.std() * np.sqrt(TRADING_DAYS))

        # Sharpe (risk-free = 0 para simplificar)
        m.sharpe_ratio = (
            float(ret.mean() / ret.std() * np.sqrt(TRADING_DAYS))
            if ret.std() > 0 else np.nan
        )

        # Sortino
        downside = ret[ret < 0]
        downside_std = downside.std() * np.sqrt(TRADING_DAYS)
        m.sortino_ratio = (
            float(ret.mean() * TRADING_DAYS / downside_std)
            if downside_std > 0 else np.nan
        )

        # Drawdown
        rolling_max = eq.cummax()
        drawdown = (eq - rolling_max) / rolling_max
        m.max_drawdown = float(drawdown.min())
        m.avg_drawdown = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

        # Calmar
        m.calmar_ratio = (
            float(m.cagr / abs(m.max_drawdown))
            if m.max_drawdown < 0 else np.nan
        )

        # Ulcer Index
        m.ulcer_index = float(np.sqrt((drawdown ** 2).mean()))

        # Recovery Factor
        if m.max_drawdown < 0:
            m.recovery_factor = float(m.total_return / abs(m.max_drawdown))

        # Profit Factor (via retornos diários positivos/negativos)
        pos_sum = ret[ret > 0].sum()
        neg_sum = abs(ret[ret < 0].sum())
        m.profit_factor = float(pos_sum / neg_sum) if neg_sum > 0 else np.nan

    # -----------------------------------------------------------------------
    # 2. Distribuição de Trades
    # -----------------------------------------------------------------------

    def _trade_distribution(self, m: FullMetrics):
        trades = self.trades
        m.n_trades = len(trades)
        m.n_completed_trades = len(trades)
        if self.result.signals is not None and len(self.result.signals):
            m.n_signals = int((self.result.signals.fillna(0) != 0).sum())
            prev = self.result.signals.shift(fill_value=0)
            m.n_entries = int(((self.result.signals != 0) & (prev == 0)).sum())

        if not trades:
            return

        pnls = np.array([t.pnl_pct for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        m.win_rate      = float(len(wins) / len(pnls))
        m.avg_win       = float(wins.mean())  if len(wins)   > 0 else np.nan
        m.avg_loss      = float(losses.mean()) if len(losses) > 0 else np.nan
        m.payoff_ratio  = float(abs(m.avg_win / m.avg_loss)) if m.avg_loss and m.avg_loss != 0 else np.nan
        m.expectancy    = float(pnls.mean())
        m.median_trade  = float(np.median(pnls))
        m.best_trade    = float(pnls.max())
        m.worst_trade   = float(pnls.min())

    # -----------------------------------------------------------------------
    # 3. Dependência de Eventos Raros
    # -----------------------------------------------------------------------

    def _rare_events(self, m: FullMetrics):
        if not self.trades:
            return

        pnls = np.array([t.pnl_pct for t in self.trades])
        total_profit = pnls[pnls > 0].sum()

        if total_profit <= 0:
            return

        sorted_desc = np.sort(pnls)[::-1]

        m.pct_profit_top5  = float(sorted_desc[:5].clip(min=0).sum() / total_profit)
        m.pct_profit_top10 = float(sorted_desc[:10].clip(min=0).sum() / total_profit)

        n_top20 = max(1, int(len(pnls) * 0.2))
        m.pct_profit_top20pct = float(sorted_desc[:n_top20].clip(min=0).sum() / total_profit)

        # Gini
        m.gini_trades = self._gini(pnls)

    @staticmethod
    def _gini(arr: np.ndarray) -> float:
        arr = np.sort(np.abs(arr))
        n = len(arr)
        if n == 0 or arr.sum() == 0:
            return np.nan
        idx = np.arange(1, n + 1)
        return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)

    # -----------------------------------------------------------------------
    # 4. Eficiência do Capital
    # -----------------------------------------------------------------------

    def _capital_efficiency(self, m: FullMetrics):
        pos = self.positions.dropna()
        m.avg_exposure      = float(pos.mean())
        m.max_exposure      = float(pos.max()) if len(pos) else 0.0
        m.pct_time_invested = float((pos > 0).mean())
        m.pct_time_cash     = float((pos == 0).mean())

        if self.trades:
            durations = [t.duration for t in self.trades]
            m.avg_trade_duration = float(np.mean(durations))
            m.max_trade_duration = float(np.max(durations))

            days_invested = sum(durations)
            if days_invested > 0 and not np.isnan(self.result.equity.iloc[-1]):
                m.return_per_day_invested = float(
                    (self.result.equity.iloc[-1] - 1) / days_invested
                )

    # -----------------------------------------------------------------------
    # 5. Métricas de Fluxo
    # -----------------------------------------------------------------------

    def _flow(self, m: FullMetrics):
        eq  = self.equity.dropna()
        ret = self.returns.dropna()
        n_years = len(eq) / TRADING_DAYS

        if n_years > 0:
            m.trades_per_year = (len(self.trades) if len(self.trades) > 0 else m.n_entries) / n_years

        # Intervalos entre trades
        if len(self.trades) > 1:
            entry_dates = [t.entry_date for t in self.trades]
            intervals = [(entry_dates[i+1] - entry_dates[i]).days for i in range(len(entry_dates)-1)]
            m.avg_interval_between_trades = float(np.mean(intervals))
            m.max_interval_between_trades = float(np.max(intervals))

        # Retornos mensais
        monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        if len(monthly) > 0:
            m.monthly_return_mean   = float(monthly.mean())
            m.monthly_return_median = float(monthly.median())
            m.monthly_return_std    = float(monthly.std())
            m.pct_months_positive   = float((monthly > 0).mean())

        # Trimestres
        quarterly = ret.resample("QE").apply(lambda x: (1 + x).prod() - 1)
        if len(quarterly) > 0:
            m.pct_quarters_positive = float((quarterly > 0).mean())

        # Anos
        annual = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        if len(annual) > 0:
            m.pct_years_positive = float((annual > 0).mean())

        # Classificação de fluxo
        tpy = m.trades_per_year if not np.isnan(m.trades_per_year) else 0
        if tpy >= 24:
            m.flow_classification = "recorrente"
        elif tpy >= 6:
            m.flow_classification = "moderado"
        else:
            m.flow_classification = "esporádico"

    # -----------------------------------------------------------------------
    # 6. Métricas Temporais
    # -----------------------------------------------------------------------

    def _temporal(self, m: FullMetrics):
        eq = self.equity.dropna()

        # Longest flat period (maior tempo sem nova máxima)
        roll_max = eq.cummax()
        is_flat = eq < roll_max
        flat_lengths = []
        count = 0
        for v in is_flat:
            if v:
                count += 1
            else:
                if count > 0:
                    flat_lengths.append(count)
                count = 0
        if count > 0:
            flat_lengths.append(count)

        m.longest_flat_period_days = float(max(flat_lengths)) if flat_lengths else 0.0

        # Tempo médio e máximo de recuperação de drawdown
        rolling_max = eq.cummax()
        in_dd = eq < rolling_max
        recovery_times = []
        dd_start = None

        for i, (date, val) in enumerate(in_dd.items()):
            if val and dd_start is None:
                dd_start = i
            elif not val and dd_start is not None:
                recovery_times.append(i - dd_start)
                dd_start = None

        if recovery_times:
            m.avg_drawdown_recovery_days = float(np.mean(recovery_times))
            m.max_drawdown_recovery_days = float(np.max(recovery_times))

    # -----------------------------------------------------------------------
    # 7. Psicológico
    # -----------------------------------------------------------------------

    def _psychological(self, m: FullMetrics):
        eq = self.equity.dropna()
        rolling_max = eq.cummax()
        drawdown = (eq - rolling_max) / rolling_max

        # Pain Index = média dos drawdowns quadráticos (intensidade × duração implícita)
        m.pain_index = float(np.sqrt((drawdown ** 2).mean()) * 100)

    # -----------------------------------------------------------------------
    # 8. Operacional
    # -----------------------------------------------------------------------

    def _operational(self, m: FullMetrics):
        n_years = len(self.equity) / TRADING_DAYS
        m.decisions_per_year = (m.n_entries + m.n_completed_trades) / n_years if n_years > 0 else 0

        # Complexity Score: normaliza número de parâmetros + frequência de decisões
        param_score = min(m.n_parameters / 5, 1.0)
        freq_score  = min(m.decisions_per_year / 100, 1.0)
        m.complexity_score = float((param_score + freq_score) / 2)

        if m.complexity_score < 0.33:
            m.complexity_label = "baixa manutenção"
        elif m.complexity_score < 0.66:
            m.complexity_label = "média manutenção"
        else:
            m.complexity_label = "alta manutenção"

    # -----------------------------------------------------------------------
    # Risco de Ruína
    # -----------------------------------------------------------------------

    def _ruin_risk(self, m: FullMetrics):
        if not self.trades:
            return

        pnls = np.array([t.pnl_pct for t in self.trades])

        # Pior sequência de perdas consecutivas
        consecutive = 0
        max_consecutive = 0
        cumulative_loss = 0.0
        worst_streak = 0.0

        for p in pnls:
            if p < 0:
                consecutive += 1
                cumulative_loss += p
                max_consecutive = max(max_consecutive, consecutive)
                worst_streak    = min(worst_streak, cumulative_loss)
            else:
                consecutive   = 0
                cumulative_loss = 0.0

        m.max_consecutive_losses = max_consecutive
        m.worst_loss_streak      = float(worst_streak)

        # Estimativa simples de risco de ruína via fórmula Kelly inversa
        wr = m.win_rate if not np.isnan(m.win_rate) else 0.5
        pr = m.payoff_ratio if not np.isnan(m.payoff_ratio) else 1.0
        if pr > 0 and wr < 1:
            edge = wr - (1 - wr) / pr
            # Risk of ruin approximation: ((1-edge)/(1+edge))^capital_units
            # Simplificado: base em max_drawdown
            m.ruin_risk = float(abs(m.max_drawdown)) if not np.isnan(m.max_drawdown) else 1.0
        else:
            m.ruin_risk = 1.0

        cfg_thresh = {"low": 0.05, "high": 0.20}
        if m.ruin_risk < cfg_thresh["low"]:
            m.ruin_classification = "baixo"
        elif m.ruin_risk < cfg_thresh["high"]:
            m.ruin_classification = "médio"
        else:
            m.ruin_classification = "alto"

        # Texto explicativo útil para o dashboard
        if m.ruin_risk < 0.05:
            m.ruin_classification = "baixo"
        elif m.ruin_risk < 0.20:
            m.ruin_classification = "médio"
        else:
            m.ruin_classification = "alto"

    # -----------------------------------------------------------------------
    # Scores por objetivo
    # -----------------------------------------------------------------------

    def _objective_scores(self, m: FullMetrics):

        # Crescimento de capital
        growth = [
            m.cagr,
            m.sharpe_ratio,
            -abs(m.max_drawdown)
        ]

        m.growth_score = np.nanmean(growth)


        # Renda constante
        income = [
            m.monthly_return_mean,
            m.pct_months_positive,
            m.trades_per_year / 100
        ]

        m.income_score = np.nanmean(income)



        # Preservação de capital
        safety = [
            m.sharpe_ratio,
            -abs(m.max_drawdown),
            -m.volatility_annual
        ]

        m.safety_score = np.nanmean(safety)

    # -----------------------------------------------------------------------
    # Utilitário: resumo como dict
    # -----------------------------------------------------------------------

    def to_dict(self, m: FullMetrics) -> dict:
        return {k: v for k, v in m.__dict__.items()}

    def full_report(self) -> pd.Series:
        m = self.compute()
        return pd.Series(self.to_dict(m))
