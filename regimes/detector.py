"""
regimes/detector.py
--------------------
Detecção e classificação de regimes de mercado.

Dois modos:
  1. Supervisionado  : regras explícitas (Bull, Bear, Sideways, etc.)
  2. Não-supervisionado: clustering (K-Means sobre features de mercado)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Classificação Supervisionada
# ---------------------------------------------------------------------------

def classify_regime_supervised(df: pd.DataFrame, ma_period: int = 200) -> pd.Series:
    """
    Classifica o regime de mercado dia a dia usando regras explícitas.

    Regimes:
      - Bull Trend      : preço acima da MA, momentum positivo, baixa volatilidade
      - Bear Trend      : preço abaixo da MA, momentum negativo
      - Alta Volatilidade: volatilidade acima do 80º percentil histórico
      - Baixa Volatilidade: volatilidade abaixo do 20º percentil histórico
      - Recuperação     : drawdown diminuindo após período de queda
      - Capitulação     : drawdown acelerando, momentum muito negativo
      - Sideways        : padrão residual

    Parameters
    ----------
    df : pd.DataFrame
        Colunas: close. Index: DatetimeIndex.
    ma_period : int
        Período da média para filtro de tendência.

    Returns
    -------
    pd.Series com rótulo de regime para cada data.
    """
    close = df["close"]

    # Features básicas
    ma      = close.rolling(ma_period).mean()
    ret_1m  = close.pct_change(21)
    ret_3m  = close.pct_change(63)
    vol_21  = close.pct_change().rolling(21).std() * np.sqrt(252)

    # Drawdown
    roll_max = close.cummax()
    dd = (close - roll_max) / roll_max

    # Percentis de volatilidade
    vol_p80 = vol_21.rolling(252).quantile(0.80)
    vol_p20 = vol_21.rolling(252).quantile(0.20)

    regime = pd.Series("Sideways", index=df.index)

    for i in range(ma_period, len(df)):
        r1 = ret_1m.iloc[i]
        r3 = ret_3m.iloc[i]
        v  = vol_21.iloc[i]
        vp80 = vol_p80.iloc[i]
        vp20 = vol_p20.iloc[i]
        above_ma = close.iloc[i] > ma.iloc[i]
        drawdown = dd.iloc[i]

        if pd.isna(r1) or pd.isna(v):
            regime.iloc[i] = "Sideways"
            continue

        # Prioridade de classificação
        if v > vp80:
            if r1 < -0.10:
                regime.iloc[i] = "Capitulação"
            else:
                regime.iloc[i] = "Alta Volatilidade"
        elif v < vp20:
            if above_ma and r3 > 0:
                regime.iloc[i] = "Euforia"
            else:
                regime.iloc[i] = "Baixa Volatilidade"
        elif above_ma and r1 > 0 and r3 > 0:
            regime.iloc[i] = "Bull Trend"
        elif not above_ma and r1 < 0 and r3 < 0:
            regime.iloc[i] = "Bear Trend"
        elif drawdown < -0.10 and r1 > r3:
            regime.iloc[i] = "Recuperação"
        else:
            regime.iloc[i] = "Sideways"

    return regime


# ---------------------------------------------------------------------------
# 2. Clustering Não-Supervisionado
# ---------------------------------------------------------------------------

def classify_regime_unsupervised(
    df: pd.DataFrame,
    n_clusters: int = 5,
    ma_period: int = 50,
    random_state: int = 42,
) -> pd.Series:
    """
    Detecta regimes via K-Means clustering sobre features de mercado.

    Features usadas:
      - Retorno 1 mês
      - Retorno 3 meses
      - Volatilidade realizada 21d
      - Distância da média (z-score)
      - Drawdown atual

    Parameters
    ----------
    n_clusters : int
        Número de regimes (clusters).

    Returns
    -------
    pd.Series com índice numérico de regime (0 a n_clusters-1).
    """
    close = df["close"]

    features = pd.DataFrame(index=df.index)
    features["ret_1m"]     = close.pct_change(21)
    features["ret_3m"]     = close.pct_change(63)
    features["vol_21"]     = close.pct_change().rolling(21).std() * np.sqrt(252)
    features["dist_ma"]    = (close / close.rolling(ma_period).mean()) - 1
    features["drawdown"]   = (close - close.cummax()) / close.cummax()

    features = features.dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    regime = pd.Series(labels, index=features.index, name="regime_cluster")

    # Nomear clusters por retorno médio (cluster 0 = menor retorno)
    cluster_ret = {}
    for c in range(n_clusters):
        mask = regime == c
        cluster_ret[c] = features.loc[mask, "ret_3m"].mean()

    sorted_clusters = sorted(cluster_ret, key=lambda x: cluster_ret[x])
    name_map = {
        sorted_clusters[0]: "Bear",
        sorted_clusters[-1]: "Bull",
    }
    if n_clusters >= 3:
        name_map[sorted_clusters[len(sorted_clusters)//2]] = "Neutral"

    regime_named = regime.map(lambda x: name_map.get(x, f"Cluster_{x}"))
    return regime_named


# ---------------------------------------------------------------------------
# Análise de performance por regime
# ---------------------------------------------------------------------------

def performance_by_regime(
    strategy_returns: pd.Series,
    regime: pd.Series,
) -> pd.DataFrame:
    """
    Calcula o retorno médio da estratégia em cada regime.

    Parameters
    ----------
    strategy_returns : pd.Series
        Retornos diários da estratégia.
    regime : pd.Series
        Regime de mercado (mesmo índice).

    Returns
    -------
    pd.DataFrame com métricas por regime.
    """
    combined = pd.DataFrame({
        "return": strategy_returns,
        "regime": regime,
    }).dropna()

    rows = []
    for r in combined["regime"].unique():
        mask = combined["regime"] == r
        ret  = combined.loc[mask, "return"]
        rows.append({
            "regime":          r,
            "n_days":          int(mask.sum()),
            "mean_daily_ret":  float(ret.mean()),
            "ann_return":      float((1 + ret.mean()) ** 252 - 1),
            "volatility":      float(ret.std() * np.sqrt(252)),
            "pct_positive":    float((ret > 0).mean()),
        })

    df_result = pd.DataFrame(rows).set_index("regime").sort_values("ann_return", ascending=False)
    return df_result
