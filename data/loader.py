"""
data/loader.py
--------------
Carregamento e normalização dos CSVs de cada ativo.
Cada CSV deve ter colunas: Date, Open, High, Low, Close, Volume
(nomes configuráveis em config.yaml).
"""

import os
import yaml
import pandas as pd
from typing import Optional, List


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DataLoader:
    """Carrega e normaliza os dados de ativos a partir de CSVs."""

    def __init__(self, data_dir: str = "data/raw", config_path: str = "config.yaml"):
        self.data_dir = data_dir
        self.config = load_config(config_path)
        self.col_map = self.config["data"]["ohlcv_columns"]
        self.date_col = self.config["data"]["date_column"]
        self.date_fmt = self.config["data"]["date_format"]

    def load(
        self,
        asset: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Carrega o CSV de um ativo e retorna DataFrame normalizado.

        Colunas de saída padronizadas: open, high, low, close, volume
        Index: DatetimeIndex (Date)

        Parameters
        ----------
        asset : str
            Nome do ativo (ex: 'BTCUSD'). Procura por BTCUSD.csv na pasta data_dir.
        start : str, optional
            Data de início no formato YYYY-MM-DD.
        end : str, optional
            Data de fim no formato YYYY-MM-DD.

        Returns
        -------
        pd.DataFrame
        """
        path = self._find_file(asset)
        df = pd.read_csv(path, parse_dates=[self.date_col])
        df = df.rename(columns={self.date_col: "date"})
        df = df.rename(columns={v: k for k, v in self.col_map.items()})
        df = df.set_index("date").sort_index()

        # Manter apenas colunas OHLCV
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]

        # Remover linhas com close nulo
        df = df.dropna(subset=["close"])

        # Filtro de período
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]

        return df

    def load_all(
        self,
        assets: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Carrega todos os ativos configurados (ou lista específica).

        Returns
        -------
        dict : {asset_name: DataFrame}
        """
        if assets is None:
            assets = self.config["assets"]

        result = {}
        for asset in assets:
            try:
                result[asset] = self.load(asset, start=start, end=end)
                print(f"[OK] {asset}: {len(result[asset])} linhas carregadas.")
            except FileNotFoundError as e:
                print(f"[AVISO] {asset}: arquivo não encontrado — {e}")
        return result

    def _find_file(self, asset: str) -> str:
        """Procura o CSV do ativo (case-insensitive, múltiplas extensões)."""
        for fname in os.listdir(self.data_dir):
            name, ext = os.path.splitext(fname)
            if name.upper() == asset.upper() and ext.lower() == ".csv":
                return os.path.join(self.data_dir, fname)
        raise FileNotFoundError(
            f"CSV para '{asset}' não encontrado em '{self.data_dir}'. "
            f"Arquivos disponíveis: {os.listdir(self.data_dir)}"
        )

    def available_assets(self) -> List[str]:
        """Lista os ativos com CSV disponível na pasta data_dir."""
        assets = []
        for fname in os.listdir(self.data_dir):
            if fname.lower().endswith(".csv"):
                assets.append(os.path.splitext(fname)[0].upper())
        return sorted(assets)
