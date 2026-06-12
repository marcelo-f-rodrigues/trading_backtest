"""
data/loader.py
--------------
Carregamento e normalização dos CSVs de cada ativo.

Suporta dois formatos de CSV:

1. Formato "padrão": colunas Date, Open, High, Low, Close, Volume,
   números com ponto decimal, datas ISO (YYYY-MM-DD).

2. Formato "Investing.com" (em português): colunas
   Data, Último, Abertura, Máxima, Mínima, Vol., Var%,
   números no formato "1.234,56" (ponto = milhar, vírgula = decimal),
   volume como "12,34K" / "1,5M" / "1,2B", datas "dd.mm.yyyy".

O loader detecta automaticamente o formato e os arquivos são localizados
por "alias" (substring, case-insensitive) configurado em config.yaml,
já que nomes de arquivos baixados (ex: "Dados Históricos - Bitcoin.csv")
raramente coincidem com o nome do ativo (ex: "BTCUSD").
"""

import os
import re
import yaml
import pandas as pd
import numpy as np
from typing import Optional, List


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Parsers auxiliares para o formato Investing.com
# ---------------------------------------------------------------------------

# Possíveis nomes de colunas (PT/EN) para cada campo padronizado
_COLUMN_CANDIDATES = {
    "date":   ["data", "date"],
    "close":  ["último", "ultimo", "close", "price", "fechamento"],
    "open":   ["abertura", "open"],
    "high":   ["máxima", "maxima", "high"],
    "low":    ["mínima", "minima", "low"],
    "volume": ["vol.", "vol", "volume"],
}


def _normalize_colname(col: str) -> str:
    return str(col).strip().lower()


def _parse_investing_number(value) -> float:
    """
    Converte strings no formato Investing.com para float.

    Exemplos:
        "64.123,45"  -> 64123.45
        "1,234.56"   -> 1234.56  (também suportado, caso o arquivo venha em EN)
        "1,5K"       -> 1500.0
        "2,3M"       -> 2300000.0
        "1,2B"       -> 1200000000.0
        "1.234"      -> 1234.0
        "-1,23%"     -> -0.0123  (percentuais são tratados separadamente)
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if s == "" or s == "-":
        return np.nan

    s = s.replace("%", "")

    multiplier = 1.0
    if s and s[-1].upper() in ("K", "M", "B"):
        suffix = s[-1].upper()
        multiplier = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[suffix]
        s = s[:-1]

    s = s.strip()

    has_comma = "," in s
    has_dot = "." in s

    if has_comma and has_dot:
        # Determina qual é o separador decimal pela posição (o último é o decimal)
        if s.rfind(",") > s.rfind("."):
            # formato BR: 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:
            # formato EN: 1,234.56
            s = s.replace(",", "")
    elif has_comma:
        # Só vírgula: assume decimal BR -> 64123,45
        s = s.replace(",", ".")
    # Só ponto, ou nenhum separador: mantém como está

    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan


def _detect_date_format(sample: str) -> Optional[str]:
    """Detecta o formato de data a partir de uma amostra de string."""
    s = str(sample).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return "%Y-%m-%d"
    if re.match(r"^\d{2}\.\d{2}\.\d{4}", s):
        return "%d.%m.%Y"
    if re.match(r"^\d{2}/\d{2}/\d{4}", s):
        # Ambíguo (pode ser dd/mm/yyyy ou mm/dd/yyyy); assume dd/mm (BR)
        return "%d/%m/%Y"
    return None


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """Carrega e normaliza os dados de ativos a partir de CSVs."""

    def __init__(self, data_dir: str = "data/raw", config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        data_cfg = self.config.get("data", {})

        self.data_dir = data_cfg.get("directory", data_dir)
        self.asset_aliases: dict[str, str] = data_cfg.get("asset_aliases", {})

    # -------------------------------------------------------------------
    # Localização de arquivos
    # -------------------------------------------------------------------

    def _find_file(self, asset: str) -> str:
        """
        Procura o CSV de um ativo na pasta data_dir.

        Estratégia de busca (em ordem):
          1. Match exato do nome do arquivo (case-insensitive): ASSET.csv
          2. Alias configurado em config.yaml -> substring no nome do arquivo
          3. O próprio nome do ativo como substring no nome do arquivo
        """
        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(".csv")]

        # 1. Match exato
        for fname in files:
            name, _ = os.path.splitext(fname)
            if name.upper() == asset.upper():
                return os.path.join(self.data_dir, fname)

        # 2. Alias configurado
        alias = self.asset_aliases.get(asset.upper())
        if alias:
            alias_lower = alias.lower()
            for fname in files:
                if alias_lower in fname.lower():
                    return os.path.join(self.data_dir, fname)

        # 3. Nome do ativo como substring
        asset_lower = asset.lower()
        for fname in files:
            if asset_lower in fname.lower():
                return os.path.join(self.data_dir, fname)

        raise FileNotFoundError(
            f"CSV para '{asset}' não encontrado em '{self.data_dir}'. "
            f"Configure um alias em config.yaml -> data.asset_aliases.{asset.upper()}. "
            f"Arquivos disponíveis: {files}"
        )

    # -------------------------------------------------------------------
    # Carregamento e normalização
    # -------------------------------------------------------------------

    def load(
        self,
        asset: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Carrega o CSV de um ativo e retorna DataFrame normalizado.

        Colunas de saída padronizadas: open, high, low, close, volume
        (volume pode estar ausente, dependendo da fonte).
        Index: DatetimeIndex (date), ordenado ascendente.

        Parameters
        ----------
        asset : str
            Nome do ativo (ex: 'BTCUSD').
        start, end : str, optional
            Filtro de datas no formato YYYY-MM-DD.

        Returns
        -------
        pd.DataFrame
        """
        path = self._find_file(asset)
        df_raw = pd.read_csv(path, dtype=str)

        # Normaliza nomes de colunas para comparação
        col_lookup = {_normalize_colname(c): c for c in df_raw.columns}

        # Identifica colunas correspondentes
        col_map = {}
        for std_name, candidates in _COLUMN_CANDIDATES.items():
            for cand in candidates:
                if cand in col_lookup:
                    col_map[std_name] = col_lookup[cand]
                    break

        if "date" not in col_map or "close" not in col_map:
            raise ValueError(
                f"Não foi possível identificar colunas de data/close em '{path}'. "
                f"Colunas encontradas: {list(df_raw.columns)}"
            )

        # --- Datas ---
        date_series = df_raw[col_map["date"]].astype(str).str.strip()
        date_fmt = _detect_date_format(date_series.iloc[0])
        if date_fmt:
            dates = pd.to_datetime(date_series, format=date_fmt, errors="coerce")
        else:
            dates = pd.to_datetime(date_series, errors="coerce", dayfirst=True)

        # --- Construção do DataFrame normalizado ---
        df = pd.DataFrame(index=dates)
        df.index.name = "date"

        for std_name in ["open", "high", "low", "close", "volume"]:
            if std_name in col_map:
                df[std_name] = df_raw[col_map[std_name]].apply(_parse_investing_number).values

        df = df[~df.index.isna()]
        df = df.dropna(subset=["close"])
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Se faltar OHL, usa close como fallback (necessário para algumas estratégias)
        for col in ["open", "high", "low"]:
            if col not in df.columns:
                df[col] = df["close"]

        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]

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
            assets = self.config.get("assets", [])

        result = {}
        for asset in assets:
            try:
                df = self.load(asset, start=start, end=end)
                result[asset] = df
                print(f"[OK] {asset}: {len(df)} linhas carregadas "
                      f"({df.index.min().date()} -> {df.index.max().date()})")
            except (FileNotFoundError, ValueError) as e:
                print(f"[AVISO] {asset}: {e}")
        return result

    def available_files(self) -> List[str]:
        """Lista os arquivos CSV disponíveis na pasta data_dir."""
        return sorted(f for f in os.listdir(self.data_dir) if f.lower().endswith(".csv"))
