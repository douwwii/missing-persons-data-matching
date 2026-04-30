from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd


def read_table(path_or_buffer: str | Path | BinaryIO, sheet_name: str | int | None = 0) -> pd.DataFrame:
    name = getattr(path_or_buffer, "name", str(path_or_buffer))
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path_or_buffer, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path_or_buffer, sep=None, engine="python", encoding="utf-8-sig")
    raise ValueError("Format non supporte. Utilisez un fichier CSV, XLS ou XLSX.")


def list_excel_sheets(path_or_buffer: str | Path | BinaryIO) -> list[str]:
    return pd.ExcelFile(path_or_buffer).sheet_names

