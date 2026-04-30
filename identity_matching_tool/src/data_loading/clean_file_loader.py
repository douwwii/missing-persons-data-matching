from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models import MissingPersonRecord
from src.preprocessing.safe_parsing import (
    none_if_missing,
    parse_name_tokens,
    parse_optional_date,
    parse_optional_float,
)


class CleanFileLoadingError(ValueError):
    """Raised when a clean CSV cannot be loaded with a user-readable message."""


def load_clean_csv(uploaded_file_or_path: Any) -> pd.DataFrame:
    try:
        if isinstance(uploaded_file_or_path, (str, Path)):
            return pd.read_csv(uploaded_file_or_path)
        return pd.read_csv(uploaded_file_or_path)
    except UnicodeDecodeError as exc:
        raise CleanFileLoadingError(
            "Le fichier CSV ne peut pas etre lu avec l'encodage attendu. "
            "Verifiez que le fichier exporte par l'etape 1 est bien un CSV lisible."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise CleanFileLoadingError("Le fichier CSV est vide.") from exc
    except pd.errors.ParserError as exc:
        raise CleanFileLoadingError(
            "Le fichier CSV semble mal forme. Verifiez le separateur, les guillemets et l'export de l'etape 1."
        ) from exc
    except OSError as exc:
        raise CleanFileLoadingError(f"Le fichier CSV est introuvable ou inaccessible : {exc}") from exc
    except Exception as exc:
        raise CleanFileLoadingError(f"Impossible de charger le fichier CSV : {exc}") from exc


def dataframe_to_records(df: pd.DataFrame) -> list[MissingPersonRecord]:
    records: list[MissingPersonRecord] = []
    for _, row in df.iterrows():
        raw_data = row.where(pd.notna(row), None).to_dict()
        records.append(
            MissingPersonRecord(
                case_id=str(none_if_missing(row.get("case_id")) or ""),
                full_name_raw=str(none_if_missing(row.get("full_name_raw")) or ""),
                full_name_normalized=str(none_if_missing(row.get("full_name_normalized")) or ""),
                first_name_raw=_optional_str(row.get("first_name_raw")),
                first_name_normalized=_optional_str(row.get("first_name_normalized")),
                last_name_raw=_optional_str(row.get("last_name_raw")),
                last_name_normalized=_optional_str(row.get("last_name_normalized")),
                name_tokens=parse_name_tokens(row.get("name_tokens")),
                disappearance_city_verified=str(none_if_missing(row.get("disappearance_city_verified")) or ""),
                disappearance_department_verified=_optional_str(row.get("disappearance_department_verified")),
                latitude=parse_optional_float(row.get("latitude")),
                longitude=parse_optional_float(row.get("longitude")),
                disappearance_date_parsed=parse_optional_date(row.get("disappearance_date_parsed")),
                raw_data=raw_data,
            )
        )
    return records


def _optional_str(value: Any) -> str | None:
    value = none_if_missing(value)
    if value is None:
        return None
    return str(value)

