from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd

from src.models import MissingPersonRecord, PhoneticSimilarityResult
from src.preprocessing.safe_parsing import none_if_missing

PHONETIC_CACHE_DIR = Path("cache")
PHONETIC_CACHE_FILE = PHONETIC_CACHE_DIR / "phonetic_ipa_cache.csv"
_PANPHON_DISTANCE: Any | None = None
_IPA_CACHE_FIELD = "_phonetic_ipa_cache"
_IPA_CACHE_VERSION = "reordered_ipa_tokens_v1"
PHONETIC_SCORE_POWER = 2.5


def compute_phonetic_similarity(
    record_a: MissingPersonRecord,
    record_b: MissingPersonRecord,
    lang: str = "spa-Latn",
) -> PhoneticSimilarityResult:
    name_a = _clean_present_value(record_a.full_name_normalized)
    name_b = _clean_present_value(record_b.full_name_normalized)
    if not name_a or not name_b:
        return PhoneticSimilarityResult(
            score_final=None,
            explanation="Similarite phonetique non calculable : nom complet normalise manquant.",
            status="missing_name",
        )

    ipa_a = _cached_ipa(record_a, lang)
    ipa_b = _cached_ipa(record_b, lang)
    if ipa_a is None or ipa_b is None:
        ipa_a_result = _compute_ipa(name_a, lang)
        ipa_b_result = _compute_ipa(name_b, lang)
        _store_record_ipa(record_a, lang, ipa_a_result)
        _store_record_ipa(record_b, lang, ipa_b_result)
        if ipa_a_result["status"] != "ok":
            return PhoneticSimilarityResult(
                score_final=None,
                explanation=str(ipa_a_result["explanation"]),
                status=str(ipa_a_result["status"]),
            )
        if ipa_b_result["status"] != "ok":
            return PhoneticSimilarityResult(
                score_final=None,
                explanation=str(ipa_b_result["explanation"]),
                status=str(ipa_b_result["status"]),
            )
        ipa_a = str(ipa_a_result["ipa"])
        ipa_b = str(ipa_b_result["ipa"])

    distance_result = _compute_panphon_distance(ipa_a, ipa_b)
    if distance_result["status"] != "ok":
        return PhoneticSimilarityResult(
            score_final=None,
            ipa_a=ipa_a,
            ipa_b=ipa_b,
            explanation=str(distance_result["explanation"]),
            status=str(distance_result["status"]),
        )

    distance = float(distance_result["distance"])
    normalizer = max(len(ipa_a), len(ipa_b), 1)
    raw_score = 1 - min(distance / normalizer, 1)
    score = raw_score ** PHONETIC_SCORE_POWER
    return PhoneticSimilarityResult(
        score_final=_clamp(score),
        ipa_a=ipa_a,
        ipa_b=ipa_b,
        explanation=(
            "Comparaison phonetique sur les tokens du nom convertis en IPA puis tries par ordre IPA, "
            f"distance {distance:.3f}, score brut {raw_score:.3f}, score calibre puissance "
            f"{PHONETIC_SCORE_POWER:.1f}. L'enqueteur doit verifier si ce reordonnancement est pertinent."
        ),
        status="ok",
    )


def precompute_ipa_for_records(
    records: list[MissingPersonRecord],
    lang: str = "spa-Latn",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, int]:
    """Precompute and persist IPA per record; pair scoring still uses PanPhon distance."""
    cache_lookup = _load_ipa_cache_lookup()
    rows_to_store: list[dict[str, Any]] = []
    stats = {"total": len(records), "cached": 0, "computed": 0, "missing": 0, "unavailable": 0}

    transliterator_result = _build_transliterator(lang)
    for index, record in enumerate(records, start=1):
        name = _clean_present_value(record.full_name_normalized)
        if not name:
            result = {
                "status": "missing_name",
                "ipa": None,
                "explanation": "Similarite phonetique non calculable : nom complet normalise manquant.",
            }
            _store_record_ipa(record, lang, result)
            stats["missing"] += 1
        else:
            cache_key = _make_ipa_cache_key(record.case_id, name, lang)
            cached = cache_lookup.get(cache_key)
            if cached is not None:
                result = {
                    "status": str(cached.get("status") or "unavailable"),
                    "ipa": _clean_present_value(cached.get("ipa")),
                    "explanation": str(cached.get("explanation") or "IPA recupere depuis le cache."),
                }
                _store_record_ipa(record, lang, result)
                stats["cached"] += 1
            elif transliterator_result["status"] != "ok":
                result = {
                    "status": transliterator_result["status"],
                    "ipa": None,
                    "explanation": transliterator_result["explanation"],
                }
                _store_record_ipa(record, lang, result)
                rows_to_store.append(_ipa_cache_row(record.case_id, name, lang, result))
                stats["unavailable"] += 1
            else:
                try:
                    ipa = _reordered_ipa_for_name(name, transliterator_result["transliterator"])
                    result = {
                        "status": "ok",
                        "ipa": ipa,
                        "explanation": "IPA pre-calcule depuis les tokens du nom tries par ordre IPA.",
                    }
                    stats["computed"] += 1
                except Exception as exc:
                    result = {
                        "status": "unavailable",
                        "ipa": None,
                        "explanation": f"IPA non calculable : erreur Epitran ({exc}).",
                    }
                    stats["unavailable"] += 1
                _store_record_ipa(record, lang, result)
                rows_to_store.append(_ipa_cache_row(record.case_id, name, lang, result))

        if progress_callback is not None:
            progress_callback("Pre-calcul IPA", index, len(records))

    _store_ipa_cache_rows(rows_to_store)
    return stats


def _compute_ipa(name: str, lang: str) -> dict[str, Any]:
    transliterator_result = _build_transliterator(lang)
    if transliterator_result["status"] != "ok":
        return transliterator_result
    try:
        return {
            "status": "ok",
            "ipa": _reordered_ipa_for_name(name, transliterator_result["transliterator"]),
            "explanation": "IPA calcule depuis les tokens du nom tries par ordre IPA.",
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "ipa": None,
            "explanation": f"Similarite phonetique non calculable : erreur Epitran ({exc}).",
        }


def _build_transliterator(lang: str) -> dict[str, Any]:
    try:
        import epitran
    except ImportError:
        return {
            "status": "unavailable",
            "ipa": None,
            "explanation": "Similarite phonetique indisponible : Epitran n'est pas installe.",
        }

    try:
        with _panphon_utf8_csv_reads():
            transliterator = epitran.Epitran(lang)
    except UnicodeDecodeError as exc:
        return {
            "status": "unavailable",
            "ipa": None,
            "explanation": (
                "Similarite phonetique non calculable : erreur d'encodage lors du chargement "
                f"des ressources Epitran/PanPhon ({exc})."
            ),
        }
    except Exception:
        return {
            "status": "invalid_language",
            "ipa": None,
            "explanation": f"Similarite phonetique non calculable : langue non supportee ou invalide ({lang}).",
        }
    return {"status": "ok", "transliterator": transliterator, "explanation": ""}


def _compute_panphon_distance(ipa_a: str, ipa_b: str) -> dict[str, Any]:
    global _PANPHON_DISTANCE
    try:
        import panphon.distance
    except ImportError:
        return {
            "status": "unavailable",
            "distance": None,
            "explanation": "Similarite phonetique indisponible : PanPhon n'est pas installe.",
        }

    try:
        with _panphon_utf8_csv_reads():
            if _PANPHON_DISTANCE is None:
                _PANPHON_DISTANCE = panphon.distance.Distance()
            distance = _PANPHON_DISTANCE.feature_edit_distance(ipa_a, ipa_b)
    except UnicodeDecodeError as exc:
        return {
            "status": "unavailable",
            "distance": None,
            "explanation": (
                "Similarite phonetique non calculable : erreur d'encodage lors du chargement "
                f"des ressources PanPhon ({exc})."
            ),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "distance": None,
            "explanation": f"Similarite phonetique non calculable : erreur PanPhon ({exc}).",
        }
    return {"status": "ok", "distance": distance, "explanation": ""}


def _cached_ipa(record: MissingPersonRecord, lang: str) -> str | None:
    cache = record.raw_data.get(_IPA_CACHE_FIELD)
    if not isinstance(cache, dict):
        return None
    entry = cache.get(lang)
    if not isinstance(entry, dict) or entry.get("status") != "ok":
        return None
    return _clean_present_value(entry.get("ipa"))


def _store_record_ipa(record: MissingPersonRecord, lang: str, result: dict[str, Any]) -> None:
    cache = record.raw_data.setdefault(_IPA_CACHE_FIELD, {})
    if isinstance(cache, dict):
        cache[lang] = {
            "status": result.get("status"),
            "ipa": result.get("ipa"),
            "explanation": result.get("explanation"),
        }


def _load_ipa_cache_lookup() -> dict[str, dict[str, Any]]:
    if not PHONETIC_CACHE_FILE.exists():
        return {}
    try:
        cache = pd.read_csv(PHONETIC_CACHE_FILE, dtype=str, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return {}
    if cache.empty or "cache_key" not in cache.columns:
        return {}
    cache = cache.drop_duplicates(subset=["cache_key"], keep="last")
    return {str(row["cache_key"]): row.to_dict() for _, row in cache.iterrows()}


def _store_ipa_cache_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    PHONETIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame()
    if PHONETIC_CACHE_FILE.exists():
        try:
            existing = pd.read_csv(PHONETIC_CACHE_FILE, dtype=str, low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
            existing = pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    updated = updated.drop_duplicates(subset=["cache_key"], keep="last")
    updated.to_csv(PHONETIC_CACHE_FILE, index=False)


def _ipa_cache_row(case_id: str, name: str, lang: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "lang": lang,
        "name_hash": _hash_text(name),
        "cache_key": _make_ipa_cache_key(case_id, name, lang),
        "ipa": result.get("ipa"),
        "status": result.get("status"),
        "explanation": result.get("explanation"),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def _make_ipa_cache_key(case_id: str, name: str, lang: str) -> str:
    return f"{_IPA_CACHE_VERSION}::{lang}::{case_id}::{_hash_text(name)}"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reordered_ipa_for_name(name: str, transliterator: Any) -> str:
    token_ipas = []
    for token in name.split():
        ipa = _clean_present_value(transliterator.transliterate(token))
        if ipa:
            token_ipas.append(ipa)
    if not token_ipas:
        return _clean_present_value(transliterator.transliterate(name)) or ""
    return " ".join(sorted(token_ipas))


@contextmanager
def _panphon_utf8_csv_reads() -> Iterator[None]:
    """Force UTF-8 for PanPhon's internal CSV resources on Windows locales."""
    try:
        import pandas as pd
    except ImportError:
        yield
        return

    original_read_csv = pd.read_csv
    original_path_open = Path.open

    def read_csv_with_utf8(*args, **kwargs):
        if "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return original_read_csv(*args, **kwargs)

    def path_open_with_utf8(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if (
            "b" not in mode
            and encoding is None
            and self.suffix.lower() == ".csv"
            and "panphon" in {part.lower() for part in self.parts}
        ):
            encoding = "utf-8"
        return original_path_open(self, mode, buffering, encoding, errors, newline)

    pd.read_csv = read_csv_with_utf8
    Path.open = path_open_with_utf8
    try:
        yield
    finally:
        pd.read_csv = original_read_csv
        Path.open = original_path_open


def _clean_present_value(value: object) -> str | None:
    value = none_if_missing(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
