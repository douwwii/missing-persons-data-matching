from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from identity_cleaning.normalization import compact_text, is_missing, normalize_text

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only without optional dependency
    fuzz = None
    process = None
    from difflib import SequenceMatcher


@dataclass(frozen=True)
class CityReference:
    ref_id: str
    city_raw: str
    department_raw: str
    latitude: float
    longitude: float
    country_raw: str = ""
    city_normalized: str = ""
    city_compact: str = ""
    department_normalized: str = ""
    country_normalized: str = ""

    @classmethod
    def create(
        cls,
        ref_id: str,
        city_raw: object,
        department_raw: object,
        latitude: object,
        longitude: object,
        country_raw: object = "",
    ) -> "CityReference":
        return cls(
            ref_id=str(ref_id),
            city_raw="" if is_missing(city_raw) else str(city_raw).strip(),
            department_raw="" if is_missing(department_raw) else str(department_raw).strip(),
            latitude=float(latitude),
            longitude=float(longitude),
            country_raw="" if is_missing(country_raw) else str(country_raw).strip(),
            city_normalized=normalize_text(city_raw),
            city_compact=compact_text(city_raw),
            department_normalized=normalize_text(department_raw),
            country_normalized=normalize_text(country_raw),
        )


@dataclass(frozen=True)
class CityCandidate:
    reference: CityReference
    score: float
    explanation: str


@dataclass(frozen=True)
class CityResolution:
    status: str
    candidates: list[CityCandidate]
    selected: CityReference | None
    score: float | None
    explanation: str
    needs_human_validation: bool = False


def build_city_references(
    df: pd.DataFrame,
    city_col: str,
    department_col: str,
    latitude_col: str,
    longitude_col: str,
    country_col: str | None = None,
) -> list[CityReference]:
    references: list[CityReference] = []
    for idx, row in df.iterrows():
        city = row.get(city_col)
        latitude = row.get(latitude_col)
        longitude = row.get(longitude_col)
        if is_missing(city) or is_missing(latitude) or is_missing(longitude):
            continue
        try:
            reference = CityReference.create(
                ref_id=str(idx),
                city_raw=city,
                department_raw=row.get(department_col),
                latitude=latitude,
                longitude=longitude,
                country_raw=row.get(country_col) if country_col else "",
            )
        except (TypeError, ValueError):
            continue
        references.append(reference)
    return references


def _country_matches(reference: CityReference, country_normalized: str) -> bool:
    return not country_normalized or not reference.country_normalized or reference.country_normalized == country_normalized


_LOCATION_STOPWORDS = {"de", "del", "la", "le", "el", "les", "los", "las", "du", "des", "da", "do", "dos", "das", "y", "e"}


def _meaningful_tokens(value: str) -> list[str]:
    raw_tokens = [token for token in value.split() if token not in _LOCATION_STOPWORDS]
    tokens = raw_tokens or value.split()
    merged: list[str] = []
    single_letters: list[str] = []
    for token in tokens:
        if len(token) == 1:
            single_letters.append(token)
            continue
        if single_letters:
            merged.append("".join(single_letters))
            single_letters = []
        merged.append(token)
    if single_letters:
        merged.append("".join(single_letters))
    return merged


def _token_set_score(left: str, right: str) -> float:
    left_tokens = set(_meaningful_tokens(left))
    right_tokens = set(_meaningful_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall) * 100


def _contiguous_initials(tokens: list[str]) -> set[str]:
    initials: set[str] = set()
    for start in range(len(tokens)):
        current = ""
        for token in tokens[start:]:
            current += token[0]
            if len(current) > 1:
                initials.add(current)
    return initials


def _abbreviation_score(left: str, right: str) -> float:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0

    def coverage(source_tokens: list[str], target_tokens: list[str]) -> float:
        target_set = set(target_tokens)
        target_initials = _contiguous_initials(target_tokens)
        matched = 0.0
        for token in source_tokens:
            if token in target_set:
                matched += 1.0
            elif 1 < len(token) <= 5 and token in target_initials:
                matched += 1.0
        return matched / len(source_tokens) * 100

    return max(coverage(left_tokens, right_tokens), coverage(right_tokens, left_tokens))


def _query_token_coverage(query: str, reference: str) -> float:
    query_tokens = _meaningful_tokens(query)
    reference_tokens = _meaningful_tokens(reference)
    if not query_tokens or not reference_tokens:
        return 0.0
    reference_set = set(reference_tokens)
    reference_initials = _contiguous_initials(reference_tokens)
    matched = 0.0
    for token in query_tokens:
        if token in reference_set:
            matched += 1.0
        elif 1 < len(token) <= 5 and token in reference_initials:
            matched += 1.0
    return matched / len(query_tokens) * 100


def _text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    token_score = _token_set_score(left, right)
    abbreviation_score = _abbreviation_score(left, right)
    if fuzz is not None:
        char_score = float(max(fuzz.ratio(left, right), fuzz.token_sort_ratio(left, right)))
    else:
        char_score = SequenceMatcher(None, left, right).ratio() * 100
    return max(char_score, token_score * 0.90 + abbreviation_score * 0.10, abbreviation_score * 0.94)


def _city_similarity(query_city: str, query_compact: str, reference: CityReference) -> float:
    compact_score = _text_similarity(query_compact, reference.city_compact)
    spaced_score = _text_similarity(query_city, reference.city_normalized)
    base_score = max(compact_score, spaced_score)
    query_coverage = _query_token_coverage(query_city, reference.city_normalized)

    if not query_compact or not reference.city_compact:
        return base_score

    shorter = min(len(query_compact), len(reference.city_compact))
    longer = max(len(query_compact), len(reference.city_compact))
    length_ratio = shorter / longer if longer else 1.0
    contains_bonus = 0.0
    containment_score = 0.0

    # Containment is useful for names like "cali" vs "santiago de cali", but it
    # is capped so a substring alone cannot become a verified match.
    if query_compact in reference.city_compact:
        contains_bonus = 8.0
        containment_score = 62.0 + length_ratio * 28.0
        if length_ratio < 0.82:
            base_score = min(base_score, length_ratio * 100)
    elif reference.city_compact in query_compact:
        contains_bonus = 3.0
        if length_ratio < 0.82:
            base_score = min(base_score, length_ratio * 100)

    if query_coverage >= 80:
        base_score = max(base_score, 86.0 + query_coverage * 0.10)
    elif 0 < query_coverage < 80:
        base_score = min(base_score, 70.0 + query_coverage * 0.15)

    return min(100.0, max(base_score * 0.80 + length_ratio * 12 + contains_bonus, containment_score, query_coverage * 0.94))


def _candidate_score(query_city: str, query_compact: str, query_department: str, reference: CityReference) -> tuple[float, float, float]:
    city_score = _city_similarity(query_city, query_compact, reference)
    if not query_department:
        return city_score, city_score, 0.0

    department_score = _text_similarity(query_department, reference.department_normalized)
    if not reference.department_normalized:
        combined_score = city_score * 0.80
    else:
        combined_score = city_score * 0.65 + department_score * 0.35
    return combined_score, city_score, department_score


def resolve_city(
    city_raw: object,
    department_raw: object,
    references: Iterable[CityReference],
    country_raw: object = "",
    fuzzy_limit: int = 5,
    fuzzy_threshold: float = 58,
) -> CityResolution:
    if is_missing(city_raw):
        return CityResolution("missing_city", [], None, None, "Ville de disparition absente.", False)

    refs = list(references)
    city_norm = normalize_text(city_raw)
    city_compact = compact_text(city_raw)
    dept_norm = normalize_text(department_raw)
    country_norm = normalize_text(country_raw)
    scoped_refs = [ref for ref in refs if _country_matches(ref, country_norm)]

    exact_city = [ref for ref in scoped_refs if ref.city_normalized == city_norm or ref.city_compact == city_compact]
    if dept_norm:
        exact_city_dept = [ref for ref in exact_city if ref.department_normalized == dept_norm]
        if len(exact_city_dept) == 1:
            ref = exact_city_dept[0]
            return CityResolution(
                "exact_city_department_match",
                [CityCandidate(ref, 100.0, "Ville et departement identiques apres normalisation.")],
                ref,
                100.0,
                "Ville et departement verifies dans la base de reference.",
                False,
            )
        if len(exact_city_dept) > 1:
            candidates = [
                CityCandidate(ref, 100.0, "Plusieurs references partagent la meme ville et le meme departement.")
                for ref in exact_city_dept
            ]
            return CityResolution("ambiguous_city_department_match", candidates, None, 100.0, "Plusieurs villes de reference possibles.", True)

    if len(exact_city) == 1:
        ref = exact_city[0]
        if dept_norm:
            department_score = _text_similarity(dept_norm, ref.department_normalized)
            candidate = CityCandidate(
                ref,
                round(72.0 + department_score * 0.28, 2),
                f"Ville identique; departement similaire ({department_score:.0f}/100).",
            )
            return CityResolution(
                "city_match_department_mismatch",
                [candidate],
                None,
                candidate.score,
                "Validation humaine requise: la ville correspond, mais le departement n'est pas identique.",
                True,
            )
        return CityResolution(
            "exact_city_match",
            [CityCandidate(ref, 96.0, "Ville identique apres normalisation; departement absent ou compatible sans confirmation exacte.")],
            ref,
            96.0,
            "Ville unique trouvee dans la base de reference.",
            False,
        )
    if len(exact_city) > 1:
        if dept_norm:
            ranked = []
            for ref in exact_city:
                department_score = _text_similarity(dept_norm, ref.department_normalized)
                ranked.append(
                    CityCandidate(
                        ref,
                        round(70.0 + department_score * 0.3, 2),
                        "Ville identique; score departement utilise pour departager les candidats.",
                    )
                )
            ranked.sort(key=lambda candidate: candidate.score, reverse=True)
            return CityResolution(
                "ambiguous_city_match",
                ranked[:fuzzy_limit],
                None,
                ranked[0].score,
                "Plusieurs villes portent ce nom; le departement ne permet pas de choisir automatiquement.",
                True,
            )
        candidates = [
            CityCandidate(ref, 95.0, "Ville identique, mais plusieurs departements possibles.")
            for ref in exact_city
        ]
        return CityResolution("ambiguous_city_match", candidates, None, 95.0, "Plusieurs villes portent ce nom.", True)

    scored: list[CityCandidate] = []
    for ref in scoped_refs:
        score, city_score, department_score = _candidate_score(city_norm, city_compact, dept_norm, ref)
        department_is_helpful = bool(dept_norm and department_score >= 72)
        city_is_plausible = city_score >= 50
        if score >= fuzzy_threshold and (city_is_plausible or department_is_helpful):
            explanation = f"Ville similaire ({city_score:.0f}/100)"
            if dept_norm:
                explanation += f", departement similaire ({department_score:.0f}/100)"
            scored.append(CityCandidate(ref, round(score, 2), explanation + "."))
    scored.sort(key=lambda candidate: candidate.score, reverse=True)
    candidates = scored[:fuzzy_limit]

    if not candidates:
        return CityResolution("unresolved_city", [], None, None, "Aucune ville plausible trouvee dans la base de reference.", True)

    return CityResolution(
        "fuzzy_city_candidates",
        candidates,
        None,
        candidates[0].score,
        "Validation humaine requise pour choisir parmi les candidats.",
        True,
    )
