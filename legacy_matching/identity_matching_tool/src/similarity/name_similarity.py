from __future__ import annotations

from src.models import MissingPersonRecord, NameSimilarityResult
from src.preprocessing.safe_parsing import none_if_missing, parse_name_tokens

PRIMARY_NAME_WEIGHT = 0.40
REORDERED_NAME_WEIGHT = 0.45
TOKEN_ALIGNMENT_WEIGHT = 0.15
NAME_SCORE_POWER = 2.0


def compute_name_similarity(record_a: MissingPersonRecord, record_b: MissingPersonRecord) -> NameSimilarityResult:
    """Compute interpretable written-name similarity from already-normalized fields."""
    full_name_a = _clean_present_value(record_a.full_name_normalized)
    full_name_b = _clean_present_value(record_b.full_name_normalized)
    tokens_a = _record_tokens(record_a)
    tokens_b = _record_tokens(record_b)

    direct_structured_score = _structured_score(record_a, record_b)
    direct_full_name_score = None if direct_structured_score is not None else _score_if_available(full_name_a, full_name_b)
    reordered_full_name_score = _reordered_score(tokens_a, tokens_b)
    token_alignment_score, matched_tokens = _token_alignment_score(tokens_a, tokens_b)

    weighted_scores = _weighted_name_scores(
        direct_structured_score,
        direct_full_name_score,
        reordered_full_name_score,
        token_alignment_score,
    )
    if not weighted_scores:
        return NameSimilarityResult(
            score_final=0.0,
            direct_structured_score=None,
            direct_full_name_score=None,
            reordered_full_name_score=None,
            token_alignment_score=None,
            order_mismatch_suspected=False,
            best_method=None,
            matched_tokens=[],
            explanation="Name similarity cannot be computed: normalized name and tokens are missing.",
            status="missing_name",
        )

    score_final = sum(score * weight for score, weight, _ in weighted_scores) / sum(
        weight for _, weight, _ in weighted_scores
    )
    best_method = "weighted_name_score"
    order_mismatch_suspected = _order_mismatch_suspected(direct_full_name_score, reordered_full_name_score)
    explanation = _build_explanation(score_final, weighted_scores, order_mismatch_suspected, token_alignment_score)

    return NameSimilarityResult(
        score_final=_clamp(score_final),
        direct_structured_score=_clamp_optional(direct_structured_score),
        direct_full_name_score=_clamp_optional(direct_full_name_score),
        reordered_full_name_score=_clamp_optional(reordered_full_name_score),
        token_alignment_score=_clamp_optional(token_alignment_score),
        order_mismatch_suspected=order_mismatch_suspected,
        best_method=best_method,
        matched_tokens=matched_tokens,
        explanation=explanation,
        status="ok",
    )


def jaro_winkler_similarity(value_a: str, value_b: str) -> float:
    """Compute a Jaro-Winkler similarity score between two strings."""
    if value_a == value_b:
        return 1.0
    if not value_a or not value_b:
        return 0.0

    jaro = _jaro_similarity(value_a, value_b)
    prefix_length = 0
    for char_a, char_b in zip(value_a, value_b):
        if char_a != char_b:
            break
        prefix_length += 1
        if prefix_length == 4:
            break
    return _clamp(jaro + prefix_length * 0.1 * (1 - jaro))


def calibrated_name_similarity(value_a: str, value_b: str) -> float:
    """Apply the name-score calibration power to Jaro-Winkler output."""
    return jaro_winkler_similarity(value_a, value_b) ** NAME_SCORE_POWER


def _jaro_similarity(value_a: str, value_b: str) -> float:
    """Compute the base Jaro similarity before the Winkler prefix boost."""
    len_a = len(value_a)
    len_b = len(value_b)
    if len_a == 0 or len_b == 0:
        return 0.0

    match_distance = max(0, max(len_a, len_b) // 2 - 1)
    a_matches = [False] * len_a
    b_matches = [False] * len_b

    matches = 0
    for index_a, char_a in enumerate(value_a):
        start = max(0, index_a - match_distance)
        end = min(index_a + match_distance + 1, len_b)
        for index_b in range(start, end):
            if b_matches[index_b] or char_a != value_b[index_b]:
                continue
            a_matches[index_a] = True
            b_matches[index_b] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    index_b = 0
    for index_a in range(len_a):
        if not a_matches[index_a]:
            continue
        while not b_matches[index_b]:
            index_b += 1
        if value_a[index_a] != value_b[index_b]:
            transpositions += 1
        index_b += 1

    return (
        matches / len_a
        + matches / len_b
        + (matches - transpositions / 2) / matches
    ) / 3


def _structured_score(record_a: MissingPersonRecord, record_b: MissingPersonRecord) -> float | None:
    """Score first-name to first-name and last-name to last-name when both exist."""
    first_a = _clean_present_value(record_a.first_name_normalized)
    first_b = _clean_present_value(record_b.first_name_normalized)
    last_a = _clean_present_value(record_a.last_name_normalized)
    last_b = _clean_present_value(record_b.last_name_normalized)
    if not all([first_a, first_b, last_a, last_b]):
        return None
    return (calibrated_name_similarity(first_a, first_b) + calibrated_name_similarity(last_a, last_b)) / 2


def _score_if_available(value_a: str | None, value_b: str | None) -> float | None:
    """Score two strings only when both values are present."""
    if not value_a or not value_b:
        return None
    return calibrated_name_similarity(value_a, value_b)


def _reordered_score(tokens_a: list[str], tokens_b: list[str]) -> float | None:
    """Score full names after sorting their tokens alphabetically."""
    if not tokens_a or not tokens_b:
        return None
    reordered_a = " ".join(sorted(tokens_a))
    reordered_b = " ".join(sorted(tokens_b))
    return calibrated_name_similarity(reordered_a, reordered_b)


def _token_alignment_score(tokens_a: list[str], tokens_b: list[str]) -> tuple[float | None, list[dict[str, float | str]]]:
    """Greedily align closest tokens and penalize unmatched extra tokens."""
    if not tokens_a or not tokens_b:
        return None, []

    pairs: list[tuple[float, int, int]] = []
    for index_a, token_a in enumerate(tokens_a):
        for index_b, token_b in enumerate(tokens_b):
            pairs.append((calibrated_name_similarity(token_a, token_b), index_a, index_b))
    pairs.sort(reverse=True, key=lambda item: item[0])

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched_tokens: list[dict[str, float | str]] = []
    for score, index_a, index_b in pairs:
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matched_tokens.append(
            {
                "token_a": tokens_a[index_a],
                "token_b": tokens_b[index_b],
                "score": _clamp(score),
            }
        )

    if not matched_tokens:
        return 0.0, []

    average_score = sum(float(match["score"]) for match in matched_tokens) / len(matched_tokens)
    length_penalty = min(len(tokens_a), len(tokens_b)) / max(len(tokens_a), len(tokens_b))
    return _clamp(average_score * length_penalty), matched_tokens


def _weighted_name_scores(
    direct_structured_score: float | None,
    direct_full_name_score: float | None,
    reordered_full_name_score: float | None,
    token_alignment_score: float | None,
) -> list[tuple[float, float, str]]:
    """Return available name subscores with their configured contribution weights."""
    weighted_scores: list[tuple[float, float, str]] = []
    if direct_structured_score is not None:
        weighted_scores.append((direct_structured_score, PRIMARY_NAME_WEIGHT, "direct_structured"))
    elif direct_full_name_score is not None:
        weighted_scores.append((direct_full_name_score, PRIMARY_NAME_WEIGHT, "direct_full_name"))
    if reordered_full_name_score is not None:
        weighted_scores.append((reordered_full_name_score, REORDERED_NAME_WEIGHT, "reordered_full_name"))
    if token_alignment_score is not None:
        weighted_scores.append((token_alignment_score, TOKEN_ALIGNMENT_WEIGHT, "token_alignment"))
    return [(score, weight, label) for score, weight, label in weighted_scores if weight > 0]


def _record_tokens(record: MissingPersonRecord) -> list[str]:
    """Return normalized name tokens, falling back to full_name_normalized splitting."""
    tokens = parse_name_tokens(record.name_tokens)
    if tokens:
        return tokens
    full_name = _clean_present_value(record.full_name_normalized)
    if not full_name:
        return []
    return [token for token in full_name.split() if token]


def _clean_present_value(value: object) -> str | None:
    """Convert a present value to stripped text and keep missing values as None."""
    value = none_if_missing(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _order_mismatch_suspected(direct_score: float | None, reordered_score: float | None) -> bool:
    """Detect names that score much better after token reordering."""
    if direct_score is None or reordered_score is None:
        return False
    return direct_score < 0.88 and reordered_score >= 0.92 and reordered_score - direct_score >= 0.08


def _build_explanation(
    score_final: float,
    weighted_scores: list[tuple[float, float, str]],
    order_mismatch_suspected: bool,
    token_alignment_score: float | None,
) -> str:
    """Build a plain-English explanation for the written-name score."""
    components = ", ".join(f"{label}={score:.3f} weight={weight:.2f}" for score, weight, label in weighted_scores)
    parts = [f"Final name score {score_final:.3f}, computed as a weighted average ({components})."]
    if order_mismatch_suspected:
        parts.append(
            "The direct score is lower, but the reordered score is high: "
            "name elements may be inverted or in a different order."
        )
    if token_alignment_score is not None and token_alignment_score >= 0.80:
        parts.append("Token alignment indicates that several main name elements match.")
    return " ".join(parts)


def _clamp_optional(value: float | None) -> float | None:
    """Clamp a score when present and preserve None when absent."""
    if value is None:
        return None
    return _clamp(value)


def _clamp(value: float) -> float:
    """Constrain a numeric score to the inclusive 0..1 range."""
    return max(0.0, min(1.0, float(value)))
