"""Trade matching logic for agreement calculations."""

from typing import Any, Callable, TypeVar

from src.models.trade import group_trades_by_key

# Generic type for similarity results (float, tuple, or dataclass)
T = TypeVar("T")


def find_best_matches(
    trades_a: list[dict],
    trades_b: list[dict],
    similarity_fn: Callable[[dict, dict], T],
) -> list[tuple[dict, dict, T]]:
    """
    Find best matching pairs of trades using a greedy approach.

    For each possible pair, calculates similarity and then greedily
    matches pairs in order of highest similarity first.

    Args:
        trades_a: List of trades from annotator A
        trades_b: List of trades from annotator B
        similarity_fn: Function to calculate similarity between two trades

    Returns:
        List of (trade_a, trade_b, similarity_score) tuples
    """
    if not trades_a or not trades_b:
        return []

    # Calculate similarity matrix
    similarity_matrix: list[list[T]] = []
    for trade_a in trades_a:
        row = []
        for trade_b in trades_b:
            similarity = similarity_fn(trade_a, trade_b)
            row.append(similarity)
        similarity_matrix.append(row)

    # Collect all pairs with their similarities
    all_pairs: list[tuple[int, int, T]] = []
    for i in range(len(trades_a)):
        for j in range(len(trades_b)):
            all_pairs.append((i, j, similarity_matrix[i][j]))

    # Sort by similarity score (descending)
    all_pairs.sort(key=lambda x: _extract_score(x[2]), reverse=True)

    # Greedily match pairs
    matches: list[tuple[dict, dict, T]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()

    for i, j, score in all_pairs:
        if i not in used_a and j not in used_b:
            matches.append((trades_a[i], trades_b[j], score))
            used_a.add(i)
            used_b.add(j)

    return matches


def match_trades_by_group(
    trades_a: list[dict],
    trades_b: list[dict],
    similarity_fn: Callable[[dict, dict], T],
) -> list[tuple[dict, dict, T]]:
    """
    Match trades across all primary key groups.

    Groups trades by primary key, then finds best matches within each group.

    Args:
        trades_a: All trades from annotator A
        trades_b: All trades from annotator B
        similarity_fn: Function to calculate similarity between two trades

    Returns:
        List of (trade_a, trade_b, similarity_score) tuples
    """
    grouped_a = group_trades_by_key(trades_a)
    grouped_b = group_trades_by_key(trades_b)

    all_keys = set(grouped_a.keys()) | set(grouped_b.keys())
    all_matches: list[tuple[dict, dict, T]] = []

    for key in all_keys:
        key_trades_a = grouped_a.get(key, [])
        key_trades_b = grouped_b.get(key, [])

        matches = find_best_matches(key_trades_a, key_trades_b, similarity_fn)
        all_matches.extend(matches)

    return all_matches


def _extract_score(similarity: Any) -> float:
    """Extract numeric score from similarity result for sorting.

    Handles:
    - float: returns as-is
    - tuple: extracts last numeric element (per_field or per_label format)
    - UnifiedSimilarity dataclass: extracts overall_score attribute
    """
    if isinstance(similarity, (int, float)):
        return float(similarity)
    if isinstance(similarity, tuple):
        # For per_label: (agreement_dict, count_dict, score)
        # For per_field: (field_scores_dict, score)
        if len(similarity) == 3:
            return similarity[2]
        elif len(similarity) == 2:
            return similarity[1]
    # Handle dataclass with overall_score attribute (UnifiedSimilarity)
    if hasattr(similarity, "overall_score"):
        return similarity.overall_score
    return 0.0
