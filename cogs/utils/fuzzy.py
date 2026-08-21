"""Fuzzy name matching for user-supplied profile names.

Discord's autocomplete never constrains what a user can submit, so every command that
takes a profile name has to cope with free text. The matcher here backs both halves of
that: it ranks the autocomplete choices (so a typo is caught before submission) and it
feeds the "Did you mean?" recovery view (for what still slips through).

Deliberately dependency-free and I/O-free. `rank` is called from the autocomplete hot
path, so it only ever touches strings the caller already has in memory -- resolving
display names, PIDs, or appearance data for a candidate is the caller's job, and should
be done *after* ranking so it only happens for the survivors.

Cost: the cheap tiers are a casefold plus a substring test. `difflib` is consulted only
for candidates that no cheaper tier matched, which is the case that was previously
returning nothing at all.
"""

from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Scores are banded rather than continuous so that a weaker tier can never outrank a
# stronger one no matter how flattering its internal ratio is. A substring hit on a long
# name must still lose to a prefix hit on a short one.
_EXACT = 1.0
_CASEFOLD = 0.98
_NORMALISED = 0.95
_PREFIX_BASE = 0.85
_SUBSTRING_BASE = 0.70
_TOKEN_BASE = 0.55
_DIFFLIB_CEILING = 0.54

#: Below this, a suggestion is more confusing than helpful and is dropped.
DEFAULT_CUTOFF = 0.40

#: Discord's hard cap on choices in one autocomplete response.
MAX_CHOICES = 25


def _normalise(text: str) -> str:
    """Casefold and strip everything that isn't alphanumeric.

    Collapses the punctuation and spacing differences that users reliably get wrong --
    "Dr. Aris Thorne", "dr aris thorne" and "DrArisThorne" all normalise together.
    """
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _tokens(text: str) -> List[str]:
    """Split on anything non-alphanumeric, dropping empties."""
    out: List[str] = []
    current: List[str] = []
    for ch in text.casefold():
        if ch.isalnum():
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def _length_bonus(query_len: int, candidate_len: int) -> float:
    """A small tie-breaker favouring candidates the query covers more of.

    Kept under the width of a single band (0.1) so it can only reorder within a tier,
    never promote a candidate past one that matched more strongly.
    """
    if candidate_len <= 0:
        return 0.0
    return 0.09 * min(1.0, query_len / candidate_len)


def score(query: str, candidate: str) -> float:
    """Score `candidate` against `query`, in [0.0, 1.0]. Higher is a better match.

    An empty query scores every candidate equally at the substring band, which is what
    makes "user opened the picker and hasn't typed yet" fall out of the same code path
    as a real search.
    """
    if not query:
        return _SUBSTRING_BASE

    if query == candidate:
        return _EXACT

    q_fold = query.casefold()
    c_fold = candidate.casefold()
    if q_fold == c_fold:
        return _CASEFOLD

    q_norm = _normalise(query)
    c_norm = _normalise(candidate)
    if q_norm and q_norm == c_norm:
        return _NORMALISED

    if c_fold.startswith(q_fold):
        return _PREFIX_BASE + _length_bonus(len(q_fold), len(c_fold))

    # Normalised prefix catches "dr.aris" against "Dr Aris Thorne", where the raw
    # casefold prefix test fails on the punctuation alone.
    if q_norm and c_norm.startswith(q_norm):
        return _PREFIX_BASE + _length_bonus(len(q_norm), len(c_norm))

    if q_fold in c_fold:
        return _SUBSTRING_BASE + _length_bonus(len(q_fold), len(c_fold))

    if q_norm and q_norm in c_norm:
        return _SUBSTRING_BASE + _length_bonus(len(q_norm), len(c_norm))

    # Token overlap handles word-order mistakes and partial recall -- "thorne aris"
    # or just "thorne" against "Dr Aris Thorne".
    q_tokens = _tokens(query)
    c_tokens = _tokens(candidate)
    if q_tokens and c_tokens:
        c_set = set(c_tokens)
        matched = 0
        for token in q_tokens:
            if token in c_set:
                matched += 1
            elif any(ct.startswith(token) for ct in c_tokens):
                # A partial token still counts, at half weight -- "thor" for "thorne".
                matched += 0.5
        if matched:
            coverage = matched / len(q_tokens)
            # Spans 0.55 to 0.69 -- above the difflib ceiling, below the substring band.
            return _TOKEN_BASE + 0.14 * coverage

    # Only now, for candidates nothing cheaper matched, pay for edit-distance. This is
    # the tier that catches an actual typo: "Aliec" against "Alice".
    ratio = SequenceMatcher(None, q_norm or q_fold, c_norm or c_fold).ratio()
    return min(_DIFFLIB_CEILING, ratio * _DIFFLIB_CEILING)


def rank(
    query: str,
    candidates: Iterable[str],
    *,
    limit: int = MAX_CHOICES,
    cutoff: float = DEFAULT_CUTOFF,
) -> List[Tuple[str, float]]:
    """Rank `candidates` against `query`, best first.

    Returns (candidate, score) pairs above `cutoff`, truncated to `limit`. Ties break
    on the candidate's own sort order so repeated calls are stable.
    """
    scored: List[Tuple[str, float]] = []
    for candidate in candidates:
        s = score(query, candidate)
        if s >= cutoff:
            scored.append((candidate, s))

    scored.sort(key=lambda pair: (-pair[1], pair[0].casefold()))
    return scored[:limit]


def rank_keyed(
    query: str,
    candidates: Sequence[Tuple[str, str]],
    *,
    limit: int = MAX_CHOICES,
    cutoff: float = DEFAULT_CUTOFF,
) -> List[Tuple[str, float]]:
    """`rank`, for candidates carrying an opaque key alongside their matchable text.

    Takes (key, text) pairs and returns (key, score). Used where the thing being matched
    is not the thing being returned -- a session participant matches on its profile name
    but is identified by an "owner_id:name" value.

    Where the same key appears with several matchable texts (a profile has both an
    internal name and a display name), the best-scoring one wins and the key appears
    once.
    """
    best: Dict[str, float] = {}
    for key, text in candidates:
        s = score(query, text)
        if s >= cutoff and s > best.get(key, -1.0):
            best[key] = s

    scored = sorted(best.items(), key=lambda pair: (-pair[1], pair[0].casefold()))
    return scored[:limit]


def best_match(
    query: str,
    candidates: Iterable[str],
    *,
    threshold: float = _NORMALISED,
) -> Optional[str]:
    """The single unambiguously-correct candidate, or None.

    `threshold` defaults to the normalised-exact band, so this only ever auto-resolves
    a difference in case, spacing, or punctuation -- never a genuine typo, which stays
    the user's call to confirm.
    """
    ranked = rank(query, candidates, limit=2, cutoff=threshold)
    if not ranked:
        return None
    # Two candidates can normalise identically ("Alice" / "alice"), and picking one for
    # the user would be a coin flip. Only a tie *at the same score* is ambiguous though:
    # querying "Alice" against both scores the exact match above the casefold one, and
    # that is a clear winner rather than a toss-up.
    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
        return None
    return ranked[0][0]
