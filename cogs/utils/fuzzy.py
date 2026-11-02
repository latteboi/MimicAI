"""Fuzzy name matching for user-supplied profile names.

Discord's autocomplete never constrains what a user can submit, so every command that
takes a profile name has to cope with free text. The matcher here backs both halves of
that: it ranks the autocomplete choices (so a typo is caught before submission) and it
feeds the "Did you mean?" recovery view (for what still slips through).

Deliberately dependency-free and I/O-free. `rank` is called from the autocomplete hot
path, so it only ever touches strings the caller already has in memory -- resolving
display names, PIDs, or appearance data for a candidate is the caller's job, and should
be done *after* ranking so it only happens for the survivors.

Cost: the cheap tiers are a casefold plus a substring test. Edit distance is paid only
for candidates that no cheaper tier matched, and only over the diagonal band its budget
allows, so it costs a few cells per query character whatever the name's length.
"""

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
# A typo: within the edit budget of the whole name, or else of its start or one of its
# words. Each further edit costs a step, and the weakest still clears DEFAULT_CUTOFF.
_TYPO_WHOLE = 0.54
_TYPO_PARTIAL = 0.47
_TYPO_STEP = 0.02

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


def _typo_budget(query_len: int) -> int:
    """How many edits a query of this length may carry and still mean a name.

    Counted, not a ratio: one slip in "mrf" is a third of it, and a ratio cut-off long
    enough to be safe on "sherlok" refused every three-letter name a single typo.
    """
    return 0 if query_len < 3 else 1 if query_len < 6 else 2 if query_len < 12 else 3


def _typo_distance(q: str, c: str, k: int) -> Tuple[int, int]:
    """Edits from `q` to `c`, and to the closest start of `c` -- each capped at k + 1.

    Optimal string alignment: a swap of neighbours ("aliec") is one edit. Only cells
    within k of the diagonal are filled, as every cell outside it is over the cap already.
    The start counts because autocomplete is typed left to right: "mrf" is one slip from
    "mr. grizz" begun, though nowhere near all of it.
    """
    n, m = len(q), len(c)
    over = k + 1
    if n - m > k:
        return over, over
    prev2: List[int] = []
    prev = [j if j <= k else over for j in range(m + 1)]
    for i in range(1, n + 1):
        row = [over] * (m + 1)
        if i <= k:
            row[0] = i
        for j in range(max(1, i - k), min(m, i + k) + 1):
            d = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (q[i - 1] != c[j - 1]))
            if i > 1 and j > 1 and q[i - 1] == c[j - 2] and q[i - 2] == c[j - 1]:
                d = min(d, prev2[j - 2] + 1)
            row[j] = min(d, over)
        prev2, prev = prev, row
    return prev[m], min(prev[max(0, n - k):m + 1])


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

    # Only now, for candidates nothing cheaper matched, pay for edit distance. This is
    # the tier that catches an actual typo: "Aliec" against "Alice". Against each word as
    # well as the whole name: "thron" is one slip from "Dr Aris Thorne"'s last word, and
    # as a fraction of the whole name it is no match at all. A whole-name slip outranks a
    # partial one, so "mrf" offers "mrc" before "mr. grizz".
    k = _typo_budget(len(q_norm))
    if not k:
        return 0.0
    whole, start = _typo_distance(q_norm, c_norm or c_fold, k)
    if whole <= k:
        return _TYPO_WHOLE - _TYPO_STEP * max(0, whole - 1)
    partial = min([start, *(d for token in (c_tokens if len(c_tokens) > 1 else ())
                            for d in _typo_distance(q_norm, token, k))])
    if partial <= k:
        return _TYPO_PARTIAL - _TYPO_STEP * max(0, partial - 1)
    return 0.0


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
