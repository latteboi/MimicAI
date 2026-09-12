"""Every text-output model OpenRouter lists, held once in memory for the model pickers.

Built from OpenRouter's documented API only: `/api/v1/models` (text output is its default
filter), `/api/v1/endpoints/zdr`, and `/api/v1/models/user` read with the bot owner's key.
OpenRouter's website has richer internal endpoints -- per-host training and retention
policy among them -- but its Terms (§7) prohibit software that scrapes the Site, so
nothing here reads them.

So whether some host serves a model without training on prompts is inferred rather than
read. A zero-retention host keeps nothing to train on, and `/models/user` leaves out
whatever the account's privacy settings exclude -- which, with providers that may train
turned off, is every model only such providers serve. A model known either way is offered
to everyone; the rest only to the bot owner, who alone can open a server to them
(`may_pick_training_models`).

Four ways to browse, plus one per model author:

- **Most Popular** is this bot's own usage count. Until anything has been used it shows
  OpenRouter's global ranking instead, and says so.
- **Trending** is the biggest climb in OpenRouter's popularity rank against the oldest
  daily snapshot inside a week. It needs two snapshots to say anything.
- **Cheapest** is input plus output price per million tokens, free first. Router models
  that price per request (-1) go last.
- **By author** is each author's models in popularity order.

The sync runs in a thread: the full listing is ~700 KB and parses to several MB of dicts,
of which only a slim record per model is kept.
"""
import datetime
import os
import time
from collections import Counter
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import orjson

from ...managers.storage_manager import IOManager

#: Snapshots older than this are dropped; Trending compares against the oldest one kept.
TRENDING_WINDOW_DAYS = 7
#: Only models currently this popular or better can trend. A model absent from the old
#: snapshot starts at the bottom, so without a floor every new listing would "climb".
TRENDING_MAX_RANK = 150
#: Unsaved usage is written at most this often. The count only orders a list, so losing a
#: few minutes of it to a crash costs nothing; rewriting it on every turn cost a disk
#: write per generation.
USAGE_FLUSH_INTERVAL_SECONDS = 300.0

#: Bumped when ModelInfo's fields change; a saved catalogue in another format is ignored
#: until the next sync rewrites it, rather than misread positionally.
CATALOGUE_FORMAT = 2

BROWSE_POPULAR = "popular"
BROWSE_TRENDING = "trending"
BROWSE_CHEAPEST = "cheapest"
AUTHOR_PREFIX = "author:"


class ModelInfo(NamedTuple):
    id: str
    name: str
    author: str
    context: int
    image_input: bool
    prompt_1m: float
    completion_1m: float
    moderated: bool
    reasoning_mandatory: bool
    reasoning_default: Optional[str]
    knowledge_cutoff: Optional[str]
    expires: Optional[str]
    zdr_hosts: int
    alias_of: Optional[str]
    open_weights: bool
    #: False when the bot owner's `/models/user` listed it, so some host serves it without
    #: training; True when a listing that did filter left it out; None when no usable
    #: listing has ever covered it.
    trains: Optional[bool]


class AuthorInfo(NamedTuple):
    slug: str
    name: str
    models: int
    min_price: Optional[float]
    max_price: Optional[float]
    zdr_models: int


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _author_display(slug: str, names: List[str]) -> str:
    """The author as their model names spell it ("Anthropic: Claude ..."), else the slug."""
    prefixes = Counter(n.split(": ", 1)[0] for n in names if ": " in n)
    if prefixes:
        return prefixes.most_common(1)[0][0]
    return slug.replace("-", " ").title()


def format_price(per_million: float) -> str:
    if per_million >= 10:
        return f"${per_million:.0f}"
    if per_million >= 1:
        return f"${per_million:.2f}"
    if per_million == 0:
        return "$0"
    text = f"{per_million:.4f}".rstrip("0").rstrip(".")
    # Rounding a real price down to "$0" would read as free.
    return f"${text}" if text != "0" else "<$0.0001"


def format_context(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:g}M ctx"
    if tokens >= 1000:
        return f"{tokens // 1000}K ctx"
    return f"{tokens} ctx" if tokens else ""


def price_text(info: ModelInfo) -> str:
    if info.prompt_1m < 0 or info.completion_1m < 0:
        return "Variable price"
    if info.prompt_1m == 0 and info.completion_1m == 0:
        return "Free"
    return f"{format_price(info.prompt_1m)} in · {format_price(info.completion_1m)} out /1M"


def _blended(info: ModelInfo) -> float:
    if info.prompt_1m < 0 or info.completion_1m < 0:
        return float("inf")
    return info.prompt_1m + info.completion_1m


def _open_to_all(info: ModelInfo) -> bool:
    """Whether some host is known to serve this model without training on prompts.

    Unknown counts as no: a model nothing has vouched for is offered to the bot owner alone.
    """
    return bool(info.zdr_hosts) or info.trains is False


def _author_rows(models: Dict[str, ModelInfo], ids: Iterable[str]) -> List[AuthorInfo]:
    """One row per author of `ids`, A-Z, each counting only the models in `ids`."""
    by_author: Dict[str, List[str]] = {}
    for model_id in ids:
        by_author.setdefault(models[model_id].author, []).append(model_id)
    authors = []
    for slug, author_ids in by_author.items():
        priced = [_blended(models[m]) for m in author_ids if _blended(models[m]) != float("inf")]
        authors.append(AuthorInfo(
            slug=slug, name=_author_display(slug, [models[m].name for m in author_ids]),
            models=len(author_ids), min_price=min(priced) if priced else None,
            max_price=max(priced) if priced else None,
            zdr_models=sum(1 for m in author_ids if models[m].zdr_hosts)))
    authors.sort(key=lambda a: a.name.casefold())
    return authors


def _models_without_training(account_body: Optional[bytes], listed: Set[str]) -> Optional[Set[str]]:
    """The ids `/models/user` returned, when that listing can say which models avoid training.

    OpenRouter filters it by the account's privacy settings. It answers the question only
    when those settings exclude providers that may train -- a listing holding every model
    says they do not -- and only when it is whole: one cut short by paging cannot say which
    of the rest are missing. Anything else is None, never a guess.
    """
    if not account_body:
        return None
    try:
        body = orjson.loads(account_body)
    except orjson.JSONDecodeError:
        return None
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return None
    total = body.get("total_count")
    if isinstance(total, int) and total > len(data):
        return None
    ids = {m["id"] for m in data if isinstance(m, dict) and m.get("id")}
    if not ids or listed <= ids:
        return None
    return ids


class OpenRouterCatalogue:
    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "openrouter_catalogue.json")
        self.history_path = os.path.join(data_dir, "openrouter_rank_history.json")
        #: The usage counter keeps the file name it has always had.
        self.usage_path = os.path.join(data_dir, "openrouter_models.json")

        self.models: Dict[str, ModelInfo] = {}
        self.updated_at: Optional[str] = None
        self._popular: List[str] = []
        self._cheapest: List[str] = []
        self._trending: List[str] = []
        self.trending_days = 0
        self._authors: List[AuthorInfo] = []
        self._by_author: Dict[str, List[str]] = {}
        #: The models offered to everyone, and their authors' rows counted over only those.
        self._open: frozenset = frozenset()
        self._open_authors: List[AuthorInfo] = []
        #: Unix time of the last sync whose account listing could tell which models avoid
        #: training, and whether the latest sync was one.
        self.training_checked_at: Optional[int] = None
        self.training_current = False

        self.usage: Counter = Counter()
        self._usage_loaded = False
        self._usage_dirty = False
        self._usage_flushed_at = 0.0

    # --- Loading and syncing (run these in a thread) ------------------------------

    def load(self) -> None:
        """Reads the saved catalogue, rank history and usage count."""
        self._load_usage()
        data = IOManager.read_json(self.path) or {}
        if data.get("format") != CATALOGUE_FORMAT:
            return
        records = data.get("models") or []
        models = {}
        for record in records:
            try:
                info = ModelInfo(*record)
            except TypeError:
                continue
            models[info.id] = info
        self.training_checked_at = data.get("training_checked_at")
        if models:
            self._install(models, data.get("popular") or [], data.get("updated_at"),
                          IOManager.read_json(self.history_path) or {})

    def apply_listing(self, models_body: bytes, zdr_body: Optional[bytes],
                      today: Optional[datetime.date] = None,
                      account_body: Optional[bytes] = None) -> Dict[str, Dict[str, float]]:
        """Parses a fresh listing, saves it, and returns OpenRouter prices for the pricing cache.

        `models_body` must come from `/api/v1/models?sort=most-popular`: its order is the
        popularity ranking, and nothing else in the response carries one. `account_body` is
        `/api/v1/models/user` as the bot owner's account sees it; when it cannot say which
        models avoid training, each model keeps what the last one that could said.
        """
        listing = orjson.loads(models_body).get("data") or []
        zdr_counts: Counter = Counter()
        if zdr_body:
            try:
                for endpoint in orjson.loads(zdr_body).get("data") or []:
                    if endpoint.get("model_id"):
                        zdr_counts[endpoint["model_id"]] += 1
            except orjson.JSONDecodeError:
                pass

        # Aliases are left out of the comparison: whether the account listing repeats them
        # says nothing about its privacy settings.
        without_training = _models_without_training(
            account_body, {raw.get("id") for raw in listing if raw.get("id") and not raw.get("alias_target")})
        self.training_current = without_training is not None
        previous = self.models

        models: Dict[str, ModelInfo] = {}
        popular: List[str] = []
        rates: Dict[str, Dict[str, float]] = {}
        for raw in listing:
            model_id = raw.get("id")
            if not model_id:
                continue
            pricing = raw.get("pricing") or {}
            reasoning = raw.get("reasoning") or {}
            if without_training is not None:
                trains = model_id not in without_training
            else:
                trains = previous[model_id].trains if model_id in previous else None
            info = ModelInfo(
                id=model_id,
                name=raw.get("name") or model_id,
                author=model_id.split("/", 1)[0],
                context=int(raw.get("context_length") or 0),
                image_input="image" in ((raw.get("architecture") or {}).get("input_modalities") or []),
                prompt_1m=_float(pricing.get("prompt")) * 1_000_000,
                completion_1m=_float(pricing.get("completion")) * 1_000_000,
                moderated=bool((raw.get("top_provider") or {}).get("is_moderated")),
                reasoning_mandatory=bool(reasoning.get("mandatory")),
                reasoning_default=reasoning.get("default_effort"),
                knowledge_cutoff=raw.get("knowledge_cutoff"),
                expires=raw.get("expiration_date"),
                zdr_hosts=zdr_counts.get(model_id, 0),
                alias_of=raw.get("alias_target"),
                open_weights=bool(raw.get("hugging_face_id")),
                trains=trains,
            )
            models[model_id] = info
            popular.append(model_id)
            if info.prompt_1m >= 0 and info.completion_1m >= 0:
                rates[f"OPENROUTER/{model_id}"] = {"input_1m": info.prompt_1m,
                                                    "output_1m": info.completion_1m}

        if not models:
            return rates
        if without_training is not None:
            self.training_checked_at = int(time.time())

        today = today or datetime.datetime.now(datetime.timezone.utc).date()
        history = IOManager.read_json(self.history_path) or {}
        history[today.isoformat()] = {model_id: rank for rank, model_id in enumerate(popular)}
        cutoff = today - datetime.timedelta(days=TRENDING_WINDOW_DAYS)
        kept_history = {}
        for day, ranks in history.items():
            parsed = _parse_day(day)
            if parsed is not None and parsed >= cutoff:
                kept_history[day] = ranks
        history = kept_history

        updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        IOManager.write_json({"format": CATALOGUE_FORMAT, "updated_at": updated_at, "popular": popular,
                              "training_checked_at": self.training_checked_at,
                              "models": [list(info) for info in models.values()]}, self.path)
        IOManager.write_json(history, self.history_path)
        self._install(models, popular, updated_at, history)
        return rates

    def _install(self, models: Dict[str, ModelInfo], popular: List[str], updated_at: Optional[str],
                 history: Dict[str, Dict[str, int]]) -> None:
        """Rebuilds every derived ordering, then swaps them in together."""
        listed = [model_id for model_id in popular if model_id in models]
        seen = set(listed)
        listed += [model_id for model_id in models if model_id not in seen]
        position = {model_id: i for i, model_id in enumerate(listed)}
        # An alias is another id for a model already listed; showing both duplicates it.
        visible = [model_id for model_id in listed if not models[model_id].alias_of]

        cheapest = sorted(visible, key=lambda m: (_blended(models[m]), position[m]))

        trending, days = _trending(visible, history)

        by_author: Dict[str, List[str]] = {}
        for model_id in visible:
            by_author.setdefault(models[model_id].author, []).append(model_id)
        open_ids = frozenset(m for m in visible if _open_to_all(models[m]))
        authors = _author_rows(models, visible)
        open_authors = _author_rows(models, [m for m in visible if m in open_ids])

        self.models = models
        self._popular = visible
        self._cheapest = cheapest
        self._trending = trending
        self.trending_days = days
        self._by_author = by_author
        self._authors = authors
        self._open = open_ids
        self._open_authors = open_authors
        self.updated_at = updated_at

    # --- Browsing ---------------------------------------------------------------
    #
    # Every read that lists models takes `show_training`: `may_pick_training_models` for
    # whoever the picker writes for. Keyword-only and required, so a new picker cannot
    # forget to ask and offer everyone the bot owner's list.

    def authors(self, *, show_training: bool) -> List[AuthorInfo]:
        return self._authors if show_training else self._open_authors

    def browse(self, key: str, *, show_training: bool) -> Tuple[List[str], Optional[str]]:
        """The model ids for a browse key, and a note to show above them (or None)."""
        def offered(ids: Iterable[str]) -> List[str]:
            return list(ids) if show_training else [m for m in ids if m in self._open]

        if key == BROWSE_TRENDING:
            if self.trending_days < 1:
                return [], "Trending needs a second day of rankings. Check back tomorrow."
            note = None if self.trending_days >= TRENDING_WINDOW_DAYS else \
                f"Trending over {self.trending_days} of {TRENDING_WINDOW_DAYS} days so far."
            return offered(self._trending), note
        if key == BROWSE_CHEAPEST:
            return offered(self._cheapest), None
        if key.startswith(AUTHOR_PREFIX):
            return offered(self._by_author.get(key[len(AUTHOR_PREFIX):], [])), None
        used = offered(model_id for model_id, _count in self.usage.most_common()
                       if not self.models or (model_id in self.models and not self.models[model_id].alias_of))
        if used:
            return used, None
        return offered(self._popular), "Nothing listed here has been used on this bot yet, so this is OpenRouter's own ranking."

    def open_to_all(self, model_id: str) -> bool:
        """Whether anyone may be offered this model. An id the listing lacks is not judged."""
        info = self.models.get(model_id)
        return info is None or _open_to_all(info)

    def training_status(self) -> Tuple[int, int, Optional[int]]:
        """(models offered to everyone, models listed, when training was last checked)."""
        return len(self._open), len(self._popular), self.training_checked_at

    def label(self, model_id: str, *, drop_author: bool = False) -> str:
        info = self.models.get(model_id)
        if not info:
            return model_id[:100]
        name = info.name
        if drop_author and ": " in name:
            name = name.split(": ", 1)[1]
        return name[:100]

    def describe(self, model_id: str) -> Optional[str]:
        """The price first, then the few facts worth deciding on, in 100 characters."""
        info = self.models.get(model_id)
        if not info:
            return None
        parts = [price_text(info)]
        if format_context(info.context):
            parts.append(format_context(info.context))
        if info.zdr_hosts:
            parts.append("🔒 ZDR")
        elif not _open_to_all(info):
            parts.append("⚠ may train")
        if info.image_input:
            parts.append("👁 images")
        if info.moderated:
            parts.append("🛡 moderated")
        if info.expires:
            parts.append(f"⏳ ends {info.expires[:10]}")
        return " · ".join(parts)[:100]

    def describe_author(self, author: AuthorInfo) -> str:
        parts = [f"{author.models} model{'s' if author.models != 1 else ''}"]
        if author.min_price is not None and author.max_price is not None:
            low, high = format_price(author.min_price), format_price(author.max_price)
            parts.append(low + " /1M" if low == high else f"{low}–{high} /1M")
        if author.zdr_models:
            parts.append(f"{author.zdr_models} with ZDR")
        return " · ".join(parts)[:100]

    def detail_lines(self, model_id: str) -> Optional[str]:
        """Everything the picker's embed says about a chosen model."""
        info = self.models.get(model_id)
        if not info:
            return None
        lines = [" · ".join(p for p in (price_text(info), format_context(info.context)) if p)]
        facts = []
        if info.image_input:
            facts.append("Sees images")
        facts.append(f"{info.zdr_hosts} zero-retention host{'s' if info.zdr_hosts != 1 else ''}"
                     if info.zdr_hosts else "No zero-retention host")
        if info.moderated:
            facts.append("Moderated")
        if info.open_weights:
            facts.append("Open weights")
        lines.append(" · ".join(facts))
        if not _open_to_all(info):
            lines.append(("No host is known to serve it without training on prompts"
                          if info.trains else "Not yet checked for a host that avoids training on prompts")
                         + ", so it is offered to the bot owner alone")
        extra = []
        if info.reasoning_mandatory:
            extra.append("Reasoning always on")
        elif info.reasoning_default:
            extra.append(f"Reasoning default: {info.reasoning_default}")
        if info.knowledge_cutoff:
            extra.append(f"Knowledge to {info.knowledge_cutoff}")
        if info.expires:
            extra.append(f"Retires {info.expires[:10]}")
        if extra:
            lines.append(" · ".join(extra))
        return "\n".join(lines)

    # --- Usage ------------------------------------------------------------------

    def _load_usage(self) -> None:
        if self._usage_loaded:
            return
        data = IOManager.read_json(self.usage_path) or {}
        self.usage = Counter({k: int(v) for k, v in data.items() if isinstance(v, (int, float))})
        self._usage_loaded = True
        self._usage_flushed_at = time.monotonic()

    def record_use(self, model_id: str) -> bool:
        """Counts one successful call. True when the count is due to be written."""
        if not model_id:
            return False
        self.usage[model_id] += 1
        self._usage_dirty = True
        return time.monotonic() - self._usage_flushed_at >= USAGE_FLUSH_INTERVAL_SECONDS

    def flush_usage(self) -> None:
        if not self._usage_dirty:
            return
        self._usage_dirty = False
        self._usage_flushed_at = time.monotonic()
        IOManager.write_json(dict(self.usage), self.usage_path)


def _parse_day(value: str) -> Optional[datetime.date]:
    try:
        return datetime.date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _trending(visible: List[str], history: Dict[str, Dict[str, int]]) -> Tuple[List[str], int]:
    """Models ranked by how far they climbed since the oldest snapshot in the window."""
    days = sorted((d, ranks) for d, ranks in ((_parse_day(k), v) for k, v in history.items()) if d)
    if len(days) < 2:
        return [], 0
    newest_day, newest = days[-1]
    oldest_day, oldest = days[0]
    bottom = len(oldest)
    climbs = []
    for model_id in visible:
        rank_now = newest.get(model_id)
        if rank_now is None or rank_now >= TRENDING_MAX_RANK:
            continue
        climb = oldest.get(model_id, bottom) - rank_now
        if climb > 0:
            climbs.append((-climb, rank_now, model_id))
    climbs.sort()
    return [model_id for _c, _r, model_id in climbs], (newest_day - oldest_day).days
