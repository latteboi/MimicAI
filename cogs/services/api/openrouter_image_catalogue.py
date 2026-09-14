"""Every OpenRouter image model this bot can use, held once in memory for the image pickers.

The image counterpart of `openrouter_catalogue`, built the same way, from OpenRouter's
documented API only:

- `/api/v1/images/models` -- what each model accepts (`supported_parameters`);
- `/api/v1/models?output_modalities=image&sort=most-popular` -- the ranking and the names;
- `/api/v1/images/models/{id}/endpoints`, once per model -- who hosts it, and the price;
- `/api/v1/models/user?output_modalities=image`, read with the bot owner's key.

Two rules decide what is listed at all, and both answer something the text catalogue never
has to.

**One host company, or not listed.** A text request carries `provider.data_collection:
"deny"`, which keeps it off a host that trains whichever host would otherwise serve it. The
Image API takes no such field, so an image request routes by the privacy settings of the
account whose key pays -- a server's, not the bot owner's. What the bot owner's listing says
about a model with one host holds for that host under any key; about a model with two, it
only says that one of them avoids training. That leaves out the Gemini image models, which
the Google tab runs natively anyway.

**Discord has to be able to show the result.** A model that answers only in SVG is left out,
and so is one that refuses a request without a reference image.

Whether a listed model avoids training is read off the account listing, but only in a sync
whose *text* listing proved the account filters training hosts at all. Every image host may
well be allowed, and a listing that holds every image model cannot tell that apart from
privacy settings that filter nothing.
"""
import datetime
import os
import time
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import orjson

from ...managers.storage_manager import IOManager
from ...utils.constants import IMAGE_QUALITY_LEVELS, IMAGE_RASTER_FORMATS, IMAGE_SIZES_ALL
from ...utils.helpers import install_openrouter_image_caps
from .openrouter_catalogue import (
    AUTHOR_PREFIX, AuthorInfo, _author_display, format_price,
)

#: Bumped when ImageModelInfo's fields change, for the reason CATALOGUE_FORMAT is.
IMAGE_CATALOGUE_FORMAT = 1


class ImageModelInfo(NamedTuple):
    id: str
    name: str
    author: str
    #: What it accepts, as `image_model_caps` hands it to the pickers and the request path.
    #: Sizes stop at IMAGE_SIZES_ALL's 2K. "auto" is kept out of ratios and qualities: it is
    #: the absence of a choice, which a blank setting already sends.
    ratios: Tuple[str, ...]
    sizes: Tuple[str, ...]
    qualities: Tuple[str, ...]
    formats: Tuple[str, ...]
    max_refs: int
    #: One generated image's price in OpenRouter's own unit -- "image", "megapixel" or
    #: "token" -- as the cheapest and dearest rate its host lists; None when it lists none.
    price_unit: Optional[str]
    price_min: Optional[float]
    price_max: Optional[float]
    #: As OpenRouterCatalogue's: False when a filtering account listing kept it, True when
    #: one left it out, None when none ever covered it.
    trains: Optional[bool]


def _enum(params: dict, key: str) -> Tuple[str, ...]:
    spec = params.get(key)
    if isinstance(spec, dict) and spec.get("type") == "enum":
        return tuple(str(v) for v in spec.get("values") or [])
    return ()


def _range_bound(params: dict, key: str, bound: str) -> int:
    spec = params.get(key)
    if not isinstance(spec, dict):
        return 0
    try:
        return int(spec.get(bound) or 0)
    except (TypeError, ValueError):
        return 0


def _hosts_and_price(body: Optional[bytes]):
    """(host companies, price unit, cheapest, dearest) from one endpoints body; None if unreadable.

    Companies rather than endpoints: AI Studio's standard, flex and priority tiers are one
    host under one policy, and routing between them is not what the one-host rule is about.
    """
    if not body:
        return None
    try:
        parsed = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    record = parsed["data"] if isinstance(parsed.get("data"), dict) else parsed
    endpoints = record.get("endpoints")
    if not isinstance(endpoints, list):
        return None
    hosts: Set[str] = set()
    rates: Dict[str, List[float]] = {}
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        slug = str(endpoint.get("provider_slug") or endpoint.get("provider_name") or "")
        if slug:
            hosts.add(slug.split("/", 1)[0])
        for price in endpoint.get("pricing") or []:
            if not (isinstance(price, dict) and price.get("billable") == "output_image"):
                continue
            try:
                rates.setdefault(str(price.get("unit")), []).append(float(price.get("cost_usd")))
            except (TypeError, ValueError):
                continue
    if not rates:
        return hosts, None, None, None
    # One unit per model in practice. Should a host ever mix them, the first listed wins
    # rather than a per-image rate being compared with a per-megapixel one.
    unit, values = next(iter(rates.items()))
    return hosts, unit, min(values), max(values)


def _whole_listing_ids(account_body: Optional[bytes]) -> Optional[Set[str]]:
    """The ids an account listing returned, or None when it is missing, unreadable or cut short.

    Unlike the text catalogue's test, a listing holding every model is an answer here: the
    caller only asks when the text listing already proved the settings filter.
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
    return {m["id"] for m in data if isinstance(m, dict) and m.get("id")}


def _ranking(ranking_body: Optional[bytes]) -> Tuple[List[str], Dict[str, str]]:
    if not ranking_body:
        return [], {}
    try:
        data = orjson.loads(ranking_body).get("data") or []
    except (orjson.JSONDecodeError, AttributeError):
        return [], {}
    ids, names = [], {}
    for raw in data:
        if isinstance(raw, dict) and raw.get("id"):
            ids.append(raw["id"])
            if raw.get("name"):
                names[raw["id"]] = raw["name"]
    return ids, names


def _open_to_all(info: ImageModelInfo) -> bool:
    """Whether its one host is known not to train on prompts. Unknown counts as no."""
    return info.trains is False


def _caps(info: ImageModelInfo) -> dict:
    """The `image_model_caps` shape. Nothing on this API takes thinking, a search tool or sampling."""
    return {'sizes': info.sizes, 'ratios': info.ratios, 'thinking': False, 'modalities': (),
            'grounding': False, 'image_search': False, 'quality': info.qualities,
            'formats': info.formats, 'max_refs': info.max_refs, 'sampling': False}


def _from_record(record) -> ImageModelInfo:
    info = ImageModelInfo(*record)
    # JSON has no tuples; the caps registry and the pickers compare against these.
    return info._replace(ratios=tuple(info.ratios), sizes=tuple(info.sizes),
                         qualities=tuple(info.qualities), formats=tuple(info.formats))


_UNIT_SUFFIXES = {"image": "/image", "megapixel": "/MP", "token": "/1M tokens"}


def image_price_text(info: ImageModelInfo) -> str:
    if info.price_unit is None or info.price_min is None:
        return "Price not listed"
    scale = 1_000_000 if info.price_unit == "token" else 1
    low, high = info.price_min * scale, (info.price_max or info.price_min) * scale
    suffix = _UNIT_SUFFIXES.get(info.price_unit, f"/{info.price_unit}")
    if high > low:
        return f"{format_price(low)}–{format_price(high).lstrip('$')}{suffix}"
    return f"{format_price(low)}{suffix}"


def _sizes_text(info: ImageModelInfo) -> str:
    if not info.sizes:
        return "one fixed size"
    return info.sizes[0] if len(info.sizes) == 1 else f"{info.sizes[0]}–{info.sizes[-1]}"


class OpenRouterImageCatalogue:
    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "openrouter_image_catalogue.json")
        self.models: Dict[str, ImageModelInfo] = {}
        self.updated_at: Optional[str] = None
        self._popular: List[str] = []
        self._by_author: Dict[str, List[str]] = {}
        self._authors: List[AuthorInfo] = []
        self._open: frozenset = frozenset()
        self._open_authors: List[AuthorInfo] = []
        #: As OpenRouterCatalogue's: when an account listing last said which models train,
        #: and whether the latest sync's did.
        self.training_checked_at: Optional[int] = None
        self.training_current = False

    # --- Loading and syncing (run these in a thread) ------------------------------

    def load(self) -> None:
        data = IOManager.read_json(self.path) or {}
        if data.get("format") != IMAGE_CATALOGUE_FORMAT:
            return
        models = {}
        for record in data.get("models") or []:
            try:
                info = _from_record(record)
            except (TypeError, ValueError):
                continue
            models[info.id] = info
        self.training_checked_at = data.get("training_checked_at")
        if models:
            self._install(models, data.get("popular") or [], data.get("updated_at"))

    def apply_listing(self, models_body: bytes, ranking_body: Optional[bytes],
                      endpoint_bodies: Dict[str, Optional[bytes]], account_body: Optional[bytes],
                      account_filters_training: bool) -> None:
        """Parses a fresh image listing, saves it, and installs it.

        `account_filters_training` is the same sync's `OpenRouterCatalogue.training_current`:
        whether the bot owner's text listing showed their privacy settings excluding hosts
        that may train. Only then does their image listing say anything -- see the module
        docstring. A model whose endpoints could not be read this time keeps its last
        record rather than vanishing from every picker until tomorrow's sync.
        """
        listing = orjson.loads(models_body).get("data") or []
        ranking, names = _ranking(ranking_body)
        kept = _whole_listing_ids(account_body) if account_filters_training else None
        self.training_current = kept is not None
        previous = self.models

        models: Dict[str, ImageModelInfo] = {}
        for raw in listing:
            model_id = raw.get("id") if isinstance(raw, dict) else None
            if not model_id:
                continue
            params = raw.get("supported_parameters") or {}
            formats = _enum(params, "output_format")
            if formats and not set(formats) & set(IMAGE_RASTER_FORMATS):
                continue
            if _range_bound(params, "input_references", "min") > 0:
                continue

            hosted = _hosts_and_price(endpoint_bodies.get(model_id))
            if hosted is None:
                if model_id not in previous:
                    continue
                last = previous[model_id]
                unit, low, high = last.price_unit, last.price_min, last.price_max
            else:
                hosts, unit, low, high = hosted
                if len(hosts) != 1:
                    continue

            if kept is not None:
                trains = model_id not in kept
            else:
                trains = previous[model_id].trains if model_id in previous else None

            resolutions, qualities = _enum(params, "resolution"), _enum(params, "quality")
            models[model_id] = ImageModelInfo(
                id=model_id,
                name=names.get(model_id) or raw.get("name") or model_id,
                author=model_id.split("/", 1)[0],
                ratios=tuple(r for r in _enum(params, "aspect_ratio") if r != "auto"),
                sizes=tuple(s for s in IMAGE_SIZES_ALL if s in resolutions),
                qualities=tuple(q for q in IMAGE_QUALITY_LEVELS if q in qualities),
                formats=tuple(f for f in formats if f in IMAGE_RASTER_FORMATS),
                max_refs=max(0, _range_bound(params, "input_references", "max")),
                price_unit=unit, price_min=low, price_max=high,
                trains=trains,
            )

        if not models:
            return
        if kept is not None:
            self.training_checked_at = int(time.time())

        popular = [m for m in ranking if m in models]
        seen = set(popular)
        popular += [m for m in models if m not in seen]
        updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        IOManager.write_json({"format": IMAGE_CATALOGUE_FORMAT, "updated_at": updated_at,
                              "popular": popular, "training_checked_at": self.training_checked_at,
                              "models": [list(info) for info in models.values()]}, self.path)
        self._install(models, popular, updated_at)

    def _install(self, models: Dict[str, ImageModelInfo], popular: List[str],
                 updated_at: Optional[str]) -> None:
        """Rebuilds the orderings and the caps registry, then swaps them in together."""
        listed = [m for m in popular if m in models]
        seen = set(listed)
        listed += [m for m in models if m not in seen]

        by_author: Dict[str, List[str]] = {}
        for model_id in listed:
            by_author.setdefault(models[model_id].author, []).append(model_id)
        open_ids = frozenset(m for m in listed if _open_to_all(models[m]))

        self.models = models
        self._popular = listed
        self._by_author = by_author
        self._authors = _author_rows(models, listed)
        self._open = open_ids
        self._open_authors = _author_rows(models, [m for m in listed if m in open_ids])
        self.updated_at = updated_at
        # Every listed model, the bot owner's-only ones included: caps say what a model
        # accepts, not who may pick it.
        install_openrouter_image_caps({m: _caps(info) for m, info in models.items()})

    # --- Browsing -----------------------------------------------------------------
    #
    # The surface ModelPickerMixin already reads off OpenRouterCatalogue, so the image
    # category browses through the same code. `show_training` is required for the same
    # reason it is there.

    def authors(self, *, show_training: bool) -> List[AuthorInfo]:
        return self._authors if show_training else self._open_authors

    def browse(self, key: str, *, show_training: bool) -> Tuple[List[str], Optional[str]]:
        """Image models for a browse key, and a note to show above them (or None).

        Most Popular is OpenRouter's own ranking: image calls are not counted on this bot.
        A text browse key carried over from another category reads as Most Popular.
        """
        def offered(ids: Iterable[str]) -> List[str]:
            return list(ids) if show_training else [m for m in ids if m in self._open]

        if not self.models:
            return [], "The OpenRouter image model list has not loaded yet. Try again shortly."
        if key.startswith(AUTHOR_PREFIX):
            return offered(self._by_author.get(key[len(AUTHOR_PREFIX):], [])), None
        return offered(self._popular), None

    def open_to_all(self, model_id: str) -> bool:
        """For a typed id in a picker: an id the listing lacks is not judged here."""
        info = self.models.get(model_id)
        return info is None or _open_to_all(info)

    def is_open(self, model_id: str) -> bool:
        """For the model factory: listed, and its one host known not to train.

        Stricter than `open_to_all`, because an image request has no `data_collection` to
        fall back on when the listing does not know a model.
        """
        info = self.models.get(model_id)
        return info is not None and _open_to_all(info)

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
        """The price first, then what the model draws and takes, in 100 characters."""
        info = self.models.get(model_id)
        if not info:
            return None
        parts = [image_price_text(info), _sizes_text(info)]
        if info.qualities:
            parts.append(info.qualities[0] if len(info.qualities) == 1
                         else f"{info.qualities[0]}–{info.qualities[-1]} quality")
        parts.append(f"{info.max_refs} ref{'s' if info.max_refs != 1 else ''}")
        if not _open_to_all(info):
            parts.append("⚠ may train")
        return " · ".join(parts)[:100]

    def describe_author(self, author: AuthorInfo) -> str:
        return f"{author.models} image model{'s' if author.models != 1 else ''}"[:100]

    def detail_lines(self, model_id: str) -> Optional[str]:
        """Everything the picker's embed says about a chosen image model."""
        info = self.models.get(model_id)
        if not info:
            return None
        lines = [f"{image_price_text(info)} · {_sizes_text(info)}"]
        facts = [f"Takes up to {info.max_refs} reference image{'s' if info.max_refs != 1 else ''}"]
        if info.qualities:
            facts.append("Quality: " + ", ".join(info.qualities))
        lines.append(" · ".join(facts))
        if info.ratios:
            lines.append("Ratios: " + ", ".join(info.ratios))
        if not _open_to_all(info):
            lines.append(("Its host may train on prompts" if info.trains
                          else "Not yet checked whether its host trains on prompts")
                         + ", so it is offered to the bot owner alone")
        return "\n".join(lines)


def _author_rows(models: Dict[str, ImageModelInfo], ids: Iterable[str]) -> List[AuthorInfo]:
    """One row per author of `ids`, A-Z, counting only the models in `ids`."""
    by_author: Dict[str, List[str]] = {}
    for model_id in ids:
        by_author.setdefault(models[model_id].author, []).append(model_id)
    rows = [AuthorInfo(slug=slug, name=_author_display(slug, [models[m].name for m in author_ids]),
                       models=len(author_ids), min_price=None, max_price=None, zdr_models=0)
            for slug, author_ids in by_author.items()]
    rows.sort(key=lambda a: a.name.casefold())
    return rows
