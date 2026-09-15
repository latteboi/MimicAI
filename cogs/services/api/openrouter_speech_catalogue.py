"""Every OpenRouter speech model this bot can use, held once in memory for the speech pickers.

The speech counterpart of `openrouter_image_catalogue`, built the same way from OpenRouter's
documented API only:

- `/api/v1/models?output_modalities=speech&sort=most-popular` -- the ranking, the names, and
  the voices each model takes (`supported_voices`);
- `/api/v1/models/{id}/endpoints`, once per model -- who hosts it, the price, and whether a
  host takes a voice sample (`supports_voice_cloning`);
- `/api/v1/models/user?output_modalities=speech`, read with the bot owner's key.

The speech endpoint takes no `provider.data_collection` either, so the image catalogue's two
answers hold here for the same reasons: **one host company, or not listed**, and the account
listing says which models avoid training only in a sync whose text listing proved the account
filters at all. See that module's docstring.
"""
import datetime
import os
import time
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import orjson

from ...managers.storage_manager import IOManager
from .openrouter_catalogue import AUTHOR_PREFIX, AuthorInfo, format_price
from .openrouter_image_catalogue import _author_rows, _whole_listing_ids

#: Bumped when SpeechModelInfo's fields change, for the reason CATALOGUE_FORMAT is.
SPEECH_CATALOGUE_FORMAT = 1

#: Voice names a model's detail lines spell out before counting the rest.
_VOICES_NAMED = 6


class SpeechModelInfo(NamedTuple):
    id: str
    name: str
    author: str
    #: The voice ids the model takes, in OpenRouter's order. Empty for a model that lists
    #: none, which speaks in a voice of its own choosing.
    voices: Tuple[str, ...]
    #: The cheapest input and output rates its host lists, per unit. OpenRouter does not say
    #: what the unit is -- characters for most speech models, tokens for Gemini -- so the
    #: pickers show both per million without naming it.
    price_in: Optional[float]
    price_out: Optional[float]
    #: Whether its host takes a voice sample with the request.
    clones: bool
    #: As OpenRouterImageCatalogue's: False when a filtering account listing kept it, True
    #: when one left it out, None when none ever covered it.
    trains: Optional[bool]


def _hosting(body: Optional[bytes]):
    """(host companies, input rate, output rate, takes a sample) from one endpoints body;
    None if unreadable.

    Companies rather than endpoints: Mistral's standard, EU and zero-retention endpoints are
    one host under one policy. The company leads the endpoint's `tag` ("mistral/eu").
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
    rates_in: List[float] = []
    rates_out: List[float] = []
    clones = False
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        company = str(endpoint.get("tag") or endpoint.get("provider_name") or "").split("/", 1)[0]
        if company:
            hosts.add(company.lower())
        pricing = endpoint.get("pricing") if isinstance(endpoint.get("pricing"), dict) else {}
        for key, rates in (("prompt", rates_in), ("completion", rates_out)):
            try:
                rates.append(float(pricing[key]))
            except (KeyError, TypeError, ValueError):
                continue
        clones = clones or endpoint.get("supports_voice_cloning") is True
    return (hosts, min(rates_in) if rates_in else None,
            min(rates_out) if rates_out else None, clones)


def _open_to_all(info: SpeechModelInfo) -> bool:
    """Whether its one host is known not to train on prompts. Unknown counts as no."""
    return info.trains is False


def _from_record(record) -> SpeechModelInfo:
    info = SpeechModelInfo(*record)
    # JSON has no tuples; the voice screens compare against these.
    return info._replace(voices=tuple(info.voices))


def speech_price_text(info: SpeechModelInfo) -> str:
    if info.price_in is None:
        return "Price not listed"
    if not info.price_in and not info.price_out:
        return "Free"
    text = f"{format_price(info.price_in * 1_000_000)} in"
    if info.price_out:
        text += f" · {format_price(info.price_out * 1_000_000)} out"
    return text + " /1M"


def _voices_text(info: SpeechModelInfo) -> str:
    if not info.voices:
        return "no voices listed"
    return f"{len(info.voices)} voice{'s' if len(info.voices) != 1 else ''}"


class OpenRouterSpeechCatalogue:
    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "openrouter_speech_catalogue.json")
        self.models: Dict[str, SpeechModelInfo] = {}
        self.updated_at: Optional[str] = None
        self._popular: List[str] = []
        self._by_author: Dict[str, List[str]] = {}
        self._authors: List[AuthorInfo] = []
        self._open: frozenset = frozenset()
        self._open_authors: List[AuthorInfo] = []
        #: As OpenRouterImageCatalogue's: when an account listing last said which models
        #: train, and whether the latest sync's did.
        self.training_checked_at: Optional[int] = None
        self.training_current = False

    # --- Loading and syncing (run these in a thread) ------------------------------

    def load(self) -> None:
        data = IOManager.read_json(self.path) or {}
        if data.get("format") != SPEECH_CATALOGUE_FORMAT:
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

    def apply_listing(self, models_body: bytes, endpoint_bodies: Dict[str, Optional[bytes]],
                      account_body: Optional[bytes], account_filters_training: bool) -> None:
        """Parses a fresh speech listing, saves it, and installs it.

        The listing arrives sorted by popularity, so it is the ranking as well. Otherwise as
        `OpenRouterImageCatalogue.apply_listing`: the account listing counts only when
        `account_filters_training` says the same sync's text listing proved the settings
        filter, and a model whose endpoints could not be read keeps its last record rather
        than vanishing from every picker until tomorrow's sync.
        """
        listing = orjson.loads(models_body).get("data") or []
        kept = _whole_listing_ids(account_body) if account_filters_training else None
        self.training_current = kept is not None
        previous = self.models

        models: Dict[str, SpeechModelInfo] = {}
        for raw in listing:
            model_id = raw.get("id") if isinstance(raw, dict) else None
            if not model_id:
                continue

            hosting = _hosting(endpoint_bodies.get(model_id))
            if hosting is None:
                if model_id not in previous:
                    continue
                last = previous[model_id]
                price_in, price_out, clones = last.price_in, last.price_out, last.clones
            else:
                hosts, price_in, price_out, clones = hosting
                if len(hosts) != 1:
                    continue

            if kept is not None:
                trains = model_id not in kept
            else:
                trains = previous[model_id].trains if model_id in previous else None

            voices = raw.get("supported_voices")
            models[model_id] = SpeechModelInfo(
                id=model_id,
                name=raw.get("name") or model_id,
                author=model_id.split("/", 1)[0],
                voices=tuple(str(v) for v in voices) if isinstance(voices, list) else (),
                price_in=price_in, price_out=price_out, clones=clones,
                trains=trains,
            )

        if not models:
            return
        if kept is not None:
            self.training_checked_at = int(time.time())

        popular = list(models)
        updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        IOManager.write_json({"format": SPEECH_CATALOGUE_FORMAT, "updated_at": updated_at,
                              "popular": popular, "training_checked_at": self.training_checked_at,
                              "models": [list(info) for info in models.values()]}, self.path)
        self._install(models, popular, updated_at)

    def _install(self, models: Dict[str, SpeechModelInfo], popular: List[str],
                 updated_at: Optional[str]) -> None:
        """Rebuilds the orderings, then swaps them in together."""
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

    # --- Browsing -----------------------------------------------------------------
    #
    # The surface ModelPickerMixin reads off the text and image catalogues, so the TTS
    # category browses through the same code.

    def authors(self, *, show_training: bool) -> List[AuthorInfo]:
        return self._authors if show_training else self._open_authors

    def browse(self, key: str, *, show_training: bool) -> Tuple[List[str], Optional[str]]:
        """Speech models for a browse key, and a note to show above them (or None).

        Most Popular is OpenRouter's own ranking: speech calls are not counted on this bot.
        A text browse key carried over from another category reads as Most Popular.
        """
        def offered(ids: Iterable[str]) -> List[str]:
            return list(ids) if show_training else [m for m in ids if m in self._open]

        if not self.models:
            return [], "The OpenRouter speech model list has not loaded yet. Try again shortly."
        if key.startswith(AUTHOR_PREFIX):
            return offered(self._by_author.get(key[len(AUTHOR_PREFIX):], [])), None
        return offered(self._popular), None

    def open_to_all(self, model_id: str) -> bool:
        """For a typed id in a picker: an id the listing lacks is not judged here."""
        info = self.models.get(model_id)
        return info is None or _open_to_all(info)

    def is_open(self, model_id: str) -> bool:
        """For the model factory: listed, and its one host known not to train.

        Stricter than `open_to_all`, because a speech request has no `data_collection` to
        fall back on when the listing does not know a model.
        """
        info = self.models.get(model_id)
        return info is not None and _open_to_all(info)

    def voices(self, model_id: str) -> Optional[Tuple[str, ...]]:
        """The voices a listed model takes, or None for an id the listing does not know."""
        info = self.models.get(model_id)
        return info.voices if info else None

    def clones(self, model_id: str) -> bool:
        """Whether a listed model's host takes a voice sample with the request."""
        info = self.models.get(model_id)
        return bool(info and info.clones)

    def cloning_models(self) -> List[str]:
        """The listed models that can speak with a voice sample, most popular first."""
        return [m for m in self._popular if self.models[m].clones]

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
        """The price first, then how many voices it takes, in 100 characters."""
        info = self.models.get(model_id)
        if not info:
            return None
        parts = [speech_price_text(info), _voices_text(info)]
        if info.clones:
            parts.append("clones voices")
        if not _open_to_all(info):
            parts.append("⚠ may train")
        return " · ".join(parts)[:100]

    def describe_author(self, author: AuthorInfo) -> str:
        return f"{author.models} speech model{'s' if author.models != 1 else ''}"[:100]

    def detail_lines(self, model_id: str) -> Optional[str]:
        """Everything the picker's embed says about a chosen speech model."""
        info = self.models.get(model_id)
        if not info:
            return None
        lines = [f"{speech_price_text(info)} · {_voices_text(info)}"]
        if info.voices:
            named = ", ".join(info.voices[:_VOICES_NAMED])
            rest = len(info.voices) - _VOICES_NAMED
            lines.append(f"Voices: {named}" + (f" and {rest} more" if rest > 0 else ""))
        if info.clones:
            lines.append("Clones voices: a profile with a voice sample speaks in that voice.")
        lines.append("Sent the reply alone: the Director's Desk reaches Google speech models only.")
        if not _open_to_all(info):
            lines.append(("Its host may train on prompts" if info.trains
                          else "Not yet checked whether its host trains on prompts")
                         + ", so it is offered to the bot owner alone")
        return "\n".join(lines)
