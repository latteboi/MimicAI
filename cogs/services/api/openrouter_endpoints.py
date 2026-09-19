"""One OpenRouter text model's endpoints, for pinning a model to one of them.

Read from `/api/v1/models/{id}/endpoints`, the documented listing the image and speech
syncs already call -- but on demand, when someone opens the Hosts screen, and never in
the daily sync: there are several hundred text models and a pin is set on a handful.

An endpoint is a host *at a tier*. Gemini 3.8 Flash lists six: Google AI Studio and
Google Vertex, each at flex, standard and priority. Its `tag` names exactly one
(`google-ai-studio/flex`, `google-vertex/global/priority`), and `provider.order`
accepts that same tag -- so the tag is the whole of what a pin stores, and a pin names
its tier as well as its host.
"""
from typing import NamedTuple, Optional, Tuple

import orjson

from ...utils.constants import OPENROUTER_SERVICE_TIER_VALUES
from ...utils.helpers import OPENROUTER_ENDPOINT_TAG
from .openrouter_catalogue import format_price

#: The documented listing, with the model id filled in.
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{}/endpoints"


class Endpoint(NamedTuple):
    tag: str
    host: str
    #: Per million tokens, discount already applied. None when OpenRouter sent no price.
    prompt_1m: Optional[float]
    completion_1m: Optional[float]
    #: Percentage over the last 30 minutes, as OpenRouter reports it; None when absent.
    uptime: Optional[float]
    #: "flex", "priority", or "" for the standard tier.
    tier: str
    #: False when the endpoint lists its parameters and `temperature` is not among them.
    #: OpenRouter drops an unsupported parameter rather than refusing the request, so a
    #: profile pinned here would lose its temperature without a word said.
    takes_temperature: bool


def _per_million(value) -> Optional[float]:
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


def parse_endpoints(body: Optional[bytes]) -> Optional[Tuple[Endpoint, ...]]:
    """Every endpoint in one listing body, cheapest first; None if the body is unreadable.

    An endpoint whose tag would be refused at the wire is left out rather than offered,
    so nothing can be pinned that `resolve_openrouter_endpoint` would then ignore.
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
    raw = record.get("endpoints")
    if not isinstance(raw, list):
        return None
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tag = item.get("tag")
        if not (isinstance(tag, str) and OPENROUTER_ENDPOINT_TAG.fullmatch(tag)):
            continue
        pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
        suffix = tag.rsplit("/", 1)[-1] if "/" in tag else ""
        params = item.get("supported_parameters")
        uptime = item.get("uptime_last_30m")
        out.append(Endpoint(
            tag=tag,
            host=str(item.get("provider_name") or tag.split("/", 1)[0]),
            prompt_1m=_per_million(pricing.get("prompt")),
            completion_1m=_per_million(pricing.get("completion")),
            uptime=float(uptime) if isinstance(uptime, (int, float)) and not isinstance(uptime, bool) else None,
            tier=suffix if suffix in OPENROUTER_SERVICE_TIER_VALUES else "",
            takes_temperature=not isinstance(params, list) or "temperature" in params,
        ))

    def cost(e: Endpoint):
        if e.prompt_1m is None or e.completion_1m is None:
            return (1, 0.0, e.tag)
        return (0, e.prompt_1m + e.completion_1m, e.tag)

    # One per tag, the cheapest. OpenRouter can list a host twice under one tag
    # (DeepSeek V4.1 Flash has two `baseten/fp8`, differing only in uptime), and a pin
    # stores nothing but the tag, so the two are one choice -- and a dropdown offering
    # both is refused outright by Discord for the repeated option value.
    seen = set()
    return tuple(e for e in sorted(out, key=cost) if not (e.tag in seen or seen.add(e.tag)))


def option_label(endpoint: Endpoint) -> str:
    """The host, then whatever its tag says beyond the host slug: "Google · Global · Priority"."""
    extra = [part.upper() if part.startswith(("fp", "int", "bf")) else part.title()
             for part in endpoint.tag.split("/")[1:]]
    return " · ".join([endpoint.host, *extra])[:100]


def option_description(endpoint: Endpoint) -> str:
    """Price, uptime and anything that would quietly change a reply, in 100 characters."""
    if endpoint.prompt_1m is None or endpoint.completion_1m is None:
        parts = ["Price not listed"]
    else:
        parts = [f"{format_price(endpoint.prompt_1m)} in · "
                 f"{format_price(endpoint.completion_1m)} out /1M"]
    if endpoint.uptime is not None:
        parts.append(f"{endpoint.uptime:.1f}% up")
    if not endpoint.takes_temperature:
        parts.append("ignores temperature")
    return " · ".join(parts)[:100]


def base_model_id(model_id: str) -> str:
    """The id the listing is keyed by: `google/gemini-3.8-flash:floor` lists as the model."""
    return model_id.split(":", 1)[0]
