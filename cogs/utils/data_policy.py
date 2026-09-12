"""Whether a conversation's Discord messages may reach an AI provider that trains on them.

Two provider routes may use what they are sent to train models: Google's free (unpaid)
Gemini tier, and the OpenRouter hosts whose policy allows training. Both are closed to a
server's traffic until the bot owner opts that server in, per provider, on the data policy
screen `/privacy` gives them. See GEMINI_FREE_TIER_BLOCKED in constants for why.

A Global Chat is a conversation wherever its card is opened, so it answers to the same
policy: the server's when it is opened in one, and closed anywhere else, where there is no
server to opt in. Everything else a DM sends a provider is its own user's command input
under their own key, and is not gated.

The decision is the bot owner's alone. Discord's terms bind the developer operating the
application, not a server, and Discord's permission -- the only thing that could make
opening either route compliant -- would be granted to that developer. So a record anyone
else wrote opens nothing, and nobody else can write one.

The choice lives in the server's plaintext `index.json` under `training_opt_in`, keyed by
provider. Sparse: an absent entry means "not allowed", and a revoked one is removed rather
than written as False, so the file records only decisions somebody actually made.
"""
import time
from typing import Any, Dict, Optional

from .constants import defaultConfig

TRAINING_OPT_IN_KEY = "training_opt_in"

#: The two providers the bot owner can open, in the order the data policy screen lists them.
TRAINING_PROVIDERS = ("gemini", "openrouter")


def may_set_data_policy(user_id: Optional[int]) -> bool:
    """Whether this user's opt-in counts: the bot owner's, and nobody else's."""
    try:
        return user_id is not None and int(user_id) == int(defaultConfig.DISCORD_OWNER_ID)
    except (TypeError, ValueError):
        return False


def may_pick_training_models(user_id: Optional[int]) -> bool:
    """Whether a model picker offers this user OpenRouter models that may train on prompts.

    The bot owner's, and nobody else's. Only the bot owner can open a server to a host that
    trains, so to anyone else a model no other host serves is one the data policy refuses
    wherever the bot owner has not opened that route.
    """
    return may_set_data_policy(user_id)


def may_save_free_gemini_key(user_id: Optional[int]) -> bool:
    """Whether /settings accepts a free-tier Gemini key from this user: the bot owner's alone.

    A free-tier key is held back from every server and every Global Chat until the bot owner
    opens that server to it, and nobody else can. Anyone else's would save, read as set up,
    and never carry a conversation.
    """
    return may_set_data_policy(user_id)


def opt_in_record(server_index: Optional[Dict[str, Any]], provider: str) -> Optional[Dict[str, Any]]:
    """Who opted this server in to `provider`, and when, or None if the bot owner has not."""
    entry = ((server_index or {}).get(TRAINING_OPT_IN_KEY) or {}).get(provider)
    if isinstance(entry, dict) and entry.get("allowed") and may_set_data_policy(entry.get("by")):
        return entry
    return None


def training_opt_in(server_index: Optional[Dict[str, Any]], provider: str) -> bool:
    """True when the bot owner has opted this server in to `provider`'s training-capable route."""
    return opt_in_record(server_index, provider) is not None


def set_training_opt_in(server_index: Dict[str, Any], provider: str, allowed: bool,
                        user_id: int) -> None:
    """Records the bot owner's decision on the index, in place."""
    if provider not in TRAINING_PROVIDERS:
        raise ValueError(f"Unknown training provider {provider!r}")
    if not may_set_data_policy(user_id):
        raise PermissionError("Only the bot owner can change a server's data policy")
    entries = server_index.setdefault(TRAINING_OPT_IN_KEY, {})
    if allowed:
        entries[provider] = {"allowed": True, "by": int(user_id), "at": int(time.time())}
    else:
        entries.pop(provider, None)
        if not entries:
            server_index.pop(TRAINING_OPT_IN_KEY, None)


def is_paid_gemini_slot(slot: Optional[Dict[str, Any]]) -> bool:
    """A key slot validated as billing-enabled.

    Anything else is treated as the free tier, including a slot saved before tiers were
    recorded: the settings screens already show a missing tier as "Free", and guessing
    paid would send a server's messages somewhere it never agreed to.
    """
    return (slot or {}).get("tier") == "paid"


def openrouter_data_collection(server_index: Optional[Dict[str, Any]]) -> Optional[str]:
    """The `provider.data_collection` value a request answering to this server's policy carries.

    "deny" unless the server opted in; None when it has, which sends nothing and leaves
    routing to the key owner's own OpenRouter privacy settings, as before. `server_index`
    is None for a Global Chat opened outside a server, which nothing can opt in.
    """
    return None if training_opt_in(server_index, "openrouter") else "deny"
