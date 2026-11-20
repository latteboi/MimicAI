from ..utils.constants import *

import discord
from discord import ui
import asyncio
import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple

from .base_components import (
    BlockedGuard, PageJumpModal, TabbedView, add_button, add_select, paged_nav_options,
)
from ..utils.birthdays import MONTH_NAMES, format_birthday, parse_birthday, valid_birthday
from ..utils.helpers import _get_user_hash, _resolve_zoneinfo, suppress_link_previews
from ..utils.data_policy import is_paid_gemini_slot, may_save_free_gemini_key
from ..utils import openrouter_signin
from ..managers.subscriptions import (ACTIVE, carried_over, period_usage, runs_out_at, tier_name,
                                      used_share)
from ..utils.helpers import subscription_curated
from ..services.api import stripe
from ..services.api.stripe import StripeError
from .base_components import dismissible, text_embed

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog


class OllamaHostModal(ui.Modal, title="Set Ollama Host URL"):
    host_input = ui.TextInput(label="Ollama API URL", placeholder="http://127.0.0.1:11434 (Blank for default)", required=False)
    
    def __init__(self, view):
        super().__init__()
        self.parent_view = view
        
        if hasattr(view, 'profile_name') and view.profile_name != "BULK_APPLY":
            cfg = view.cog.profile_manager._get_profile_config(view.user_id, view.profile_name, getattr(view, 'is_borrowed', False)) or {}
            self.host_input.default = cfg.get("ollama_host_url", OLLAMA_LOCAL_URL)
        else:
            self.host_input.default = getattr(view, 'models_state', {}).get("ollama_host_url", OLLAMA_LOCAL_URL)

    async def on_submit(self, interaction: discord.Interaction):
        # The host is where every message a profile sees would be sent, and the probe
        # below dials it from the bot's own network: the bot owner's call alone.
        if not self.parent_view.cog.user_settings.may_use_ollama(
                getattr(self.parent_view, "user_id", interaction.user.id)):
            await interaction.response.send_message(embed=text_embed(OLLAMA_OWNER_ONLY), ephemeral=True)
            return
        url = self.host_input.value.strip()
        if not url:
            url = OLLAMA_LOCAL_URL
        elif not url.startswith("http"):
            url = "http://" + url
        
        self.parent_view._save_changes("ollama_host_url", url)
        
        await interaction.response.defer()
        self.parent_view.ollama_working = "processing"
        
        await self.parent_view._update_ollama_status()
        self.parent_view._build_view()
        await interaction.edit_original_response(**self.parent_view._picker_render())

class SubmitAPIKeyModal(ui.Modal, title="Submit API Key"):
    key_input = ui.TextInput(label="API Key", placeholder="Paste your API key here...", required=True)

    def __init__(self, cog: 'MimicCog', slot_id: str, provider: str, view: Optional[ui.View] = None,
                 assign_personal: bool = False):
        super().__init__()
        self.cog = cog
        self.slot_id = slot_id
        self.provider = provider
        self.view = view
        # `/start` pastes a key to make it work, so it is always Personal. `/settings` makes
        # it Personal only where there is none (`save_api_key`): a key nothing points at
        # does nothing. Either way a server is claimed by `KeyScopeMixin._claim_server`.
        self.assign_personal = assign_personal

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        raw_key = self.key_input.value.strip()

        if not raw_key.startswith(("AIzaSy", "AQ.", "sk-or-")):
            await interaction.followup.send(embed=text_embed("❌ **Invalid Format.** Keys must start with `AIzaSy`, `AQ.`, or `sk-or-`."), ephemeral=True)
            return

        saved, msg = await save_api_key(self.cog, interaction.user.id, self.slot_id, self.provider,
                                        raw_key, self.assign_personal)
        if saved and isinstance(self.view, KeyScopeMixin):
            msg += self.view._key_landed(self.slot_id, interaction.guild)
        if saved and self.view:
            await self.view._repaint()

        await interaction.followup.send(embed=text_embed(msg), ephemeral=True)


async def save_api_key(cog: 'MimicCog', user_id: int, slot_id: str, provider: str, raw_key: str,
                       assign_personal: bool) -> Tuple[bool, str]:
    """Validates a key and saves it to a slot: where a pasted key and an OpenRouter sign-in
    both end. Returns (saved, the line to tell the user). A caller then adds
    `KeyScopeMixin._claim_server`'s line: skipped, a key added in a server stays off it."""
    is_valid, err, tier = await cog.api_service._validate_api_keys(
        raw_key if provider == "gemini" else None,
        raw_key if provider == "openrouter" else None
    )
    if not is_valid:
        return False, f"❌ **Validation Failed:** {suppress_link_previews(str(err))}"

    # Refused before anything is written -- see may_save_free_gemini_key.
    if (provider == "gemini" and not is_paid_gemini_slot({"tier": tier})
            and not may_save_free_gemini_key(user_id)):
        return False, f"❌ {GEMINI_FREE_TIER_KEY_REFUSED}"

    user_data = cog.storage_manager._get_user_keys_data(user_id)
    user_data.setdefault("slots", {})[slot_id] = {
        "key": raw_key,
        "provider": provider,
        "tier": tier
    }
    personal = user_data.setdefault("personal_assignments", {})
    # Asked for, or the first key of its provider: a key nothing points at does nothing.
    made_personal = (assign_personal or not personal.get(provider)) and personal.get(provider) != slot_id
    if made_personal:
        personal[provider] = slot_id
    cog.storage_manager._save_user_keys_data(user_id, user_data)

    cog.decrypted_key_cache[(user_id, slot_id)] = raw_key

    # has_personal_key is set by _save_user_keys_data above, together with the
    # stat stamp _index_is_consistent checks it against. Setting the flag here
    # without the stamp is what this used to do, and it bought a full index
    # repair on the next hourly pass.

    msg = f"✅ {provider.title()} key saved to slot `{slot_id}` ({tier.title()} Tier)."
    if made_personal:
        msg += "\nIt is now your **Personal** key."
    if provider == "gemini" and not is_paid_gemini_slot({"tier": tier}):
        msg += "\nGoogle may train on what a free-tier key is sent, so it is not used in servers or Global Chat."
    return True, msg

class OverrideConfirmView(BlockedGuard, ui.View):
    """Asks before taking over assignments another key holds, then applies them through
    the view that asked -- `/settings` -> API Keys or `/start`'s key step."""

    def __init__(self, parent_view: "KeyScopeMixin", slot_id: str, new_scopes):
        super().__init__(timeout=120)
        self.cog = parent_view.cog
        self.parent_view = parent_view
        self.slot_id = slot_id
        # A snapshot: the view that asked can go on changing ticks behind this prompt.
        self.new_scopes = set(new_scopes)

    @staticmethod
    def prompt(conflicts: List[str]) -> str:
        return ("⚠️ **Key Assignment Override**\nAssigning this key will overwrite existing "
                "assignments for the following scopes:\n"
                + "\n".join(f"- {c}" for c in conflicts) + "\n\nDo you want to proceed?")

    @ui.button(label="Yes, Override", style=discord.ButtonStyle.danger)
    async def confirm_override(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.defer(ephemeral=True)
        self.parent_view._apply_assignments(self.slot_id, self.new_scopes)
        await self.parent_view._repaint()
        await interaction.edit_original_response(content=None, embed=dismissible(text_embed("✅ Assignments saved successfully.")), view=None)

    @ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel_override(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.edit_message(content=None, embed=dismissible(text_embed("❌ Assignment update cancelled.")), view=None)

def _defaults_view(v):
    # Imported here, not at module scope: gui_defaults adopts ModelPickerMixin from
    # gui_profiles, and gui_profiles imports OllamaHostModal from this module. A
    # top-level import would close that cycle.
    from .gui_defaults import SettingsDefaultsView
    return SettingsDefaultsView(v.cog, v.original_interaction)


class SettingsBaseView(TabbedView):
    TABS = (
        ("Home", "home", lambda v: SettingsHomeView(v.cog, v.original_interaction)),
        ("About Me", "about", lambda v: SettingsAboutView(v.cog, v.original_interaction)),
        ("API Keys", "api", lambda v: SettingsAPIView(v.cog, v.original_interaction)),
        ("Override Defaults", "defaults", _defaults_view),
        ("Child Bots", "bots", lambda v: SettingsChildBotView(v.cog, v.original_interaction)),
    )

    @classmethod
    def _tabs_for(cls, source) -> tuple:
        # Registering a child bot hands over a bot token, so the tab is the bot owner's alone.
        if source.user_id == int(defaultConfig.DISCORD_OWNER_ID):
            return cls.TABS
        return tuple(t for t in cls.TABS if t[1] != "bots")

class SettingsHomeView(SettingsBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "home")

    async def update_display(self):
        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        slots = user_data.get("slots", {})
        
        has_gem = any(s.get("provider") == "gemini" for s in slots.values())
        has_or = any(s.get("provider") == "openrouter" for s in slots.values())
        
        stat_gemini = "✅ **`Set`**" if has_gem else "❌ `Not Set`"
        stat_or = "✅ **`Set`**" if has_or else "❌ `Not Set`"
        
        is_owner = self.user_id == int(defaultConfig.DISCORD_OWNER_ID)
        if is_owner:
            child_bots = [b for b in self.cog.child_bots.values() if b['owner_id'] == self.user_id]
            bot_text = f"You own **{len(child_bots)}** Child Bots." if child_bots else "You do not own any Child Bots."
        
        primary_count = 0
        for g in self.cog.bot.guilds:
            idx = self.cog.server_manager._get_server_index(str(g.id))
            for assigned in idx.get("assigned_keys", {}).values():
                if assigned.get("user_id") == self.user_id:
                    primary_count += 1
                    break

        embed = discord.Embed(title="MimicAI Control Panel", description="Manage your API keys and defaults from one place.", color=discord.Color.dark_teal())
        embed.set_thumbnail(url=THINKING_THUMBNAIL_URL)
        
        embed.add_field(name="API Key Slots", value=f"**Google Gemini:** {stat_gemini}\n**OpenRouter:** {stat_or}", inline=True)
        if is_owner:
            embed.add_field(name="Child Bots", value=bot_text, inline=True)
        embed.add_field(name="Server Contributions", value=f"Active Assignments: `{primary_count} servers`", inline=False)
        
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

def _standing_text(cog: 'MimicCog', user_id: int) -> str:
    """Whether the user is restricted, since when, and until when.

    Never why, and never by whom: those are the operator's note. The Privacy Policy
    discloses that a reason is held, not that it is shown. Set membership decides, not
    the record, so an elapsed block reads as lifted before the sweep has pruned it.
    """
    strikes = cog.server_manager.active_strikes(user_id)
    strike_line = (f"**{len(strikes)} strike{'s' if len(strikes) != 1 else ''}** of {STRIKE_THRESHOLD}: "
                   f"a profile of yours was Refused by its content rating. The oldest expires "
                   f"<t:{int(strikes[0]['at'] + STRIKE_LIFETIME_SECONDS)}:D>." if strikes else None)
    if user_id not in cog.generation_blocked:
        return strike_line or "No restrictions."
    entry = cog.server_manager.blacklist_entry(user_id) or {}
    lines = ["**Restricted.** Only `/privacy`, `/terms` and `/settings` are available to you."]
    try:
        since = int(datetime.datetime.fromisoformat(entry["at"]).timestamp())
        lines.append(f"Since <t:{since}:D>")
    except (KeyError, TypeError, ValueError):
        pass
    until = entry.get("until")
    lines.append(f"Lifts <t:{int(until)}:R>" if until else "No end date.")
    if strike_line:
        lines.append(strike_line)
    return "\n".join(lines)


#: Under the choice, in Override Defaults and in `/start`'s step: what it does, and what it leaves.
PROVIDER_PREFERENCE_NOTE = (
    "Every model runs Primary \u2192 Fallback on this provider. A profile with its **Final "
    "Fallback** turned on (Params, off unless you turn it on) tries the other one last, so "
    "one provider's outage or a spent key cannot silence it. New profiles start on this "
    "provider; a model you picked yourself stays picked.")


def build_about_embed(cog: 'MimicCog', user_id: int) -> discord.Embed:
    """About Me, with the user's standing.

    The About tab renders this, and so does `/settings` for a blocked user in place of
    the tabs, with no view attached -- nothing on it is theirs to change while blocked.
    """
    about = cog.user_settings.get_user_about(user_id)
    stored_tz = about.get("timezone")

    # The resolved value is what actually gets used, so it is what is shown --
    # "unset" is reported as the fallback it resolves to, not as a blank.
    tz_name = stored_tz or "UTC"
    try:
        tz_obj, _ = _resolve_zoneinfo(tz_name)
        now_str = datetime.datetime.now(tz_obj).strftime("%I:%M %p (%Z)")
    except Exception:
        now_str = "Unknown"

    tz_value = (f"`{tz_name}` -- local time `{now_str}`" if stored_tz
                else f"`Not set` -- using `UTC`, local time `{now_str}`")

    embed = discord.Embed(
        title="About Me",
        description=("Your own settings, separate from any character's. These follow "
                     "you into every server."),
        color=discord.Color.dark_teal())
    embed.set_thumbnail(url=THINKING_THUMBNAIL_URL)

    embed.add_field(
        name="\N{GLOBE WITH MERIDIANS} Your Timezone",
        value=(f"{tz_value}\n"
               "Timestamps your messages carry into a character's history. A "
               "character's *own* clock is set per profile, under "
               "`/profile manage` -> Timezone."),
        inline=False)

    birthday = format_birthday(about.get("birthday"))
    embed.add_field(
        name="\N{BIRTHDAY CAKE} Your Birthday",
        value=(f"`{birthday or 'Not set'}`\n"
               "A character you are talking with knows it the day before, on the day and the "
               "day after, going by your timezone. Only the day and month are kept."),
        inline=False)

    embed.add_field(
        name="\N{BUST IN SILHOUETTE} Your Identifiers",
        value=(f"**Discord ID:** `{user_id}`\n"
               f"**Handle characters see:** `{_get_user_hash(user_id)}`"),
        inline=False)

    embed.add_field(name="\N{SCALES} Standing", value=_standing_text(cog, user_id),
                    inline=False)
    return embed


class SettingsAboutView(SettingsBaseView):
    """The user's own settings, as opposed to any character's.

    Everything here is stored sparsely in `index.json["about"]` -- plaintext, already
    cached in `cog.user_indices`, and read on the turn path. See the block above
    UserSettings.get_user_about for why it is not an encrypted shard and not a key
    inside "defaults".
    """

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "about")
        self._build_view()

    def _build_view(self):
        self.clear_items()
        add_button(self, "Set Timezone", self._act_timezone, style=discord.ButtonStyle.primary,
                   row=0, emoji="\N{GLOBE WITH MERIDIANS}")

        about = self.cog.user_settings.get_user_about(self.user_id)
        if about.get("timezone"):
            add_button(self, "Clear Timezone", self._act_clear_timezone,
                       style=discord.ButtonStyle.secondary, row=0)

        add_button(self, "Set Birthday", self._act_birthday, style=discord.ButtonStyle.primary,
                   row=1, emoji="\N{BIRTHDAY CAKE}")
        if about.get("birthday"):
            add_button(self, "Clear Birthday", self._act_clear_birthday,
                       style=discord.ButtonStyle.secondary, row=1)

        self._add_nav_buttons()

    async def update_display(self):
        await self.original_interaction.edit_original_response(
            content=None, embed=build_about_embed(self.cog, self.user_id), view=self)

    async def _act_timezone(self, i: discord.Interaction):
        # Deferred, not module scope: gui_profiles imports OllamaHostModal from this
        # module, so a top-level import would close the cycle. Same reason
        # nav_defaults defers gui_defaults.
        from .gui_profiles import UserTimezoneView
        view = UserTimezoneView(self.cog, self)
        await i.response.send_message(embed=text_embed(view._get_header_content()),
                                      view=view, ephemeral=True)

    async def _act_clear_timezone(self, i: discord.Interaction):
        await i.response.defer()
        about = self.cog.user_settings.get_user_about(self.user_id)
        about.pop("timezone", None)
        self.cog.user_settings.save_user_about(self.user_id, about)
        self._build_view()
        await self.update_display()

    async def _act_birthday(self, i: discord.Interaction):
        await i.response.send_modal(UserBirthdayModal(self.cog, self))

    async def _act_clear_birthday(self, i: discord.Interaction):
        await i.response.defer()
        about = self.cog.user_settings.get_user_about(self.user_id)
        about.pop("birthday", None)
        self.cog.user_settings.save_user_about(self.user_id, about)
        self._build_view()
        await self.update_display()


class UserBirthdayModal(ui.Modal, title="Your Birthday"):
    """Day and month only, into `index.json["about"]["birthday"]`.

    No year: a character can mention a birthday without the bot holding anyone's age.
    Day and month are separate boxes, so no date can be read the wrong way round.
    """

    def __init__(self, cog: 'MimicCog', parent_about_view):
        super().__init__()
        self.cog = cog
        self.parent_about_view = parent_about_view
        current = valid_birthday(
            cog.user_settings.get_user_about(parent_about_view.user_id).get("birthday")) or {}
        self.day = ui.TextInput(label="Day", default=str(current.get("day", "")),
                                required=False, max_length=2, placeholder="1-31")
        self.month = ui.TextInput(
            label="Month", default=MONTH_NAMES[current["month"] - 1] if current.get("month") else "",
            required=False, max_length=9, placeholder="March, Mar or 3")
        self.add_item(self.day)
        self.add_item(self.month)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            birthday = parse_birthday(self.day.value, self.month.value, allow_year=False)
        except ValueError as e:
            await interaction.response.send_message(
                embed=text_embed(f"❌ **Invalid Input:** {suppress_link_previews(str(e))}"), ephemeral=True)
            return
        user_id = self.parent_about_view.user_id
        about = self.cog.user_settings.get_user_about(user_id)
        if birthday:
            about["birthday"] = birthday
        else:
            about.pop("birthday", None)
        self.cog.user_settings.save_user_about(user_id, about)

        await interaction.response.defer()
        self.parent_about_view._build_view()
        await self.parent_about_view.update_display()

def _fit_list(items: List[str], sep: str, limit: int) -> str:
    """Joins as many items as fit in `limit` characters, then says how many did not."""
    out, used = [], 0
    for n, item in enumerate(items):
        if used + len(item) + len(sep) > limit:
            return sep.join(out + [f"…and {len(items) - n} more"])
        out.append(item)
        used += len(item) + len(sep)
    return sep.join(out)


#: The four key slots, (slot id, label, provider): two per provider, in the order API Keys
#: lists them, under the subscription.
KEY_SLOTS = (
    ("openrouter_key_1", "OpenRouter Key 1", "openrouter"),
    ("openrouter_key_2", "OpenRouter Key 2", "openrouter"),
    ("google_key_1", "Google Gemini Key 1", "gemini"),
    ("google_key_2", "Google Gemini Key 2", "gemini"),
)
#: The MimicAI subscription is assigned like a slot, but holds no pasted key: it is not in
#: KEY_SLOTS, so `_slot_for` never puts one there. See `StorageManager._key_in_slot`.
SLOT_PROVIDER = {**{slot: provider for slot, _label, provider in KEY_SLOTS},
                 SUBSCRIPTION_SLOT: "openrouter"}
KEY_SLOT_IDS = frozenset(slot for slot, _label, _provider in KEY_SLOTS)
SLOT_LABELS = {**{slot: label for slot, label, _provider in KEY_SLOTS},
               SUBSCRIPTION_SLOT: "MimicAI Subscription"}


class KeyScopeMixin:
    """Where one key slot applies: Personal, and servers you administer.

    One picker for `/settings` -> API Keys, which stages ticks until Save Assignments,
    and `/start`'s key step, which applies each pick as it is made (`_scopes_picked`)
    and always keeps Personal (`_OFFER_PERSONAL`). The dropdown is paged the way Set
    Models pages its models: page controls first, then the fixed Personal row, then one
    page of servers. It used to list the first 24 and stop, and touching it dropped any
    assignment beyond them.

    Needs `cog`, `user_id` and `update_display`; `_repaint` is what everything that
    changes an assignment calls afterwards, so a host that shows more than the picker
    overrides that.
    """

    _SCOPE_NAV_VALUES = ("scope_prev", "scope_jump", "scope_next")
    #: 3 controls + Personal + 21 servers = Discord's 25.
    _SERVERS_PER_PAGE = 21
    _OFFER_PERSONAL = True

    def _init_scopes(self):
        self.selected_slot = None
        # Staged across every page until saved; see _load_scopes.
        self.selected_scopes = set()
        self.scope_page = 0
        self.admin_guilds = []
        for g in self.cog.bot.guilds:
            m = g.get_member(self.user_id)
            if m and m.guild_permissions.administrator:
                self.admin_guilds.append(g)
        # Sorted, so a server stays on the same page and can be found by name.
        self.admin_guilds.sort(key=lambda g: g.name.casefold())
        # The OpenRouter sign-in link on screen, the slot it fills, and the wait for its code.
        self._sign_in_url = self._sign_in_slot = self._sign_in_task = None

    def _slot_for(self, provider: str) -> str:
        filled = (self.cog.storage_manager._get_user_keys_data(self.user_id) or {}).get("slots") or {}
        mine = [slot for slot, _label, p in KEY_SLOTS if p == provider]
        # An empty slot first; with both full, the first is replaced, as Edit Key would.
        return next((s for s in mine if s not in filled), mine[0])

    def _key_landed(self, slot: str, guild: Optional[discord.Guild]) -> str:
        """After a key is saved to `slot` from this view, pasted or signed in: claims the
        server it was added in and reloads the ticks. The line to tell the user."""
        msg = self._claim_server(slot, guild)
        if not self._saved_scopes(SLOT_PROVIDER[slot], slot):
            msg += ("\nIt isn't used anywhere yet: pick where under *Assign this key to...*, "
                    "then press **Save Assignments**.")
        # What is saved may have just changed under the ticks this view staged.
        self._load_scopes()
        return msg

    def _sign_in_link(self, slot: Optional[str] = None) -> str:
        """Sign in with OpenRouter's link, and the wait for its code, begun when the link is
        first drawn: a link button opens the browser without an interaction reaching us.
        Kept across repaints, since the user may already have it open. It fills `slot`, or
        with None, `_slot_for` when the key lands; a link for another slot ends this one."""
        if self._sign_in_url is not None:
            if slot == self._sign_in_slot:
                return self._sign_in_url
            # Still waiting, so nothing is lost: once a code lands the url is None.
            self._sign_in_task.cancel()
        self._sign_in_url, state, verifier = openrouter_signin.start()
        self._sign_in_slot = slot
        self._sign_in_task = asyncio.create_task(self._await_sign_in(slot, state, verifier))
        return self._sign_in_url

    async def _await_sign_in(self, slot: Optional[str], state: str, verifier: str):
        # As long as the view lives: the interaction token it repaints through lasts that long.
        code = await openrouter_signin.receive_code(self.cog.bot, state, self.timeout or 900)
        # Spent either way; the next repaint draws a fresh link.
        self._sign_in_url = None
        if code is None:
            return
        slot = slot or self._slot_for("openrouter")
        try:
            key = await openrouter_signin.exchange(code, verifier)
            # `/start` makes every key Personal (it offers no Personal tick); `/settings`
            # only where there is none, as a paste does.
            saved, message = await save_api_key(self.cog, self.user_id, slot, "openrouter", key,
                                                assign_personal=not self._OFFER_PERSONAL)
        except openrouter_signin.SignInFailed as e:
            saved, message = False, f"❌ {e}"
        if saved:
            message += self._key_landed(slot, self.original_interaction.guild)
            message += ("\n\nFree models work now. For the rest, add credit at "
                        "<https://openrouter.ai/settings/credits>.")
        try:
            # A view left by a tab switch, or timed out, would paint over what replaced it.
            if not self.is_finished():
                await self._repaint()
            await self.original_interaction.followup.send(embed=text_embed(message), ephemeral=True)
        except discord.HTTPException:
            pass  # The token has lapsed. The key is saved: reopening the panel shows it.

    def _saved_scopes(self, provider: str, slot: Optional[str] = None) -> set:
        """Where `slot` (the selected one by default) is assigned now, as dropdown values."""
        slot = slot or self.selected_slot
        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        scopes = set()
        if user_data.get("personal_assignments", {}).get(provider) == slot:
            scopes.add("personal")
        for g in self.admin_guilds:
            idx = self.cog.server_manager._get_server_index(str(g.id))
            assigned = idx.get("assigned_keys", {}).get(provider)
            if assigned and assigned.get("user_id") == self.user_id and assigned.get("slot") == slot:
                scopes.add(str(g.id))
        return scopes

    def _claim_server(self, slot: str, guild: Optional[discord.Guild]) -> str:
        """A key added in a server you run that has none becomes its key too, unasked: a key
        saved and never assigned was the commonest way a server ended up with none. Never
        takes a server from another key, nor gives it one its data policy would hold back.
        The line to tell the user, or ''."""
        provider = SLOT_PROVIDER[slot]
        st = self.cog.storage_manager
        if (guild is None or not any(g.id == guild.id for g in self.admin_guilds)
                or st.guild_has_text_key(guild.id)
                or (provider == "gemini" and not st._gemini_slot_allowed_in_guild(guild.id, self.user_id, slot))
                or self._assignment_conflicts(slot, {str(guild.id)})):
            return ""
        self._apply_assignments(slot, self._saved_scopes(provider, slot) | {str(guild.id)})
        return (f"\nIt is now **{guild.name}**'s key too, so everyone there can talk to the characters "
                "on it. `/settings` → API Keys changes where it applies.")

    def _load_scopes(self):
        """Resets the staged ticks to what is saved.

        Called when a slot is chosen or deleted, never from a rebuild: a page turn
        rebuilds the view, and reloading there would throw away ticks not yet saved.
        """
        self.selected_scopes = set()
        if self.selected_slot:
            self.selected_scopes = self._saved_scopes(SLOT_PROVIDER[self.selected_slot])

    def _scope_pages(self) -> int:
        return max(1, (len(self.admin_guilds) - 1) // self._SERVERS_PER_PAGE + 1)

    def _page_guilds(self) -> list:
        start = self.scope_page * self._SERVERS_PER_PAGE
        return self.admin_guilds[start:start + self._SERVERS_PER_PAGE]

    def _scope_labels(self, scopes) -> List[str]:
        """Display names for dropdown values: Personal first, then servers in page order."""
        labels = ["Personal"] if "personal" in scopes else []
        return labels + [f"Server: {g.name}" for g in self.admin_guilds if str(g.id) in scopes]

    def _add_scope_select(self, row: int, placeholder: str = "Assign this key to..."):
        num_pages = self._scope_pages()
        self.scope_page = max(0, min(self.scope_page, num_pages - 1))
        options = paged_nav_options(self.scope_page, num_pages,
                                    values=self._SCOPE_NAV_VALUES, nav_suffix=" of servers")
        if self._OFFER_PERSONAL:
            # "Personal" is not "in a DM": it is the key your own Global Chat and your
            # profiles' background work run on, wherever you happen to run them.
            options.append(discord.SelectOption(
                label="Personal", value="personal",
                description="Your Global Chat (anywhere) and your profiles' background work"[:100],
                default=("personal" in self.selected_scopes)))
        for g in self._page_guilds():
            options.append(discord.SelectOption(label=f"Server: {g.name}"[:100], value=str(g.id),
                                                default=(str(g.id) in self.selected_scopes)))
        add_select(self, options, self.scope_select_callback, placeholder=placeholder,
                   min_values=0, max_values=len(options), row=row)

    async def scope_select_callback(self, interaction: discord.Interaction):
        values = set(interaction.data['values'])
        # A submission can only speak for the rows it showed, so it settles this page
        # and leaves every other page's ticks alone. Ticks made alongside a page
        # control count too.
        on_page = {str(g.id) for g in self._page_guilds()}
        if self._OFFER_PERSONAL:
            on_page.add("personal")
        self.selected_scopes = (self.selected_scopes - on_page) | (values & on_page)

        prev_value, jump_value, next_value = self._SCOPE_NAV_VALUES
        if jump_value in values:
            async def on_jump(i: discord.Interaction, page: int):
                self.scope_page = page
                await i.response.defer()
                await self._repaint()

            await interaction.response.send_modal(
                PageJumpModal(self._scope_pages(), on_jump, zero_indexed=True))
        else:
            if next_value in values:
                self.scope_page += 1
            elif prev_value in values:
                self.scope_page -= 1
            await interaction.response.defer()
        await self._scopes_picked(interaction)
        # Repainted through the original message, so the ticks just made reach the
        # embed even while the jump prompt is open.
        await self._repaint()

    async def _scopes_picked(self, interaction: discord.Interaction):
        """After a submission, its interaction already answered. Staging does nothing."""

    async def _repaint(self):
        self.setup_items()
        await self.update_display()

    @staticmethod
    def _rival(slot: str):
        """(provider, test on its assigned slot) of the assignment that may not share a scope
        with `slot`, or (None, None). A scope runs on the MimicAI subscription or on its own
        keys, never both: the subscription and a Gemini key displace each other."""
        if slot == SUBSCRIPTION_SLOT:
            return "gemini", lambda assigned: True
        if SLOT_PROVIDER[slot] == "gemini":
            return "openrouter", lambda assigned: assigned == SUBSCRIPTION_SLOT
        return None, None

    def _assignment_conflicts(self, slot: str, scopes) -> List[str]:
        """Scopes in `scopes` another key or user holds now, in the confirm prompt's words."""
        provider = SLOT_PROVIDER[slot]
        rival, displaces = self._rival(slot)
        rival_words = ("the MimicAI subscription, which this key replaces" if rival == "openrouter"
                       else "a Gemini key, which the subscription replaces")
        conflicts = []
        if "personal" in scopes:
            user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
            personal = user_data.get("personal_assignments", {})
            curr_personal = personal.get(provider)
            if curr_personal and curr_personal != slot:
                conflicts.append(f"Personal Scope (Currently uses {SLOT_LABELS.get(curr_personal, curr_personal)})")
            if rival and personal.get(rival) and displaces(personal[rival]):
                conflicts.append(f"Personal Scope (Currently uses {rival_words})")
        for g in self.admin_guilds:
            if str(g.id) not in scopes:
                continue
            assigned_keys = self.cog.server_manager._get_server_index(str(g.id)).get("assigned_keys", {})
            assigned = assigned_keys.get(provider)
            if assigned and (assigned.get("user_id") != self.user_id or assigned.get("slot") != slot):
                conflicts.append(f"Server: {g.name} (Currently assigned by another key/user)")
            if rival and assigned_keys.get(rival) and displaces(assigned_keys[rival].get("slot")):
                conflicts.append(f"Server: {g.name} (Currently uses {rival_words})")
        return conflicts

    def _apply_assignments(self, slot: str, scopes):
        """Makes `slot` apply exactly to `scopes`, among the scopes this view listed.

        Only servers you administer are touched. The override prompt used to walk every
        server the bot is in, so confirming one takeover also dropped this key from a
        server you had since stopped administering, which no dropdown could show you.
        """
        provider = SLOT_PROVIDER[slot]
        rival, displaces = self._rival(slot)
        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        personal = user_data.setdefault("personal_assignments", {})
        if ("personal" in scopes and rival and personal.get(rival)
                and displaces(personal[rival])):
            del personal[rival]
            self.cog.storage_manager._save_user_keys_data(self.user_id, user_data)
        if "personal" in scopes and personal.get(provider) != slot:
            personal[provider] = slot
            self.cog.storage_manager._save_user_keys_data(self.user_id, user_data)
        elif "personal" not in scopes and personal.get(provider) == slot:
            del personal[provider]
            self.cog.storage_manager._save_user_keys_data(self.user_id, user_data)

        mine = {"user_id": self.user_id, "slot": slot}
        for guild in self.admin_guilds:
            guild_id_str = str(guild.id)
            server_index = self.cog.server_manager._get_server_index(guild_id_str)
            assigned = server_index.get("assigned_keys", {}).get(provider)
            rival_assigned = server_index.get("assigned_keys", {}).get(rival) if rival else None
            if (guild_id_str in scopes and rival_assigned
                    and displaces(rival_assigned.get("slot"))):
                del server_index["assigned_keys"][rival]
                self.cog.server_manager._save_server_index(guild_id_str, server_index)
                self.cog.server_key_pointers.pop((guild.id, rival), None)
            if guild_id_str in scopes:
                # `/start` applies on every pick: an unchanged server is not rewritten.
                if assigned == mine and NO_KEY_NOTICE_FLAG not in server_index:
                    continue
                server_index.setdefault("assigned_keys", {})[provider] = dict(mine)
                # A server that loses this key later is told again, once.
                server_index.pop(NO_KEY_NOTICE_FLAG, None)
                self.cog.server_manager._save_server_index(guild_id_str, server_index)
                self.cog.server_key_pointers[(guild.id, provider)] = (self.user_id, slot)
            elif assigned and assigned.get("user_id") == self.user_id and assigned.get("slot") == slot:
                del server_index["assigned_keys"][provider]
                self.cog.server_manager._save_server_index(guild_id_str, server_index)
                self.cog.server_key_pointers.pop((guild.id, provider), None)


class SettingsAPIView(KeyScopeMixin, SettingsBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "api")
        self._init_scopes()
        #: Opened as the subscription's own screen (`/subscribe`, `/start`): no slot list, no tabs.
        self.dedicated = False
        #: One line over the card, once: the thanks after a purchase on this same message.
        self.notice: Optional[str] = None
        if (cog.user_settings.provider_choice(self.user_id) == PROVIDER_SUBSCRIPTION
                and (cog.subscriptions.sells or self._subscription_row())):
            self.show_subscription()  # what they chose in /start opens first
        else:
            self.setup_items()

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4 or self.dedicated: self.remove_item(item)

        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        slots_data = user_data.get("slots", {})

        # Row 0: Slot Selection, the subscription first where there is one
        slot_options = []
        sub, sells = self._subscription_row(), self.cog.subscriptions.sells
        if sub or sells:
            slot_options.append(discord.SelectOption(
                label=SLOT_LABELS[SUBSCRIPTION_SLOT], value=SUBSCRIPTION_SLOT, emoji="⭐",
                description=(f"{tier_name(sub.get('tier'))}, {sub.get('status', '')}" if sub
                             else "Not subscribed"),
                default=(self.selected_slot == SUBSCRIPTION_SLOT)))
        for slot_id, label, provider in KEY_SLOTS:
            data = slots_data.get(slot_id)
            if data:
                tier = data.get("tier", "free").title()
                desc = f"Set ({tier} Tier)"
                emoji = "🟢" if provider == "gemini" else "🟣"
            else:
                desc = "Empty"
                emoji = "⚪"
            slot_options.append(discord.SelectOption(label=label, value=slot_id, description=desc, emoji=emoji, default=(self.selected_slot == slot_id)))

        if not self.dedicated:
            add_select(self, slot_options, self.slot_select_callback,
                       placeholder="Select an API Key Slot...", row=0)

        if self.selected_slot == SUBSCRIPTION_SLOT:
            if sub:
                # Assigned like a key; there is nothing to paste or delete.
                self._add_scope_select(row=1)
                add_button(self, "Save Assignments", self.save_assignments_callback,
                           style=discord.ButtonStyle.success, row=2)
            # Anyone with an active row sees Manage, never a second Checkout -- paused or not.
            if self.cog.subscriptions.billing and (sub or {}).get("status") == "active":
                if (sub or {}).get("source") == "stripe" and defaultConfig.STRIPE_PORTAL_URL:
                    self.add_item(ui.Button(label="Manage ↗", url=defaultConfig.STRIPE_PORTAL_URL,
                                            row=2))
            elif sells:
                add_button(self, "Subscribe", self.subscribe_callback, emoji="⭐",
                           style=discord.ButtonStyle.primary, row=2)
        # Row 1: Scope Multi-Select
        elif self.selected_slot and self.selected_slot in slots_data:
            self._add_scope_select(row=1)

            # Row 2: Actions
            add_button(self, "Edit Key", self.edit_key_callback,
                       style=discord.ButtonStyle.primary, row=2)
            
            add_button(self, "Delete Key", self.delete_key_callback,
                       style=discord.ButtonStyle.danger, row=2)
            
            add_button(self, "Save Assignments", self.save_assignments_callback,
                       style=discord.ButtonStyle.success, row=2)
        elif self.selected_slot:
            add_button(self, "Submit Key", self.edit_key_callback,
                       style=discord.ButtonStyle.success, row=2)
        if self._offers_sign_in():
            # A link: Discord opens it, and no interaction reaches us.
            self.add_item(ui.Button(label="Sign in with OpenRouter ↗", row=2,
                                    url=self._sign_in_link(self.selected_slot)))

    def _offers_sign_in(self) -> bool:
        return (self.selected_slot in KEY_SLOT_IDS and openrouter_signin.enabled()
                and SLOT_PROVIDER[self.selected_slot] == "openrouter")

    def _subscription_row(self) -> Optional[dict]:
        """This user's subscription record, active or lapsed, where the subscription runs."""
        subs = self.cog.subscriptions
        return subs.row(self.user_id) if subs.enabled else None

    def show_subscription(self, dedicated: bool = False) -> None:
        """Opens on the subscription slot. `dedicated` (`/subscribe`, `/start`) makes it the
        subscription's own screen, with no slot list or tabs to read as API Key Management."""
        self.selected_slot = SUBSCRIPTION_SLOT
        self.dedicated = self.dedicated or dedicated
        self._load_scopes()
        self.setup_items()

    def _subscription_card(self) -> discord.Embed:
        """The subscription's card: tier, use as a share (never US$: the allowance reads as usage,
        not a balance), pace, and what happens at 100%. Its colour is its state."""
        sub = self._subscription_row() or {}
        paid, ends, renews = sub.get("source") == "stripe", sub.get("ends"), sub.get("renews")
        colour = discord.Colour.gold()
        if not sub:
            lines = ["**Not subscribed.** " + (
                "Your characters reply without an API key of your own, on models curated for "
                "roleplay. **⭐ Subscribe** shows the tiers." if self.cog.subscriptions.sells
                else SALES_PAUSED)]
        elif sub.get("status") == ACTIVE:
            head = f"**{tier_name(sub.get('tier'))}** · ✅ Active"
            if ends:
                head += f" · Ends <t:{ends}:D> (cancelled)"
                colour = discord.Colour.dark_gold()
            elif renews:
                head += f" · {'Renews' if paid else 'Allowance resets'} <t:{renews}:D>"
            lines = [head]
            if period_usage(sub)[1] > 0:
                share = used_share(sub)
                bar = f"{usage_bar(share)} {share:.0%} used"
                if sub.get("synced"):
                    bar += f" (as of <t:{int(sub['synced'])}:R>)"
                lines.append(bar)
                until = f" until <t:{renews}:D>" if renews and not ends else ""
                notes = []
                if share >= 1:
                    notes.append(f"**Used up:** your characters don't reply where it's assigned{until}.")
                elif share >= 0.8:
                    notes.append(f"At 100%, your characters stop replying where it's assigned{until}.")
                out = runs_out_at(sub)
                if out:
                    notes.append(f"Running fast: at this rate it runs out around <t:{out}:D>.")
                if carried_over(sub):
                    notes.append("Includes last month's unused allowance.")
                lines += [f"-# * {n}" for n in notes]
                if share >= 0.8:
                    colour = discord.Colour.orange()
        else:
            lines = ["**Lapsed.** Where it was assigned is kept, so **⭐ Subscribe** restores it."]
            colour = discord.Colour.greyple()
        if self.notice:
            lines.insert(0, self.notice + "\n")
        embed = discord.Embed(title="⭐ MimicAI Subscription", colour=colour, description="\n".join(lines))
        footer = ("Runs its curated models in place of a profile's own (Set Models → API: "
                  "Subscription), never in age-restricted channels. Web searches use the most.")
        if paid and sub.get("status") == ACTIVE and defaultConfig.STRIPE_PORTAL_URL:
            footer += (" Manage opens Stripe: sign in with the email you paid with to cancel, "
                       "switch tier or change card. MimicAI catches up within the hour.")
        return embed.set_footer(text=footer)

    async def update_display(self):
        await self.original_interaction.edit_original_response(content=None, embed=self.build_embed(),
                                                               view=self)

    def build_embed(self) -> discord.Embed:
        if self.selected_slot == SUBSCRIPTION_SLOT:
            embed = self._subscription_card()
        else:
            description = ("Your MimicAI Subscription and 4 API key slots. Assign each to your "
                           "Personal account or servers you run."
                           if self.cog.subscriptions.sells or self._subscription_row() else
                           "Manage your 4 API key slots and assign them to your Personal account or Servers you administrate.")
            embed = discord.Embed(title="API Key Management", description=description, color=discord.Color.blue())

        if self.selected_slot:
            user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
            slot_data = user_data.get("slots", {}).get(self.selected_slot)
            label = SLOT_LABELS[self.selected_slot]
            is_sub = self.selected_slot == SUBSCRIPTION_SLOT

            if slot_data or (is_sub and self._subscription_row()):
                provider = SLOT_PROVIDER[self.selected_slot]
                if not is_sub:
                    tier = slot_data.get("tier", "free").title()
                    value = f"**Status:** ✅ Set\n**Tier:** `{tier}`"
                    if provider == "gemini" and not is_paid_gemini_slot(slot_data):
                        # Assigning one to a server looks like it worked, and then nothing
                        # there can use it -- see cogs/utils/data_policy.
                        value += ("\n-# Google may train on what a free-tier key is sent, so it "
                                  "is not used in servers or Global Chat.")
                    embed.add_field(name=f"Slot: {label}", value=value, inline=False)
                
                saved = self._saved_scopes(provider)
                assignments = [f"- {a}" for a in self._scope_labels(saved)]
                assign_str = _fit_list(assignments, "\n", 1000) if assignments else "None"
                embed.add_field(name="Paying for" if is_sub else "Current Assignments",
                                value=assign_str, inline=False)

                # Ticks on another page are out of sight, and an unsaved tick is the
                # most common way a key ends up doing nothing, so say what Save will do.
                if self.selected_scopes != saved:
                    lines = []
                    for heading, scopes in (("Add", self.selected_scopes - saved),
                                            ("Remove", saved - self.selected_scopes)):
                        if scopes:
                            lines.append(f"**{heading}:** "
                                         + _fit_list(self._scope_labels(scopes), ", ", 400))
                    lines.append("-# Nothing changes until you press Save Assignments.")
                    embed.add_field(name="Unsaved Changes", value="\n".join(lines), inline=False)
            elif not is_sub:
                how = ("Click 'Submit Key' to paste a key, or 'Sign in with OpenRouter' to have "
                       "OpenRouter make one and send it here, with nothing to copy."
                       if self._offers_sign_in() else "Click 'Submit Key' to add a key to this slot.")
                embed.add_field(name=f"Slot: {label}", value=f"**Status:** ❌ Empty\n{how}", inline=False)
        else:
            embed.add_field(name="Overview", value="Select a slot from the dropdown above to view or manage it.", inline=False)
        return embed

    async def subscribe_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await SubscribeView(self).open(interaction)

    async def slot_select_callback(self, interaction: discord.Interaction):
        self.selected_slot = interaction.data['values'][0]
        self.scope_page = 0
        self._load_scopes()
        self.setup_items()
        await interaction.response.defer()
        await self.update_display()

    async def edit_key_callback(self, interaction: discord.Interaction):
        modal = SubmitAPIKeyModal(self.cog, self.selected_slot, SLOT_PROVIDER[self.selected_slot], view=self)
        await interaction.response.send_modal(modal)

    async def delete_key_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        provider = SLOT_PROVIDER[self.selected_slot]
        
        if self.selected_slot in user_data.get("slots", {}):
            del user_data["slots"][self.selected_slot]
            
        if user_data.get("personal_assignments", {}).get(provider) == self.selected_slot:
            del user_data["personal_assignments"][provider]
            
        self.cog.storage_manager._save_user_keys_data(self.user_id, user_data)
        self.cog.decrypted_key_cache.pop((self.user_id, self.selected_slot), None)
        
        for guild in self.cog.bot.guilds:
            guild_id_str = str(guild.id)
            server_index = self.cog.server_manager._get_server_index(guild_id_str)
            assigned = server_index.get("assigned_keys", {}).get(provider)
            if assigned and assigned.get("user_id") == self.user_id and assigned.get("slot") == self.selected_slot:
                del server_index["assigned_keys"][provider]
                self.cog.server_manager._save_server_index(guild_id_str, server_index)
                self.cog.server_key_pointers.pop((guild.id, provider), None)

        self._load_scopes()
        self.setup_items()
        await self.update_display()
        await interaction.followup.send(embed=text_embed("✅ Key and all its assignments deleted."), ephemeral=True)

    async def save_assignments_callback(self, interaction: discord.Interaction):
        conflicts = self._assignment_conflicts(self.selected_slot, self.selected_scopes)
        if conflicts:
            await interaction.response.send_message(
                embed=text_embed(OverrideConfirmView.prompt(conflicts)),
                view=OverrideConfirmView(self, self.selected_slot, self.selected_scopes),
                ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        self._apply_assignments(self.selected_slot, self.selected_scopes)
        await self._repaint()
        await interaction.followup.send(embed=text_embed("✅ Assignments saved successfully."), ephemeral=True)

def usage_bar(share: float, width: int = 10) -> str:
    """`▰▰▰▱▱▱▱▱▱▱` for 0.3: a period's use at a glance."""
    filled = round(min(1.0, max(0.0, share)) * width)
    return "▰" * filled + "▱" * (width - filled)


#: The Subscribe screen's account of what draws on the allowance: every call where it is
#: assigned (tests/MIMIC_SUBSCRIPTION.md, Open questions 6), not replies alone.
ALLOWANCE_COVERS = (
    "-# * The memories your characters write (LTMs)\n"
    "-# * Long-chat summaries (compaction)\n"
    "-# * Web searches *(uses more allowance)*\n"
    "-# * Reading attachments and recalling memories\n"
    "-# * A session's critic or proactivity, if on.\n"
    "-# ***Assigned to a server, it pays for everyone's chats there.***\n"
    "-# ***Not included: image gen, text-to-speech, Adult-rated profiles and use in age-restricted channels.***\n")

STRIPE_DOWN = "Stripe didn't answer just now, so nothing changed. Try again in a minute."
#: The owner paused new sales from `/mod` (`SubscriptionStore.paused`).
SALES_PAUSED = "New subscriptions are paused for now. Check back soon."


class SubscribeView(BlockedGuard, ui.View):
    """Subscribe, from the subscription slot: the tiers at Stripe's prices, and under them Pay and
    Check, greyed until a tier is picked. `_watch` notices the payment by itself; Check, which
    reconciles at once, is the backup. Stripe is the seller (Managed Payments)."""

    def __init__(self, parent: SettingsAPIView):
        super().__init__(timeout=900)
        self.parent, self.cog, self.user_id = parent, parent.cog, parent.user_id
        self.prices: dict = {}
        self._watcher: Optional[asyncio.Task] = None
        # (tier, Checkout Session) behind the Pay button, once a tier is picked.
        self._checkout: Optional[tuple] = None

    async def open(self, interaction: discord.Interaction) -> None:
        subs = self.cog.subscriptions
        # Paid a minute ago and never pressed Check? A second Checkout would bill twice.
        problem = await subs.reconcile(self.user_id)
        if not problem and subs.active(self.user_id):
            await self._subscribed(interaction)
            return
        if subs.paused:
            await interaction.edit_original_response(embed=text_embed(SALES_PAUSED))
            return
        try:
            self.prices = await stripe.prices(SUBSCRIPTION_TIERS)
        except StripeError as e:
            print(f"Subscribe: {e}")
        if problem or not self.prices:
            await interaction.edit_original_response(embed=text_embed(
                STRIPE_DOWN if problem or not stripe.enabled() else
                "The subscription isn't on sale yet."))
            return
        await interaction.edit_original_response(embed=self._show_tiers(), view=self)

    def _show_tiers(self) -> discord.Embed:
        """The tiers, with Pay and Check live once `_checkout` holds a pick."""
        self.clear_items()
        picked, session = self._checkout or (None, None)
        lines, options = [], []
        for tier, allowance in SUBSCRIPTION_TIERS.items():
            price = self.prices.get(tier)
            if not price:
                continue
            # Managed Payments adds tax on top unless the Price says it is included.
            per = f" a {(price.get('recurring') or {}).get('interval', 'month')}"
            cost = (f"{price['currency'].upper()} {price['unit_amount'] / 100:.2f}".removesuffix(".00")
                    + ("" if price.get("tax_behavior") == "inclusive" else " + tax") + per)
            count = round(allowance * SUBSCRIPTION_REPLIES_PER_USD, -2)
            replies, daily = f"~{count:,.0f} replies", f"about {round(count / 30, -1):,.0f} a day"
            lines.append(f"**{tier_name(tier)}** · {replies} · {daily} · {cost}")
            options.append(discord.SelectOption(label=f"{tier_name(tier)}: {replies}{per}", value=tier,
                                                description=f"{cost} · {daily}"[:100],
                                                default=tier == picked))
        add_select(self, options, self._pick, placeholder="Choose a tier...", row=0)
        if session:
            self.add_item(ui.Button(label=f"Pay for {tier_name(picked)} on Stripe ↗",
                                    url=session["url"], row=1))
        else:
            self.add_item(ui.Button(label="Pay on Stripe ↗", disabled=True, row=1))
        add_button(self, "Check", self._check, disabled=not session, row=1)
        description = "Your characters reply without an API key of your own."
        if session:
            description += ("\n\n✅ Choose a tier → **② Pay on Stripe** → ③ Done\n-# This updates by "
                            "itself once you've paid; if it doesn't, press **Check**.")
        embed = discord.Embed(title="⭐ MimicAI Subscription", colour=discord.Colour.gold(),
                              description=description)
        embed.add_field(name="Tiers", inline=False, value="\n".join(lines) + (
            "\n-# Rough counts: long chats, extra features, and models may use more. Measured on a simulated chat of 100 replies "
            "(50 messages of memory, a rolling summary) on DeepSeek V4.1 Flash, then trimmed for "
            "memories and searches.\n"))
        embed.add_field(name="Models · curated for casual talk and roleplay", inline=False,
                        value=self._models_line() or "-# * Curated for high performance in roleplay.")
        embed.add_field(name="Your allowance also covers", value=ALLOWANCE_COVERS, inline=False)
        embed.set_footer(text="18+ · Paid securely through Stripe: checkout and receipts say \"Link\" · "
                              "Cancel anytime and keep it to the end of the month · Unused replies "
                              "carry over, up to a month's worth · Paying accepts the Terms.")
        return embed

    def _models_line(self) -> str:
        """The models the bot owner curates, one bullet each: what a subscriber gets, kept true
        by a swap."""
        catalogue = self.cog.api_service.catalogue
        return "\n".join(f"-# * {catalogue.label(m.removeprefix('OPENROUTER/'), drop_author=True)}"
                         for m in subscription_curated(self.cog, "primary_model"))

    async def _pick(self, interaction: discord.Interaction):
        tier = interaction.data["values"][0]
        await interaction.response.defer()
        if self.cog.subscriptions.paused:
            # Paused while this was open: no new Checkout, but one already made stays payable.
            await interaction.followup.send(embed=text_embed(SALES_PAUSED), ephemeral=True)
            return
        if self._checkout:
            # A new pick ends the last one's Checkout: a tab still open on it would bill twice.
            try:
                await stripe.expire_checkout(self._checkout[1]["id"])
            except StripeError:
                await self._confirm(interaction)  # paid already, or Stripe is down: keep it
                if not self.is_finished():
                    await interaction.edit_original_response(embed=self._show_tiers(), view=self)
                return
            self._watcher.cancel()
            self._checkout = None
        customer = ((self.cog.subscriptions.row(self.user_id) or {}).get("ref") or {}).get("customer")
        try:
            session = await stripe.checkout(self.prices[tier]["id"], str(self.user_id), customer)
        except StripeError as e:
            print(f"Subscribe: no Checkout for {self.user_id}: {e}")
            await interaction.edit_original_response(embed=self._show_tiers(), view=self)
            await interaction.followup.send(embed=text_embed(STRIPE_DOWN), ephemeral=True)
            return
        self._checkout = (tier, session)
        await interaction.edit_original_response(embed=self._show_tiers(), view=self)
        self._watcher = asyncio.create_task(self._watch(interaction, session["id"]))

    async def _watch(self, interaction: discord.Interaction, session_id: str) -> None:
        """Polls the Checkout Session until it is paid, then reconciles. Stops short of the
        interaction token's 15 minutes; past that, the hourly reconcile or `/subscribe` catches it."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 840
        while not self.is_finished() and loop.time() < deadline:
            await asyncio.sleep(5)
            try:
                status = await stripe.checkout_status(session_id)
            except StripeError:
                continue  # Check is still there
            if status == "expired":
                return
            if status == "complete":
                subs = self.cog.subscriptions
                if not await subs.reconcile(self.user_id) and subs.active(self.user_id):
                    try:
                        await self._subscribed(interaction)
                    except discord.HTTPException:
                        pass
                    return

    async def _check(self, interaction: discord.Interaction):
        await interaction.response.defer()
        await self._confirm(interaction)

    async def _confirm(self, interaction: discord.Interaction):
        subs = self.cog.subscriptions
        problem = await subs.reconcile(self.user_id)
        if subs.active(self.user_id):
            await self._subscribed(interaction)
            return
        await interaction.followup.send(embed=text_embed(STRIPE_DOWN if problem else (
            "Stripe hasn't confirmed a payment yet. If you've paid, wait a moment and press "
            "**Check** again.")), ephemeral=True)

    async def _subscribed(self, interaction: discord.Interaction) -> None:
        if self.is_finished():
            return  # `_watch` and Check both got there
        self.stop()
        p = self.parent
        guild = interaction.guild
        # It pays where it was bought: for you, and the server you bought it in if you run it.
        # A scope another key holds is asked about, never taken (OverrideConfirmView).
        want = {"personal"} | ({str(guild.id)} if guild and any(
            g.id == guild.id for g in p.admin_guilds) else set())
        held = {s for s in want if p._assignment_conflicts(SUBSCRIPTION_SLOT, {s})}
        saved = p._saved_scopes("openrouter", SUBSCRIPTION_SLOT)
        if want - held:
            p._apply_assignments(SUBSCRIPTION_SLOT, saved | (want - held))
        where = " and ".join(f"**{l.removeprefix('Server: ')}**"
                             for l in p._scope_labels(want - held))
        thanks = ("✅ **You're subscribed. Thank you!**"
                  + (f" It's now assigned to {where}." if where else ""))
        dedicated = getattr(p, "dedicated", False)
        if dedicated:
            # `/subscribe`'s own message: it becomes the card, carrying the thanks this once
            # (a repaint through `p` would paint over them).
            p.notice = thanks
            p._load_scopes()
            p.setup_items()
            await interaction.edit_original_response(embed=p.build_embed(), view=p)
            p.notice = None
        else:
            await interaction.edit_original_response(view=None, embed=text_embed(
                thanks + "\n-# `/subscribe` changes where it applies."))
        if held:
            await interaction.followup.send(
                embed=text_embed(OverrideConfirmView.prompt(
                    p._assignment_conflicts(SUBSCRIPTION_SLOT, held))),
                view=OverrideConfirmView(p, SUBSCRIPTION_SLOT, saved | want), ephemeral=True)
        if dedicated:
            return
        p._load_scopes()
        try:
            await p._repaint()
        except discord.HTTPException:
            pass  # that screen's 15 minutes are up; `/settings` shows the same


class SettingsChildBotView(SettingsBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "bots")
        self.selected_bot_id = None
        self.setup_items()

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        if self.user_id != int(defaultConfig.DISCORD_OWNER_ID):
            return

        # Row 0: Select Bot
        options = []
        user_bot_items = [(bid, b) for bid, b in self.cog.child_bots.items() if b['owner_id'] == self.user_id]
        
        for bid, b_data in user_bot_items:
            bot_user = self.cog.bot.get_user(int(bid))
            name = bot_user.name if bot_user else f"ID: {bid}"
            options.append(discord.SelectOption(label=f"{name} ({b_data.get('profile_name')})", value=bid, default=(bid == self.selected_bot_id)))

        if options:
            add_select(self, options[:25], self.select_bot, placeholder="Select a child bot...",
                       row=0)

        # Row 1: Actions
        add_button(self, "Create New Child Bot", self.create_bot, style=discord.ButtonStyle.green,
                   row=1)

        if self.selected_bot_id:
            # [REMOVED] Manage Approved Servers button
            
            add_button(self, "Unlink & Delete", self.delete_bot, style=discord.ButtonStyle.danger,
                       row=1)

    async def update_display(self):
        if self.user_id != int(defaultConfig.DISCORD_OWNER_ID):
            embed = discord.Embed(title="Child Bots", description="Only the bot owner can create Child Bots.", color=discord.Color.red())
            await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)
            return

        embed = discord.Embed(title="My Child Bots", description="Manage your linked bot applications.", color=discord.Color.dark_magenta())
        if self.selected_bot_id:
            bot_user = self.cog.bot.get_user(int(self.selected_bot_id))
            name = bot_user.name if bot_user else self.selected_bot_id
            b_data = self.cog.child_bots.get(self.selected_bot_id)
            p_name = b_data.get('profile_name')
            embed.add_field(name="Selected Bot", value=f"**Name:** `{name}`\n**Linked Profile:** `{p_name}`", inline=False)
        else:
            embed.add_field(name="Overview", value="Select a bot from the dropdown to manage it, or create a new one.\n*(Note: You will need the **PID** of the profile you want to link, which can be found in `/profile manage`.)*", inline=False)
        
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def select_bot(self, i: discord.Interaction):
        self.selected_bot_id = i.data['values'][0]
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def create_bot(self, i: discord.Interaction):
        self._build_child_bot_list_ui = lambda x: self.update_rebuild(x) 
        modal = ChildBotCreateModal(self.cog, self)
        await i.response.send_modal(modal)

    async def update_rebuild(self, i: discord.Interaction):
        # Callback for the modal to refresh UI
        self.setup_items()
        await self.update_display()

    async def delete_bot(self, i: discord.Interaction):
        bot_to_delete = self.cog.child_bots.get(self.selected_bot_id)
        if bot_to_delete:
            owner_id = bot_to_delete['owner_id']
            profile_name = bot_to_delete['profile_name']
            
            self.cog.profile_manager._delete_child_bot_config(owner_id, profile_name)
            await asyncio.to_thread(self.cog.child_bot_manager._load_child_bots)
            await self.cog.manager_queue.put({"action": "shutdown_bot", "bot_id": self.selected_bot_id})
        
        self.selected_bot_id = None
        self.setup_items()
        await self.update_display()
        await i.response.send_message(embed=text_embed("Bot deleted."), ephemeral=True)

class ChildBotCreateModal(ui.Modal, title="Create a New Child Bot"):
    def __init__(self, cog: 'MimicCog', view: 'SettingsChildBotView'):
        super().__init__()
        self.cog = cog
        self.parent_view = view
        self.profile_id_input = ui.TextInput(label="Profile ID (PID)", placeholder="e.g. A1B2C3D4E5F6789 or X1B2C3D4E5F6789", required=True, min_length=16, max_length=16)
        self.token_input = ui.TextInput(
            label="Bot Token", 
            placeholder="Applications -> Bot -> Token", 
            style=discord.TextStyle.paragraph, 
            required=True
        )
        self.add_item(self.profile_id_input)
        self.add_item(self.token_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        if interaction.user.id != int(defaultConfig.DISCORD_OWNER_ID):
            await interaction.followup.send(embed=text_embed("Error: Only the bot owner can create Child Bots."), ephemeral=True)
            return
            
        token = self.token_input.value.strip()
        pid = self.profile_id_input.value.strip().upper()
        owner_id = interaction.user.id

        if not (pid.startswith("A") or pid.startswith("X")):
            await interaction.followup.send(embed=text_embed("Error: Child bots can only be linked to Personal Profiles (PIDs starting with 'A') or System Profiles (PIDs starting with 'X'). Borrowed profiles (PIDs starting with 'B' or 'C') cannot be linked -- duplicate one first to get your own Personal Profile."), ephemeral=True)
            return

        index = self.cog.profile_manager._get_user_index(owner_id)
        profile_name = None
        for cat in ["personal", "system"]:
            if isinstance(index.get(cat), dict):
                for name, mapped_pid in index[cat].items():
                    if mapped_pid == pid:
                        profile_name = name
                        break
            if profile_name: break

        if not profile_name:
            await interaction.followup.send(embed=text_embed(f"Error: You do not own a profile with the PID '{pid}'."), ephemeral=True)
            return

        temp_client = discord.Client(intents=discord.Intents.none())
        try:
            await temp_client.login(token)
            bot_user_id = str(temp_client.user.id)
            await temp_client.close()
        except discord.LoginFailure:
            await interaction.followup.send(embed=text_embed("Error: The provided token is invalid. Please double-check it."), ephemeral=True)
            return
        except Exception as e:
            await temp_client.close()
            await interaction.followup.send(embed=text_embed(f"Error: An unexpected error occurred while validating the token: {suppress_link_previews(str(e))}"), ephemeral=True)
            return

        if bot_user_id in self.cog.child_bots:
            await interaction.followup.send(embed=text_embed("Error: This bot application is already registered as a child bot."), ephemeral=True)
            return

        try:
            encrypted_token = self.cog.fernet.encrypt(token.encode()).decode()
        except Exception:
            encrypted_token = token
        
        bot_config = {
            "token_encrypted": encrypted_token,
            "approved_servers": [],
            "bot_id": bot_user_id
        }
        
        self.cog.profile_manager._set_child_bot_config(owner_id, profile_name, bot_config)
        await asyncio.to_thread(self.cog.child_bot_manager._load_child_bots)

        new_bot_config = self.cog.child_bots.get(bot_user_id)
        if new_bot_config:
            await self.cog.manager_queue.put({
                "action": "launch_bot",
                "bot_id": bot_user_id,
                "token": token,
                "config": new_bot_config
            })
        
        invite = invite_url(bot_user_id, CHILD_BOT_PERMISSIONS)
        await interaction.followup.send(embed=text_embed(f"Success! Child bot '{temp_client.user.name}' has been linked to profile '{profile_name}'.\n[Invite it to a server](<{invite}>)"), ephemeral=True)
        await self.parent_view.update_rebuild(interaction)

class ParentActivityModal(ui.Modal):
    def __init__(self, cog, act_type):
        super().__init__(title="Set Activity Details")
        self.cog = cog
        self.act_type = act_type
        
        self.text_input = ui.TextInput(label="Activity Text", placeholder="e.g. the conversation", required=True, max_length=128)
        self.add_item(self.text_input)
        
        if act_type == "streaming":
            self.url_input = ui.TextInput(label="Twitch/YouTube URL", placeholder="https://twitch.tv/example", required=True)
            self.add_item(self.url_input)

    async def on_submit(self, interaction: discord.Interaction):
        text = self.text_input.value.strip()
        url = getattr(self, "url_input", None)
        url_val = url.value.strip() if url else None
        
        presence = self.cog.server_manager._load_parent_presence()
        presence["activity_type"] = self.act_type
        presence["activity_text"] = text
        presence["activity_url"] = url_val
        self.cog.server_manager._save_parent_presence(presence)
        
        status_val = presence.get("status", "online")
        status_map = {"online": discord.Status.online, "idle": discord.Status.idle, "dnd": discord.Status.dnd, "invisible": discord.Status.invisible}
        
        activity = self.cog.server_manager._build_activity_from_dict(presence)
        await self.cog.bot.change_presence(status=status_map.get(status_val, discord.Status.online), activity=activity)
        
        await interaction.response.send_message(embed=text_embed(f"Activity set to **{self.act_type.title()} {text}**."), ephemeral=True)

class ParentPresenceView(BlockedGuard, ui.View):
    def __init__(self, cog):
        super().__init__(timeout=300)
        self.cog = cog
        
        status_options =[
            discord.SelectOption(label="Online", value="online", emoji="🟢"),
            discord.SelectOption(label="Idle", value="idle", emoji="🌙"),
            discord.SelectOption(label="Do Not Disturb", value="dnd", emoji="⛔"),
            discord.SelectOption(label="Invisible", value="invisible", emoji="🔘")
        ]
        self.status_select = ui.Select(placeholder="Change Online Status...", options=status_options, row=0)
        self.status_select.callback = self.status_callback
        self.add_item(self.status_select)

        activity_options =[
            discord.SelectOption(label="Playing...", value="playing", emoji="🎮"),
            discord.SelectOption(label="Watching...", value="watching", emoji="📺"),
            discord.SelectOption(label="Listening to...", value="listening", emoji="🎧"),
            discord.SelectOption(label="Competing in...", value="competing", emoji="🏆"),
            discord.SelectOption(label="Streaming...", value="streaming", emoji="🟪")
        ]
        self.activity_select = ui.Select(placeholder="Set Activity Type...", options=activity_options, row=1)
        self.activity_select.callback = self.activity_callback
        self.add_item(self.activity_select)

        add_button(self, "Clear Activity", self.clear_callback, style=discord.ButtonStyle.danger,
                   row=2)

    async def status_callback(self, interaction: discord.Interaction):
        status_map = {
            "online": discord.Status.online, "idle": discord.Status.idle,
            "dnd": discord.Status.dnd, "invisible": discord.Status.invisible
        }
        status_val = self.status_select.values[0]
        
        presence = self.cog.server_manager._load_parent_presence()
        presence["status"] = status_val
        self.cog.server_manager._save_parent_presence(presence)
        
        activity = self.cog.server_manager._build_activity_from_dict(presence)
        await self.cog.bot.change_presence(status=status_map[status_val], activity=activity)
        await interaction.response.send_message(embed=text_embed(f"Status changed to **{status_val.title()}**."), ephemeral=True)

    async def activity_callback(self, interaction: discord.Interaction):
        act_type = self.activity_select.values[0]
        await interaction.response.send_modal(ParentActivityModal(self.cog, act_type))

    async def clear_callback(self, interaction: discord.Interaction):
        presence = self.cog.server_manager._load_parent_presence()
        presence["activity_type"] = None
        presence["activity_text"] = None
        presence["activity_url"] = None
        self.cog.server_manager._save_parent_presence(presence)
        
        status_val = presence.get("status", "online")
        status_map = {"online": discord.Status.online, "idle": discord.Status.idle, "dnd": discord.Status.dnd, "invisible": discord.Status.invisible}
        await self.cog.bot.change_presence(status=status_map.get(status_val, discord.Status.online), activity=None)
        await interaction.response.send_message(embed=text_embed("Activity cleared."), ephemeral=True)

class ShutdownConfirmView(BlockedGuard, ui.View):
    def __init__(self, cog: 'MimicCog'):
        super().__init__(timeout=60)
        self.cog = cog

    @ui.button(label="Yes, Shutdown", style=discord.ButtonStyle.danger)
    async def confirm_shutdown(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.edit_message(content=None, embed=text_embed("Shutting down child bots and main instance..."), view=None)
        
        # 1. Close child processes first
        for bot_id in list(self.cog.child_bots.keys()):
            await self.cog.manager_queue.put({"action": "shutdown_bot", "bot_id": bot_id})
        
        await asyncio.sleep(2)

        # 2. Flush all in-memory sessions to disk
        self.cog.dirty_sessions.clear()
        for session_key, session_data in self.cog.global_chat_sessions.items():
            await self.cog.session_manager._save_session_to_disk(session_key, 'global_chat', session_data)
        
        for ch_id, session_data in self.cog.multi_profile_channels.items():
            if session_data.get("is_hydrated"):
                session_type = session_data.get("type", "multi")
                unified_log = session_data.get("unified_log")
                if unified_log is not None:
                    dummy_session_key = (ch_id, None, None)
                    await self.cog.session_manager._save_session_to_disk(dummy_session_key, session_type, unified_log)

        # 3. Force stop all loops and close
        if self.cog.has_lock:
            try:
                if os.path.exists(COG_LOCK_FILE_PATH):
                    os.remove(COG_LOCK_FILE_PATH)
            except: pass
        
        await self.cog.bot.change_presence(status=discord.Status.offline)
        await self.cog.bot.close()

    async def on_timeout(self):
        pass
