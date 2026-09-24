from ..utils.constants import *

import discord
from discord import ui
import asyncio
import datetime
from typing import TYPE_CHECKING, List, Optional

from .base_components import (
    BlockedGuard, PageJumpModal, TabbedView, add_button, add_select, paged_nav_options,
)
from ..utils.birthdays import MONTH_NAMES, format_birthday, parse_birthday, valid_birthday
from ..utils.helpers import _get_user_hash, _resolve_zoneinfo, suppress_link_previews
from ..utils.data_policy import is_paid_gemini_slot, may_save_free_gemini_key

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
        if not self.parent_view.cog.profile_manager.may_use_ollama(
                getattr(self.parent_view, "user_id", interaction.user.id)):
            await interaction.response.send_message(OLLAMA_OWNER_ONLY, ephemeral=True)
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
        # `/start` pastes a key to make it work, and a key nothing points at does
        # nothing; `/settings` leaves where it goes to the dropdown.
        self.assign_personal = assign_personal

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        raw_key = self.key_input.value.strip()
        
        if not raw_key.startswith(("AIzaSy", "AQ.", "sk-or-")):
            await interaction.followup.send("❌ **Invalid Format.** Keys must start with `AIzaSy`, `AQ.`, or `sk-or-`.", ephemeral=True)
            return

        is_valid, err, tier = await self.cog.api_service._validate_api_keys(
            raw_key if self.provider == "gemini" else None, 
            raw_key if self.provider == "openrouter" else None
        )
        
        if not is_valid:
            await interaction.followup.send(f"❌ **Validation Failed:** {suppress_link_previews(str(err))}", ephemeral=True)
            return

        # Refused before anything is written -- see may_save_free_gemini_key.
        if (self.provider == "gemini" and not is_paid_gemini_slot({"tier": tier})
                and not may_save_free_gemini_key(interaction.user.id)):
            await interaction.followup.send(f"❌ {GEMINI_FREE_TIER_KEY_REFUSED}", ephemeral=True)
            return

        user_data = self.cog.storage_manager._get_user_keys_data(interaction.user.id)
        user_data.setdefault("slots", {})[self.slot_id] = {
            "key": raw_key,
            "provider": self.provider,
            "tier": tier
        }
        if self.assign_personal:
            user_data.setdefault("personal_assignments", {})[self.provider] = self.slot_id
        self.cog.storage_manager._save_user_keys_data(interaction.user.id, user_data)
        
        self.cog.decrypted_key_cache[(interaction.user.id, self.slot_id)] = raw_key

        # has_personal_key is set by _save_user_keys_data above, together with the
        # stat stamp _index_is_consistent checks it against. Setting the flag here
        # without the stamp is what this used to do, and it bought a full index
        # repair on the next hourly pass.

        msg = f"✅ {self.provider.title()} key saved to slot `{self.slot_id}` ({tier.title()} Tier)."
        if self.assign_personal:
            msg += "\nIt is now your **Personal** key."
        if self.provider == "gemini" and not is_paid_gemini_slot({"tier": tier}):
            msg += "\nGoogle may train on what a free-tier key is sent, so it is not used in servers or Global Chat."

        if self.view:
            await self.view._repaint()

        await interaction.followup.send(msg, ephemeral=True)

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
        await interaction.edit_original_response(content="✅ Assignments saved successfully.", view=None)

    @ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel_override(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.edit_message(content="❌ Assignment update cancelled.", view=None)

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
        ("Defaults", "defaults", _defaults_view),
        ("Child Bots", "bots", lambda v: SettingsChildBotView(v.cog, v.original_interaction)),
    )

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
        
        if self.user_id == int(defaultConfig.DISCORD_OWNER_ID):
            child_bots = [b for b in self.cog.child_bots.values() if b['owner_id'] == self.user_id]
            bot_text = f"You own **{len(child_bots)}** Child Bots." if child_bots else "You do not own any Child Bots."
        else:
            bot_text = "Child Bots are restricted to the bot owner."
        
        primary_count = 0
        for g in self.cog.bot.guilds:
            idx = self.cog.server_manager._get_server_index(str(g.id))
            for assigned in idx.get("assigned_keys", {}).values():
                if assigned.get("user_id") == self.user_id:
                    primary_count += 1
                    break

        embed = discord.Embed(title="MimicAI Control Panel", description="Manage your API keys and personal bots from one place.", color=discord.Color.dark_teal())
        embed.set_thumbnail(url=THINKING_THUMBNAIL_URL)
        
        embed.add_field(name="API Key Slots", value=f"**Google Gemini:** {stat_gemini}\n**OpenRouter:** {stat_or}", inline=True)
        embed.add_field(name="Child Bots", value=bot_text, inline=True)
        embed.add_field(name="Server Contributions", value=f"Active Assignments: `{primary_count} servers`", inline=False)
        
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

def _standing_text(cog: 'MimicCog', user_id: int) -> str:
    """Whether the user is restricted, since when, and until when.

    Never why, and never by whom: those are the operator's note. The Privacy Policy
    discloses that a reason is held, not that it is shown. Set membership decides, not
    the record, so an elapsed block reads as lifted before the sweep has pruned it.
    """
    if user_id not in cog.generation_blocked:
        return "No restrictions."
    entry = cog.server_manager.blacklist_entry(user_id) or {}
    lines = ["**Restricted.** Only `/privacy`, `/terms` and `/settings` are available to you."]
    try:
        since = int(datetime.datetime.fromisoformat(entry["at"]).timestamp())
        lines.append(f"Since <t:{since}:D>")
    except (KeyError, TypeError, ValueError):
        pass
    until = entry.get("until")
    lines.append(f"Lifts <t:{int(until)}:R>" if until else "No end date.")
    return "\n".join(lines)


#: Under the choice, in About Me and in `/start`'s step: what it does, and what it leaves.
PROVIDER_PREFERENCE_NOTE = (
    "Every model runs Primary \u2192 Fallback on this provider. A profile with its **Final "
    "Fallback** turned on (Params, off unless you turn it on) tries the other one last, so "
    "one provider's outage or a spent key cannot silence it. New profiles start on this "
    "provider; a model you picked yourself stays picked. "
    "Holding a key for only the other provider overrides it for new profiles.")


def provider_wording(provider: Optional[str]) -> str:
    return MODEL_PROVIDERS.get(provider or "", "Not chosen -- using Google")


def provider_options(current: Optional[str]) -> List[discord.SelectOption]:
    """The two providers, the stored one ticked. One list for About Me and `/start`."""
    blurbs = {"gemini": "Gemini through your Google key.",
              "openrouter": "The same Gemini models through your OpenRouter key."}
    return [discord.SelectOption(label=label, value=value, description=blurbs[value],
                                 default=value == current)
            for value, label in MODEL_PROVIDERS.items()]


def build_about_embed(cog: 'MimicCog', user_id: int) -> discord.Embed:
    """About Me, with the user's standing.

    The About tab renders this, and so does `/settings` for a blocked user in place of
    the tabs, with no view attached -- nothing on it is theirs to change while blocked.
    """
    about = cog.profile_manager.get_user_about(user_id)
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

    embed.add_field(
        name="\N{TWISTED RIGHTWARDS ARROWS} Your Preferred Provider",
        value=f"`{provider_wording(about.get('provider'))}`\n{PROVIDER_PREFERENCE_NOTE}",
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
    ProfileManager.get_user_about for why it is not an encrypted shard and not a key
    inside "defaults".
    """

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "about")
        self._build_view()

    def _build_view(self):
        self.clear_items()
        add_button(self, "Set Timezone", self._act_timezone, style=discord.ButtonStyle.primary,
                   row=0, emoji="\N{GLOBE WITH MERIDIANS}")

        about = self.cog.profile_manager.get_user_about(self.user_id)
        if about.get("timezone"):
            add_button(self, "Clear Timezone", self._act_clear_timezone,
                       style=discord.ButtonStyle.secondary, row=0)

        add_button(self, "Set Birthday", self._act_birthday, style=discord.ButtonStyle.primary,
                   row=1, emoji="\N{BIRTHDAY CAKE}")
        if about.get("birthday"):
            add_button(self, "Clear Birthday", self._act_clear_birthday,
                       style=discord.ButtonStyle.secondary, row=1)

        add_select(self, provider_options(about.get("provider")), self._act_provider,
                   placeholder="Preferred provider...", row=2)

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
        await i.response.send_message(content=view._get_header_content(),
                                      view=view, ephemeral=True)

    async def _act_clear_timezone(self, i: discord.Interaction):
        await i.response.defer()
        about = self.cog.profile_manager.get_user_about(self.user_id)
        about.pop("timezone", None)
        self.cog.profile_manager.save_user_about(self.user_id, about)
        self._build_view()
        await self.update_display()

    async def _act_provider(self, i: discord.Interaction):
        await i.response.defer()
        self.cog.profile_manager.set_provider_preference(self.user_id, i.data["values"][0])
        self._build_view()
        await self.update_display()

    async def _act_birthday(self, i: discord.Interaction):
        await i.response.send_modal(UserBirthdayModal(self.cog, self))

    async def _act_clear_birthday(self, i: discord.Interaction):
        await i.response.defer()
        about = self.cog.profile_manager.get_user_about(self.user_id)
        about.pop("birthday", None)
        self.cog.profile_manager.save_user_about(self.user_id, about)
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
            cog.profile_manager.get_user_about(parent_about_view.user_id).get("birthday")) or {}
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
                f"❌ **Invalid Input:** {suppress_link_previews(str(e))}", ephemeral=True)
            return
        user_id = self.parent_about_view.user_id
        about = self.cog.profile_manager.get_user_about(user_id)
        if birthday:
            about["birthday"] = birthday
        else:
            about.pop("birthday", None)
        self.cog.profile_manager.save_user_about(user_id, about)

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


#: The four key slots, (slot id, label, provider): two per provider.
KEY_SLOTS = (
    ("google_key_1", "Google Gemini Key 1", "gemini"),
    ("google_key_2", "Google Gemini Key 2", "gemini"),
    ("openrouter_key_1", "OpenRouter Key 1", "openrouter"),
    ("openrouter_key_2", "OpenRouter Key 2", "openrouter"),
)
SLOT_PROVIDER = {slot: provider for slot, _label, provider in KEY_SLOTS}


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

    def _saved_scopes(self, provider: str) -> set:
        """Where the selected slot is assigned now, as dropdown values."""
        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        scopes = set()
        if user_data.get("personal_assignments", {}).get(provider) == self.selected_slot:
            scopes.add("personal")
        for g in self.admin_guilds:
            idx = self.cog.server_manager._get_server_index(str(g.id))
            assigned = idx.get("assigned_keys", {}).get(provider)
            if assigned and assigned.get("user_id") == self.user_id and assigned.get("slot") == self.selected_slot:
                scopes.add(str(g.id))
        return scopes

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

    def _assignment_conflicts(self, slot: str, scopes) -> List[str]:
        """Scopes in `scopes` another key or user holds now, in the confirm prompt's words."""
        provider = SLOT_PROVIDER[slot]
        conflicts = []
        if "personal" in scopes:
            user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
            curr_personal = user_data.get("personal_assignments", {}).get(provider)
            if curr_personal and curr_personal != slot:
                conflicts.append(f"Personal Scope (Currently uses {curr_personal})")
        for g in self.admin_guilds:
            if str(g.id) not in scopes:
                continue
            assigned = self.cog.server_manager._get_server_index(str(g.id)).get(
                "assigned_keys", {}).get(provider)
            if assigned and (assigned.get("user_id") != self.user_id or assigned.get("slot") != slot):
                conflicts.append(f"Server: {g.name} (Currently assigned by another key/user)")
        return conflicts

    def _apply_assignments(self, slot: str, scopes):
        """Makes `slot` apply exactly to `scopes`, among the scopes this view listed.

        Only servers you administer are touched. The override prompt used to walk every
        server the bot is in, so confirming one takeover also dropped this key from a
        server you had since stopped administering, which no dropdown could show you.
        """
        provider = SLOT_PROVIDER[slot]
        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        personal = user_data.setdefault("personal_assignments", {})
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
        self.setup_items()

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        slots_data = user_data.get("slots", {})

        # Row 0: Slot Selection
        slot_options = []
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

        add_select(self, slot_options, self.slot_select_callback,
                   placeholder="Select an API Key Slot...", row=0)

        # Row 1: Scope Multi-Select
        if self.selected_slot and self.selected_slot in slots_data:
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

    async def update_display(self):
        embed = discord.Embed(title="API Key Management", description="Manage your 4 API key slots and assign them to your Personal account or Servers you administrate.", color=discord.Color.blue())
        
        if self.selected_slot:
            user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
            slot_data = user_data.get("slots", {}).get(self.selected_slot)
            label = next(l for s, l, p in KEY_SLOTS if s == self.selected_slot)
            
            if slot_data:
                tier = slot_data.get("tier", "free").title()
                provider = SLOT_PROVIDER[self.selected_slot]
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
                embed.add_field(name="Current Assignments", value=assign_str, inline=False)

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
            else:
                embed.add_field(name=f"Slot: {label}", value="**Status:** ❌ Empty\nClick 'Submit Key' to add a key to this slot.", inline=False)
        else:
            embed.add_field(name="Overview", value="Select a slot from the dropdown above to view or manage it.", inline=False)

        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

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
        await interaction.followup.send("✅ Key and all its assignments deleted.", ephemeral=True)

    async def save_assignments_callback(self, interaction: discord.Interaction):
        conflicts = self._assignment_conflicts(self.selected_slot, self.selected_scopes)
        if conflicts:
            await interaction.response.send_message(
                OverrideConfirmView.prompt(conflicts),
                view=OverrideConfirmView(self, self.selected_slot, self.selected_scopes),
                ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        self._apply_assignments(self.selected_slot, self.selected_scopes)
        await self._repaint()
        await interaction.followup.send("✅ Assignments saved successfully.", ephemeral=True)

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
        await i.response.send_message("Bot deleted.", ephemeral=True)

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
            await interaction.followup.send("Error: Only the bot owner can create Child Bots.", ephemeral=True)
            return
            
        token = self.token_input.value.strip()
        pid = self.profile_id_input.value.strip().upper()
        owner_id = interaction.user.id

        if not (pid.startswith("A") or pid.startswith("X")):
            await interaction.followup.send("Error: Child bots can only be linked to Personal Profiles (PIDs starting with 'A') or System Profiles (PIDs starting with 'X'). Borrowed profiles (PIDs starting with 'B' or 'C') cannot be linked -- duplicate one first to get your own Personal Profile.", ephemeral=True)
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
            await interaction.followup.send(f"Error: You do not own a profile with the PID '{pid}'.", ephemeral=True)
            return

        temp_client = discord.Client(intents=discord.Intents.none())
        try:
            await temp_client.login(token)
            bot_user_id = str(temp_client.user.id)
            await temp_client.close()
        except discord.LoginFailure:
            await interaction.followup.send("Error: The provided token is invalid. Please double-check it.", ephemeral=True)
            return
        except Exception as e:
            await temp_client.close()
            await interaction.followup.send(f"Error: An unexpected error occurred while validating the token: {suppress_link_previews(str(e))}", ephemeral=True)
            return

        if bot_user_id in self.cog.child_bots:
            await interaction.followup.send("Error: This bot application is already registered as a child bot.", ephemeral=True)
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
        
        await interaction.followup.send(f"Success! Child bot '{temp_client.user.name}' has been linked to profile '{profile_name}'.", ephemeral=True)
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
        
        await interaction.response.send_message(f"Activity set to **{self.act_type.title()} {text}**.", ephemeral=True)

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
        await interaction.response.send_message(f"Status changed to **{status_val.title()}**.", ephemeral=True)

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
        await interaction.response.send_message("Activity cleared.", ephemeral=True)

class ShutdownConfirmView(BlockedGuard, ui.View):
    def __init__(self, cog: 'MimicCog'):
        super().__init__(timeout=60)
        self.cog = cog

    @ui.button(label="Yes, Shutdown", style=discord.ButtonStyle.danger)
    async def confirm_shutdown(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.edit_message(content="Shutting down child bots and main instance...", view=None)
        
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
