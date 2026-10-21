from ..utils.constants import *

import asyncio
import time
import discord
from discord import ui
import datetime
from collections import Counter
from string import Formatter
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Optional, get_args
from ..managers.storage_manager import IOManager
from ..managers.server_manager import record_lifted
from ..utils.helpers import (_sanitise_filename, birth_time, is_real_model, prune_openrouter_endpoints,
                             resolve_openrouter_endpoint, resolve_thinking_params, system_model,
                             system_settings)
from ..utils.loop_probe import rss_bytes
from ..utils.user_defaults import auto_wording, model_slot_defaults, other_provider
from .base_components import (BlockedGuard, TabbedView, TimeoutCleanupMixin, add_button,
                              add_select, build_confirm_view, invalidate_model_cache)

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog

from .gui_profiles import (ModelPickerMixin, OpenRouterHostView, ProfileManageView,
                           SingleProfileThinkingView, ThinkingPickerMixin)

class ModBaseView(TabbedView):
    """Base for every /mod tab.

    target_user_id is the moderated user, and every tab carries it so a trip out
    to Stats or Docs and back to Profiles does not lose it. Nothing but the
    Profiles tab reads it; the rest exist only to ferry it.
    """

    REBUILD = "_build_view"

    TABS = (
        ("Stats", "stats", lambda v: ModStatsView(v.cog, v.original_interaction, target_user_id=v.target_user_id)),
        ("Profiles", "profiles", lambda v: ModProfilesView(v.cog, v.original_interaction, target_user_id=v.target_user_id)),
        ("Prompts", "prompts", lambda v: ModPromptsView(v.cog, v.original_interaction, target_user_id=v.target_user_id)),
        ("Docs", "docs", lambda v: ModDocsView(v.cog, v.original_interaction, target_user_id=v.target_user_id)),
        ("Blacklist", "blacklist", lambda v: ModBlacklistView(v.cog, v.original_interaction, target_user_id=v.target_user_id)),
    )

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, current_tab: str,
                 target_user_id: Optional[int] = None):
        # Set before the nav bar is built: the tab factories read it off self.
        self.target_user_id = target_user_id
        super().__init__(cog, interaction, current_tab)


class ModDocsCreateModal(ui.Modal, title="Create Document"):
    category = ui.TextInput(label="Category (Subfolder)", placeholder="e.g. apis, commands", required=True, max_length=50)
    filename = ui.TextInput(label="File Name (without .txt)", placeholder="e.g. google_gemini", required=True, max_length=50)
    content = ui.TextInput(label="Content", style=discord.TextStyle.paragraph, required=True, max_length=4000)
    
    def __init__(self, view):
        super().__init__()
        self.parent_view = view
        
    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        cat = _sanitise_filename(self.category.value.strip().lower())
        name = _sanitise_filename(self.filename.value.strip().lower())
        if not cat or not name:
            await interaction.followup.send("Invalid category or file name.", ephemeral=True)
            return
        
        target_dir = os.path.join(DOCS_DIR, cat)
        os.makedirs(target_dir, exist_ok=True)
        filepath = os.path.join(target_dir, f"{name}.txt")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.content.value)
            
        self.parent_view.cog.bot.loop.create_task(self.parent_view.cog.help_service._load_and_embed_docs())
        
        self.parent_view.selected_category = cat
        self.parent_view.selected_file = f"{name}.txt"
        self.parent_view._build_view()
        await self.parent_view.update_display()

class ModDocsEditModal(ui.Modal, title="Edit Document"):
    content = ui.TextInput(label="Content", style=discord.TextStyle.paragraph, required=True, max_length=4000)
    
    def __init__(self, view, current_content):
        super().__init__()
        self.parent_view = view
        self.content.default = current_content
        
    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        cat = self.parent_view.selected_category
        name = self.parent_view.selected_file
        filepath = os.path.join(DOCS_DIR, cat, name)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.content.value)
            
        self.parent_view.cog.bot.loop.create_task(self.parent_view.cog.help_service._load_and_embed_docs())
        
        self.parent_view._build_view()
        await self.parent_view.update_display()

class ModDocsView(ModBaseView):
    def __init__(self, cog, interaction, target_user_id: Optional[int] = None):
        super().__init__(cog, interaction, "docs", target_user_id=target_user_id)
        self.selected_category = None
        self.selected_file = None
        self.current_page = 0
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        # 1. Scan Categories (Subfolders)
        categories = []
        if os.path.exists(DOCS_DIR):
            categories = sorted([d for d in os.listdir(DOCS_DIR) if os.path.isdir(os.path.join(DOCS_DIR, d))])
            
        if categories and not self.selected_category:
            self.selected_category = categories[0]
            
        # 2. Build Category Dropdown
        if categories:
            cat_opts = [discord.SelectOption(label=cat.upper(), value=cat, default=(cat == self.selected_category)) for cat in categories[:25]]
            async def cat_cb(i: discord.Interaction):
                self.selected_category = i.data['values'][0]
                self.selected_file = None
                self.current_page = 0
                self._build_view()
                await i.response.defer()
                await self.update_display()
            add_select(self, cat_opts, cat_cb, placeholder="Select Category...", row=0)

        # 3. Scan Files within Selected Category
        files = []
        if self.selected_category:
            cat_dir = os.path.join(DOCS_DIR, self.selected_category)
            if os.path.exists(cat_dir):
                files = sorted([f for f in os.listdir(cat_dir) if f.endswith(".txt")])
                
        if files and not self.selected_file:
            self.selected_file = files[0]

        # 4. Build File Dropdown with Sliding Window Pagination
        if files:
            num_pages = (len(files) - 1) // DROPDOWN_MAX_OPTIONS + 1
            if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
            
            start = self.current_page * DROPDOWN_MAX_OPTIONS
            page_files = files[start : start + DROPDOWN_MAX_OPTIONS]
            
            file_opts = []
            for f in page_files:
                label = f.replace(".txt", "").replace("_", " ").title()
                file_opts.append(discord.SelectOption(label=label[:100], value=f, default=(f == self.selected_file)))
                
            async def file_cb(i: discord.Interaction):
                self.selected_file = i.data['values'][0]
                self._build_view()
                await i.response.defer()
                await self.update_display()
            add_select(self, file_opts, file_cb, placeholder="Select Document...", row=1)

            if num_pages > 1:
                self._add_page_controls(num_pages, 2, repaint=False)

        # 5. Build Action Buttons
        action_row = 3 if len(files) > DROPDOWN_MAX_OPTIONS else 2
        
        async def create_cb(i: discord.Interaction):
            await i.response.send_modal(ModDocsCreateModal(self))
        add_button(self, "Create New", create_cb, style=discord.ButtonStyle.green, row=action_row)

        async def edit_cb(i: discord.Interaction):
            filepath = os.path.join(DOCS_DIR, self.selected_category, self.selected_file)
            content = ""
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            await i.response.send_modal(ModDocsEditModal(self, content))
        add_button(self, "Edit", edit_cb, style=discord.ButtonStyle.primary, row=action_row,
                   disabled=not self.selected_file)

        async def delete_cb(i: discord.Interaction):
            async def confirm_action(confirm_i: discord.Interaction):
                filepath = os.path.join(DOCS_DIR, self.selected_category, self.selected_file)
                if os.path.exists(filepath):
                    os.remove(filepath)

                # Re-index
                self.cog.bot.loop.create_task(self.cog.help_service._load_and_embed_docs())

                self.selected_file = None
                self._build_view()
                await confirm_i.response.edit_message(content="Deleted successfully.", view=None, embed=None)
                await self.update_display()

            confirm_view = build_confirm_view("Confirm Delete", confirm_action)
            await i.response.send_message(f"Are you sure you want to delete `{self.selected_file}`?", view=confirm_view, ephemeral=True)
        add_button(self, "Delete", delete_cb, style=discord.ButtonStyle.danger, row=action_row,
                   disabled=not self.selected_file)

        self._add_nav_buttons()

    def _get_embed(self) -> discord.Embed:
        embed = discord.Embed(title="Global System Documentation", color=discord.Color.dark_magenta())
        
        if self.selected_file and self.selected_category:
            filepath = os.path.join(DOCS_DIR, self.selected_category, self.selected_file)
            content = "No content."
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                except Exception as e:
                    content = f"Error reading file: {e}"
            
            clean_name = self.selected_file.replace(".txt", "").replace("_", " ").title()
            embed.set_author(name=f"Category: {self.selected_category.upper()} | Document: {clean_name}")
            
            if len(content) > 2048:
                embed.description = content[:2045] + "..."
            else:
                embed.description = content
        else:
            embed.description = "No documentation files found. Click 'Create New' to build your first RAG shard."
            
        return embed

    async def update_display(self):
        await self.original_interaction.edit_original_response(embed=self._get_embed(), view=self)

def _parse_block_duration(raw: str) -> Optional[float]:
    """"7d" / "12h" / "30m" -> an absolute epoch time. Blank or "0" means permanent.

    Returns None for permanent *and* for anything unparseable, because the two want the
    same outcome: a block the operator has to lift by hand. A typo must never shorten a
    block, and there is no reading of "7x" that is safer than "until I say otherwise".
    """
    raw = (raw or "").strip().lower()
    if not raw or raw == "0":
        return None
    units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    unit = units.get(raw[-1])
    if unit is None:
        return None
    try:
        amount = float(raw[:-1])
    except ValueError:
        return None
    if amount <= 0:
        return None
    return time.time() + amount * unit


class ModBlockUserModal(ui.Modal):
    user_id_input = ui.TextInput(label="Discord User ID", required=True, max_length=25)
    reason_input = ui.TextInput(label="Reason", required=False, max_length=300,
                                style=discord.TextStyle.paragraph,
                                placeholder="Your note. Never shown to them.")
    duration_input = ui.TextInput(label="Duration (30m / 12h / 7d — blank = permanent)",
                                  required=False, max_length=10)

    def __init__(self, view, scope: str):
        super().__init__(title=f"Block User — {BLACKLIST_SCOPE_LABELS[scope]}")
        self.parent_view = view
        self.scope = scope

    async def on_submit(self, i: discord.Interaction):
        uid_str = self.user_id_input.value.strip()
        if not uid_str.isdigit():
            await i.response.send_message("Invalid ID. Must be numeric.", ephemeral=True)
            return

        uid = int(uid_str)
        if uid == int(defaultConfig.DISCORD_OWNER_ID):
            # Nothing else in the dashboard can be undone from a locked-out account.
            await i.response.send_message(
                "You cannot block the bot owner — nothing would be left to unblock it with.",
                ephemeral=True)
            return

        self.parent_view.cog.server_manager.block_user(
            uid, scope=self.scope, reason=self.reason_input.value.strip(),
            by=i.user.id, until=_parse_block_duration(self.duration_input.value))

        self.parent_view.selected_id = uid
        self.parent_view.mode = "users"
        self.parent_view._build_view()
        await i.response.edit_message(embed=self.parent_view._get_embed(), view=self.parent_view)


class ModBlockGuildModal(ui.Modal):
    guild_id_input = ui.TextInput(label="Discord Server ID", required=True, max_length=25)
    reason_input = ui.TextInput(label="Reason", required=False, max_length=300,
                                style=discord.TextStyle.paragraph)

    def __init__(self, view, action: str):
        super().__init__(title="Block Server — " + ("Leave" if action == GUILD_BLOCK_LEAVE
                                                    else "Quarantine"))
        self.parent_view = view
        self.action = action

    async def on_submit(self, i: discord.Interaction):
        gid_str = self.guild_id_input.value.strip()
        if not gid_str.isdigit():
            await i.response.send_message("Invalid ID. Must be numeric.", ephemeral=True)
            return

        gid = int(gid_str)
        cog = self.parent_view.cog
        cog.server_manager.block_guild(gid, action=self.action,
                                       reason=self.reason_input.value.strip(), by=i.user.id)

        self.parent_view.selected_id = gid
        self.parent_view.mode = "guilds"
        self.parent_view._build_view()
        await i.response.edit_message(embed=self.parent_view._get_embed(), view=self.parent_view)

        # After the repaint, not before: leaving tears down sessions and talks to every
        # child bot, and the dashboard should not sit on an unacknowledged interaction
        # for the length of that.
        if self.action == GUILD_BLOCK_LEAVE:
            guild = cog.bot.get_guild(gid)
            if guild:
                await cog.server_manager.leave_blocked_guild(guild)


class ModBanEffectsView(BlockedGuard, ui.View):
    """The opt-in half of a ban: what to do about the things that outlive the person.

    Blocking someone stops them talking to the bot. It does not stop their child bots
    speaking, their published profiles being borrowed, or the sessions their profiles
    are seated in from rotating -- none of those paths asks who owns them. Each is a
    separate decision because each has a different cost, and unpublishing in particular
    cannot be undone from here.
    """

    def __init__(self, cog, user_id: int):
        super().__init__(timeout=300)
        self.cog = cog
        self.user_id = user_id
        self.effects = {"stop_bots": False, "unpublish": False, "suspend_sessions": False}
        self._build_view()

    LABELS = (("stop_bots", "Stop their child bots"),
              ("unpublish", "Unpublish from Public Library"),
              ("suspend_sessions", "Suspend sessions they are seated in"))

    def _build_view(self):
        self.clear_items()
        for key, label in self.LABELS:
            async def cb(i: discord.Interaction, key=key):
                self.effects[key] = not self.effects[key]
                self._build_view()
                await i.response.edit_message(view=self)
            add_button(self, f"{'☑' if self.effects[key] else '☐'} {label}", cb,
                       row=self.LABELS.index((key, label)),
                       style=(discord.ButtonStyle.primary if self.effects[key]
                              else discord.ButtonStyle.secondary))

        async def apply_cb(i: discord.Interaction):
            await i.response.defer()
            applied = await self.cog.server_manager.apply_ban_effects(
                self.user_id, **self.effects)
            await i.edit_original_response(
                content="\n".join(applied) if applied else "Nothing selected.", view=None)

        add_button(self, "Apply", apply_cb, row=3, style=discord.ButtonStyle.danger,
                   disabled=not any(self.effects.values()))


class ModBlacklistView(ModBaseView):
    """Users and servers, in one screen.

    The two lists are the same shape and the same gestures, and a server block is
    something the operator reaches for in the same moment as a user block -- one guild
    inviting fresh accounts is the case a per-user list cannot answer on its own.
    """

    #: The record's sections, in the order the mode button cycles them. These are the
    #: record's own section keys, not display names -- see _records.
    MODES = ("users", "guilds", "strikes")
    MODE_LABELS = {"users": "Users", "guilds": "Servers", "strikes": "Strikes"}

    def __init__(self, cog, interaction, target_user_id: Optional[int] = None):
        super().__init__(cog, interaction, "blacklist", target_user_id=target_user_id)
        self.mode = self.MODES[0]
        self.selected_id = None
        self.current_page = 0
        self._build_view()

    def _records(self) -> Dict[str, Dict]:
        """The mode is the record key, deliberately -- a second spelling for the same
        thing renders an empty list rather than an error."""
        return self.cog.blacklist_records.get(self.mode, {})

    def _entry(self, entry_id: int) -> Optional[Dict]:
        return self._records().get(str(entry_id))

    def _label_for(self, entry_id: int, entry: Dict) -> str:
        if self.mode == "strikes":
            user = self.cog.bot.get_user(entry_id)
            live = len(self.cog.server_manager.active_strikes(entry_id))
            return f"⚠️ {user.name if user else 'Unknown User'} ({entry_id}) · {live} of {STRIKE_THRESHOLD}"
        if self.mode == "users":
            user = self.cog.bot.get_user(entry_id)
            name = user.name if user else "Unknown User"
            marker = "⏳" if entry.get("until") else ""
        else:
            guild = self.cog.bot.get_guild(entry_id)
            name = guild.name if guild else "Unknown Server"
            marker = "🚪" if entry.get("action") == GUILD_BLOCK_LEAVE else "🔇"
        if record_lifted(entry):
            marker = "✔️"
        return f"{marker} {name} ({entry_id})".strip()

    def _build_view(self):
        self.clear_items()
        is_users = self.mode == "users"
        following = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]

        async def mode_cb(i: discord.Interaction):
            self.mode = following
            self.selected_id = None
            self.current_page = 0
            self._build_view()
            await i.response.edit_message(embed=self._get_embed(), view=self)
        add_button(self, f"{self.MODE_LABELS[following]} ▶", mode_cb, row=0,
                   style=discord.ButtonStyle.secondary)

        if is_users:
            for scope in BLACKLIST_SCOPES:
                async def block_cb(i: discord.Interaction, scope=scope):
                    await i.response.send_modal(ModBlockUserModal(self, scope))
                add_button(self, f"Block ({BLACKLIST_SCOPE_LABELS[scope]})", block_cb, row=0,
                           style=discord.ButtonStyle.primary)
        elif self.mode == "guilds":
            for action in GUILD_BLOCK_ACTIONS:
                async def gblock_cb(i: discord.Interaction, action=action):
                    await i.response.send_modal(ModBlockGuildModal(self, action))
                add_button(self, "Block (Leave)" if action == GUILD_BLOCK_LEAVE
                           else "Block (Quarantine)", gblock_cb, row=0,
                           style=discord.ButtonStyle.primary)

        entries = sorted(self._records().items(), key=lambda kv: kv[0])
        if entries:
            num_pages = (len(entries) - 1) // DROPDOWN_MAX_OPTIONS + 1
            if self.current_page >= num_pages:
                self.current_page = max(0, num_pages - 1)

            start = self.current_page * DROPDOWN_MAX_OPTIONS
            options = []
            for id_str, entry in entries[start:start + DROPDOWN_MAX_OPTIONS]:
                try:
                    entry_id = int(id_str)
                except ValueError:
                    continue
                options.append(discord.SelectOption(
                    label=self._label_for(entry_id, entry)[:100], value=id_str,
                    default=(self.selected_id == entry_id)))

            if options:
                async def sel_cb(i: discord.Interaction):
                    self.selected_id = int(i.data['values'][0])
                    self._build_view()
                    await i.response.edit_message(embed=self._get_embed(), view=self)
                add_select(self, options, sel_cb, row=1, placeholder={
                    "users": "Select a blocked user...", "guilds": "Select a blocked server...",
                    "strikes": "Select an account with strikes..."}[self.mode])

            if num_pages > 1:
                self._add_page_controls(num_pages, 2)

        async def remove_cb(i: discord.Interaction):
            if self.selected_id is not None:
                if self.mode == "strikes":
                    self.cog.server_manager.remove_strikes(self.selected_id)
                elif is_users:
                    self.cog.server_manager.unblock_user(self.selected_id)
                else:
                    self.cog.server_manager.unblock_guild(self.selected_id)
            self.selected_id = None
            self._build_view()
            await i.response.edit_message(embed=self._get_embed(), view=self)
        add_button(self, "Remove", remove_cb, row=3, style=discord.ButtonStyle.danger,
                   disabled=self.selected_id is None)

        if is_users:
            async def effects_cb(i: discord.Interaction):
                await i.response.send_message(
                    f"Effects to apply for `{self.selected_id}`. None of these is undone "
                    f"by unblocking them.",
                    view=ModBanEffectsView(self.cog, self.selected_id), ephemeral=True)
            add_button(self, "Ban Effects...", effects_cb, row=3,
                       disabled=self.selected_id is None)

        self._add_nav_buttons()

    def _get_embed(self):
        if self.mode == "strikes":
            return self._strikes_embed()
        is_users = self.mode == "users"
        embed = discord.Embed(
            title="Blocked Users" if is_users else "Blocked Servers",
            color=discord.Color.dark_red())

        entry = self._entry(self.selected_id) if self.selected_id is not None else None
        if entry:
            lines = [f"**{self._label_for(self.selected_id, entry)}**"]
            if is_users:
                lines.append("Scope: "
                             f"`{BLACKLIST_SCOPE_LABELS.get(entry.get('scope'), 'Full block')}`")
            else:
                lines.append("Action: "
                             f"`{GUILD_BLOCK_LABELS.get(entry.get('action'), 'Leave the server')}`")
            lines.append(f"Reason: {entry.get('reason') or '*none recorded*'}")
            if entry.get("at"):
                lines.append(f"Blocked: `{entry['at'][:19].replace('T', ' ')} UTC`")
            if entry.get("by"):
                lines.append(f"By: <@{entry['by']}>")
            if record_lifted(entry):
                until = int(entry["until"])
                lines.append(f"Lifted: <t:{until}:R> · record deleted "
                             f"<t:{until + BLACKLIST_LIFTED_RETENTION_SECONDS}:R>")
                lines.append("\nUse 'Remove' to delete this record now.")
            else:
                if entry.get("until"):
                    lines.append(f"Lifts: <t:{int(entry['until'])}:R>")
                lines.append("\nUse 'Remove' to lift this block. Its record is kept "
                             f"{BLACKLIST_LIFTED_RETENTION_SECONDS // 86400} days, as the "
                             "Privacy Policy says.")
            embed.description = "\n".join(lines)
        else:
            lifted = sum(record_lifted(e) for e in self._records().values())
            total = len(self._records()) - lifted
            kept = f" · ✔️ lifted, kept on file: **{lifted}**" if lifted else ""
            if is_users:
                embed.description = (
                    f"Blocked users: **{total}**{kept}\n\n"
                    "A blocked user is never told. They keep `/privacy`, `/terms`, and a "
                    "`/settings` page showing that they are restricted and the dates — "
                    "not the reason, and not who set it.\n\n"
                    f"**{BLACKLIST_SCOPE_LABELS[BLACKLIST_SCOPE_GENERATION]}** is enforced "
                    "as a full block for now: most commands can reach a model call.\n\n"
                    "Blocks may be temporary; they lift on their own within 30 seconds of "
                    "expiring.")
            else:
                embed.description = (
                    f"Blocked servers: **{total}**{kept}\n\n"
                    "🚪 **Leave** — departs now, on re-invite, and on boot; child bots go "
                    "too.\n"
                    "🔇 **Quarantine** — stays, generates nothing. Reversible without a "
                    "re-invite.")
        return embed

    def _strikes_embed(self) -> discord.Embed:
        embed = discord.Embed(title="Strikes", color=discord.Color.dark_red())
        sm = self.cog.server_manager
        if self.selected_id is not None:
            user = self.cog.bot.get_user(self.selected_id)
            lines = [f"**{user.name if user else 'Unknown User'}** (`{self.selected_id}`)"]
            for strike in sm.active_strikes(self.selected_id):
                lines.append(f"• `{strike['name']}` (`{strike['pid']}`) — refused <t:{int(strike['at'])}:D>, "
                             f"expires <t:{int(strike['at'] + STRIKE_LIFETIME_SECONDS)}:D>")
            lines.append("\nUse 'Remove' to remove all of this account's strikes. A block they "
                         "caused stays until you lift it under Users.")
            embed.description = "\n".join(lines)
            return embed
        embed.description = (
            f"Accounts with strikes: **{len(self._records())}**\n\n"
            "A strike lands when one of an account's profiles is Refused by its content rating "
            f"-- once per profile -- and lasts {STRIKE_LIFETIME_SECONDS // 86400} days. At "
            f"{STRIKE_THRESHOLD} the account is blocked automatically, with no end date, until "
            "you lift it. Clear Refusal under a profile's Content Safety removes its strike. "
            "The account sees its strike count in `/settings`.")
        return embed

    async def update_display(self):
        await self.original_interaction.edit_original_response(embed=self._get_embed(), view=self)


#: Read at import, which is boot: close enough for an uptime line.
_STARTED_AT = time.time()

UNREGISTERED_LABEL = "⚠️ Not registered"


def _scan_users() -> List[Tuple[int, Optional[str], int, int, Optional[float]]]:
    """(id, provider, personal, borrowed, joined) for every user directory. Run off the loop.

    `joined` is `about.registered_at`, else the directory's birth time (None if the
    filesystem will not say) -- the case for everyone who registered before the stamp.

    `provider` is None for a directory that predates `/start` -- data, but never registered,
    so locked out of creating more until they run it. Listed, not skipped: a mod who cannot
    see them cannot tell them why.

    `is_registered`'s test against the raw `index.json`, never `_get_user_index`: that
    caches each one, pushing every active user out of the LRU, and may repair -- a
    stats screen must not write.
    """
    rows = []
    if not os.path.isdir(USERS_DIR):
        return rows
    for name in os.listdir(USERS_DIR):
        if not name.isdigit():
            continue
        index = IOManager.read_json(os.path.join(USERS_DIR, name, "index.json"))
        if not isinstance(index, dict):
            index = {}
        about = index.get("about")
        about = about if isinstance(about, dict) else {}
        provider = about.get("provider")
        rows.append((int(name), provider if provider in PROVIDER_CHOICES else None,
                     len(index.get("personal") or {}), len(index.get("borrowed") or {}),
                     about.get("registered_at") or birth_time(os.path.join(USERS_DIR, name))))
    return rows


def _paginate(lines: List[str], empty: str) -> List[str]:
    # 20 two-line rows split by a blank line, worst case (100-character server name) ~3.8k of
    # the embed's 4096.
    return ["\n\n".join(lines[i:i + 20]) for i in range(0, len(lines), 20)] or [empty]


class ModStatsView(ModBaseView):
    CATEGORIES = ("Servers", "Users")

    def __init__(self, cog, interaction, target_user_id: Optional[int] = None):
        super().__init__(cog, interaction, "stats", target_user_id=target_user_id)
        self.selected_category = "Servers"
        self.current_page = 0
        # Read from disk when Users is first opened, not on every /mod.
        self._users: Optional[List[Tuple[int, Optional[str], int, int, Optional[float]]]] = None
        self._build_view()

    def _pages(self) -> List[str]:
        if self.selected_category == "Servers":
            now = datetime.datetime.now(datetime.timezone.utc)
            guilds = sorted(self.cog.bot.guilds, key=lambda g: (g.me and g.me.joined_at) or now)
            live = Counter(getattr(getattr(self.cog.bot.get_channel(ch), "guild", None), "id", None)
                           for ch in self.cog.multi_profile_channels)
            lines = []
            for j, guild in enumerate(guilds, start=1):
                joined = guild.me.joined_at.strftime("%d/%m/%Y") if (guild.me and guild.me.joined_at) else "Unknown"
                lines.append(f"{j}. **{guild.name}** · `{guild.id}`\n"
                             f"Members `{guild.member_count or '?'}` · Sessions `{live[guild.id]}` · "
                             f"Joined `{joined}`")
            return _paginate(lines, "No server data available.")

        rows = sorted(self._users or [], key=lambda r: (r[2], r[3]), reverse=True)
        lines = []
        for j, (uid, provider, personal, borrowed, joined) in enumerate(rows, start=1):
            user = self.cog.bot.get_user(uid)
            when = (datetime.datetime.fromtimestamp(joined, datetime.timezone.utc).strftime("%d/%m/%Y")
                    if joined else "Unknown")
            lines.append(f"{j}. **{user.name if user else 'Unknown User'}** · `{uid}`\n"
                         f"{PROVIDER_CHOICES.get(provider, UNREGISTERED_LABEL)} · "
                         f"Personal `{personal}` · Borrowed `{borrowed}` · Joined `{when}`")
        return _paginate(lines, "No users.")

    def _build_view(self):
        self.clear_items()

        cat_opts = [discord.SelectOption(label=cat, value=cat, default=(cat == self.selected_category)) for cat in self.CATEGORIES]
        async def cat_cb(i: discord.Interaction):
            await i.response.defer()
            self.selected_category = i.data['values'][0]
            self.current_page = 0
            if self.selected_category == "Users" and self._users is None:
                self._users = await asyncio.to_thread(_scan_users)
            self._build_view()
            await self.update_display()
        add_select(self, cat_opts, cat_cb, placeholder="Select Category...", row=0)

        num_pages = len(self._pages())
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)

        self._add_page_controls(num_pages, 1)
        self._add_wipe_button()
        self._add_nav_buttons()

    def _add_wipe_button(self):
        """Instance-wide `/suspend`.

        The per-channel command only reaches the channel it is typed in, so recovering
        from a bad deploy meant visiting every session by hand. This runs the same
        teardown across every channel the bot holds a session for.
        """
        live = len(self.cog.multi_profile_channels)
        # Never disabled at zero live sessions: the disk sweep still has work to do
        # when every session is dehydrated or its channel has gone uncached.
        async def wipe_cb(i: discord.Interaction):
            async def confirm(confirm_i: discord.Interaction):
                await confirm_i.response.edit_message(
                    content=f"Suspending `{live}` session(s) across every server...",
                    view=None, embed=None)
                channels, guilds, swept = await self.cog.session_manager.suspend_all_sessions()
                await confirm_i.edit_original_response(
                    content=(f"**Global suspend complete.**\n"
                             f"• Live sessions suspended: `{channels}`\n"
                             f"• Servers affected: `{guilds}`\n"
                             f"• Session log directories removed: `{swept}`\n\n"
                             "Every channel is now silent until a session is configured again."),
                    view=None)
                # The button's count is now stale -- repaint the dashboard rather than
                # leave a live-looking control behind on it.
                self._build_view()
                await self.update_display()

            await i.response.send_message(
                f"**Suspend every session on this instance?**\n"
                f"`{live}` live channel session(s) will be torn down exactly as `/suspend` does: "
                "chat logs deleted, child bots told to drop the channel, Freewill triggers "
                "disabled. Any session log left on disk for a channel the bot can no longer "
                "see is swept too. Profiles, memories and settings are untouched.\n\n"
                "This cannot be undone.",
                view=build_confirm_view("Wipe Every Session", confirm), ephemeral=True)
        add_button(self, f"Wipe All Sessions ({live})", wipe_cb, style=discord.ButtonStyle.danger,
                   row=2)

    def _get_embed(self):
        embed = discord.Embed(title="MimicAI Statistics", color=discord.Color.gold())
        pages = self._pages()
        embed.description = pages[self.current_page]

        sessions = self.cog.multi_profile_channels.values()
        clients = self.cog.child_bot_manager.clients.values()
        rss = rss_bytes()
        embed.add_field(name="Instance", inline=False, value=(
            f"Up since <t:{int(_STARTED_AT)}:R>"
            + (f" · Memory `{rss / 2**20:.0f} MB`" if rss else "") + "\n"
            f"Sessions `{len(sessions)}` live, `{sum(map(self.cog.session_manager.is_started, sessions))}` started · "
            f"Child bots `{sum(c.is_ready() for c in clients)}` online of `{len(self.cog.child_bots)}`"))

        if self.selected_category == "Users" and self._users is not None:
            by_provider = Counter(r[1] for r in self._users)
            unregistered = by_provider.pop(None, 0)
            embed.add_field(name="Users", inline=False, value=(
                f"Registered `{by_provider.total()}` — "
                + " · ".join(f"{PROVIDER_CHOICES[p]} `{n}`" for p, n in by_provider.most_common())
                + (f" · {UNREGISTERED_LABEL} `{unregistered}`" if unregistered else "") + "\n"
                f"Personal profiles `{sum(r[2] for r in self._users)}` · "
                f"Borrows `{sum(r[3] for r in self._users)}`"))
        return embed

    async def update_display(self):
        await self.original_interaction.edit_original_response(embed=self._get_embed(), view=self)

class ModProfilesModal(ui.Modal, title="Enter User ID"):
    user_id_input = ui.TextInput(label="Discord User ID", required=True)
    def __init__(self, view):
        super().__init__()
        self.parent_view = view
    async def on_submit(self, i: discord.Interaction):
        uid_str = self.user_id_input.value.strip()
        if not uid_str.isdigit():
            await i.response.send_message("Invalid ID. Must be numeric.", ephemeral=True)
            return
        self.parent_view.target_user_id = int(uid_str)
        # Reset the page, or switching from a user with four pages of profiles to one
        # with a single page lands on an empty window.
        self.parent_view.current_page = 0
        self.parent_view._build_view()
        await i.response.edit_message(embed=self.parent_view._get_embed(), view=self.parent_view)

#: Why a refusal waits on the operator, as the Review Queue says it.
_REVIEW_WORDING = {"disputed": "second model disagreed", "unconfirmed": "second model could not answer",
                   "appealed": "owner appealed"}


class ModProfilesView(ModBaseView):
    def __init__(self, cog, interaction, target_user_id: Optional[int] = None):
        super().__init__(cog, interaction, "profiles", target_user_id=target_user_id)
        self.current_page = 0
        #: `refused_profiles`' rows -- (owner_id, name, listed at, why flagged, why a refusal
        #: awaits review) -- once Review Queue has
        #: been pressed. Read in a thread on the press: each entry is checked against its
        #: profile's own rating.
        self.refused_rows: Optional[List[Tuple[int, str, int, Optional[str], Optional[str]]]] = None
        self._build_view()

    def _resolve_profiles(self) -> List[str]:
        """Every profile name the moderated user can be managed through, sorted.

        System (class X) profiles are included: they live in the bot owner's own
        index, so this only ever yields any when the moderated user is the owner,
        but leaving them out made them unreachable from /mod entirely.
        """
        if not self.target_user_id:
            return []
        index = self.cog.profile_manager._get_user_index(self.target_user_id)
        names = set(index.get("personal", [])) | set(index.get("borrowed", [])) | set(index.get("system", []))
        return sorted(names)

    def _build_view(self):
        self.clear_items()
        
        async def enter_cb(i: discord.Interaction):
            await i.response.send_modal(ModProfilesModal(self))
        add_button(self, "Enter User ID", enter_cb, style=discord.ButtonStyle.success, row=0)

        # Instance-wide maintenance, deliberately manual. Nothing runs this on boot:
        # it is a one-off baseline reset the operator performs once, which is why no
        # rating record carries a schema marker recording whether it has happened.
        async def reset_cb(i: discord.Interaction):
            async def confirm(confirm_i: discord.Interaction):
                await confirm_i.response.edit_message(
                    content="Resetting every content rating on this instance...",
                    view=None, embed=None)
                counts = await asyncio.to_thread(
                    self.cog.profile_manager.reset_all_content_ratings)

                delisted = counts["delisted"]
                lines = [
                    "**Reset complete.**",
                    f"• Profiles scanned: `{counts['scanned']}`",
                    f"• Reset to Unrated: `{counts['reset']}`",
                    f"• Borrowed profiles skipped: `{counts['skipped_borrowed']}`",
                    f"• Legacy `safety_level` fields stripped: `{counts['legacy_fields']}`",
                    f"• Public Library entries delisted: `{len(delisted)}`",
                ]
                if counts["errors"]:
                    lines.append(f"• ⚠️ Errors: `{counts['errors']}` (see the console log)")
                lines.append("\nEvery profile is now Unrated. Owners re-submit from "
                             "`/profile manage` -> Home -> Content Safety.")
                await confirm_i.edit_original_response(content="\n".join(lines), view=None)

            await i.response.send_message(
                "**Reset every content rating on this instance?**\n"
                "Every profile becomes **Unrated** — including Adult and Refused ones — and the "
                "entire Public Library is delisted, because an Unrated profile cannot be "
                "published. Existing borrows keep working.\n\n"
                "Owners re-submit their own profiles afterwards; an 18+ declaration can be "
                "re-applied in one click with no API call.\n\n"
                "This cannot be undone.",
                view=build_confirm_view("Reset Everything", confirm), ephemeral=True)
        add_button(self, "Reset All Content Ratings", reset_cb, style=discord.ButtonStyle.danger,
                   row=0)

        # Refusals awaiting review -- the second model disagreed or could not answer, or the
        # owner appealed -- and profiles no model gave a verdict for: the operator's review
        # queue. Opening one lands on its dashboard in mod view, where Content Safety offers
        # Clear or Uphold, or a verdict for a flagged one.
        async def refused_cb(i: discord.Interaction):
            await i.response.defer()
            self.refused_rows = await asyncio.to_thread(self.cog.profile_manager.refused_profiles)
            self._build_view()
            await i.edit_original_response(embed=self._get_embed(), view=self)
        add_button(self, "Review Queue", refused_cb, style=discord.ButtonStyle.secondary,
                   row=0, emoji="⛔")

        if self.refused_rows:
            async def refused_sel(i: discord.Interaction):
                owner, _, pname = i.data['values'][0].partition(":")
                self.target_user_id = int(owner)
                await i.response.defer()
                pm_view = ProfileManageView(self.cog, self.original_interaction, pname, False,
                                            target_user_id=int(owner), is_mod_view=True)
                embed = await self.cog.profile_manager._build_profile_manage_embed(
                    self.original_interaction, pname, target_user_id=int(owner))
                await self.original_interaction.edit_original_response(embed=embed, view=pm_view)
            options = []
            for owner, name, at, flagged, review in self.refused_rows[:DROPDOWN_MAX_OPTIONS]:
                user = self.cog.bot.get_user(owner)
                day = f"{datetime.datetime.fromtimestamp(at, datetime.timezone.utc):%Y-%m-%d}"
                options.append(discord.SelectOption(
                    label=f"{name} — {user.name if user else owner}"[:100], value=f"{owner}:{name}",
                    emoji="⚠️" if flagged else "⛔",
                    description=(f"Flagged {day}: {flagged}" if flagged
                                 else f"Refused {day}: {_REVIEW_WORDING.get(review, review)}")[:100]))
            add_select(self, options, refused_sel, placeholder="Review a profile...", row=3)
        
        if self.target_user_id:
            index = self.cog.profile_manager._get_user_index(self.target_user_id)
            profiles = self._resolve_profiles()
            
            if profiles:
                num_pages = (len(profiles) - 1) // DROPDOWN_MAX_OPTIONS + 1
                if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
                
                start = self.current_page * DROPDOWN_MAX_OPTIONS
                page_items = profiles[start : start + DROPDOWN_MAX_OPTIONS]
                
                options = [discord.SelectOption(label=p[:100], value=p[:100]) for p in page_items]
                async def sel_cb(i: discord.Interaction):
                    pname = i.data['values'][0]
                    is_b = pname in index.get("borrowed", [])
                    await i.response.defer()
                    pm_view = ProfileManageView(self.cog, self.original_interaction, pname, is_b, target_user_id=self.target_user_id, is_mod_view=True)
                    embed = await self.cog.profile_manager._build_profile_manage_embed(self.original_interaction, pname, target_user_id=self.target_user_id)
                    await self.original_interaction.edit_original_response(embed=embed, view=pm_view)
                add_select(self, options, sel_cb, placeholder="Select a profile to manage...",
                           row=1)

                if num_pages > 1:
                    self._add_page_controls(num_pages, 2)

        self._add_nav_buttons()

    def _get_embed(self):
        embed = discord.Embed(title="Moderator Profile Dashboard", color=discord.Color.red())
        if self.refused_rows is not None:
            flagged = sum(1 for row in self.refused_rows if row[3])
            embed.add_field(
                name="⛔ Review queue",
                value=(f"**{len(self.refused_rows) - flagged}** refused and **{flagged}** flagged, "
                       "oldest first.\n⛔ One model found it sexualises a minor or a real "
                       "person, and the second disagreed or could not answer, or its owner "
                       "appealed. It runs nowhere; Uphold it under Content Safety to strike "
                       "it, or Clear it.\n⚠️ No rating "
                       "model gave a verdict; it stays Pending until you rate or refuse it there."
                       if self.refused_rows else "None awaiting review."),
                inline=False)
        if not self.target_user_id:
            embed.description = "Click the button below to enter a User ID to manage."
            return embed

        user = self.cog.bot.get_user(self.target_user_id)
        uname = user.name if user else "Unknown"
        profiles = self._resolve_profiles()
        if profiles:
            embed.description = (f"Managing User: **{uname}** (`{self.target_user_id}`)\n"
                                 f"**{len(profiles)}** profile(s). Select one below.")
        else:
            embed.description = (f"Managing User: **{uname}** (`{self.target_user_id}`)\n"
                                 "This user has no profiles — check the ID, or they have not "
                                 "created any yet.")
        return embed

    async def update_display(self):
        await self.original_interaction.edit_original_response(embed=self._get_embed(), view=self)

# --- Global prompt metadata -------------------------------------------------
#
# Every entry is (label, key, default_text). The placeholder table below records
# how each prompt is consumed downstream, which the edit modal validates against
# before saving -- see _validate_prompt_placeholders.

# Prompts are grouped because there are more of them than a single Discord select
# can hold (25 options), and the list is still growing.
MOD_PROMPT_CATEGORIES = [
    ("Core Instructions", [
        ("Session Rules", "CONTEXT_RULES", DEFAULT_SESSION_RULES),
        ("Enhance Awareness", "ENHANCE_AWARENESS", DEFAULT_ENHANCE_AWARENESS),
        ("Enhance Awareness · Group", "ENHANCE_AWARENESS_GROUP", DEFAULT_ENHANCE_AWARENESS_GROUP),
        ("Current Time", "TIME_CONTEXT", DEFAULT_CURRENT_TIME),
        ("Birthday Context", "BIRTHDAY_CONTEXT", DEFAULT_BIRTHDAY_CONTEXT),
        ("Negative Constraints", "NEGATIVE_CONSTRAINTS", DEFAULT_NEGATIVE_CONSTRAINTS),
        ("Training Data Injection", "TRAINING_DATA_INJECTION", DEFAULT_TRAINING_DATA_INJECTION),
        ("Neuro-Endocrine Engine", "NEURO_DEFINITION", DEFAULT_NEURO_DEFINITION),
        ("Neuro-Endocrine State", "NEURO_ENGINE", DEFAULT_NEURO_INSTRUCTION),
        ("Bonds", "BOND_DEFINITION", DEFAULT_BOND_DEFINITION),
        ("Bond State", "BOND_STATE", DEFAULT_BOND_INSTRUCTION),
        ("Content Policy (non-18+ channels)", "CONTENT_POLICY", DEFAULT_CONTENT_POLICY),
    ]),
    ("Turn Flow", [
        ("Whisper Injection", "WHISPER_INJECTION", DEFAULT_WHISPER_INJECTION),
        ("Whisper Recap", "WHISPER_RECAP", DEFAULT_WHISPER_RECAP),
        ("Kickstart: Start", "KICKSTART_START", DEFAULT_KICKSTART_START),
        ("Kickstart: Continue", "KICKSTART_CONTINUE", DEFAULT_KICKSTART_CONTINUE),
        ("Kickstart: Idle", "KICKSTART_IDLE", DEFAULT_KICKSTART_IDLE),
        ("Kickstart: Unanswered", "KICKSTART_UNANSWERED", DEFAULT_KICKSTART_UNANSWERED),
        ("AI Director Prompt", "DIRECTOR_USER_PROMPT", DEFAULT_DIRECTOR_USER_PROMPT),
        ("Speak: Re-voice (strict)", "SPEAK_REWRITE_STRICT", DEFAULT_SPEAK_REWRITE_STRICT),
        ("Speak: Re-voice (loose)", "SPEAK_REWRITE_LOOSE", DEFAULT_SPEAK_REWRITE_LOOSE),
        ("Session Synopsis: System", "SESSION_SYNOPSIS", DEFAULT_SESSION_SYNOPSIS_PROMPT),
        ("Session Synopsis: User", "SESSION_SYNOPSIS_USER", DEFAULT_SESSION_SYNOPSIS_USER_PROMPT),
    ]),
    ("Image Generation", [
        ("Image: Present (own)", "IMAGE_PRESENT", DEFAULT_IMAGE_PRESENT),
        ("Image: Present (other's)", "IMAGE_PRESENT_OTHER", DEFAULT_IMAGE_PRESENT_OTHER),
        ("Image: Failed", "IMAGE_FAILED", DEFAULT_IMAGE_FAILED),
        ("Image: Appearance Preamble", "IMAGE_APPEARANCE", DEFAULT_IMAGE_APPEARANCE),
        ("Image: Grounding Preamble", "IMAGE_GROUNDING", DEFAULT_IMAGE_GROUNDING),
    ]),
    ("Grounding & Critics", [
        # RAG (tool mode): what the character is told, and what the researcher is told.
        ("Web Search \u00b7 Character Brief", "SEARCH_INSTRUCTION", DEFAULT_SEARCH_INSTRUCTION),
        ("Web Search \u00b7 Researcher", "WEB_SEARCH_RESEARCH", DEFAULT_WEB_SEARCH_RESEARCH),
        # Legacy RAG: the gate that decides and searches in front of every round.
        ("Legacy RAG (Text)", "WEB_GROUNDING_TEXT", DEFAULT_WEB_GROUNDING_TEXT),
        ("Legacy RAG (Visual)", "WEB_GROUNDING_VISUAL", DEFAULT_WEB_GROUNDING_VISUAL),
        ("Legacy RAG Payload", "GROUNDING_RAG_PAYLOAD", DEFAULT_GROUNDING_RAG_PAYLOAD),
        ("Attachment Describer (Simulated vision)", "MEDIA_DESCRIPTION", DEFAULT_MEDIA_DESCRIPTION),
        ("Anti-Repetition Critic", "ANTI_REPETITION", DEFAULT_ANTI_REPETITION_PROMPT),
        ("Content Classifier (18+ gating)", "CONTENT_CLASSIFIER", DEFAULT_CONTENT_CLASSIFIER_PROMPT),
        ("Content Classifier · Minor Check", "CONTENT_CONFIRMER", DEFAULT_CONTENT_CONFIRMER_PROMPT),
        ("Content Classifier · Real Person Check", "CONTENT_CONFIRMER_REAL_PERSON",
         DEFAULT_REAL_PERSON_CONFIRMER_PROMPT),
    ]),
    ("Memory & Training", [
        ("LTM Summarization", "LTM_SUMMARIZATION_INSTRUCTIONS", DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS),
        # Read by prompt_builder since Memory Search shipped, but never listed here, so
        # the one prompt an operator would most want to retune was unreachable.
        ("Recalled Memories", "ARCHIVE_CONTEXT", DEFAULT_ARCHIVE_CONTEXT),
        ("Memory Search \u00b7 Character Brief", "RECALL_INSTRUCTION", DEFAULT_RECALL_INSTRUCTION),
        ("Memory Flag \u00b7 Character Brief", "MEMORY_CAPTURE_INSTRUCTION", DEFAULT_MEMORY_CAPTURE_INSTRUCTION),
        ("Memory Flag \u00b7 Written Note", "MEMORY_WRITTEN_NOTE", DEFAULT_MEMORY_WRITTEN_NOTE),
        ("Training Analyst", "TRAINING_ANALYST", DEFAULT_TRAINING_ANALYST_PROMPT),
    ]),
    ("Bot Utilities", [
        ("Profile Generator", "PROFILE_GENERATOR", DEFAULT_PROFILE_GENERATOR_PROMPT),
        ("Character Card Import", "CARD_IMPORT", DEFAULT_CARD_IMPORT_PROMPT),
        ("Help Mode Protocol", "HELP_MODE_INJECTION", DEFAULT_HELP_MODE_INJECTION),
    ]),
]

# Flattened (label, key, default), in category order.
MOD_PROMPT_DEFINITIONS = [entry for _cat, entries in MOD_PROMPT_CATEGORIES for entry in entries]

# key -> (required placeholder names, substitution mode).
#
# "format" prompts are passed through str.format() at generation time, so an
# unknown field name or an unbalanced brace in a moderator's custom text raises
# KeyError/ValueError deep inside the turn path -- for CONTEXT_RULES that is
# every turn of every profile. "replace" prompts go through str.replace() and so
# tolerate stray braces; only the presence of the placeholder matters.
# Keys absent from this table are consumed verbatim and need no validation.
MOD_PROMPT_PLACEHOLDERS: Dict[str, Tuple[Set[str], str]] = {
    "CONTEXT_RULES": ({"profile_id_placeholder"}, "format"),
    "TIME_CONTEXT": ({"time_str"}, "format"),
    "BIRTHDAY_CONTEXT": ({"birthdays"}, "format"),
    "MEMORY_WRITTEN_NOTE": ({"when"}, "format"),
    "ARCHIVE_CONTEXT": ({"memories"}, "format"),
    "NEGATIVE_CONSTRAINTS": ({"constraints"}, "format"),
    "TRAINING_DATA_INJECTION": ({"examples_block"}, "format"),
    "NEURO_ENGINE": ({"d", "c", "o", "a"}, "format"),
    "BOND_STATE": ({"bonds"}, "format"),
    "WHISPER_INJECTION": ({"whisper_content"}, "format"),
    "WHISPER_RECAP": ({"whispers"}, "format"),
    "DIRECTOR_USER_PROMPT": ({"history"}, "format"),
    "SPEAK_REWRITE_STRICT": ({"source_text"}, "format"),
    "SPEAK_REWRITE_LOOSE": ({"source_text"}, "format"),
    "SESSION_SYNOPSIS": ({"max_words"}, "format"),
    "SESSION_SYNOPSIS_USER": ({"previous_synopsis", "transcript"}, "format"),
    "IMAGE_PRESENT": ({"prompt"}, "format"),
    "IMAGE_PRESENT_OTHER": ({"name", "prompt"}, "format"),
    "IMAGE_FAILED": ({"prompt", "reason"}, "format"),
    "IMAGE_APPEARANCE": ({"appearance", "prompt"}, "format"),
    "IMAGE_GROUNDING": ({"prompt", "grounding"}, "format"),
    "GROUNDING_RAG_PAYLOAD": ({"transcript", "query"}, "format"),
    "ANTI_REPETITION": ({"char_name"}, "format"),
    "TRAINING_ANALYST": ({"verbosity", "examples_block"}, "format"),
    "PROFILE_GENERATOR": ({"prompt"}, "format"),
    "CARD_IMPORT": ({"prompt"}, "format"),
    "HELP_MODE_INJECTION": ({"docs"}, "replace"),
}

# Discord's own ceilings: a text input holds 4000 characters and a modal holds
# five components.
MODAL_TEXT_INPUT_MAX = 4000
MODAL_MAX_TEXT_INPUTS = 5
MODAL_PROMPT_CAPACITY = MODAL_TEXT_INPUT_MAX * MODAL_MAX_TEXT_INPUTS


def _split_prompt_for_modal(text: str, max_len: int = MODAL_TEXT_INPUT_MAX,
                            max_parts: int = MODAL_MAX_TEXT_INPUTS) -> Tuple[List[str], List[str]]:
    """Splits text into at most max_parts chunks of at most max_len characters.

    Prompts outgrew the single 4000-character text input this modal used to be,
    and Discord rejects a modal whose default value exceeds the field's
    max_length -- an over-long entry could not be opened at all. The worst
    offender was DEFAULT_HELP_MODE_INJECTION, which carried the whole dashboard
    tree inline; that tree is now generated and substituted at injection time, so
    the default fits comfortably again. The split stays because an operator
    override is free to grow past the limit.

    Cuts are preferred at line boundaries. Returns (chunks, joiners), where
    joiners[i] is the text consumed between chunks[i] and chunks[i + 1] -- a
    newline for a line-boundary cut, empty for a hard cut inside an over-long
    line -- so interleaving the two reproduces the input exactly.
    """
    if len(text) <= max_len:
        return [text], []

    chunks: List[str] = []
    joiners: List[str] = []
    remaining = text

    while len(remaining) > max_len and len(chunks) < max_parts - 1:
        # +1 so a newline sitting exactly on the boundary is still a clean cut.
        cut = remaining[:max_len + 1].rfind("\n")
        if cut <= 0:
            chunks.append(remaining[:max_len])
            joiners.append("")
            remaining = remaining[max_len:]
        else:
            chunks.append(remaining[:cut])
            joiners.append("\n")
            remaining = remaining[cut + 1:]

    chunks.append(remaining)
    return chunks, joiners


def _join_prompt_parts(chunks: List[str], joiners: List[str]) -> str:
    """Inverse of _split_prompt_for_modal."""
    out = []
    for i, chunk in enumerate(chunks):
        out.append(chunk)
        if i < len(joiners):
            out.append(joiners[i])
    return "".join(out)


def _validate_prompt_placeholders(key: str, text: str) -> Optional[str]:
    """Returns an error message if text would break its consumer, else None."""
    spec = MOD_PROMPT_PLACEHOLDERS.get(key)
    if not spec:
        return None
    required, mode = spec

    if mode == "replace":
        missing = sorted(f"{{{name}}}" for name in required if f"{{{name}}}" not in text)
        if missing:
            return f"This prompt must still contain {', '.join(f'`{m}`' for m in missing)}."
        return None

    try:
        found = {field for _, field, _, _ in Formatter().parse(text) if field is not None}
    except ValueError as e:
        return (f"Unbalanced or malformed braces: {e}. "
                "Write `{{` and `}}` for a literal brace.")

    unknown = sorted(f for f in found if f not in required)
    if unknown:
        allowed = ", ".join(f"`{{{name}}}`" for name in sorted(required))
        bad = ", ".join(f"`{{{f}}}`" for f in unknown)
        return (f"Unknown placeholder(s) {bad}. This prompt only accepts {allowed}. "
                "Write `{{` and `}}` for a literal brace.")

    missing = sorted(f"{{{name}}}" for name in required if name not in found)
    if missing:
        return f"This prompt must still contain {', '.join(f'`{m}`' for m in missing)}."
    return None


class ModPromptModal(ui.Modal, title="Edit Global Prompt"):
    def __init__(self, view, key, default_text):
        super().__init__()
        self.parent_view = view
        self.key = key

        curr_val = self.parent_view.cog.global_prompts.get(key, default_text)
        chunks, self.joiners = _split_prompt_for_modal(curr_val)

        self.prompt_inputs: List[ui.TextInput] = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            label = "Prompt (blank to reset to default)" if total == 1 else f"Prompt — part {idx} of {total}"
            field = ui.TextInput(
                label=label[:45],
                style=discord.TextStyle.paragraph,
                default=chunk,
                required=False,
                max_length=MODAL_TEXT_INPUT_MAX,
            )
            self.prompt_inputs.append(field)
            self.add_item(field)

    async def on_submit(self, i: discord.Interaction):
        val = _join_prompt_parts([f.value for f in self.prompt_inputs], self.joiners).strip()

        if val:
            error = _validate_prompt_placeholders(self.key, val)
            if error:
                await i.response.send_message(
                    f"❌ `{self.key}` was **not** saved.\n{error}", ephemeral=True)
                return
            self.parent_view.cog.global_prompts[self.key] = val
        else:
            self.parent_view.cog.global_prompts.pop(self.key, None)

        self.parent_view.cog.server_manager._save_global_prompts()
        action = "Updated" if val else "Reset to default"
        await i.response.send_message(f"{action} `{self.key}` successfully.", ephemeral=True)

        # Repaint the dashboard so the customised/default markers reflect the save.
        self.parent_view._build_view()
        await self.parent_view.update_display()

class ModPromptsView(ModBaseView):
    def __init__(self, cog, interaction, target_user_id: Optional[int] = None):
        super().__init__(cog, interaction, "prompts", target_user_id=target_user_id)
        self.selected_category = MOD_PROMPT_CATEGORIES[0][0]
        self._build_view()

    def _entries_for_category(self):
        return next((entries for cat, entries in MOD_PROMPT_CATEGORIES if cat == self.selected_category),
                    MOD_PROMPT_CATEGORIES[0][1])

    def _build_view(self):
        self.clear_items()

        # Category select (row 0). There are more prompts than one Discord select
        # can hold, so the list is grouped rather than paginated -- a prompt keeps
        # a stable home instead of drifting between pages as entries are added.
        cat_opts = []
        for cat, entries in MOD_PROMPT_CATEGORIES:
            n_custom = sum(1 for _l, k, _d in entries if k in self.cog.global_prompts)
            cat_opts.append(discord.SelectOption(
                label=cat,
                value=cat,
                description=f"{len(entries)} prompt(s), {n_custom} customised",
                default=(cat == self.selected_category),
            ))
        async def cat_cb(i: discord.Interaction):
            self.selected_category = i.data['values'][0]
            self._build_view()
            await i.response.edit_message(embed=self._get_embed(), view=self)
        add_select(self, cat_opts, cat_cb, placeholder="Select a category...", row=0)

        # Prompt select (row 1)
        entries = self._entries_for_category()
        options = []
        for lbl, key, _default in entries:
            is_overridden = key in self.cog.global_prompts
            options.append(discord.SelectOption(
                label=lbl[:100],
                value=key,
                description="Customised" if is_overridden else "Default",
                emoji="✏️" if is_overridden else None,
            ))

        async def sel_cb(i: discord.Interaction):
            key = i.data['values'][0]
            default_text = next(d for _l, k, d in MOD_PROMPT_DEFINITIONS if k == key)
            current = self.cog.global_prompts.get(key, default_text)

            # A stored prompt longer than the modal can hold could only arrive by
            # hand-editing system_prompts.json; opening it anyway would silently
            # truncate the tail on save.
            if len(current) > MODAL_PROMPT_CAPACITY:
                await i.response.send_message(
                    f"❌ `{key}` is {len(current):,} characters, which exceeds the "
                    f"{MODAL_PROMPT_CAPACITY:,}-character limit a Discord modal can hold. "
                    "Shorten it in `mod/system_prompts.json` before editing it here.",
                    ephemeral=True)
                return

            await i.response.send_modal(ModPromptModal(self, key, default_text))
        add_select(self, options, sel_cb, placeholder="Select a prompt to edit...", row=1)

        async def models_cb(i: discord.Interaction):
            view = ModSystemModelsView(self.cog, self.original_interaction, target_user_id=self.target_user_id)
            await i.response.edit_message(**view._picker_render())
        add_button(self, "System Models\u2026", models_cb, style=discord.ButtonStyle.primary, row=2)
        self._add_nav_buttons()

    def _get_embed(self):
        overridden = [lbl for lbl, key, _ in MOD_PROMPT_DEFINITIONS if key in self.cog.global_prompts]
        embed = discord.Embed(
            title="Global System Prompts",
            description=("Modify the internal hardcoded instructions. Leave a prompt completely "
                         "blank to revert to its default value.\n"
                         "-# The models no profile chooses are under **System Models…**."),
            color=discord.Color.purple(),
        )

        entries = self._entries_for_category()
        lines = []
        for lbl, key, default in entries:
            current = self.cog.global_prompts.get(key)
            mark = "✏️" if current is not None else "▫️"
            lines.append(f"{mark} **{lbl}** — {len(current if current is not None else default):,} chars")
        embed.add_field(name=self.selected_category, value="\n".join(lines), inline=False)

        embed.add_field(
            name=f"Customised overall ({len(overridden)}/{len(MOD_PROMPT_DEFINITIONS)})",
            value=(", ".join(overridden) if overridden
                   else "None — every prompt is running its built-in default."),
            inline=False,
        )
        return embed

    async def update_display(self):
        await self.original_interaction.edit_original_response(embed=self._get_embed(), view=self)


# --- System Models ------------------------------------------------------------

def _slots(*pairs) -> tuple:
    """(key, wording) pairs as the picker's triples. The shipped value depends on whose
    chain is shown, so it is `ModSystemModelsView._shipped`'s to answer, not the table's."""
    return tuple((key, wording, None) for key, wording in pairs)


#: A System profile's slots are that profile's own config; every other key is the file's.
_PROFILE_KEYS = ("primary_model", "fallback_model", "ollama_host_url")

#: The categories shipping a chain per provider preference, which the audience button switches.
_BY_PROVIDER_CATEGORIES = ("describer", "classifier", "generator")

#: MimicAI Auto, one category per tier and slot: each is a Primary and a Fallback. Response's
#: tab is the tier's own `auto_<tier>`; the others follow it, as AUTO_TIER_TABS names them.
#: (tab, wording, description, tier, slot).
_AUTO_SLOT_TABS = {
    "grounding_rag_model": ("Grounding", "The researcher behind RAG web searches."),
    "critic_model": ("Critic", "Screens replies for semantic repetition."),
    "ltm_model": ("LTM", "Turns conversations into long-term memories."),
}
_AUTO_TABS = tuple(
    tab for tier, _wording, desc in AUTO_TIERS
    for tab in ((AUTO_TIER_TABS[tier, "primary_model"], auto_wording(AUTO_MODEL_PREFIX + tier), desc,
                 tier, "primary_model"),
                *((AUTO_TIER_TABS[tier, slot], f"{auto_wording(AUTO_MODEL_PREFIX + tier)} · {wording}",
                   slot_desc, tier, slot) for slot, (wording, slot_desc) in _AUTO_SLOT_TABS.items())))
_AUTO_CATEGORIES = {tab: tier for tab, _w, _d, tier, _slot in _AUTO_TABS}
_AUTO_KEYS = frozenset(key for keys in AUTO_TIER_KEYS.values() for key in keys)
_AUTO_NOTE = ("OpenRouter models only. Every profile on this tier runs your choice from its next "
              "reply, sent with this tier's Hosts & Tier rather than its own, and any Thinking set "
              "here in place of its own.")
_AUTO_UNSET_NOTE = (" A Primary of None stops offering the tier, and refuses the profiles already "
                    "on it until one is set.")
_AUTO_SLOT_NOTES = {"grounding_rag_model": " Pick a model that calls the web search: one that "
                                           "answers without searching does so silently."}

#: Why a tab greys its Thinking or Hosts & Tier: what its models are sent carries neither.
#: An embedding has no reasoning, and its OpenRouter hosts are chosen per task type
#: (`embeddings._OPENROUTER_TASKS`), which a pin would override. A key check is a one-token
#: ping on Google's API with the pasted key.
_NO_THINKING = {"embedding": "Thinking: embeddings take none",
                "key_check": "Thinking: a key check sends none"}
_NO_HOSTS = {"embedding": "Hosts & Tier: fixed per task for embeddings",
             "key_check": "Hosts & Tier: Google only"}


async def _noop(interaction: discord.Interaction):
    await interaction.response.defer()


def _budget_wording(budget, short: bool = False) -> str:
    return ("Dyn" if short else "Dynamic") if budget == -1 else str(budget)


#: What each category tells the operator before they choose.
_SYSTEM_MODEL_NOTES = {
    "describer": "Pick models that read images -- and audio, for voice messages to be described.",
    "classifier": ("Sees the persona and the avatar, so pick models that read images. Runs on the "
                   "profile owner's own key alone, never yours: a user with none is not rated."),
    "generator": ("Writes `/profile generate` drafts and sorts imported character cards, on the "
                  "user's own key. A user's Override Defaults for Response win over these."),
    "embedding": ("⚠️ Changing this degrades recall of **every existing memory and "
                  "training example**. A Google model also runs through OpenRouter's `google/` "
                  "copy; any other OpenRouter model runs on OpenRouter keys only."),
    "system": "Saved to the System profile itself. Its other settings are in `/profile`.",
    "key_check": ("Called on Google's API with the pasted key, so Google models only. The billing "
                  "check must be one an unbilled key is refused: an image model."),
    **{tab: _AUTO_NOTE + _AUTO_SLOT_NOTES.get(slot, "")
            + (" New profiles on OpenRouter start on this tier." if tier == "free" else _AUTO_UNSET_NOTE)
       for tab, _w, _d, tier, slot in _AUTO_TABS},
}


class ModSystemModelsView(BlockedGuard, TimeoutCleanupMixin, ModelPickerMixin, ui.View):
    """The models no profile chooses, as Set Models presents a profile's.

    Opened from the Prompts tab onto the same message, with a Back, as Thinking opens from
    Set Models: /mod's nav bar is a full row of five, and a picker needs every row it has.
    """

    _CATEGORY_LABELS = (
        ("describer", "Attachment Describer", "Reads an attachment for a profile whose models cannot."),
        ("classifier", "Content Classifier", "Rates every profile General or 18+."),
        ("generator", "Profile Generator", "Drafts profiles, and sorts imported character cards."),
        ("embedding", "Embeddings", "The vectors behind memory, training-example and /help recall."),
        ("system", "System Profiles", "The models behind each System profile's replies."),
        ("key_check", "Key Checks", "Tests a pasted Gemini key, then whether it has billing."),
        *((tab, wording, desc) for tab, wording, desc, _tier, _slot in _AUTO_TABS),
    )
    _CATEGORY_KEYS = {
        "describer": _slots(("describer_model", "Primary"), ("describer_fallback_model", "Fallback")),
        "classifier": _slots(("classifier_model", "Primary"), ("classifier_fallback_model", "Fallback")),
        "generator": _slots(("generator_model", "Primary"), ("generator_fallback_model", "Fallback")),
        "embedding": _slots(("embedding_model", "Embedding Model")),
        "system": _slots(("primary_model", "Primary"), ("fallback_model", "Fallback")),
        "key_check": _slots(("key_check_model", "Validity Check"),
                            ("key_tier_probe_model", "Billing Check")),
        **{tab: _slots(*zip(AUTO_TIER_KEYS[tier, slot], ("Primary", "Fallback")))
           for tab, _w, _d, tier, slot in _AUTO_TABS},
    }
    #: Everything but the System profiles runs for every user, and Ollama answers the bot
    #: owner's own profiles only -- which the System profiles are.
    _NO_OLLAMA_CATEGORIES = ("describer", "classifier", "generator", "embedding", "key_check")
    _NO_RETRY_KEYS = ("describer_fallback_model", "classifier_fallback_model", "generator_fallback_model",
                      *(key for (tier, _slot), keys in AUTO_TIER_KEYS.items()
                        for key in keys if tier != "free" or key != keys[0]))
    #: No tab has a row to spare for a browse list: two slots and two button rows fill
    #: Discord's five, the System Profiles tab spends one on choosing its profile, and no
    #: list holds embedding models. The model list pages on its own.
    _BROWSE_ROW_AVAILABLE = False

    is_borrowed = False

    def __init__(self, cog, interaction: discord.Interaction, target_user_id: Optional[int] = None):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.target_user_id = target_user_id
        self.view_mode = self.preferred_api(cog, self.user_id)
        #: Whose chain the describer and classifier tabs show: the users preferring Google,
        #: or OpenRouter. Opens on the operator's own side.
        self.audience = "openrouter" if self.view_mode == "openrouter" else "gemini"
        self.category = "describer"
        self.ollama_working = None
        #: An embedding model waiting on the warning, or None.
        self.pending_embedding: Optional[str] = None
        #: The System profile the System Profiles tab edits; OllamaHostModal reads it too.
        self.profile_name = "mimicguide"
        self.cog.profile_manager._get_or_create_system_profile(self.profile_name)
        self._build_view()

    # --- Where a slot's value lives ------------------------------------------

    def _system_profiles(self) -> List[str]:
        # ponytail: the first 25 by name, one dropdown's worth; page them if an instance
        # ever holds more.
        return sorted(self.cog.profile_manager._system_index())[:25]

    def _system_config(self) -> Dict:
        return self.cog.profile_manager._get_profile_config(self.user_id, self.profile_name, False) or {}

    def _value(self, key: str) -> str:
        if key in _PROFILE_KEYS:
            # An unset slot runs what the owner's provider ships, as `model_chain` reads it.
            return self._system_config().get(key) or model_slot_defaults(
                self.cog.profile_manager.provider_preference(self.user_id)).get(key)
        return system_model(self.cog, key, self.audience)

    def _shipped(self, key: str) -> str:
        """What Reset puts back: a new System profile's model, else the build's for this audience."""
        if key in _PROFILE_KEYS:
            return SYSTEM_PROFILE_FALLBACK_MODEL if key == "fallback_model" else SYSTEM_PROFILE_MODEL
        return SYSTEM_MODEL_DEFAULTS.get(key) or SYSTEM_MODEL_DEFAULTS_BY_PROVIDER[self.audience][key]

    def _save_changes(self, key: str, value):
        if key in _PROFILE_KEYS:
            config = self._system_config()
            config[key] = value
            # A pin goes when its model leaves the profile, as Set Models prunes one.
            if "openrouter_endpoints" in config:
                prune_openrouter_endpoints(config)
            self.cog.profile_manager._save_profile_config(self.user_id, self.profile_name, config, False)
            invalidate_model_cache(self.cog, self.user_id, self.profile_name)
        elif value == self._value(key):
            return
        elif key == "embedding_model":
            # Not written until the warning is confirmed -- see `_build_view`.
            self.pending_embedding = value
        else:
            self._store(key, value)

    def _store(self, key: str, value: str):
        """Sparse: the shipped value is no override, so the file holds only what differs --
        a per-preference key under its audience's name, dropped with the last of them."""
        shared = key in SYSTEM_MODEL_DEFAULTS
        section = self.cog.system_models if shared else self.cog.system_models.setdefault(self.audience, {})
        if value == self._shipped(key):
            section.pop(key, None)
        else:
            section[key] = value
        if not shared and not section:
            self.cog.system_models.pop(self.audience, None)
        self.cog.server_manager._save_system_models()
        if key in _AUTO_KEYS:
            self._store_auto_pins(self.cog.system_models.get(AUTO_ENDPOINTS_KEY) or {})
        elif self.category in _BY_PROVIDER_CATEGORIES:
            self._prune_pins()

    # --- A tab's own Thinking and Hosts & Tier ---------------------------------

    def _settings(self) -> Dict:
        """This tab's thinking and Hosts & Tier, under a profile's key names: the System
        profile's own config, else the file's entry for the tab -- under its audience's name
        where the tab ships a chain per preference. Read only; `_put_settings` writes."""
        if self.category == "system":
            return self._system_config()
        return system_settings(self.cog, self.category,
                               self.audience if self.category in _BY_PROVIDER_CATEGORIES else None)

    def _put_settings(self, changes: Dict):
        """Applies `changes` to this tab's settings, sparse as `_store` is: None or an empty
        value is no key, a tab left with none is no entry, and no change writes nothing."""
        system = self.category == "system"
        target = self._system_config() if system else dict(self._settings())
        before = dict(target)
        for key, value in changes.items():
            if value is None or value == "" or value == {}:
                target.pop(key, None)
            else:
                target[key] = value
        if target == before:
            return
        if system:
            self.cog.profile_manager._save_profile_config(self.user_id, self.profile_name, target, False)
            invalidate_model_cache(self.cog, self.user_id, self.profile_name)
            return
        files = self.cog.system_models
        side = self.audience if self.category in _BY_PROVIDER_CATEGORIES else None
        section = dict(files.get(side) or {}) if side else files
        if target:
            section[self.category] = target
        else:
            section.pop(self.category, None)
        if side and section:
            files[side] = section
        elif side:
            files.pop(side, None)
        self.cog.server_manager._save_system_models()

    def _prune_pins(self):
        """Drops a pin whose model has left this tab, as `prune_openrouter_endpoints` does a
        profile's."""
        pins = self._settings().get("openrouter_endpoints") or {}
        held = set(self._pinnable_openrouter_models())
        if any(model not in held for model in pins):
            self._put_settings({"openrouter_endpoints": {m: t for m, t in pins.items() if m in held}})

    def _tab_wording(self) -> str:
        wording = next(l for v, l, _d in self._CATEGORY_LABELS if v == self.category)
        if self.category in _BY_PROVIDER_CATEGORIES:
            return f"{wording} \u00b7 {MODEL_PROVIDERS[self.audience]} users"
        return wording

    def _tab_footer(self) -> str:
        if self.category == "system":
            return f"{self.profile_name} \u00b7 changes save as you make them"
        return f"Bot-wide \u00b7 {self._tab_wording()} \u00b7 changes save as you make them"

    def _thinking_line(self) -> Optional[str]:
        """The tab's thinking, Primary then Fallback, as it runs; None where it takes none."""
        if self.category in _NO_THINKING:
            return None
        settings, parts = self._settings(), []
        for role, label in (("primary", "Primary"), ("fallback", "Fallback")):
            if self._auto_tier():
                level_key, budget_key = THINKING_SLOT_KEYS["response"][role]
                level, budget = settings.get(level_key), settings.get(budget_key)
                parts.append(f"{label} `Each profile's own`" if level is None and budget is None else
                             f"{label} `{level.title() if level else 'Own'}`/"
                             f"`{'Own' if budget is None else _budget_wording(budget, short=True)}`")
                continue
            if self.category == "system":
                r = resolve_thinking_params(settings, "response", role, cog=self.cog)
            else:
                r = resolve_thinking_params(settings, self.category, role, cog=self.cog)
            parts.append(f"{label} `{r['thinking_level'].title()}`/"
                         f"`{_budget_wording(r['thinking_budget'], short=True)}`")
        return " \u2192 ".join(parts)

    def _add_thinking_button(self, row: int):
        """Thinking, on every tab -- greyed where the tab's models take none. The System
        profiles open their own, every slot of them, as Set Models does."""
        if self.category in _NO_THINKING:
            add_button(self, _NO_THINKING[self.category], _noop, row=row, disabled=True)
            return

        async def think_cb(i: discord.Interaction):
            if self.category == "system":
                view = SingleProfileThinkingView(self.cog, self.original_interaction, self.profile_name,
                                                 user_id=self.user_id, parent_model_view=self)
            else:
                view = ModThinkingView(self)
            await i.response.edit_message(**view._render())

        lit = self.category != "system" and any(
            self._settings().get(key) is not None
            for pair in THINKING_SLOT_KEYS["response"].values() for key in pair)
        add_button(self, "Thinking\u2026", think_cb, row=row,
                   style=discord.ButtonStyle.blurple if lit else discord.ButtonStyle.secondary)

    # --- MimicAI Auto's Hosts & Tier -------------------------------------------

    def _auto_tier(self) -> Optional[str]:
        return _AUTO_CATEGORIES.get(self.category)

    def _put(self, key: str, value: Dict):
        """Sparse, as `_store`: an empty mapping is no entry."""
        if value:
            self.cog.system_models[key] = value
        else:
            self.cog.system_models.pop(key, None)
        self.cog.server_manager._save_system_models()

    def _auto_pins(self) -> Dict[str, str]:
        pins = self.cog.system_models.get(AUTO_ENDPOINTS_KEY)
        return dict(pins) if isinstance(pins, dict) else {}

    def _store_auto_pins(self, pins: Dict[str, str]):
        """Kept only for a model some tier still runs, as `prune_openrouter_endpoints` keeps a
        profile's, so a model swapped out does not leave its pin behind."""
        held = {str(system_model(self.cog, key)).removeprefix("OPENROUTER/") for key in _AUTO_KEYS}
        self._put(AUTO_ENDPOINTS_KEY, {m: tag for m, tag in pins.items() if m in held})

    def _pinnable_slots(self) -> List[tuple]:
        return [(key, wording, self._value(key)) for key, wording, _d in self._CATEGORY_KEYS[self.category]]

    def _current_service_tier(self):
        if not self._auto_tier():
            return self._settings().get("openrouter_service_tier") or ""
        return (self.cog.system_models.get(AUTO_SERVICE_TIERS_KEY) or {}).get(self._auto_tier()) or ""

    def _set_service_tier(self, value):
        if not self._auto_tier():
            self._put_settings({"openrouter_service_tier": value})
            return
        tiers = dict(self.cog.system_models.get(AUTO_SERVICE_TIERS_KEY) or {})
        if value:
            tiers[self._auto_tier()] = value
        else:
            tiers.pop(self._auto_tier(), None)
        self._put(AUTO_SERVICE_TIERS_KEY, tiers)

    def _add_openrouter_buttons(self, *, row: int):
        """Hosts & Tier, on every tab -- greyed where the tab's models take none. Auto's are
        the owner's per tier; every other tab's are its own (ModHostView)."""
        if self.category in _NO_HOSTS:
            add_button(self, _NO_HOSTS[self.category], _noop, row=row, disabled=True)
            return
        auto = self._auto_tier() is not None
        pins = {"openrouter_endpoints": self._auto_pins()} if auto else self._settings()
        lit = bool(self._current_service_tier()) or any(
            resolve_openrouter_endpoint(pins, m) for m in self._pinnable_openrouter_models())
        self._add_hosts_button(row=row, view_cls=ModAutoHostView if auto else ModHostView, lit=lit)

    def _is_changed(self, key: str) -> bool:
        return self._value(key) != self._shipped(key)

    # --- Mixin contract ------------------------------------------------------

    def _ollama_host_url(self) -> Optional[str]:
        return self._system_config().get("ollama_host_url")

    def _get_selection_feedback_message(self) -> str:
        """Unused -- this view renders an embed -- but named by the mixin."""
        return ""

    def _api_modes(self) -> List[str]:
        # The key checks are raw calls to Google's API with the key being checked, and
        # MimicAI Auto runs on OpenRouter keys.
        if self.category == "key_check":
            return ["google"]
        return ["openrouter"] if self._auto_tier() else super()._api_modes()

    def _allows_no_fallback(self, target_config_key: str) -> bool:
        return target_config_key in self._NO_RETRY_KEYS

    def _tier_applies(self) -> bool:
        """Never on the API row: every tab carries Hosts & Tier beside its Thinking, whatever
        API it is browsing, so `_build_view` adds it."""
        return False

    def _get_top_models(self, provider: str, target_config_key: str) -> List[str]:
        if target_config_key == "key_tier_probe_model":
            return list(get_args(IMAGE_MODELS))
        return super()._get_top_models(provider, target_config_key)

    def _create_model_options(self, current_val: str, target_config_key: str) -> List[discord.SelectOption]:
        if target_config_key != "embedding_model":
            return super()._create_model_options(current_val, target_config_key)
        # Neither tab's list is embedding models: typed, or the shipped one.
        opts = [discord.SelectOption(label="Custom Model...", value="custom_option",
                                     description="Enter manually via modal"),
                discord.SelectOption(label=f"Current: {self.strip_prefix(current_val)}"[:100],
                                     value=current_val, default=True)]
        if current_val != EMBEDDING_MODEL_NAME:
            opts.append(discord.SelectOption(
                label=f"Shipped: {self.strip_prefix(EMBEDDING_MODEL_NAME)}"[:100],
                value=EMBEDDING_MODEL_NAME))
        return opts

    def refuse_custom_model(self, key: str, value: str) -> Optional[str]:
        """What CustomModelModal asks before the profile rules. See `_NO_OLLAMA_CATEGORIES`."""
        if key in _PROFILE_KEYS:
            return None
        if value.startswith("OLLAMA/"):
            return ("Ollama can't hold a system model: these run for every user, and Ollama "
                    "answers the bot owner's own profiles only.")
        if key in _AUTO_KEYS and not value.startswith("OPENROUTER/"):
            return "MimicAI Auto runs on OpenRouter keys, so its tiers take OpenRouter models."
        if key in ("key_check_model", "key_tier_probe_model") and not value.startswith("GOOGLE/"):
            return "The key checks call Google's API with the pasted key, so they take a Google model."
        return None

    # --- Rendering -----------------------------------------------------------

    def _blind_models(self) -> List[str]:
        """This tab's OpenRouter models the catalogue says cannot read an image."""
        if self.category not in ("describer", "classifier"):
            return []
        blind = []
        for key, _wording, _default in self._CATEGORY_KEYS[self.category]:
            value = self._value(key)
            if not str(value).startswith("OPENROUTER/"):
                continue
            info = self.cog.api_service.catalogue.models.get(self.strip_prefix(value))
            if info is not None and not info.image_input:
                blind.append(self.strip_prefix(value))
        return blind

    def embed(self) -> discord.Embed:
        if self.pending_embedding is not None:
            return discord.Embed(
                title="Change the Embedding Model?", colour=discord.Colour.red(),
                description=(
                    f"`{self.strip_prefix(self._value('embedding_model'))}` → "
                    f"`{self.strip_prefix(self.pending_embedding)}`\n\n"
                    "Every memory and training example on this bot was embedded by the current "
                    "model, and a stored vector does not record which model made it. After the "
                    "change, recall compares new queries against old vectors and returns poor "
                    "matches **with no error**, until each one is saved again. Nothing "
                    "re-embeds them.\n\n"
                    "The `/help` documentation re-embeds itself straight away, on your key."))

        wording, description = next((l, d) for v, l, d in self._CATEGORY_LABELS if v == self.category)
        whose = ""
        if self.category in _BY_PROVIDER_CATEGORIES:
            whose = (f"Showing the chain for profiles whose owner prefers "
                     f"**{MODEL_PROVIDERS[self.audience]}**.\n")
        elif self.category == "system":
            whose = f"Editing **{self.profile_name}**.\n"
        e = discord.Embed(title="System Models", colour=discord.Colour.purple(),
                          description=f"**{wording}** — {description}\n{whose}"
                                      f"-# {_SYSTEM_MODEL_NOTES[self.category]}")
        for key, slot, _default in self._CATEGORY_KEYS[self.category]:
            mark = " ✏️" if self._is_changed(key) else ""
            e.add_field(name=f"{slot}{mark}", value=f"`{self.display_model(self._value(key))}`", inline=True)
        thinking = self._thinking_line()
        if thinking:
            e.add_field(name="Thinking", value=thinking, inline=False)
        blind = self._blind_models()
        if blind:
            e.add_field(name="⚠️ Cannot read images", inline=False,
                        value=", ".join(f"`{m}`" for m in blind) + " -- the chain passes over "
                              f"{'them' if len(blind) > 1 else 'it'} whenever there is a file to read.")
        self._add_openrouter_details(e, [(slot, self._value(key))
                                         for key, slot, _d in self._CATEGORY_KEYS[self.category]])
        e.set_footer(text="Bot-wide · ✏️ differs from the shipped model · saves as you choose")
        return e

    def _add_system_profile_select(self, row: int):
        names = self._system_profiles()
        if self.profile_name not in names:
            # Converted back to a personal profile, or deleted, since this screen opened.
            self.profile_name = "mimicguide" if "mimicguide" in names else names[0]
        select = ui.Select(placeholder="Choose a System profile...", row=row, options=[
            discord.SelectOption(label=f"Profile: {name}"[:100], value=name,
                                 default=(name == self.profile_name)) for name in names])

        async def callback(i: discord.Interaction):
            self.profile_name = select.values[0]
            # Each profile names its own Ollama host.
            self.ollama_working = None
            self._build_view()
            await i.response.edit_message(**self._picker_render())

        select.callback = callback
        self.add_item(select)

    def _build_view(self):
        self.clear_items()

        if self.pending_embedding is not None:
            async def confirm_cb(i: discord.Interaction):
                self._store("embedding_model", self.pending_embedding)
                self.pending_embedding = None
                # As a Docs edit does: the documentation is the one archive that can be
                # rebuilt from its source.
                self.cog.bot.loop.create_task(self.cog.help_service._load_and_embed_docs())
                self._build_view()
                await i.response.edit_message(**self._picker_render())

            async def cancel_cb(i: discord.Interaction):
                self.pending_embedding = None
                self._build_view()
                await i.response.edit_message(**self._picker_render())

            add_button(self, "Change Embedding Model", confirm_cb, style=discord.ButtonStyle.danger, row=0)
            add_button(self, "Cancel", cancel_cb, row=0)
            return

        if self.view_mode not in self._api_modes():
            self.view_mode = self._api_modes()[0]

        self._add_category_select(0)
        row = 1
        if self.category == "system":
            self._add_system_profile_select(row)
            row += 1
        slots = self._CATEGORY_KEYS[self.category]
        for key, wording, _default in slots:
            self.add_item(self.GenericModelSelect(
                f"Select {wording} Model...", self._create_model_options(self._value(key), key), row, key))
            row += 1

        # Every tab but System Profiles has two rows left: the API row, then the tab's own
        # Thinking and Hosts & Tier with Reset and Back. System Profiles has one for all of
        # them, so its Ollama tab leaves off Hosts & Tier, which pins OpenRouter models, and
        # the Ollama Guide, which the profile's own Set Models offers, to fit Discord's five.
        last_row = row if self.category == "system" else row + 1
        crowded = self.category == "system" and self.view_mode == "ollama"

        if self.category in _BY_PROVIDER_CATEGORIES:
            async def audience_cb(i: discord.Interaction):
                self.audience = other_provider(self.audience)
                # The models a side ships are that side's, so the list follows it.
                self.view_mode = "openrouter" if self.audience == "openrouter" else "google"
                self._build_view()
                await i.response.edit_message(**self._picker_render())

            add_button(self, f"For: {MODEL_PROVIDERS[self.audience]} users", audience_cb,
                       style=discord.ButtonStyle.success, row=row)

        self._add_api_buttons(row=row, guide=not crowded)
        self._add_thinking_button(last_row)
        if not crowded:
            self._add_openrouter_buttons(row=last_row)

        async def reset_cb(i: discord.Interaction):
            # Through `_save_changes`, so resetting the embedding model is warned about too.
            for key, _wording, _default in slots:
                self._save_changes(key, self._shipped(key))
            self._build_view()
            await i.response.edit_message(**self._picker_render())

        add_button(self, "Reset to Shipped", reset_cb, row=last_row,
                   disabled=not any(self._is_changed(key) for key, _w, _d in slots))

        async def back_cb(i: discord.Interaction):
            view = ModPromptsView(self.cog, self.original_interaction, target_user_id=self.target_user_id)
            await i.response.edit_message(embed=view._get_embed(), view=view)

        add_button(self, "Back", back_cb, row=last_row)


class ModThinkingView(BlockedGuard, ThinkingPickerMixin, ui.View):
    """Thinking for one System Models tab, bot-wide: how hard its Primary and its Fallback
    reason. Opened onto the same message with a Back, as from Set Models.

    Kept under the response slot's keys in the tab's own settings (`helpers.system_settings`).
    A describer, classifier or generator tab resolves them as its own slot, whose shipped
    default an unset key reads. An Auto tab's stand in for each profile's own on that tier
    (`helpers.auto_thinking`), so unset is "Unchanged": the profile keeps its own.
    """

    def __init__(self, parent: ModSystemModelsView):
        super().__init__(timeout=600)
        self.parent = parent
        self.cog = parent.cog
        self.user_id = parent.user_id
        self.slot = parent.category
        self._OFFERS_UNCHANGED = parent._auto_tier() is not None
        self._build_view()

    def _slot_keys(self, role: str = "primary"):
        return THINKING_SLOT_KEYS["response"][role]

    def _slot_model(self, _slot: str, role: str = "primary"):
        keys = self.parent._CATEGORY_KEYS[self.parent.category]
        return self.parent._value(keys[role == "fallback"][0])

    def _current(self, key: str):
        return self.parent._settings().get(key)

    def _write(self, key: str, value):
        self.parent._put_settings({key: value})

    def _render(self) -> Dict:
        return {"content": None, "embed": self.embed(), "view": self}

    def _all_keys(self) -> List[str]:
        return [key for role, _label in self._ROLE_LABELS for key in self._slot_keys(role)]

    def embed(self) -> discord.Embed:
        parent, settings = self.parent, self.parent._settings()
        auto = self._OFFERS_UNCHANGED
        e = discord.Embed(title="Thinking / Reasoning", colour=discord.Colour.purple())
        e.description = f"**{parent._tab_wording()}** \u2014 " + (
            "set here, each runs in place of a profile's own on this tier; left Unchanged, the "
            "profile keeps its own." if auto else
            "effort and budget are stored per model, so a cheap fallback need not think as hard "
            "as the primary it stands in for.")
        own = "`Each profile's own`"
        for role, label in self._ROLE_LABELS:
            level_key, budget_key = self._slot_keys(role)
            if auto:
                level, budget = settings.get(level_key), settings.get(budget_key)
                effort = f"`{level.title()}`" if level else own
                spend = own if budget is None else f"`{_budget_wording(budget)}`"
            else:
                r = resolve_thinking_params(settings, self.slot, role, cog=self.cog)
                unset = " *(shipped)*" if role == "primary" else " *(inherited)*"
                effort = f"`{r['thinking_level'].title()}`" + ("" if settings.get(level_key) else unset)
                spend = (f"`{_budget_wording(r['thinking_budget'])}`"
                         + ("" if settings.get(budget_key) is not None else unset))
            model = self._slot_model(self.slot, role)
            note = f"This model {self._caps_note(role)}." if is_real_model(model) else "No model set."
            e.add_field(name=f"{label} \u2014 {parent.display_model(model)}"[:256], inline=False,
                        value=f"Effort: {effort}\nBudget: {spend}\n-# {note}")
        e.set_footer(text=parent._tab_footer())
        return e

    def _build_view(self):
        self.clear_items()
        self._add_level_select("primary", 0)
        self._add_level_select("fallback", 1)
        self._add_budget_button(2)

        async def reset_cb(i: discord.Interaction):
            self.parent._put_settings(dict.fromkeys(self._all_keys()))
            self._build_view()
            await i.response.edit_message(**self._render())

        add_button(self, "Reset to Profiles' Own" if self._OFFERS_UNCHANGED else "Reset to Shipped",
                   reset_cb, row=2, disabled=all(self._current(k) is None for k in self._all_keys()))

        async def back_cb(i: discord.Interaction):
            self.parent._build_view()
            await i.response.edit_message(**self.parent._picker_render())

        add_button(self, "\u2190 Back to Models", back_cb, row=2)


class ModHostView(OpenRouterHostView):
    """Hosts & Tier for a System Models tab other than MimicAI Auto's: its own pins and Default
    Tier, kept as a profile keeps them -- on the System profile itself, or in the file under
    the tab (`helpers.system_settings`), where the describer, classifier and generator read
    them at the call."""

    @property
    def _TIER_REACH(self) -> str:
        if self.parent.category == "system":
            return OpenRouterHostView._TIER_REACH
        return "Every OpenRouter model on this tab, for every user it runs for."

    def _chosen(self, model_id: str) -> str:
        return resolve_openrouter_endpoint(self.parent._settings(), model_id) or self._AUTO

    def _choose(self, model_id: str, chosen: str):
        pins = dict(self.parent._settings().get("openrouter_endpoints") or {})
        if chosen == self._AUTO:
            pins.pop(model_id, None)
        else:
            pins[model_id] = chosen
        self.parent._put_settings({"openrouter_endpoints": pins})

    def _footer(self) -> str:
        return self.parent._tab_footer()


class ModAutoHostView(OpenRouterHostView):
    """Hosts & Tier for one MimicAI Auto tier, bot-wide: what every profile on it is sent
    with. Pins are the owner's, keyed by model id across the tiers as a profile's are
    across its slots, and read at the call by `helpers.openrouter_routing`."""

    _TIER_REACH = "Both of this tier's models, for every profile on it, wherever a model is left on Auto."

    def _chosen(self, model_id: str) -> str:
        return (resolve_openrouter_endpoint({"openrouter_endpoints": self.parent._auto_pins()}, model_id)
                or self._AUTO)

    def _choose(self, model_id: str, chosen: str):
        pins = self.parent._auto_pins()
        if chosen == self._AUTO:
            pins.pop(model_id, None)
        else:
            pins[model_id] = chosen
        self.parent._store_auto_pins(pins)

    def _footer(self) -> str:
        return (f"Bot-wide \u00b7 {auto_wording(AUTO_MODEL_PREFIX + self.parent._auto_tier())} "
                "\u00b7 changes save as you make them")
