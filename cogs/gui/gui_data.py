from ..utils.constants import *

import discord
from discord import ui
import datetime
import traceback
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional
from ..utils.helpers import _resolve_zoneinfo, suppress_link_previews
from ..managers.memory_manager import (EMBEDDING_FAILED_MSG, NO_EMBEDDING_KEY_MSG,
                                       encode_embedding_b64, entry_date)

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog
    from .gui_profiles import ProfileManageView

from .base_components import (PAGE_NAV_VALUES, BaseBulkProfileView, BlockedGuard, PageJumpModal, add_button,
                              add_select, build_confirm_view, paged_nav_options)
from .gui_data_policy import DataPolicyView, add_data_policy_fields
from ..utils.data_policy import may_set_data_policy

class EditLtmModal(ui.Modal, title="Edit Long-Term Memory"):
    summary_field = ui.TextInput(label="Memory Summary", style=discord.TextStyle.paragraph, required=True, max_length=2000)

    def __init__(self, cog, profile_owner_id: int, profile_name: str, ltm_id: str, current_summary: str):
        super().__init__()
        self.cog: MimicCog = cog
        self.profile_owner_id = profile_owner_id
        self.profile_name = profile_name
        self.ltm_id = ltm_id
        self.summary_field.default = current_summary

    async def on_submit(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True, thinking=True)
        new_summary = self.summary_field.value
        
        # No guild hunt. This used to walk bot.guilds for any server the *invoker*
        # shares with the bot, which says nothing about whether that server has a key
        # assigned -- so in a DM it reliably picked a keyless guild and reported the
        # miss as a broken embedding. The owner's own key is the answer instead.
        if not self.cog.storage_manager._embedding_routes(i.guild_id, self.profile_owner_id):
            await i.followup.send(NO_EMBEDDING_KEY_MSG, ephemeral=True)
            return

        new_embedding = await self.cog.memory_manager._get_embedding(
            new_summary, i.guild_id, task_type="RETRIEVAL_DOCUMENT", owner_id=self.profile_owner_id)
        if not new_embedding:
            await i.followup.send(EMBEDDING_FAILED_MSG, ephemeral=True)
            return

        b64_emb = encode_embedding_b64(new_embedding)
        success = await self.cog.memory_manager.update_ltm(self.profile_owner_id, self.profile_name, self.ltm_id, new_summary, b64_emb)
        if success:
            await i.followup.send(f"LTM entry `{self.ltm_id}` for profile '{self.profile_name}' has been updated.", ephemeral=True)
        else:
            await i.followup.send(f"Failed to find and update LTM entry `{self.ltm_id}`.", ephemeral=True)
    
    async def on_error(self, i: discord.Interaction, e: Exception):
        print(f"EditLtmModal error: {e}"); traceback.print_exc()
        await i.followup.send("An error occurred with the LTM edit form.", ephemeral=True)

class AddLtmModal(ui.Modal, title="Add Long-Term Memory"):
    summary_field = ui.TextInput(label="Memory Summary", style=discord.TextStyle.paragraph, required=True, max_length=2000)

    def __init__(self, cog, profile_owner_id: int, profile_name: str, guild_id: Optional[int]):
        super().__init__()
        self.cog: MimicCog = cog
        self.profile_owner_id = profile_owner_id
        self.profile_name = profile_name
        self.guild_id = guild_id

    async def on_submit(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True, thinking=True)

        # A memory is filed under the guild it formed in, and retrieval only ever looks
        # in the current guild's bucket -- so _add_ltm drops one with no guild rather
        # than store something nothing can recall. That drop was silent, and this modal
        # reported "LTM entry added" on top of it. Refuse up front instead: the work
        # below (a limit check, an embedding, a shard write) all leads nowhere here.
        if not self.guild_id:
            await i.followup.send(
                "A memory has to be filed under a server, because that is where it can "
                "be recalled. Run `/profile manage` in the server this memory belongs to "
                "and add it there.", ephemeral=True)
            return

        # Refused at the limit rather than rolled over: automatic creation drops the
        # oldest to make room, but a memory typed by hand should not silently cost one.
        limit = defaultConfig.LIMIT_LTM
        current_count = len(await self.cog.memory_manager.load_ltms(self.profile_owner_id, self.profile_name))
        if current_count >= limit:
            await i.followup.send(
                f"**Limit Reached.**\nYou have **{current_count}** memories (Limit: {limit}).\n"
                "You cannot manually add more memories while at or above the limit. "
                "Please delete old memories first.", ephemeral=True)
            return

        summary = self.summary_field.value
        
        if not self.cog.storage_manager._embedding_routes(self.guild_id, self.profile_owner_id):
            await i.followup.send(NO_EMBEDDING_KEY_MSG, ephemeral=True)
            return

        embedding = await self.cog.memory_manager._get_embedding(
            summary, self.guild_id, task_type="RETRIEVAL_DOCUMENT", owner_id=self.profile_owner_id)
        if not embedding:
            await i.followup.send(EMBEDDING_FAILED_MSG, ephemeral=True)
            return

        await self.cog.memory_manager._add_ltm(
            self.profile_owner_id, self.profile_name, summary, encode_embedding_b64(embedding),
            self.guild_id, i.user.id, i.user.display_name, source="manual")
        await i.followup.send(f"Memory added for '{self.profile_name}' "
                              f"({current_count + 1}/{limit}).", ephemeral=True)

    async def on_error(self, i: discord.Interaction, e: Exception):
        print(f"AddLtmModal error: {e}"); traceback.print_exc()
        await i.followup.send("An error occurred with the LTM add form.", ephemeral=True)

class AddTrainingExampleModal(ui.Modal, title="Add Profile Training Example"): 
    user_input_field=ui.TextInput(label="User Input Example",style=discord.TextStyle.paragraph,required=True,max_length=1000)
    chatbot_response_field=ui.TextInput(label="Desired Chatbot Response",style=discord.TextStyle.paragraph,required=True,max_length=2000)
    def __init__(self, cog, profile_owner_id: int, profile_name: str, guild_id: Optional[int]):
        super().__init__()
        self.cog:MimicCog=cog
        self.profile_owner_id = profile_owner_id
        self.profile_name = profile_name
        self.guild_id = guild_id
    async def on_submit(self,i:discord.Interaction):
        await i.response.defer(ephemeral=True,thinking=True)
        # add_new_training_example checks the limit itself, before the embedding call.
        s,m=await self.cog.memory_manager.add_new_training_example(self.profile_owner_id, self.profile_name, self.user_input_field.value, self.chatbot_response_field.value, self.guild_id)
        await i.followup.send(m,ephemeral=True)
    async def on_error(self,i:discord.Interaction,e:Exception):print(f"AddTrainExModal err:{e}");traceback.print_exc();await i.followup.send('Oops!',ephemeral=True)

class EditTrainingExampleModal(ui.Modal, title="Edit Profile Training Example"):
    user_input_field = ui.TextInput(label="User Input Example", style=discord.TextStyle.paragraph, required=True, max_length=1000)
    chatbot_response_field = ui.TextInput(label="Desired Chatbot Response", style=discord.TextStyle.paragraph, required=True, max_length=2000)

    def __init__(self, cog, profile_owner_id: int, profile_name: str, example_id: str, current_user_input: str, current_bot_response: str, guild_id: Optional[int]):
        super().__init__()
        self.cog: MimicCog = cog
        self.profile_owner_id = profile_owner_id
        self.profile_name = profile_name
        self.example_id = example_id
        self.guild_id = guild_id
        self.user_input_field.default = current_user_input
        self.chatbot_response_field.default = current_bot_response

    async def on_submit(self,i:discord.Interaction):
        await i.response.defer(ephemeral=True,thinking=True)
        s,m=await self.cog.memory_manager.update_training_example(self.profile_owner_id, self.profile_name, self.example_id, self.user_input_field.value, self.chatbot_response_field.value, self.guild_id)
        await i.followup.send(m,ephemeral=True)

    async def on_error(self, i: discord.Interaction, e: Exception):
        print(f"EditTrainingExampleModal error: {e}")
        traceback.print_exc()
        await i.followup.send("An error occurred with the edit form.", ephemeral=True)

class SearchDataModal(ui.Modal, title="Search"):
    search_input = ui.TextInput(label="Search term (leave blank to clear)", required=False, max_length=100)

    def __init__(self, parent_view: 'DataManageView'):
        super().__init__()
        self.parent_view = parent_view
        if self.parent_view.search_term:
            self.search_input.default = self.parent_view.search_term

    async def on_submit(self, interaction: discord.Interaction):
        self.parent_view.search_term = self.search_input.value.strip() or None
        self.parent_view.page = 0
        self.parent_view.open_id = None
        await self.parent_view._show(interaction)

class TestMatchModal(ui.Modal, title="Test"):
    phrase = ui.TextInput(label="Something someone might say", style=discord.TextStyle.paragraph,
                          required=True, max_length=500)

    def __init__(self, parent_view: 'DataManageView'):
        super().__init__(title="Test Recall" if parent_view.mode == 'ltm' else "Test Match")
        self.parent_view = parent_view

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.parent_view.run_test(interaction, self.phrase.value)

#: Entries per page. Five code blocks of up to MAX_LTM_SUMMARY_CHARACTERS each stay
#: under an embed description's 4096.
DATA_PAGE_SIZE = 5
#: Servers per page of the server dropdown: 3 page controls + 22 = Discord's 25.
SERVER_PAGE_SIZE = 22
SERVER_NAV_VALUES = ("server_prev", "server_jump", "server_next")
#: The Delete dropdown's two bulk picks: every entry on this page, or every one shown.
SELECT_PAGE, SELECT_ALL = "select_page", "select_all"

def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"

def _code(text: str) -> str:
    """A code block, with any backtick swapped for its look-alike so none can close it early."""
    return "```\n" + text.replace("`", "ˋ") + "\n```"

class DataManageView(BlockedGuard, ui.View):
    """A profile's long-term memories or training examples: a page of five, or one opened.

    Memories are listed one server at a time, since a memory is only ever recalled in
    the server it formed in -- so a server is chosen before anything is shown, starting
    on the one this was opened in when it has any. The shard is read, decrypted and
    grouped off the event loop once per change, never per page: it holds up to
    LIMIT_LTM entries.
    """

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, profile_name: str, is_borrowed: bool, mode: Optional[Literal['training', 'ltm']] = None, parent_manage_view: Optional['ProfileManageView'] = None, target_user_id: Optional[int] = None):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        # The profile's owner, which is not the invoker when /mod drives this view.
        # Only the LTM author stamp (in AddLtmModal) still uses the invoker's id.
        self.user_id = target_user_id or interaction.user.id
        self.guild_id = interaction.guild_id
        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        self.parent_manage_view = parent_manage_view
        self.mode: Literal['training', 'ltm'] = mode or ('ltm' if is_borrowed else 'training')
        self.entries: List[Dict] = []                  # newest first, plain text
        self.servers: List[Tuple[str, str, int]] = []  # (guild id, name, memories), LTM only
        self.server_id: Optional[str] = None
        self.search_term: Optional[str] = None
        self.page = 0
        self.server_page = 0
        self.open_id: Optional[str] = None
        self._clock = None

    # --- data -------------------------------------------------------------------

    async def _reload(self):
        mm = self.cog.memory_manager
        if self.mode == 'training':
            self.entries = await mm.load_training_examples(self.user_id, self.profile_name)
            return
        self.entries = await mm.load_ltms(self.user_id, self.profile_name)
        counts: Dict[str, int] = {}
        for e in self.entries:
            if e.get("context_id"):
                counts[str(e["context_id"])] = counts.get(str(e["context_id"]), 0) + 1
        here = str(self.guild_id) if self.guild_id else None
        servers = []
        for sid, n in counts.items():
            guild = self.cog.bot.get_guild(int(sid)) if sid.isdigit() else None
            servers.append((sid, guild.name if guild else f"Server {sid}", n))
        servers.sort(key=lambda s: (s[0] != here, -s[2], s[1].lower()))
        self.servers = servers
        ids = {sid for sid, _, _ in self.servers}
        if self.server_id not in ids:
            self.server_id = here if here in ids else None

    def _text(self, e: Dict) -> str:
        return e.get("sum", "") if self.mode == 'ltm' else f"{e.get('u_in', '')} {e.get('b_out', '')}"

    def _visible(self) -> List[Dict]:
        items = self.entries
        if self.mode == 'ltm':
            items = [e for e in items if str(e.get("context_id")) == self.server_id] if self.server_id else []
        if self.search_term:
            needle = self.search_term.lower()
            items = [e for e in items if needle in self._text(e).lower()]
        return items

    def _opened(self) -> Optional[Dict]:
        return next((e for e in self.entries if e.get("id") == self.open_id), None) if self.open_id else None

    def _server_name(self, sid: Optional[str]) -> str:
        return next((name for s, name, _ in self.servers if s == sid), f"Server {sid}")

    def _date(self, e: Dict, key: str = "created_ts") -> str:
        """On the viewer's clock, from About Me -- not UTC, and not the character's."""
        if self._clock is None:
            self._clock, _ = _resolve_zoneinfo(
                self.cog.profile_manager.user_timezone(self.original_interaction.user.id))
        return entry_date({"created_ts": e.get(key) or e.get("ts")}, self._clock,
                          "%d %b %Y, %I:%M %p %Z") or "undated"

    @staticmethod
    def _source(e: Dict) -> str:
        by = f" by {e['usr']}" if e.get("usr") else ""
        return {"auto": "auto", "memorise": f"/memorise{by}", "manual": f"added{by}"}.get(e.get("src"), "")

    # --- rendering ----------------------------------------------------------------

    def _embed(self, items: List[Dict]) -> discord.Embed:
        opened = self._opened()
        if opened:
            return self._detail_embed(opened)
        noun = "Long-Term Memories" if self.mode == 'ltm' else "Training Examples"
        embed = discord.Embed(title=f"{noun} · {self.profile_name}", color=discord.Color.dark_teal())
        limit = defaultConfig.LIMIT_LTM if self.mode == 'ltm' else defaultConfig.LIMIT_TRAINING
        embed.set_footer(text=f"{len(self.entries)} / {limit} {'memories across every server' if self.mode == 'ltm' else 'examples'}")

        if self.mode == 'ltm' and not self.server_id:
            embed.description = ("Choose a server. A memory is only ever recalled in the server it formed in."
                                 if self.servers else "No memories yet.")
            return embed

        start = self.page * DATA_PAGE_SIZE
        header = []
        if self.mode == 'ltm':
            header.append(f"**{self._server_name(self.server_id)}** · {len(items)} memories")
        if self.search_term:
            header.append(f"Search: `{self.search_term}` · {len(items)} found")
        lines = []
        for n, e in enumerate(items[start:start + DATA_PAGE_SIZE], start + 1):
            if self.mode == 'ltm':
                meta = " · ".join(x for x in (self._date(e), self._source(e)) if x)
                lines.append(f"**{n}.** {meta}\n{_code(_clip(e.get('sum', ''), MAX_LTM_SUMMARY_CHARACTERS))}")
            else:
                lines.append(f"**{n}.** {self._date(e)}\n" + _code(
                    f"User: {_clip(e.get('u_in', ''), 200)}\n"
                    f"{_clip(self.profile_name, 40)}: {_clip(e.get('b_out', ''), 400)}"))
        body = "\n\n".join(lines) or ("Nothing matches." if self.search_term else "Nothing here yet.")
        embed.description = "\n".join(header + ([""] if header else []) + [body])
        return embed

    def _detail_embed(self, e: Dict) -> discord.Embed:
        edited = e.get("modified_ts") and e.get("modified_ts") != e.get("created_ts")
        made = self._date(e) + (f" · edited {self._date(e, 'modified_ts')}" if edited else "")
        if self.mode == 'ltm':
            embed = discord.Embed(title=f"Memory `{e.get('id')}`", description=e.get("sum", "")[:4000],
                                  color=discord.Color.dark_teal())
            embed.add_field(name="Made", value=made, inline=True)
            embed.add_field(name="Server", value=self._server_name(str(e.get("context_id"))), inline=True)
            if self._source(e):
                embed.add_field(name="Source", value=self._source(e), inline=True)
        else:
            embed = discord.Embed(title=f"Example `{e.get('id')}`",
                                  description=f"**User**\n{e.get('u_in', '')}\n\n**{self.profile_name}**\n{e.get('b_out', '')}"[:4000],
                                  color=discord.Color.dark_teal())
            embed.add_field(name="Made", value=made, inline=True)
        return embed

    def _build_components(self, items: List[Dict]):
        self.clear_items()
        opened = self._opened()
        if opened:
            add_button(self, "Edit", self.edit_callback, style=discord.ButtonStyle.primary, row=0)
            add_button(self, "Delete", self.delete_callback, style=discord.ButtonStyle.danger, row=0)
            add_button(self, "Back to list", self.close_callback, row=0, emoji="⬅️")
            return

        # Paged the way Set Models pages its models: the page controls are the dropdown's
        # first options, so no row of buttons is spent on them.
        if self.mode == 'ltm' and self.servers:
            start = self.server_page * SERVER_PAGE_SIZE
            add_select(self, paged_nav_options(self.server_page, self._pages(self.servers, SERVER_PAGE_SIZE),
                                               values=SERVER_NAV_VALUES, nav_suffix=" of servers")
                       + [discord.SelectOption(label=_clip(name, 80), description=f"{n} memories",
                                               value=sid, default=sid == self.server_id)
                          for sid, name, n in self.servers[start:start + SERVER_PAGE_SIZE]],
                       self.server_callback, placeholder="Choose a server…", row=0)

        page_items = items[self.page * DATA_PAGE_SIZE:(self.page + 1) * DATA_PAGE_SIZE]
        if page_items:
            first = self.page * DATA_PAGE_SIZE + 1

            def options():   # once per dropdown, so the two share no option objects
                return paged_nav_options(self.page, self._pages(items)) + [
                    discord.SelectOption(label=_clip(f"{n}. {self._text(e)}", 100),
                                         value=str(e.get("id")), description=self._date(e))
                    for n, e in enumerate(page_items, first)]
            add_select(self, options(), self.open_callback, placeholder="Open…", row=1)
            delete = options()
            bulk = [discord.SelectOption(label="☑ Select Page", value=SELECT_PAGE,
                                         description=f"All {len(page_items)} on this page")]
            if len(items) > len(page_items):   # on one page, Select All would say the same thing
                shown = (f"matching `{_clip(self.search_term, 40)}`" if self.search_term
                         else f"in {_clip(self._server_name(self.server_id), 60)}" if self.mode == 'ltm'
                         else "for this profile")
                bulk.append(discord.SelectOption(label="☑ Select All", value=SELECT_ALL,
                                                 description=f"All {len(items)} {shown}"[:100]))
            nav = len(delete) - len(page_items)
            delete[nav:nav] = bulk
            add_select(self, delete, self.delete_selected_callback, placeholder="Delete…",
                       max_values=len(delete), row=2)

        showing = self.mode == 'training' or self.server_id
        if showing:
            add_button(self, "Search", self.search_callback, row=3, emoji="\U0001f50d")
            add_button(self, "Test recall" if self.mode == 'ltm' else "Test match", self.test_callback,
                       row=3, emoji="\U0001f9e0" if self.mode == 'ltm' else "\U0001f9ea",
                       disabled=not self.entries)

        add_button(self, "Add", self.add_callback, style=discord.ButtonStyle.success, row=4, emoji="➕")
        if self.mode == 'training' and not self.is_borrowed and self.entries:
            async def analyse_cb(i): await i.response.send_modal(AnalyseExamplesModal(self))
            add_button(self, "Analyse", analyse_cb, style=discord.ButtonStyle.blurple, row=4)
        if self.parent_manage_view:
            add_button(self, "Back to Dashboard", self.back_callback, row=4, emoji="⬅️")

    @staticmethod
    def _pages(items: list, size: int = DATA_PAGE_SIZE) -> int:
        return max(1, -(-len(items) // size))

    async def _show(self, interaction: discord.Interaction):
        # Filtered once per repaint, and the pages clamped before either half reads them.
        items = self._visible()
        self.page = max(0, min(self.page, self._pages(items) - 1))
        self.server_page = max(0, min(self.server_page, self._pages(self.servers, SERVER_PAGE_SIZE) - 1))
        embed = self._embed(items)
        self._build_components(items)
        if interaction.response.is_done():
            await interaction.edit_original_response(embed=embed, view=self)
        else:
            await interaction.response.edit_message(embed=embed, view=self)

    async def start(self):
        if not self.original_interaction.response.is_done():
            await self.original_interaction.response.defer()
        await self._reload()
        await self._show(self.original_interaction)

    async def _refresh(self):
        """After a change made through a modal or a confirmation: re-read, then repaint
        the screen itself, which belongs to the interaction that opened it."""
        await self._reload()
        if self.open_id and not self._opened():
            self.open_id = None
        await self._show(self.original_interaction)

    # --- callbacks ------------------------------------------------------------------

    async def _turn_page(self, interaction: discord.Interaction, value: str) -> bool:
        """Previous, next or jump, when `value` is one of a dropdown's page controls."""
        for attr, values, pages in (("page", PAGE_NAV_VALUES, lambda: self._pages(self._visible())),
                                    ("server_page", SERVER_NAV_VALUES,
                                     lambda: self._pages(self.servers, SERVER_PAGE_SIZE))):
            if value not in values:
                continue
            step = (-1, 0, 1)[values.index(value)]
            if step:
                setattr(self, attr, getattr(self, attr) + step)
                await self._show(interaction)
                return True

            async def jump(i: discord.Interaction, page: int):
                setattr(self, attr, page)
                await self._show(i)
            await interaction.response.send_modal(PageJumpModal(pages(), jump, zero_indexed=True))
            return True
        return False

    async def server_callback(self, interaction: discord.Interaction):
        if await self._turn_page(interaction, interaction.data['values'][0]):
            return
        self.server_id = interaction.data['values'][0]
        self.page, self.open_id, self.search_term = 0, None, None
        await self._show(interaction)

    async def open_callback(self, interaction: discord.Interaction):
        if await self._turn_page(interaction, interaction.data['values'][0]):
            return
        self.open_id = interaction.data['values'][0]
        await self._show(interaction)

    async def close_callback(self, interaction: discord.Interaction):
        self.open_id = None
        await self._show(interaction)

    async def search_callback(self, interaction: discord.Interaction):
        await interaction.response.send_modal(SearchDataModal(self))

    async def test_callback(self, interaction: discord.Interaction):
        await interaction.response.send_modal(TestMatchModal(self))

    async def back_callback(self, i: discord.Interaction):
        await i.response.defer()
        embed = await self.cog.profile_manager._build_profile_manage_embed(self.original_interaction, self.profile_name, target_user_id=self.parent_manage_view.user_id)
        self.parent_manage_view._build_view()
        await self.original_interaction.edit_original_response(embed=embed, view=self.parent_manage_view)

    async def add_callback(self, interaction: discord.Interaction):
        if self.mode == 'training':
            modal = AddTrainingExampleModal(self.cog, self.user_id, self.profile_name, self.guild_id)
        else:
            modal = AddLtmModal(self.cog, self.user_id, self.profile_name, self.guild_id)

        original_on_submit = modal.on_submit
        async def on_submit_refresh(i: discord.Interaction):
            await original_on_submit(i)
            # Newest first, so the new entry heads page one -- of the server it was
            # filed under, which is the one this screen was opened in.
            self.page, self.open_id, self.search_term = 0, None, None
            if self.mode == 'ltm' and self.guild_id:
                self.server_id = str(self.guild_id)
            await self._refresh()
        modal.on_submit = on_submit_refresh
        await interaction.response.send_modal(modal)

    async def edit_callback(self, interaction: discord.Interaction):
        item = self._opened()
        if not item:
            await interaction.response.send_message("Could not find the selected item to edit.", ephemeral=True)
            return
        if self.mode == 'training':
            modal = EditTrainingExampleModal(self.cog, self.user_id, self.profile_name, item["id"],
                                             item.get("u_in", ""), item.get("b_out", ""), self.guild_id)
        else:
            modal = EditLtmModal(self.cog, self.user_id, self.profile_name, item["id"], item.get("sum", ""))
        original_on_submit = modal.on_submit
        async def on_submit_refresh(i: discord.Interaction):
            await original_on_submit(i)
            await self._refresh()
        modal.on_submit = on_submit_refresh
        await interaction.response.send_modal(modal)

    async def _confirm_delete(self, interaction: discord.Interaction, ids: List[str], prompt: str):
        async def confirm(i: discord.Interaction):
            mm = self.cog.memory_manager
            delete = mm.delete_ltms if self.mode == 'ltm' else mm.delete_training_examples
            gone = await delete(self.user_id, self.profile_name, ids)
            await i.response.edit_message(content=f"Deleted {gone} item(s)." if gone else "Nothing was found to delete.",
                                          view=None, embed=None)
            await self._refresh()
        await interaction.response.send_message(prompt, view=build_confirm_view(f"Delete {len(ids)}", confirm),
                                                ephemeral=True)

    async def delete_callback(self, interaction: discord.Interaction):
        if self.open_id:
            await self._confirm_delete(interaction, [self.open_id],
                                       f"**Delete `{self.open_id}`?** This is permanent.")

    async def delete_selected_callback(self, interaction: discord.Interaction):
        # A page control picked beside entries is ignored: the deletion is what was meant,
        # and it still waits on a confirmation. Select All outranks Select Page, which
        # outranks single picks -- each already includes the next.
        values = list(interaction.data['values'])
        items = self._visible()
        if SELECT_ALL in values:
            what = f"matching `{self.search_term}` " if self.search_term else ""
            where = f"from '{self._server_name(self.server_id)}' " if self.mode == 'ltm' else ""
            await self._confirm_delete(
                interaction, [e["id"] for e in items if e.get("id")],
                f"**Delete all {len(items)} {what}{where}for '{self.profile_name}'?** This is permanent.")
            return
        if SELECT_PAGE in values:
            ids = [e["id"] for e in items[self.page * DATA_PAGE_SIZE:(self.page + 1) * DATA_PAGE_SIZE]
                   if e.get("id")]
        else:
            ids = [v for v in values if v not in PAGE_NAV_VALUES]
        if not ids:
            await self._turn_page(interaction, values[0])
            return
        await self._confirm_delete(interaction, ids, f"**Delete {len(ids)} selected item(s)?** This is permanent.")

    async def run_test(self, interaction: discord.Interaction, query: str):
        """Test recall / Test match: the stored entries nearest `query`, scored, against
        the profile's own threshold -- which is what decides what a turn is sent."""
        if not self.cog.storage_manager._embedding_routes(interaction.guild_id, self.user_id):
            await interaction.followup.send(NO_EMBEDDING_KEY_MSG, ephemeral=True)
            return
        mm = self.cog.memory_manager
        cfg = self.cog.profile_manager._get_profile_config(self.user_id, self.profile_name, self.is_borrowed) or {}
        if self.mode == 'ltm':
            ranked = await mm.rank_ltms(self.user_id, self.profile_name, self.server_id, query, interaction.guild_id)
            threshold = float(cfg.get("ltm_relevance_threshold", 0.75))
            size = int(cfg.get("ltm_context_size", 3))
            note = (f"Threshold {threshold} · up to {size} per turn. A turn also skips memories it "
                    "recalled recently, and prefers ones that differ from each other.")
            text = lambda e: self.cog.storage_manager._decrypt_data(e.get("sum", ""))
        else:
            ranked = await mm.rank_training_examples(self.user_id, self.profile_name, query, interaction.guild_id)
            threshold = float(cfg.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD))
            size = int(cfg.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE))
            note = f"Threshold {threshold} · up to {size} per turn."
            text = lambda e: self.cog.storage_manager._decrypt_data(e.get("u_in", ""))
        if ranked is None:
            await interaction.followup.send(EMBEDDING_FAILED_MSG, ephemeral=True)
            return
        lines = [f"{'✅' if sim >= threshold else '▫️'} `{sim:.2f}` {_clip(text(e), 150)}"
                 for sim, e in ranked]
        embed = discord.Embed(title="Test Recall" if self.mode == 'ltm' else "Test Match",
                              description=f"> {_clip(query, 200)}\n\n" + ("\n".join(lines) or "Nothing stored here to match."),
                              color=discord.Color.dark_teal())
        embed.set_footer(text=note)
        await interaction.followup.send(embed=embed, ephemeral=True)

class AnalyseExamplesModal(ui.Modal, title="Analyse Training Examples"):
    def __init__(self, parent_view: 'DataManageView'):
        super().__init__()
        self.parent_view = parent_view
        self.count_input = ui.TextInput(label="Number of Examples to Process", placeholder="Default: 10", default="10", required=True, min_length=1, max_length=3)
        self.verbosity_input = ui.TextInput(label="Target Verbosity (50 - 3000 chars)", placeholder="Default: 800", default="800", required=True, min_length=2, max_length=4)
        self.model_input = ui.TextInput(label="Analysis Model (optional)", placeholder="Blank: this profile's LTM model, with fallback", required=False)
        self.add_item(self.count_input)
        self.add_item(self.verbosity_input)
        self.add_item(self.model_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            count = int(self.count_input.value)
            verbosity = int(self.verbosity_input.value)
            model_name = self.model_input.value.strip()
            
            if not (50 <= verbosity <= 3000): raise ValueError("Verbosity out of range.")
            if count < 1: raise ValueError("Count too low.")
            if model_name and not model_name.upper().startswith(("GOOGLE/", "OPENROUTER/")):
                raise ValueError("Model must start with GOOGLE/ or OPENROUTER/.")
        except ValueError as e:
            await interaction.response.send_message(f"❌ **Invalid Input:** {suppress_link_previews(str(e))}", ephemeral=True); return

        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.parent_view.cog.memory_manager._execute_training_analysis(
            interaction, self.parent_view.user_id, self.parent_view.profile_name, count, verbosity,
            model_name or None)

class PrivacyDashboardView(BlockedGuard, ui.View):
    # Open to a blocked user. Export and deletion are theirs whatever their standing --
    # Privacy Policy §5 promises it -- and neither button reaches a model. The server's
    # data policy is read-only here; the bot owner's button opens its own screen, which is
    # not exempt.
    block_exempt = True

    def __init__(self, cog: 'MimicCog', user_id: int, guild: Optional[discord.Guild] = None):
        super().__init__(timeout=300)
        self.cog = cog
        self.user_id = user_id
        #: The server /privacy was run in, whose data policy it shows. None in a DM.
        self.guild = guild
        if guild is not None and may_set_data_policy(user_id):
            add_button(self, "Change Data Policy", self.open_data_policy, row=1, emoji="🛡️")

    def embed(self) -> discord.Embed:
        e = discord.Embed(
            title="Privacy & Data Dashboard",
            description="Request a full export of your data or permanently delete your account and all associated profiles, memories, and settings.",
            color=discord.Color.red()
        )
        if self.guild is not None:
            e.description += (
                "\n\n**This server's data policy** decides whether its messages may reach AI "
                "providers that train on them. Only the bot's owner can change it.")
            add_data_policy_fields(e, self.cog.server_manager._get_server_index(str(self.guild.id)))
            e.set_footer(text=self.guild.name)
        return e

    async def open_data_policy(self, interaction: discord.Interaction):
        if not may_set_data_policy(interaction.user.id):
            await interaction.response.send_message(
                "Only this bot's owner can change a server's data policy.", ephemeral=True)
            return

        def back():
            dashboard = PrivacyDashboardView(self.cog, self.user_id, self.guild)
            return dashboard.embed(), dashboard

        view = DataPolicyView(self.cog, self.guild, self.user_id, on_back=back)
        await interaction.response.edit_message(embed=view.embed(), view=view)

    @ui.button(label="Request Data Export", style=discord.ButtonStyle.blurple, emoji="📥")
    async def export_data(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.cog.profile_manager._execute_privacy_export(self.user_id, interaction)

    @ui.button(label="Delete My Account", style=discord.ButtonStyle.danger, emoji="⚠️")
    async def delete_account(self, interaction: discord.Interaction, button: ui.Button):
        modal = AccountDeleteModal(self.cog, self.user_id)
        await interaction.response.send_modal(modal)

class AccountDeleteModal(ui.Modal, title="Permanently Delete Account"):
    confirm_input = ui.TextInput(label="Type 'DELETE' to confirm", placeholder="DELETE", required=True, max_length=6)

    def __init__(self, cog: 'MimicCog', user_id: int):
        super().__init__()
        self.cog = cog
        self.user_id = user_id

    async def on_submit(self, interaction: discord.Interaction):
        if self.confirm_input.value != "DELETE":
            await interaction.response.send_message("❌ Deletion cancelled. You must type 'DELETE' exactly.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.cog.profile_manager._execute_account_deletion(self.user_id, interaction)

class ExportPassphraseModal(ui.Modal, title="Self-Hosted Export"):
    passphrase_input = ui.TextInput(label="Enter a strong passphrase", placeholder="Used to decrypt on your self-hosted instance", required=True, min_length=8, max_length=100)

    def __init__(self, parent_view):
        super().__init__()
        self.parent_view = parent_view

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        passphrase = self.passphrase_input.value
        await self.parent_view.cog.profile_manager._execute_export(interaction, list(self.parent_view.selected_profiles), self.parent_view.export_filters, passphrase=passphrase)

class ImportPassphraseModal(ui.Modal, title="Enter Passphrase"):
    passphrase_input = ui.TextInput(label="Passphrase", placeholder="Enter the passphrase used for export", required=True, min_length=8, max_length=100)

    def __init__(self, cog, file_bytes: bytes):
        super().__init__()
        self.cog = cog
        self.file_bytes = file_bytes

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.cog.profile_manager._execute_import(interaction, file_bytes=self.file_bytes, passphrase=self.passphrase_input.value)

class BulkExportView(BaseBulkProfileView):
    def __init__(self, cog: 'MimicCog', user_id: int):
        super().__init__(cog, user_id, include_borrowed=False)
        self.export_filters = set()
        self._build_view()

    def _build_view(self):
        self.clear_items()
        self._build_profile_select_ui(row=0)
        
        filter_options = [
            discord.SelectOption(label="Long-Term Memories", value="ltm", description="Optional. Include compiled conversation memories.", default="ltm" in self.export_filters),
            discord.SelectOption(label="Training Examples", value="training", description="Optional. Include training input/output style examples.", default="training" in self.export_filters)
        ]
        
        add_select(self, filter_options, self.filter_callback,
                   placeholder="Optional. Select additional memories to export...", min_values=1,
                   max_values=len(filter_options), row=2)

        add_button(self, "Standard Export", self.export_master_callback,
                   style=discord.ButtonStyle.primary, row=3)

        add_button(self, "Export for Self-Hosted", self.export_selfhost_callback,
                   style=discord.ButtonStyle.secondary, row=3)

    async def filter_callback(self, interaction: discord.Interaction):
        self.export_filters = set(interaction.data['values'])
        await interaction.response.defer()

    async def export_master_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        if not self.selected_profiles:
            await interaction.followup.send("Select at least one profile to export.", ephemeral=True)
            return
        await self.cog.profile_manager._execute_export(interaction, list(self.selected_profiles), self.export_filters)

    async def export_selfhost_callback(self, interaction: discord.Interaction):
        if not self.selected_profiles:
            await interaction.response.send_message("Select at least one profile to export.", ephemeral=True)
            return
        await interaction.response.send_modal(ExportPassphraseModal(self))
