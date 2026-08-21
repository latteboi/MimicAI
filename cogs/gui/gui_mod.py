from ..utils.constants import *

import discord
from discord import ui
import datetime
from string import Formatter
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Optional
from ..utils.helpers import _sanitise_filename
from .base_components import TimeoutCleanupMixin, build_pagination_controls, build_tab_nav_bar, build_confirm_view

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog

from .gui_profiles import ProfileManageView

class ModBaseView(TimeoutCleanupMixin, ui.View):
    """Base for every /mod tab.

    target_user_id is the moderated user, and every tab carries it so a trip out
    to Stats or Docs and back to Profiles does not lose it. Nothing but the
    Profiles tab reads it; the rest exist only to ferry it.
    """

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, current_tab: str,
                 target_user_id: Optional[int] = None):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.current_tab = current_tab
        self.target_user_id = target_user_id
        self._add_nav_buttons()

    def _add_page_controls(self, num_pages: int, row: int, *, repaint: bool = True):
        """Attach prev/next page buttons.

        Four tabs each carried their own pair of near-identical `p_cb`/`n_cb` closures.
        `repaint=True` edits the message in place off the click; `repaint=False` defers
        and repaints through `update_display`, which is what the Docs tab did.
        """
        async def p_cb(i: discord.Interaction):
            await self._turn_page(i, -1, repaint)

        async def n_cb(i: discord.Interaction):
            await self._turn_page(i, 1, repaint)

        build_pagination_controls(self, self.current_page, num_pages, row, p_cb, n_cb)

    async def _turn_page(self, i: discord.Interaction, delta: int, repaint: bool):
        self.current_page += delta
        self._build_view()
        if repaint:
            await i.response.edit_message(embed=self._get_embed(), view=self)
        else:
            await i.response.defer()
            await self.update_display()

    def _add_nav_buttons(self):
        ModBaseView.add_nav_to_other_view(
            self, self.cog, self.original_interaction, self.current_tab, self.target_user_id)

    @staticmethod
    def add_nav_to_other_view(target_view, cog, interaction, current_tab, target_user_id: Optional[int] = None):
        async def nav_stats(i: discord.Interaction):
            await i.response.defer(); view = ModStatsView(cog, interaction, target_user_id=target_user_id); await view.update_display()
        async def nav_profiles(i: discord.Interaction):
            await i.response.defer(); view = ModProfilesView(cog, interaction, target_user_id=target_user_id); await view.update_display()
        async def nav_prompts(i: discord.Interaction):
            await i.response.defer(); view = ModPromptsView(cog, interaction, target_user_id=target_user_id); await view.update_display()
        async def nav_docs(i: discord.Interaction):
            await i.response.defer(); view = ModDocsView(cog, interaction, target_user_id=target_user_id); await view.update_display()
        async def nav_blacklist(i: discord.Interaction):
            await i.response.defer(); view = ModBlacklistView(cog, interaction, target_user_id=target_user_id); await view.update_display()

        build_tab_nav_bar(target_view, current_tab, [
            ("Stats", "stats", nav_stats),
            ("Profiles", "profiles", nav_profiles),
            ("Prompts", "prompts", nav_prompts),
            ("Docs", "docs", nav_docs),
            ("Blacklist", "blacklist", nav_blacklist),
        ])

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
            cat_select = ui.Select(placeholder="Select Category...", options=cat_opts, row=0)
            
            async def cat_cb(i: discord.Interaction):
                self.selected_category = i.data['values'][0]
                self.selected_file = None
                self.current_page = 0
                self._build_view()
                await i.response.defer()
                await self.update_display()
                
            cat_select.callback = cat_cb
            self.add_item(cat_select)

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
                
            file_select = ui.Select(placeholder="Select Document...", options=file_opts, row=1)
            
            async def file_cb(i: discord.Interaction):
                self.selected_file = i.data['values'][0]
                self._build_view()
                await i.response.defer()
                await self.update_display()
                
            file_select.callback = file_cb
            self.add_item(file_select)

            if num_pages > 1:
                self._add_page_controls(num_pages, 2, repaint=False)

        # 5. Build Action Buttons
        action_row = 3 if len(files) > DROPDOWN_MAX_OPTIONS else 2
        
        btn_create = ui.Button(label="Create New", style=discord.ButtonStyle.green, row=action_row)
        async def create_cb(i: discord.Interaction):
            await i.response.send_modal(ModDocsCreateModal(self))
        btn_create.callback = create_cb
        self.add_item(btn_create)

        btn_edit = ui.Button(label="Edit", style=discord.ButtonStyle.primary, disabled=(not self.selected_file), row=action_row)
        async def edit_cb(i: discord.Interaction):
            filepath = os.path.join(DOCS_DIR, self.selected_category, self.selected_file)
            content = ""
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            await i.response.send_modal(ModDocsEditModal(self, content))
        btn_edit.callback = edit_cb
        self.add_item(btn_edit)

        btn_delete = ui.Button(label="Delete", style=discord.ButtonStyle.danger, disabled=(not self.selected_file), row=action_row)
        async def delete_cb(i: discord.Interaction):
            async def confirm_action(confirm_i: discord.Interaction):
                filepath = os.path.join(DOCS_DIR, self.selected_category, self.selected_file)
                if os.path.exists(filepath):
                    os.remove(filepath)

                # Re-index
                self.cog.bot.loop.create_task(self.cog.help_service._load_and_embed_docs())

                self.selected_file = None
                self._build_view()
                await confirm_i.response.edit_message(content=f"Deleted successfully.", view=None, embed=None)
                await self.update_display()

            confirm_view = build_confirm_view("Confirm Delete", confirm_action)
            await i.response.send_message(f"Are you sure you want to delete `{self.selected_file}`?", view=confirm_view, ephemeral=True)
            
        btn_delete.callback = delete_cb
        self.add_item(btn_delete)

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

class ModBlacklistModal(ui.Modal, title="Enter User ID"):
    user_id_input = ui.TextInput(label="Discord User ID", required=True)
    def __init__(self, view):
        super().__init__()
        self.parent_view = view
        
    async def on_submit(self, i: discord.Interaction):
        uid_str = self.user_id_input.value.strip()
        if not uid_str.isdigit():
            await i.response.send_message("Invalid ID. Must be numeric.", ephemeral=True)
            return
            
        uid = int(uid_str)
        if uid not in self.parent_view.cog.global_blacklist:
            self.parent_view.cog.global_blacklist.add(uid)
            self.parent_view.cog.server_manager._save_blacklist()
            
        self.parent_view.selected_user_id = uid
        self.parent_view._build_view()
        await i.response.edit_message(embed=self.parent_view._get_embed(), view=self.parent_view)

class ModBlacklistView(ModBaseView):
    def __init__(self, cog, interaction, target_user_id: Optional[int] = None):
        super().__init__(cog, interaction, "blacklist", target_user_id=target_user_id)
        self.selected_user_id = None
        self.current_page = 0
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        blacklist = sorted(list(self.cog.global_blacklist))
        
        btn_enter = ui.Button(label="Enter User ID", style=discord.ButtonStyle.primary, row=0)
        async def enter_cb(i: discord.Interaction):
            await i.response.send_modal(ModBlacklistModal(self))
        btn_enter.callback = enter_cb
        self.add_item(btn_enter)
        
        btn_remove = ui.Button(label="Remove", style=discord.ButtonStyle.danger, disabled=(self.selected_user_id is None), row=0)
        async def remove_cb(i: discord.Interaction):
            if self.selected_user_id in self.cog.global_blacklist:
                self.cog.global_blacklist.discard(self.selected_user_id)
                self.cog.server_manager._save_blacklist()
            self.selected_user_id = None
            self._build_view()
            await i.response.edit_message(embed=self._get_embed(), view=self)
        btn_remove.callback = remove_cb
        self.add_item(btn_remove)
        
        if blacklist:
            num_pages = (len(blacklist) - 1) // DROPDOWN_MAX_OPTIONS + 1
            if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
            
            start = self.current_page * DROPDOWN_MAX_OPTIONS
            page_items = blacklist[start : start + DROPDOWN_MAX_OPTIONS]
            
            options = []
            for uid in page_items:
                user = self.cog.bot.get_user(uid)
                uname = user.name if user else "Unknown User"
                options.append(discord.SelectOption(label=f"{uname} ({uid})", value=str(uid), default=(self.selected_user_id == uid)))
                
            sel = ui.Select(placeholder="Select a blacklisted user...", options=options, row=1)
            
            async def sel_cb(i: discord.Interaction):
                self.selected_user_id = int(i.data['values'][0])
                self._build_view()
                await i.response.edit_message(embed=self._get_embed(), view=self)
            
            sel.callback = sel_cb
            self.add_item(sel)

            if num_pages > 1:
                self._add_page_controls(num_pages, 2)

        self._add_nav_buttons()

    def _get_embed(self):
        embed = discord.Embed(title="Global Blacklist", color=discord.Color.dark_red())
        if self.selected_user_id:
            user = self.cog.bot.get_user(self.selected_user_id)
            uname = user.name if user else "Unknown User"
            embed.description = f"Selected User: **{uname}** (`{self.selected_user_id}`)\nClick 'Remove' to pardon them."
        else:
            embed.description = f"Total Blacklisted Users: **{len(self.cog.global_blacklist)}**\nEnter a User ID to add them to the blacklist, or select an existing one below to remove them."
        return embed

    async def update_display(self):
        await self.original_interaction.edit_original_response(embed=self._get_embed(), view=self)

class ModStatsView(ModBaseView):
    def __init__(self, cog, interaction, target_user_id: Optional[int] = None):
        super().__init__(cog, interaction, "stats", target_user_id=target_user_id)
        self.selected_category = "Servers"
        self.current_page = 0
        self.content_dict = {}
        self._load_data()
        self._build_view()

    def _load_data(self):
        guilds_sorted = sorted(self.cog.bot.guilds, key=lambda g: g.me.joined_at if (g.me and g.me.joined_at) else datetime.datetime.now(datetime.timezone.utc))
        
        servers_pages = []
        if not guilds_sorted:
            servers_pages.append("No server data available.")
        else:
            for i in range(0, len(guilds_sorted), 25):
                chunk = guilds_sorted[i:i + 25]
                lines = []
                for j, guild in enumerate(chunk, start=i + 1):
                    join_str = guild.me.joined_at.strftime("%d/%m/%Y") if (guild.me and guild.me.joined_at) else "Unknown"
                    lines.append(f"{j}. **{guild.name}** (`{guild.id}`) — Joined: `{join_str}`")
                servers_pages.append("\n".join(lines))
        
        user_stats = []
        if os.path.isdir(self.cog.USERS_DIR):
            for user_id_str in os.listdir(self.cog.USERS_DIR):
                if user_id_str.isdigit():
                    user_id = int(user_id_str)
                    index = self.cog.profile_manager._get_user_index(user_id)
                    profile_count = len(index.get("personal", {}))
                    if profile_count > 0:
                        user_obj = self.cog.bot.get_user(user_id)
                        user_name = user_obj.name if user_obj else "Unknown User"
                        user_stats.append({"id": user_id, "name": user_name, "count": profile_count})
        
        user_stats.sort(key=lambda x: x["count"], reverse=True)
        
        users_pages = []
        if not user_stats:
            users_pages.append("No user data available.")
        else:
            for i in range(0, len(user_stats), 25):
                chunk = user_stats[i:i + 25]
                lines = []
                for j, u_stat in enumerate(chunk, start=i + 1):
                    lines.append(f"{j}. **{u_stat['name']}** (`{u_stat['id']}`) — Personal Profiles: `{u_stat['count']}`")
                users_pages.append("\n".join(lines))

        self.content_dict = {"Servers": servers_pages, "Users": users_pages}

    def _build_view(self):
        self.clear_items()

        cat_opts = [discord.SelectOption(label=cat, value=cat, default=(cat == self.selected_category)) for cat in self.content_dict.keys()]
        cat_sel = ui.Select(placeholder="Select Category...", options=cat_opts, row=0)
        async def cat_cb(i: discord.Interaction):
            self.selected_category = i.data['values'][0]
            self.current_page = 0
            self._build_view()
            await i.response.edit_message(embed=self._get_embed(), view=self)
        cat_sel.callback = cat_cb
        self.add_item(cat_sel)

        pages = self.content_dict[self.selected_category]
        num_pages = len(pages)
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)

        self._add_page_controls(num_pages, 1)
        self._add_nav_buttons()

    def _get_embed(self):
        embed = discord.Embed(title="MimicAI Statistics", color=discord.Color.gold())
        pages = self.content_dict[self.selected_category]
        embed.description = pages[self.current_page]
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

class ModProfilesView(ModBaseView):
    def __init__(self, cog, interaction, target_user_id: Optional[int] = None):
        super().__init__(cog, interaction, "profiles", target_user_id=target_user_id)
        self.current_page = 0
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
        
        btn_enter = ui.Button(label="Enter User ID", style=discord.ButtonStyle.success, row=0)
        async def enter_cb(i: discord.Interaction):
            await i.response.send_modal(ModProfilesModal(self))
        btn_enter.callback = enter_cb
        self.add_item(btn_enter)
        
        if self.target_user_id:
            index = self.cog.profile_manager._get_user_index(self.target_user_id)
            profiles = self._resolve_profiles()
            
            if profiles:
                num_pages = (len(profiles) - 1) // DROPDOWN_MAX_OPTIONS + 1
                if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
                
                start = self.current_page * DROPDOWN_MAX_OPTIONS
                page_items = profiles[start : start + DROPDOWN_MAX_OPTIONS]
                
                options = [discord.SelectOption(label=p[:100], value=p[:100]) for p in page_items]
                sel = ui.Select(placeholder="Select a profile to manage...", options=options, row=1)
                
                async def sel_cb(i: discord.Interaction):
                    pname = i.data['values'][0]
                    is_b = pname in index.get("borrowed", [])
                    await i.response.defer()
                    pm_view = ProfileManageView(self.cog, self.original_interaction, pname, is_b, target_user_id=self.target_user_id, is_mod_view=True)
                    embed = await self.cog.profile_manager._build_profile_manage_embed(self.original_interaction, pname, target_user_id=self.target_user_id)
                    await self.original_interaction.edit_original_response(embed=embed, view=pm_view)
                
                sel.callback = sel_cb
                self.add_item(sel)

                if num_pages > 1:
                    self._add_page_controls(num_pages, 2)

        self._add_nav_buttons()

    def _get_embed(self):
        embed = discord.Embed(title="Moderator Profile Dashboard", color=discord.Color.red())
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
        ("Context Rules", "CONTEXT_RULES", DEFAULT_CONTEXT_RULES),
        ("Time Context", "TIME_CONTEXT", DEFAULT_TIME_CONTEXT),
        ("Negative Constraints", "NEGATIVE_CONSTRAINTS", DEFAULT_NEGATIVE_CONSTRAINTS),
        ("Training Data Injection", "TRAINING_DATA_INJECTION", DEFAULT_TRAINING_DATA_INJECTION),
        ("Neuro-Endocrine Engine", "NEURO_ENGINE", DEFAULT_NEURO_INSTRUCTION),
        ("Content Policy (non-18+ channels)", "CONTENT_POLICY", DEFAULT_CONTENT_POLICY),
    ]),
    ("Turn Flow", [
        ("Whisper Injection", "WHISPER_INJECTION", DEFAULT_WHISPER_INJECTION),
        ("Whisper Recap", "WHISPER_RECAP", DEFAULT_WHISPER_RECAP),
        ("Kickstart: Start", "KICKSTART_START", DEFAULT_KICKSTART_START),
        ("Kickstart: Continue", "KICKSTART_CONTINUE", DEFAULT_KICKSTART_CONTINUE),
        ("Kickstart: Idle", "KICKSTART_IDLE", DEFAULT_KICKSTART_IDLE),
        ("AI Director Prompt", "DIRECTOR_USER_PROMPT", DEFAULT_DIRECTOR_USER_PROMPT),
    ]),
    ("Image Generation", [
        ("Image: Present (own)", "IMAGE_PRESENT", DEFAULT_IMAGE_PRESENT),
        ("Image: Present (other's)", "IMAGE_PRESENT_OTHER", DEFAULT_IMAGE_PRESENT_OTHER),
        ("Image: Failed", "IMAGE_FAILED", DEFAULT_IMAGE_FAILED),
        ("Image: Appearance Preamble", "IMAGE_APPEARANCE", DEFAULT_IMAGE_APPEARANCE),
        ("Image: Grounding Preamble", "IMAGE_GROUNDING", DEFAULT_IMAGE_GROUNDING),
    ]),
    ("Grounding & Critics", [
        ("Web Grounding (Text)", "WEB_GROUNDING_TEXT", DEFAULT_WEB_GROUNDING_TEXT),
        ("Web Grounding (Visual)", "WEB_GROUNDING_VISUAL", DEFAULT_WEB_GROUNDING_VISUAL),
        ("Grounding RAG Payload", "GROUNDING_RAG_PAYLOAD", DEFAULT_GROUNDING_RAG_PAYLOAD),
        ("Anti-Repetition Critic", "ANTI_REPETITION", DEFAULT_ANTI_REPETITION_PROMPT),
        ("Auto-Moderator Critic", "AUTO_MODERATOR", DEFAULT_AUTO_MODERATOR_PROMPT),
    ]),
    ("Memory & Training", [
        ("LTM Summarization", "LTM_SUMMARIZATION_INSTRUCTIONS", DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS),
        ("Training Analyst", "TRAINING_ANALYST", DEFAULT_TRAINING_ANALYST_PROMPT),
    ]),
    ("Bot Utilities", [
        ("Profile Generator", "PROFILE_GENERATOR", DEFAULT_PROFILE_GENERATOR_PROMPT),
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
    "NEGATIVE_CONSTRAINTS": ({"constraints"}, "format"),
    "TRAINING_DATA_INJECTION": ({"examples_block"}, "format"),
    "NEURO_ENGINE": ({"d", "c", "o", "a"}, "format"),
    "WHISPER_INJECTION": ({"whisper_content"}, "format"),
    "WHISPER_RECAP": ({"whispers"}, "format"),
    "DIRECTOR_USER_PROMPT": ({"history"}, "format"),
    "IMAGE_PRESENT": ({"prompt"}, "format"),
    "IMAGE_PRESENT_OTHER": ({"name", "prompt"}, "format"),
    "IMAGE_FAILED": ({"prompt", "reason"}, "format"),
    "IMAGE_APPEARANCE": ({"appearance", "prompt"}, "format"),
    "IMAGE_GROUNDING": ({"prompt", "grounding"}, "format"),
    "GROUNDING_RAG_PAYLOAD": ({"transcript", "query"}, "format"),
    "ANTI_REPETITION": ({"char_name"}, "format"),
    "TRAINING_ANALYST": ({"verbosity", "examples_block"}, "format"),
    "PROFILE_GENERATOR": ({"prompt"}, "format"),
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

    Prompts outgrew the single 4000-character text input this modal used to be:
    DEFAULT_HELP_MODE_INJECTION is 4703 characters, and Discord rejects a modal
    whose default value exceeds the field's max_length, so that entry could not
    be opened at all.

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
        cat_sel = ui.Select(placeholder="Select a category...", options=cat_opts, row=0)

        async def cat_cb(i: discord.Interaction):
            self.selected_category = i.data['values'][0]
            self._build_view()
            await i.response.edit_message(embed=self._get_embed(), view=self)

        cat_sel.callback = cat_cb
        self.add_item(cat_sel)

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

        sel = ui.Select(placeholder="Select a prompt to edit...", options=options, row=1)

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

        sel.callback = sel_cb
        self.add_item(sel)
        self._add_nav_buttons()

    def _get_embed(self):
        overridden = [lbl for lbl, key, _ in MOD_PROMPT_DEFINITIONS if key in self.cog.global_prompts]
        embed = discord.Embed(
            title="Global System Prompts",
            description=("Modify the internal hardcoded instructions. Leave a prompt completely "
                         "blank to revert to its default value."),
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
