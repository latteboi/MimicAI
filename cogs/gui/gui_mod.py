from ..utils.constants import *

import discord
from discord import ui
import datetime
from typing import TYPE_CHECKING
from ..utils.helpers import _sanitise_filename
from .base_components import build_pagination_controls, build_tab_nav_bar, build_confirm_view

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog

from .gui_profiles import ProfileManageView

class ModBaseView(ui.View):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, current_tab: str):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.current_tab = current_tab
        self._add_nav_buttons()

    def _add_nav_buttons(self):
        build_tab_nav_bar(self, self.current_tab, [
            ("Stats", "stats", self.nav_stats),
            ("Profiles", "profiles", self.nav_profiles),
            ("Prompts", "prompts", self.nav_prompts),
            ("Docs", "docs", self.nav_docs),
            ("Blacklist", "blacklist", self.nav_blacklist),
        ])

    async def nav_stats(self, i: discord.Interaction):
        await i.response.defer(); view = ModStatsView(self.cog, self.original_interaction); await view.update_display()

    async def nav_profiles(self, i: discord.Interaction):
        await i.response.defer(); view = ModProfilesView(self.cog, self.original_interaction); await view.update_display()

    async def nav_prompts(self, i: discord.Interaction):
        await i.response.defer(); view = ModPromptsView(self.cog, self.original_interaction); await view.update_display()
        
    async def nav_docs(self, i: discord.Interaction):
        await i.response.defer(); view = ModDocsView(self.cog, self.original_interaction); await view.update_display()
        
    async def nav_blacklist(self, i: discord.Interaction):
        await i.response.defer(); view = ModBlacklistView(self.cog, self.original_interaction); await view.update_display()
        
    @staticmethod
    def add_nav_to_other_view(target_view, cog, interaction, current_tab):
        async def nav_stats(i: discord.Interaction):
            await i.response.defer(); view = ModStatsView(cog, interaction); await view.update_display()
        async def nav_profiles(i: discord.Interaction):
            await i.response.defer(); view = ModProfilesView(cog, interaction); await view.update_display()
        async def nav_prompts(i: discord.Interaction):
            await i.response.defer(); view = ModPromptsView(cog, interaction); await view.update_display()
        async def nav_docs(i: discord.Interaction):
            await i.response.defer(); view = ModDocsView(cog, interaction); await view.update_display()
        async def nav_blacklist(i: discord.Interaction):
            await i.response.defer(); view = ModBlacklistView(cog, interaction); await view.update_display()

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
    def __init__(self, cog, interaction):
        super().__init__(cog, interaction, "docs")
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
                async def p_cb(i: discord.Interaction):
                    self.current_page -= 1
                    self._build_view()
                    await i.response.defer()
                    await self.update_display()
                async def n_cb(i: discord.Interaction):
                    self.current_page += 1
                    self._build_view()
                    await i.response.defer()
                    await self.update_display()
                    
                build_pagination_controls(self, self.current_page, num_pages, 2, p_cb, n_cb)

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
    def __init__(self, cog, interaction):
        super().__init__(cog, interaction, "blacklist")
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
                async def p_cb(i: discord.Interaction):
                    self.current_page -= 1
                    self._build_view()
                    await i.response.edit_message(embed=self._get_embed(), view=self)
                async def n_cb(i: discord.Interaction):
                    self.current_page += 1
                    self._build_view()
                    await i.response.edit_message(embed=self._get_embed(), view=self)
                    
                build_pagination_controls(self, self.current_page, num_pages, 2, p_cb, n_cb)

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
    def __init__(self, cog, interaction):
        super().__init__(cog, interaction, "stats")
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

        async def p_cb(i: discord.Interaction):
            self.current_page -= 1
            self._build_view()
            await i.response.edit_message(embed=self._get_embed(), view=self)
        async def n_cb(i: discord.Interaction):
            self.current_page += 1
            self._build_view()
            await i.response.edit_message(embed=self._get_embed(), view=self)

        build_pagination_controls(self, self.current_page, num_pages, 1, p_cb, n_cb)
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
            await i.response.send_message("Invalid ID.", ephemeral=True)
            return
        self.parent_view.target_user_id = int(uid_str)
        self.parent_view._build_view()
        await i.response.edit_message(embed=self.parent_view._get_embed(), view=self.parent_view)

class ModProfilesView(ModBaseView):
    def __init__(self, cog, interaction):
        super().__init__(cog, interaction, "profiles")
        self.target_user_id = None
        self.current_page = 0
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        btn_enter = ui.Button(label="Enter User ID", style=discord.ButtonStyle.success, row=0)
        async def enter_cb(i: discord.Interaction):
            await i.response.send_modal(ModProfilesModal(self))
        btn_enter.callback = enter_cb
        self.add_item(btn_enter)
        
        if self.target_user_id:
            index = self.cog.profile_manager._get_user_index(self.target_user_id)
            profiles = list(index.get("personal", [])) + list(index.get("borrowed", []))
            profiles.sort()
            
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
                    async def p_cb(i: discord.Interaction):
                        self.current_page -= 1
                        self._build_view()
                        await i.response.edit_message(embed=self._get_embed(), view=self)
                    async def n_cb(i: discord.Interaction):
                        self.current_page += 1
                        self._build_view()
                        await i.response.edit_message(embed=self._get_embed(), view=self)
                        
                    build_pagination_controls(self, self.current_page, num_pages, 2, p_cb, n_cb)

        self._add_nav_buttons()

    def _get_embed(self):
        embed = discord.Embed(title="Moderator Profile Dashboard", color=discord.Color.red())
        if self.target_user_id:
            user = self.cog.bot.get_user(self.target_user_id)
            uname = user.name if user else "Unknown"
            embed.description = f"Managing User: **{uname}** (`{self.target_user_id}`)\nSelect a profile below."
        else:
            embed.description = "Click the button below to enter a User ID to manage."
        return embed

    async def update_display(self):
        await self.original_interaction.edit_original_response(embed=self._get_embed(), view=self)

class ModPromptModal(ui.Modal, title="Edit Global Prompt"):
    def __init__(self, view, key, default_text):
        super().__init__()
        self.parent_view = view
        self.key = key
        
        curr_val = self.parent_view.cog.global_prompts.get(key, default_text)
        self.prompt_input = ui.TextInput(
            label="Prompt (Blank to reset to default)", 
            style=discord.TextStyle.paragraph, 
            default=curr_val, 
            required=False,
            max_length=4000
        )
        self.add_item(self.prompt_input)

    async def on_submit(self, i: discord.Interaction):
        val = self.prompt_input.value.strip()
        if val:
            self.parent_view.cog.global_prompts[self.key] = val
        else:
            self.parent_view.cog.global_prompts.pop(self.key, None)
        
        self.parent_view.cog.server_manager._save_global_prompts()
        await i.response.send_message(f"Updated `{self.key}` successfully.", ephemeral=True)

class ModPromptsView(ModBaseView):
    def __init__(self, cog, interaction):
        super().__init__(cog, interaction, "prompts")
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        prompt_keys =[
            ("LTM Summarization", "LTM_SUMMARIZATION_INSTRUCTIONS", DEFAULT_LTM_SUMMARIZATION_INSTRUCTIONS),
            ("Context Rules", "CONTEXT_RULES", DEFAULT_CONTEXT_RULES),
            ("Training Data Injection", "TRAINING_DATA_INJECTION", DEFAULT_TRAINING_DATA_INJECTION),
            ("Auto-Moderator Critic", "AUTO_MODERATOR", DEFAULT_AUTO_MODERATOR_PROMPT),
            ("Anti-Repetition Critic", "ANTI_REPETITION", DEFAULT_ANTI_REPETITION_PROMPT),
            ("Web Grounding (Text)", "WEB_GROUNDING_TEXT", DEFAULT_WEB_GROUNDING_TEXT),
            ("Web Grounding (Visual)", "WEB_GROUNDING_VISUAL", DEFAULT_WEB_GROUNDING_VISUAL),
            ("Profile Generator", "PROFILE_GENERATOR", DEFAULT_PROFILE_GENERATOR_PROMPT),
            ("Training Analyst", "TRAINING_ANALYST", DEFAULT_TRAINING_ANALYST_PROMPT),
            ("Whisper Injection", "WHISPER_INJECTION", DEFAULT_WHISPER_INJECTION),
            ("Neuro-Endocrine Engine", "NEURO_ENGINE", DEFAULT_NEURO_INSTRUCTION),
            ("Help Mode Protocol", "HELP_MODE_INJECTION", DEFAULT_HELP_MODE_INJECTION)
        ]
        
        options = [discord.SelectOption(label=lbl, value=key) for lbl, key, _ in prompt_keys]
        sel = ui.Select(placeholder="Select a prompt to edit...", options=options, row=0)
        
        async def sel_cb(i: discord.Interaction):
            key = i.data['values'][0]
            default_text = next(d for l, k, d in prompt_keys if k == key)
            await i.response.send_modal(ModPromptModal(self, key, default_text))
            
        sel.callback = sel_cb
        self.add_item(sel)
        self._add_nav_buttons()

    def _get_embed(self):
        embed = discord.Embed(title="Global System Prompts", description="Modify the internal hardcoded instructions. Leave a prompt completely blank to revert to its default value.", color=discord.Color.purple())
        return embed

    async def update_display(self):
        await self.original_interaction.edit_original_response(embed=self._get_embed(), view=self)
