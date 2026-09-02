from ..utils.constants import *

import discord
from discord import ui
import asyncio
from typing import TYPE_CHECKING, List, Optional

from .base_components import TimeoutCleanupMixin, build_tab_nav_bar

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

    def __init__(self, cog: 'MimicCog', slot_id: str, provider: str, view: Optional[ui.View] = None):
        super().__init__()
        self.cog = cog
        self.slot_id = slot_id
        self.provider = provider
        self.view = view

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
            await interaction.followup.send(f"❌ **Validation Failed:** {err}", ephemeral=True)
            return
        
        user_data = self.cog.storage_manager._get_user_keys_data(interaction.user.id)
        user_data.setdefault("slots", {})[self.slot_id] = {
            "key": raw_key,
            "provider": self.provider,
            "tier": tier
        }
        self.cog.storage_manager._save_user_keys_data(interaction.user.id, user_data)
        
        self.cog.decrypted_key_cache[(interaction.user.id, self.slot_id)] = raw_key
        
        idx = self.cog.profile_manager._get_user_index(interaction.user.id)
        idx["has_personal_key"] = True
        self.cog.profile_manager._save_user_index(interaction.user.id, idx)
        
        msg = f"✅ {self.provider.title()} key saved to slot `{self.slot_id}` ({tier.title()} Tier)."

        if self.view:
            self.view.setup_items()
            await self.view.update_display()
            
        await interaction.followup.send(msg, ephemeral=True)

class OverrideConfirmView(ui.View):
    def __init__(self, cog: 'MimicCog', user_id: int, slot_id: str, provider: str, new_scopes: List[str], parent_view: ui.View):
        super().__init__(timeout=120)
        self.cog = cog
        self.user_id = user_id
        self.slot_id = slot_id
        self.provider = provider
        self.new_scopes = new_scopes
        self.parent_view = parent_view

    @ui.button(label="Yes, Override", style=discord.ButtonStyle.danger)
    async def confirm_override(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.defer(ephemeral=True)
        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        
        if "personal" in self.new_scopes:
            user_data.setdefault("personal_assignments", {})[self.provider] = self.slot_id
        else:
            if user_data.get("personal_assignments", {}).get(self.provider) == self.slot_id:
                del user_data["personal_assignments"][self.provider]
                
        self.cog.storage_manager._save_user_keys_data(self.user_id, user_data)
        
        for scope in self.new_scopes:
            if scope != "personal":
                server_index = self.cog.server_manager._get_server_index(scope)
                server_index.setdefault("assigned_keys", {})[self.provider] = {"user_id": self.user_id, "slot": self.slot_id}
                self.cog.server_manager._save_server_index(scope, server_index)
                self.cog.server_key_pointers[(int(scope), self.provider)] = (self.user_id, self.slot_id)
                
        for guild in self.cog.bot.guilds:
            guild_id_str = str(guild.id)
            if guild_id_str not in self.new_scopes:
                server_index = self.cog.server_manager._get_server_index(guild_id_str)
                assigned = server_index.get("assigned_keys", {}).get(self.provider)
                if assigned and assigned.get("user_id") == self.user_id and assigned.get("slot") == self.slot_id:
                    del server_index["assigned_keys"][self.provider]
                    self.cog.server_manager._save_server_index(guild_id_str, server_index)
                    self.cog.server_key_pointers.pop((guild.id, self.provider), None)

        self.parent_view.setup_items()
        await self.parent_view.update_display()
        await interaction.edit_original_response(content="✅ Assignments saved successfully.", view=None)

    @ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel_override(self, interaction: discord.Interaction, button: ui.Button):
        await interaction.response.edit_message(content="❌ Assignment update cancelled.", view=None)

class SettingsBaseView(TimeoutCleanupMixin, ui.View):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, current_tab: str):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.current_tab = current_tab
        self._add_nav_buttons()

    def _add_nav_buttons(self):
        build_tab_nav_bar(self, self.current_tab, [
            ("Home", "home", self.nav_home),
            ("API Keys", "api", self.nav_api),
            ("Child Bots", "bots", self.nav_bots),
        ])

    async def nav_home(self, i: discord.Interaction):
        await i.response.defer()
        view = SettingsHomeView(self.cog, self.original_interaction)
        await view.update_display()

    async def nav_api(self, i: discord.Interaction):
        await i.response.defer()
        view = SettingsAPIView(self.cog, self.original_interaction)
        await view.update_display()

    async def nav_bots(self, i: discord.Interaction):
        await i.response.defer()
        view = SettingsChildBotView(self.cog, self.original_interaction)
        await view.update_display()

class SettingsHomeView(SettingsBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "home")

    async def update_display(self):
        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        slots = user_data.get("slots", {})
        
        has_gem = any(s.get("provider") == "gemini" for s in slots.values())
        has_or = any(s.get("provider") == "openrouter" for s in slots.values())
        
        stat_gemini = f"✅ **`Set`**" if has_gem else "❌ `Not Set`"
        stat_or = f"✅ **`Set`**" if has_or else "❌ `Not Set`"
        
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

class SettingsAPIView(SettingsBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "api")
        self.selected_slot = None
        self.selected_scopes = set()
        self.slots_config = [
            ("google_key_1", "Google Gemini Key 1", "gemini"),
            ("google_key_2", "Google Gemini Key 2", "gemini"),
            ("openrouter_key_1", "OpenRouter Key 1", "openrouter"),
            ("openrouter_key_2", "OpenRouter Key 2", "openrouter")
        ]
        self.admin_guilds = []
        for g in self.cog.bot.guilds:
            m = g.get_member(self.user_id)
            if m and m.guild_permissions.administrator:
                self.admin_guilds.append(g)
        self.setup_items()

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        slots_data = user_data.get("slots", {})

        # Row 0: Slot Selection
        slot_options = []
        for slot_id, label, provider in self.slots_config:
            data = slots_data.get(slot_id)
            if data:
                tier = data.get("tier", "free").title()
                desc = f"Set ({tier} Tier)"
                emoji = "🟢" if provider == "gemini" else "🟣"
            else:
                desc = "Empty"
                emoji = "⚪"
            slot_options.append(discord.SelectOption(label=label, value=slot_id, description=desc, emoji=emoji, default=(self.selected_slot == slot_id)))

        slot_select = ui.Select(placeholder="Select an API Key Slot...", options=slot_options, row=0)
        slot_select.callback = self.slot_select_callback
        self.add_item(slot_select)

        # Row 1: Scope Multi-Select
        if self.selected_slot and self.selected_slot in slots_data:
            provider = next(p for s, l, p in self.slots_config if s == self.selected_slot)
            
            current_assignments = []
            if user_data.get("personal_assignments", {}).get(provider) == self.selected_slot:
                current_assignments.append("personal")
                
            for g in self.admin_guilds:
                idx = self.cog.server_manager._get_server_index(str(g.id))
                assigned = idx.get("assigned_keys", {}).get(provider)
                if assigned and assigned.get("user_id") == self.user_id and assigned.get("slot") == self.selected_slot:
                    current_assignments.append(str(g.id))
            
            self.selected_scopes = set(current_assignments)
            
            scope_options = [discord.SelectOption(label="Personal", value="personal", default=("personal" in self.selected_scopes))]
            for g in self.admin_guilds[:24]:
                scope_options.append(discord.SelectOption(label=f"Server: {g.name}"[:100], value=str(g.id), default=(str(g.id) in self.selected_scopes)))
                
            if scope_options:
                scope_select = ui.Select(placeholder="Assign this key to...", options=scope_options, min_values=0, max_values=len(scope_options), row=1)
                scope_select.callback = self.scope_select_callback
                self.add_item(scope_select)

            # Row 2: Actions
            btn_edit = ui.Button(label="Edit Key", style=discord.ButtonStyle.primary, row=2)
            btn_edit.callback = self.edit_key_callback
            self.add_item(btn_edit)
            
            btn_del = ui.Button(label="Delete Key", style=discord.ButtonStyle.danger, row=2)
            btn_del.callback = self.delete_key_callback
            self.add_item(btn_del)
            
            btn_save = ui.Button(label="Save Assignments", style=discord.ButtonStyle.success, row=2)
            btn_save.callback = self.save_assignments_callback
            self.add_item(btn_save)
        elif self.selected_slot:
            btn_add = ui.Button(label="Submit Key", style=discord.ButtonStyle.success, row=2)
            btn_add.callback = self.edit_key_callback
            self.add_item(btn_add)

    async def update_display(self):
        embed = discord.Embed(title="API Key Management", description="Manage your 4 API key slots and assign them to your Personal account or Servers you administrate.", color=discord.Color.blue())
        
        if self.selected_slot:
            user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
            slot_data = user_data.get("slots", {}).get(self.selected_slot)
            label = next(l for s, l, p in self.slots_config if s == self.selected_slot)
            
            if slot_data:
                tier = slot_data.get("tier", "free").title()
                embed.add_field(name=f"Slot: {label}", value=f"**Status:** ✅ Set\n**Tier:** `{tier}`", inline=False)
                
                provider = next(p for s, l, p in self.slots_config if s == self.selected_slot)
                assignments = []
                if user_data.get("personal_assignments", {}).get(provider) == self.selected_slot:
                    assignments.append("Personal")
                for g in self.admin_guilds:
                    idx = self.cog.server_manager._get_server_index(str(g.id))
                    assigned = idx.get("assigned_keys", {}).get(provider)
                    if assigned and assigned.get("user_id") == self.user_id and assigned.get("slot") == self.selected_slot:
                        assignments.append(f"Server: {g.name}")
                
                assign_str = "\n".join(f"- {a}" for a in assignments) if assignments else "None"
                embed.add_field(name="Current Assignments", value=assign_str, inline=False)
            else:
                embed.add_field(name=f"Slot: {label}", value="**Status:** ❌ Empty\nClick 'Submit Key' to add a key to this slot.", inline=False)
        else:
            embed.add_field(name="Overview", value="Select a slot from the dropdown above to view or manage it.", inline=False)

        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def slot_select_callback(self, interaction: discord.Interaction):
        self.selected_slot = interaction.data['values'][0]
        self.setup_items()
        await interaction.response.defer()
        await self.update_display()

    async def scope_select_callback(self, interaction: discord.Interaction):
        self.selected_scopes = set(interaction.data['values'])
        await interaction.response.defer()

    async def edit_key_callback(self, interaction: discord.Interaction):
        provider = next(p for s, l, p in self.slots_config if s == self.selected_slot)
        modal = SubmitAPIKeyModal(self.cog, self.selected_slot, provider, view=self)
        await interaction.response.send_modal(modal)

    async def delete_key_callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        provider = next(p for s, l, p in self.slots_config if s == self.selected_slot)
        
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
                
        self.setup_items()
        await self.update_display()
        await interaction.followup.send("✅ Key and all its assignments deleted.", ephemeral=True)

    async def save_assignments_callback(self, interaction: discord.Interaction):
        provider = next(p for s, l, p in self.slots_config if s == self.selected_slot)
        user_data = self.cog.storage_manager._get_user_keys_data(self.user_id)
        
        conflicts = []
        
        if "personal" in self.selected_scopes:
            curr_personal = user_data.get("personal_assignments", {}).get(provider)
            if curr_personal and curr_personal != self.selected_slot:
                conflicts.append(f"Personal Scope (Currently uses {curr_personal})")
                
        for scope in self.selected_scopes:
            if scope != "personal":
                server_index = self.cog.server_manager._get_server_index(scope)
                assigned = server_index.get("assigned_keys", {}).get(provider)
                if assigned and (assigned.get("user_id") != self.user_id or assigned.get("slot") != self.selected_slot):
                    guild = self.cog.bot.get_guild(int(scope))
                    g_name = guild.name if guild else scope
                    conflicts.append(f"Server: {g_name} (Currently assigned by another key/user)")
                    
        if conflicts:
            conflict_str = "\n".join(f"- {c}" for c in conflicts)
            msg = f"⚠️ **Key Assignment Override**\nAssigning this key will overwrite existing assignments for the following scopes:\n{conflict_str}\n\nDo you want to proceed?"
            view = OverrideConfirmView(self.cog, self.user_id, self.selected_slot, provider, list(self.selected_scopes), self)
            await interaction.response.send_message(msg, view=view, ephemeral=True)
            return
            
        await interaction.response.defer(ephemeral=True)
        
        if "personal" in self.selected_scopes:
            user_data.setdefault("personal_assignments", {})[provider] = self.selected_slot
        else:
            if user_data.get("personal_assignments", {}).get(provider) == self.selected_slot:
                del user_data["personal_assignments"][provider]
                
        self.cog.storage_manager._save_user_keys_data(self.user_id, user_data)
        
        for scope in self.selected_scopes:
            if scope != "personal":
                server_index = self.cog.server_manager._get_server_index(scope)
                server_index.setdefault("assigned_keys", {})[provider] = {"user_id": self.user_id, "slot": self.selected_slot}
                self.cog.server_manager._save_server_index(scope, server_index)
                self.cog.server_key_pointers[(int(scope), provider)] = (self.user_id, self.selected_slot)
                
        for guild in self.admin_guilds:
            guild_id_str = str(guild.id)
            if guild_id_str not in self.selected_scopes:
                server_index = self.cog.server_manager._get_server_index(guild_id_str)
                assigned = server_index.get("assigned_keys", {}).get(provider)
                if assigned and assigned.get("user_id") == self.user_id and assigned.get("slot") == self.selected_slot:
                    del server_index["assigned_keys"][provider]
                    self.cog.server_manager._save_server_index(guild_id_str, server_index)
                    self.cog.server_key_pointers.pop((guild.id, provider), None)

        self.setup_items()
        await self.update_display()
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
            select = ui.Select(placeholder="Select a child bot...", options=options[:25], row=0)
            select.callback = self.select_bot
            self.add_item(select)

        # Row 1: Actions
        btn_create = ui.Button(label="Create New Child Bot", style=discord.ButtonStyle.green, row=1)
        btn_create.callback = self.create_bot
        self.add_item(btn_create)

        if self.selected_bot_id:
            # [REMOVED] Manage Approved Servers button
            
            btn_del = ui.Button(label="Unlink & Delete", style=discord.ButtonStyle.danger, row=1)
            btn_del.callback = self.delete_bot
            self.add_item(btn_del)

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
            await interaction.followup.send(f"Error: An unexpected error occurred while validating the token: {e}", ephemeral=True)
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

class ParentPresenceView(ui.View):
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

        clear_btn = ui.Button(label="Clear Activity", style=discord.ButtonStyle.danger, row=2)
        clear_btn.callback = self.clear_callback
        self.add_item(clear_btn)

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

class ShutdownConfirmView(ui.View):
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
