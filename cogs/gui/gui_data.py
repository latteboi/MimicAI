from ..utils.constants import *

import discord
from discord import ui
import datetime
import traceback
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional
from ..managers.memory_manager import encode_embedding_b64

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog
    from .gui_profiles import ProfileManageView

from .base_components import BaseBulkProfileView, PageJumpModal, build_confirm_view, compute_window_slice

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
        
        guild_id = i.guild_id
        if not guild_id:
            user = self.cog.bot.get_user(i.user.id)
            if user:
                for guild in self.cog.bot.guilds:
                    if guild.get_member(user.id):
                        guild_id = guild.id
                        break
        if not guild_id:
            await i.followup.send("Could not determine a valid context to get an API key. Please try editing from a server.", ephemeral=True)
            return

        new_embedding = await self.cog.memory_manager._get_embedding(new_summary, guild_id, task_type="RETRIEVAL_DOCUMENT")
        if not new_embedding:
            await i.followup.send("Failed to generate embedding for the new summary. The memory was not updated.", ephemeral=True)
            return

        b64_emb = encode_embedding_b64(new_embedding)
        success = self.cog.memory_manager.update_ltm(self.profile_owner_id, self.profile_name, self.ltm_id, new_summary, b64_emb)
        if success:
            await i.followup.send(f"LTM entry `{self.ltm_id}` for profile '{self.profile_name}' has been updated.", ephemeral=True)
        else:
            await i.followup.send(f"Failed to find and update LTM entry `{self.ltm_id}`.", ephemeral=True)
    
    async def on_error(self, i: discord.Interaction, e: Exception):
        print(f"EditLtmModal error: {e}"); traceback.print_exc()
        await i.followup.send("An error occurred with the LTM edit form.", ephemeral=True)

class AddLtmModal(ui.Modal, title="Add Long-Term Memory"):
    summary_field = ui.TextInput(label="Memory Summary", style=discord.TextStyle.paragraph, required=True, max_length=2000)

    def __init__(self, cog, profile_owner_id: int, profile_name: str, guild_id: int):
        super().__init__()
        self.cog: MimicCog = cog
        self.profile_owner_id = profile_owner_id
        self.profile_name = profile_name
        self.guild_id = guild_id

    async def on_submit(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True, thinking=True)
        
        # [NEW] Manual Hard Block Check
        user_id_str = str(self.profile_owner_id)
        ltm_shard = self.cog.memory_manager._load_ltm_shard(user_id_str, self.profile_name)
        current_count = len(ltm_shard.get("guild", [])) if ltm_shard else 0
        
        limit = defaultConfig.LIMIT_LTM

        if current_count >= limit:
            msg = f"**Limit Reached.**\n"
            msg += f"You have **{current_count}** memories (Limit: {limit}).\n"
            msg += "You cannot manually add more memories while at or above the limit. Please delete old memories first."
            await i.followup.send(msg, ephemeral=True)
            return

        summary = self.summary_field.value
        
        embedding = await self.cog.memory_manager._get_embedding(summary, self.guild_id, task_type="RETRIEVAL_DOCUMENT")
        if not embedding:
            await i.followup.send("Failed to generate embedding for the summary. The memory was not added.", ephemeral=True)
            return

        b64_emb = encode_embedding_b64(embedding)
        
        # The _add_ltm method now handles the rolling window logic automatically.
        self.cog.memory_manager._add_ltm(self.profile_owner_id, self.profile_name, summary, b64_emb, self.guild_id, i.user.id, i.user.display_name)
        
        # Fetch new count for feedback
        ltm_shard = self.cog.memory_manager._load_ltm_shard(str(self.profile_owner_id), self.profile_name)
        count = len(ltm_shard.get("guild", [])) if ltm_shard else 0
        limit = defaultConfig.LIMIT_LTM
        
        msg = f"LTM entry added for '{self.profile_name}'."
        if count >= limit:
            msg += f"\nNote: You have reached the {limit} memory limit. The oldest memory was automatically replaced."
            
        await i.followup.send(msg, ephemeral=True)

    async def on_error(self, i: discord.Interaction, e: Exception):
        print(f"AddLtmModal error: {e}"); traceback.print_exc()
        await i.followup.send("An error occurred with the LTM add form.", ephemeral=True)

class AddTrainingExampleModal(ui.Modal, title="Add Profile Training Example"): 
    user_input_field=ui.TextInput(label="User Input Example",style=discord.TextStyle.paragraph,required=True,max_length=1000)
    chatbot_response_field=ui.TextInput(label="Desired Chatbot Response",style=discord.TextStyle.paragraph,required=True,max_length=2000)
    def __init__(self, cog, profile_owner_id: int, profile_name: str, guild_id: int):
        super().__init__()
        self.cog:MimicCog=cog
        self.profile_owner_id = profile_owner_id
        self.profile_name = profile_name
        self.guild_id = guild_id
    async def on_submit(self,i:discord.Interaction):
        await i.response.defer(ephemeral=True,thinking=True)
        
        # [NEW] Manual Hard Block Check
        # Although add_new_training_example has a check, we do it here to provide a better UI response
        # and prevent the embedding API call if blocked.
        user_id_str = str(self.profile_owner_id)
        training_shard = self.cog.memory_manager._load_training_shard(user_id_str, self.profile_name) or []
        current_count = len(training_shard)
        
        limit = defaultConfig.LIMIT_TRAINING

        if current_count >= limit:
            msg = f"**Limit Reached.**\n"
            msg += f"You have **{current_count}** training examples (Limit: {limit}).\n"
            msg += "You cannot add more examples. Please delete existing ones first."
            await i.followup.send(msg, ephemeral=True)
            return

        s,m=await self.cog.memory_manager.add_new_training_example(self.profile_owner_id, self.profile_name, self.user_input_field.value, self.chatbot_response_field.value, self.guild_id)
        await i.followup.send(m,ephemeral=True)
    async def on_error(self,i:discord.Interaction,e:Exception):print(f"AddTrainExModal err:{e}");traceback.print_exc();await i.followup.send('Oops!',ephemeral=True)

class EditTrainingExampleModal(ui.Modal, title="Edit Profile Training Example"):
    user_input_field = ui.TextInput(label="User Input Example", style=discord.TextStyle.paragraph, required=True, max_length=1000)
    chatbot_response_field = ui.TextInput(label="Desired Chatbot Response", style=discord.TextStyle.paragraph, required=True, max_length=2000)

    def __init__(self, cog, profile_owner_id: int, profile_name: str, example_id: str, current_user_input: str, current_bot_response: str, guild_id: int):
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

class SearchDataModal(ui.Modal, title="Search Data"):
    search_input = ui.TextInput(label="Enter search term (leave blank to clear)", required=False, max_length=100)

    def __init__(self, parent_view: 'DataManageView'):
        super().__init__()
        self.parent_view = parent_view
        if self.parent_view.search_term:
            self.search_input.default = self.parent_view.search_term

    async def on_submit(self, interaction: discord.Interaction):
        search_term = self.search_input.value.strip()
        self.parent_view.search_term = search_term if search_term else None
        self.parent_view.current_page = 1
        await self.parent_view._update_view(interaction)

class DataManageView(ui.View):
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
        
        if mode:
            self.mode: Literal['training', 'ltm'] = mode
        else:
            self.mode: Literal['training', 'ltm'] = 'ltm' if self.is_borrowed else 'training'
        self.current_page = 1
        self.items_per_page = 1
        self.max_pages = 1
        self.current_item_id: Optional[str] = None
        self.full_data_list: List[Dict] = []
        self.search_term: Optional[str] = None
        self.displayed_data_list: List[Dict] = []
        self.ltm_filter: Optional[str] = "all"

    async def start(self):
        await self._update_view(self.original_interaction)

    async def _update_view(self, interaction: discord.Interaction):
        embed, page_items, ltm_filter_options = await self._build_embed()
        self._build_components(page_items, ltm_filter_options)
        if interaction.response.is_done():
            await interaction.edit_original_response(embed=embed, view=self)
        else:
            await interaction.response.edit_message(embed=embed, view=self)

    async def _build_embed(self) -> Tuple[discord.Embed, List[Dict], List[discord.SelectOption]]:
        user_id_str = str(self.user_id)
        title_prefix = ""
        ltm_filter_options = []

        if self.mode == 'training':
            self.full_data_list = self.cog.memory_manager._load_training_shard(user_id_str, self.profile_name) or []
            title_prefix = "Training Examples"
            self.displayed_data_list = self.full_data_list
        else: # ltm
            ltm_shard = self.cog.memory_manager._load_ltm_shard(user_id_str, self.profile_name)
            self.full_data_list = ltm_shard.get("guild", []) if ltm_shard else []
            title_prefix = "Long-Term Memories"

            server_filters = {}
            for item in self.full_data_list:
                server_id = item.get('context_id')
                if server_id and server_id not in server_filters:
                    try:
                        guild = self.cog.bot.get_guild(int(server_id))
                        server_filters[server_id] = guild.name if guild else f"Server ID: {server_id}"
                    except (ValueError, TypeError):
                        continue
            
            ltm_filter_options.append(discord.SelectOption(label="All Memories", value="all"))
            for server_id, server_name in sorted(server_filters.items(), key=lambda item: item[1]):
                ltm_filter_options.append(discord.SelectOption(label=f"Server: {server_name}", value=f"server_{server_id}"))

            for option in ltm_filter_options:
                if option.value == self.ltm_filter:
                    option.default = True

            if self.ltm_filter == "all":
                self.displayed_data_list = self.full_data_list
            elif self.ltm_filter and self.ltm_filter.startswith("server_"):
                filter_server_id = self.ltm_filter.split("_", 1)[1]
                self.displayed_data_list =[item for item in self.full_data_list if str(item.get('context_id')) == filter_server_id]
            else:
                self.displayed_data_list = self.full_data_list


        # After scope filtering, apply search term filtering on the result
        if self.search_term:
            search_term_lower = self.search_term.lower()
            
            # Note: We filter the already-scope-filtered 'displayed_data_list'
            search_filtered_list = []
            for item in self.displayed_data_list:
                content_to_search = ""
                if self.mode == 'training':
                    content_to_search = self.cog.storage_manager._decrypt_data(item.get('u_in', '')) + " " + self.cog.storage_manager._decrypt_data(item.get('b_out', ''))
                else: # ltm
                    content_to_search = self.cog.storage_manager._decrypt_data(item.get('sum', ''))
                
                if search_term_lower in content_to_search.lower():
                    search_filtered_list.append(item)
            self.displayed_data_list = search_filtered_list
        
        self.max_pages = len(self.displayed_data_list) or 1
        self.current_page = max(1, min(self.current_page, self.max_pages))
        start_index = self.current_page - 1
        
        page_items = self.displayed_data_list[start_index : start_index + 1]
        self.current_item_id = page_items[0].get('id') if page_items else None

        embed = discord.Embed(title=f"{title_prefix} for '{self.profile_name}'", color=discord.Color.dark_teal())
        embed.set_footer(text=f"Item {self.current_page}/{self.max_pages} | Total: {len(self.full_data_list)}")

        if not page_items:
            embed.description = f"No {title_prefix.lower()} found."
        else:
            item = page_items[0]
            item_id = item.get('id', 'N/A')

            created_ts_str = item.get('created_ts') or item.get('ts')
            modified_ts_str = item.get('modified_ts')
            ts_display = ""
            
            created_dt = None
            if created_ts_str:
                try:
                    created_dt = datetime.datetime.fromisoformat(created_ts_str)
                    ts_display += f" | Created: {created_dt.strftime('%d/%m/%y')} UTC"
                except ValueError:
                    pass

            if modified_ts_str:
                try:
                    modified_dt = datetime.datetime.fromisoformat(modified_ts_str)
                    if created_dt and (modified_dt - created_dt).total_seconds() > 5:
                        ts_display += f" | Modified: {modified_dt.strftime('%d/%m/%y')} UTC"
                except ValueError:
                    pass

            if self.mode == 'training':
                user_input = self.cog.storage_manager._decrypt_data(item.get('u_in', ''))
                bot_response = self.cog.storage_manager._decrypt_data(item.get('b_out', ''))
                embed.add_field(name=f"ID: `{item_id}`{ts_display}", value=f"**User Input:**\n{user_input}", inline=False)
                embed.add_field(name="Bot Response:", value=bot_response, inline=False)
            else: # ltm
                content = self.cog.storage_manager._decrypt_data(item.get('sum', ''))
                
                display_content = content
                if len(content) > 950:
                    display_content = content[:950] + "... (truncated)"

                embed.add_field(name=f"ID: `{item_id}`{ts_display}", value=f"**Summary:**\n{display_content}", inline=False)
        
        return embed, page_items, ltm_filter_options

    def _build_components(self, page_items: List[Dict], ltm_filter_options: List[discord.SelectOption]):
        self.clear_items()

        # Row 0: Navigation and Mode
        if not self.is_borrowed:
            training_button = ui.Button(label="Training", style=discord.ButtonStyle.green if self.mode == 'training' else discord.ButtonStyle.grey, custom_id="mode_training", row=0)
            training_button.callback = self.mode_button_callback
            self.add_item(training_button)

            ltm_button = ui.Button(label="LTMs", style=discord.ButtonStyle.green if self.mode == 'ltm' else discord.ButtonStyle.grey, custom_id="mode_ltm", row=0)
            ltm_button.callback = self.mode_button_callback
            self.add_item(ltm_button)

        prev_button = ui.Button(label="◀", style=discord.ButtonStyle.secondary, disabled=(self.current_page <= 1), row=0)
        prev_button.callback = self.prev_page_callback
        self.add_item(prev_button)

        page_button = ui.Button(label=f"{self.current_page}/{self.max_pages}", style=discord.ButtonStyle.secondary, row=0)
        page_button.callback = self.page_button_callback
        self.add_item(page_button)

        next_button = ui.Button(label="▶", style=discord.ButtonStyle.secondary, disabled=(self.current_page >= self.max_pages), row=0)
        next_button.callback = self.next_page_callback
        self.add_item(next_button)

        # [NEW] Move Analyse button to Row 1 (Only visible in training mode)
        if self.mode == 'training' and not self.is_borrowed:
            analyse_button = ui.Button(label="Analyse", style=discord.ButtonStyle.blurple, row=1)
            async def analyse_cb(i): await i.response.send_modal(AnalyseExamplesModal(self))
            analyse_button.callback = analyse_cb
            self.add_item(analyse_button)

        # Row 1: LTM Filter
        if self.mode == 'ltm' and ltm_filter_options:
            ltm_filter_select = ui.Select(placeholder="Filter memories by scope...", options=ltm_filter_options, row=1)
            ltm_filter_select.callback = self.ltm_filter_callback
            self.add_item(ltm_filter_select)

            # New sliding window logic
            start_slice_index, end_slice_index = compute_window_slice(self.current_page - 1, len(self.displayed_data_list))

            items_for_dropdown = self.displayed_data_list[start_slice_index:end_slice_index]

            options = []
            for i, item in enumerate(items_for_dropdown):
                item_id = item.get('id', 'N/A')
                absolute_page_number = start_slice_index + i + 1
                
                if self.mode == 'training':
                    content = self.cog.storage_manager._decrypt_data(item.get('u_in', ''))[:80]
                    label = f"Ex ({item_id}): {content}..."
                else:
                    content = self.cog.storage_manager._decrypt_data(item.get('sum', ''))[:80]
                    label = f"LTM ({item_id}): {content}..."
                
                option = discord.SelectOption(label=label, value=str(absolute_page_number))
                if absolute_page_number == self.current_page:
                    option.default = True
                options.append(option)

            if options:
                select = ui.Select(placeholder="Quick Navigation...", options=options, row=2)
                select.callback = self.select_callback
                self.add_item(select)

        # Row 3: Action Buttons
        search_button = ui.Button(label="🔍 Search", style=discord.ButtonStyle.secondary, row=3)
        search_button.callback = self.search_callback
        self.add_item(search_button)

        add_button = ui.Button(label="Add New", style=discord.ButtonStyle.success, row=3)
        add_button.callback = self.add_callback
        self.add_item(add_button)

        edit_button = ui.Button(label="Edit", style=discord.ButtonStyle.primary, row=3, disabled=(not page_items))
        edit_button.callback = self.edit_callback
        self.add_item(edit_button)

        delete_button = ui.Button(label="Delete", style=discord.ButtonStyle.danger, row=3, disabled=(not page_items))
        delete_button.callback = self.delete_callback
        self.add_item(delete_button)

        delete_all_button = ui.Button(label="Delete All (Filtered)", style=discord.ButtonStyle.danger, row=3, disabled=True)
        if self.mode == 'ltm' and self.ltm_filter and self.ltm_filter.startswith("server_") and self.displayed_data_list:
            delete_all_button.disabled = False
        delete_all_button.callback = self.delete_all_callback
        self.add_item(delete_all_button)

        if self.parent_manage_view:
            back_button = ui.Button(label="Back to Dashboard", style=discord.ButtonStyle.secondary, emoji="⬅️", row=4)
            async def back_callback(i: discord.Interaction):
                await i.response.defer()
                embed = await self.cog.profile_manager._build_profile_manage_embed(self.original_interaction, self.profile_name, target_user_id=self.parent_manage_view.user_id)
                self.parent_manage_view._build_view()
                await self.original_interaction.edit_original_response(embed=embed, view=self.parent_manage_view)
            back_button.callback = back_callback
            self.add_item(back_button)

    async def delete_all_callback(self, interaction: discord.Interaction):
        if not (self.mode == 'ltm' and self.ltm_filter and self.ltm_filter.startswith("server_")):
            await interaction.response.send_message("This action is only available when filtering LTMs by a specific server.", ephemeral=True)
            return

        items_to_delete = self.displayed_data_list
        if not items_to_delete:
            await interaction.response.send_message("There are no items matching the current filter to delete.", ephemeral=True)
            return

        async def confirm_action(i: discord.Interaction):
            owner_id_str = str(self.user_id)
            ltm_data = self.cog.memory_manager._load_ltm_shard(owner_id_str, self.profile_name)
            if not ltm_data:
                await i.response.edit_message(content="Could not load LTM data.", view=None)
                return

            ids_to_delete = {item['id'] for item in items_to_delete}
            context_type = "guild"
            original_list = ltm_data.get(context_type, [])

            new_list = [item for item in original_list if item.get("id") not in ids_to_delete]

            ltm_data[context_type] = new_list
            self.cog.memory_manager._save_ltm_shard(owner_id_str, self.profile_name, ltm_data)

            await i.response.edit_message(content=f"Successfully deleted {len(ids_to_delete)} LTM entries.", view=None)

            self.current_page = 1
            await self._update_view(self.original_interaction)

        confirm_view = build_confirm_view(f"Confirm Delete All ({len(items_to_delete)})", confirm_action)

        try:
            filter_server_id = self.ltm_filter.split("_", 1)[1]
            guild = self.cog.bot.get_guild(int(filter_server_id))
            server_name = guild.name if guild else f"ID: {filter_server_id}"
        except (IndexError, ValueError):
            server_name = "the selected server"

        await interaction.response.send_message(
            f"**Are you sure you want to delete all {len(items_to_delete)} LTMs for profile '{self.profile_name}' from '{server_name}'?**\nThis action is permanent.",
            view=confirm_view,
            ephemeral=True
        )


    async def ltm_filter_callback(self, interaction: discord.Interaction):
        self.ltm_filter = interaction.data['values'][0]
        self.current_page = 1
        await self._update_view(interaction)

    async def mode_button_callback(self, interaction: discord.Interaction):
        self.mode = 'ltm' if interaction.data['custom_id'] == 'mode_ltm' else 'training'
        self.current_page = 1
        await self._update_view(interaction)

    async def page_button_callback(self, interaction: discord.Interaction):
        async def _jump(i: discord.Interaction, page: int):
            self.current_page = page
            await self._update_view(i)

        await interaction.response.send_modal(PageJumpModal(self.max_pages, _jump))

    async def prev_page_callback(self, interaction: discord.Interaction):
        if self.current_page > 1:
            self.current_page -= 1
            await self._update_view(interaction)

    async def next_page_callback(self, interaction: discord.Interaction):
        if self.current_page < self.max_pages:
            self.current_page += 1
            await self._update_view(interaction)

    async def select_callback(self, interaction: discord.Interaction):
        self.current_page = int(interaction.data['values'][0])
        await self._update_view(interaction)

    async def add_callback(self, interaction: discord.Interaction):
        if self.mode == 'training':
            modal = AddTrainingExampleModal(self.cog, self.user_id, self.profile_name, self.guild_id)
        else: # ltm
            modal = AddLtmModal(self.cog, self.user_id, self.profile_name, self.guild_id)
        
        original_on_submit = modal.on_submit
        async def on_submit_refresh(i: discord.Interaction):
            await original_on_submit(i)
            if not i.response.is_done(): await i.response.defer()
            self.current_page = self.max_pages + 1 # Go to the new item
            await self._update_view(self.original_interaction)
        
        modal.on_submit = on_submit_refresh
        await interaction.response.send_modal(modal)

    async def edit_callback(self, interaction: discord.Interaction):
        if not self.current_item_id: return
        
        item_to_edit = next((item for item in self.full_data_list if item.get("id") == self.current_item_id), None)
        modal = None
        if item_to_edit:
            if self.mode == 'training':
                modal = EditTrainingExampleModal(self.cog, self.user_id, self.profile_name, self.current_item_id, self.cog.storage_manager._decrypt_data(item_to_edit.get("u_in", "")), self.cog.storage_manager._decrypt_data(item_to_edit.get("b_out", "")), self.guild_id)
            else: # ltm
                modal = EditLtmModal(self.cog, self.user_id, self.profile_name, self.current_item_id, self.cog.storage_manager._decrypt_data(item_to_edit.get("sum", "")))
        
        if modal:
            original_on_submit = modal.on_submit
            async def on_submit_refresh(i: discord.Interaction):
                await original_on_submit(i)
                if not i.response.is_done(): await i.response.defer()
                await self._update_view(self.original_interaction)
            modal.on_submit = on_submit_refresh
            await interaction.response.send_modal(modal)
        else:
            await interaction.response.send_message("Could not find the selected item to edit.", ephemeral=True)

    async def delete_callback(self, interaction: discord.Interaction):
        if not self.current_item_id: return
        
        async def confirm_delete(i: discord.Interaction):
            user_id_str = str(self.user_id)
            deleted = False
            item_id_to_delete = self.current_item_id
            if self.mode == 'training':
                training_shard = self.cog.memory_manager._load_training_shard(user_id_str, self.profile_name) or []
                new_list = [item for item in training_shard if item.get("id") != item_id_to_delete]
                if len(new_list) < len(training_shard):
                    self.cog.memory_manager._save_training_shard(user_id_str, self.profile_name, new_list)
                    deleted = True
            else: # ltm
                context_type: Literal["guild", "dm"] = "guild" if self.guild_id else "dm"
                ltm_shard = self.cog.memory_manager._load_ltm_shard(user_id_str, self.profile_name)
                if ltm_shard:
                    data_list = ltm_shard.get(context_type, [])
                    new_list = [item for item in data_list if item.get("id") != item_id_to_delete]
                    if len(new_list) < len(data_list):
                        ltm_shard[context_type] = new_list
                        self.cog.memory_manager._save_ltm_shard(user_id_str, self.profile_name, ltm_shard)
                        deleted = True

            if deleted:
                await i.response.edit_message(content=f"Item `{item_id_to_delete}` deleted.", view=None, embed=None)
                self.current_page = max(1, self.current_page - 1)
                await self._update_view(self.original_interaction)
            else:
                await i.response.edit_message(content="Could not find item to delete.", view=None, embed=None)

        confirm_view = build_confirm_view("Confirm Deletion", confirm_delete)
        await interaction.response.send_message(f"**Are you sure you want to delete item `{self.current_item_id}`?**", view=confirm_view, ephemeral=True)

    async def search_callback(self, interaction: discord.Interaction):
        modal = SearchDataModal(self)
        await interaction.response.send_modal(modal)

class AnalyseExamplesModal(ui.Modal, title="Analyse Training Examples"):
    def __init__(self, parent_view: 'DataManageView'):
        super().__init__()
        self.parent_view = parent_view
        self.count_input = ui.TextInput(label="Number of Examples to Process", placeholder="Default: 10", default="10", required=True, min_length=1, max_length=3)
        self.verbosity_input = ui.TextInput(label="Target Verbosity (50 - 3000 chars)", placeholder="Default: 800", default="800", required=True, min_length=2, max_length=4)
        self.model_input = ui.TextInput(label="Analysis Model", placeholder="Default: GOOGLE/gemini-2.5-flash-lite", default="GOOGLE/gemini-2.5-flash-lite", required=True)
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
            if not (model_name.upper().startswith("GOOGLE/") or model_name.upper().startswith("OPENROUTER/")):
                raise ValueError("Model must start with GOOGLE/ or OPENROUTER/.")
        except ValueError as e:
            await interaction.response.send_message(f"❌ **Invalid Input:** {e}", ephemeral=True); return

        await interaction.response.defer(ephemeral=True, thinking=True)
        await self.parent_view.cog.memory_manager._execute_training_analysis(interaction, self.parent_view.profile_name, count, verbosity, model_name)

class PrivacyDashboardView(ui.View):
    def __init__(self, cog: 'MimicCog', user_id: int):
        super().__init__(timeout=300)
        self.cog = cog
        self.user_id = user_id

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
        
        filter_select = ui.Select(
            placeholder="Optional. Select additional memories to export...",
            min_values=1,
            max_values=len(filter_options),
            options=filter_options,
            row=2
        )
        filter_select.callback = self.filter_callback
        self.add_item(filter_select)

        export_master_btn = ui.Button(label="Standard Export", style=discord.ButtonStyle.primary, row=3)
        export_master_btn.callback = self.export_master_callback
        self.add_item(export_master_btn)

        export_selfhost_btn = ui.Button(label="Export for Self-Hosted", style=discord.ButtonStyle.secondary, row=3)
        export_selfhost_btn.callback = self.export_selfhost_callback
        self.add_item(export_selfhost_btn)

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
