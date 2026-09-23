from ..utils.constants import *

import discord
from discord import ui
import asyncio
import datetime
import uuid
import time
from typing import TYPE_CHECKING, List, Optional

from ..utils.discord_cdn import unsigned_attachment_url
from .base_components import (PageJumpModal, TabbedView, add_button, add_select,
                              build_pagination_controls, bulk_select_options,
                              paged_nav_options, resolve_bulk_select)
from .gui_start import NewProfileModal

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog


class RedeemCodeModal(ui.Modal, title="Redeem a Share Code"):
    share_code_input = ui.TextInput(label="Enter the share code", required=True, min_length=12, max_length=16)
    name_input = ui.TextInput(label="Desired Profile/Internal Name (Optional)", required=False, placeholder="Leave blank to auto-generate.", max_length=30)

    def __init__(self, cog: 'MimicCog'):
        super().__init__()
        self.cog = cog

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        code = self.share_code_input.value.strip()
        desired_name_raw = self.name_input.value.lower().strip() if self.name_input.value else ""
        
        share_data = self.cog.share_codes.get(code)
        if not share_data or time.time() > share_data["expires_at"]:
            await interaction.followup.send("This share code is invalid or has expired.", ephemeral=True)
            return
        
        owner_id_str = share_data["owner_id"]
        pids_to_borrow = share_data.get("pids", [])
        names_to_borrow = share_data.get("profile_names", share_data.get("profile_name", []))
        if not isinstance(names_to_borrow, list):
            names_to_borrow = [names_to_borrow]

        if owner_id_str == str(interaction.user.id):
            await interaction.followup.send("You cannot borrow a profile from yourself.", ephemeral=True)
            return

        owner = await self.cog.bot.fetch_user(int(owner_id_str))
        sharer_name = owner.name if owner else "Unknown"

        index = self.cog.profile_manager._get_user_index(interaction.user.id)
        
        if desired_name_raw:
            is_valid, err_msg = self.cog.profile_manager._is_valid_profile_name(desired_name_raw)
            if not is_valid:
                await interaction.followup.send(f"❌ **Invalid Name:** {err_msg}", ephemeral=True)
                return
            if desired_name_raw in index.get("personal", []) or desired_name_raw in index.get("borrowed", []):
                await interaction.followup.send("A profile with that name already exists.", ephemeral=True)
                return

        borrowed_field = index.get("borrowed", {})
        current_borrowed = len(borrowed_field) if isinstance(borrowed_field, dict) else len(borrowed_field)
        
        limit = defaultConfig.LIMIT_BORROWED

        if current_borrowed + len(names_to_borrow) > limit:
            await interaction.followup.send(
                f"**Limit Reached.**\n"
                f"Redeeming this code would put you at {current_borrowed + len(names_to_borrow)}/{limit} borrowed profiles.\n"
                f"Please delete some profiles first.",
                ephemeral=True
            )
            return

        accepted_profiles = []
        failed_profiles = {}

        for idx, fallback_name in enumerate(names_to_borrow):
            target_pid = pids_to_borrow[idx] if idx < len(pids_to_borrow) else None
            current_name = self.cog.profile_manager._get_name_from_pid(int(owner_id_str), target_pid) if target_pid else fallback_name
            if not current_name: current_name = fallback_name

            owner_index = self.cog.profile_manager._get_user_index(int(owner_id_str))
            owner_profile_data = self.cog.profile_manager._get_profile_config(int(owner_id_str), current_name, False)
            
            if not owner_profile_data or current_name not in owner_index.get("personal", []):
                failed_profiles[fallback_name] = "Original profile deleted by owner."
                continue

            if desired_name_raw and len(names_to_borrow) == 1:
                desired_name = desired_name_raw
            else:
                desired_name = self.cog.profile_manager._generate_unique_local_name(interaction.user.id, current_name, sharer_name)
            
            if await self.cog.profile_manager._accept_share_request(interaction, int(owner_id_str), target_pid, current_name, desired_name, is_public_borrow=False):
                accepted_profiles.append(f"`{fallback_name}` (as `{desired_name}`)")

        if accepted_profiles:
            del self.cog.share_codes[code]
        
        message = ""
        if accepted_profiles:
            message += f"✅ **Successfully redeemed code!**\nBorrowed: {', '.join(accepted_profiles)}"
        if failed_profiles:
            message += "\n\n⚠️ **Issues:**\n" + "\n".join([f"`{p}`: {r}" for p, r in failed_profiles.items()])
        
        # Empty when every refusal was already explained by _accept_share_request.
        if message:
            await interaction.followup.send(message.strip(), ephemeral=True)

class HubBaseView(TabbedView):
    TABS = (
        ("Home", "home", lambda v: HubHomeView(v.cog, v.original_interaction)),
        ("Public Library", "library", lambda v: HubPublicLibraryView(v.cog, v.original_interaction)),
        ("Incoming Shares", "incoming", lambda v: HubIncomingView(v.cog, v.original_interaction)),
        ("Manage My Shares", "manage", lambda v: HubShareManagerView(v.cog, v.original_interaction)),
        ("Profile Cloning", "cloning", lambda v: HubCloningView(v.cog, v.original_interaction)),
    )

class HubHomeView(HubBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "home")

    async def update_display(self):
        total_public = len(self.cog.public_profiles)
        
        unique_creators = {d["owner_id"] for d in self.cog.profile_manager._iter_public_entries()}
        unique_creators_count = len(unique_creators)
        
        index = self.cog.profile_manager._get_user_index(self.user_id)
        
        user_owned_dict = index.get("personal", {})
        user_owned = len(user_owned_dict) if isinstance(user_owned_dict, dict) else len(user_owned_dict)
        
        user_borrowed_dict = index.get("borrowed", {})
        user_borrowed = len(user_borrowed_dict) if isinstance(user_borrowed_dict, dict) else len(user_borrowed_dict)
        
        embed = discord.Embed(title="MimicAI Profile Hub", description=defaultConfig.MIMIC_NEWS, color=discord.Color.gold())
        embed.set_thumbnail(url=THINKING_THUMBNAIL_URL)
        
        embed.add_field(name="Global Stats", value=f"`{total_public} Public Profiles`\n`{unique_creators_count} Creators`", inline=True)
        embed.add_field(name="Your Stats", value=f"`{user_owned} Personal Profiles`\n`{user_borrowed} Borrowed Profiles`", inline=True)
        
        embed.set_footer(text="Use the navigation buttons below to explore.")

        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def redeem_callback(self, i: discord.Interaction):
        modal = RedeemCodeModal(self.cog)
        await i.response.send_modal(modal)

class HubPublicLibraryView(HubBaseView):
    #: Profiles per dropdown page: 25 options, less the three page controls above them.
    PER_PAGE = 22

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, filtered_list=None):
        super().__init__(cog, interaction, "library")
        self.all_public = []
        self._load_public_data()
        self.filtered_list = filtered_list if filtered_list is not None else self.all_public
        
        # Two positions, because they move independently: the profile the embed shows,
        # and the dropdown page being looked through. They used to be one number, which
        # is why the dropdown was a window sliding round the shown profile and its
        # "page" counter counted profiles.
        self.selected_index = 0
        self.current_page = 0
        
        self.setup_items()

    def _load_public_data(self):
        raw_list = list(self.cog.profile_manager._iter_public_entries())
        raw_list.sort(key=lambda x: x['published_at'], reverse=True)
        self.all_public = raw_list

    def _num_pages(self) -> int:
        return max(1, (len(self.filtered_list) - 1) // self.PER_PAGE + 1)

    def _has_borrowed(self, p_info) -> bool:
        """Whether the viewer already holds a borrow of this listing: one borrow-index lookup.

        This runs on every repaint. It used to read the config of every profile the
        viewer had borrowed, and matched on `original_profile_name` -- a snapshot, so a
        listing renamed after the borrow offered Borrow again.
        """
        return any(borrower_id == self.user_id for borrower_id, _ in
                   self.cog.profile_manager._borrowers_of(p_info['owner_id'], p_info.get('original_pid')))

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        total = len(self.filtered_list)
        self.selected_index = max(0, min(self.selected_index, total - 1))
        num_pages = self._num_pages()
        self.current_page = max(0, min(self.current_page, num_pages - 1))

        if total:
            start = self.current_page * self.PER_PAGE
            options = paged_nav_options(self.current_page, num_pages, nav_suffix=" of profiles")
            for abs_index, p in enumerate(self.filtered_list[start:start + self.PER_PAGE], start):
                owner = self.cog.bot.get_user(p['owner_id'])
                options.append(discord.SelectOption(
                    label=p['profile_name'][:100], value=str(abs_index),
                    description=f"by {owner.name}"[:100] if owner else None,
                    default=(abs_index == self.selected_index)))
            add_select(self, options, self.select_callback,
                       placeholder="Select a profile to view...", row=0)

            add_button(self, "◀", self.prev_profile_cb, row=1,
                       disabled=(self.selected_index == 0))
            add_button(self, "▶", self.next_profile_cb, row=1,
                       disabled=(self.selected_index >= total - 1))

            p_info = self.filtered_list[self.selected_index]
            if self.user_id == p_info['owner_id']:
                add_button(self, "Edit Intro", self.edit_intro_cb, row=1,
                           style=discord.ButtonStyle.blurple)
            elif self._has_borrowed(p_info):
                add_button(self, "Borrowed", self.borrow_cb, row=1,
                           style=discord.ButtonStyle.grey, disabled=True)
            else:
                add_button(self, "Borrow", self.borrow_cb, row=1, style=discord.ButtonStyle.green)

        add_button(self, "Search", self.search_cb, row=1)
        add_button(self, "Generate", self.generate_cb, row=1,
                   style=discord.ButtonStyle.blurple, emoji="✨")

    async def update_display(self):
        if not self.filtered_list:
            embed = discord.Embed(title="Public Library", description="No profiles found.", color=discord.Color.red())
            await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)
            return

        if self.selected_index >= len(self.filtered_list):
            self.selected_index = 0
        
        p_info = self.filtered_list[self.selected_index]
        owner_id = p_info['owner_id']
        original_pid = p_info.get('original_pid')
        
        # Load the heavy config data dynamically ONLY for the profile actively being viewed
        cfg_data = {}
        if original_pid:
            # profile.json.gz, and the config nested inside it. This read used to name
            # a `config.json.gz` that has never been written -- the profile file has
            # been unified since before the hub existed -- so cfg_data was always {},
            # and every library entry rendered with no avatar and the raw profile name
            # instead of the creator's chosen display name.
            profile_data = self.cog.profile_manager._get_profile_by_pid(owner_id, original_pid) or {}
            cfg_data = profile_data.get("config") or {}
        
        owner = self.cog.bot.get_user(owner_id) or await self.cog.bot.fetch_user(owner_id)
        owner_name = owner.name if owner else "Unknown"
        
        disp_name = cfg_data.get("custom_display_name", p_info['profile_name'])
        avatar_url = unsigned_attachment_url(cfg_data.get("custom_avatar_url"))

        byline = f"Created by **{owner_name}**"
        borrowers = len({borrower_id for borrower_id, _ in
                         self.cog.profile_manager._borrowers_of(owner_id, original_pid)})
        if borrowers:
            byline += f" \u00B7 Borrowed by {borrowers} {'user' if borrowers == 1 else 'users'}"
        intro = cfg_data.get("library_intro")
        description = f"{intro}\n\n{byline}" if intro else byline

        embed = discord.Embed(title=disp_name, description=description, color=discord.Color.random())
        if avatar_url: embed.set_image(url=avatar_url)
        embed.set_footer(text=f"ID: {p_info['id']} | {self.selected_index + 1} of {len(self.filtered_list)}")
        
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def _show_profile(self, i: discord.Interaction, index: int):
        """Show one listing and turn the dropdown to the page it is on."""
        self.selected_index = index
        self.current_page = index // self.PER_PAGE
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def select_callback(self, i: discord.Interaction):
        value = i.data['values'][0]
        if value == "jump_page":
            async def _jump(inner: discord.Interaction, page: int):
                self.current_page = page
                self.setup_items()
                await inner.response.edit_message(view=self)

            await i.response.send_modal(PageJumpModal(self._num_pages(), _jump, zero_indexed=True))
            return
        if value in ("prev_page", "next_page"):
            # Only the dropdown moves; the profile on show stays put.
            self.current_page += -1 if value == "prev_page" else 1
            self.setup_items()
            await i.response.edit_message(view=self)
            return
        await self._show_profile(i, int(value))

    async def prev_profile_cb(self, i: discord.Interaction):
        await self._show_profile(i, max(0, self.selected_index - 1))

    async def next_profile_cb(self, i: discord.Interaction):
        await self._show_profile(i, min(len(self.filtered_list) - 1, self.selected_index + 1))

    async def borrow_cb(self, i: discord.Interaction):
        if self.selected_index >= len(self.filtered_list): return
        
        p_info = self.filtered_list[self.selected_index]
        if i.user.id == p_info['owner_id']:
            await i.response.send_message("You cannot borrow your own profile.", ephemeral=True)
            return
        
        if self._has_borrowed(p_info):
            await i.response.send_message("You already have this profile.", ephemeral=True)
            return

        modal = BorrowNameModal(self.cog, self.original_interaction, p_info['owner_id'], p_info.get('original_pid'), p_info['profile_name'], is_public_borrow=True)
        await i.response.send_modal(modal)

    async def edit_intro_cb(self, i: discord.Interaction):
        # Deferred: gui_profiles imports this module at load time.
        from .gui_profiles import LibraryIntroModal

        p_info = self.filtered_list[self.selected_index]
        if i.user.id != p_info['owner_id']:
            return
        # Resolved from the PID: an entry's stored name is a snapshot of when it was
        # published, and the profile may have been renamed since.
        name = self.cog.profile_manager._get_name_from_pid(p_info['owner_id'], p_info.get('original_pid'))
        if not name:
            await i.response.send_message("That profile no longer exists.", ephemeral=True)
            return

        async def repaint(_interaction: discord.Interaction):
            await self.update_display()

        await i.response.send_modal(LibraryIntroModal(self.cog, i.user.id, name, on_saved=repaint))

    async def generate_cb(self, i: discord.Interaction):
        await i.response.send_modal(NewProfileModal(self.cog, generate=True))

    async def search_cb(self, i: discord.Interaction):
        modal = ui.Modal(title="Search Public Library")
        inp = ui.TextInput(label="Search Term", required=False)
        modal.add_item(inp)
        async def on_submit(mi: discord.Interaction):
            term = inp.value.lower()
            if term:
                self.filtered_list = [p for p in self.all_public if term in p['profile_name'].lower()]
            else:
                self.filtered_list = self.all_public
            self.selected_index = self.current_page = 0
            self.setup_items()
            await mi.response.defer()
            await self.update_display()
        modal.on_submit = on_submit
        await i.response.send_modal(modal)

class HubIncomingView(HubBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "incoming")
        self.selected_sharer_id = None
        self.current_page = 0
        self.setup_items()

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        shares = self.cog.profile_shares.get(str(self.user_id), [])
        
        # Group
        sharers = {}
        for s in shares:
            sharers.setdefault(s['sharer_id'], []).append(s)
        
        sharer_ids = list(sharers.keys())
        num_pages = (len(sharer_ids) - 1) // DROPDOWN_MAX_OPTIONS + 1
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_ids = sharer_ids[start : start + DROPDOWN_MAX_OPTIONS]

        # Row 0: Dropdown
        if page_ids:
            options = []
            for sid in page_ids:
                u = self.cog.bot.get_user(sid)
                name = u.name if u else f"ID: {sid}"
                count = len(sharers[sid])
                options.append(discord.SelectOption(label=f"{name} ({count} profiles)", value=str(sid), default=(sid == self.selected_sharer_id)))

            add_select(self, options, self.select_sharer,
                       placeholder="Select a user to review shares...", min_values=1,
                       max_values=1, row=0)

        # Row 1: Pagination (if needed)
        build_pagination_controls(self, self.current_page, num_pages, 1, self.prev_page, self.next_page)

        # Row 2: Actions
        action_row = 2 if num_pages > 1 else 1 # Move up if no pagination
        
        if self.selected_sharer_id:
            acc_btn = ui.Button(label="Accept All", style=discord.ButtonStyle.green, row=action_row)
            rej_btn = ui.Button(label="Reject All", style=discord.ButtonStyle.danger, row=action_row)
            back_btn = ui.Button(label="Cancel Selection", style=discord.ButtonStyle.grey, row=action_row)
            
            acc_btn.callback = self.accept_all
            rej_btn.callback = self.reject_all
            back_btn.callback = self.clear_selection
            
            self.add_item(acc_btn)
            self.add_item(rej_btn)
            self.add_item(back_btn)
        else:
            add_button(self, "Redeem Share Code", self.redeem_code_callback,
                       style=discord.ButtonStyle.secondary, row=action_row, emoji="🔑")

    async def update_display(self):
        shares = self.cog.profile_shares.get(str(self.user_id), [])
        
        if self.selected_sharer_id:
            u = self.cog.bot.get_user(self.selected_sharer_id)
            name = u.name if u else "Unknown"
            user_shares = [s['profile_name'] for s in shares if s['sharer_id'] == self.selected_sharer_id]
            desc = f"**Pending shares from {name}:**\n" + ", ".join([f"`{n}`" for n in user_shares])
            embed = discord.Embed(title="Reviewing Shares", description=desc, color=discord.Color.blue())
        elif not shares:
            embed = discord.Embed(title="Incoming Shares", description="You have no direct share requests pending.\n\nHave a code? Click the button below.", color=discord.Color.dark_grey())
        else:
            embed = discord.Embed(title="Incoming Shares", description="Select a user from the dropdown to accept or reject their shared profiles.", color=discord.Color.blue())
            
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def select_sharer(self, i: discord.Interaction):
        self.selected_sharer_id = int(i.data['values'][0])
        self.setup_items()
        await i.response.defer()
        await self.update_display()


    async def clear_selection(self, i: discord.Interaction):
        self.selected_sharer_id = None
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def redeem_code_callback(self, i: discord.Interaction):
        modal = RedeemCodeModal(self.cog)
        await i.response.send_modal(modal)

    async def accept_all(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True)
        sharer_id = self.selected_sharer_id
        shares = [s for s in self.cog.profile_shares.get(str(self.user_id), []) if s['sharer_id'] == sharer_id]
        
        limit = defaultConfig.LIMIT_BORROWED
        index = self.cog.profile_manager._get_user_index(self.user_id)
        
        borrowed_field = index.get("borrowed", {})
        current_borrowed = len(borrowed_field) if isinstance(borrowed_field, dict) else len(borrowed_field)

        if current_borrowed + len(shares) > limit:
            await i.followup.send(f"Limit Reached. Accepting these would exceed your limit of {limit} borrowed profiles.", ephemeral=True)
            return

        accepted = []
        sharer_user = self.cog.bot.get_user(sharer_id)
        sharer_name = sharer_user.name if sharer_user else "User"

        for s in shares:
            fallback_name = s['profile_name']
            target_pid = s.get('original_pid')
            current_name = self.cog.profile_manager._get_name_from_pid(sharer_id, target_pid) if target_pid else fallback_name
            if not current_name: current_name = fallback_name

            sharer_index = self.cog.profile_manager._get_user_index(sharer_id)
            if current_name not in sharer_index.get("personal", []):
                await self.cog.profile_manager._reject_share_request(self.original_interaction, sharer_id, target_pid, fallback_name, notify_sharer=False)
                continue

            local_name = self.cog.profile_manager._generate_unique_local_name(self.user_id, current_name, sharer_name)
            if await self.cog.profile_manager._accept_share_request(self.original_interaction, sharer_id, target_pid, current_name, local_name, is_public_borrow=False):
                accepted.append(current_name)
        
        msg = f"Accepted: {', '.join(accepted)}" if accepted else "No valid profiles found."
        await i.followup.send(msg, ephemeral=True)
        self.selected_sharer_id = None
        self.setup_items()
        await self.update_display()

    async def reject_all(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True)
        sharer_id = self.selected_sharer_id
        shares =[s for s in self.cog.profile_shares.get(str(self.user_id), []) if s['sharer_id'] == sharer_id]
        for s in shares:
            await self.cog.profile_manager._reject_share_request(self.original_interaction, sharer_id, s.get('original_pid'), s['profile_name'], notify_sharer=False)
        await i.followup.send("Rejected shares.", ephemeral=True)
        self.selected_sharer_id = None
        self.setup_items()
        await self.update_display()

class HubShareManagerView(HubBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "manage")
        self.mode = "private"
        self.selected_profiles = []
        self.selected_users = []
        self.processing = False
        
        #: What the dropdown offers in the current mode, filled by `load_profiles`: only
        #: profiles whose rating lets them be shared (or published), because listing the
        #: rest offered a choice every button then refused.
        self.personal_profiles: List[str] = []
        self._loaded_mode: Optional[str] = None
        self.current_page = 0
        
        self.setup_items()

    def _eligible_profiles(self) -> List[str]:
        """Blocking: one rating read per personal profile."""
        pm = self.cog.profile_manager
        capability = "share" if self.mode == "private" else "publish"
        names = sorted(pm._get_user_index(self.user_id).get("personal", []))
        eligible = {name for name in names if pm.content_capability(self.user_id, name, capability)[0]}
        if self.mode == "public":
            # Already listed stays offered whatever its rating, or it could not be
            # deselected to unpublish it.
            eligible |= {name for name, _ in self._get_user_public_profiles()} & set(names)
        return sorted(eligible)

    async def load_profiles(self):
        self.personal_profiles = await asyncio.to_thread(self._eligible_profiles)
        self._loaded_mode = self.mode
        self.setup_items()

    def _get_user_public_profiles(self):
        """Scans public_profiles for entries owned by this user, returning (name, is_locked) pairs."""
        return [(d["profile_name"], d["status"] == "locked")
                for d in self.cog.profile_manager._iter_public_entries(self.user_id)]

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        # Row 0: Mode
        style = discord.ButtonStyle.blurple if self.mode == "private" else discord.ButtonStyle.green
        label = "Mode: Private Sharing" if self.mode == "private" else "Mode: Public Publishing"
        add_button(self, label, self.toggle_mode, style=style, row=0)

        # Row 0, not beside the action buttons: in private mode row 2 already carries
        # the three pagination controls plus Send and Get Code, which is Discord's
        # five-per-row limit exactly. Sitting directly above the select reads as
        # belonging to it anyway.
        if self.selected_profiles:
            add_button(self, f"Clear ({len(self.selected_profiles)})", self.clear_selection,
                       style=discord.ButtonStyle.secondary, row=0)

        # Row 1: Paginated Profiles.
        #
        # Page size is SHARE_PAGE_SIZE rather than the full 25, because the two bulk
        # rows occupy option slots: Discord caps a select at 25 options total, so a
        # full page plus the sentinels would be rejected outright.
        num_pages = max(1, (len(self.personal_profiles) - 1) // SHARE_PAGE_SIZE + 1)
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)

        start = self.current_page * SHARE_PAGE_SIZE
        page_profiles = self.personal_profiles[start : start + SHARE_PAGE_SIZE]

        options = []
        if page_profiles:
            options = bulk_select_options(
                page_profiles, self.personal_profiles,
                lambda p: p in self.selected_profiles,
                page_description="Toggle selection for every profile on this page.",
                all_description="Toggle selection for every profile you own.")

            for p in page_profiles:
                options.append(discord.SelectOption(
                    label=p, value=p, default=(p in self.selected_profiles)))

        if options:
            add_select(self, options, self.select_profiles, placeholder="Select profiles...",
                       min_values=0, max_values=len(options), row=1)

        # Row 2: Pagination Buttons (if needed) AND Action Buttons
        build_pagination_controls(self, self.current_page, num_pages, 2, self.prev_page, self.next_page)

        if self.mode == "private":
            send_btn = ui.Button(label="Send", style=discord.ButtonStyle.green, row=2)
            code_btn = ui.Button(label="Get Code", style=discord.ButtonStyle.secondary, row=2)
            send_btn.callback = self.send_private
            code_btn.callback = self.generate_code
            self.add_item(send_btn)
            self.add_item(code_btn)
        else:
            add_button(self, "Apply Changes", self.apply_public, style=discord.ButtonStyle.green,
                       row=2)
            # One intro at a time, so only with one profile selected. Row 2 holds at most
            # the three page controls and Apply Changes beside it.
            add_button(self, "Edit Intro", self.edit_intro, row=2,
                       disabled=len(self.selected_profiles) != 1)

        # Row 3: User Select (Private)
        if self.mode == "private":
            user_sel = ui.UserSelect(placeholder="Select recipients...", min_values=1, max_values=10, row=3)
            user_sel.callback = self.select_users
            self.add_item(user_sel)

    async def update_display(self):
        if self._loaded_mode != self.mode:
            await self.load_profiles()
        await self.original_interaction.edit_original_response(content=None, embed=self.build_embed(), view=self)

    def build_embed(self) -> discord.Embed:
        desc = "Manage how you share your profiles.\n\n"
        if self.mode == "private":
            desc += ("**Private Mode:** Share specifically with friends via DM or Code.\n"
                     "Only profiles rated **General** or **Exempt** are listed.")
        else:
            desc += ("**Public Mode:** Publish your profiles to the global library for anyone to borrow.\n"
                     "Only profiles rated **General** or **Exempt** are listed, with any already published. "
                     "Select a single profile to give it an intro for its listing.")
        if not self.personal_profiles:
            desc += (f"\n\nNone of your profiles can be {'shared' if self.mode == 'private' else 'published'} "
                     f"yet. Rate one from `/profile manage` → Home → **Content Safety**.")
            
        embed = discord.Embed(title="Share Manager", description=desc, color=discord.Color.teal())
        
        full_text = ", ".join(self.selected_profiles)
        # A field value, so 1024. This cut at 4000, the description's limit, and Select All
        # on a large account made Discord refuse the whole embed.
        if len(full_text) > 1024: full_text = full_text[:1021] + "..."
        if not full_text: full_text = "None"
        embed.add_field(name="Selected Profiles", value=full_text, inline=False)

        if self.mode == "public":
            public_names = [f"{name}{' (Locked)' if is_locked else ''}" for name, is_locked in self._get_user_public_profiles()]

            val = ", ".join(public_names) if public_names else "None"
            if len(val) > 1024: val = val[:1021] + "..."
            embed.add_field(name="Your Currently Public Profiles", value=val, inline=False)

        return embed

    async def toggle_mode(self, i: discord.Interaction):
        # [UPDATED] Free users CAN toggle to Public to UNPUBLISH. Validation happens on Apply.
        if self.mode == "private":
            self.mode = "public"
            self.selected_profiles = [name for name, _ in self._get_user_public_profiles()]
        else:
            self.mode = "private"
            self.selected_profiles = []
        
        self.current_page = 0
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def edit_intro(self, i: discord.Interaction):
        # Deferred: gui_profiles imports this module at load time.
        from .gui_profiles import LibraryIntroModal

        if len(self.selected_profiles) != 1:
            return
        await i.response.send_modal(LibraryIntroModal(self.cog, self.user_id, self.selected_profiles[0]))

    async def select_profiles(self, i: discord.Interaction):
        start = self.current_page * SHARE_PAGE_SIZE
        self.selected_profiles = sorted(resolve_bulk_select(
            i.data.get('values', []),
            self.personal_profiles[start : start + SHARE_PAGE_SIZE],
            self.personal_profiles, self.selected_profiles))
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def clear_selection(self, i: discord.Interaction):
        self.selected_profiles = []
        self.setup_items()
        await i.response.defer()
        await self.update_display()


    async def select_users(self, i: discord.Interaction):
        self.selected_users = i.data['values'] 
        await i.response.defer()

    async def send_private(self, i: discord.Interaction):
        if self.processing: return
        self.processing = True

        for item in self.children:
            if isinstance(item, ui.Button) and item.label in ["Send", "Send Request", "Get Code"]: item.disabled = True
        await i.response.edit_message(view=self)

        if not self.selected_profiles or not self.selected_users:
            await i.followup.send("Select profiles and recipients first.", ephemeral=True)
            self.processing = False
            self.setup_items()
            await self.update_display()
            return
        
        shareable, refused = self._partition_shareable(self.selected_profiles)
        if not shareable:
            self.processing = False
            self.setup_items()
            await self.update_display()
            lines = "\n".join(f"• **{n}** -- {r}" for n, r in refused.items())
            await i.followup.send(
                f"None of the selected profiles can be shared yet.\n{lines}", ephemeral=True)
            return

        success_count = 0
        for recipient_id_str in self.selected_users:
            recipient_id = int(recipient_id_str)
            recipient = self.cog.bot.get_user(recipient_id)
            if not recipient:
                try:
                    recipient = await self.cog.bot.fetch_user(recipient_id)
                except Exception:
                    pass
            if not recipient or recipient.bot or recipient.id == self.user_id: continue

            self.cog.profile_shares.setdefault(str(recipient_id), [])
            newly_shared = []
            for profile_name in shareable:
                pid = self.cog.profile_manager._get_pid_from_name_any(self.user_id, profile_name)
                existing = next((s for s in self.cog.profile_shares[str(recipient_id)] if s['sharer_id'] == self.user_id and (s.get('original_pid') == pid or s.get('profile_name') == profile_name)), None)
                if not existing:
                    share_req = {"sharer_id": self.user_id, "original_pid": pid, "profile_name": profile_name, "shared_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                    self.cog.profile_shares[str(recipient_id)].append(share_req)
                    newly_shared.append(profile_name)
            
            if newly_shared:
                try:
                    await recipient.send(f"**{self.original_interaction.user.name}** shared profile(s) with you: {', '.join(newly_shared)}. Check `/profile hub`.")
                    success_count += 1
                except discord.Forbidden: pass
            
            self.cog.profile_manager._save_profile_share_shard(str(recipient_id), self.cog.profile_shares[str(recipient_id)])

        self.processing = False
        self.setup_items()
        await self.update_display()
        report = f"Sent to {success_count} users."
        if refused:
            lines = "\n".join(f"• **{n}** -- {r}" for n, r in refused.items())
            report += f"\n\n**Left out:**\n{lines}"
        await i.followup.send(report, ephemeral=True)

    def _partition_shareable(self, names):
        """Splits a selection into (allowed, {name: reason}) for the share gates.

        Sharing is the point at which a profile stops being only its owner's
        business, so it is gated on the rating exactly as publishing is: Adult may be
        neither. Both gates read content_capability so the refusal wording is the
        same sentence the Content Safety dashboard shows. The dropdown already hides
        what fails here; this catches a rating that changed after it was drawn.
        """
        allowed, refused = [], {}
        for name in names:
            ok, reason = self.cog.profile_manager.content_capability(self.user_id, name, "share")
            if ok:
                allowed.append(name)
            else:
                refused[name] = reason
        return allowed, refused

    async def generate_code(self, i: discord.Interaction):
        if not self.selected_profiles:
            await i.response.send_message("Select at least one profile.", ephemeral=True); return

        shareable, refused = self._partition_shareable(self.selected_profiles)
        if not shareable:
            lines = "\n".join(f"• **{n}** -- {r}" for n, r in refused.items())
            await i.response.send_message(
                f"None of the selected profiles can be shared yet.\n{lines}", ephemeral=True)
            return

        code = f"SHR-{uuid.uuid4().hex[:8].upper()}"
        pids =[self.cog.profile_manager._get_pid_from_name_any(self.user_id, p) for p in shareable]
        self.cog.profile_manager.register_share_code(
            code, {"owner_id": str(self.user_id), "pids": pids,
                   "profile_names": shareable, "expires_at": time.time() + 300})

        msg = f"Share Code: `{code}`\nExpires in 5 minutes.\nIncludes: {', '.join(shareable)}"
        if refused:
            lines = "\n".join(f"• **{n}** -- {r}" for n, r in refused.items())
            msg += f"\n\n**Left out:**\n{lines}"
        await i.response.send_message(msg, ephemeral=True)

    async def apply_public(self, i: discord.Interaction):
        if self.processing: return
        self.processing = True

        for item in self.children:
            if isinstance(item, ui.Button) and item.label in ["Apply Changes", "Apply Publishing Changes"]: item.disabled = True
        await i.response.edit_message(view=self)

        user_id_str = str(self.user_id)
        
        # Names of everything this user already has published. Read through the
        # normaliser: the old inline version resolved string entries via a
        # name.txt that is never written, so this set came back empty and every
        # already-public profile landed in `to_publish` -- re-running the paid
        # moderator on it and, if it had since been set to 18+, reporting a
        # failure for a profile that was published all along.
        current_public_set = {d["profile_name"]
                              for d in self.cog.profile_manager._iter_public_entries(self.user_id)}
        
        target_set = set(self.selected_profiles)
        to_publish = target_set - current_public_set
        to_unpublish = current_public_set - target_set
        
        published_list = []
        failed_list = {}

        if to_publish:
            # Publishing no longer analyses anything. It reads the rating the owner
            # already obtained, because that rating *is* the safety analysis -- the
            # separate auto-moderator pass that used to run here answered an adjacent
            # question with a second prompt, and the two could disagree on the same
            # profile with no way for the owner to tell which had objected.
            #
            # The consequence is that publishing is now instant and free: no API call,
            # no avatar download, no "Analysing profiles for safety..." wait. A
            # profile that has not been rated is refused with the same sentence the
            # Content Safety dashboard shows, and rating it is a click away there.
            def evaluate_profile(name: str):
                allowed, reason = self.cog.profile_manager.content_capability(
                    self.user_id, name, "publish")
                return name, allowed, reason

            results = [evaluate_profile(name) for name in to_publish]

            for name, is_safe, reason in results:
                if is_safe:
                    pid_entry = self.cog.profile_manager._get_pid_from_name_any(self.user_id, name)
                    self.cog.public_profiles[pid_entry] = f"{self.user_id}:{pid_entry}"
                    published_list.append(name)
                else:
                    failed_list[name] = reason
            
            if published_list:
                self.cog.profile_manager._save_public_index()

        unpublished_list = []
        for name in to_unpublish:
            target_pid = self.cog.profile_manager._get_pid_from_name_any(self.user_id, name)
            ids_to_del = []
            for pid, info in self.cog.public_profiles.items():
                if isinstance(info, str) and ":" in info:
                    if info == f"{self.user_id}:{target_pid}":
                        ids_to_del.append(pid)
                # By PID, as the string form above: the entry's name is a snapshot from
                # publishing, so a profile renamed since could not be unpublished, and
                # a later profile given the old name unpublished the renamed one.
                elif (isinstance(info, dict) and str(info.get("owner_id")) == user_id_str
                      and (info.get("original_pid") or info.get("original_profile_id")) == target_pid):
                    ids_to_del.append(pid)
                    
            for pid in ids_to_del:
                del self.cog.public_profiles[pid]
                unpublished_list.append(name)
        
        if unpublished_list:
            self.cog.profile_manager._save_public_index()

        self.processing = False
        self.setup_items()
        await self.update_display()

        report_embed = discord.Embed(title="Publishing Report", color=discord.Color.blue())
        if published_list: report_embed.add_field(name=f"✅ Published ({len(published_list)})", value=", ".join(published_list), inline=False)
        if unpublished_list: report_embed.add_field(name=f"⛔ Unpublished ({len(unpublished_list)})", value=", ".join(unpublished_list), inline=False)
        if failed_list:
            errs = "\n".join([f"• **{n}**: {r}" for n, r in failed_list.items()])
            if len(errs) > 1000: errs = errs[:997] + "..."
            report_embed.add_field(name=f"⚠️ Failed ({len(failed_list)})", value=errs, inline=False)
            report_embed.color = discord.Color.orange()
        if not (published_list or unpublished_list or failed_list): report_embed.description = "No changes were made."

        await i.followup.send(embed=report_embed, ephemeral=True)

class HubCloningView(HubBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "cloning")
        self.selected_profile = None
        self.current_page = 0
        
        index = self.cog.profile_manager._get_user_index(self.user_id)
        self.personal_profiles = sorted(list(index.get("personal", [])))
        self.setup_items()

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        num_pages = (len(self.personal_profiles) - 1) // DROPDOWN_MAX_OPTIONS + 1
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        
        start = self.current_page * DROPDOWN_MAX_OPTIONS
        page_profiles = self.personal_profiles[start : start + DROPDOWN_MAX_OPTIONS]

        options = []
        for p in page_profiles:
            options.append(discord.SelectOption(label=p, value=p, default=(p == self.selected_profile)))
        
        if options:
            add_select(self, options, self.select_profile_cb,
                       placeholder="Select a personal profile to clone...", row=0)

        build_pagination_controls(self, self.current_page, num_pages, 1, self.prev_page, self.next_page)

        action_row = 2 if num_pages > 1 else 1
        
        add_button(self, "Generate Clone Code", self.generate_clone_code_cb,
                   style=discord.ButtonStyle.green, row=action_row,
                   disabled=not self.selected_profile)

        add_button(self, "Redeem Clone Code", self.redeem_clone_code_cb,
                   style=discord.ButtonStyle.blurple, row=action_row)

    async def update_display(self):
        embed = discord.Embed(title="Profile Cloning", description="Clone profiles to create independent copies. Cloned profiles will not copy memories or child bot configurations.", color=discord.Color.dark_purple())
        embed.add_field(name="Selected Profile", value=f"`{self.selected_profile or 'None'}`", inline=False)
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

    async def select_profile_cb(self, i: discord.Interaction):
        self.selected_profile = i.data['values'][0]
        self.setup_items()
        await i.response.defer()
        await self.update_display()


    async def generate_clone_code_cb(self, i: discord.Interaction):
        if not self.selected_profile: return
        allowed, reason = self.cog.profile_manager.content_capability(
            self.user_id, self.selected_profile, "share")
        if not allowed:
            await i.response.send_message(
                f"**'{self.selected_profile}' cannot be cloned.**\n{reason}", ephemeral=True)
            return
        code = f"CLN-{uuid.uuid4().hex[:8].upper()}"
        
        pid = self.cog.profile_manager._get_pid_from_name_any(self.user_id, self.selected_profile)
        
        if not hasattr(self.cog, "clone_codes"):
            self.cog.clone_codes = {}
        
        self.cog.clone_codes[code] = {
            "owner_id": self.user_id,
            "pid": pid,
            "profile_name": self.selected_profile,
            "expires_at": time.time() + 300
        }
        
        await i.response.send_message(f"Clone Code Generated: `{code}`\nProvide this to another user. Valid for 5 minutes.", ephemeral=True)

    async def redeem_clone_code_cb(self, i: discord.Interaction):
        modal = RedeemCloneCodeModal(self.cog, self)
        await i.response.send_modal(modal)

class RedeemCloneCodeModal(ui.Modal, title="Redeem Clone Code"):
    code_input = ui.TextInput(label="Enter Clone Code", required=True, min_length=12, max_length=16)
    name_input = ui.TextInput(label="Local Profile Name", required=True, min_length=1, max_length=30)

    def __init__(self, cog: 'MimicCog', parent_view: HubCloningView):
        super().__init__()
        self.cog = cog
        self.parent_view = parent_view

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        code = self.code_input.value.strip()
        desired_name = self.name_input.value.lower().strip()

        is_valid, err_msg = self.cog.profile_manager._is_valid_profile_name(desired_name)
        if not is_valid:
            await interaction.followup.send(f"❌ **Invalid Name:** {err_msg}", ephemeral=True)
            return

        clone_codes = getattr(self.cog, "clone_codes", {})
        share_data = clone_codes.get(code)
        if not share_data or time.time() > share_data["expires_at"]:
            await interaction.followup.send("This clone code is invalid or has expired.", ephemeral=True)
            return

        owner_id = share_data["owner_id"]
        pid = share_data["pid"]
        
        if owner_id == interaction.user.id:
            await interaction.followup.send("You cannot clone your own profile.", ephemeral=True)
            return

        index = self.cog.profile_manager._get_user_index(interaction.user.id)
        if desired_name in index.get("personal", []) or desired_name in index.get("borrowed", []):
            await interaction.followup.send("A profile with that name already exists.", ephemeral=True)
            return

        limit = defaultConfig.LIMIT_PROFILES
        if len(index.get("personal", [])) >= limit:
            await interaction.followup.send(f"You have reached your personal profile limit of {limit}.", ephemeral=True)
            return

        success, msg = await self.cog.profile_manager._execute_clone_handshake(owner_id, pid, interaction.user.id, desired_name)
        if success:
            clone_codes.pop(code, None)
            self.parent_view.setup_items()
            await self.parent_view.update_display()
        
        await interaction.followup.send(msg, ephemeral=True)

class BorrowNameModal(ui.Modal, title="Name Your Borrowed Profile"):
    profile_name_input = ui.TextInput(label="Enter a unique local name", required=True, min_length=1, max_length=50)
    
    def __init__(self, cog: 'MimicCog', original_interaction: discord.Interaction, sharer_id: int, target_pid: Optional[str], fallback_name: str, is_public_borrow: bool = False):
        super().__init__()
        self.cog = cog
        self.original_interaction = original_interaction
        self.sharer_id = sharer_id
        self.target_pid = target_pid
        self.fallback_name = fallback_name
        self.is_public_borrow = is_public_borrow

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        desired_name = self.profile_name_input.value.lower().strip()
        
        is_valid, err_msg = self.cog.profile_manager._is_valid_profile_name(desired_name)
        if not is_valid:
            await interaction.followup.send(f"❌ **Invalid Name:** {err_msg}", ephemeral=True)
            return

        index = self.cog.profile_manager._get_user_index(interaction.user.id)
        if desired_name in index.get("personal", []) or desired_name in index.get("borrowed",[]):
            await interaction.followup.send(f"You already have a profile named '{desired_name}'. Please choose a different name.", ephemeral=True)
            return

        if await self.cog.profile_manager._accept_share_request(interaction, self.sharer_id, self.target_pid, self.fallback_name, desired_name, self.is_public_borrow):
            await interaction.followup.send(f"✅ Successfully borrowed profile **{self.fallback_name}** and named it **{desired_name}**.", ephemeral=True)
