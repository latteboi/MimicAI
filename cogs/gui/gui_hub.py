from ..utils.constants import *

import discord
from discord import ui
import asyncio
import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from ..utils.discord_cdn import unsigned_attachment_url
from .base_components import (BlockedGuard, PageJumpModal, TabbedView, add_button, add_select,
                              build_pagination_controls, bulk_select_options,
                              paged_nav_options, refuse_unregistered, resolve_bulk_select)
from .gui_start import NewProfileModal

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog


class HubBaseView(TabbedView):
    #: The Library first: browsing is what most people open the hub for, and publishing
    #: lives on it (Publish Yours) rather than behind a mode toggle on the Sharing tab.
    #: Five tabs is Discord's five to a row, so a sixth needs a home inside one of these.
    TABS = (
        ("Public Library", "library", lambda v: HubPublicLibraryView(v.cog, v.original_interaction)),
        ("Incoming Shares", "incoming", lambda v: HubIncomingView(v.cog, v.original_interaction)),
        ("Profile Sharing", "manage", lambda v: HubShareManagerView(v.cog, v.original_interaction)),
        ("Profile Cloning", "clone", lambda v: HubCloneManagerView(v.cog, v.original_interaction)),
        ("Stats", "stats", lambda v: HubStatsView(v.cog, v.original_interaction)),
    )


def _borrower_count(pm, owner_id: int, pid: Optional[str]) -> int:
    """People borrowing one profile, counted once each: one borrow-index lookup."""
    return len({borrower_id for borrower_id, _ in pm._borrowers_of(owner_id, pid)}) if pid else 0


class HubStatsView(HubBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "stats")

    def build_embed(self) -> discord.Embed:
        """Blocking: one owner-index read per creator, as the Library's own open."""
        pm = self.cog.profile_manager
        listings = list(pm._iter_public_entries())
        reach = {d["id"]: _borrower_count(pm, d["owner_id"], d["original_pid"]) for d in listings}
        creators = {d["owner_id"] for d in listings}

        index = pm._get_user_index(self.user_id)
        personal = list(index.get("personal", {}))
        borrowed = len(index.get("borrowed", {}))
        waiting = len(pm._pending_shares(self.user_id))
        mine = [d for d in listings if d["owner_id"] == self.user_id]
        # Every profile of yours, listed or only shared privately: reach is reach.
        yours = {name: {b for b, _ in pm._borrowers_of(self.user_id, pm._get_pid_from_name(self.user_id, name))}
                 for name in personal}
        borrowers = set().union(*yours.values()) if yours else set()

        embed = discord.Embed(title="Profile Hub Stats", description=defaultConfig.MIMIC_NEWS or None,
                              color=discord.Color.gold())
        embed.set_thumbnail(url=THINKING_THUMBNAIL_URL)
        library = (f"`{len(listings)} Public Profiles`\n`{len(creators)} Creators`\n"
                   f"`{sum(reach.values())} Borrows`")
        top = max(listings, key=lambda d: reach[d["id"]], default=None)
        if top and reach[top["id"]]:
            owner = self.cog.bot.get_user(top["owner_id"])
            library += (f"\nMost borrowed: **{top['profile_name']}**"
                        f"{f' by {owner.name}' if owner else ''} ({reach[top['id']]})")
        embed.add_field(name="The Library", value=library, inline=True)
        embed.add_field(name="Yours", value=(
            f"`{len(personal)} Personal Profiles`\n`{borrowed} Borrowed Profiles`\n"
            f"`{waiting} Shares Waiting`\n`{len(mine)} Published`"), inline=True)
        favourite = max(yours, key=lambda n: len(yours[n]), default=None)
        reach_text = (f"`{len(borrowers)} {'person borrows' if len(borrowers) == 1 else 'people borrow'} "
                      f"your profiles`")
        if favourite and yours[favourite]:
            reach_text += f"\nMost borrowed: **{favourite}** ({len(yours[favourite])})"
        embed.add_field(name="Your Reach", value=reach_text, inline=False)
        embed.set_footer(text="Publish from Public Library → Publish Yours, or from /profile manage → Sharing.")
        return embed

    async def update_display(self):
        embed = await asyncio.to_thread(self.build_embed)
        await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)

class HubPublicLibraryView(HubBaseView):
    #: Profiles per dropdown page: 25 options, less the three page controls above them.
    PER_PAGE = 22

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, filtered_list=None):
        super().__init__(cog, interaction, "library")
        #: "borrowed" (most borrowers first) or "new" (newest first).
        self.sort = "borrowed"
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
        # ponytail: one owner-index read per creator, on the loop, every Library open --
        # cheap until creators outgrow the 200-entry user_indices LRU; then cache this list.
        self.all_public = self._sorted(self.cog.profile_manager._iter_public_entries())

    def _sorted(self, entries):
        """Newest first, or most borrowed first with newest breaking ties."""
        newest = sorted(entries, key=lambda x: x['published_at'], reverse=True)
        if self.sort != "borrowed":
            return newest
        pm = self.cog.profile_manager
        return sorted(newest, key=lambda x: _borrower_count(pm, x['owner_id'], x.get('original_pid')),
                      reverse=True)

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
        add_button(self, "Sort: Most Borrowed" if self.sort == "borrowed" else "Sort: Newest",
                   self.sort_cb, row=2)
        add_button(self, "Publish Yours", self.publish_cb, row=2,
                   style=discord.ButtonStyle.green, emoji="🌐")

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
        if await refuse_unregistered(self.cog, i):
            return
        
        p_info = self.filtered_list[self.selected_index]
        if i.user.id == p_info['owner_id']:
            await i.response.send_message("You cannot borrow your own profile.", ephemeral=True)
            return
        
        if self._has_borrowed(p_info):
            await i.response.send_message("You already have this profile.", ephemeral=True)
            return

        await i.response.defer(ephemeral=True)
        local_name = self.cog.profile_manager._generate_unique_local_name(i.user.id, p_info['profile_name'])
        if await self.cog.profile_manager._accept_share_request(i, p_info['owner_id'], p_info.get('original_pid'), p_info['profile_name'], local_name, is_public_borrow=True):
            await i.followup.send(f"✅ Borrowed **{p_info['profile_name']}** as **{local_name}**.", ephemeral=True)

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
        await i.response.send_modal(NewProfileModal(self.cog))

    def _matches(self, entry, term: str) -> bool:
        """The term in the profile's name, its creator's, its display name or its intro.
        Blocking: the last two can read the profile's file."""
        owner = self.cog.bot.get_user(entry['owner_id'])
        return (term in entry['profile_name'].lower()
                or (owner is not None and term in owner.name.lower())
                or term in self.cog.profile_manager.library_search_text(
                    entry['owner_id'], entry.get('original_pid')))

    async def search_cb(self, i: discord.Interaction):
        modal = ui.Modal(title="Search Public Library")
        inp = ui.TextInput(label="Name, creator or words from the intro", required=False)
        modal.add_item(inp)
        async def on_submit(mi: discord.Interaction):
            await mi.response.defer()
            term = inp.value.lower().strip()
            if term:
                self.filtered_list = await asyncio.to_thread(
                    lambda: [p for p in self.all_public if self._matches(p, term)])
            else:
                self.filtered_list = self.all_public
            self.selected_index = self.current_page = 0
            self.setup_items()
            await self.update_display()
        modal.on_submit = on_submit
        await i.response.send_modal(modal)

    async def sort_cb(self, i: discord.Interaction):
        self.sort = "new" if self.sort == "borrowed" else "borrowed"
        searching = self.filtered_list is not self.all_public
        self.all_public = self._sorted(self.all_public)
        self.filtered_list = self._sorted(self.filtered_list) if searching else self.all_public
        self.selected_index = self.current_page = 0
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def publish_cb(self, i: discord.Interaction):
        if await refuse_unregistered(self.cog, i):
            return
        await i.response.defer()
        view = HubPublishView(self.cog, self.original_interaction)
        await view.update_display()

class HubIncomingView(HubBaseView):
    """Shares waiting for you, and who may send them.

    A share is only ever offered: it waits here, and nobody is messaged to say so --
    see `ProfileManager.accepts_share`. Closing turns away new ones and leaves those
    already waiting to be answered; blocking someone takes theirs back.
    """

    #: One user dropdown holds the whole block list: its ticks are the list.
    BLOCK_LIMIT = DROPDOWN_MAX_OPTIONS

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "incoming")
        self.selected_sharer_id = None
        self.current_page = 0
        self.setup_items()

    def _about(self):
        return self.cog.profile_manager.get_user_about(self.user_id)

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        shares = self.cog.profile_manager._pending_shares(self.user_id)
        
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
            add_button(self, "Accept All", self.accept_all, style=discord.ButtonStyle.green,
                       row=action_row)
            add_button(self, "Reject All", self.reject_all, style=discord.ButtonStyle.danger,
                       row=action_row)
            add_button(self, "Block Sender", self.block_sender, style=discord.ButtonStyle.danger,
                       row=action_row, emoji="🚫")
            add_button(self, "Cancel Selection", self.clear_selection, row=action_row)
            return

        if not self.cog.profile_manager.is_registered(self.user_id):
            return
        about = self._about()
        closed = bool(about.get("shares_closed"))
        add_button(self, f"Incoming Shares: {'Closed' if closed else 'Open'}", self.toggle_open,
                   style=discord.ButtonStyle.secondary if closed else discord.ButtonStyle.green,
                   row=action_row)
        blocked = (about.get("blocked_sharers") or [])[:self.BLOCK_LIMIT]
        block_select = ui.UserSelect(
            placeholder="Blocked users -- tick to block, untick to unblock...",
            min_values=0, max_values=self.BLOCK_LIMIT, row=3,
            default_values=[discord.Object(id=uid) for uid in blocked])
        block_select.callback = self.set_blocked
        self.add_item(block_select)

    async def update_display(self):
        shares = self.cog.profile_manager._pending_shares(self.user_id)
        
        if self.selected_sharer_id:
            u = self.cog.bot.get_user(self.selected_sharer_id)
            name = u.name if u else "Unknown"
            user_shares = [f"`{s['profile_name']}`" + (" (clone)" if s.get("kind") == "clone" else "")
                           for s in shares if s['sharer_id'] == self.selected_sharer_id]
            desc = (f"**Pending shares from {name}:**\n" + ", ".join(user_shares)
                    + "\n\n-# A borrow follows its owner's edits; a (clone) becomes your own "
                    "independent copy, without memories.\n"
                    "-# **Block Sender** rejects these and stops them sharing with you again.")
            embed = discord.Embed(title="Reviewing Shares", description=desc, color=discord.Color.blue())
            await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)
            return

        if not self.cog.profile_manager.is_registered(self.user_id):
            embed = discord.Embed(title="Incoming Shares", description=NOT_REGISTERED,
                                  color=discord.Color.dark_grey())
            await self.original_interaction.edit_original_response(content=None, embed=embed, view=self)
            return

        about = self._about()
        if about.get("shares_closed"):
            status = ("**Closed.** Nobody can offer you a profile. Shares already waiting stay "
                      "until you answer them.")
        else:
            status = ("**Open.** Anyone who has set MimicAI up can offer you a profile, "
                      "unless you block them. Nobody is messaged; offers wait here.")
        waiting = ("Select a user from the dropdown to accept or reject their shared profiles."
                   if shares else "You have no shares waiting.")
        blocked = len(about.get("blocked_sharers") or [])
        embed = discord.Embed(
            title="Incoming Shares", color=discord.Color.blue() if shares else discord.Color.dark_grey(),
            description=f"{status}\n\n{waiting}\n\n-# Blocked: {blocked} of {self.BLOCK_LIMIT}. "
                        f"Blocking someone also takes back what they have waiting.")
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

    async def toggle_open(self, i: discord.Interaction):
        if await refuse_unregistered(self.cog, i):
            return
        pm = self.cog.profile_manager
        pm.set_shares_closed(self.user_id, not self._about().get("shares_closed"))
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def set_blocked(self, i: discord.Interaction):
        if await refuse_unregistered(self.cog, i):
            return
        self.cog.profile_manager.set_blocked_sharers(
            self.user_id, [int(v) for v in (i.data or {}).get("values", [])])
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def block_sender(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True)
        if await refuse_unregistered(self.cog, i):
            return
        blocked = self._about().get("blocked_sharers") or []
        if len(blocked) >= self.BLOCK_LIMIT:
            await i.followup.send(f"Your block list is full ({self.BLOCK_LIMIT}). Unblock someone "
                                  "on this tab first.", ephemeral=True)
            return
        self.cog.profile_manager.set_blocked_sharers(self.user_id, [*blocked, self.selected_sharer_id])
        await i.followup.send("Blocked. Their shares are gone, and they cannot offer you more.",
                              ephemeral=True)
        self.selected_sharer_id = None
        self.setup_items()
        await self.update_display()

    async def accept_all(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True)
        if await refuse_unregistered(self.cog, i):
            return
        pm = self.cog.profile_manager
        sharer_id = self.selected_sharer_id
        shares = [s for s in pm._pending_shares(self.user_id) if s['sharer_id'] == sharer_id]
        clones = sum(1 for s in shares if s.get("kind") == "clone")

        # A borrow counts towards the borrowed limit, a clone towards the personal one.
        index = pm._get_user_index(self.user_id)
        for field, count, limit in (("borrowed", len(shares) - clones, defaultConfig.LIMIT_BORROWED),
                                    ("personal", clones, defaultConfig.LIMIT_PROFILES)):
            if count and len(index.get(field, {})) + count > limit:
                await i.followup.send(f"Limit Reached. Accepting these would exceed your limit of {limit} {field} profiles.", ephemeral=True)
                return

        accepted, failed = [], []
        sharer_index = pm._get_user_index(sharer_id)

        for s in shares:
            kind = s.get("kind")
            fallback_name = s['profile_name']
            target_pid = s.get('original_pid')
            current_name = (pm._get_name_from_pid(sharer_id, target_pid) if target_pid else None) or fallback_name

            if current_name not in sharer_index.get("personal", []):
                await pm._reject_share_request(self.original_interaction, sharer_id, target_pid, fallback_name, kind)
                continue

            # The source's own name, amended only where this user already has it.
            local_name = pm._generate_unique_local_name(self.user_id, current_name)
            label = current_name if local_name == current_name else f"{current_name} as {local_name}"
            if kind == "clone":
                ok, message = await pm._execute_clone_handshake(sharer_id, target_pid, self.user_id, local_name)
                if ok:
                    accepted.append(f"{label} (clone)")
                else:
                    failed.append(f"• **{current_name}** -- {message}")
            elif await self.cog.profile_manager._accept_share_request(self.original_interaction, sharer_id, target_pid, current_name, local_name, is_public_borrow=False):
                accepted.append(label)

        msg = f"Accepted: {', '.join(accepted)}" if accepted else "No valid profiles found."
        if failed:
            msg += "\n\n**Not accepted:**\n" + "\n".join(failed)
        await i.followup.send(msg, ephemeral=True)
        self.selected_sharer_id = None
        self.setup_items()
        await self.update_display()

    async def reject_all(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True)
        sharer_id = self.selected_sharer_id
        shares =[s for s in self.cog.profile_manager._pending_shares(self.user_id) if s['sharer_id'] == sharer_id]
        for s in shares:
            await self.cog.profile_manager._reject_share_request(self.original_interaction, sharer_id, s.get('original_pid'), s['profile_name'], s.get("kind"))
        await i.followup.send("Rejected shares.", ephemeral=True)
        self.selected_sharer_id = None
        self.setup_items()
        await self.update_display()

class HubShareManagerView(HubBaseView):
    """Profile Sharing: offer your profiles as borrows. Its two siblings share the screen
    and flip one class attribute each -- Profile Cloning (`CLONE`) and the Library's
    Publish Yours (`PUBLISH`) -- so the three send buttons never sit side by side: one
    lends a borrow, one gives a copy away for good, one lists it for everyone."""
    TAB = "manage"
    CLONE = False
    PUBLISH = False

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, self.TAB)
        self.selected_profiles = []
        self.selected_users = []
        self.processing = False
        
        #: What the dropdown offers, filled by `load_profiles`: whatever is rated for it,
        #: plus whatever has no verdict yet -- rated by the press that sends it. Adult is
        #: left out, because listing it offered a choice every button then refused.
        self.personal_profiles: List[str] = []
        #: The listed ones a press will rate first, so the dropdown can say so.
        self.needs_rating: Set[str] = set()
        self._loaded = False
        self.current_page = 0
        
        self.setup_items()

    @property
    def capability(self) -> str:
        return "publish" if self.PUBLISH else "share"

    def _eligible_profiles(self) -> Tuple[List[str], Set[str]]:
        """(offered, of those the ones needing a rating). Blocking: one rating read each."""
        pm = self.cog.profile_manager
        names = sorted(pm._get_user_index(self.user_id).get("personal", []))
        offered, unrated = set(), set()
        for name in names:
            verdict, _ = pm._content_rating_state(self.user_id, name)
            if CONTENT_RATING_CAPABILITIES[verdict][self.capability]:
                offered.add(name)
            elif verdict in (CONTENT_RATING_UNRATED, CONTENT_RATING_PENDING):
                offered.add(name)
                unrated.add(name)
        if self.PUBLISH:
            # Already listed stays offered whatever its rating, or it could not be
            # deselected to unpublish it.
            offered |= {name for name, _ in self._get_user_public_profiles()} & set(names)
        return sorted(offered), unrated

    async def load_profiles(self):
        self.personal_profiles, self.needs_rating = await asyncio.to_thread(self._eligible_profiles)
        self._loaded = True
        self.setup_items()

    def _get_user_public_profiles(self):
        """Scans public_profiles for entries owned by this user, returning (name, is_locked) pairs."""
        return [(d["profile_name"], d["status"] == "locked")
                for d in self.cog.profile_manager._iter_public_entries(self.user_id)]

    def setup_items(self):
        for item in self.children[:]:
            if item.row != 4: self.remove_item(item)

        # Row 0 holds Clear and the page controls, directly above the select they act on:
        # rows 2 and 3 are recipients then Send, so Send reads as sending to them.
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
        build_pagination_controls(self, self.current_page, num_pages, 0, self.prev_page, self.next_page)

        start = self.current_page * SHARE_PAGE_SIZE
        page_profiles = self.personal_profiles[start : start + SHARE_PAGE_SIZE]

        options = []
        if page_profiles:
            options = bulk_select_options(
                page_profiles, self.personal_profiles,
                lambda p: p in self.selected_profiles,
                page_description="Toggle selection for every profile on this page.",
                all_description="Toggle selection for every profile you own.")

            press = "publish" if self.PUBLISH else "send"
            for p in page_profiles:
                options.append(discord.SelectOption(
                    label=p, value=p, default=(p in self.selected_profiles),
                    description=f"Not rated yet: rated when you {press}" if p in self.needs_rating else None))

        if options:
            add_select(self, options, self.select_profiles, placeholder="Select profiles...",
                       min_values=0, max_values=len(options), row=1)

        if self.PUBLISH:
            add_button(self, "Apply Changes", self.apply_public, style=discord.ButtonStyle.green,
                       row=2)
            # One intro at a time, so only with one profile selected.
            add_button(self, "Edit Intro", self.edit_intro, row=2,
                       disabled=len(self.selected_profiles) != 1)
            add_button(self, "← Library", self.back_to_library, row=2)
            return
        user_sel = ui.UserSelect(placeholder="Select recipients...", min_values=1, max_values=10, row=2)
        user_sel.callback = self.select_users
        self.add_item(user_sel)
        if self.CLONE:
            add_button(self, "Send as Clone", self.send_clone, style=discord.ButtonStyle.blurple, row=3)
        else:
            add_button(self, "Send", self.send_private, style=discord.ButtonStyle.green, row=3)

    async def update_display(self):
        if not self._loaded:
            await self.load_profiles()
        await self.original_interaction.edit_original_response(content=None, embed=self.build_embed(), view=self)

    def build_embed(self) -> discord.Embed:
        rating = ("Profiles not rated yet are rated when you press it, which takes a few "
                  "seconds; Adult 18+ profiles cannot be ")
        if self.PUBLISH:
            title = "Publish to the Library"
            desc = ("List your profiles in the Public Library for anyone to borrow, and deselect "
                    "one to take it out. Existing borrows keep working either way.\n"
                    f"{rating}published. Select a single profile to give it an intro.")
        elif self.CLONE:
            title = "Profile Cloning"
            desc = ("Give your profiles away as copies.\n\n"
                    "**Send as Clone** offers people who have set MimicAI up an independent copy of "
                    "their own, without memories: they can edit it freely and it never follows your "
                    "edits. The offer waits in their Incoming Shares until they answer it; nobody is "
                    "messaged. To lend a profile that follows your edits, use **Profile Sharing**.\n"
                    f"{rating}cloned.")
        else:
            title = "Profile Sharing"
            desc = ("Offer profiles to people who have set MimicAI up. An offer waits in their "
                    "Incoming Shares until they answer it; nobody is messaged.\n"
                    "**Send** lends a borrow that follows your edits. To give them an independent "
                    "copy instead, use **Profile Cloning**; to list a profile for everyone, "
                    "**Public Library** → **Publish Yours**.\n"
                    f"{rating}shared.")
        if not self.personal_profiles:
            desc += "\n\nYou have no profiles that can be offered here."
            
        embed = discord.Embed(title=title, description=desc, color=discord.Color.teal())
        
        full_text = ", ".join(self.selected_profiles)
        # A field value, so 1024. This cut at 4000, the description's limit, and Select All
        # on a large account made Discord refuse the whole embed.
        if len(full_text) > 1024: full_text = full_text[:1021] + "..."
        if not full_text: full_text = "None"
        embed.add_field(name="Selected Profiles", value=full_text, inline=False)

        if self.PUBLISH:
            public_names = [f"{name}{' (Locked)' if is_locked else ''}" for name, is_locked in self._get_user_public_profiles()]

            val = ", ".join(public_names) if public_names else "None"
            if len(val) > 1024: val = val[:1021] + "..."
            embed.add_field(name="Your Currently Public Profiles", value=val, inline=False)

        return embed

    async def edit_intro(self, i: discord.Interaction):
        # Deferred: gui_profiles imports this module at load time.
        from .gui_profiles import LibraryIntroModal

        if len(self.selected_profiles) != 1:
            return
        await i.response.send_modal(LibraryIntroModal(self.cog, self.user_id, self.selected_profiles[0]))

    async def back_to_library(self, i: discord.Interaction):
        await i.response.defer()
        await HubPublicLibraryView(self.cog, self.original_interaction).update_display()

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

    async def send_clone(self, i: discord.Interaction):
        await self.send_private(i, kind="clone")

    async def _rate_selection(self, i: discord.Interaction, names) -> Tuple[List[str], Dict[str, str]]:
        """Splits a selection into (allowed, {name: reason}) for this screen's gate.

        Through `rate_for_distribution`, so a profile with no verdict is rated by the
        press rather than refused, and one edited since its verdict is not let out on it.
        Its refusal wording is the sentence the Content Safety dashboard shows. At most
        RATING_CONCURRENCY at once, each may wait on the classifier; while any needs a
        rating, one ephemeral message counts them off as they land.
        """
        pm = self.cog.profile_manager
        unrated = [n for n in names if n in self.needs_rating]
        progress = None
        if unrated:
            progress = await i.followup.send(
                f"Rating {len(unrated)} profile{'s' if len(unrated) != 1 else ''}, "
                f"{RATING_CONCURRENCY} at a time. This takes a few seconds each.",
                ephemeral=True, wait=True)
        gate = asyncio.Semaphore(RATING_CONCURRENCY)

        async def rate(name):
            async with gate:
                return name, await pm.rate_for_distribution(self.user_id, name, self.capability)

        answers, lines = {}, []
        # Edited from here, not inside `rate`: a rate-limited edit must not hold a slot.
        for done in asyncio.as_completed([rate(n) for n in names]):
            name, answer = await done
            answers[name] = answer
            if progress and name in self.needs_rating:
                lines.append(f"{'✅' if answer[0] else '⚠️'} **{name}**")
                try:
                    await progress.edit(content=f"Rated {len(lines)} of {len(unrated)}\n"
                                                + "\n".join(lines)[-1800:])
                except discord.HTTPException:
                    progress = None   # Gone or refused: the report still lands.
        allowed = [name for name in names if answers[name][0]]
        refused = {name: answers[name][1] or "Not available." for name in names if not answers[name][0]}
        return allowed, refused

    async def send_private(self, i: discord.Interaction, kind: Optional[str] = None):
        """Offers the selection to each recipient; `kind` is the share's, sparse as
        `ProfileManager._is_share_of` reads it."""
        if self.processing: return
        self.processing = True

        for item in self.children:
            if isinstance(item, ui.Button) and str(item.label).startswith("Send"): item.disabled = True
        await i.response.edit_message(view=self)

        if not self.selected_profiles or not self.selected_users:
            await i.followup.send("Select profiles and recipients first.", ephemeral=True)
            self.processing = False
            self.setup_items()
            await self.update_display()
            return
        
        shareable, refused = await self._rate_selection(i, self.selected_profiles)
        self._loaded = False   # A rating may have landed: re-read what needs one.
        if not shareable:
            self.processing = False
            self.setup_items()
            await self.update_display()
            lines = "\n".join(f"• **{n}** -- {r}" for n, r in refused.items())
            await i.followup.send(
                f"None of the selected profiles can be shared yet.\n{lines}", ephemeral=True)
            return

        pm = self.cog.profile_manager
        pids = {name: pm._get_pid_from_name_any(self.user_id, name) for name in shareable}
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        offered, turned_away = [], []
        for recipient_id_str in self.selected_users:
            recipient_id = int(recipient_id_str)
            if recipient_id == self.user_id:
                continue
            waiting = pm._pending_shares(recipient_id)
            mine = [s for s in waiting if s.get('sharer_id') == self.user_id]
            new = [n for n in shareable
                   if not any(pm._is_share_of(s, self.user_id, pids[n], n, kind) for s in mine)]
            # Unregistered, closed, blocked or full: one answer for all four, so a
            # sender cannot tell a block from anything else.
            if (not pm.accepts_share(recipient_id, self.user_id)
                    or len(mine) + len(new) > LIMIT_PENDING_SHARES):
                turned_away.append(f"<@{recipient_id}>")
                continue
            if new:
                waiting += [{"sharer_id": self.user_id, "original_pid": pids[n],
                             "profile_name": n, "shared_at": now, **({"kind": kind} if kind else {})}
                            for n in new]
                self.cog.profile_shares[str(recipient_id)] = waiting
                pm._save_profile_share_shard(str(recipient_id), waiting)
            offered.append(f"<@{recipient_id}>")

        self.processing = False
        self.setup_items()
        await self.update_display()
        report = (f"Offered{' as a clone' if kind else ''} to {', '.join(offered)}. It waits in their Incoming Shares "
                  f"(`/profile hub`) until they answer." if offered else "Nothing was sent.")
        if turned_away:
            report += (f"\n\nNot sent to {', '.join(turned_away)}: they are not taking shares "
                       f"right now.")
        if refused:
            lines = "\n".join(f"• **{n}** -- {r}" for n, r in refused.items())
            report += f"\n\n**Left out:**\n{lines}"
        await i.followup.send(report, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())

    async def apply_public(self, i: discord.Interaction):
        if self.processing: return
        self.processing = True

        for item in self.children:
            if isinstance(item, ui.Button) and item.label == "Apply Changes": item.disabled = True
        await i.response.edit_message(view=self)

        pm = self.cog.profile_manager
        # Read through the normaliser: the old inline version resolved string entries via
        # a name.txt that is never written, so every already-public profile landed in
        # `to_publish` and was judged again.
        current_public_set = {d["profile_name"] for d in pm._iter_public_entries(self.user_id)}
        target_set = set(self.selected_profiles)
        to_publish = sorted(target_set - current_public_set)
        to_unpublish = sorted(current_public_set - target_set)

        # Past the day's limit is refused before rating, so it spends no classifier call.
        left, refusal = pm.publishes_left(self.user_id)
        over_limit = {name: refusal for name in to_publish[left:]}
        # Publishing analyses nothing of its own. It reads the rating, because that rating
        # *is* the safety analysis -- a profile with none is rated here, by this press.
        allowed, failed_list = await self._rate_selection(i, to_publish[:left])
        failed_list.update(over_limit)
        published_list = []
        for name in allowed:
            pid_entry = pm._get_pid_from_name_any(self.user_id, name)
            self.cog.public_profiles[pid_entry] = f"{self.user_id}:{pid_entry}"
            published_list.append(name)
        if published_list:
            pm._save_public_index()
            pm.note_publishes(self.user_id, len(published_list))

        unpublished_list = [name for name in to_unpublish if pm.unpublish_profile(self.user_id, name)]

        self.processing = False
        self._loaded = False
        self.selected_profiles = [name for name, _ in self._get_user_public_profiles()]
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

class HubCloneManagerView(HubShareManagerView):
    TAB = "clone"
    CLONE = True


class HubPublishView(HubShareManagerView):
    """The Library's Publish Yours: opens on what is already listed, so deselecting one
    and pressing Apply Changes unpublishes it."""
    TAB = "library"
    PUBLISH = True

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction)
        self.selected_profiles = [name for name, _ in self._get_user_public_profiles()]
        self.setup_items()


class BorrowDefaultsView(BlockedGuard, ui.View):
    """Asked by `_accept_share_request` when the borrower's defaults would change what the
    author chose. Answered or not, the borrow goes ahead: unanswered keeps the original's."""

    def __init__(self, cog: 'MimicCog'):
        super().__init__(timeout=120)
        self.cog = cog
        self.use_mine = False

    async def _answer(self, interaction: discord.Interaction, use_mine: bool):
        self.use_mine = use_mine
        await interaction.response.edit_message(
            content="Using your defaults." if use_mine else "Keeping the original's settings.",
            view=None)
        self.stop()

    @ui.button(label="Use my defaults", style=discord.ButtonStyle.primary)
    async def mine(self, interaction: discord.Interaction, _button: ui.Button):
        await self._answer(interaction, True)

    @ui.button(label="Keep the original's", style=discord.ButtonStyle.secondary)
    async def original(self, interaction: discord.Interaction, _button: ui.Button):
        await self._answer(interaction, False)
