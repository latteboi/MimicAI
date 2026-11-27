from ..utils.constants import *

import discord
from discord import ui
import asyncio
import datetime
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from ..utils.helpers import default_profile_avatar_url
from .base_components import (BlockedGuard, TimeoutCleanupMixin,
                              add_button, add_select, build_confirm_view,
                              build_pagination_controls, build_tab_nav_bar, bulk_select_options,
                              refuse_unregistered, resolve_bulk_select,
                              add_topgg_links, send_status, settle_status)
from .gui_start import open_generate
from .base_components import closed_embed, dismissible, embed_panel, text_embed

#: Said to a borrower wherever a borrow is taken: what they are and are not getting.
BORROW_TERMS = "-# A borrow follows its owner's edits, and ends if they delete it or withdraw it from you."

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog


class HubBaseView(BlockedGuard, ui.LayoutView):
    """The hub's tabs, in Components V2: a panel, the tab's controls, then the tab bar.

    Every tab, not only the Library: the V2 flag never comes off a message, and every tab is
    reached on the Library's. A tab puts its controls in `row(n)` from `_build_rows` -- a V2
    button lives in an action row, never loose -- and paints through `_paint`, with a panel
    or an embed (`embed_panel`). `setup_items` lays the whole message out again, so an
    `edit_message(view=self)` straight after it carries the panel too: a V2 message is
    nothing but components, and one sent without its panel loses its words.
    """

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

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, current_tab: str):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        #: The newest interaction known to have updated this message, and the newest press,
        #: which may have (`_repaint_via`).
        self._via, self._press = interaction, None
        self.user_id = interaction.user.id
        self.current_tab = current_tab
        self.current_page = 0
        self._rows: Dict[int, ui.ActionRow] = {}
        self._panel: Optional[ui.Container] = None
        self._tabs = ui.ActionRow()
        build_tab_nav_bar(self._tabs, current_tab,
                          [(label, key, self._go(factory)) for label, key, factory in self.TABS])

    def _go(self, factory):
        async def callback(i: discord.Interaction):
            await i.response.defer()
            await self._switch(factory(self), i)
        return callback

    async def _switch(self, view: 'HubBaseView', press: discord.Interaction):
        """Paint `view` over this message through `press`, then stop this one: left running,
        its timeout would later paint its own stale panel over whatever is showing by then."""
        view._via, view._press = self._via, press
        await view.update_display()
        self.stop()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not await super().interaction_check(interaction):
            return False
        self._press = interaction
        return True

    def _repaint_via(self) -> discord.Interaction:
        """The interaction to edit this message through: the newest press, once it has
        answered by updating this message, else the last one that did. An interaction's
        token lasts fifteen minutes from its own press; repainting through the one that
        opened the hub left it dead to anyone using it for longer. A modal's submission is
        not a press the view sees, so whatever takes one sets `_press` to it."""
        answered = getattr(getattr(self._press, "response", None), "type", None)
        if answered in (discord.InteractionResponseType.deferred_message_update,
                        discord.InteractionResponseType.message_update):
            self._via = self._press
        return self._via

    def row(self, n: int) -> ui.ActionRow:
        """Control row `n`, under the panel; rows are laid out in order, the tab bar last."""
        return self._rows.setdefault(n, ui.ActionRow())

    def _build_rows(self):
        """This tab's controls, into `row(n)`. None by default."""

    def setup_items(self):
        self._rows = {}
        self._build_rows()
        self._lay_out()

    def _lay_out(self, controls: bool = True):
        self.clear_items()
        if self._panel is not None:
            self.add_item(self._panel)
        if controls:
            for n in sorted(self._rows):
                if self._rows[n].children:
                    self.add_item(self._rows[n])
            self.add_item(self._tabs)

    async def _paint(self, panel):
        """Show `panel` -- a `ui.Container`, or an embed drawn as one -- above the controls."""
        self._panel = panel if isinstance(panel, ui.Container) else embed_panel(panel)
        self._lay_out()
        await self._repaint_via().edit_original_response(view=self)

    async def _turn_page(self, i: discord.Interaction, delta: int):
        self._press = i   # a page jump's modal submission, as well as a press
        self.current_page += delta
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def update_display(self):
        raise NotImplementedError

    async def on_timeout(self):
        """The words stay and the controls go -- a V2 message with nothing in it is refused --
        and any control inside the panel (the Library's Borrow buttons) is greyed."""
        if self._panel is None:
            return
        self._lay_out(controls=False)
        for item in self.walk_children():
            if hasattr(item, "disabled"):
                item.disabled = True
        try:
            await self._repaint_via().edit_original_response(view=self)
        except discord.HTTPException:
            pass   # gone, or past even the newest token's fifteen minutes


def _borrower_count(cog, owner_id: int, pid: Optional[str]) -> int:
    """People borrowing one profile, counted once each: one borrow-index lookup."""
    return len({borrower_id for borrower_id, _ in cog.borrows._borrowers_of(owner_id, pid)}) if pid else 0


class HubStatsView(HubBaseView):
    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction):
        super().__init__(cog, interaction, "stats")

    def build_embed(self) -> discord.Embed:
        """Blocking: one owner-index read per creator, as the Library's own open."""
        pm = self.cog.profile_manager
        listings = list(self.cog.library._iter_public_entries())
        reach = {d["id"]: _borrower_count(self.cog, d["owner_id"], d["original_pid"]) for d in listings}
        creators = {d["owner_id"] for d in listings}

        index = pm._get_user_index(self.user_id)
        personal = list(index.get("personal", {}))
        borrowed = len(index.get("borrowed", {}))
        waiting = len(self.cog.sharing._pending_shares(self.user_id))
        mine = [d for d in listings if d["owner_id"] == self.user_id]
        # Every profile of yours, listed or only shared privately: reach is reach.
        yours = {name: {b for b, _ in self.cog.borrows._borrowers_of(self.user_id, pm._get_pid_from_name(self.user_id, name))}
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
        await self._paint(embed)

#: The Library's orders, in the order Sort steps through them.
LIBRARY_SORTS = {"borrowed": "Most Borrowed", "new": "Newest", "updated": "Recently Updated"}


class HubPublicLibraryView(HubBaseView):
    """The Public Library: a page of cards, each a listing's words beside its picture with its
    own Borrow button underneath (Edit Intro on your own, greyed once borrowed). A card reads
    its listing's public entry (`LibraryManager._listing`), so a page opens no profile.

    Four to a page because Discord allows a message forty components: a card is five
    (section, text, picture, row, button), and the page row and the tab bar twelve more.
    """

    PER_PAGE = 4

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, filtered_list=None):
        super().__init__(cog, interaction, "library")
        #: A key of LIBRARY_SORTS.
        self.sort = "borrowed"
        #: The search in force, lower-cased; "" for none.
        self.search = ""
        #: Creator id -> name, or None where Discord has none (`_resolve_owners`).
        self._owners: Dict[int, Optional[str]] = {}
        # On the loop this once, so the view has controls before its first paint; every
        # repaint after reads the Library again in a worker (`update_display`).
        self._take_in(self._sorted(self.cog.library._iter_public_entries()))
        if filtered_list is not None:
            self.filtered_list = filtered_list
        self.setup_items()

    def _take_in(self, entries: List[dict]):
        """`entries`, sorted, as the Library on show, under the search in force."""
        self.all_public = entries
        self.filtered_list = ([p for p in entries if self._matches(p, self.search)]
                              if self.search else entries)
        #: Listings per creator, counted over the whole Library, not a search's results.
        self._shelf = Counter(p['owner_id'] for p in entries)

    def _sorted(self, entries):
        """Newest first, most recently updated first, or most borrowed first with newest
        breaking ties."""
        if self.sort == "updated":
            return sorted(entries, key=lambda x: x['updated_at'], reverse=True)
        newest = sorted(entries, key=lambda x: x['published_at'], reverse=True)
        if self.sort != "borrowed":
            return newest
        return sorted(newest, key=lambda x: _borrower_count(self.cog, x['owner_id'], x.get('original_pid')),
                      reverse=True)

    def _num_pages(self) -> int:
        return max(1, (len(self.filtered_list) - 1) // self.PER_PAGE + 1)

    def _page(self) -> List[dict]:
        """The listings on the page on show, holding the page inside the list first: a search
        or a re-sort can leave it past the end."""
        self.current_page = max(0, min(self.current_page, self._num_pages() - 1))
        start = self.current_page * self.PER_PAGE
        return self.filtered_list[start:start + self.PER_PAGE]

    def _has_borrowed(self, p_info) -> bool:
        """Whether the viewer already holds a borrow of this listing: one borrow-index lookup.

        This runs on every repaint. It used to read the config of every profile the
        viewer had borrowed, and matched on `original_profile_name` -- a snapshot, so a
        listing renamed after the borrow offered Borrow again.
        """
        return any(borrower_id == self.user_id for borrower_id, _ in
                   self.cog.borrows._borrowers_of(p_info['owner_id'], p_info.get('original_pid')))

    async def _resolve_owners(self, entries):
        """The creators' names for these listings, fetched once each where Discord's cache has
        none: a creator need not share a server with the bot any more."""
        async def name_of(uid: int):
            user = self.cog.bot.get_user(uid)
            if user is None:
                try:
                    user = await self.cog.bot.fetch_user(uid)
                except discord.HTTPException:
                    pass
            self._owners[uid] = user.name if user else None

        await asyncio.gather(*(name_of(uid) for uid in {p['owner_id'] for p in entries} - self._owners.keys()))

    def _card(self, p_info) -> Tuple[str, str]:
        """(words, picture URL) for one listing's card."""
        pid, owner_id, name = p_info.get('original_pid'), p_info['owner_id'], p_info['profile_name']
        listing = self.cog.library._listing(owner_id, pid) if pid else {}
        avatar = listing.get("avatar") or ""
        # A link Discord will not draw refuses the whole message, not just its card.
        if not avatar.startswith(("https://", "http://")):
            avatar = default_profile_avatar_url(name)
        owner = (self._owners[owner_id] if owner_id in self._owners
                 else getattr(self.cog.bot.get_user(owner_id), "name", None))
        # Escaped: a name is the author's to choose, and V2 text is markdown throughout. The
        # intro is not -- formatting it is the author's to choose too.
        byline = [f"by **{discord.utils.escape_markdown(owner or 'Unknown')}**",
                  f"{self._shelf[owner_id]} in the Library"]
        borrowers = _borrower_count(self.cog, owner_id, pid)
        if borrowers:
            byline.append(f"Borrowed by {borrowers} {'user' if borrowers == 1 else 'users'}")
        listed, updated = int(p_info.get('published_at') or 0), int(p_info.get('updated_at') or 0)
        # Updated is when its rating last landed, and listing it was one: only a later one says more.
        dates = [*([f"Listed <t:{listed}:R>"] if listed else []),
                 *([f"Updated <t:{updated}:R>"] if updated > listed else [])]
        lines = (f"### {discord.utils.escape_markdown(listing.get('display_name') or name)}",
                 listing.get("intro") or "",
                 "-# " + " · ".join(byline),
                 "-# " + " · ".join(dates) if dates else "")
        return "\n".join(filter(None, lines)), avatar

    def _take(self, p_info) -> Dict[str, Any]:
        """A card's own button: its owner's Edit Intro, else Borrow, greyed once held."""
        if self.user_id == p_info['owner_id']:
            return dict(label="Edit Intro", style=discord.ButtonStyle.blurple,
                        callback=lambda i: self._edit_intro(i, p_info))
        if self._has_borrowed(p_info):
            return dict(label="Borrowed", style=discord.ButtonStyle.grey, disabled=True,
                        callback=lambda i: self._borrow(i, p_info))
        return dict(label="Borrow", style=discord.ButtonStyle.green,
                    callback=lambda i: self._borrow(i, p_info))

    def _build_rows(self):
        page = self._page()
        panel = ui.Container(accent_colour=discord.Colour.gold())
        panel.add_item(ui.TextDisplay("## Public Library\n" + (
            f"-# Page {self.current_page + 1} of {self._num_pages()} · {len(self.filtered_list)} "
            f"profiles · {LIBRARY_SORTS[self.sort]}" if page else "No profiles found.")))
        for n, p_info in enumerate(page):
            if n:   # A gap between one card's button and the next card's name.
                panel.add_item(ui.Separator(visible=False, spacing=discord.SeparatorSpacing.large))
            words, picture = self._card(p_info)
            panel.add_item(ui.Section(ui.TextDisplay(words), accessory=ui.Thumbnail(picture)))
            take = ui.ActionRow()
            add_button(take, **self._take(p_info))
            panel.add_item(take)
        self._panel = panel

        build_pagination_controls(self.row(0), self.current_page, self._num_pages(), 0, self._turn_page)
        add_button(self.row(0), "Search", self.search_cb)
        add_button(self.row(0), f"Sort: {LIBRARY_SORTS[self.sort]}", self.sort_cb)
        add_button(self.row(1), "Generate", self.generate_cb, style=discord.ButtonStyle.blurple, emoji="✨")
        add_button(self.row(1), "Publish Yours", self.publish_cb, style=discord.ButtonStyle.green, emoji="🌐")

    async def update_display(self):
        # Read again at every repaint, so a page shows the Library as it stands: a listing
        # taken down since leaves, a new one arrives. Owner indexes in a worker, where a
        # cache miss reads a file; the search's creator names off Discord's cache on the loop.
        self._take_in(await asyncio.to_thread(
            lambda: self._sorted(self.cog.library._iter_public_entries())))
        await self._resolve_owners(self._page())
        self.setup_items()
        await self._repaint_via().edit_original_response(view=self)

    async def _borrow(self, i: discord.Interaction, p_info):
        if await refuse_unregistered(self.cog, i):
            return
        if i.user.id == p_info['owner_id']:
            await i.response.send_message(embed=text_embed("You cannot borrow your own profile."), ephemeral=True)
            return
        if self._has_borrowed(p_info):
            await i.response.send_message(embed=text_embed("You already have this profile."), ephemeral=True)
            return

        await i.response.defer(ephemeral=True)
        local_name = self.cog.transfer._generate_unique_local_name(i.user.id, p_info['profile_name'])
        if await self.cog.sharing._accept_share_request(i, p_info['owner_id'], p_info.get('original_pid'), p_info['profile_name'], local_name, is_public_borrow=True):
            await i.followup.send(embed=text_embed(
                f"✅ Borrowed **{p_info['profile_name']}** as **{local_name}**.\n{BORROW_TERMS}"), ephemeral=True)
            await self.update_display()   # its card reads Borrowed now

    async def _edit_intro(self, i: discord.Interaction, p_info):
        # Deferred: gui_profiles imports this module at load time.
        from .gui_profiles import LibraryIntroModal

        if i.user.id != p_info['owner_id']:
            return
        # Resolved from the PID: an entry's stored name is a snapshot of when it was
        # published, and the profile may have been renamed since.
        name = self.cog.profile_manager._get_name_from_pid(p_info['owner_id'], p_info.get('original_pid'))
        if not name:
            await i.response.send_message(embed=text_embed("That profile no longer exists."), ephemeral=True)
            return

        async def repaint(submitted: discord.Interaction):
            self._press = submitted
            await self.update_display()

        await i.response.send_modal(LibraryIntroModal(self.cog, i.user.id, name, on_saved=repaint))

    async def generate_cb(self, i: discord.Interaction):
        await open_generate(self.cog, i)

    def _matches(self, entry, term: str) -> bool:
        """The term in the profile's name, its creator's, its display name or its intro --
        the last two copied into the listing's entry, so this reads no file."""
        owner = self.cog.bot.get_user(entry['owner_id'])
        return (term in entry['profile_name'].lower()
                or (owner is not None and term in owner.name.lower())
                or term in self.cog.library.library_search_text(
                    entry['owner_id'], entry.get('original_pid')))

    async def search_cb(self, i: discord.Interaction):
        modal = ui.Modal(title="Search Public Library")
        inp = ui.TextInput(label="Name, creator or words from the intro", required=False)
        modal.add_item(inp)
        async def on_submit(mi: discord.Interaction):
            await mi.response.defer()
            self.search, self.current_page, self._press = inp.value.lower().strip(), 0, mi
            await self.update_display()
        modal.on_submit = on_submit
        await i.response.send_modal(modal)

    async def sort_cb(self, i: discord.Interaction):
        order = list(LIBRARY_SORTS)
        self.sort = order[(order.index(self.sort) + 1) % len(order)]
        self._take_in(self._sorted(self.all_public))
        self.current_page = 0
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def publish_cb(self, i: discord.Interaction):
        if await refuse_unregistered(self.cog, i):
            return
        await i.response.defer()
        await self._switch(HubPublishView(self.cog, self.original_interaction), i)


class HubIncomingView(HubBaseView):
    """Shares waiting for you, and who may send them.

    A share is only ever offered: it waits here, and nobody is messaged to say so --
    see `SharingManager.accepts_share`. Closing turns away new ones and leaves those
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
        return self.cog.user_settings.get_user_about(self.user_id)

    def _build_rows(self):

        shares = self.cog.sharing._pending_shares(self.user_id)
        
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

            add_select(self.row(0), options, self.select_sharer,
                       placeholder="Select a user to review shares...", min_values=1,
                       max_values=1, row=0)

        # Row 1: Pagination (if needed)
        build_pagination_controls(self.row(1), self.current_page, num_pages, 1, self._turn_page)

        # Row 2: Actions
        action_row = 2 if num_pages > 1 else 1 # Move up if no pagination
        
        if self.selected_sharer_id:
            add_button(self.row(action_row), "Accept All", self.accept_all, style=discord.ButtonStyle.green,
                       row=action_row)
            add_button(self.row(action_row), "Reject All", self.reject_all, style=discord.ButtonStyle.danger,
                       row=action_row)
            add_button(self.row(action_row), "Block Sender", self.block_sender, style=discord.ButtonStyle.danger,
                       row=action_row, emoji="🚫")
            add_button(self.row(action_row), "Cancel Selection", self.clear_selection, row=action_row)
            return

        if not self.cog.user_settings.is_registered(self.user_id):
            return
        about = self._about()
        closed = bool(about.get("shares_closed"))
        add_button(self.row(action_row), f"Incoming Shares: {'Closed' if closed else 'Open'}", self.toggle_open,
                   style=discord.ButtonStyle.secondary if closed else discord.ButtonStyle.green,
                   row=action_row)
        blocked = (about.get("blocked_sharers") or [])[:self.BLOCK_LIMIT]
        block_select = ui.UserSelect(
            placeholder="Blocked users -- tick to block, untick to unblock...",
            min_values=0, max_values=self.BLOCK_LIMIT, row=3,
            default_values=[discord.Object(id=uid) for uid in blocked])
        block_select.callback = self.set_blocked
        self.row(3).add_item(block_select)

    async def update_display(self):
        shares = self.cog.sharing._pending_shares(self.user_id)
        
        if self.selected_sharer_id:
            u = self.cog.bot.get_user(self.selected_sharer_id)
            name = u.name if u else "Unknown"
            user_shares = [f"`{s['profile_name']}`" + (" (clone)" if s.get("kind") == "clone" else "")
                           for s in shares if s['sharer_id'] == self.selected_sharer_id]
            desc = (f"**Pending shares from {name}:**\n" + ", ".join(user_shares)
                    + f"\n\n{BORROW_TERMS}\n-# A (clone) becomes your own independent copy, "
                    "without memories.\n"
                    "-# **Block Sender** rejects these and stops them sharing with you again.")
            embed = discord.Embed(title="Reviewing Shares", description=desc, color=discord.Color.blue())
            await self._paint(embed)
            return

        if not self.cog.user_settings.is_registered(self.user_id):
            embed = discord.Embed(title="Incoming Shares", description=NOT_REGISTERED,
                                  color=discord.Color.dark_grey())
            await self._paint(embed)
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
        await self._paint(embed)

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
        self.cog.sharing.set_shares_closed(self.user_id, not self._about().get("shares_closed"))
        self.setup_items()
        await i.response.defer()
        await self.update_display()

    async def set_blocked(self, i: discord.Interaction):
        if await refuse_unregistered(self.cog, i):
            return
        self.cog.sharing.set_blocked_sharers(
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
            await i.followup.send(embed=text_embed(f"Your block list is full ({self.BLOCK_LIMIT}). Unblock someone "
                                  "on this tab first."), ephemeral=True)
            return
        self.cog.sharing.set_blocked_sharers(self.user_id, [*blocked, self.selected_sharer_id])
        await i.followup.send(embed=text_embed("Blocked. Their shares are gone, and they cannot offer you more."),
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
        shares = [s for s in self.cog.sharing._pending_shares(self.user_id) if s['sharer_id'] == sharer_id]
        clones = sum(1 for s in shares if s.get("kind") == "clone")

        # A borrow counts towards the borrowed limit, a clone towards the personal one.
        index = pm._get_user_index(self.user_id)
        for field, count, limit in (("borrowed", len(shares) - clones, defaultConfig.LIMIT_BORROWED),
                                    ("personal", clones, defaultConfig.LIMIT_PROFILES)):
            if count and len(index.get(field, {})) + count > limit:
                await i.followup.send(embed=text_embed(f"Limit Reached. Accepting these would exceed your limit of {limit} {field} profiles."), ephemeral=True)
                return

        accepted, failed = [], []
        sharer_index = pm._get_user_index(sharer_id)

        for s in shares:
            kind = s.get("kind")
            fallback_name = s['profile_name']
            target_pid = s.get('original_pid')
            current_name = (pm._get_name_from_pid(sharer_id, target_pid) if target_pid else None) or fallback_name

            if current_name not in sharer_index.get("personal", []):
                await self.cog.sharing._reject_share_request(i, sharer_id, target_pid, fallback_name, kind)
                continue

            # The source's own name, amended only where this user already has it.
            local_name = self.cog.transfer._generate_unique_local_name(self.user_id, current_name)
            label = current_name if local_name == current_name else f"{current_name} as {local_name}"
            if kind == "clone":
                ok, message = await self.cog.sharing._execute_clone_handshake(sharer_id, target_pid, self.user_id, local_name)
                if ok:
                    accepted.append(f"{label} (clone)")
                else:
                    failed.append(f"• **{current_name}** -- {message}")
            elif await self.cog.sharing._accept_share_request(i, sharer_id, target_pid, current_name, local_name, is_public_borrow=False):
                accepted.append(label)

        msg = f"Accepted: {', '.join(accepted)}" if accepted else "No valid profiles found."
        if failed:
            msg += "\n\n**Not accepted:**\n" + "\n".join(failed)
        await i.followup.send(embed=text_embed(msg), ephemeral=True)
        self.selected_sharer_id = None
        self.setup_items()
        await self.update_display()

    async def reject_all(self, i: discord.Interaction):
        await i.response.defer(ephemeral=True)
        sharer_id = self.selected_sharer_id
        shares =[s for s in self.cog.sharing._pending_shares(self.user_id) if s['sharer_id'] == sharer_id]
        for s in shares:
            await self.cog.sharing._reject_share_request(i, sharer_id, s.get('original_pid'), s['profile_name'], s.get("kind"))
        await i.followup.send(embed=text_embed("Rejected shares."), ephemeral=True)
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
        #: plus whatever has no verdict yet -- rated by the press that sends it. Not Allowed
        #: is left out, because listing it offered a choice every button then refused.
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
        """(offered, of those the ones needing a rating). Verdicts off the catalogue: what is
        offered, not what is let out -- the press rates for distribution itself."""
        pm = self.cog.profile_manager
        names = sorted(pm._get_user_index(self.user_id).get("personal", []))
        offered, unrated = set(), set()
        for name in names:
            verdict = pm.catalogue_entry(self.user_id, name)[1]
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
                for d in self.cog.library._iter_public_entries(self.user_id)]

    def _build_rows(self):

        # Row 0 holds Clear and the page controls, directly above the select they act on:
        # rows 2 and 3 are recipients then Send, so Send reads as sending to them.
        if self.selected_profiles:
            add_button(self.row(0), f"Clear ({len(self.selected_profiles)})", self.clear_selection,
                       style=discord.ButtonStyle.secondary, row=0)

        # Row 1: Paginated Profiles.
        #
        # Page size is SHARE_PAGE_SIZE rather than the full 25, because the two bulk
        # rows occupy option slots: Discord caps a select at 25 options total, so a
        # full page plus the sentinels would be rejected outright.
        num_pages = max(1, (len(self.personal_profiles) - 1) // SHARE_PAGE_SIZE + 1)
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        build_pagination_controls(self.row(0), self.current_page, num_pages, 0, self._turn_page)

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
            add_select(self.row(1), options, self.select_profiles, placeholder="Select profiles...",
                       min_values=0, max_values=len(options), row=1)

        if self.PUBLISH:
            add_button(self.row(2), "Apply Changes", self.apply_public, style=discord.ButtonStyle.green,
                       row=2)
            # One intro at a time, so only with one profile selected.
            add_button(self.row(2), "Edit Intro", self.edit_intro, row=2,
                       disabled=len(self.selected_profiles) != 1)
            add_button(self.row(2), "← Library", self.back_to_library, row=2)
            return
        user_sel = ui.UserSelect(placeholder="Select recipients...", min_values=1, max_values=10, row=2)
        user_sel.callback = self.select_users
        self.row(2).add_item(user_sel)
        if self.CLONE:
            add_button(self.row(3), "Send as Clone", self.send_clone, style=discord.ButtonStyle.blurple, row=3)
        else:
            add_button(self.row(3), "Send", self.send_private, style=discord.ButtonStyle.green, row=3)

    async def update_display(self):
        if not self._loaded:
            await self.load_profiles()
        await self._paint(self.build_embed())

    def build_embed(self) -> discord.Embed:
        if self.PUBLISH:
            title = "Publish to the Library"
            desc = ("List your profiles in the Public Library for anyone to borrow.\n"
                    "• Select profiles to list them, deselect to take them out, then press "
                    "**Apply Changes**.\n"
                    "• Select just one to give it an intro with **Edit Intro**.\n"
                    "-# Existing borrows keep working either way.\n"
                    "-# Unrated profiles are rated when you apply (a few seconds each); "
                    "Not Allowed ones cannot be published.")
        else:
            send, verb = ("Send as Clone", "cloned") if self.CLONE else ("Send", "shared")
            title, lead, aside = (
                ("Profile Cloning",
                 "Give someone their own copy. It has no memories, they can edit it, and your "
                 "edits never reach it.",
                 "To lend one that follows your edits instead: **Profile Sharing**.")
                if self.CLONE else
                ("Profile Sharing",
                 "Lend profiles to other users. A borrow follows your edits.",
                 "To give a copy away: **Profile Cloning**. To list it for everyone: "
                 "**Public Library → Publish Yours**."))
            desc = (f"{lead}\n"
                    f"• Pick profiles and people, then press **{send}**.\n"
                    "• It waits in their **Incoming Shares**; nobody is messaged.\n"
                    f"-# {aside}\n"
                    "-# Recipients need MimicAI set up. Unrated profiles are rated when you send "
                    f"(a few seconds each); Not Allowed ones cannot be {verb}.")
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
        await self._switch(HubPublicLibraryView(self.cog, self.original_interaction), i)

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
        unrated = [n for n in names if n in self.needs_rating]
        progress = None
        if unrated:
            progress = await i.followup.send(
                embed=text_embed(f"Rating {len(unrated)} profile{'s' if len(unrated) != 1 else ''}, "
                f"{RATING_CONCURRENCY} at a time. This takes a few seconds each."),
                ephemeral=True, wait=True)
        gate = asyncio.Semaphore(RATING_CONCURRENCY)

        async def rate(name):
            async with gate:
                return name, await self.cog.content_rating.rate_for_distribution(self.user_id, name, self.capability)

        answers, lines = {}, []
        # Edited from here, not inside `rate`: a rate-limited edit must not hold a slot.
        for done in asyncio.as_completed([rate(n) for n in names]):
            name, answer = await done
            answers[name] = answer
            if progress and name in self.needs_rating:
                lines.append(f"{'✅' if answer[0] else '⚠️'} **{name}**")
                try:
                    await progress.edit(embed=text_embed(f"Rated {len(lines)} of {len(unrated)}\n"
                                                         + "\n".join(lines)[-3800:]))
                except discord.HTTPException:
                    progress = None   # Gone or refused: the report still lands.
        allowed = [name for name in names if answers[name][0]]
        refused = {name: answers[name][1] or "Not available." for name in names if not answers[name][0]}
        return allowed, refused

    async def send_private(self, i: discord.Interaction, kind: Optional[str] = None):
        """Offers the selection to each recipient; `kind` is the share's, sparse as
        `SharingManager._is_share_of` reads it."""
        if self.processing: return
        self.processing = True

        for item in self.walk_children():
            if isinstance(item, ui.Button) and str(item.label).startswith("Send"): item.disabled = True
        await i.response.edit_message(view=self)

        if not self.selected_profiles or not self.selected_users:
            await i.followup.send(embed=text_embed("Select profiles and recipients first."), ephemeral=True)
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
                embed=text_embed(f"None of the selected profiles can be shared yet.\n{lines}"), ephemeral=True)
            return

        pm = self.cog.profile_manager
        pids = {name: pm._get_pid_from_name_any(self.user_id, name) for name in shareable}
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        offered, turned_away = [], []
        for recipient_id_str in self.selected_users:
            recipient_id = int(recipient_id_str)
            if recipient_id == self.user_id:
                continue
            waiting = self.cog.sharing._pending_shares(recipient_id)
            mine = [s for s in waiting if s.get('sharer_id') == self.user_id]
            new = [n for n in shareable
                   if not any(self.cog.sharing._is_share_of(s, self.user_id, pids[n], n, kind) for s in mine)]
            # Unregistered, closed, blocked or full: one answer for all four, so a
            # sender cannot tell a block from anything else.
            if (not self.cog.sharing.accepts_share(recipient_id, self.user_id)
                    or len(mine) + len(new) > LIMIT_PENDING_SHARES):
                turned_away.append(f"<@{recipient_id}>")
                continue
            if new:
                waiting += [{"sharer_id": self.user_id, "original_pid": pids[n],
                             "profile_name": n, "shared_at": now, **({"kind": kind} if kind else {})}
                            for n in new]
                self.cog.profile_shares[str(recipient_id)] = waiting
                self.cog.sharing._save_profile_share_shard(str(recipient_id), waiting)
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
        await i.followup.send(embed=text_embed(report), ephemeral=True, allowed_mentions=discord.AllowedMentions.none())

    async def apply_public(self, i: discord.Interaction):
        if self.processing: return
        self.processing = True

        for item in self.walk_children():
            if isinstance(item, ui.Button) and item.label == "Apply Changes": item.disabled = True
        await i.response.edit_message(view=self)

        # Read through the normaliser: the old inline version resolved string entries via
        # a name.txt that is never written, so every already-public profile landed in
        # `to_publish` and was judged again.
        current_public_set = {d["profile_name"] for d in self.cog.library._iter_public_entries(self.user_id)}
        target_set = set(self.selected_profiles)
        to_publish = sorted(target_set - current_public_set)
        to_unpublish = sorted(current_public_set - target_set)

        # Past the day's limit is refused before rating, so it spends no classifier call.
        left, refusal = self.cog.library.publishes_left(self.user_id)
        over_limit = {name: refusal for name in to_publish[left:]}
        # Publishing analyses nothing of its own. It reads the rating, because that rating
        # *is* the safety analysis -- a profile with none is rated here, by this press.
        allowed, failed_list = await self._rate_selection(i, to_publish[:left])
        failed_list.update(over_limit)
        published_list = []
        for name in allowed:
            if self.cog.library.list_public(self.user_id, name):
                published_list.append(name)
        if published_list:
            self.cog.library._save_public_index()
            self.cog.library.note_publishes(self.user_id, len(published_list))

        unpublished_list = [name for name in to_unpublish if self.cog.library.unpublish_profile(self.user_id, name)]

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

        if published_list:
            report_embed.description = PUBLISHED_REVIEW_LINE
            await i.followup.send(embed=report_embed, ephemeral=True,
                                  view=add_topgg_links(ui.View(timeout=None)))
            return
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


class BorrowersView(BlockedGuard, TimeoutCleanupMixin, ui.View):
    """Who holds a borrow of one of your profiles, and withdrawing it from them.

    The people you shared it with are named: you chose them. Library borrowers are a count
    and are withdrawn together, never picked out -- borrowing from the Library tells its
    author nothing about who did. `borrow_origin` tells the two apart, and anything it
    cannot place is counted, not named.
    """

    #: Names per dropdown page: 25 options, less Select Page and Select All.
    PER_PAGE = DROPDOWN_MAX_OPTIONS - 2

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, owner_id: int, profile_name: str):
        super().__init__(timeout=300)
        self.cog, self.original_interaction = cog, interaction
        self.owner_id, self.profile_name = owner_id, profile_name
        self.selected: Set[str] = set()
        self.page = 0
        self.load()

    def load(self):
        """Every borrow of the profile, by borrower and how it arrived: index lookups only."""
        pm = self.cog.profile_manager
        pid = pm._get_pid_from_name(self.owner_id, self.profile_name)
        self.private: Dict[str, List[str]] = {}
        self.library: Dict[str, List[str]] = {}
        for borrower_id, borrow_pid in self.cog.borrows._borrowers_of(self.owner_id, pid):
            side = self.private if pm.borrow_origin(borrow_pid) == "private" else self.library
            side.setdefault(str(borrower_id), []).append(borrow_pid)
        self.listed = self.cog.library._is_profile_public(self.owner_id, self.profile_name)
        self.selected &= set(self.private)
        self.setup_items()

    def _people(self) -> List[Tuple[str, str]]:
        """(borrower id, name) for everyone it was shared with, by name."""
        named = []
        for uid in self.private:
            user = self.cog.bot.get_user(int(uid))
            named.append((uid, user.name if user else f"ID: {uid}"))
        return sorted(named, key=lambda p: p[1].lower())

    def _page(self) -> List[Tuple[str, str]]:
        return self._people()[self.page * self.PER_PAGE:(self.page + 1) * self.PER_PAGE]

    def setup_items(self):
        self.clear_items()
        num_pages = max(1, (len(self.private) - 1) // self.PER_PAGE + 1)
        self.page = max(0, min(self.page, num_pages - 1))
        page = self._page()
        if page:
            options = bulk_select_options([uid for uid, _ in page], list(self.private),
                                          self.selected.__contains__,
                                          page_description="Tick or untick everyone on this page.",
                                          all_description="Tick or untick everyone you shared it with.")
            options += [discord.SelectOption(label=name[:100], value=uid, default=uid in self.selected)
                        for uid, name in page]
            add_select(self, options, self.select_people, row=0, min_values=0, max_values=len(options),
                       placeholder="People you shared it with: tick to withdraw...")
        build_pagination_controls(self, self.page, num_pages, 1, self._turn_page)
        add_button(self, f"Withdraw from {len(self.selected)} Selected", self.withdraw_selected,
                   row=2, style=discord.ButtonStyle.danger, disabled=not self.selected)
        if self.library:
            label = "Unpublish & Withdraw Library Borrows" if self.listed else "Withdraw Library Borrows"
            add_button(self, f"{label} ({len(self.library)})", self.withdraw_library,
                       row=2, style=discord.ButtonStyle.danger)

    def build_embed(self) -> discord.Embed:
        private, library = len(self.private), len(self.library)
        lines = [f"**Shared privately with:** {private} {'person' if private == 1 else 'people'}",
                 f"**Borrowed from the Library:** {library} {'person' if library == 1 else 'people'}"
                 + (" (who is not shown)" if library else ""),
                 "",
                 "-# Withdrawing deletes their copy, with the memories it made, and cannot be undone. "
                 "They are told the next time they open `/profile hub`."]
        if self.listed:
            lines.append("-# It is in the Public Library, so anyone can borrow it from there "
                         "until you unpublish it.")
        return discord.Embed(title=f"Borrowers: {self.profile_name}"[:256], description="\n".join(lines),
                             color=discord.Color.blue())

    async def select_people(self, i: discord.Interaction):
        self.selected = resolve_bulk_select(i.data.get("values", []), [uid for uid, _ in self._page()],
                                            list(self.private), self.selected)
        self.setup_items()
        await i.response.edit_message(view=self)

    async def _turn_page(self, i: discord.Interaction, delta: int):
        self.page += delta
        self.setup_items()
        await i.response.edit_message(view=self)

    async def withdraw_selected(self, i: discord.Interaction):
        await self._confirm(i, {p for uid in self.selected for p in self.private.get(uid, [])},
                            len(self.selected), unpublish=False)

    async def withdraw_library(self, i: discord.Interaction):
        await self._confirm(i, {p for pids in self.library.values() for p in pids},
                            len(self.library), unpublish=self.listed)

    async def _confirm(self, i: discord.Interaction, borrow_pids: Set[str], people: int, unpublish: bool):
        text = (f"Withdraw **{self.profile_name}** from {people} {'person' if people == 1 else 'people'}? "
                "Their copies are deleted, with the memories they made. This cannot be undone.")
        if unpublish:
            text += "\n\nIt also leaves the Public Library, so nobody can borrow it again."

        async def go(ci: discord.Interaction):
            await send_status(ci, f"Withdrawing {self.profile_name}...")
            ended = await self.cog.sharing.withdraw_borrows(self.owner_id, self.profile_name,
                                                            borrow_pids, unpublish)
            await settle_status(ci, embed=closed_embed(
                "Withdrawn", f"{ended} {'copy' if ended == 1 else 'copies'} of **{self.profile_name}** deleted."
                             + (" It is out of the Public Library." if unpublish else "")), view=None)
            self.load()
            try:
                await self.original_interaction.edit_original_response(embed=self.build_embed(), view=self)
            except discord.HTTPException:
                pass

        await i.response.send_message(embed=text_embed(text), view=build_confirm_view("Withdraw", go, i),
                                      ephemeral=True)


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
            content=None, embed=dismissible(text_embed("Using your defaults." if use_mine else "Keeping the original's settings.")),
            view=None)
        self.stop()

    @ui.button(label="Use my defaults", style=discord.ButtonStyle.primary)
    async def mine(self, interaction: discord.Interaction, _button: ui.Button):
        await self._answer(interaction, True)

    @ui.button(label="Keep the original's", style=discord.ButtonStyle.secondary)
    async def original(self, interaction: discord.Interaction, _button: ui.Button):
        await self._answer(interaction, False)
