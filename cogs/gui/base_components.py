import traceback

import discord
from discord import ui
from typing import Awaitable, Callable, Optional

def build_tab_nav_bar(target_view: ui.View, current_tab: str, tabs, row: int = 4):
    """Attaches a row of tab-navigation buttons to target_view.

    Each entry in tabs is (label, tab_key, async_callback). The button for the
    currently active tab is styled primary and disabled; all others are secondary.
    """
    for label, tab_key, callback in tabs:
        add_button(target_view, label, callback, row=row,
                   style=(discord.ButtonStyle.primary if current_tab == tab_key
                          else discord.ButtonStyle.secondary),
                   disabled=(current_tab == tab_key))

def add_button(view: ui.View, label: str, callback, *, row: int = 0,
               style: discord.ButtonStyle = discord.ButtonStyle.secondary,
               disabled: bool = False, emoji=None, custom_id: Optional[str] = None):
    """Construct a button, wire its callback and add it, in one call.

    The `btn = ui.Button(...)` / `btn.callback = cb` / `add_item(btn)` triple was
    written out roughly 140 times across this package. Returns the button for the few
    callers that keep a reference to it.
    """
    btn = ui.Button(label=label, style=style, row=row, disabled=disabled,
                    emoji=emoji, custom_id=custom_id)
    btn.callback = callback
    view.add_item(btn)
    return btn


def add_select(view: ui.View, options, callback, *, row: int = 0,
               placeholder: Optional[str] = None, min_values: int = 1,
               max_values: int = 1, disabled: bool = False,
               custom_id: Optional[str] = None):
    """`add_button`'s counterpart for a string select."""
    # Not passed as custom_id=None: ui.Select type-checks it and rejects None, where
    # ui.Button quietly generates one.
    extra = {"custom_id": custom_id} if custom_id else {}
    select = ui.Select(options=options, row=row, placeholder=placeholder,
                       min_values=min_values, max_values=max_values,
                       disabled=disabled, **extra)
    select.callback = callback
    view.add_item(select)
    return select


def build_confirm_view(button_label: str, on_confirm) -> ui.View:
    """Builds a single-button (danger-styled) confirmation view wired to on_confirm, timeout=60."""
    view = ui.View(timeout=60)
    add_button(view, button_label, on_confirm, style=discord.ButtonStyle.danger)
    return view

def invalidate_model_cache(cog, user_id: int, profile_name: Optional[str] = None):
    """Drop cached model instances so the next turn rebuilds them from the edited config.

    A cached model carries the system instruction and sampling parameters it was built
    with, so every in-place profile write has to evict it or the setting only takes
    effect once the LRU forgets it.

    Eight call sites had grown four different key predicates: two dropped
    `channel_models` and left `channel_model_last_profile_key` behind, and two widened
    to every profile the user owns in order to change one character's timezone. Keys are
    `(channel_id, owner_id, profile_name)`, with older two-element entries still in the
    cache -- naming a profile narrows to that character, omitting it clears the user.
    """
    stale = [k for k in cog.channel_models
             if isinstance(k, tuple) and len(k) >= 2 and k[1] == user_id
             and (profile_name is None or (len(k) >= 3 and k[2] == profile_name))]
    for k in stale:
        cog.channel_models.pop(k, None)
        cog.channel_model_last_profile_key.pop(k, None)

def compute_window_slice(center_index: int, total_items: int, window_size: int = 25):
    """Computes a [start, end) slice of window_size items centered on center_index (a 0-based
    index into the total_items-length sequence), clamped so the window never runs past either end."""
    half_window = window_size // 2
    start = max(0, center_index - half_window)
    end = start + window_size
    if end > total_items:
        end = total_items
        start = max(0, end - window_size)
    return start, end

class TimeoutCleanupMixin:
    """Strips a view's controls when it times out.

    discord.py stops dispatching to a timed-out view but leaves its buttons sitting in
    the channel, so the user goes on pressing controls that silently do nothing --
    indistinguishable from the bot having broken. Only two of the GUI's ~64 views
    handled this; the rest just went quiet.

    Requires self.original_interaction, and edits through it, which is the same call
    the adopting views already repaint themselves with -- so it is guaranteed to be
    addressing the message they own rather than some later followup.

    Mix in ahead of ui.View so this on_timeout wins the MRO. A view that defines its
    own on_timeout still overrides this one.
    """

    timeout_message: Optional[str] = None

    async def on_timeout(self):
        interaction = getattr(self, "original_interaction", None)
        if interaction is None:
            return
        payload = {"view": None}
        if self.timeout_message:
            payload["content"] = self.timeout_message
        try:
            await interaction.edit_original_response(**payload)
        except Exception:
            # Ephemeral responses expire and messages get deleted; a view we can no
            # longer edit is exactly the case this is cleaning up after, so there is
            # nothing to report.
            pass


#: The two bulk sentinels a paginated multi-select carries above its real options.
SELECT_PAGE = "toggle_page"
SELECT_ALL = "toggle_all"


def bulk_select_options(page_items, all_items, is_selected, *,
                        page_description="Toggle selection for all profiles on this page.",
                        all_description="Toggle selection for all profiles in this source."):
    """The two "Select Page" / "Select All" sentinels, labelled for the current state.

    `is_selected` is called per item. Each label reads as the *undoing* half once its
    scope is fully selected, which is what makes one click reversible by the same click.
    """
    page_selected = bool(page_items) and all(is_selected(i) for i in page_items)
    all_selected = bool(all_items) and all(is_selected(i) for i in all_items)
    return [
        discord.SelectOption(
            label="Unselect Page" if page_selected else "Select Page",
            value=SELECT_PAGE, description=page_description, emoji="\U0001F4C4"),
        discord.SelectOption(
            label="Unselect All" if all_selected else "Select All",
            value=SELECT_ALL, description=all_description, emoji="\U0001F4DA"),
    ]


def resolve_bulk_select(values, page_values, all_values, selected):
    """Apply one multi-select submission to `selected`, returning the new set.

    Four copies of this rule had grown across the package -- the bulk profile picker,
    the share manager and both of the session editor's pickers -- and each had to
    rediscover the part that is not obvious: **a submission only restates the page it
    was made on**. Discord sends the values of the visible options and nothing else, so
    a selection made on page 1 is absent from page 2's payload and must survive it. A
    copy that forgets drops every off-page choice on the next click.

    The sentinels short-circuit and never combine with names: picking "Select All"
    alongside three entries is one gesture with one meaning.
    """
    values, page_values = set(values), set(page_values)
    selected = set(selected)

    if SELECT_PAGE in values:
        return selected - page_values if page_values <= selected else selected | page_values
    if SELECT_ALL in values:
        everything = set(all_values)
        return selected - everything if everything <= selected else selected | everything

    # Only this page's membership is being restated; other pages are not in the payload.
    return (selected - page_values) | (values - {"none"})


class ReportErrorMixin:
    """Tells the user when a control raised, instead of the interaction just dying.

    discord.py's default `on_error` logs and returns, so a failed button looks exactly
    like a button that does nothing. Two views carried a byte-identical copy of this;
    the message is deliberately generic because it is reached from every control.
    """

    error_message = "An unexpected error occurred with this view."

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: ui.Item):
        print(f"Error in {type(self).__name__}: {error}")
        traceback.print_exc()
        try:
            if interaction.response.is_done():
                await interaction.followup.send(self.error_message, ephemeral=True)
            else:
                await interaction.response.send_message(self.error_message, ephemeral=True)
        except Exception:
            # The interaction token can be dead by the time we get here; there is
            # nowhere left to report to, and raising out of an error handler is worse.
            pass


class TabbedView(TimeoutCleanupMixin, ui.View):
    """Base for a screen whose tabs are sibling views: /hub, /settings, /mod.

    Each of the three carried its own copy of this constructor and a nav method per
    tab whose body -- defer, construct the sibling, repaint -- differed only in the
    class it named. Subclasses declare `TABS` instead:

        TABS = (("Home", "home", lambda v: HubHomeView(v.cog, v.original_interaction)),)

    The factory is handed the view it is navigating away from, so a screen that
    ferries state between its tabs (the moderated user id, say) reads it off there.
    """

    #: ((label, tab_key, factory), ...) in the order the nav bar renders them.
    TABS: tuple = ()

    #: The method that re-renders this screen's controls. Named two ways across the
    #: adopters, and probing for it would silently pick the wrong one on a screen
    #: that grew both.
    REBUILD = "setup_items"

    def __init__(self, cog, interaction: discord.Interaction, current_tab: str):
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.current_tab = current_tab
        self._add_nav_buttons()

    def _add_nav_buttons(self):
        self.attach_nav(self, self.current_tab, source=self)

    @classmethod
    def attach_nav(cls, target_view: ui.View, current_tab: str, *, source):
        """Render this screen's nav bar onto `target_view`.

        `source` is what the tab factories read. Normally the view itself, but
        ProfileManageView passes a stand-in so the /mod nav bar can be grafted onto a
        screen that is not one of the mod tabs.
        """
        def nav(factory):
            async def callback(i: discord.Interaction):
                await i.response.defer()
                await factory(source).update_display()
            return callback

        build_tab_nav_bar(target_view, current_tab,
                          [(label, key, nav(factory)) for label, key, factory in cls.TABS])

    def _add_page_controls(self, num_pages: int, row: int, *, repaint: bool = True):
        """Attach prev/next page buttons.

        `repaint=True` edits the message in place off the click; `repaint=False`
        defers and repaints through `update_display`.
        """
        async def p_cb(i: discord.Interaction):
            await self._turn_page(i, -1, repaint)

        async def n_cb(i: discord.Interaction):
            await self._turn_page(i, 1, repaint)

        build_pagination_controls(self, self.current_page, num_pages, row, p_cb, n_cb)

    async def _turn_page(self, i: discord.Interaction, delta: int, repaint: bool = False):
        self.current_page += delta
        getattr(self, self.REBUILD)()
        if repaint:
            await i.response.edit_message(embed=self._get_embed(), view=self)
        else:
            await i.response.defer()
            await self.update_display()

    async def prev_page(self, i: discord.Interaction):
        await self._turn_page(i, -1)

    async def next_page(self, i: discord.Interaction):
        await self._turn_page(i, 1)

    # HubPublicLibraryView wires its buttons to the *_cb spelling.
    prev_page_cb = prev_page
    next_page_cb = next_page


class PageJumpModal(ui.Modal):
    """Jump-to-page prompt for any paginated view.

    Replaces three near-identical copies (session audit, data manager, public library)
    that differed only in where they read the page count from, whether their page
    counter was 0- or 1-based, and how they repainted afterwards -- all three now
    arrive as arguments.

    on_jump is awaited as on_jump(interaction, page), with page already converted to
    the caller's indexing convention.
    """

    def __init__(self, max_pages: int, on_jump: Callable[..., Awaitable[None]], *,
                 title: str = "Jump to Page", label: str = "Page Number",
                 zero_indexed: bool = False):
        super().__init__(title=title)
        self.max_pages = max(1, int(max_pages or 1))
        self.on_jump = on_jump
        self.zero_indexed = zero_indexed
        self.page_input = ui.TextInput(
            label=label,
            placeholder=f"Enter a number between 1 and {self.max_pages}",
            required=True,
            min_length=1,
            max_length=5,
        )
        self.add_item(self.page_input)

    async def on_submit(self, interaction: discord.Interaction):
        # Parsing is validated separately from running on_jump on purpose. The copies
        # this replaces wrapped both in one try/except ValueError, so a ValueError
        # raised anywhere downstream in the repaint was reported to the user as
        # "please enter a valid number".
        raw = (self.page_input.value or "").strip()
        try:
            page = int(raw)
        except ValueError:
            page = None

        if page is None or page < 1 or page > self.max_pages:
            await interaction.response.send_message(
                f"❌ Please enter a valid number between 1 and {self.max_pages}.",
                ephemeral=True,
            )
            return

        await self.on_jump(interaction, page - 1 if self.zero_indexed else page)


def build_pagination_controls(view: ui.View, current_page: int, num_pages: int, row: int, prev_cb, next_cb, page_cb=None):
    if num_pages <= 1: return
    prev_btn = ui.Button(label="◀", style=discord.ButtonStyle.secondary, disabled=(current_page == 0), row=row)
    page_lbl = ui.Button(label=f"{current_page + 1}/{num_pages}", style=discord.ButtonStyle.grey, disabled=(page_cb is None), row=row)
    next_btn = ui.Button(label="▶", style=discord.ButtonStyle.secondary, disabled=(current_page >= num_pages - 1), row=row)
    prev_btn.callback = prev_cb
    next_btn.callback = next_cb
    if page_cb: page_lbl.callback = page_cb
    view.add_item(prev_btn)
    view.add_item(page_lbl)
    view.add_item(next_btn)

class ConfigModal(ui.Modal):
    def __init__(self, cog, profile_name, is_borrowed, title, fields, parser, callback=None, target_user_id=None):
        super().__init__(title=title[:45])
        self.cog = cog
        self.profile_name = profile_name
        self.is_borrowed = is_borrowed
        self.parser = parser
        self.callback = callback
        self.target_user_id = target_user_id
        for f in fields:
            self.add_item(ui.TextInput(**f))

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            raw_values = {c.custom_id: c.value for c in self.children}
            updates = self.parser(raw_values)
            config_updates = updates.get("config", {})
            prompt_updates = updates.get("prompts", {})
        except ValueError as e:
            await interaction.followup.send(f"❌ **Invalid Input:** {e}", ephemeral=True)
            return
        except Exception:
            await interaction.followup.send("❌ Error parsing input.", ephemeral=True)
            return

        uid = self.target_user_id or interaction.user.id

        if self.profile_name == "BULK_APPLY":
            if self.callback: await self.callback(interaction, updates)
            return

        if config_updates:
            target = self.cog.profile_manager._get_profile_config(uid, self.profile_name, self.is_borrowed)
            if target:
                target.update(config_updates)
                self.cog.profile_manager._save_profile_config(uid, self.profile_name, target, self.is_borrowed)
                invalidate_model_cache(self.cog, uid, self.profile_name)

        if prompt_updates and not self.is_borrowed:
            prompts = self.cog.profile_manager._get_profile_prompts(uid, self.profile_name)
            if prompts:
                prompts.update(prompt_updates)
                self.cog.profile_manager._save_profile_prompts(uid, self.profile_name, prompts)

        await interaction.followup.send(f"✅ Settings updated for '{self.profile_name}'.", ephemeral=True)
        if self.callback: await self.callback(interaction)

class ActionTextInputModal(ui.Modal):
    def __init__(self, title: str, label: str, placeholder: str, on_submit_callback, default: Optional[str] = None, required: bool = True):
        super().__init__(title=title)
        self.on_submit_callback = on_submit_callback
        self.input = ui.TextInput(
            label=label,
            placeholder=placeholder,
            default=default,
            style=discord.TextStyle.paragraph,
            required=required
        )
        self.add_item(self.input)

    async def on_submit(self, interaction: discord.Interaction):
        await self.on_submit_callback(interaction, self.input.value)

class DropdownContentView(ui.View):
    def __init__(self, content_dict: dict, title: str, link_button_label: Optional[str] = None,
                 link_button_url: Optional[str] = None, start_category: Optional[str] = None,
                 start_page: Optional[str] = None):
        super().__init__(timeout=600)
        self.content_dict = content_dict
        self.view_title = title
        self.link_button_label = link_button_label
        self.link_button_url = link_button_url

        # `start_category` / `start_page` open the browser on one page rather than at
        # the beginning, which is what lets a "Read more" button elsewhere hand the
        # reader the paragraph it was talking about instead of the table of contents.
        # Validated rather than trusted: a caller naming a page that has since been
        # renamed opens the guide at the top, which is a worse answer but not a crash.
        self.selected_category = (start_category if start_category in self.content_dict
                                  else list(self.content_dict.keys())[0])
        pages = self.content_dict[self.selected_category]
        self.selected_page = start_page if start_page in pages else list(pages.keys())[0]
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        cat_options = [discord.SelectOption(label=cat[:100], value=cat[:100], default=(cat == self.selected_category)) for cat in self.content_dict.keys()]
        async def cat_callback(interaction: discord.Interaction):
            self.selected_category = interaction.data['values'][0]
            self.selected_page = list(self.content_dict[self.selected_category].keys())[0]
            self._build_view()
            await interaction.response.edit_message(embed=self.get_embed(), view=self)
        add_select(self, cat_options, cat_callback, placeholder="Select Category...", row=0)
        
        page_options = [discord.SelectOption(label=page[:100], value=page[:100], default=(page == self.selected_page)) for page in self.content_dict[self.selected_category].keys()]
        async def page_callback(interaction: discord.Interaction):
            self.selected_page = interaction.data['values'][0]
            self._build_view()
            await interaction.response.edit_message(embed=self.get_embed(), view=self)
        add_select(self, page_options, page_callback, placeholder="Select Page...", row=1)

        if self.link_button_label and self.link_button_url:
            btn = ui.Button(label=self.link_button_label, url=self.link_button_url, row=2)
            self.add_item(btn)

    def get_embed(self) -> discord.Embed:
        content = self.content_dict[self.selected_category][self.selected_page]
        embed = discord.Embed(title=self.selected_page, description=content, color=discord.Color.blurple())
        embed.set_author(name=self.view_title)
        return embed

class InviteView(ui.View):
    def __init__(self, invite_url: str):
        super().__init__(timeout=None)
        btn = ui.Button(label="Add MimicAI to Server", url=invite_url, style=discord.ButtonStyle.link)
        self.add_item(btn)

class BaseBulkProfileView(ui.View):
    def __init__(self, cog, user_id, include_borrowed=True, timeout=300, exclude_public=False):
        super().__init__(timeout=timeout)
        self.cog = cog
        self.user_id = user_id
        self.include_borrowed = include_borrowed
        self.exclude_public = exclude_public
        self.selected_profiles = set()
        self.current_page = 0
        self.view_source = 'personal'
        self._load_profile_lists()

    def _load_profile_lists(self):
        """Reads the profile index into the two source lists and their cached options.

        Split out of `__init__` so a view whose scope can change after construction --
        the bulk wizard, where Personal/Borrowed/Both is chosen as a first step -- can
        re-scope in place instead of rebuilding the view object and losing the message
        it is attached to.
        """
        index = self.cog.profile_manager._get_user_index(self.user_id)
        self.personal_profiles = sorted(list(index.get("personal", [])))

        # Profiles held in the Public Library are withheld from settings that would
        # invalidate their listing -- the 18+ declaration being the one that does. A
        # bulk sweep over "all my profiles" silently flipped published profiles to
        # 18+, which the publish gate rejects, so they had to be found and reverted
        # one at a time. Resolved in a single pass over the public index rather than
        # a _is_profile_public call per profile.
        self.excluded_public = []
        if self.exclude_public:
            published = {d["profile_name"]
                         for d in self.cog.profile_manager._iter_public_entries(self.user_id)}
            self.excluded_public = [n for n in self.personal_profiles if n in published]
            self.personal_profiles = [n for n in self.personal_profiles if n not in published]

        self.borrowed_profiles = sorted(list(index.get("borrowed", []))) if self.include_borrowed else []

        # Pre-compute options once to save massive UI overhead
        self._cached_personal_opts = [discord.SelectOption(label=p, value=p) for p in self.personal_profiles]
        self._cached_borrowed_opts = [discord.SelectOption(label=p, value=p) for p in self.borrowed_profiles]

    async def _edit(self, interaction: discord.Interaction):
        """Re-renders this view onto the message it already occupies.

        A hook rather than a literal `edit_message(content=...)` repeated at each of
        the five call sites below, because the bulk wizard renders as an embed: without
        it, paging or selecting a profile would replace that embed with a plain-text
        selection summary.
        """
        await interaction.response.edit_message(
            content=self._get_selection_feedback_message(), view=self)

    def _get_active_list(self):
        return self.personal_profiles if self.view_source == 'personal' else self.borrowed_profiles

    def _build_profile_select_ui(self, row=1):
        active_list = self._get_active_list()
        cached_opts = self._cached_personal_opts if self.view_source == 'personal' else self._cached_borrowed_opts
        
        per_page = 23
        num_pages = max(1, (len(active_list) - 1) // per_page + 1)
        if self.current_page >= num_pages: self.current_page = max(0, num_pages - 1)
        start = self.current_page * per_page
        
        page_items = active_list[start : start + per_page]
        page_opts = cached_opts[start : start + per_page]
        
        options = []
        if page_items:
            options = bulk_select_options(page_items, active_list,
                                          lambda p: p in self.selected_profiles)
            # Default state set directly on the cached objects
            for opt in page_opts:
                opt.default = (opt.value in self.selected_profiles)
                options.append(opt)
        else:
            options = [discord.SelectOption(label="No profiles found", value="none", default=False)]

        placeholder = f"Select {self.view_source} profiles..."
        add_select(self, options, self.profile_select_callback, placeholder=placeholder,
                   min_values=0, max_values=len(options) if page_items else 1, row=row,
                   disabled=not page_items, custom_id="profile_select")

        btn_row = row + 1
        
        # No standalone page counter, and none baked into the Source label either:
        # build_pagination_controls already puts one between the arrows, so both of
        # those were a second copy of the same number sitting next to the first. It
        # also frees the slot that used to take this row to its five-button cap.
        if self.include_borrowed:
            style = discord.ButtonStyle.blurple if self.view_source == 'personal' else discord.ButtonStyle.green
            add_button(self, f"Source: {self.view_source.title()}", self.toggle_source_callback,
                       style=style, row=btn_row, custom_id="toggle_source")

        async def p_cb(i: discord.Interaction):
            self.current_page -= 1
            self._build_view()
            await self._edit(i)

        async def n_cb(i: discord.Interaction):
            self.current_page += 1
            self._build_view()
            await self._edit(i)

        build_pagination_controls(self, self.current_page, num_pages, btn_row, p_cb, n_cb)

        # Only when there is something to clear. "Select All" across both sources is
        # one click, and undoing it by paging through every page to unselect is not;
        # the dropdown sentinels only ever toggle the source currently in view.
        if self.selected_profiles:
            add_button(self, "Clear", self.clear_selection_callback,
                       style=discord.ButtonStyle.secondary, row=btn_row,
                       custom_id="clear_selection")

    async def clear_selection_callback(self, interaction: discord.Interaction):
        """Drops the whole selection, both sources, every page."""
        self.selected_profiles.clear()
        self._build_view()
        await self._edit(interaction)

    async def toggle_source_callback(self, interaction: discord.Interaction):
        self.view_source = 'borrowed' if self.view_source == 'personal' else 'personal'
        self.current_page = 0
        self._build_view()
        await self._edit(interaction)

    async def profile_select_callback(self, interaction: discord.Interaction):
        per_page = 23
        active_list = self._get_active_list()
        start = self.current_page * per_page
        self.selected_profiles = resolve_bulk_select(
            interaction.data.get('values', []),
            active_list[start : start + per_page], active_list, self.selected_profiles)

        self._build_view()
        await self._edit(interaction)

    def _get_selection_feedback_message(self) -> str:
        count = len(self.selected_profiles)
        if count == 0: return "Select profiles to apply the action to."
        profile_list = sorted(list(self.selected_profiles))
        message = f"**Selected Profiles ({count}):**\n" + "\n".join(f"- `{name}`" for name in profile_list[:10])
        if count > 10: message += f"\n...and {count - 10} more."
        return message
    
    def _build_view(self):
        raise NotImplementedError
