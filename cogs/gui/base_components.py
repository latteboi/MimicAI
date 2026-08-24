import discord
from discord import ui
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

if TYPE_CHECKING:
    # This only runs during "hinting" and prevents the circular crash
    from ..MimicCog import MimicCog

def build_tab_nav_bar(target_view: ui.View, current_tab: str, tabs, row: int = 4):
    """Attaches a row of tab-navigation buttons to target_view.

    Each entry in tabs is (label, tab_key, async_callback). The button for the
    currently active tab is styled primary and disabled; all others are secondary.
    """
    for label, tab_key, callback in tabs:
        btn = ui.Button(
            label=label,
            style=discord.ButtonStyle.primary if current_tab == tab_key else discord.ButtonStyle.secondary,
            row=row,
            disabled=(current_tab == tab_key),
        )
        btn.callback = callback
        target_view.add_item(btn)

def build_confirm_view(button_label: str, on_confirm) -> ui.View:
    """Builds a single-button (danger-styled) confirmation view wired to on_confirm, timeout=60."""
    view = ui.View(timeout=60)
    btn = ui.Button(label=button_label, style=discord.ButtonStyle.danger)
    btn.callback = on_confirm
    view.add_item(btn)
    return view

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
                keys_to_clear = [k for k in self.cog.channel_models.keys() if isinstance(k, tuple) and len(k) == 3 and k[1] == uid and k[2] == self.profile_name]
                for k in keys_to_clear:
                    self.cog.channel_models.pop(k, None)
                    self.cog.channel_model_last_profile_key.pop(k, None)

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
    def __init__(self, content_dict: dict, title: str, link_button_label: Optional[str] = None, link_button_url: Optional[str] = None):
        super().__init__(timeout=600)
        self.content_dict = content_dict
        self.view_title = title
        self.link_button_label = link_button_label
        self.link_button_url = link_button_url
        
        self.selected_category = list(self.content_dict.keys())[0]
        self.selected_page = list(self.content_dict[self.selected_category].keys())[0]
        self._build_view()

    def _build_view(self):
        self.clear_items()
        
        cat_options = [discord.SelectOption(label=cat[:100], value=cat[:100], default=(cat == self.selected_category)) for cat in self.content_dict.keys()]
        cat_select = ui.Select(placeholder="Select Category...", options=cat_options, row=0)
        
        async def cat_callback(interaction: discord.Interaction):
            self.selected_category = interaction.data['values'][0]
            self.selected_page = list(self.content_dict[self.selected_category].keys())[0]
            self._build_view()
            await interaction.response.edit_message(embed=self.get_embed(), view=self)
            
        cat_select.callback = cat_callback
        self.add_item(cat_select)
        
        page_options = [discord.SelectOption(label=page[:100], value=page[:100], default=(page == self.selected_page)) for page in self.content_dict[self.selected_category].keys()]
        page_select = ui.Select(placeholder="Select Page...", options=page_options, row=1)
        
        async def page_callback(interaction: discord.Interaction):
            self.selected_page = interaction.data['values'][0]
            self._build_view()
            await interaction.response.edit_message(embed=self.get_embed(), view=self)
            
        page_select.callback = page_callback
        self.add_item(page_select)

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
            page_set = set(page_items)
            page_selected = page_set.issubset(self.selected_profiles)
            page_label = "Unselect Page" if page_selected else "Select Page"
            options.append(discord.SelectOption(label=page_label, value="toggle_page", description="Toggle selection for all profiles on this page.", emoji="📄"))
            
            all_set = set(active_list)
            all_selected = all_set.issubset(self.selected_profiles)
            all_label = "Unselect All" if all_selected else "Select All"
            options.append(discord.SelectOption(label=all_label, value="toggle_all", description="Toggle selection for all profiles in this source.", emoji="📚"))

            # Update default state directly on the cached objects
            for opt in page_opts:
                opt.default = (opt.value in self.selected_profiles)
                options.append(opt)
        else:
            options = [discord.SelectOption(label="No profiles found", value="none", default=False)]

        placeholder = f"Select {self.view_source} profiles..."
        select = ui.Select(placeholder=placeholder, min_values=0, max_values=len(options) if page_items else 1, options=options, custom_id="profile_select", row=row, disabled=(not page_items))
        select.callback = self.profile_select_callback
        self.add_item(select)

        btn_row = row + 1
        
        # No standalone page counter, and none baked into the Source label either:
        # build_pagination_controls already puts one between the arrows, so both of
        # those were a second copy of the same number sitting next to the first. It
        # also frees the slot that used to take this row to its five-button cap.
        if self.include_borrowed:
            style = discord.ButtonStyle.blurple if self.view_source == 'personal' else discord.ButtonStyle.green
            mode_btn = ui.Button(label=f"Source: {self.view_source.title()}", style=style,
                                 custom_id="toggle_source", row=btn_row)
            mode_btn.callback = self.toggle_source_callback
            self.add_item(mode_btn)

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
            clear_btn = ui.Button(label="Clear", style=discord.ButtonStyle.secondary,
                                  custom_id="clear_selection", row=btn_row)
            clear_btn.callback = self.clear_selection_callback
            self.add_item(clear_btn)

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
        vals = interaction.data.get('values', [])
        if "none" in vals: vals = []
        
        per_page = 23
        active_list = self._get_active_list()
        start = self.current_page * per_page
        page_items = set(active_list[start : start + per_page])
        
        if "toggle_page" in vals:
            if page_items.issubset(self.selected_profiles): self.selected_profiles.difference_update(page_items)
            else: self.selected_profiles.update(page_items)
        elif "toggle_all" in vals:
            all_set = set(active_list)
            if all_set.issubset(self.selected_profiles): self.selected_profiles.difference_update(all_set)
            else: self.selected_profiles.update(all_set)
        else:
            self.selected_profiles.difference_update(page_items)
            self.selected_profiles.update(vals)
            
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
