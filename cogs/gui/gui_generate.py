"""The draft `/profile generate` shows before anything is saved.

Generation used to write the profile the moment the model answered, so the only way to
see what came back was to open the dashboard of a profile that already existed, and the
only way to try again was to delete it first. The draft now sits on this view: Save writes
it, Regenerate rolls a fresh one from the same concept, Refine rewrites it to a request,
and Discard -- or simply walking away -- leaves nothing behind.
"""
import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

import discord
from discord import ui

from ..services.profile_generation import (ProfileGenerationError, generate_draft,
                                           save_draft)
from ..utils.helpers import suppress_link_previews
from .base_components import BlockedGuard, TimeoutCleanupMixin, add_button

if TYPE_CHECKING:
    from ..MimicCog import MimicCog

#: Per persona field. Six of these, the intro and the footer stay inside Discord's 6000
#: characters for a whole embed; the draft itself is saved in full.
FIELD_PREVIEW_CHARS = 700

_PREVIEW_FIELDS = (("Backstory", "backstory"), ("Personality", "personality_traits"),
                   ("Likes", "likes"), ("Dislikes", "dislikes"), ("Appearance", "appearance"))


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


class RefineDraftModal(ui.Modal, title="Refine the Draft"):
    request_input = ui.TextInput(
        label="What should change?", style=discord.TextStyle.paragraph, max_length=500,
        placeholder="e.g. Make her older, less cynical, and give her a pet crow.")

    def __init__(self, parent_view: "GeneratedProfileView"):
        super().__init__()
        self.parent_view = parent_view

    async def on_submit(self, interaction: discord.Interaction):
        await self.parent_view.redraft(interaction, (self.request_input.value or "").strip())


class GeneratedProfileView(BlockedGuard, TimeoutCleanupMixin, ui.View):
    timeout_message = "This draft expired, and nothing was saved. Run `/profile generate` again."

    def __init__(self, cog: 'MimicCog', interaction: discord.Interaction, profile_name: str,
                 concept: str, draft: Dict[str, Any], model_used: str,
                 appearance: Optional[Dict[str, str]] = None):
        # Ten minutes, inside the fifteen an interaction token lasts, so the expiry
        # notice can still be written onto the message.
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.profile_name = profile_name
        self.concept = concept
        #: The display name and avatar the user gave, laid over every draft.
        self.appearance = appearance or {}
        self.draft = {**draft, **self.appearance}
        self.model_used = model_used
        self.busy = False
        self.saved = False
        self._build_view()

    def _build_view(self):
        self.clear_items()
        if self.saved:
            add_button(self, "Open Dashboard", self.open_dashboard,
                       style=discord.ButtonStyle.blurple)
            return
        add_button(self, "Save", self.save, style=discord.ButtonStyle.green, emoji="✅",
                   disabled=self.busy)
        add_button(self, "Regenerate", self.regenerate, style=discord.ButtonStyle.blurple,
                   emoji="\U0001F504", disabled=self.busy)
        add_button(self, "Refine", self.refine, emoji="✏️", disabled=self.busy)
        add_button(self, "Discard", self.discard, disabled=self.busy)

    def build_embed(self, status: Optional[str] = None) -> discord.Embed:
        draft = self.draft
        lines = []
        if status:
            lines.append(f"**{status}**")
        elif self.saved:
            lines.append(f"✅ **Saved as `{self.profile_name}`.**")
        else:
            lines.append(f"A draft, to be saved as `{self.profile_name}`. "
                         f"Nothing is written until you press **Save**.")
        if draft.get("library_intro"):
            lines.append(f"*{draft['library_intro']}*")

        title = draft.get("display_name") or self.profile_name
        emoji = draft.get("placeholder_emoji")
        embed = discord.Embed(title=_clip(f"{emoji} {title}" if emoji else title, 256),
                              description="\n\n".join(lines), color=discord.Color.blurple())
        if draft.get("avatar_url"):
            embed.set_thumbnail(url=draft["avatar_url"])

        persona = draft.get("persona") or {}
        for label, key in _PREVIEW_FIELDS:
            if persona.get(key):
                embed.add_field(name=label, value=_clip(persona[key], FIELD_PREVIEW_CHARS),
                                inline=False)
        if draft.get("ai_instructions"):
            embed.add_field(name="Instructions",
                            value=_clip(draft["ai_instructions"], FIELD_PREVIEW_CHARS), inline=False)

        embed.set_footer(text=_clip(f"Concept: {self.concept} · Written by {self.model_used}", 300))
        return embed

    async def _refuse_while_busy(self, interaction: discord.Interaction) -> bool:
        if not self.busy:
            return False
        await interaction.response.send_message("Still working on the last one.", ephemeral=True)
        return True

    async def redraft(self, interaction: discord.Interaction, request: Optional[str] = None):
        """Regenerate from the concept, or with `request`, rewrite the draft on screen."""
        if await self._refuse_while_busy(interaction):
            return
        self.busy = True
        self._build_view()
        await interaction.response.edit_message(
            embed=self.build_embed("✏️ Refining…" if request
                                   else "\U0001F504 Regenerating…"),
            view=self)

        failure = None
        try:
            draft, self.model_used = await generate_draft(
                self.cog, self.user_id, self.concept,
                previous=self.draft if request else None, request=request)
            self.draft = {**draft, **self.appearance}
        except ProfileGenerationError as e:
            failure = suppress_link_previews(str(e))
        finally:
            self.busy = False
            self._build_view()

        # Through this press's token, not the command's: a draft reworked a few times
        # can outlive the fifteen minutes the original interaction's token lasts.
        await interaction.edit_original_response(embed=self.build_embed(), view=self)
        if failure:
            await interaction.followup.send(f"❌ **Generation Failed:** {failure}", ephemeral=True)

    async def regenerate(self, interaction: discord.Interaction):
        await self.redraft(interaction)

    async def refine(self, interaction: discord.Interaction):
        if await self._refuse_while_busy(interaction):
            return
        await interaction.response.send_modal(RefineDraftModal(self))

    async def save(self, interaction: discord.Interaction):
        if await self._refuse_while_busy(interaction):
            return
        self.busy = True
        self._build_view()
        await interaction.response.edit_message(view=self)

        failure = await asyncio.to_thread(
            save_draft, self.cog, self.user_id, self.profile_name, self.draft)
        self.busy = False
        self.saved = failure is None
        if self.saved:
            # What is left is a shortcut to a profile that exists; expiry just removes it.
            self.timeout_message = None
        self._build_view()
        await interaction.edit_original_response(embed=self.build_embed(), view=self)
        if failure:
            await interaction.followup.send(f"❌ {failure}", ephemeral=True)

    async def discard(self, interaction: discord.Interaction):
        self.stop()
        await interaction.response.edit_message(
            content="Draft discarded. Nothing was saved.", embed=None, view=None)

    async def open_dashboard(self, interaction: discord.Interaction):
        self.stop()
        await interaction.response.defer(thinking=True, ephemeral=True)
        await self.cog._open_profile_manage(interaction, self.profile_name, repaint=True)
