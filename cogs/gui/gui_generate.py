"""The draft `/profile generate` shows before anything is saved.

Generation used to write the profile the moment the model answered, so the only way to
see what came back was to open the dashboard of a profile that already existed, and the
only way to try again was to delete it first. The draft now sits on this view: Save writes
it, Regenerate rolls a fresh one from the same concept, Refine rewrites it to a request,
and Discard -- or simply walking away -- leaves nothing behind.

`/import` of a character card shows its draft here too (`card`): Regenerate sorts the card
again, and Save also writes its lorebook and example exchanges (as fixed examples).
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
                 appearance: Optional[Dict[str, Any]] = None, card: Optional[Dict[str, Any]] = None,
                 image: Optional[Dict[str, str]] = None):
        # Ten minutes, inside the fifteen an interaction token lasts, so the expiry
        # notice can still be written onto the message.
        super().__init__(timeout=600)
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.profile_name = profile_name
        self.concept = concept
        #: The display name and avatar the user gave, laid over every draft -- and for a
        #: card, its lorebook, fixed examples and 18+ tag, which no redraft changes.
        self.appearance = appearance or {}
        #: The card being imported (`character_card.read_card`), or None for a concept.
        self.card = card
        self.template_key = "CARD_IMPORT" if card else "PROFILE_GENERATOR"
        #: The image drafted from (`generate_draft`'s part), shown again to every redraft.
        self.image = image
        if card:
            self.timeout_message = "This draft expired, and nothing was saved. Run `/import` again."
        self.draft = {**draft, **self.appearance}
        self.model_used = model_used
        self.busy = False
        self.saved = False
        self.published = False
        self._build_view()

    def _build_view(self):
        self.clear_items()
        if self.saved:
            add_button(self, "Open Dashboard", self.open_dashboard,
                       style=discord.ButtonStyle.blurple)
            # A concept's draft only: a card is someone else's character, and one tagged
            # 18+ could not be published anyway.
            if not self.card and not self.published:
                add_button(self, "Publish to Library", self.publish, style=discord.ButtonStyle.green,
                           emoji="🌐", disabled=self.busy)
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
            lines.append(f"✅ **Saved as `{self.profile_name}`.**"
                         + (" 🌐 Listed in the Public Library." if self.published else ""))
        else:
            lines.append(f"A draft, to be saved as `{self.profile_name}`. "
                         f"Nothing is written until you press **Save**.")
        if draft.get("library_intro"):
            lines.append(f"*{draft['library_intro']}*")

        title = draft.get("display_name") or self.profile_name
        embed = discord.Embed(title=_clip(title, 256),
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

        if self.card:
            extras = []
            if draft.get("lorebook"):
                extras.append(f"**Lorebook:** {len(draft['lorebook'])} entries")
            if draft.get("fixed_examples"):
                extras.append(f"**Fixed examples:** {len(draft['fixed_examples'])}")
            if draft.get("adult"):
                extras.append("**Adult 18+:** the card is tagged NSFW, so it is declared 18+ on Save")
            if extras:
                embed.add_field(name="Also from the card", value="\n".join(extras), inline=False)
            sorted_by = f"Sorted by {self.model_used}" if self.model_used else "Filed as written"
            embed.set_footer(text=_clip(f"Imported from {self.card['name']} · {sorted_by}", 300))
        else:
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
                previous=self.draft if request else None, request=request,
                template_key=self.template_key, image=self.image)
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

    async def publish(self, interaction: discord.Interaction):
        """Rates the new profile -- it is Unrated -- and lists it if the verdict allows."""
        if await self._refuse_while_busy(interaction):
            return
        self.busy = True
        self._build_view()
        await interaction.response.edit_message(
            embed=self.build_embed("\U0001F310 Rating and publishing…"), view=self)
        self.published, reason = await self.cog.profile_manager.publish_profile(
            self.user_id, self.profile_name)
        self.busy = False
        self._build_view()
        await interaction.edit_original_response(embed=self.build_embed(), view=self)
        if not self.published:
            await interaction.followup.send(f"Not published. {reason or ''}".strip(), ephemeral=True)

    async def discard(self, interaction: discord.Interaction):
        self.stop()
        await interaction.response.edit_message(
            content="Draft discarded. Nothing was saved.", embed=None, view=None)

    async def open_dashboard(self, interaction: discord.Interaction):
        self.stop()
        await interaction.response.defer(thinking=True, ephemeral=True)
        await self.cog._open_profile_manage(interaction, self.profile_name, repaint=True)
