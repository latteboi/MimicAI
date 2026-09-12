"""The data policy screen: whether a server's messages may reach AI providers that train on them.

The bot owner's alone, opened from /privacy; cogs/utils/data_policy says why a server's
administrators cannot open a route. Everyone else reads the same fields on /privacy.
"""
import asyncio
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import discord
from discord import ui

from .base_components import BlockedGuard, add_button
from ..utils.data_policy import (
    TRAINING_PROVIDERS, may_set_data_policy, opt_in_record, set_training_opt_in, training_opt_in,
)

if TYPE_CHECKING:
    from ..MimicCog import MimicCog


#: provider -> (name, what "allowed" opens up, what stays true while it is closed).
_PROVIDER_COPY = {
    "gemini": (
        "Google Gemini free tier",
        "Google may use what a free-tier (unpaid) Gemini key is sent to improve its "
        "products and train its models, and human reviewers may read it.",
        "Free-tier keys are not used for this server's messages. Billing-enabled keys "
        "are unaffected.",
    ),
    "openrouter": (
        "OpenRouter hosts that may train",
        "Some OpenRouter hosts may keep prompts and train on them.",
        "This server's OpenRouter requests only go to hosts that don't. A model that no "
        "such host serves stops working here.",
    ),
}


def add_data_policy_fields(embed: discord.Embed, server_index) -> None:
    """One field per provider: whether this server has opened it, and what that means.

    The bot owner's screen and every user's /privacy both render through here, so the two
    cannot describe the same server differently.
    """
    for provider in TRAINING_PROVIDERS:
        name, opens, closed = _PROVIDER_COPY[provider]
        record = opt_in_record(server_index, provider)
        if record:
            status = f"✅ **Allowed** by <@{record['by']}> <t:{record['at']}:R>"
            detail = opens
        else:
            status = "🔒 **Not allowed**"
            detail = closed
        embed.add_field(name=name, value=f"{status}\n-# {detail}", inline=False)


class DataPolicyView(BlockedGuard, ui.View):
    """One embed, one toggle per provider; turning a provider on asks for confirmation.

    Opening a route is the consequential direction, so it takes a second press on a
    screen that says exactly what it means. Closing one is immediate.
    """

    def __init__(self, cog: 'MimicCog', guild: discord.Guild, user_id: int,
                 on_back: Optional[Callable[[], Tuple[discord.Embed, ui.View]]] = None):
        super().__init__(timeout=300)
        self.cog = cog
        self.guild = guild
        self.user_id = user_id
        #: Rebuilds the /privacy screen this was opened from, for the Back button.
        self.on_back = on_back
        #: The provider awaiting confirmation, or None on the main screen.
        self.confirming: Optional[str] = None
        self._build()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not await super().interaction_check(interaction):
            return False
        return interaction.user.id == self.user_id and may_set_data_policy(interaction.user.id)

    def _index(self):
        return self.cog.server_manager._get_server_index(str(self.guild.id))

    def embed(self) -> discord.Embed:
        index = self._index()
        if self.confirming:
            name, opens, _closed = _PROVIDER_COPY[self.confirming]
            e = discord.Embed(
                title=f"Allow {name}?",
                description=(
                    f"{opens}\n\n"
                    "Allowing it means this server's messages -- conversations, Global Chat "
                    "cards opened here, memories, web research, image prompts and speech -- "
                    "may be used that way.\n\n"
                    "Discord's Developer Policy does not allow message content to be used to "
                    "train AI models without Discord's permission. Only allow this if Discord "
                    "has given this application that permission."
                ),
                color=discord.Color.orange())
            e.set_footer(text=self.guild.name)
            return e

        e = discord.Embed(
            title="Data Policy",
            description=(
                "Whether this server's messages may be sent to AI providers that can train "
                "on them. Both start closed, and only the bot's owner can open either. "
                "Everyone can read this in `/privacy`."
            ),
            color=discord.Color.blurple())
        add_data_policy_fields(e, index)
        e.add_field(name="OpenRouter models offered to everyone else",
                    value=self._openrouter_offer_text(), inline=False)
        e.set_footer(text=self.guild.name)
        return e

    def _openrouter_offer_text(self) -> str:
        """How many OpenRouter models the pickers offer people other than the bot owner, and why."""
        shown, total, checked_at = self.cog.api_service.catalogue.training_status()
        if checked_at:
            how = ("The rest have no host known to serve them without training on prompts, "
                   "going by what your OpenRouter account's privacy settings allowed "
                   f"<t:{int(checked_at)}:R>.")
        else:
            how = ("Only zero-retention models, until the bot can tell which others avoid "
                   "training. It reads OpenRouter's model list through your Personal "
                   "OpenRouter key, filtered by that account's privacy settings -- turn off "
                   "providers that may train on inputs there.")
        return f"**{shown}** of {total}. {how}"[:1024]

    def _build(self):
        self.clear_items()
        if self.confirming:
            provider = self.confirming

            async def confirm(i: discord.Interaction):
                await self._set(i, provider, True)

            async def cancel(i: discord.Interaction):
                self.confirming = None
                self._build()
                await i.response.edit_message(embed=self.embed(), view=self)

            add_button(self, "Allow", confirm, style=discord.ButtonStyle.danger)
            add_button(self, "Cancel", cancel)
            return

        index = self._index()
        for provider in TRAINING_PROVIDERS:
            name = _PROVIDER_COPY[provider][0]
            if training_opt_in(index, provider):
                async def revoke(i: discord.Interaction, provider=provider):
                    await self._set(i, provider, False)
                add_button(self, f"Stop allowing {name}", revoke, style=discord.ButtonStyle.success)
            else:
                async def ask(i: discord.Interaction, provider=provider):
                    self.confirming = provider
                    self._build()
                    await i.response.edit_message(embed=self.embed(), view=self)
                add_button(self, f"Allow {name}…", ask, style=discord.ButtonStyle.secondary)

        on_back = self.on_back
        if on_back:
            async def back(i: discord.Interaction):
                embed, view = on_back()
                await i.response.edit_message(embed=embed, view=view)
            add_button(self, "Back", back, row=1)

    async def _set(self, interaction: discord.Interaction, provider: str, allowed: bool):
        index = self._index()
        set_training_opt_in(index, provider, allowed, interaction.user.id)
        await asyncio.to_thread(self.cog.server_manager._save_server_index, str(self.guild.id), index)
        self.confirming = None
        self._build()
        await interaction.response.edit_message(embed=self.embed(), view=self)
