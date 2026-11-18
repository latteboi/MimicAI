"""`/help`: the manual, MimicGuide, and support tickets, from one dashboard.

A router, as `/start` is: Docs opens the documentation browser, Ask MimicGuide the same
lookup `/help ask` ran, and MimicGuide here the same toggle -- each lives on the cog, and
this only opens it. Tickets are the one thing that lives here, over `TicketStore`.

Everything happens on this message. Nobody is sent a DM: a reply waits for its reader
to open `/help`, where the count says it is there. The bot owner sees the Inbox -- every
user's open tickets -- where everyone else sees their own.

One view, four pages (home, the user's tickets, one ticket, the owner's inbox), each a
repaint of the same ephemeral message through `original_interaction`.
"""
import asyncio
import time
from typing import Dict, List, Optional, Tuple

import discord
from discord import ui

from ..managers.ticket_store import OPERATOR, USER
from ..utils.constants import (LIMIT_OPEN_TICKETS, LIMIT_TICKET_MESSAGES, TICKET_CATEGORIES,
                               TICKET_MESSAGE_MAX_CHARS, TICKET_OPEN_COOLDOWN_SECONDS,
                               TICKET_RETENTION_DAYS, defaultConfig)
from .base_components import (BlockedGuard, DropdownContentView, TimeoutCleanupMixin, add_button,
                              add_select, build_pagination_controls, cast_refusal, refuse_unregistered,
                              text_embed)

#: Inbox rows per page: one dropdown's worth.
INBOX_PAGE = 25
#: Room for a ticket's conversation in one embed description, below its 4096.
CONVERSATION_ROOM = 3800
UNREAD = "🔔"


#: The bot owner's system profile that answers from the docs.
GUIDE_PROFILE = "mimicguide"


def is_operator(user_id: int) -> bool:
    return user_id == int(defaultConfig.DISCORD_OWNER_ID)


def guide_seated(cog, channel_id: int) -> bool:
    session = cog.multi_profile_channels.get(channel_id)
    owner_id = int(defaultConfig.DISCORD_OWNER_ID)
    return bool(session) and any(p.get("owner_id") == owner_id and p.get("profile_name") == GUIDE_PROFILE
                                 for p in session["profiles"])


def guide_refusal(cog, interaction: discord.Interaction) -> Optional[str]:
    """Why this member cannot add MimicGuide to this channel, or take it out, or None.

    The cast editor's own gate: it changes the cast. There are no sessions in DMs, and
    joining a server with no key seats a guide that can never answer -- leaving needs none.
    """
    if interaction.guild is None:
        return "Sessions run in servers, so MimicGuide joins one there, not in a DM."
    refusal = cast_refusal(cog, interaction)
    if refusal:
        return refusal
    if not guide_seated(cog, interaction.channel_id):
        return cog._session_key_block(interaction.guild, interaction.user.id)
    return None


def _when(ts: float) -> str:
    return f"<t:{int(ts)}:R>"


def _row_line(ticket_id: str, row: Dict, operator: bool) -> str:
    unread = row.get("unread_owner" if operator else "unread_user")
    parts = [f"**#{ticket_id}**", TICKET_CATEGORIES.get(row["category"], row["category"])]
    if operator:
        parts.append(f"<@{row['user']}>")
    parts.append("Open" if row["status"] == "open" else "Closed")
    parts.append(_when(row["updated"]))
    return (f"{UNREAD} " if unread else "") + " · ".join(parts)


def _user_name(cog, user_id: int) -> str:
    user = cog.bot.get_user(user_id)
    return user.display_name if user else f"user {user_id}"


class HelpView(BlockedGuard, TimeoutCleanupMixin, ui.View):
    def __init__(self, cog, interaction: discord.Interaction):
        super().__init__(timeout=600)  # under the 15 minutes an interaction token lasts
        self.cog = cog
        self.original_interaction = interaction
        self.user_id = interaction.user.id
        self.operator = is_operator(self.user_id)
        self.page = "home"
        self.ticket_id: Optional[str] = None
        self.body: Optional[Dict] = None
        self.show_closed = False
        self.inbox_page = 0
        self._build()

    # --- drawing ----------------------------------------------------------------------

    def render(self) -> Dict:
        return {"embed": getattr(self, f"_{self.page}_embed")(), "view": self}

    async def update_display(self):
        self._build()
        await self.original_interaction.edit_original_response(**self.render())

    def _build(self):
        self.clear_items()
        getattr(self, f"_{self.page}_items")()

    def _ticket_block(self) -> Optional[str]:
        """Why this user cannot open a ticket now, or None."""
        if not self.cog.user_settings.is_registered(self.user_id):
            return "Tickets need an account: create yours in `/start` first."
        if len(self.cog.tickets.rows(self.user_id, "open")) >= LIMIT_OPEN_TICKETS:
            return f"You have {LIMIT_OPEN_TICKETS} tickets open. Close one, or add to it."
        last = self.cog.ticket_times.get(self.user_id)
        if last and time.time() - last < TICKET_OPEN_COOLDOWN_SECONDS:
            left = int(TICKET_OPEN_COOLDOWN_SECONDS - (time.time() - last)) // 60 + 1
            return f"You opened one a moment ago. A new ticket is possible in {left} minute(s)."
        return None

    # home

    def _home_embed(self) -> discord.Embed:
        lines = [
            "📖 **Docs**: the manual, by topic.",
            "💬 **Ask MimicGuide**: one question, answered from the docs. Only you see the answer.",
            "🧭 **MimicGuide here**: adds MimicGuide to this channel's session for follow-up "
            "questions, or takes it out again.",
        ]
        if self.operator:
            open_rows = self.cog.tickets.rows(status="open")
            lines.append(f"📥 **Inbox**: {len(open_rows)} open, "
                         f"{self.cog.tickets.unread()} with something new.")
        else:
            lines.append("🎫 **Tickets**: write to the people who run this bot. Their replies "
                         "show up here, in `/help`.")
        notes = []
        refusal = guide_refusal(self.cog, self.original_interaction)
        if refusal:
            notes.append(f"-# MimicGuide here: {refusal}")
        if not self.operator and not self.cog.user_settings.is_registered(self.user_id):
            notes.append("-# Tickets: create your account in `/start` first.")
        return text_embed("\n".join(lines + ([""] + notes if notes else [])), title="Help")

    def _home_items(self):
        add_button(self, "Docs", self._docs, emoji="📖")
        add_button(self, "Ask MimicGuide", self._ask, emoji="💬")
        seated = guide_seated(self.cog, self.original_interaction.channel_id)
        add_button(self, "Remove MimicGuide" if seated else "MimicGuide here", self._toggle,
                   emoji="🧭", disabled=guide_refusal(self.cog, self.original_interaction) is not None)
        if self.operator:
            unread = self.cog.tickets.unread()
            add_button(self, f"Inbox ({unread} new)" if unread else "Inbox", self._show_inbox,
                       emoji="📥", style=discord.ButtonStyle.primary)
        else:
            unread = self.cog.tickets.unread(self.user_id)
            add_button(self, f"Tickets ({unread} new)" if unread else "Tickets", self._show_tickets,
                       emoji="🎫", style=discord.ButtonStyle.primary,
                       disabled=not self.cog.user_settings.is_registered(self.user_id))

    # the user's tickets

    def _tickets_embed(self) -> discord.Embed:
        rows = self.cog.tickets.rows(self.user_id)
        text = ("\n".join(_row_line(tid, row, False) for tid, row in rows) if rows else
                "None yet. **New Ticket** writes to the people who run this bot; their replies "
                "show up here, in `/help`.")
        block = self._ticket_block()
        if block:
            text += f"\n\n-# New Ticket: {block}"
        embed = text_embed(text, title="Your tickets")
        embed.set_footer(text=f"A closed ticket is deleted {TICKET_RETENTION_DAYS} days after it closes.")
        return embed

    def _tickets_items(self):
        rows = self.cog.tickets.rows(self.user_id)
        if rows:
            add_select(self, [discord.SelectOption(
                label=f"#{tid} · {TICKET_CATEGORIES.get(row['category'], row['category'])}"[:100],
                description=("Open" if row["status"] == "open" else "Closed")
                + (" · something new" if row.get("unread_user") else ""),
                value=tid) for tid, row in rows[:25]],
                self._pick, placeholder="Open a ticket...", row=0)
        add_button(self, "New Ticket", self._new_ticket, emoji="✏️", row=1,
                   style=discord.ButtonStyle.success, disabled=self._ticket_block() is not None)
        add_button(self, "Back", self._home, row=1)

    # the operator's inbox

    def _inbox_rows(self) -> List[Tuple[str, Dict]]:
        rows = self.cog.tickets.rows(status="closed" if self.show_closed else "open")
        return sorted(rows, key=lambda item: (not item[1].get("unread_owner"), -item[1]["updated"]))

    def _inbox_embed(self) -> discord.Embed:
        rows = self._inbox_rows()
        pages = max(1, -(-len(rows) // INBOX_PAGE))
        self.inbox_page = min(self.inbox_page, pages - 1)
        shown = rows[self.inbox_page * INBOX_PAGE:(self.inbox_page + 1) * INBOX_PAGE]
        text = ("\n".join(_row_line(tid, row, True) for tid, row in shown) if shown
                else ("No closed tickets." if self.show_closed else "No open tickets."))
        return text_embed(text, title="Inbox · " + ("closed" if self.show_closed else "open"))

    def _inbox_items(self):
        rows = self._inbox_rows()
        pages = max(1, -(-len(rows) // INBOX_PAGE))
        shown = rows[self.inbox_page * INBOX_PAGE:(self.inbox_page + 1) * INBOX_PAGE]
        if shown:
            add_select(self, [discord.SelectOption(
                label=f"#{tid} · {_user_name(self.cog, row['user'])}"[:100],
                description=TICKET_CATEGORIES.get(row["category"], row["category"])
                + (" · something new" if row.get("unread_owner") else ""),
                value=tid) for tid, row in shown], self._pick, placeholder="Open a ticket...", row=0)

        async def turn(i: discord.Interaction, delta: int):
            await i.response.defer()
            self.inbox_page = max(0, min(pages - 1, self.inbox_page + delta))
            await self.update_display()
        build_pagination_controls(self, self.inbox_page, pages, 1, turn)
        add_button(self, "Show open" if self.show_closed else "Show closed", self._flip_inbox, row=2)
        add_button(self, "Back", self._home, row=2)

    # one ticket

    def _ticket_embed(self) -> discord.Embed:
        row = self.cog.tickets.row(self.ticket_id)
        if row is None or self.body is None:
            return text_embed("This ticket no longer exists.", title=f"#{self.ticket_id}")
        title = f"#{self.ticket_id} · {TICKET_CATEGORIES.get(row['category'], row['category'])}"
        embed = text_embed(self._conversation(row), title=title)
        embed.add_field(name="Status", value="Open" if row["status"] == "open" else "Closed")
        if self.operator:
            guild = self.cog.bot.get_guild(row["guild"]) if row.get("guild") else None
            embed.add_field(name="From", value=f"<@{row['user']}>"
                            + (f" in {guild.name}" if guild else " (not from a server)" if not row.get("guild") else ""))
        trace = (self.body or {}).get("trace")
        if trace:
            embed.add_field(name="Message", value=trace["url"]
                            + ("" if trace.get("turn") else " (no trace kept)"), inline=False)
        return embed

    def _conversation(self, row: Dict) -> str:
        """The messages, newest last, as many of the latest as fit."""
        blocks = []
        for message in self.body["messages"]:
            mine = (message["from"] == OPERATOR) == self.operator
            who = "You" if mine else ("Support" if message["from"] == OPERATOR else f"<@{row['user']}>")
            blocks.append(f"**{who}** · {_when(message['at'])}\n{message['text']}")
        shown, room = [], CONVERSATION_ROOM
        for block in reversed(blocks):
            if len(block) + 2 > room and shown:
                break
            shown.insert(0, block[:room])
            room -= len(block) + 2
        hidden = len(blocks) - len(shown)
        return (f"-# {hidden} earlier message(s) not shown\n\n" if hidden else "") + "\n\n".join(shown)

    def _ticket_items(self):
        row = self.cog.tickets.row(self.ticket_id)
        is_open = bool(row) and row["status"] == "open"
        full = bool(row) and row["messages"] >= LIMIT_TICKET_MESSAGES
        add_button(self, "Reply", self._reply, emoji="↩️", style=discord.ButtonStyle.primary,
                   disabled=not is_open or full)
        add_button(self, "Close Ticket", self._close, emoji="✅", disabled=not is_open)
        if self.operator and ((self.body or {}).get("trace") or {}).get("turn"):
            add_button(self, "View Trace", self._trace, emoji="🔍")
        add_button(self, "Back", self._back)

    # --- actions ----------------------------------------------------------------------

    async def _docs(self, i: discord.Interaction):
        from ..utils.content import HELP_CATEGORIES
        view = DropdownContentView(HELP_CATEGORIES, "MimicAI Help & Documentation")
        await i.response.send_message(embed=view.get_embed(), view=view, ephemeral=True)

    async def _ask(self, i: discord.Interaction):
        await i.response.send_modal(AskModal(self.cog))

    async def _toggle(self, i: discord.Interaction):
        await self.cog._toggle_guide(i)
        await self.update_display()

    async def _go(self, i: discord.Interaction, page: str):
        await i.response.defer()
        self.page = page
        await self.update_display()

    async def _home(self, i):
        await self._go(i, "home")

    async def _show_tickets(self, i):
        if await refuse_unregistered(self.cog, i):
            return
        await self._go(i, "tickets")

    async def _show_inbox(self, i):
        await self._go(i, "inbox")

    async def _back(self, i):
        await self._go(i, "inbox" if self.operator else "tickets")

    async def _flip_inbox(self, i):
        self.show_closed, self.inbox_page = not self.show_closed, 0
        await self._go(i, "inbox")

    async def _pick(self, i: discord.Interaction):
        await i.response.defer()
        await self.open_ticket(i.data["values"][0])

    async def open_ticket(self, ticket_id: str):
        """Shows `ticket_id` on this message, read off the loop, and marks it read."""
        row = self.cog.tickets.row(ticket_id)
        # Only the operator reads someone else's. A stale dropdown is the one way here.
        if row is None or (not self.operator and row["user"] != self.user_id):
            self.page = "inbox" if self.operator else "tickets"
            await self.update_display()
            return
        self.ticket_id, self.page = ticket_id, "ticket"
        self.body = await asyncio.to_thread(self.cog.tickets.read, ticket_id)
        await asyncio.to_thread(self.cog.tickets.mark_read, ticket_id, OPERATOR if self.operator else USER)
        await self.update_display()

    async def _new_ticket(self, i: discord.Interaction):
        block = self._ticket_block()
        if block:
            await i.response.send_message(embed=text_embed(block), ephemeral=True)
            return
        await i.response.send_modal(NewTicketModal(self))

    async def _reply(self, i: discord.Interaction):
        await i.response.send_modal(ReplyModal(self))

    async def _close(self, i: discord.Interaction):
        await i.response.defer()
        await asyncio.to_thread(self.cog.tickets.close, self.ticket_id, OPERATOR if self.operator else USER)
        await self.update_display()

    async def _trace(self, i: discord.Interaction):
        from .gui_sessions import GenerationTraceView
        trace = (self.body or {}).get("trace") or {}
        view = GenerationTraceView(self.cog, i, trace["turn"], trace.get("prompt_url"))
        await i.response.send_message(embed=view.build_embed(), view=view, ephemeral=True)


class AskModal(ui.Modal):
    """Ask MimicGuide: one question, answered by `MimicCog._ask_guide`."""

    def __init__(self, cog):
        super().__init__(title="Ask MimicGuide")
        self.cog = cog
        self.question = ui.TextInput(label="Your question", style=discord.TextStyle.paragraph,
                                     max_length=1000, placeholder="e.g. How do I give a character a voice?")
        self.add_item(self.question)

    async def on_submit(self, interaction: discord.Interaction):
        await self.cog._ask_guide(interaction, self.question.value.strip())


class NewTicketModal(ui.Modal):
    def __init__(self, view: HelpView):
        super().__init__(title="New Ticket")
        self.view = view
        self.category = ui.RadioGroup(options=[
            discord.RadioGroupOption(label=label, value=value, default=value == "bug")
            for value, label in TICKET_CATEGORIES.items()])
        self.add_item(ui.Label(text="What is it about?", component=self.category))
        self.message = ui.TextInput(label="Message", style=discord.TextStyle.paragraph,
                                    max_length=TICKET_MESSAGE_MAX_CHARS,
                                    placeholder="What happened, and what you expected instead.")
        self.add_item(self.message)
        self.link = ui.TextInput(label="Message link (optional)", required=False, max_length=200,
                                 placeholder="Right-click a character's reply → Copy Message Link")
        self.add_item(self.link)

    async def on_submit(self, interaction: discord.Interaction):
        view, cog = self.view, self.view.cog
        # Checked again here, not only when the button was drawn: minutes may have passed.
        if await refuse_unregistered(cog, interaction):
            return
        block = view._ticket_block()
        if block or self.category.value not in TICKET_CATEGORIES:
            await interaction.response.send_message(
                embed=text_embed(block or "Choose what the ticket is about."), ephemeral=True)
            return
        await interaction.response.defer()
        trace, note = None, None
        if (self.link.value or "").strip():
            trace, note = await cog._ticket_trace(interaction.user.id, self.link.value.strip())
        ticket_id, refusal = await asyncio.to_thread(
            cog.tickets.open, interaction.user.id, interaction.guild_id, self.category.value,
            self.message.value.strip(), trace)
        if refusal:
            await interaction.followup.send(embed=text_embed(refusal), ephemeral=True)
            return
        cog.ticket_times[interaction.user.id] = time.time()
        await view.open_ticket(ticket_id)
        if note:
            await interaction.followup.send(embed=text_embed(note), ephemeral=True)


class ReplyModal(ui.Modal):
    def __init__(self, view: HelpView):
        super().__init__(title=f"Reply to #{view.ticket_id}")
        self.view = view
        self.message = ui.TextInput(label="Message", style=discord.TextStyle.paragraph,
                                    max_length=TICKET_MESSAGE_MAX_CHARS)
        self.add_item(self.message)

    async def on_submit(self, interaction: discord.Interaction):
        view = self.view
        refusal = await asyncio.to_thread(view.cog.tickets.reply, view.ticket_id,
                                          OPERATOR if view.operator else USER, self.message.value.strip())
        if refusal:
            await interaction.response.send_message(embed=text_embed(refusal), ephemeral=True)
            return
        await interaction.response.defer()
        await view.open_ticket(view.ticket_id)
