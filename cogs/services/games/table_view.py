"""Rendering the shared table: the status embed and the hand-panel signature.

Split out of `game_service.py` because none of it touches the cog -- it turns a
`Game` into an embed and nothing else, so it can be exercised without a bot. The
service keeps the parts that need a channel: posting, editing and resinking.

`private_view` never reaches here. Everything below is built from public state, which
is what stops a hand leaking into the shared message.
"""

import time
from typing import Optional

import discord

from . import dice, eights, neuro
from .eights import RuleSet
from .table import Game, NAMES

#: Colour swatches for the status embed. Wild shows the declared colour, so the black
#: square only ever appears for the top card itself, never for `active_colour`.
_SWATCH = {
    eights.RED: "\U0001F7E5", eights.YELLOW: "\U0001F7E8",
    eights.GREEN: "\U0001F7E9", eights.BLUE: "\U0001F7E6", eights.WILD: "⬛",
}
_EMBED_COLOUR = {
    eights.RED: 0xC42F35, eights.YELLOW: 0xB77C05,
    eights.GREEN: 0x217D47, eights.BLUE: 0x1F6DBE, eights.WILD: 0x2B2F36,
}
_LABEL = {
    eights.SKIP: "Skip", eights.REVERSE: "Reverse", eights.DRAW_TWO: "Draw Two",
    eights.WILD_PLAIN: "Wild", eights.DRAW_FOUR: "Wild Draw Four",
}


def card_label(card) -> str:
    colour, value = card
    return f"{_SWATCH.get(colour, '')} {_LABEL.get(value, value)}".strip()


def panel_signature(game: Game, seat_id: str) -> tuple:
    """Everything a hand panel renders, as one comparable value.

    Deliberately covers the *board* as well as the hand. A seat's cards are only
    one of the things its panel shows -- the top card, the active colour, the
    pending draw and whose turn it is are all on the embed, and every control's
    enabled state is derived from them. Watching the hand alone is what let a panel
    sit frozen on a six-move-old board.
    """
    state = game.state
    seat_state = state.seat(seat_id)
    if game.kind == "dice":
        return (tuple(seat_state.hand), state.bid, state.current.seat_id
                if state.phase == "playing" else None, state.phase, state.round_no)
    hand = tuple(sorted(seat_state.hand)) if seat_state else ()
    return (hand, state.top, state.active_colour, state.pending_draw,
            state.current.seat_id if state.phase == "playing" else None,
            state.phase, bool(game.last_call_armed.get(seat_id)))


def house_rule_summary(rules: RuleSet) -> str:
    """The non-default rules in play, for the table footer.

    Only the deviations: a table running the defaults says nothing, and everyone
    can see at a glance what they actually agreed to when it is not.
    """
    if isinstance(rules, dice.RuleSet):
        return _dice_rule_summary(rules)
    default = RuleSet()
    bits = []
    if not rules.stack_draw_two: bits.append("no stacking")
    if rules.stack_draw_four: bits.append("D4 stacks")
    if rules.draw_to_match: bits.append("draw to match")
    if rules.strict_draw_four: bits.append("strict D4")
    if not rules.play_after_draw: bits.append("no play after draw")
    if rules.turn_seconds != default.turn_seconds:
        bits.append(f"{int(rules.turn_seconds)}s turns")
    return " · ".join(bits)


def _dice_rule_summary(rules: "dice.RuleSet") -> str:
    default = dice.RuleSet()
    bits = []
    if rules.dice_per_seat != default.dice_per_seat:
        bits.append(f"{rules.dice_per_seat} dice each")
    if not rules.ones_wild: bits.append("ones not wild")
    if rules.turn_seconds != default.turn_seconds:
        bits.append(f"{int(rules.turn_seconds)}s turns")
    return " · ".join(bits)


def _footer(game: Game, text: str, final: bool) -> str:
    house = house_rule_summary(game.state.rules)
    if house:
        text += f" · {house}"
    if game.deadline and not final:
        text += f" · {max(0, int(game.deadline - time.monotonic()))}s to play"
    return text


def _dice_cups(held) -> str:
    return " ".join(dice.GLYPHS[d - 1] for d in held)


def build_dice_embed(game: Game, final: bool = False,
                     note: Optional[str] = None) -> discord.Embed:
    """The shared Mimic Dice table. Built from `public_view` only -- never a cup this
    round. The last cups lifted are shown in full: the table has already seen them."""
    view = dice.public_view(game.state)

    def name(seat_id) -> str:
        seat = game.seat(seat_id) if seat_id else None
        return seat.display if seat else "someone"

    if final:
        title = (f"{NAMES['dice']} — {name(view['winner'])} wins" if view["winner"]
                 else f"{NAMES['dice']} — game over")
    else:
        title = f"{NAMES['dice']} — {len(game.seats)} at the table"
    embed = discord.Embed(title=title, colour=0x6B4FA0)

    board = []
    if view["bid"]:
        quantity, face = view["bid"]
        board.append(f"**Bid** {dice.GLYPHS[face - 1]} {dice.bid_text(quantity, face)}"
                     f" — {name(view['bidder'])}")
    elif view["current_seat"]:
        board.append(f"**Bid** none yet — {name(view['current_seat'])} opens")
    board.append(f"**Dice in play** {view['dice_in_play']}"
                 + (" · ones are wild" if view["ones_wild"] else ""))
    embed.add_field(name="Board", value="\n".join(board), inline=False)

    reveal = view["reveal"]
    if reveal:
        verdict = ("short" if reveal["loser"] == reveal["bidder"] else "good")
        head = (f"{name(reveal['caller'])} called Liar on {name(reveal['bidder'])}'s "
                f"{dice.bid_text(*reveal['bid'])} — **{reveal['found']}** there, the "
                f"bid was {verdict}. {name(reveal['loser'])} lost a die.")
        cups = "\n".join(f"{name(sid)} {_dice_cups(held)}"
                         for sid, held in reveal["dice"].items())
        embed.add_field(name="Last cups lifted", value=f"{head}\n{cups}"[:1024],
                        inline=False)

    lines = []
    for seat_view in view["seats"]:
        held = seat_view["dice"]
        if not held:
            lines.append(f" ~~{name(seat_view['seat_id'])}~~ · out")
            continue
        here = "▸ " if seat_view["seat_id"] == view["current_seat"] else " "
        mood = neuro.describe(game.neuro.get(seat_view["seat_id"], neuro.BASELINE))
        lines.append(f"{here}**{name(seat_view['seat_id'])}** · 🎲 {held}  *{mood}*")
    embed.add_field(name="Seats", value="\n".join(lines) or "—", inline=False)

    if note:
        embed.add_field(name="​", value=note, inline=False)
    embed.set_footer(text=_footer(
        game, f"round {view['round_no']} · turn {view['turn_no']}", final))
    return embed


def build_embed(game: Game, final: bool = False,
                note: Optional[str] = None) -> discord.Embed:
    """The shared table view. Built from `public_view` only -- never a hand."""
    if game.kind == "dice":
        return build_dice_embed(game, final=final, note=note)
    view = eights.public_view(game.state)
    colour = _EMBED_COLOUR.get(view["active_colour"], 0x2B2F36)

    if final:
        winner_id = view["winner"]
        winner = game.seat(winner_id) if winner_id else None
        title = f"Mimic Eights — {winner.display} wins" if winner else "Mimic Eights — game over"
    else:
        title = f"Mimic Eights — {len(game.seats)} at the table"

    embed = discord.Embed(title=title, colour=colour)

    top = view["top"]
    arrow = "⟳ clockwise" if view["direction"] > 0 else "⟲ anticlockwise"
    board = [
        f"**Top card** {card_label(top)}",
        f"**Colour** {_SWATCH.get(view['active_colour'], '')} "
        f"{view['active_colour'].capitalize()}",
        f"**Order** {arrow}",
    ]
    if view["pending_draw"]:
        board.append(f"**Pending** ⚠️ {view['pending_draw']} to draw")
    embed.add_field(name="Board", value="\n".join(board), inline=False)

    lines = []
    for seat_view in view["seats"]:
        seat = game.seat(seat_view["seat_id"])
        name = seat.display if seat else seat_view["seat_id"]
        here = "▸ " if seat_view["seat_id"] == view["current_seat"] else " "
        count = seat_view["cards"]
        tail = "  — **Last Card!**" if seat_view["called_last"] else ""
        mood = neuro.describe(game.neuro.get(seat_view["seat_id"], neuro.BASELINE))
        lines.append(f"{here}**{name}** · {count} card{'' if count == 1 else 's'}"
                     f"{tail}  *{mood}*")
    embed.add_field(name="Seats", value="\n".join(lines) or "—", inline=False)

    if note:
        embed.add_field(name="​", value=note, inline=False)

    embed.set_footer(text=_footer(
        game, f"turn {view['turn_no']} · {view['draw_pile_size']} in the pile", final))
    return embed
