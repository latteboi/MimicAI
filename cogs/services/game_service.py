"""Drives a table game inside a Discord channel.

The rules, the policy and the neuro deltas are pure and live under `services/games/`;
everything here is the part that has to know about channels, tasks and embeds. The
split is the point -- nothing in this file decides what a legal move is, and nothing
under `games/` knows what a channel is.

Three things are worth knowing before reading further.

**The roster is a snapshot.** `/play` opens a `Lobby` -- a guest list with no rules
engine behind it -- and the cast is copied at the moment the host presses Start, not
when the command ran. From then on the roster owns who is at the table. Removing a
profile from the session mid-game does not unseat it; adding one does not seat it. That
turns every session mutation from a correctness problem into a casting question.

**A human seat is not special.** Seats are dealt from `Lobby.humans` in the order people
sat down, and the run loop dispatches on `seat.kind` rather than on a single known
player, so a table of six people and no profiles runs the same code as a table of one.

**The game is not in the session.** It lives in `cog.active_games`, a bounded cache
keyed by channel, and it dies on restart. Nothing about a live hand is persisted, so
none of the `structural=True` flush discipline applies to it -- there is no tail to
write. Two things cross into the session, and only two: `context_block` is read by
`_construct_system_instructions` as standing `<game_context>`, and a notable beat goes
on the channel's `task_queue` as a `game_beat` trigger -- once per loud moment for the
character it happened to, and once at the end for the whole table. No mechanical event
is ever logged, so the session reads as a conversation that happens to be about Mimic Eights
rather than as a transcript of bookkeeping.

**Nothing here generates dialogue.** `_request_reaction` queues a trigger and returns;
the multi-profile worker owns the round that follows, which is what gives a reaction
the same instructions, training, LTM, critic, placeholder and typing indicator as any
other reply -- and what stops it talking over a round the channel was already running.

**A hand never leaves this module.** `eights.private_view` is called here and nowhere
else, and what it returns goes to `policy.choose` and to the seat's own ephemeral
message. It is never put in a prompt, never logged, and never rendered into the shared
status embed or `<game_context>`, both of which are built from `public_view` alone.
"""

import asyncio
import random
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import discord

from ..utils.constants import (
    DEFAULT_GAME_CONTEXT, DEFAULT_GAME_FINALE_USER, DEFAULT_GAME_OPENING_LIVE,
    DEFAULT_GAME_OPENING_OVER, DEFAULT_GAME_REACTION_USER, GAME_BEAT_STALE_SECONDS,
    GAME_EMBED_MIN_INTERVAL_SECONDS,
    GAME_EPILOGUE_CACHE_MAX_SIZE, GAME_EPILOGUE_SECONDS, GAME_FINALE_MAX_WORDS,
    GAME_FINALE_STALE_SECONDS, GAME_MAX_CONCURRENT,
    GAME_REACTION_MAX_WORDS, GAME_TABLE_REPOST_MIN_SECONDS,
    GAME_TURN_PACE_SECONDS, GAME_LAST_CALL_WORDS,
)
from .games import ledger as ledger_mod, neuro, policy, eights
#: Re-exported: the dataclasses moved to games/table.py, but `from .game_service
#: import Game, Lobby` is what gui_games and the channel listener already say.
from .games.table import Game, Lobby, Panel, Seat
from .games import narration, table_view
from .games._shared import Event, Ev, IllegalMove
from .games.neuro import PRESETS, Temperament
from .games.eights import RuleSet

#: How often a human turn wakes to redraw. Short enough that the footer's countdown
#: reads as a clock and a buried table comes back quickly, long enough that a
#: three-quarter-minute turn costs a handful of edits rather than one a second.
TURN_TICK_SECONDS = 5.0


card_label = table_view.card_label



class GameService:
    """Owns the lifecycle of channel games.

    Holds a back-reference to the parent cog for shared state and Discord dispatch,
    per the transitional dependency-injection pattern the other services use.
    """

    def __init__(self, cog):
        self.cog = cog
        #: Closing tables, kept warm for GAME_EPILOGUE_SECONDS after `_finish` pops the
        #: game. See `context_block` for why the block has to outlive the game. Written
        #: once per game ending, which is where the bound is trimmed -- a plain
        #: OrderedDict rather than the cog's LRUCache because `context_block` is read
        #: from a worker thread, and LRUCache reorders on read.
        self._epilogues: "OrderedDict[int, tuple]" = OrderedDict()

    # ------------------------------------------------------------------ roster

    def _temperament_for(self, config: Dict[str, Any], profile_name: str) -> Temperament:
        """The profile's authored temperament, or a stable stand-in.

        There is no temperament field on a profile yet -- that arrives with the
        settings surface in a later phase. Until then the name is hashed to a preset so
        a table is visibly varied rather than five identical steady players, and so the
        same profile always plays the same way. Deterministic, and replaced the moment
        `game_temperament` is authored.
        """
        named = (config or {}).get("game_temperament")
        if named in PRESETS:
            return PRESETS[named]
        pool = ("steady", "steady", "reckless", "anxious", "vindictive", "meek")
        return PRESETS[pool[sum(map(ord, profile_name)) % len(pool)]]

    def _display_for(self, owner_id: int, profile_name: str) -> str:
        """What the table calls this profile.

        The same name the channel already sees on its posts -- the appearance's
        `custom_display_name`, resolved through the borrow indirection like every other
        surface. The internal profile name is only a fallback: a seat labelled with it
        is the one place in the channel where a character is named something its own
        author renamed away from.

        `_get_user_appearance` memoises into `cog.user_appearances`, so this is a dict
        hit for every profile the channel has already spoken as, and one config read for
        one that has not. Same cost class as `_load_profile_state` beside it, on the same
        deal path -- and now also on every lobby repaint, via `roster_preview`, which is
        what the memoisation is carrying.
        """
        try:
            appearance = self.cog.profile_manager._get_user_appearance(
                owner_id, profile_name) or {}
            return appearance.get("custom_display_name") or profile_name
        except Exception:
            return profile_name

    def _build_roster(self, session: Dict[str, Any], limit: int) -> List[Seat]:
        """Snapshot the session's cast into seats.

        Deduplicates by `(owner_id, profile_name)`: the same profile can legitimately
        appear twice in a cast through a borrow, but it cannot sit twice at one table
        without both seats sharing a neuro state and a ledger line.
        """
        seats: List[Seat] = []
        if limit <= 0:
            # Reachable now that the caller subtracts a human count that a lobby lets
            # grow to fill the table on its own. The loop below only tests its limit
            # *after* appending, so without this a full table still seats one profile.
            return seats
        seen = set()
        for participant in session.get("profiles", []):
            owner_id = participant.get("owner_id")
            profile_name = participant.get("profile_name")
            if not owner_id or not profile_name:
                continue
            key = (owner_id, profile_name)
            if key in seen:
                continue
            seen.add(key)
            seats.append(Seat(
                seat_id=f"{owner_id}:{profile_name}",
                display=self._display_for(owner_id, profile_name),
                kind="profile",
                owner_id=owner_id,
                profile_name=profile_name,
                method=participant.get("method", "webhook"),
                bot_id=participant.get("bot_id"),
            ))
            if len(seats) >= limit:
                break
        return seats

    def _load_profile_state(self, seat: Seat):
        """Starting neuro state and temperament for one seated profile.

        The starting state is whatever the profile is actually carrying, so a character
        who has had a rough afternoon brings that to the table rather than resetting to
        a clean 50/20/50/20.
        """
        config: Dict[str, Any] = {}
        try:
            index = self.cog.profile_manager._get_user_index(seat.owner_id)
            borrowed = seat.profile_name in (index.get("borrowed") or [])
            config = self.cog.profile_manager._get_profile_config(
                seat.owner_id, seat.profile_name, borrowed) or {}
        except Exception:
            config = {}

        state = dict(neuro.BASELINE)
        stored = config.get("neuro_state")
        if isinstance(stored, dict):
            for axis in neuro.AXES:
                if isinstance(stored.get(axis), (int, float)):
                    state[axis] = max(0, min(100, int(stored[axis])))
        return state, self._temperament_for(config, seat.profile_name or "")

    # ------------------------------------------------------------------ lifecycle

    def lobby_for(self, channel_id: int) -> Optional[Lobby]:
        return self.cog.pending_lobbies.get(channel_id)

    def _table_blocked(self, channel_id: Optional[int]) -> Optional[str]:
        """Why this channel cannot open a table right now, or None."""
        if channel_id is None:
            return "Games need a channel."
        if channel_id in self.cog.active_games:
            return "A game is already running in this channel. `/play stop` ends it."
        if self.cog.pending_lobbies.get(channel_id) is not None:
            return ("A table is already forming in this channel. Sit down at it, or "
                    "`/play stop` to clear it.")
        # A lobby is a claim on one of the concurrent slots. Counting it here is what
        # stops twelve open lobbies dealing themselves into a thirteenth game.
        in_use = len(self.cog.active_games) + len(self.cog.pending_lobbies)
        if in_use >= GAME_MAX_CONCURRENT:
            return (f"This instance is already running {GAME_MAX_CONCURRENT} games. "
                    "Try again when one finishes.")
        return None

    async def open_lobby(
        self,
        interaction: discord.Interaction,
        seats_wanted: Optional[int] = None,
        rules: Optional[RuleSet] = None,
        join: bool = True,
    ) -> Optional[str]:
        """Post a table that has not been dealt yet. Returns an error string, or None.

        Nothing is on a clock. The lobby stands until the host starts it or somebody
        clears it with `/play stop` -- a countdown would only ever fire on the sitting
        that was still deciding, and `pending_lobbies` is bounded, so an abandoned one
        costs a cache slot rather than a task.
        """
        from ..gui.gui_games import LobbyView, build_lobby_embed

        channel_id = interaction.channel_id
        blocked = self._table_blocked(channel_id)
        if blocked:
            return blocked

        session = self.cog.multi_profile_channels.get(channel_id)
        if not session or session.get("type") != "multi":
            return ("There is no active session in this channel. Use `/session config` "
                    "to build a cast first — the table is drawn from it.")

        lobby = Lobby(
            channel_id=channel_id,
            guild_id=interaction.guild_id,
            host_id=interaction.user.id,
            rules=rules if rules is not None else RuleSet(),
            seats_wanted=seats_wanted,
        )
        if join:
            lobby.humans[interaction.user.id] = getattr(
                interaction.user, "display_name", "Player")

        # Registered before the send, so a second `/play eights` racing this one is
        # refused by `_table_blocked` rather than posting a second lobby.
        self.cog.pending_lobbies[channel_id] = lobby
        try:
            lobby.message = await interaction.channel.send(
                embed=build_lobby_embed(self.cog, lobby),
                view=LobbyView(self.cog, channel_id))
        except discord.HTTPException:
            self.cog.pending_lobbies.pop(channel_id, None)
            return "I could not post the table in this channel."
        return None

    def roster_preview(self, lobby: Lobby) -> List[Seat]:
        """The profiles that would fill the seats the humans have not taken.

        Recomputed on every render rather than snapshotted, so the lobby embed shows
        who gets bumped as people sit down instead of promising a cast it will not deal.
        """
        session = self.cog.multi_profile_channels.get(lobby.channel_id)
        if not session or session.get("type") != "multi":
            return []
        return self._build_roster(session, lobby.limit - len(lobby.humans))

    def cancel_lobby(self, channel_id: int) -> Optional[Lobby]:
        """Drop a forming table. Returns it if there was one, for the caller to tidy."""
        return self.cog.pending_lobbies.pop(channel_id, None)

    async def start_from_lobby(self, lobby: Lobby) -> Optional[str]:
        """Deal the table a lobby describes. Returns an error string, or None.

        The seat list is built here rather than carried on the lobby because the profile
        fill depends on how many humans ended up sitting, and that is not settled until
        the host presses Start.
        """
        session = self.cog.multi_profile_channels.get(lobby.channel_id)
        if not session or session.get("type") != "multi":
            return ("The session in this channel has gone. Use `/session config` to "
                    "build a cast, then deal again.")

        seats: List[Seat] = [
            Seat(seat_id=f"user:{user_id}", display=display, kind="human")
            for user_id, display in lobby.humans.items()
        ][:lobby.limit]
        seats.extend(self._build_roster(session, lobby.limit - len(seats)))

        if len(seats) < 2:
            return ("Mimic Eights needs at least two players. Sit down, unlock the table "
                    "so others can, or add profiles with `/session config`.")

        self.cog.pending_lobbies.pop(lobby.channel_id, None)
        return self._deal(lobby.channel_id, lobby.guild_id, lobby.host_id,
                          seats, lobby.rules)

    def _deal(self, channel_id: int, guild_id: Optional[int], host_id: int,
              seats: List[Seat], ruleset: RuleSet) -> Optional[str]:
        """Turn a settled seat list into a running game."""
        states, temperaments = {}, {}
        for seat in seats:
            if seat.kind == "human":
                # People have no persisted chemistry, but they still occupy a seat the
                # neuro table writes to, and the table embed reads a mood for every
                # seat -- so they get a resting state rather than a special case.
                states[seat.seat_id] = neuro.default_state()
                temperaments[seat.seat_id] = PRESETS["steady"]
                continue
            state, temperament = self._load_profile_state(seat)
            states[seat.seat_id] = state
            temperaments[seat.seat_id] = temperament

        seed = random.randrange(1 << 30)
        state = eights.new_game(
            [s.seat_id for s in seats], rules=ruleset, seed=seed,
            kinds={s.seat_id: s.kind for s in seats},
        )

        game = Game(
            kind="eights", state=state, seats=seats, neuro=states,
            ledger=ledger_mod.Ledger([s.seat_id for s in seats]),
            temperaments=temperaments, channel_id=channel_id,
            guild_id=guild_id, started_by=host_id,
            rng=random.Random(seed ^ 0x5EED),
        )
        self.cog.active_games[channel_id] = game

        task = self.cog.bot.loop.create_task(self._run(channel_id))
        game.task = task
        self.cog.background_tasks.add(task)
        task.add_done_callback(self.cog.background_tasks.discard)
        return None

    async def stop_game(self, channel_id: int, reason: str = "stopped") -> bool:
        """End a game early. Safe to call for a channel with no game."""
        game = self.cog.active_games.get(channel_id)
        if not game:
            return False
        game.stopping = True
        task = game.task
        if task and not task.done():
            task.cancel()
            # gather(return_exceptions=True) collects the task's own CancelledError
            # without it propagating here, and without an `except CancelledError` that
            # would also swallow a cancellation aimed at *this* coroutine.
            await asyncio.gather(task, return_exceptions=True)
        await self._finish(channel_id, note=f"Game {reason}.")
        return True

    # ------------------------------------------------------------------ session

    def context_block(self, channel_id: int) -> Optional[str]:
        """The `<game_context>` body for this channel, or None when no game is live.

        Read by `_construct_system_instructions` on every multi-profile generation in
        the channel, so the no-game path -- overwhelmingly the common one -- is a single
        `dict.get`. Note `.get` rather than `[]`: `LRUCache` reorders on `__getitem__`
        and this is called from inside `asyncio.to_thread`, where mutating the cache
        would be a data race. `dict.get` is C-level and does not reach the override.

        Built from `public_view` and the ledger only. There is no filtering step keeping
        hands out of it, because no hand is ever in scope here to filter.

        Called from a worker thread, concurrently with the run loop, so it can read a
        state the loop is midway through mutating. That is deliberate rather than
        overlooked: the worst outcome is a prompt quoting a card count one move out of
        date, and reaching across the thread boundary for a lock to prevent it would
        cost more than the thing it buys.

        Gated on the game existing rather than on it still being *playable*. Those come
        apart at exactly the moment that matters: `apply` flips the phase to "over" the
        instant somebody plays their last card, and the reaction to that -- the biggest
        beat in the game -- is generated afterwards. Gating on `phase` handed the
        winning moment a prompt with no table in it.

        For the same reason the block outlives the game itself. The finale is queued on
        the channel's task queue and runs behind whatever the channel was already
        saying, so by the time the cast actually speaks, `_finish` has long since popped
        the game -- and a table that vanishes at the buzzer is exactly the state the
        gate above exists to avoid. `_finish` leaves the closing block in `_epilogues`
        for a few minutes, which also grounds the chat that follows a game.

        Most of the weight of this block is the briefing at the end of it, and it is
        deliberately spent on every generation in the channel while a game is live. A
        character asked about a game it can only see in outline will otherwise invent
        the detail -- naming cards in hands, describing plays that never happened --
        which reads as the model hallucinating because it is. The fix is telling it
        exactly which of those things it is not being shown.
        """
        game = self.cog.active_games.get(channel_id)
        if not game or game.stopping:
            return self._epilogue_block(channel_id)
        table = narration.describe_table(game)
        over = game.state.phase != "playing"
        if over:
            winner = game.seat(game.state.winner) if game.state.winner else None
            table += (f" The game is over -- {winner.display} went out and won."
                      if winner else " The game has ended with no winner.")
        return DEFAULT_GAME_CONTEXT.format(
            opening=DEFAULT_GAME_OPENING_OVER if over else DEFAULT_GAME_OPENING_LIVE,
            table=table,
            cast=narration.describe_cast(game),
            ledger=narration.describe_ledger(game),
            events="\n".join(f"- {line}" for line in game.recent) or "- Nothing yet.",
            rules=narration.describe_rules(game),
        )

    def _epilogue_block(self, channel_id: int) -> Optional[str]:
        """The last table of a finished game, while it is still recent.

        Read-only and free of `dict` mutation, because `context_block` reaches it from
        inside `asyncio.to_thread`. Expiry is checked here rather than swept: the entry
        is unreachable once stale, and the next `_finish` in the channel overwrites it.
        """
        entry = self._epilogues.get(channel_id)
        if not entry:
            return None
        block, expires_at = entry
        return block if time.monotonic() < expires_at else None

    # ------------------------------------------------------------------ panels


    def bind_panel(self, game: Game, seat_id: str, interaction, view=None) -> None:
        """Re-point a seat's panel handle at the interaction that just touched it.

        The signature is left unknown deliberately: the click that rebinds a panel is
        usually a move, and the resulting board is not known until the run loop has
        applied it, so the panel asks to be redrawn rather than claiming to be current.
        """
        previous = game.panels.get(seat_id)
        if previous is not None and previous.view is not None and previous.view is not view:
            try:
                previous.view.stop()
            except Exception:
                pass
        game.panels[seat_id] = Panel(interaction=interaction, view=view)

    async def close_panel(self, game: Game, seat_id: str) -> None:
        """Best-effort delete of a seat's existing panel, so only one is ever alive.

        Failure is normal and ignorable: the token may have expired, or the player may
        have dismissed the message themselves. Either way there is nothing to clean up.
        """
        panel = game.panels.pop(seat_id, None)
        if not panel:
            return
        if panel.view is not None:
            try:
                panel.view.stop()
            except Exception:
                pass
        try:
            await panel.interaction.delete_original_response()
        except Exception:
            pass

    async def refresh_panel(self, game: Game, seat_id: str) -> None:
        """Push the current board and hand into a seat's open panel, unprompted.

        This is what makes an edited-in-place panel work at all: a player's own clicks
        keep their view current, but the six things that happen between their turns do
        not, and a stale panel is the thing that made the old design repost.
        """
        panel = game.panels.get(seat_id)
        if not panel or not panel.pushable or game.state.phase != "playing":
            return
        try:
            from ..gui.gui_games import HandView, build_hand_embed
            view = HandView(self.cog, game, seat_id)
            await panel.interaction.edit_original_response(
                content=None, embed=build_hand_embed(game, seat_id), view=view)
            if panel.view is not None and panel.view is not view:
                try:
                    panel.view.stop()
                except Exception:
                    pass
            panel.view = view
            panel.signature = table_view.panel_signature(game, seat_id)
        except Exception:
            # An expired or invalidated handle. Drop it rather than retrying every
            # turn for the rest of the game; the seat's next click rebinds a fresh one.
            game.panels.pop(seat_id, None)

    # ------------------------------------------------------------------ sticky table

    def nudge_table(self, channel_id: int, message_id: Optional[int] = None) -> None:
        """Something landed in the channel; put the table back at the bottom.

        Called from `on_message` *above* its `author.bot` guard, because during a game
        most of the channel's traffic is webhook and child-bot posts and none of it gets
        past that guard. Synchronous and cheap -- one dict lookup on the common path --
        because it runs for every message the gateway delivers, in every channel.
        """
        game = self.cog.active_games.get(channel_id)
        if not game or game.stopping or game.state.phase != "playing":
            return
        if game.message_id is None or message_id == game.message_id:
            return
        if game.resink_pending:
            return
        game.resink_pending = True
        task = self.cog.bot.loop.create_task(self._resink(game))
        self.cog.background_tasks.add(task)
        task.add_done_callback(self.cog.background_tasks.discard)

    async def _resink(self, game: Game) -> None:
        """Wait out the repost floor, then move the table if it is still buried.

        The floor is a rate-limit guard, not a taste threshold: a repost that arrives
        during it is deferred to the end rather than dropped, so a burst of five lines
        produces one repost immediately after the burst instead of five during it.
        """
        try:
            wait = GAME_TABLE_REPOST_MIN_SECONDS - (time.monotonic() - game._last_repost)
            if wait > 0:
                await asyncio.sleep(wait)
            if game.stopping or game.state.phase != "playing":
                return
            await self._render(game, force=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            # Cleared last, and after `_render` has already stored the new message id,
            # so the gateway echo of our own repost finds a matching id and stops.
            game.resink_pending = False

    def has_live_game(self, channel_id: int) -> bool:
        game = self.cog.active_games.get(channel_id)
        return bool(game) and game.state.phase == "playing"

    def teardown_channel(self, channel_id: int) -> bool:
        """Drop a channel's game without awaiting anything.

        `/suspend` and `/purge` both call this. It has to be safe from a synchronous
        path, so it cancels rather than awaiting: the game cache is independent of the
        session, and without this hook a suspended channel would leave a game task
        running against a session that no longer exists.
        """
        game = self.cog.active_games.pop(channel_id, None)
        # A suspended or purged channel keeps no memory of the table either: the
        # epilogue is standing context, and standing context is exactly what /purge is
        # for removing.
        self._epilogues.pop(channel_id, None)
        # A lobby is dealt from the session's cast, so a session that has just been
        # suspended or purged has nothing left to deal. Dropped here rather than left
        # for its own Start to discover, which would refuse with a stale error.
        cleared_lobby = self.cog.pending_lobbies.pop(channel_id, None) is not None
        if not game:
            return cleared_lobby
        game.stopping = True
        if game.task and not game.task.done():
            game.task.cancel()
        return True

    async def _finish(self, channel_id: int, note: Optional[str] = None) -> None:
        game = self.cog.active_games.get(channel_id)
        if not game:
            return
        # Snapshot before the pop, while `context_block` can still see a game. Nothing
        # awaits in between, so this cannot race a second `_finish` into the channel.
        epilogue = self.context_block(channel_id)
        self.cog.active_games.pop(channel_id, None)
        if epilogue:
            self._epilogues[channel_id] = (
                epilogue, time.monotonic() + GAME_EPILOGUE_SECONDS)
            while len(self._epilogues) > GAME_EPILOGUE_CACHE_MAX_SIZE:
                self._epilogues.popitem(last=False)
        try:
            await self._render(game, final=True, note=note)
        except Exception:
            pass
        # The closing frame carries the result; a hand panel left open behind it holds
        # cards from a game that no longer exists.
        for seat_id in list(game.panels):
            await self.close_panel(game, seat_id)
        await self._write_back_neuro(game)

    async def _write_back_neuro(self, game: Game) -> None:
        """Persist where the game left each profile's chemistry.

        This is the half of the feature that makes a game matter afterwards -- a
        character really is rattled for a while after a bad one. It writes through the
        normal profile save path rather than touching anything the game owns.
        """
        for seat in game.seats:
            if seat.kind != "profile" or not seat.owner_id or not seat.profile_name:
                continue
            state = game.neuro.get(seat.seat_id)
            if not state:
                continue
            try:
                index = self.cog.profile_manager._get_user_index(seat.owner_id)
                borrowed = seat.profile_name in (index.get("borrowed") or [])
                config = self.cog.profile_manager._get_profile_config(
                    seat.owner_id, seat.profile_name, borrowed)
                if not config or not config.get("neuro_engine_enabled"):
                    continue
                config["neuro_state"] = dict(state)
                await asyncio.to_thread(
                    self.cog.profile_manager._save_profile_config,
                    seat.owner_id, seat.profile_name, config, borrowed)
            except Exception:
                continue

    # ------------------------------------------------------------------ the loop

    async def _run(self, channel_id: int) -> None:
        """Play the game out, one move at a time.

        Every seat is a profile in this phase, so the loop never has to wait on a
        person; the human path and its turn timer arrive with the interface. Pacing is
        deliberate rather than incidental -- a table that resolves instantly is
        unreadable, and the embed cannot be edited faster than Discord will accept.
        """
        game: Optional[Game] = self.cog.active_games.get(channel_id)
        if not game:
            return

        try:
            await self._render(game, force=True)
            while game.state.phase == "playing" and not game.stopping:
                if channel_id not in self.cog.active_games:
                    return

                seat_id = game.state.current.seat_id
                seat = game.seat(seat_id)
                if seat is None:
                    break

                if seat.kind == "human":
                    move, extra = await self._await_human_move(game, seat)
                else:
                    await asyncio.sleep(GAME_TURN_PACE_SECONDS)
                    move, extra = policy.choose(
                        eights.private_view(game.state, seat_id), game.neuro[seat_id],
                        game.temperaments[seat_id], game.rng), []

                try:
                    _, events = eights.apply(game.state, move)
                except IllegalMove:
                    # A submitted move that no longer fits the board. The turn cannot
                    # be skipped without stalling the table, so fall back to the
                    # policy's pick rather than dropping it. The call rides across:
                    # a player who shouted Last Card should not lose it because the board
                    # moved under the card they picked.
                    called = move.call_last
                    move = policy.choose(
                        eights.private_view(game.state, seat_id), game.neuro[seat_id],
                        PRESETS["steady"], game.rng)
                    move.call_last = move.call_last or called
                    _, events = eights.apply(game.state, move)
                events = extra + events

                neuro.apply_events(game.neuro, events, [s.seat_id for s in game.seats],
                                   game.temperaments)
                game.log.extend(events)
                if game.ledger:
                    game.ledger.observe(
                        events,
                        {s.seat_id: len(s.hand) for s in game.state.seats},
                        {s.seat_id: s.display for s in game.seats})
                game.recent.extend(narration.describe_events(game, events))
                game.turns += 1
                game.lap += 1

                # A lap is one full circuit of the table, and it is still the unit the
                # chemistry settles in -- but dialogue is no longer batched to it. A
                # reaction belongs to the moment that earned it, not to the boundary
                # that happens to follow it.
                if game.lap >= len(game.seats) or game.state.phase != "playing":
                    game.lap = 0
                    neuro.decay(game.neuro)

                await self._render(game, force=narration.is_dramatic(events))
                await self._refresh_open_panels(game)

                # The winning move is not reacted to here even though WENT_OUT is loud
                # enough to qualify. The finale below covers it, and covers it better:
                # one character crowing and then the same character crowing again in
                # the aftermath round is the sort of seam nobody can unsee.
                beat = (narration.beat_for(game, events)
                        if game.state.phase == "playing" else None)
                if beat is not None:
                    try:
                        await self._request_reaction(game, *beat)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        # Dialogue is decoration. A missing key, a model outage or a
                        # malformed response must never take the table down with it, so
                        # this is caught here rather than falling through to the run
                        # loop's handler, which ends the game.
                        print(f"[Game] Reaction skipped for channel {channel_id}: {e}")

            # Only a game that ran out on its own gets an aftermath. A `/stop` is
            # somebody closing the table, and having the cast eulogise a hand the room
            # just cancelled is the opposite of reading the room.
            if not game.stopping and game.state.phase != "playing":
                try:
                    await self._request_finale(game)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[Game] Finale skipped for channel {channel_id}: {e}")

            await self._finish(channel_id)

        except asyncio.CancelledError:
            raise
        except Exception:
            traceback.print_exc()
            self.cog.active_games.pop(channel_id, None)
            try:
                channel = self.cog.bot.get_channel(channel_id)
                if channel:
                    await channel.send(
                        "The game hit an internal error and has been stopped.")
            except Exception:
                pass

    async def _await_human_move(self, game: Game, seat: Seat):
        """Wait for a seated person to choose, or play for them when the clock runs out.

        The deadline is this wait's timeout rather than a separate timer task. One
        absent player would otherwise stall a ten-minute game indefinitely, and it will
        happen in the first week -- but a per-game polling task on a 1 GB box, with
        every child bot's gateway sharing this loop, is exactly the pattern the
        deployment constraint exists to prevent. A `wait_for` costs nothing extra.
        """
        game.pending_move = None
        game.turn_event = asyncio.Event()
        game.deadline = time.monotonic() + game.state.rules.turn_seconds
        await self._render(game, force=True)
        await self.refresh_panel(game, seat.seat_id)

        try:
            # Sliced rather than one long wait. The footer's countdown is read from
            # `deadline` at render time, so a single `wait_for` left it frozen at the
            # starting figure for the whole turn -- and this is also the one stretch of
            # a game where nothing else redraws, which is exactly when the channel is
            # most likely to bury the table.
            while True:
                left = game.deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    await asyncio.wait_for(game.turn_event.wait(),
                                           timeout=min(TURN_TICK_SECONDS, left))
                    break
                except asyncio.TimeoutError:
                    await self._render(game, force=True)
        finally:
            move = game.pending_move
            game.pending_move = None
            game.turn_event = None
            game.deadline = None

        # The call is spent here rather than at the click, so it survives a click that
        # is refused, and so a player who armed it and then let the clock run out is
        # still credited -- they said it, and the table heard them.
        if move is not None:
            if move.kind == "play":
                move.call_last = move.call_last or self.take_last_call(game, seat.seat_id)
            return move, []

        auto = policy.choose(eights.private_view(game.state, seat.seat_id),
                             game.neuro[seat.seat_id], PRESETS["steady"], game.rng)
        auto.call_last = auto.call_last or self.take_last_call(game, seat.seat_id)
        return auto, [Event(Ev.TIMED_OUT, seat_id=seat.seat_id)]

    # ------------------------------------------------------------------ dialogue

    #: Events loud enough that a character would actually say something. Everything
    #: else -- a plain number card, a colour change, a quiet draw -- passes in silence,
    #: which is what keeps a fifty-turn game to a handful of generations.









    def _participant_for(self, session: Dict[str, Any], seat: Seat):
        """The seat's entry in the live session cast, by identity.

        Identity matters rather than equality: the worker rotates the cast with
        `session['profiles'].index(start_p)` and tests membership with `in`, so handing
        it a freshly built dict that merely looks the same would silently fail both.
        Returns None for a seat whose profile has since left the session -- the roster
        is a snapshot, so that is a normal state, not an error.
        """
        for participant in session.get("profiles", []):
            if (participant.get("owner_id") == seat.owner_id
                    and participant.get("profile_name") == seat.profile_name):
                return participant
        return None

    def _ensure_worker(self, channel_id: int, session: Dict[str, Any]) -> None:
        """Make sure something is going to take what we just queued.

        The channel worker is spawned lazily, by whatever first gives the session work
        -- a message, a reaction, `/trigger`, the Director. A table is a fifth
        such source, and it used to be the one that did not spawn it: a session built
        with `/session config` and then handed straight to `/play eights` has a
        `task_queue` and no `worker_task`, so every beat the game queued sat in the
        queue and the cast played the whole game in silence. Saying anything in the
        channel started the worker and the backlog arrived at once, which is exactly
        the "it only works if you talk to it first" shape.

        Idempotent: a live worker is left alone, and a finished one is replaced.
        """
        task = session.get('worker_task')
        if task is not None and not task.done():
            return
        task = self.cog.bot.loop.create_task(
            self.cog.generation_service._multi_profile_worker(channel_id))
        session['worker_task'] = task
        self.cog.background_tasks.add(task)
        task.add_done_callback(self.cog.background_tasks.discard)

    async def _request_reaction(self, game: Game, seat: Seat, beat: str) -> None:
        """Ask the channel's own worker for a reaction, rather than generating one here.

        This is the whole of the dialogue path now, and everything it used to do by hand
        it deliberately no longer does. A reaction is an ordinary multi-profile round:
        it goes on `session['task_queue']` as a trigger, and the worker builds the
        prompt the same way it builds every other one -- full instructions, training
        examples, LTM retrieval, the critic, the profile's own model and safety
        settings. The bespoke call this replaced had only the persona, which is exactly
        the half of a character that is cheapest to fake and least worth having.

        Two things follow from routing it through the queue rather than around it.
        Feedback is free: the worker dispatches the placeholder emote or the typing
        indicator before it builds anything, so a character visibly takes the floor.
        And it is serialised: a reaction can no longer talk over a round that the
        channel was already running, which is what the queue exists to prevent.

        Fire-and-forget by design. The run loop does not wait for the character to
        finish speaking -- the table keeps moving, and a beat that has gone stale by the
        time the worker reaches it is dropped there rather than acted on late.
        """
        session = self.cog.multi_profile_channels.get(game.channel_id)
        if not session:
            return
        queue = session.get("task_queue")
        if queue is None:
            return
        participant = self._participant_for(session, seat)
        if participant is None:
            return

        game.generations += 1
        await queue.put(("game_beat", {
            "channel_id": game.channel_id,
            "beat": beat,
            "content": DEFAULT_GAME_REACTION_USER.format(
                beat=beat, max_words=GAME_REACTION_MAX_WORDS),
            "queued_at": time.monotonic(),
            "stale_after": GAME_BEAT_STALE_SECONDS,
        }, participant))
        self._ensure_worker(game.channel_id, session)


    async def _request_finale(self, game: Game) -> None:
        """Ask the channel for the aftermath: one round, the whole table speaking.

        This is the one beat that is not one character's moment. A game that ends with
        nobody saying anything is the point at which the cast stops reading as players
        and starts reading as a scoreboard, and it is also the moment with the most to
        say -- so the trigger carries the seated cast and the worker seats all of them
        in a single round rather than making it single-turn.

        Deliberately outside `GAME_REACTION_MAX_CALLS`. The ceiling exists to stop a
        pathological game running away mid-play; the ending happens exactly once, and a
        game that spent its budget on Draw Fours should not fall silent at the buzzer.
        The cost is still counted, because `generations` is what the tests measure.

        Queued before `_finish`, so the closing table is still readable here; by the
        time the worker reaches the round the game is gone, which is what `_epilogues`
        is for.
        """
        session = self.cog.multi_profile_channels.get(game.channel_id)
        if not session:
            return
        queue = session.get("task_queue")
        if queue is None:
            return
        beat = narration.finale_beat(game)
        if not beat:
            return

        cast = []
        for seat in game.seats:
            if seat.kind != "profile":
                continue
            participant = self._participant_for(session, seat)
            if participant is not None and not any(p is participant for p in cast):
                cast.append(participant)
        if not cast:
            return

        # The winner leads, when the winner is one of ours. `_collect_round_triggers`
        # reads the starting override off the trigger's participant, and the character
        # who just went out is the one the table is looking at.
        winner_seat = game.seat(game.state.winner) if game.state.winner else None
        if winner_seat is not None and winner_seat.kind == "profile":
            lead = self._participant_for(session, winner_seat)
            if lead is not None:
                cast = [lead] + [p for p in cast if p is not lead]

        game.generations += len(cast)
        await queue.put(("game_beat", {
            "channel_id": game.channel_id,
            "beat": beat,
            "content": DEFAULT_GAME_FINALE_USER.format(
                beat=beat, max_words=GAME_FINALE_MAX_WORDS),
            "queued_at": time.monotonic(),
            "stale_after": GAME_FINALE_STALE_SECONDS,
            "finale": True,
            "cast": cast,
        }, cast[0]))
        self._ensure_worker(game.channel_id, session)

    # ----------------------------------------------------------- calling last card

    def arm_last_call(self, channel_id: int, user_id: int, content: str) -> bool:
        """Arm a seated player's Last Card call from something they said in the channel.

        Called from `on_message` and answers in one dict lookup on the no-game path.
        Returns whether the call was taken, so the listener can acknowledge it -- and
        stop, because a call is a control input rather than a line of dialogue. The cast
        hears it through `REACT_ON` when the call rides the next play instead.

        The call arms the seat and the next play carries it, which is the same contract
        the button had -- with `auto_call_penalty` on, the penalty lands inside
        `eights.apply` the instant a seat plays down to one card, so there is no honest
        way to accept a call *after* the play. Saying it early is therefore the only
        thing that can work, and it is also what people do at a real table.
        """
        game = self.cog.active_games.get(channel_id)
        if not game or game.stopping or game.state.phase != "playing":
            return False
        if content.strip().lower() not in GAME_LAST_CALL_WORDS:
            return False
        seat_id = f"user:{user_id}"
        if game.seat(seat_id) is None:
            return False
        game.last_call_armed[seat_id] = True
        return True

    def take_last_call(self, game: Game, seat_id: str) -> bool:
        """Spend a seat's armed call, if it has one. Arming is one-shot: a call that has
        ridden a play is gone, so a player who shouts it once cannot coast on it for the
        rest of the game."""
        return bool(game.last_call_armed.pop(seat_id, False))

    async def _refresh_open_panels(self, game: Game) -> None:
        """Push to the panels that are actually showing something out of date.

        Gated rather than pushed unconditionally, because an unconditional push is one
        API call per open panel per turn. The gate is the full render signature, not
        the hand: the board moves under a panel far more often than its cards do, and
        watching the cards alone made a panel look broken -- see `panel_signature`.
        """
        if not game.panels:
            return
        for seat_id, panel in list(game.panels.items()):
            if panel.signature != table_view.panel_signature(game, seat_id):
                await self.refresh_panel(game, seat_id)


    # ------------------------------------------------------------------ rendering

    async def _render(self, game: Game, force: bool = False, final: bool = False,
                      note: Optional[str] = None) -> None:
        """Draw or redraw the table, keeping it at the bottom of the channel.

        The table is sticky: if anything has landed under it, it is deleted and reposted
        rather than edited, so the controls are never scrolled off. Burial is detected
        from `channel.last_message_id`, which discord.py maintains from the gateway for
        every message including webhooks and other bots -- so this costs no API call and
        does not depend on a listener that would miss most of a game's traffic.

        Edits are throttled: a fifty-turn game would otherwise ask Discord for fifty of
        them in a couple of minutes. Laps, dramatic moments and reposts force a redraw;
        everything else waits for the interval.
        """
        now = time.monotonic()
        if not (force or final) and now - game._last_render < GAME_EMBED_MIN_INTERVAL_SECONDS:
            return
        game._last_render = now

        channel = self.cog.bot.get_channel(game.channel_id)
        if channel is None:
            return

        async with game.render_lock:
            await self._draw(game, channel, final=final, note=note)

    async def _draw(self, game: Game, channel, final: bool = False,
                    note: Optional[str] = None) -> None:
        """The body of `_render`, under the lock."""
        embed = table_view.build_embed(game, final=final, note=note)
        # The table carries one public button, and only when somebody could use it.
        # An all-profile table has no private hands to open.
        view = None
        if not final and any(s.kind == "human" for s in game.seats):
            from ..gui.gui_games import TableView
            view = TableView(self.cog, game.channel_id)

        buried = (game.message_id is not None
                  and getattr(channel, "last_message_id", None) not in (None, game.message_id))

        try:
            if game.message_id is None:
                await self._post_table(channel, game, embed, view)
            elif buried and not final:
                # Delete first. Posting first would leave two tables visible for a beat
                # and, if the delete then failed, permanently.
                await self._drop_table(channel, game)
                await self._post_table(channel, game, embed, view)
            else:
                message = game.message or await channel.fetch_message(game.message_id)
                game.message = message
                await message.edit(embed=embed, view=view)
        except discord.NotFound:
            # Someone deleted the table, most likely a purge. Post a fresh one unless
            # this was the closing frame.
            game.message_id, game.message = None, None
            if not final:
                try:
                    await self._post_table(channel, game, embed, view)
                except discord.HTTPException:
                    pass
        except discord.HTTPException:
            pass

    async def _post_table(self, channel, game: Game, embed, view) -> None:
        """Send a fresh table and adopt it. The id is stored before this returns, which
        is what lets `nudge_table` recognise the repost's own gateway echo."""
        sent = await channel.send(embed=embed, view=view)
        game.message = sent
        game.message_id = sent.id
        game._last_repost = time.monotonic()

    async def _drop_table(self, channel, game: Game) -> None:
        """Remove the current table. A table that has already gone is not an error."""
        message = game.message
        try:
            if message is None and game.message_id is not None:
                message = await channel.fetch_message(game.message_id)
            if message is not None:
                await message.delete()
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            pass
        finally:
            game.message, game.message_id = None, None

    #: Kept on the service because gui_games calls it there; the implementation
    #: lives with the rest of the table rendering.
    house_rule_summary = staticmethod(table_view.house_rule_summary)

