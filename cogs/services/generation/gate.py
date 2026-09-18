"""Bot-wide slots for model calls in flight, handed out in arrival order.

Every reply, fallback, whisper, regeneration and generated image reaches its model
through `_generate_with_heartbeat`, and that is the one place a slot is taken. A
call waiting on a model costs the loop almost nothing; what costs it is the burst at
either end -- the payload built, the answer parsed, the reply delivered -- and with
no limit a busy minute starts all of them at once. With one, a reply past the limit
waits its turn with its placeholder saying so, and the loop sees at most
GENERATION_SLOTS answers land together.

Only the heartbeat call takes a slot. The critic, recall, synopsis and Director calls
a turn makes around it do not: they are short, bounded by the number of live rounds,
and a turn holding a slot must never wait on a second one.

A fallback goes to the front of the queue. Its turn has already waited once.
"""

import asyncio
import collections
import time
from typing import Optional

from ...utils.constants import GENERATION_SLOTS

#: Seconds between the "all slots busy" lines, so a long busy spell prints a handful
#: rather than one per queued reply.
_BUSY_REPORT_INTERVAL = 60.0


class GenerationTicket:
    """One call's place in the gate. Made before the call starts, so the heartbeat can
    read the position while the call is still waiting inside its own task."""

    __slots__ = ("_gate", "_front", "_future", "admitted_at")

    def __init__(self, gate: "GenerationGate", front: bool):
        self._gate = gate
        self._front = front
        self._future: Optional[asyncio.Future] = None
        #: time.time() when the slot was taken; None while waiting or not yet started.
        self.admitted_at: Optional[float] = None

    @property
    def position(self) -> int:
        """1 for the next in line, 0 when not waiting."""
        return self._gate._position(self._future)

    @property
    def waiting(self) -> int:
        """How many calls are waiting in all, this one included."""
        return len(self._gate._queue)

    async def __aenter__(self):
        await self._gate._acquire(self)
        return self

    async def __aexit__(self, *_exc):
        self._gate._release()
        return False


class GenerationGate:
    def __init__(self, slots: int):
        self.slots = max(1, slots)
        self.active = 0
        self._queue: collections.deque = collections.deque()
        self._last_busy_report = 0.0

    def ticket(self, front: bool = False) -> GenerationTicket:
        return GenerationTicket(self, front)

    def _position(self, future: Optional[asyncio.Future]) -> int:
        if future is None:
            return 0
        try:
            return self._queue.index(future) + 1
        except ValueError:
            return 0

    async def _acquire(self, ticket: GenerationTicket) -> None:
        # Straight in only when nobody is waiting: a free slot with a queue behind it
        # belongs to the head of the queue, which `_release` has already handed it to.
        if self.active < self.slots and not self._queue:
            self.active += 1
            ticket.admitted_at = time.time()
            return

        future = asyncio.get_running_loop().create_future()
        ticket._future = future
        if ticket._front:
            self._queue.appendleft(future)
        else:
            self._queue.append(future)
        self._report_busy()
        try:
            await future
        except BaseException:
            if future.done() and not future.cancelled():
                # Handed the slot in the same instant the caller was cancelled. Nobody
                # will release it on this caller's behalf, so pass it on now.
                self._release()
            else:
                try:
                    self._queue.remove(future)
                except ValueError:
                    pass
            raise
        finally:
            ticket._future = None
        ticket.admitted_at = time.time()

    def _release(self) -> None:
        # The slot goes to the next waiter directly, so `active` never dips and lets a
        # newcomer in ahead of it.
        while self._queue:
            future = self._queue.popleft()
            if not future.done():
                future.set_result(None)
                return
        self.active -= 1

    def _report_busy(self) -> None:
        now = time.monotonic()
        if now - self._last_busy_report < _BUSY_REPORT_INTERVAL:
            return
        self._last_busy_report = now
        print(f"[generation-gate] all {self.slots} slots busy, {len(self._queue)} waiting "
              f"(MIMIC_GENERATION_SLOTS sets the count)", flush=True)


_gate: Optional[GenerationGate] = None


def generation_gate() -> GenerationGate:
    """The process-wide gate, made on first use at GENERATION_SLOTS."""
    global _gate
    if _gate is None:
        _gate = GenerationGate(GENERATION_SLOTS)
    return _gate
