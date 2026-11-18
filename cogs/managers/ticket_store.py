"""Support tickets: written to the bot's operator from `/help`, and answered there.

Two levels, so nothing on the way to a ticket decrypts more than that one ticket:

* `data/tickets.json`, plaintext like a user's `index.json`: who opened each ticket, its
  category, status, times and unread flags -- never a word of what it says. Read once,
  held in memory, and the answer to every count and list `/help` draws.
* `users/<id>/tickets/<n>.json.gz`, sealed: the messages, and a trace if one was attached,
  read only when the ticket is opened. Under the user's own directory, so `/privacy`'s
  export carries it, and Delete My Account and the departed-user cleanup remove it with
  everything else -- each of them calls `drop_user` beside its rmtree, for the index.

Shard first, index second, as everywhere: a crash orphans a ticket file, never lists a
ticket that is not there.
"""
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

from ..utils.constants import (DATA_DIR, LIMIT_OPEN_TICKETS, LIMIT_TICKET_MESSAGES,
                               LIMIT_TICKETS_KEPT, TICKET_RETENTION_DAYS)
from .storage_manager import IOManager, _delete_file_shard

TICKETS_INDEX_FILE = os.path.join(DATA_DIR, "tickets.json")

#: Who wrote a message, and whose unread flag a message raises.
USER, OPERATOR = "user", "owner"

#: Every read-modify-write of the index and a ticket file, and every read of the index:
#: the daily cleanup and account deletion drop rows from worker threads.
# ponytail: one lock for every ticket, held across a file write; per-ticket locks if ticket
# traffic ever contends -- at a person's typing pace it does not.
_LOCK = threading.Lock()

TICKET_GONE = "This ticket no longer exists."
TICKET_CLOSED = "This ticket is closed. Open a new one if there is more to say."
TICKET_FULL = "This ticket has reached its message limit. Open a new one to carry on."


class TicketStore:
    def __init__(self, cog):
        self.cog = cog
        self._index: Optional[Dict] = None

    # --- the index ------------------------------------------------------------------

    def _load(self) -> Dict:
        """The index, read from disk the first time. Callers hold `_LOCK`."""
        if self._index is None:
            data = IOManager.read_json(TICKETS_INDEX_FILE)
            if not (isinstance(data, dict) and isinstance(data.get("tickets"), dict)):
                data = {"next": 1, "tickets": {}}
            self._index = data
        return self._index

    def load(self) -> None:
        """Reads the index if this process has not. Sync: run it through `to_thread`."""
        with _LOCK:
            self._load()

    def _save(self) -> None:
        IOManager.write_json(self._index, TICKETS_INDEX_FILE)

    def _path(self, user_id: int, ticket_id: str) -> str:
        return os.path.join(self.cog.USERS_DIR, str(user_id), "tickets", f"{ticket_id}.json.gz")

    def row(self, ticket_id: str) -> Optional[Dict]:
        """A copy of one ticket's index row, or None."""
        with _LOCK:
            row = self._load()["tickets"].get(ticket_id)
            return dict(row) if row else None

    def rows(self, user_id: Optional[int] = None, status: Optional[str] = None) -> List[Tuple[str, Dict]]:
        """(id, row copy), most recently changed first: one user's, or everyone's."""
        with _LOCK:
            found = [(tid, dict(row)) for tid, row in self._load()["tickets"].items()
                     if (user_id is None or row["user"] == user_id)
                     and (status is None or row["status"] == status)]
        return sorted(found, key=lambda item: item[1]["updated"], reverse=True)

    def unread(self, user_id: Optional[int] = None) -> int:
        """Tickets with something new: for one user, or for the operator when None."""
        key = "unread_owner" if user_id is None else "unread_user"
        return sum(1 for _, row in self.rows(user_id) if row.get(key))

    # --- writes (sync: each runs through `asyncio.to_thread`) ------------------------

    def open(self, user_id: int, guild_id: Optional[int], category: str, text: str,
             trace: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str]]:
        """(ticket id, None) once stored, or (None, why not)."""
        now = time.time()
        with _LOCK:
            index = self._load()
            mine = [(tid, row) for tid, row in index["tickets"].items() if row["user"] == user_id]
            if sum(1 for _, row in mine if row["status"] == "open") >= LIMIT_OPEN_TICKETS:
                return None, (f"You have {LIMIT_OPEN_TICKETS} tickets open already. Close one, "
                              "or add to it, before opening another.")
            ticket_id = str(index["next"])
            body = {"id": ticket_id, "category": category, "guild": guild_id, "created": now,
                    "messages": [{"from": USER, "at": now, "text": text}]}
            if trace:
                body["trace"] = trace
            path = self._path(user_id, ticket_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            IOManager.write_json_gzip(body, path, self.cog.fernet)
            index["next"] = int(ticket_id) + 1
            index["tickets"][ticket_id] = {
                "user": user_id, "guild": guild_id, "category": category, "status": "open",
                "created": now, "updated": now, "last_from": USER, "messages": 1,
                "unread_user": False, "unread_owner": True}
            # Past the keep limit the oldest closed ones go: forgotten here, deleted after
            # the index that named them is written.
            closed = sorted((row["updated"], tid) for tid, row in mine if row["status"] == "closed")
            excess = len(mine) + 1 - LIMIT_TICKETS_KEPT
            dropped = [tid for _, tid in closed[:max(0, excess)]]
            for tid in dropped:
                del index["tickets"][tid]
            self._save()
        for tid in dropped:
            _delete_file_shard(self._path(user_id, tid))
        return ticket_id, None

    def read(self, ticket_id: str) -> Optional[Dict]:
        """The ticket's messages and trace, or None when it is gone or will not open."""
        row = self.row(ticket_id)
        if row is None:
            return None
        return IOManager.read_json_gzip(self._path(row["user"], ticket_id), self.cog.fernet)

    def reply(self, ticket_id: str, author: str, text: str) -> Optional[str]:
        """None once `author` (USER or OPERATOR) has added `text`, or why not."""
        now = time.time()
        with _LOCK:
            row = self._load()["tickets"].get(ticket_id)
            if row is None:
                return TICKET_GONE
            if row["status"] != "open":
                return TICKET_CLOSED
            if row["messages"] >= LIMIT_TICKET_MESSAGES:
                return TICKET_FULL
            path = self._path(row["user"], ticket_id)
            body = IOManager.read_json_gzip(path, self.cog.fernet)
            if body is None:
                return "This ticket could not be read."
            body["messages"].append({"from": author, "at": now, "text": text})
            IOManager.write_json_gzip(body, path, self.cog.fernet)
            row.update(updated=now, last_from=author, messages=len(body["messages"]),
                       unread_user=author == OPERATOR, unread_owner=author == USER)
            self._save()
        return None

    def close(self, ticket_id: str, by: str) -> bool:
        """True when it was open. Closed by the operator, it reads as news to the user."""
        with _LOCK:
            row = self._load()["tickets"].get(ticket_id)
            if row is None or row["status"] != "open":
                return False
            row.update(status="closed", updated=time.time(), closed_by=by)
            if by == OPERATOR:
                row["unread_user"] = True
            else:
                row["unread_owner"] = False
            self._save()
        return True

    def mark_read(self, ticket_id: str, reader: str) -> None:
        key = "unread_owner" if reader == OPERATOR else "unread_user"
        with _LOCK:
            row = self._load()["tickets"].get(ticket_id)
            if not row or not row.get(key):
                return  # nothing changes, so nothing is written
            row[key] = False
            self._save()

    def drop_user(self, user_id: int) -> None:
        """Forgets a user whose directory -- their ticket files with it -- was removed."""
        with _LOCK:
            tickets = self._load()["tickets"]
            gone = [tid for tid, row in tickets.items() if row["user"] == user_id]
            for tid in gone:
                del tickets[tid]
            if gone:
                self._save()

    def purge_expired(self, now: Optional[float] = None) -> int:
        """Deletes tickets closed and untouched for TICKET_RETENTION_DAYS. Reads the index
        alone: no directory is walked. Returns how many went."""
        cutoff = (now or time.time()) - TICKET_RETENTION_DAYS * 86400
        with _LOCK:
            tickets = self._load()["tickets"]
            expired = [(tid, row["user"]) for tid, row in tickets.items()
                       if row["status"] == "closed" and row["updated"] < cutoff]
            for tid, _ in expired:
                del tickets[tid]
            if expired:
                self._save()
        for tid, user_id in expired:
            _delete_file_shard(self._path(user_id, tid))
        return len(expired)
